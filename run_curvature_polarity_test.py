#!/usr/bin/env python3 -u
"""Curvature Polarity Test (CPT).

Tests the Curvature Polarity Hypothesis (CPH):
  κ[l] = ||a[l]|| / ||v[l]|| predicts steering polarity.
  κ < 1 → velocity-dominated → trim-tab (steering helps)
  κ > 1 → acceleration-dominated → death layer (steering hurts)

Uses existing trajectory data — 0 GPU-hours needed.
"""
from __future__ import annotations

import gc, glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cpu"


def compute_kappa(hidden_seqs):
    """Compute κ[l] = ||a[l]|| / ||v[l]|| for each layer transition.
    
    Args:
        hidden_seqs: (B, L, D) tensor of hidden states at layers 0..L-1
    Returns:
        kappa: (B, L-2) tensor of curvatures for layers 1..L-2
               κ[l] corresponds to the curvature at layer l
    """
    v = hidden_seqs[:, 1:] - hidden_seqs[:, :-1]      # (B, L-1, D): velocity at each layer
    a = v[:, 1:] - v[:, :-1]                            # (B, L-2, D): acceleration at each layer
    v_norm = v[:, :-1].norm(dim=-1)                     # (B, L-2): ||v[l]|| for l=1..L-2
    a_norm = a.norm(dim=-1)                             # (B, L-2): ||a[l]||
    kappa = a_norm / (v_norm + 1e-8)                    # (B, L-2)
    return kappa


def analyze_08b():
    """Analyze Qwen3.5-0.8B trajectory data."""
    print("=" * 60, flush=True)
    print("CURVATURE POLARITY TEST — Qwen3.5-0.8B", flush=True)
    print("=" * 60, flush=True)

    DATA_DIR = "/home/filip/Projects/Personal/AI/RankAdaptation/data/qwen35_08b_trajs"
    files = sorted(glob.glob(os.path.join(DATA_DIR, "batch_*.pt")))
    print(f"Found {len(files)} batch files", flush=True)

    all_kappa = []
    n_total = 0
    for f in files:
        d = torch.load(f, map_location="cpu")
        hs = d["hidden_seqs"].float()
        k = compute_kappa(hs)
        all_kappa.append(k)
        n_total += len(hs)
        del d, hs, k

    kappa = torch.cat(all_kappa)  # (N, L-2)
    kappa_mean = kappa.mean(dim=0).numpy()  # (L-2,): layers 1..L-2
    kappa_std = kappa.std(dim=0).numpy()
    kappa_se = kappa_std / np.sqrt(n_total)
    print(f"Total trajectories: {n_total}", flush=True)
    print(f"κ mean shape: {kappa_mean.shape}", flush=True)

    # Known per-layer results for 0.8B (FA layers only)
    # From per_layer_sweep_08b.json: baseline=30%
    known_deltas_08b = {3: -0.033, 7: -0.100, 11: -0.033, 15: -0.267, 19: -0.167, 23: 0.0}
    fa_layers = [3, 7, 11, 15, 19, 23]
    n_layers = 24

    print(f"\n{'='*60}", flush=True)
    print(f"Per-layer κ values (all {n_layers} layers)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Layer':>6} {'Type':>4} {'κ mean':>10} {'κ std':>10} {'κ < 1?':>10}", flush=True)

    fa_set = set(fa_layers)
    results = {}
    for li in range(1, n_layers - 1):  # κ defined for layers 1..L-2
        k_mean = float(kappa_mean[li - 1])
        k_std = float(kappa_std[li - 1])
        lt = "FA" if li in fa_set else "GDN"
        pred = "TRIM-TAB ✓" if k_mean < 1.0 else "DEATH"
        print(f"  L{li:3d}  {lt:>4}  {k_mean:10.4f}  {k_std:10.4f}  {pred:>10}", flush=True)
        results[li] = {"type": lt, "kappa_mean": k_mean, "kappa_std": k_std, "predicted": pred}

    # Compare with known FA results
    print(f"\n{'='*60}", flush=True)
    print(f"CPH Validation against known FA-layer Δaccuracy", flush=True)
    print(f"{'='*60}", flush=True)

    fa_kappas, fa_deltas = [], []
    for li in sorted(known_deltas_08b.keys()):
        if li in results and li in known_deltas_08b:
            k = results[li]["kappa_mean"]
            d = known_deltas_08b[li]
            fa_kappas.append(k)
            fa_deltas.append(d)
            pred = "TRIM-TAB ✓" if k < 1.0 else "DEATH"
            cph = "✓ CPH" if ((k < 1.0 and d > 0) or (k > 1.0 and d < 0)) else "✗ CPH"
            print(f"  L{li:2d}  κ={k:.4f}  Δacc={d:+.4f}  {pred:>12}  {cph:>6}", flush=True)

    if len(fa_kappas) >= 3:
        r = np.corrcoef(fa_kappas, fa_deltas)[0, 1]
        print(f"\n  Pearson ρ(κ[l], Δacc[l]) = {r:.4f} (FA layers)", flush=True)

    # L15 specific test
    l15_k = results.get(15, {}).get("kappa_mean", None)
    if l15_k is not None:
        print(f"\n  L15 prediction: κ={'>' if l15_k > 1 else '<'} 1 → {'DEATH ✓' if l15_k > 1 else 'TRIM-TAB ✗'}")

    return results


def analyze_7b():
    """Analyze Qwen2.5-7B trajectory data if available."""
    print("\n" + "=" * 60, flush=True)
    print("CURVATURE POLARITY TEST — Qwen2.5-7B", flush=True)
    print("=" * 60, flush=True)

    DATA_DIR = "/run/media/filip/B522-875D/Datasets/project_data/qwen25_7b_gen_trajs"
    files = sorted(glob.glob(os.path.join(DATA_DIR, "gen_trajs_7b_batch_*.pt")))
    if not files:
        print("7B trajectory data not found — skipping", flush=True)
        return None
    print(f"Found {len(files)} batch files", flush=True)

    all_kappa = []
    n_total = 0
    for f in files[:5]:  # Limit to 5 files for speed
        d = torch.load(f, map_location="cpu")
        hs = d["hidden_seqs"].float()
        k = compute_kappa(hs)
        all_kappa.append(k)
        n_total += len(hs)
        del d, hs, k
        gc.collect()

    kappa = torch.cat(all_kappa)
    kappa_mean = kappa.mean(dim=0).numpy()
    kappa_std = kappa.std(dim=0).numpy()
    kappa_se = kappa_std / np.sqrt(n_total)
    print(f"Total trajectories: {n_total}", flush=True)

    # Known 7B per-layer results (from recovery doc)
    known_deltas_7b = {
        0: -0.03, 1: 0.00, 2: 0.17, 3: 0.13, 4: 0.07, 5: 0.13,
        6: 0.00, 7: -0.05, 8: 0.20, 9: -0.23, 10: 0.17,
        11: 0.00, 12: 0.00, 13: 0.00, 14: 0.00, 15: 0.00,
        16: 0.00, 17: 0.00, 18: 0.00, 19: 0.00, 20: 0.00,
        21: 0.00, 22: 0.00, 23: 0.00, 24: -0.10, 25: 0.00,
        26: 0.00, 27: 0.00,
    }
    n_layers_7b = 28

    print(f"\n{'='*60}", flush=True)
    print(f"Key layers", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Layer':>6} {'κ mean':>10} {'Δacc':>10} {'Prediction':>15}", flush=True)

    layers_7b_results = {}
    for li in range(1, n_layers_7b - 1):
        k_mean = float(kappa_mean[li - 1])
        d = known_deltas_7b.get(li, 0)
        pred = "TRIM-TAB" if k_mean < 1.0 else "DEATH"
        cph = "✓" if ((k_mean < 1.0 and d > 0) or (k_mean > 1.0 and d < 0)) else "✗"
        layers_7b_results[li] = {"kappa_mean": k_mean, "delta_acc": d, "predicted": pred, "cph": cph}

        if abs(d) > 0.05 or li in [8, 9]:
            print(f"  L{li:3d}  {k_mean:10.4f}  {d:+10.4f}  {pred:>12}  {cph}", flush=True)

    # L8 vs L9 specific
    l8_k = layers_7b_results.get(8, {}).get("kappa_mean", None)
    l9_k = layers_7b_results.get(9, {}).get("kappa_mean", None)
    if l8_k and l9_k:
        print(f"\n  L8: κ={l8_k:.4f} {'< 1 ✓' if l8_k < 1 else '≥ 1 ✗'} (actual: +20pp)", flush=True)
        print(f"  L9: κ={l9_k:.4f} {'> 1 ✓' if l9_k > 1 else '≤ 1 ✗'} (actual: -23pp)", flush=True)
        print(f"  κ[L8] < κ[L9]: {l8_k < l9_k}", flush=True)

    # Correlation
    both_layers = [(layers_7b_results[l]["kappa_mean"], known_deltas_7b[l])
                   for l in range(1, n_layers_7b - 1)
                   if l in layers_7b_results and l in known_deltas_7b]
    if both_layers:
        kappas, deltas = zip(*both_layers)
        r = np.corrcoef(list(kappas), list(deltas))[0, 1]
        print(f"\n  Pearson ρ(κ[l], Δacc[l]) = {r:.4f} (all 7B layers)", flush=True)

    return layers_7b_results


if __name__ == "__main__":
    t0 = time.time()

    res_08b = analyze_08b()
    gc.collect()
    res_7b = analyze_7b()

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"Total time: {elapsed:.0f} min", flush=True)
    print(f"{'='*60}", flush=True)
