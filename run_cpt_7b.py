#!/usr/bin/env python3 -u
"""CPT validation on Qwen2.5-7B — the capable model.

Tests whether κ[l] curvature rank predicts steering polarity
on a model KNOWN to have both trim-tabs (+20pp) and death layers (-23pp).
"""
from __future__ import annotations

import gc, glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_kappa(hidden_seqs):
    v = hidden_seqs[:, 1:] - hidden_seqs[:, :-1]
    a = v[:, 1:] - v[:, :-1]
    v_norm = v[:, :-1].norm(dim=-1)
    a_norm = a.norm(dim=-1)
    return a_norm / (v_norm + 1e-8)


def main():
    t0 = time.time()

    DATA_DIR = "/run/media/filip/B522-875D/Datasets/project_data/qwen25_7b_gen_trajs"
    files = sorted(glob.glob(os.path.join(DATA_DIR, "gen_trajs_7b_batch_*.pt")))
    print(f"7B trajectory files: {len(files)}", flush=True)

    all_kappa = []
    n_total = 0
    for f in files:
        d = torch.load(f, map_location="cpu")
        hs = d["hidden_seqs"].float()
        k = compute_kappa(hs)
        all_kappa.append(k)
        n_total += len(hs)
        del d, hs
    kappa = torch.cat(all_kappa)  # (N, 27) — layers 1..27 (28 layers total)
    kappa_mean = kappa.mean(dim=0).numpy()
    kappa_std = kappa.std(dim=0).numpy()
    print(f"Total trajectories: {n_total}", flush=True)

    # Known per-layer Δacc from published results
    # From recovery doc: L8: +20pp, L9: -23pp, L2: +17pp, L3: +13pp, L5: +13pp, L10: +17pp
    known = {
        0: -3, 1: 0, 2: 17, 3: 13, 4: 7, 5: 13,
        6: 0, 7: -5, 8: 20, 9: -23, 10: 17,
        11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
        16: 0, 17: 0, 18: 0, 19: 0, 20: 0,
        21: 0, 22: 0, 23: 0, 24: -10, 25: 0, 26: 0, 27: 0,
    }
    n_layers_7b = 28

    print(f"\n{'='*80}", flush=True)
    print(f"CURVATURE POLARITY TEST — Qwen2.5-7B", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'Layer':>6} {'κ mean':>10} {'κ std':>10} {'Δacc(pp)':>10} {'Prediction':>18} {'CPH':>6}", flush=True)

    results = []
    for li in range(1, n_layers_7b - 1):
        k_m = float(kappa_mean[li - 1])
        k_s = float(kappa_std[li - 1])
        d = known.get(li, 0)
        # CPH: κ < 1 → trim-tab (+Δ), κ > 1 → death (-Δ)
        # But all κ > 1, so use RELATIVE ranking: lower κ = more likely trim-tab
        # Divide layers into tertiles by κ
        pred = "NEUTRAL"
        if d > 5:
            pred = "TRIM-TAB"
        elif d < -5:
            pred = "DEATH"

        results.append((li, k_m, k_s, d, pred))

    # Sort by κ
    results.sort(key=lambda x: x[1])

    # Compute tertiles
    n = len(results)
    tertile_size = n // 3
    for i, (li, k_m, k_s, d, pred) in enumerate(results):
        tertile = "LOW κ " if i < tertile_size else ("HIGH κ" if i >= 2 * tertile_size else "MID κ ")
        # CPH match: lower-κ tertile should have more trim-tabs
        print(f"  L{li:3d}  {k_m:10.4f}  {k_s:10.4f}  {d:+9d}pp  {pred:>18}  {tertile:>6}", flush=True)

    # Analysis by tertile
    tertile_low = results[:tertile_size]
    tertile_mid = results[tertile_size:2 * tertile_size]
    tertile_high = results[2 * tertile_size:]

    print(f"\n{'='*80}", flush=True)
    print(f"Analysis by κ tertile (lower κ → should be more trim-tab)", flush=True)
    print(f"{'='*80}", flush=True)

    for name, tert in [("LOW κ (bottom 9)", tertile_low),
                        ("MID κ", tertile_mid),
                        ("HIGH κ (top 9)", tertile_high)]:
        trim = sum(1 for _, _, _, d, _ in tert if d > 5)
        death = sum(1 for _, _, _, d, _ in tert if d < -5)
        neut = sum(1 for _, _, _, d, _ in tert if -5 <= d <= 5)
        mean_d = np.mean([d for _, _, _, d, _ in tert])
        print(f"  {name:20s}: trim={trim:2d}  death={death:2d}  neut={neut:2d}  mean Δ={mean_d:+5.1f}pp", flush=True)

    # Correlation (all layers)
    kappas = [r[1] for r in results]
    deltas = [r[3] for r in results]
    r_all = np.corrcoef(kappas, deltas)[0, 1]

    # Correlation (only layers with |Δacc| > 5pp — the non-neutral ones)
    sig = [(k, d) for k, nd, _, d, _ in results if abs(d) > 5 or d != 0]
    if len(sig) >= 3:
        ks, ds = zip(*sig)
        r_sig = np.corrcoef(list(ks), list(ds))[0, 1]
    else:
        r_sig = 0.0

    print(f"\n  ρ(κ, Δacc) — all layers:       {r_all:+.4f}", flush=True)
    print(f"  ρ(κ, Δacc) — non-neutral only: {r_sig:+.4f}", flush=True)

    # Specific: does κ rank correctly for L8 vs L9?
    l8_k = kappa_mean[8 - 1]
    l9_k = kappa_mean[9 - 1]
    print(f"\n  κ[L8]={l8_k:.4f} (trim-tab, +20pp)  κ[L9]={l9_k:.4f} (death, -23pp)", flush=True)
    print(f"  κ[L8] < κ[L9]: {l8_k < l9_k} {'✓' if l8_k < l9_k else '✗'}", flush=True)

    # Also check L2 vs L9
    l2_k = kappa_mean[2 - 1]
    print(f"  κ[L2]={l2_k:.4f} (trim-tab, +17pp)  < κ[L9]: {l2_k < l9_k} {'✓' if l2_k < l9_k else '✗'}", flush=True)

    # Check L10 (trim-tab, +17pp) vs L24 (death, -10pp)
    l10_k = kappa_mean[10 - 1]
    l24_k = kappa_mean[24 - 1]
    print(f"  κ[L10]={l10_k:.4f} (trim-tab, +17pp) < κ[L24]={l24_k:.4f} (death, -10pp): {l10_k < l24_k} {'✓' if l10_k < l24_k else '✗'}", flush=True)

    # Check L5 (trim-tab, +13pp) vs L7 (death, -5pp)
    l5_k = kappa_mean[5 - 1]
    l7_k = kappa_mean[7 - 1]
    print(f"  κ[L5]={l5_k:.4f} (trim-tab, +13pp) < κ[L7]={l7_k:.4f} (death, -5pp):  {l5_k < l7_k} {'✓' if l5_k < l7_k else '✗'}", flush=True)

    out = {
        "model": "Qwen2.5-7B",
        "n_trajectories": n_total,
        "per_layer": {str(li): {"kappa": float(kappa_mean[li-1]),
                                 "kappa_std": float(kappa_std[li-1]),
                                 "delta_acc_pp": known.get(li, 0)}
                       for li in range(1, n_layers_7b - 1)},
        "correlation_all": r_all,
        "correlation_sig": r_sig,
        "l8_vs_l9": {"kappa_l8": float(l8_k), "kappa_l9": float(l9_k),
                     "ordered_correctly": bool(l8_k < l9_k)},
    }
    with open("cpt_7b_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to cpt_7b_results.json", flush=True)
    print(f"Time: {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
