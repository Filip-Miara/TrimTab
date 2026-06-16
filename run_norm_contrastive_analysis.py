#!/usr/bin/env python3 -u
"""Norm-growth baseline + contrastive similarity analysis.

Two experiments from TSE Phase 8 (Null Hypothesis Tests):

Experiment 1 — Norm-growth baseline (H0-2):
  Decompose TT prediction into magnitude vs direction components.
  Train norm-only model to test if TT learns meaningful directions.
  Data: 0.8B trajectories, TT at R²=0.8775

Experiment 2 — Contrastive similarity (H0-1):
  Compute cosine similarity between v_correct and v_incorrect predictions.
  Data: 7B trajectories, 3 TT checkpoints (standard, correct, incorrect)
"""
from __future__ import annotations

import gc, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"


def compute_r2(v_pred, v_true):
    v_p = v_pred.float()
    v_t = v_true.float()
    mse = (v_p - v_t).pow(2).mean().item()
    var = v_t.var().item()
    return 1.0 - mse / max(var, 1e-8)


def compute_cosine_similarity(a, b):
    a_f = a.float().flatten(1)
    b_f = b.float().flatten(1)
    cos = (a_f * b_f).sum(-1) / (a_f.norm(dim=-1) * b_f.norm(dim=-1) + 1e-8)
    return cos.mean().item()


def experiment1_norm_growth():
    """Decompose 0.8B TT predictions into norm vs direction."""
    print("=" * 60, flush=True)
    print("EXPERIMENT 1: Norm-Growth Baseline (H0-2)", flush=True)
    print("=" * 60, flush=True)

    DATA_DIR = "/home/filip/Projects/Personal/AI/RankAdaptation/data/qwen35_08b_trajs"
    files = sorted(glob.glob(os.path.join(DATA_DIR, "batch_*.pt")))
    val_file = files.pop(5)  # same held-out as training

    vd = torch.load(val_file, map_location="cpu")
    val_h = vd["hidden_seqs"].float()[:500]
    val_v = vd["velocity_targets"].float()[:500]
    del vd
    print(f"Validation data: {val_h.shape}", flush=True)

    # Normalize targets
    v_mean = val_v.mean(dim=(0, 1), keepdim=True)
    v_std = val_v.std(dim=(0, 1), keepdim=True) + 1e-8
    val_v_norm = (val_v - v_mean) / v_std

    # Load trained TT
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=24, d_input=1024).to(DEVICE)
    tt.load_state_dict(torch.load("best_tt_08b.pt", map_location="cpu"), strict=False)
    tt.to(DEVICE)
    tt.eval()
    n_params = sum(p.numel() for p in tt.parameters())
    print(f"TT params: {n_params:,}", flush=True)

    val_h_gpu = val_h.to(DEVICE)
    val_v_gpu = val_v.to(DEVICE)
    val_v_norm_gpu = val_v_norm.to(DEVICE)

    with torch.no_grad():
        v_pred_norm = tt(val_h_gpu)

    # Full vector R²
    r2_full = compute_r2(v_pred_norm * v_std.to(DEVICE) + v_mean.to(DEVICE), val_v_gpu)
    print(f"\nFull-vector R²:     {r2_full:.4f}", flush=True)

    # Norm-only R²: compare predicted norm vs true norm
    pred_norm = v_pred_norm.norm(dim=-1)
    true_norm = val_v_norm_gpu.norm(dim=-1)
    mse_norm = (pred_norm - true_norm).pow(2).mean().item()
    var_norm = true_norm.var().item()
    r2_norm = 1.0 - mse_norm / max(var_norm, 1e-8)
    print(f"Norm-only R²:       {r2_norm:.4f}", flush=True)

    # Directional R²: compare normalized vectors
    pred_dir = F.normalize(v_pred_norm, dim=-1, eps=1e-8)
    true_dir = F.normalize(val_v_norm_gpu, dim=-1, eps=1e-8)
    mse_dir = (pred_dir - true_dir).pow(2).mean().item()
    var_dir = 1.0  # normalized vectors have unit variance
    r2_dir = 1.0 - mse_dir / var_dir
    print(f"Directional R²:     {r2_dir:.4f}", flush=True)

    # Cosine similarity between predicted and true velocity
    cos_sim = compute_cosine_similarity(v_pred_norm, val_v_norm_gpu)
    print(f"Pred-True cos sim:  {cos_sim:.4f}", flush=True)

    # Train a norm-only model for fair comparison
    print(f"\nTraining norm-only baseline...", flush=True)
    norm_model = nn.Sequential(
        nn.Linear(24 * 1024, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 24),
    ).to(DEVICE)

    opt = torch.optim.AdamW(norm_model.parameters(), lr=1e-3)
    n_epochs = 20

    # Training data
    train_files = files  # all remaining files
    for ep in range(n_epochs):
        norm_model.train()
        for f in train_files:
            d = torch.load(f, map_location="cpu")
            h = d["hidden_seqs"].float()
            v = d["velocity_targets"].float()
            v_n = (v - v_mean) / v_std

            perm = torch.randperm(len(h))
            for i in range(0, len(h), 256):
                idx = perm[i:i + 256]
                h_b = h[idx].to(DEVICE)
                v_b = v_n[idx].to(DEVICE)
                h_flat = h_b.reshape(len(idx), -1)
                norm_pred = norm_model(h_flat)  # (B, 24) scalar norms
                # Use global average direction per layer
                avg_dir = F.normalize(v_pred_norm.mean(dim=0, keepdim=True), dim=-1, eps=1e-8)
                v_pred = norm_pred.unsqueeze(-1) * avg_dir  # (B, 24, D)
                loss = F.mse_loss(v_pred, v_b)
                opt.zero_grad()
                loss.backward()
                opt.step()
            del d, h, v, v_n

        # Validation
        norm_model.eval()
        with torch.no_grad():
            h_flat_val = val_h_gpu.reshape(val_h.shape[0], -1)
            norm_pred_val = norm_model(h_flat_val)
            avg_dir_val = F.normalize(v_pred_norm.mean(dim=0, keepdim=True), dim=-1, eps=1e-8)
            v_pred_norm_only = norm_pred_val.unsqueeze(-1) * avg_dir_val
            r2_norm_model = compute_r2(v_pred_norm_only, val_v_norm_gpu)
        print(f"  ep={ep+1:2d} norm-only val R²={r2_norm_model:.4f}", flush=True)

    # Un-normalize for final comparison
    with torch.no_grad():
        v_pred_norm_only_raw = v_pred_norm_only * v_std.to(DEVICE) + v_mean.to(DEVICE)
    r2_norm_model_raw = compute_r2(v_pred_norm_only_raw, val_v_gpu)
    print(f"\nNorm-only model R² (raw): {r2_norm_model_raw:.4f}", flush=True)

    # Random direction baseline
    with torch.no_grad():
        rand_dir = F.normalize(torch.randn_like(v_pred_norm), dim=-1)
        v_pred_rand = v_pred_norm.norm(dim=-1, keepdim=True) * rand_dir
        v_pred_rand_raw = v_pred_rand * v_std.to(DEVICE) + v_mean.to(DEVICE)
    r2_rand = compute_r2(v_pred_rand_raw, val_v_gpu)
    print(f"Random-direction R²:    {r2_rand:.4f}", flush=True)

    results = {
        "full_r2": r2_full,
        "norm_r2": r2_norm,
        "directional_r2": r2_dir,
        "pred_true_cos_sim": cos_sim,
        "norm_model_r2": r2_norm_model_raw,
        "random_direction_r2": r2_rand,
    }

    with open("norm_growth_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to norm_growth_analysis.json", flush=True)
    return results


def experiment2_contrastive_similarity():
    """Compute cosine similarity between v_correct and v_incorrect."""
    print("\n" + "=" * 60, flush=True)
    print("EXPERIMENT 2: Contrastive Similarity (H0-1)", flush=True)
    print("=" * 60, flush=True)

    DATA_DIR = "/run/media/filip/B522-875D/Datasets/project_data/qwen25_7b_gen_trajs"
    files = sorted(glob.glob(os.path.join(DATA_DIR, "gen_trajs_7b_batch_*.pt")))
    if not files:
        print("ERROR: 7B trajectory data not found!", flush=True)
        return None
    print(f"Found {len(files)} 7B batch files", flush=True)

    # Load one validation file
    val_file = files[0]
    vd = torch.load(val_file, map_location="cpu")
    val_h = vd["hidden_seqs"].float()
    val_v = vd["velocity_targets"].float()
    print(f"7B Validation data: {val_h.shape}", flush=True)

    v_mean = val_v.mean(dim=(0, 1), keepdim=True)
    v_std = val_v.std(dim=(0, 1), keepdim=True) + 1e-8
    val_v_norm = (val_v - v_mean) / v_std

    checkpoints = {
        "standard": "best_gen_tt_7b.pt",
        "correct": "best_tt_correct.pt",
        "incorrect": "best_tt_incorrect.pt",
        "all": "best_tt_all.pt",
    }

    results = {"model_r2": {}, "cosine_similarities": {}}
    predictions = {}

    for name, ckpt in checkpoints.items():
        if not os.path.exists(ckpt):
            print(f"  Skipping {name} ({ckpt} not found)", flush=True)
            continue

        print(f"\nLoading {name} TT from {ckpt}...", flush=True)
        tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                    n_positions=28, d_input=3584).to(DEVICE)
        tt.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        tt.eval()

        val_h_gpu = val_h.to(DEVICE)
        with torch.no_grad():
            v_pred = tt(val_h_gpu[:500])
        predictions[name] = v_pred.cpu()

        r2 = compute_r2(v_pred, val_v_norm[:500].to(DEVICE))
        results["model_r2"][name] = r2
        n_params = sum(p.numel() for p in tt.parameters())
        print(f"  {name}: R²={r2:.4f} ({n_params:,} params)", flush=True)

        del tt, val_h_gpu
        gc.collect()
        torch.cuda.empty_cache()

    if len(predictions) < 2:
        print("Need at least 2 models for comparison", flush=True)
        return results

    model_names = list(predictions.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            cos = compute_cosine_similarity(predictions[a], predictions[b])
            results["cosine_similarities"][f"{a}_vs_{b}"] = cos
            print(f"  cos({a}, {b}) = {cos:.4f}", flush=True)

    with open("contrastive_similarity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to contrastive_similarity.json", flush=True)
    return results


if __name__ == "__main__":
    t0 = time.time()

    results1 = experiment1_norm_growth()
    gc.collect()
    torch.cuda.empty_cache()

    results2 = experiment2_contrastive_similarity()

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"Total time: {elapsed:.0f} min", flush=True)
    print(f"{'='*60}", flush=True)
