#!/usr/bin/env python3 -u
"""Train TrajectoryTransformer on generation trajectories.

Memory-efficient: iterates batch files on disk, never loads all at once.
One batch file (5000 trajs) used as validation set.
"""
from __future__ import annotations

import glob, json, os, sys, time, gc
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
DATA_DIR = "/run/media/filip/B522-875D/Datasets/project_data/smolm2_gen_trajs"
D_INPUT = 960
N_LAYERS = 32
BS = 128
LR = 3e-4
N_EPOCHS = 50


def load_batch_file(path, device="cpu"):
    """Load a single batch file. Returns (hidden, velocity) in float32 on device."""
    data = torch.load(path, map_location=device)
    return data["hidden_seqs"].float(), data["velocity_targets"].float()


def compute_metrics(v_pred, v_target):
    v_p = v_pred.float()
    v_t = v_target.float()
    mse = (v_p - v_t).pow(2).mean().item()
    var = v_t.var().item()
    r2 = 1.0 - mse / max(var, 1e-8)
    cos = (v_p * v_t).sum(-1) / (v_p.norm(dim=-1) * v_t.norm(dim=-1) + 1e-8)
    return r2, mse, cos.mean().item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--bs", type=int, default=BS)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--val-batch-idx", type=int, default=0,
                        help="Which batch file to hold out for validation")
    parser.add_argument("--d-model", type=int, default=512)
    args = parser.parse_args()

    batch_files = sorted(glob.glob(os.path.join(args.data_dir, "gen_trajs_batch_*.pt")))
    val_file = batch_files.pop(args.val_batch_idx)
    print(f"Train files: {len(batch_files)}", flush=True)
    print(f"Val file: {val_file} ({os.path.getsize(val_file)/1e6:.0f}MB)", flush=True)

    # Count total training tokens
    n_train_total = 0
    for f in batch_files:
        d = torch.load(f, map_location="cpu")
        n_train_total += len(d["hidden_seqs"])
        del d
    print(f"Total training trajectories: {n_train_total}", flush=True)

    # Load validation set (one batch file → stays in RAM, ~1.2GB float32)
    print(f"Loading validation set...", flush=True)
    val_h, val_v = load_batch_file(val_file)
    print(f"  Val shape: {val_h.shape}", flush=True)

    # Normalize targets using validation stats (avoids loading all training data)
    v_mean = val_v.mean(dim=(0, 1), keepdim=True)
    v_std = val_v.std(dim=(0, 1), keepdim=True) + 1e-8
    val_v_norm = (val_v - v_mean) / v_std

    # Model
    tt = TrajectoryTransformer(
        d_model=args.d_model, n_layers=6, n_heads=8,
        d_ff=args.d_model * 4, n_positions=N_LAYERS, d_input=D_INPUT,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in tt.parameters())
    print(f"\nModel: {n_params:,} params", flush=True)

    opt = torch.optim.AdamW(tt.parameters(), lr=args.lr, weight_decay=1e-4)
    best_r2 = -float("inf")
    t0 = time.time()

    # Move validation to GPU once
    val_h_gpu = val_h.to(DEVICE)
    val_v_gpu = val_v.to(DEVICE)
    val_v_norm_gpu = val_v_norm.to(DEVICE)

    for epoch in range(args.epochs):
        tt.train()
        epoch_loss = 0
        n_batches = 0

        for bf in batch_files:
            h, v = load_batch_file(bf)  # CPU float32
            v_norm = (v - v_mean) / v_std
            perm = torch.randperm(len(h))

            for i in range(0, len(h), args.bs):
                idx = perm[i:i + args.bs]
                h_gpu = h[idx].to(DEVICE)
                v_gpu = v_norm[idx].to(DEVICE)
                v_pred = tt(h_gpu)
                loss = nn.functional.mse_loss(v_pred, v_gpu)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tt.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            del h, v, v_norm, perm
            gc.collect()

        # Validation
        tt.eval()
        with torch.no_grad():
            v_pred = tt(val_h_gpu)
            r2, mse, cos = compute_metrics(
                v_pred * v_std.to(v_pred.device) + v_mean.to(v_pred.device),
                val_v_gpu)

        if r2 > best_r2:
            best_r2 = r2
            torch.save(tt.state_dict(), "best_gen_tt.pt")
            print(f"  ep={epoch+1:2d} loss={epoch_loss/n_batches:.6f} "
                  f"val_r²={r2:.4f} cos={cos:.4f} ✨ BEST {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"  ep={epoch+1:2d} loss={epoch_loss/n_batches:.6f} "
                  f"val_r²={r2:.4f} cos={cos:.4f} {time.time()-t0:.0f}s", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Best val R²: {best_r2:.4f}", flush=True)
    print(f"Params: {n_params:,} | Model: best_gen_tt.pt", flush=True)
    print(f"\nComparison:", flush=True)
    print(f"  Prompt TT (Qwen, 2B, layer-to-layer): R²=0.62", flush=True)
    print(f"  Prompt TT (Qwen, 2B, reasoning):     R²=0.75", flush=True)
    print(f"  Gen TT (SmolLM2-360M):                R²={best_r2:.4f}", flush=True)

    with open("gen_tt_results.json", "w") as f:
        json.dump({"best_val_r2": best_r2, "n_train": n_train_total}, f, indent=2)


if __name__ == "__main__":
    main()
