#!/usr/bin/env python3 -u
"""Train TrajectoryTransformer on Qwen2.5-7B generation trajectories.
Memory-efficient: iterates batch files on disk.
"""
from __future__ import annotations

import glob, gc, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
DATA_DIR = "/run/media/filip/B522-875D/Datasets/project_data/qwen25_7b_gen_trajs"
D_INPUT = 3584
N_LAYERS = 28
BS = 64
LR = 3e-4
N_EPOCHS = 50


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
    parser.add_argument("--val-batch-idx", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=768)
    args = parser.parse_args()

    batch_files = sorted(glob.glob(os.path.join(args.data_dir, "gen_trajs_7b_batch_*.pt")))
    val_file = batch_files.pop(args.val_batch_idx)
    print(f"Train files: {len(batch_files)}, Val file: {os.path.basename(val_file)}", flush=True)

    n_train_total = sum(len(torch.load(f, map_location="cpu")["hidden_seqs"]) for f in batch_files)
    print(f"Total training trajectories: {n_train_total}", flush=True)

    # Load validation on CPU
    val_data = torch.load(val_file, map_location="cpu")
    val_h = val_data["hidden_seqs"].float()
    val_v = val_data["velocity_targets"].float()
    print(f"Val shape: {val_h.shape}", flush=True)

    # Normalize targets
    v_mean = val_v.mean(dim=(0, 1), keepdim=True)
    v_std = val_v.std(dim=(0, 1), keepdim=True) + 1e-8
    val_v_norm = (val_v - v_mean) / v_std

    # Model
    tt = TrajectoryTransformer(
        d_model=args.d_model, n_layers=6, n_heads=8,
        d_ff=args.d_model * 4, n_positions=N_LAYERS, d_input=D_INPUT,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in tt.parameters())
    print(f"Model: {n_params:,} params", flush=True)

    opt = torch.optim.AdamW(tt.parameters(), lr=args.lr, weight_decay=1e-4)
    best_r2 = -float("inf")
    t0 = time.time()

    val_h_gpu = val_h.to(DEVICE)
    val_v_gpu = val_v.to(DEVICE)

    for epoch in range(args.epochs):
        tt.train()
        epoch_loss, n_batches = 0, 0

        for bf in batch_files:
            data = torch.load(bf, map_location="cpu")
            h, v = data["hidden_seqs"].float(), data["velocity_targets"].float()
            v_norm = (v - v_mean) / v_std
            perm = torch.randperm(len(h))

            for i in range(0, len(h), args.bs):
                idx = perm[i:i + args.bs]
                v_pred = tt(h[idx].to(DEVICE))
                loss = nn.functional.mse_loss(v_pred, v_norm[idx].to(DEVICE))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(tt.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            del h, v, v_norm, perm, data; gc.collect()

        tt.eval()
        with torch.no_grad():
            v_pred = tt(val_h_gpu)
            r2, mse, cos = compute_metrics(
                v_pred * v_std.to(v_pred.device) + v_mean.to(v_pred.device), val_v_gpu)

        if r2 > best_r2:
            best_r2 = r2
            torch.save(tt.state_dict(), "best_gen_tt_7b.pt")
            print(f"  ep={epoch+1:2d} loss={epoch_loss/n_batches:.6f} "
                  f"val_r²={r2:.4f} cos={cos:.4f} ✨ BEST {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"  ep={epoch+1:2d} loss={epoch_loss/n_batches:.6f} "
                  f"val_r²={r2:.4f} cos={cos:.4f} {time.time()-t0:.0f}s", flush=True)

    print(f"\nBest val R²: {best_r2:.4f}", flush=True)
    with open("gen_tt_7b_results.json", "w") as f:
        json.dump({"best_val_r2": best_r2, "n_train": n_train_total}, f, indent=2)


if __name__ == "__main__":
    main()
