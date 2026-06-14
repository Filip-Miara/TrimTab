#!/usr/bin/env python3 -u
"""Train contrastive TTs: one on correct trajectories, one on incorrect.

At inference: v_contrastive = v_correct − v_incorrect
This gives a normative steering signal pointing from incorrect→correct.
"""
from __future__ import annotations

import glob, gc, json, os, sys, time, threading
from queue import Queue
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
DATA_DIR = "/home/filip/Projects/Personal/AI/RankAdaptation/data/qwen25_7b_gen_trajs"
D_INPUT = 3584
N_LAYERS = 28
BS = 64
LR = 3e-4
N_EPOCHS = 30


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
    parser.add_argument("--mode", choices=["correct", "incorrect", "all"], required=True)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--d-input", type=int, default=D_INPUT)
    parser.add_argument("--n-layers", type=int, default=N_LAYERS)
    parser.add_argument("--bs", type=int, default=BS)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., best_tt_incorrect.pt)")
    args = parser.parse_args()

    batch_files = sorted(glob.glob(os.path.join(args.data_dir, "gen_trajs_7b_batch_*.pt")))
    val_file = batch_files.pop(0)
    print(f"Mode: {args.mode} | Train files: {len(batch_files)}, Val: {os.path.basename(val_file)}", flush=True)

    n_train_total = 0
    for f in batch_files:
        d = torch.load(f, map_location="cpu")
        if args.mode == "all":
            mask = torch.ones(len(d["is_correct"]), dtype=torch.bool)
        else:
            mask = d["is_correct"] if args.mode == "correct" else ~d["is_correct"]
        n_train_total += mask.sum().item()
        del d
    print(f"Total {args.mode} trajectories: {n_train_total}", flush=True)

    # Load validation (one batch file), split by correctness
    val_data = torch.load(val_file, map_location="cpu")
    val_mask = val_data["is_correct"] if args.mode == "correct" else (
        torch.ones(len(val_data["is_correct"]), dtype=torch.bool) if args.mode == "all" else ~val_data["is_correct"])
    val_h = val_data["hidden_seqs"][val_mask].float()
    val_v = val_data["velocity_targets"][val_mask].float()
    print(f"Val shape: {val_h.shape}", flush=True)
    if len(val_h) == 0:
        print(f"WARNING: No {args.mode} trajectories in val file! Using first 100 from train instead.", flush=True)
        d = torch.load(batch_files[0], map_location="cpu")
        m = d["is_correct"] if args.mode == "correct" else (
            torch.ones(len(d["is_correct"]), dtype=torch.bool) if args.mode == "all" else ~d["is_correct"])
        val_h = d["hidden_seqs"][m][:100].float()
        val_v = d["velocity_targets"][m][:100].float()
        del d

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
    start_epoch = 0
    best_r2 = -float("inf")

    # Resume from checkpoint if specified
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        if "model_state_dict" in ckpt:
            tt.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"]
            best_r2 = ckpt.get("best_r2", -float("inf"))
            print(f"  Resumed from epoch {start_epoch} (best R²={best_r2:.4f})", flush=True)
        else:
            # Legacy checkpoint (state_dict only)
            tt.load_state_dict(ckpt)
            print(f"  Loaded legacy checkpoint (no optimizer state, starting from epoch 0)", flush=True)

    t0 = time.time()
    val_h_gpu = val_h.to(DEVICE)
    val_v_gpu = val_v.to(DEVICE)

    def file_loader(file_list, out_queue, v_mean, v_std):
        """Background thread: load, mask, normalize, permute — push ready batches."""
        for bf in file_list:
            data = torch.load(bf, map_location="cpu")
            mask = data["is_correct"] if args.mode == "correct" else (
                torch.ones(len(data["is_correct"]), dtype=torch.bool) if args.mode == "all"
                else ~data["is_correct"])
            if mask.sum() == 0:
                del data; continue
            h = data["hidden_seqs"][mask].float()
            v = data["velocity_targets"][mask].float()
            v_norm = (v - v_mean) / v_std
            perm = torch.randperm(len(h))
            out_queue.put((h, v_norm, perm))
            del data, mask
        out_queue.put(None)

    for epoch in range(start_epoch, args.epochs):
        tt.train()
        epoch_loss, n_batches = 0, 0
        batch_q = Queue(maxsize=2)
        loader = threading.Thread(target=file_loader, args=(batch_files, batch_q, v_mean, v_std), daemon=True)
        loader.start()

        while True:
            item = batch_q.get()
            if item is None:
                break
            h, v_norm, perm = item

            h_gpu = h.to(DEVICE)
            v_norm_gpu = v_norm.to(DEVICE)
            for i in range(0, len(h), args.bs):
                idx = perm[i:i + args.bs]
                v_pred = tt(h_gpu[idx])
                loss = nn.functional.mse_loss(v_pred, v_norm_gpu[idx])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(tt.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            del h, v_norm, perm; gc.collect()
        loader.join()

        tt.eval()
        with torch.no_grad():
            v_pred = tt(val_h_gpu)
            r2, mse, cos = compute_metrics(
                v_pred * v_std.to(v_pred.device) + v_mean.to(v_pred.device), val_v_gpu)

        if r2 > best_r2:
            best_r2 = r2
            ckpt = {
                "model_state_dict": tt.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch + 1,
                "best_r2": best_r2,
                "val_r2": r2,
            }
            torch.save(ckpt, f"best_tt_{args.mode}.pt")
            print(f"  ep={epoch+1:2d} loss={epoch_loss/n_batches:.6f} "
                  f"val_r²={r2:.4f} cos={cos:.4f} ✨ BEST {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"  ep={epoch+1:2d} loss={epoch_loss/n_batches:.6f} "
                  f"val_r²={r2:.4f} cos={cos:.4f} {time.time()-t0:.0f}s", flush=True)

    print(f"\nBest {args.mode} val R²: {best_r2:.4f}", flush=True)
    with open(f"tt_{args.mode}_results.json", "w") as f:
        json.dump({"best_val_r2": best_r2, "n_train": n_train_total}, f, indent=2)


if __name__ == "__main__":
    main()
