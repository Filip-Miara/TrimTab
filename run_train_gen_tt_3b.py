#!/usr/bin/env python3 -u
"""Train TT with prefetch queue (sliding window over mmap)."""
from __future__ import annotations

import gc, json, os, sys, time, threading
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
RMMAP_DIR = "/mnt/windows/trajs_rmmap"
D_INPUT = 2048; N_LAYERS = 36; BS = 128; LR = 3e-4; N_EPOCHS = 30
CHUNK_SIZE = 4000  # ~1.2GB per chunk → 2 in VRAM = 2.4GB (fits in 2.9GB free)

def compute_metrics(v_pred, v_target):
    v_p, v_t = v_pred.float(), v_target.float()
    mse = (v_p - v_t).pow(2).mean().item()
    var = v_t.var().item()
    r2 = 1.0 - mse / max(var, 1e-8)
    cos = (v_p * v_t).sum(-1) / (v_p.norm(dim=-1) * v_t.norm(dim=-1) + 1e-8)
    return r2, mse, cos.mean().item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=BS)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--d-model", type=int, default=1280)
    parser.add_argument("--d-ff-ratio", type=int, default=2)
    args = parser.parse_args()

    meta = json.load(open(f"{RMMAP_DIR}/meta.json"))
    n_train = meta["n_train"]
    print(f"Total trajectories: {n_train}, chunk size: {CHUNK_SIZE}", flush=True)

    # Open mmap (only for sequential chunk reads, not per-batch)
    h_mmap = np.memmap(f"{RMMAP_DIR}/train_h.bin", dtype="float16", mode="r", shape=(n_train, N_LAYERS, D_INPUT))
    v_mmap = np.memmap(f"{RMMAP_DIR}/train_v.bin", dtype="float16", mode="r", shape=(n_train, N_LAYERS, D_INPUT))

    # Load val
    val_n = json.load(open(f"{RMMAP_DIR}/meta.json"))["n_val"]
    val_h_np = np.memmap(f"{RMMAP_DIR}/val_h.bin", dtype="float16", mode="r", shape=(val_n, N_LAYERS, D_INPUT))
    val_v_np = np.memmap(f"{RMMAP_DIR}/val_v.bin", dtype="float16", mode="r", shape=(val_n, N_LAYERS, D_INPUT))
    val_h = torch.from_numpy(val_h_np[:500].copy()).float()
    val_v = torch.from_numpy(val_v_np[:500].copy()).float()

    # Per-layer normalization stats from first 10K
    print("Computing per-layer stats...", flush=True)
    stat_v = torch.from_numpy(v_mmap[:10000].copy()).float()
    v_mean = stat_v.mean(dim=0, keepdim=True)
    v_std = stat_v.std(dim=0, keepdim=True) + 1e-8
    v_mean_gpu = v_mean.to(DEVICE); v_std_gpu = v_std.to(DEVICE)
    val_v_norm = (val_v - v_mean) / v_std
    layer_w = torch.ones(N_LAYERS, device=DEVICE) / N_LAYERS

    # Model
    tt = TrajectoryTransformer(d_model=args.d_model, n_layers=6, n_heads=8,
                                d_ff=args.d_model * args.d_ff_ratio,
                                n_positions=N_LAYERS, d_input=D_INPUT).to(DEVICE)
    n_params = sum(p.numel() for p in tt.parameters())
    print(f"Model: {n_params:,} params (d_model={args.d_model}, d_ff={args.d_model * args.d_ff_ratio})", flush=True)
    opt = torch.optim.AdamW(tt.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device="cuda")
    best_r2 = -float("inf"); t0 = time.time()
    val_h_gpu = val_h.to(DEVICE); val_v_gpu = val_v.to(DEVICE)
    val_v_norm_gpu = val_v_norm.to(DEVICE)

    n_chunks = (n_train + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Processing in {n_chunks} chunks of {CHUNK_SIZE}", flush=True)

    for epoch in range(args.epochs):
        tt.train(); epoch_loss = 0; n_batches = 0
        chunk_order = np.random.permutation(n_chunks)

        def _load_gpu(ch):
            s = ch * CHUNK_SIZE; e = min(s + CHUNK_SIZE, n_train)
            h = torch.from_numpy(h_mmap[s:e].copy()).to(DEVICE)
            v = torch.from_numpy(v_mmap[s:e].copy()).to(DEVICE)
            cp = torch.randperm(len(h), device=DEVICE)
            return h[cp], v[cp]

        # Double-buffer: fill both buffers upfront
        buf = [_load_gpu(chunk_order[0]), _load_gpu(chunk_order[1])]

        for ch_pos in range(n_chunks):
            ch = chunk_order[ch_pos]
            cur_h, cur_v = buf[ch_pos % 2]

            # Start loading next chunk into the other buffer
            loader = None; bi = ch_pos % 2; _r = None
            if ch_pos + 2 < n_chunks:
                next_ch = chunk_order[ch_pos + 2]
                _r = [None, None]
                def _worker(c=next_ch, r=_r):
                    r[0], r[1] = _load_gpu(c)
                loader = threading.Thread(target=_worker, daemon=True)
                loader.start()

            # Train on current buffer
            n_cur = len(cur_h)
            for bi in range(0, n_cur, args.bs):
                be = min(bi + args.bs, n_cur)
                vn = ((cur_v[bi:be] - v_mean_gpu) / v_std_gpu).to(dtype=torch.float16)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    loss = (layer_w * (tt(cur_h[bi:be]) - vn).pow(2).mean(dim=-1)).sum(dim=-1).mean()
                opt.zero_grad(); scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tt.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                epoch_loss += loss.item(); n_batches += 1

            if loader is not None and _r is not None:
                loader.join()
                buf[bi] = (_r[0], _r[1])  # overwrite freed buffer

            print(f"  ep={epoch+1:2d} ch={ch_pos+1}/{n_chunks} loss={epoch_loss/max(n_batches,1):.6f} {time.time()-t0:.0f}s", flush=True)
            gc.collect()

        # Validation
        tt.eval()
        with torch.no_grad():
            v_pred = tt(val_h_gpu)
            r2, mse, cos = compute_metrics(v_pred * v_std_gpu + v_mean_gpu, val_v_gpu)
        if r2 > best_r2:
            best_r2 = r2
            torch.save(tt.state_dict(), "best_gen_tt_3b.pt")
            print(f"  ep={epoch+1:2d} loss={epoch_loss/max(n_batches,1):.6f} val_r²={r2:.4f} cos={cos:.4f} ✨ BEST {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"  ep={epoch+1:2d} loss={epoch_loss/max(n_batches,1):.6f} val_r²={r2:.4f} cos={cos:.4f} {time.time()-t0:.0f}s", flush=True)

    print(f"\nBest val R²: {best_r2:.4f}", flush=True)
    json.dump({"best_val_r2": best_r2, "n_train": n_train}, open("gen_tt_3b_results.json", "w"), indent=2)
    print(f"Saved.", flush=True)

if __name__ == "__main__":
    main()
