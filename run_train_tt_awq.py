#!/usr/bin/env python3 -u
"""Train TrajectoryTransformer on AWQ Qwen2.5-7B trajectory data.
Uses D:\trajs_7B_AWQ\Batches batch files with optimizations enabled.
"""
import sys, os, glob, gc, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.optimization import setup_optimizations, compile_trajectory_transformer

setup_optimizations()
import torch
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
DATA_DIR = "D:\\trajs_7B_AWQ\\Batches"
D_INPUT = 3584
N_LAYERS = 28

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
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--val-batch-idx", type=int, default=0)
    args = parser.parse_args()

    # Find batch files - limit to max 8 for memory stability
    batch_files = sorted(glob.glob(os.path.join(args.data_dir, "batch_*.pt")))
    if not batch_files:
        batch_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
    if not batch_files:
        print(f"No batch files found in {args.data_dir}")
        return
    
    val_file = batch_files.pop(args.val_batch_idx)
    max_train_files = min(8, len(batch_files))
    batch_files = batch_files[:max_train_files]
    print(f"Train files: {len(batch_files)}/{max_train_files}, Val file: {os.path.basename(val_file)}", flush=True)

    # Calculate total training trajectories
    n_train_total = 0
    for f in batch_files:
        d = torch.load(f, map_location="cpu")
        n_train_total += d["hidden_seqs"].shape[0]
        del d
    print(f"Total training trajectories: {n_train_total}", flush=True)

    # Load validation
    print(f"Loading validation from {val_file}...", flush=True)
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

    # Compile disabled (access violation on Windows with large compiled graphs)
    compiled = False
    print(f"Model: eager mode ({n_params:,} params)", flush=True)

    opt = torch.optim.AdamW(tt.parameters(), lr=args.lr, weight_decay=1e-4)
    best_r2 = -float("inf")
    t0 = time.time()

    val_h_gpu = val_h.to(DEVICE)
    val_v_gpu = val_v_norm.to(DEVICE)

    for epoch in range(args.epochs):
        tt.train()
        epoch_loss, n_batches = 0, 0
        
        # Shuffle batches
        import random
        random.shuffle(batch_files)
        
        for bf_idx, bf in enumerate(batch_files):
            # Load one batch at a time to avoid OOM
            data = torch.load(bf, map_location="cpu")
            h_seq = data["hidden_seqs"].float()
            v_tgt = data["velocity_targets"].float()
            del data
            v_tgt_norm = (v_tgt - v_mean) / v_std
            
            n = h_seq.shape[0]
            for start in range(0, n, args.bs):
                end = min(start + args.bs, n)
                h_b = h_seq[start:end].to(DEVICE)
                v_b = v_tgt_norm[start:end].to(DEVICE)
                
                v_pred = tt(h_b)
                loss = torch.nn.functional.mse_loss(v_pred, v_b)
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tt.parameters(), 1.0)
                opt.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
            if (bf_idx + 1) % 2 == 0:
                print(f"  epoch {epoch+1} [{bf_idx+1}/{len(batch_files)}] "
                      f"loss={epoch_loss/max(n_batches,1):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            del h_seq, v_tgt, v_tgt_norm
            gc.collect()
        
        # Validation
        tt.eval()
        with torch.no_grad():
            v_pred = tt(val_h_gpu)
            r2, mse, cos = compute_metrics(v_pred, val_v_gpu)
        
        print(f"  Epoch {epoch+1}: train_loss={epoch_loss/n_batches:.4f} "
              f"val_r2={r2:.4f} val_cos={cos:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        
        if r2 > best_r2:
            best_r2 = r2
            save_path = f"best_tt_awq_7b.pt"
            # Save the raw model (unwrap from compile)
            model_to_save = tt._orig_mod if hasattr(tt, '_orig_mod') else tt
            torch.save(model_to_save.state_dict(), save_path)
            print(f"  -> Saved {save_path} (val_r2={r2:.4f})", flush=True)

    print(f"\nTraining complete. Best val_r2={best_r2:.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"Final checkpoint: best_tt_awq_7b.pt", flush=True)

if __name__ == "__main__":
    main()
