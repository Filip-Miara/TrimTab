#!/usr/bin/env python3 -u
"""Stable TT training on AWQ Qwen2.5-7B trajectory data.
Conservative settings: no compile, d_model=512, 4 train files, bs=16.
"""
import sys, os, glob, gc, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['HF_HOME'] = 'D:\\Datasets'
os.environ['TRANSFORMERS_CACHE'] = 'D:\\Datasets'

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
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--max-train-files", type=int, default=4)
    parser.add_argument("--val-batch-idx", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Find batch files (try various naming conventions)
    batch_files = sorted(glob.glob(os.path.join(args.data_dir, "batch_*.pt")))
    if not batch_files:
        batch_files = sorted(glob.glob(os.path.join(args.data_dir, "gen_trajs_7b_batch_*.pt")))
    if not batch_files:
        batch_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
    print(f"Found {len(batch_files)} batch files", flush=True)
    val_file = batch_files.pop(args.val_batch_idx)
    train_files = batch_files[:args.max_train_files]
    
    print(f"Train files: {len(train_files)}, Val file: {os.path.basename(val_file)}", flush=True)
    
    # Quick trajectory count from first file only (all files have same batch size)
    n_per_file = torch.load(train_files[0], map_location='cpu')['hidden_seqs'].shape[0]
    n_total = n_per_file * len(train_files)
    print(f"Training trajectories: ~{n_total} ({len(train_files)} files x ~{n_per_file}/file)", flush=True)
    
    # Load validation
    print("Loading validation data...", flush=True)
    val_data = torch.load(val_file, map_location="cpu")
    val_h = val_data["hidden_seqs"].float()
    val_v = val_data["velocity_targets"].float()
    del val_data
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
    print(f"Model: {n_params:,} params (d_model={args.d_model})", flush=True)
    
    opt = torch.optim.AdamW(tt.parameters(), lr=args.lr, weight_decay=1e-4)
    best_r2 = -float("inf")
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location='cpu')
        tt.load_state_dict(sd)
        # Infer starting epoch from filename if available
        import re
        m = re.search(r'epoch_(\d+)', args.resume)
        if m:
            start_epoch = int(m.group(1))
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}", flush=True)
    
    t_start = time.time()
    
    val_h_gpu = val_h.to(DEVICE)
    val_v_gpu = val_v_norm.to(DEVICE)
    
    for epoch in range(start_epoch, args.epochs):
        tt.train()
        epoch_loss, n_batches = 0, 0
        random.shuffle(train_files)
        
        for bf_idx, bf in enumerate(train_files):
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
                print(f"  ep {epoch+1} [{bf_idx+1}/{len(train_files)}] "
                      f"loss={epoch_loss/max(n_batches,1):.4f} ({time.time()-t_start:.0f}s)", flush=True)
            
            del h_seq, v_tgt, v_tgt_norm
            gc.collect()
            torch.cuda.empty_cache()
        
        # Validation
        tt.eval()
        with torch.no_grad():
            v_pred = tt(val_h_gpu)
            r2, mse, cos = compute_metrics(v_pred, val_v_gpu)
        
        print(f"  Epoch {epoch+1}: train_loss={epoch_loss/n_batches:.4f} "
              f"val_r2={r2:.4f} val_cos={cos:.4f} ({time.time()-t_start:.0f}s)", flush=True)
        
        if r2 > best_r2:
            best_r2 = r2
            best_path = "best_tt_awq_7b.pt"
            torch.save(tt.state_dict(), best_path)
            print(f"  -> Saved {best_path} (val_r2={r2:.4f})", flush=True)
        
        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = f"tt_awq_7b_epoch_{epoch+1}.pt"
            torch.save(tt.state_dict(), ckpt_path)
            print(f"  -> Periodic checkpoint: {ckpt_path}", flush=True)
        
        # Free GPU memory between epochs to prevent leak
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\nDone. Best val_r2={best_r2:.4f} ({time.time()-t_start:.0f}s)", flush=True)
    print(f"Checkpoint: best_tt_awq_7b.pt", flush=True)

if __name__ == "__main__":
    main()
