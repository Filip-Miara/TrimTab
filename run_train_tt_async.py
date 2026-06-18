#!/usr/bin/env python3 -u
"""TT training with async data loading to eliminate GPU idle time.

Loads next batch from disk while GPU computes on current batch.
Uses the full 7B generation trajectories from E: drive.
"""
import sys, os, glob, gc, time, random, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['HF_HOME'] = 'D:\\Datasets'

from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
D_INPUT = 3584
N_LAYERS = 28


class AsyncBatchLoader:
    """Loads batches from disk in a background thread."""
    
    def __init__(self, batch_files, v_mean, v_std, bs=64):
        self.batch_files = batch_files
        self.v_mean = v_mean
        self.v_std = v_std
        self.bs = bs
        self.batch_queue = []  # pre-loaded (h, v) pairs
        self.file_queue = []
        self.lock = threading.Lock()
        self.stop_flag = False
        self.thread = None
    
    def start(self):
        self.thread = threading.Thread(target=self._loader_loop, daemon=True)
        self.thread.start()
        # Pre-load first 2 batches
        self._refill()
        self._refill()
    
    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=5)
    
    def _refill(self):
        """Load the next file and enqueue its batches."""
        if not self.file_queue:
            return
        bf = self.file_queue.pop(0)
        data = torch.load(bf, map_location="cpu")
        h_seq = data["hidden_seqs"].float()
        v_tgt = data["velocity_targets"].float()
        v_norm = (v_tgt - self.v_mean) / self.v_std
        del data, v_tgt
        
        # Split into batches
        n = h_seq.shape[0]
        batches = []
        for start in range(0, n, self.bs):
            end = min(start + self.bs, n)
            batches.append((h_seq[start:end].clone(), v_norm[start:end].clone()))
        
        with self.lock:
            self.batch_queue.extend(batches)
        del h_seq, v_norm
    
    def _loader_loop(self):
        while not self.stop_flag:
            if len(self.file_queue) > 0 and len(self.batch_queue) < 20:
                self._refill()
            else:
                time.sleep(0.01)
    
    def get_batch(self):
        """Get next (h, v) pair, blocks if empty."""
        while True:
            with self.lock:
                if self.batch_queue:
                    return self.batch_queue.pop(0)
            time.sleep(0.005)


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
    parser.add_argument("--data-dir", type=str, 
                        default="E:\\Datasets\\project_data\\qwen25_7b_gen_trajs")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--max-train-files", type=int, default=80)
    parser.add_argument("--val-batch-idx", type=int, default=0)
    args = parser.parse_args()
    
    # Find batch files
    pattern = os.path.join(args.data_dir, "gen_trajs_7b_batch_*.pt")
    batch_files = sorted(glob.glob(pattern))
    if not batch_files:
        # Try alternative naming
        batch_files = sorted(glob.glob(os.path.join(args.data_dir, "batch_*.pt")))
    
    print(f"Found {len(batch_files)} batch files in {args.data_dir}", flush=True)
    val_file = batch_files.pop(args.val_batch_idx)
    train_files = batch_files[:args.max_train_files]
    print(f"Train files: {len(train_files)}, Val file: {os.path.basename(val_file)}", flush=True)
    
    n_total = sum(torch.load(f, map_location='cpu')['hidden_seqs'].shape[0] for f in train_files)
    print(f"Training trajectories: {n_total}", flush=True)
    
    # Load validation
    print("Loading validation data...", flush=True)
    t0 = time.time()
    val_data = torch.load(val_file, map_location="cpu")
    val_h = val_data["hidden_seqs"].float()
    val_v = val_data["velocity_targets"].float()
    del val_data
    print(f"Val shape: {val_h.shape} (loaded in {time.time()-t0:.1f}s)", flush=True)
    
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
    t_start = time.time()
    
    val_h_gpu = val_h.to(DEVICE)
    val_v_gpu = val_v_norm.to(DEVICE)
    
    for epoch in range(args.epochs):
        tt.train()
        epoch_loss = 0.0
        n_batches = 0
        random.shuffle(train_files)
        
        # Initialize async loader
        loader = AsyncBatchLoader(list(train_files), v_mean, v_std, args.bs)
        loader.file_queue = list(train_files)
        loader.start()
        
        total_minibatches = sum(
            torch.load(f, map_location='cpu')['hidden_seqs'].shape[0] // args.bs + 1
            for f in train_files
        )
        
        for mb_idx in range(total_minibatches):
            h_cpu, v_cpu = loader.get_batch()
            h_b = h_cpu.to(DEVICE, non_blocking=True)
            v_b = v_cpu.to(DEVICE, non_blocking=True)
            
            v_pred = tt(h_b)
            loss = torch.nn.functional.mse_loss(v_pred, v_b)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tt.parameters(), 1.0)
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            if (mb_idx + 1) % 50 == 0:
                print(f"  ep {epoch+1} [{mb_idx}/{total_minibatches}] "
                      f"loss={epoch_loss/max(n_batches,1):.4f} "
                      f"({time.time()-t_start:.0f}s)", flush=True)
        
        loader.stop()
        
        # Validation
        tt.eval()
        with torch.no_grad():
            v_pred = tt(val_h_gpu)
            r2, mse, cos = compute_metrics(v_pred, val_v_gpu)
        
        print(f"  Epoch {epoch+1}: train_loss={epoch_loss/n_batches:.4f} "
              f"val_r2={r2:.4f} val_cos={cos:.4f} ({time.time()-t_start:.0f}s)", flush=True)
        
        if r2 > best_r2:
            best_r2 = r2
            torch.save(tt.state_dict(), "best_tt_awq_7b.pt")
            print(f"  -> Saved best_tt_awq_7b.pt (val_r2={r2:.4f})", flush=True)
    
    print(f"\nDone. Best val_r2={best_r2:.4f} ({time.time()-t_start:.0f}s)", flush=True)
    print(f"Checkpoint: best_tt_awq_7b.pt", flush=True)


if __name__ == "__main__":
    main()
