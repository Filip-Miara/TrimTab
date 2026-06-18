#!/usr/bin/env python3 -u
"""TT training with triple-buffered GPU VRAM cache.

Pattern: GPU processes buffer N from VRAM, background thread loads
buffer N+1 into VRAM, buffer N-2 is freed. Zero per-batch CPU->GPU
transfer, zero GPU idle waiting for disk.
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
N_GPU_BUFFERS = 3  # triple buffering


def compute_metrics(v_pred, v_target):
    v_p = v_pred.float()
    v_t = v_target.float()
    mse = (v_p - v_t).pow(2).mean().item()
    var = v_t.var().item()
    r2 = 1.0 - mse / max(var, 1e-8)
    cos = (v_p * v_t).sum(-1) / (v_p.norm(dim=-1) * v_t.norm(dim=-1) + 1e-8)
    return r2, mse, cos.mean().item()


class VRAMFileCache:
    """Manages N GPU buffers for file-level data prefetching.
    
    Producer thread loads files from disk into GPU buffers.
    Consumer (main thread) trains on ready buffers.
    """
    
    def __init__(self, file_paths, v_mean, v_std, n_buffers=N_GPU_BUFFERS):
        self.file_paths = list(file_paths)
        self.v_mean = v_mean
        self.v_std = v_std
        self.n_buffers = n_buffers
        
        # State tracking
        self.buffers = [None] * n_buffers  # each: (h_gpu, v_gpu, file_name)
        self.buffer_state = ['empty'] * n_buffers  # empty, loading, ready, in_use
        self.locks = [threading.Lock() for _ in range(n_buffers)]
        self.file_queue = list(file_paths)
        self.stop_flag = False
        self.thread = None
        self.load_count = 0
    
    def start(self):
        self.thread = threading.Thread(target=self._loader_loop, daemon=True)
        self.thread.start()
        # Pre-load first 2 buffers
        for _ in range(min(2, len(self.file_paths))):
            self._load_next_available()
    
    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=10)
    
    def _find_empty_buffer(self):
        for i in range(self.n_buffers):
            with self.locks[i]:
                if self.buffer_state[i] == 'empty':
                    return i
        return -1
    
    def _find_ready_buffer(self):
        for i in range(self.n_buffers):
            with self.locks[i]:
                if self.buffer_state[i] == 'ready':
                    return i
        return -1
    
    def _load_into_buffer(self, buf_idx, file_path):
        """Load a file from disk into GPU buffer."""
        data = torch.load(file_path, map_location="cpu")
        h_seq = data["hidden_seqs"].float()
        v_tgt = data["velocity_targets"].float()
        del data
        
        v_norm = (v_tgt - self.v_mean) / self.v_std
        del v_tgt
        
        # Transfer directly to GPU
        h_gpu = h_seq.to(DEVICE, non_blocking=True)
        v_gpu = v_norm.to(DEVICE, non_blocking=True)
        del h_seq, v_norm
        
        with self.locks[buf_idx]:
            self.buffers[buf_idx] = (h_gpu, v_gpu, os.path.basename(file_path))
            self.buffer_state[buf_idx] = 'ready'
            self.load_count += 1
    
    def _load_next_available(self):
        """Load the next file into an empty buffer."""
        if not self.file_queue:
            return False
        buf_idx = self._find_empty_buffer()
        if buf_idx < 0:
            return False
        file_path = self.file_queue.pop(0)
        with self.locks[buf_idx]:
            self.buffer_state[buf_idx] = 'loading'
        
        # Do the actual load (may take time, runs in background thread)
        self._load_into_buffer(buf_idx, file_path)
        return True
    
    def _loader_loop(self):
        """Background thread: continuously fill empty buffers."""
        while not self.stop_flag:
            loaded = self._load_next_available()
            if not loaded:
                time.sleep(0.05)
    
    def acquire_ready_file(self):
        """Get the next ready file buffer. Blocks until one is available."""
        while not self.stop_flag:
            buf_idx = self._find_ready_buffer()
            if buf_idx >= 0:
                with self.locks[buf_idx]:
                    h_gpu, v_gpu, fname = self.buffers[buf_idx]
                    self.buffer_state[buf_idx] = 'in_use'
                return h_gpu, v_gpu, fname, buf_idx
            time.sleep(0.01)
        return None, None, None, -1
    
    def release_buffer(self, buf_idx):
        """Signal that we're done with this buffer (frees VRAM)."""
        with self.locks[buf_idx]:
            self.buffers[buf_idx] = None
            self.buffer_state[buf_idx] = 'empty'
        torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="D:\\project_data\\qwen25_7b_gen_trajs")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--max-train-files", type=int, default=80)
    parser.add_argument("--val-batch-idx", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    # Find batch files
    for pattern in ["gen_trajs_7b_batch_*.pt", "batch_*.pt", "*.pt"]:
        batch_files = sorted(glob.glob(os.path.join(args.data_dir, pattern)))
        if batch_files:
            break
    
    print(f"Found {len(batch_files)} batch files in {args.data_dir}", flush=True)
    val_file = batch_files.pop(args.val_batch_idx)
    train_files = batch_files[:args.max_train_files]
    print(f"Train files: {len(train_files)}, Val file: {os.path.basename(val_file)}", flush=True)
    
    # Quick count from first file
    n_per = torch.load(train_files[0], map_location='cpu')['hidden_seqs'].shape[0]
    print(f"Training trajectories: ~{n_per * len(train_files)}", flush=True)
    
    # Load validation data
    print("Loading validation data...", flush=True)
    t0 = time.time()
    val_data = torch.load(val_file, map_location="cpu")
    val_h = val_data["hidden_seqs"].float()
    val_v = val_data["velocity_targets"].float()
    del val_data
    print(f"Val shape: {val_h.shape} ({time.time()-t0:.1f}s)", flush=True)
    
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
        print(f"Resumed from {args.resume}", flush=True)
    
    t_start = time.time()
    val_h_gpu = val_h.to(DEVICE)
    val_v_gpu = val_v_norm.to(DEVICE)
    
    for epoch in range(start_epoch, args.epochs):
        tt.train()
        epoch_loss = 0.0
        n_batches = 0
        random.shuffle(train_files)
        
        # Initialize VRAM cache
        cache = VRAMFileCache(list(train_files), v_mean, v_std, N_GPU_BUFFERS)
        cache.start()
        
        files_processed = 0
        total_files = len(train_files)
        
        while files_processed < total_files:
            # Get next file from VRAM cache (blocks until ready)
            h_gpu, v_gpu, fname, buf_idx = cache.acquire_ready_file()
            if h_gpu is None:
                break
            
            n = h_gpu.shape[0]
            for start in range(0, n, args.bs):
                end = min(start + args.bs, n)
                h_b = h_gpu[start:end]   # view, no copy
                v_b = v_gpu[start:end]   # view, no copy
                
                v_pred = tt(h_b)
                loss = torch.nn.functional.mse_loss(v_pred, v_b)
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tt.parameters(), 1.0)
                opt.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            files_processed += 1
            cache.release_buffer(buf_idx)
            
            if files_processed % 4 == 0 or files_processed == total_files:
                print(f"  ep {epoch+1} [{files_processed}/{total_files}] "
                      f"loss={epoch_loss/max(n_batches,1):.4f} "
                      f"({time.time()-t_start:.0f}s, VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB)", flush=True)
        
        cache.stop()
        
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
        
        if (epoch + 1) % 5 == 0:
            torch.save(tt.state_dict(), f"tt_awq_7b_epoch_{epoch+1}.pt")
        
        # Clear GPU cache between epochs
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\nDone. Best val_r2={best_r2:.4f} ({time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
