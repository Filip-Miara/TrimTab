"""Optimization setup for TrimTab/RankAdaptation.

Apply these at import time to get zero-cost performance improvements.
Call setup_optimizations() at the start of every script.
"""
import torch
import os
import sys


def setup_optimizations():
    """Apply all zero-cost optimizations. Call once at script start."""
    
    # 1. Fix CUDA allocator (Windows/WDDM does not support expandable_segments)
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").startswith("expandable_segments"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 2. Enable TF32 matmul (~2x speedup over FP32)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 3. Enable cuDNN autotuning
    torch.backends.cudnn.benchmark = True
    
    # 4. Set HF cache to D: drive (fast SSD with space)
    os.environ.setdefault("HF_HOME", "D:\\Datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "D:\\Datasets")
    
    # Print status (optional)
    print(f"[opt] TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"[opt] cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"[opt] Float32 precision: {torch.get_float32_matmul_precision()}")
    print(f"[opt] HF_HOME: {os.environ.get('HF_HOME', 'default')}")
    print(f"[opt] PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
    
    # Check for AWQ model availability
    awq_path = "D:\\Qwen2.5-7B-AWQ\\qwen7b_awq"
    if os.path.exists(awq_path):
        print(f"[opt] AWQ Qwen2.5-7B model: AVAILABLE at {awq_path}")
    else:
        print(f"[opt] AWQ Qwen2.5-7B model: NOT FOUND")


def get_awq_model_path() -> str:
    """Return the path to the local AWQ model if available."""
    path = "D:\\Qwen2.5-7B-AWQ\\qwen7b_awq"
    if os.path.exists(path):
        return path
    return None


def compile_trajectory_transformer(model, mode="reduce-overhead"):
    """Compile a TrajectoryTransformer for ~3x speedup.
    
    Usage:
        tt = TrajectoryTransformer(...)
        tt = compile_trajectory_transformer(tt)
    """
    if hasattr(torch, 'compile'):
        print(f"[opt] Compiling TT with mode={mode}...")
        return torch.compile(model, mode=mode)
    return model


def benchmark_cuda():
    """Quick CUDA benchmark to verify optimizations are working."""
    import time
    
    # Matrix multiplication benchmark
    sizes = [(1024, 3584, 1024), (512, 3584, 576)]
    for M, N, K in sizes:
        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(10):
            c = a @ b
        torch.cuda.synchronize()
        
        # Timed
        t0 = time.perf_counter()
        for _ in range(100):
            c = a @ b
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / 100 * 1000
        
        tflops = 2 * M * N * K / 1e9 / (t / 1000)
        print(f"[opt]  ({M:>4}x{N:>4}x{K:>4}): {t:.2f}ms = {tflops:.0f} TFLOPS")
    
    # Memory bandwidth
    x = torch.randn(1000, 3584, device='cuda', dtype=torch.bfloat16)
    t0 = time.perf_counter()
    for _ in range(1000):
        y = x + 1
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 1000 * 1000
    bw = 1000 * 3584 * 2 * 2 / 1e9 / (t / 1000)  # GB/s (read+write)
    print(f"[opt]  Memory bandwidth: ~{bw:.0f} GB/s")
    
    return True
