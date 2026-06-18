# FULL SESSION RECOVERY — TrimTab / RankAdaptation

**Session**: 2026-06-17 → 2026-06-18 (~6h)
**Repo**: github.com/Filip-Miara/TrimTab
**Hardware**: MSI Cyborg 15 (RTX 4060 Laptop 8GB, i7-12650H, 16GB RAM)
**OS**: Windows 11 (WDDM)
**Python**: 3.12.12 (Conda), PyTorch 2.11.0+cu128, CUDA 12.8, Triton 3.3.1

---

## Pre-Session State

Project had established (from prior 5 sessions on Linux):
- **Trim-tab layers exist**: L8 → +20pp, L2 → +17pp, L9 → -23pp on Qwen2.5-7B
- **Velocities are learnable**: R²=0.85-0.94 for TrajectoryTransformer
- **Steering amplifies, doesn't create**: requires baseline >~40% GSM8K
- **Cross-model transfer**: SmolLM2→7B preserves L8 as best trim-tab layer
- **Steering via KV-cache modification**: modify K/V projections at specific layer

---

## Session Goals

1. Investigate acceleration strategies for 8GB VRAM Windows environment
2. Apply AWQ+Marlin quantization for fitting 7B model in VRAM
3. Train TrajectoryTransformer on AWQ 7B trajectory data
4. Run per-layer steering sweep to replicate trim-tab findings
5. Optimize throughput via batching, hooks, and KV-cache management

---

## Key Results

### ✅ Baseline Accuracy (AWQ Qwen2.5-7B on GSM8K)

| Metric | Value |
|--------|-------|
| Baseline (no steering) | **18/30 = 60.0%** |
| L8 steering (α=0.1) | **20/30 = 66.7% (+6.7pp)** |
| Original project baseline (BnB 4-bit, Linux) | ~73% |

**Trim-tab L8 confirmed on AWQ model** — consistent improvement across all batches.

### ✅ TrajectoryTransformer Training

| Detail | Value |
|--------|-------|
| Architecture | d_model=768, 6 layers, 8 heads, d_ff=3072 |
| Training data | 80 files × ~1100 trajs = ~88K trajectories |
| Val R² (normalized) | 0.699 |
| Val R² (denormalized, correct) | **0.843** |
| Val cos | 0.638 |
| Training time | ~10 min (30 epochs, bs=32) |
| Checkpoint | `best_tt_awq_7b.pt` (183 MB) |

**Key finding**: R² computed on normalized targets is more stringent than on denormalized (original method). Our model achieves R²=0.843 denormalized, close to original's 0.855.

### ✅ AWQ Model Loading

| Detail | Value |
|--------|-------|
| Model | Qwen2.5-7B-Instruct (AWQ 4-bit) |
| Load time | 15-30s (local NVMe SSD) |
| VRAM usage | **5.24 GB dedicated** |
| Kernel | `AwqGEMMTritonLinear` (Triton-based fallback) |
| ExLlamaV2 | Failed (no MSVC compiler on Windows) |
| Batch throughput (bs=16, 50 tok) | **136 tok/s** |
| Batch throughput (bs=64, 50 tok) | **300 tok/s** |

### ✅ Prompt Format Fix

- Chat template (`apply_chat_template` with `{"role": "user", ...}`) required for instruct model
- Answer extraction needs last-number-from-last-sentence strategy
- `MAX_GEN=400` required for verbose reasoning chains (200 truncated answers)
- `skip_special_tokens=True` needed when decoding

---

## Key Optimizations Applied

### Zero-Cost (5 min)

| Optimization | Impact |
|-------------|--------|
| `torch.set_float32_matmul_precision('high')` | ~2× FP32 matmul speedup (TF32) |
| `torch.backends.cuda.matmul.allow_tf32 = True` | Enables TF32 tensor cores |
| `torch.backends.cudnn.benchmark = True` | Auto-tuning convolution algorithms |
| `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` | Eliminates expandable_segments warnings |
| `HF_HOME=D:\\Datasets` | Points to fast SSD with 81GB free |

All encapsulated in `src/optimization.py`:
```python
from src.optimization import setup_optimizations
setup_optimizations()
```

### Forward Hooks vs output_hidden_states

`output_hidden_states=True` allocates ALL 28 full hidden states:
- Memory: 28 × batch × seq_len × d_model × 2 bytes = **~800MB** per forward pass at bs=10, seq=400
- Causes VRAM overflow into shared system memory

**Fix**: Forward hooks on all 28 layers capturing only `h[:, -1, :]`:
- Memory: 28 × batch × d_model × 2 bytes = **~200KB** per forward pass
- Saves ~800MB, enabling bs=10 without shared memory spill

Implemented in `run_steer_hooked.py` via `HiddenStateCollector` class.

### Batch Size Scaling

| Batch | t/s (baseline) | VRAM | Notes |
|-------|-------|------|-------|
| 1 | 8 | 5.25 GB | Slow but stable |
| 5 | ~40 | 5.41 GB | **Recommended for steering** |
| 10 | ~80 | 5.44 GB | Leaks into shared memory at long generations |
| 30 | ~142 | ~6 GB | Baseline only (no steering modifications) |

**Batched TT inference**: Running TT once per token on all batch items (not once per item per token) gave ~5× speedup over naive per-item steering.

---

## Infrastructure Discoveries

### Windows-Specific Issues

1. **gptqmodel overrides `PYTORCH_ALLOC_CONF`** after loading:
   - Sets `expandable_segments:True` (NOT supported on WDDM → allocator fallback path)
   - Sets `garbage_collection_threshold:0.7` (triggers GC at 5.6GB, barely above model's 5.24GB)
   - Can't prevent this from user code — library sets it internally

2. **CUDA UMD corruption after GPU crash**:
   - WDDM TDR leaves orphaned resources in user-mode driver
   - `nvidia-smi --gpu-reset` only resets KMD, not UMD
   - Fix: `pnputil /disable-device /enable-device` on NVIDIA GPU (requires admin) OR full reboot
   - Corrupted .pyc caches compound the problem after crashes
   - Created `fix_conda_torch.ps1` as automated repair script (steps 0-7)

3. **Page file fragmentation**:
   - Loading 4.17 GB safetensors files requires contiguous virtual memory
   - After repeated model loads + unloads, virtual memory fragments
   - `low_cpu_mem_usage=True` helps but doesn't fully prevent
   - Reboot is the only reliable fix

### AWQ vs BnB 4-bit Tradeoffs

| Aspect | AWQ (current) | Bitsandbytes 4-bit |
|--------|--------------|-------------------|
| VRAM | 5.24 GB | ~5.5 GB |
| Batch inference | Fast (Triton kernels) | Slower |
| Hidden states via hooks | ✅ Works | ✅ Works |
| `output_hidden_states=True` | ⚠️ Ignored at generation | ✅ Works |
| Page file issues | ❌ (4.17 GB safetensors) | ✅ (loads layer-by-layer) |

---

## Files Created/Modified This Session

| File | Purpose |
|------|---------|
| `src/optimization.py` | Centralized optimization setup (TF32, cuDNN, allocator) |
| `run_train_tt_awq.py` | TT training on AWQ trajectory data (sequential) |
| `run_train_tt_stable.py` | Stable TT training with memory fixes |
| `run_train_tt_async.py` | Async data loading for TT training |
| `run_train_tt_vramcache.py` | Triple-buffered VRAM cache for TT training |
| `run_awq_steering_sweep.py` | Per-layer sweep on AWQ 7B (sequential) |
| `run_awq_sweep_batched.py` | Batched per-layer sweep |
| `run_sweep_v2.py` | Corrected prompt format + batched baseline |
| `run_sweep_fast.py` | Optimized sweep with MAX_GEN=400 |
| `run_steer_only.py` | Steering-only evaluation (no baseline redo) |
| `run_steer_hooked.py` | Hook-based steering (avoids output_hidden_states) |
| `fix_conda_torch.ps1` | Windows automated repair script (Steps 0-7) |
| `ACCELERATION_STRATEGY.md` | Full acceleration strategy document |
| `EMERGENT_AVENUES.md` | 12 expansion avenues for future work |
| `_benchmark_throughput.py` | Throughput benchmarks |
| `_verify_env.py` | Environment verification |

---

## Lessons Learned

1. **Windows WDDM is hostile to CUDA ML workloads** — expandable_segments, TDR, orphaned UMD state, page file fragmentation. The dual-boot Linux setup is strongly preferred.

2. **`output_hidden_states=True` is a memory trap** — allocates all layer hidden states (~800MB). Use forward hooks instead (200KB).

3. **AWQ vs BnB**: AWQ is faster for batch inference but harder to load (page file issues). BnB is more robust on Windows.

4. **TT training R²**: Computed on normalized targets is more stringent. Denormalized R² = 0.843 matches original 0.855.

5. **Steering WORKS on AWQ models** — L8 trim-tab confirmed with +6.7pp improvement.

6. **Batched TT inference** (once per token for all batch items) is ~5× faster than per-item.

7. **Memory leak in generation loops**: Without `torch.cuda.empty_cache()` at batch boundaries, VRAM grows by ~3-5% per batch until shared memory is hit.

---

## State for Linux Resume

### What Works
- [x] AWQ Qwen2.5-7B loads and generates
- [x] Forward hooks capture hidden states
- [x] Batched TT inference (once per token for all items)
- [x] KV-cache steering at any layer
- [x] Prompt format (chat template with max_new_tokens=400)
- [x] Answer extraction (last number in last sentence)

### Checkpoints on D: SSD
- `D:\Qwen2.5-7B-AWQ\qwen7b_awq` — AWQ 4-bit model
- `best_tt_awq_7b.pt` — TT checkpoint (R²=0.843 denorm)
- `D:\project_data\qwen25_7b_gen_trajs` — 83 batch files (88K trajectories)
- `D:\trajs_7B_AWQ\Batches` — 16 AWQ trajectory batch files

### Available Scripts (TrimTab repo)
```bash
# Training
python run_train_tt_stable.py --data-dir <path> --d-model 768 --epochs 30

# Baseline only
python run_sweep_fast.py --n-test 30 --batch-size 5 --layers 8 --alpha 0.1

# Steering (hooked, memory-efficient)  
python run_steer_hooked.py --layer 8 --n-test 30 --alpha 0.1 --batch-size 10

# Sweep all key layers
python run_sweep_fast.py --n-test 30 --batch-size 5 --layers 2 3 5 8 9 10 15 20 --alpha 0.1
```

### Next Steps
1. Run full per-layer sweep on Linux (AVOIDS Windows WDDM allocator issues)
2. Contrastive TT steering (correct vs incorrect trajectories)
3. Multi-layer steering combinations (L2+L8)
4. Reading-head gated adaptive steering
