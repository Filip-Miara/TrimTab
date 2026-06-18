# Acceleration Strategy — TrimTab / RankAdaptation

**Hardware**: RTX 4060 Laptop GPU (8GB) | i7-12650H (10C/16T) | 16GB RAM
**Software**: PyTorch 2.11.0+cu128 | CUDA 12.8 | Triton 3.3.1 | Transformers 5.9.0
**Storage**: C: 327GB NVMe (11GB free) | E: 915GB HDD (16GB free) | D: External SSD (on chkdsk)
**Constraints**: WDDM driver mode (no expandable_segments) | No FP8 matmul | No AutoAWQ yet

---

## Executive Summary

**The bottleneck is NOT hardware** — it is software configuration and workflow design.

Benchmarks on the TT (62M params, d_model=768):
- **3.14x speedup** from `torch.compile` (12.8ms -> 4.1ms)
- **~2.1x speedup** from BF16 vs FP32 on matmuls
- **Near-perfect batch scaling** (82/s at bs=1 -> 4110/s at bs=64)
- Python overhead in generation loops (50% GPU utilization documented in SESSION_DEBRIEF)
- **CUDA Graphs available** for capturing generation loops

With the AWQ-Marlin env on the external SSD:
- Qwen2.5-7B fits in ~4.5GB (vs 14GB) -> leaves room for TT + reading head + KV cache
- Marlin kernels give near-FP16 inference speed
- Triton custom kernels can fuse steering operations into a single kernel
- Qwen3.5-8B becomes affordable at 4-bit (~5.1GB)

---

## Tier 1: Zero-Cost Changes (5 minutes, no code changes)

### 1.1 Fix PYTORCH_CUDA_ALLOC_CONF
**Current**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`
**Problem**: `expandable_segments:True` is NOT supported on Windows/WDDM. Causes warnings on every CUDA call.
**Fix**: Remove expandable_segments:
```
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```
**Impact**: Eliminates the "expandable_segments not supported" warnings.

### 1.2 Enable TF32
**Current**: `torch.set_float32_matmul_precision('highest')` and `cuda.matmul.allow_tf32 = False`
**Problem**: TF32 gives ~2.1x speedup over full FP32 but is disabled.
**Fix**: Add to ALL scripts:
```python
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```
**Impact**: ~2x speedup on all FP32 matmuls. Catches cases where BF16 isn't used.

### 1.3 Redirect HF_HOME to C: temporarily
**Current**: `HF_HOME=D:\Datasets` (external SSD, currently running chkdsk)
**Fix** (temporary):
```powershell
$env:HF_HOME = "C:\Users\fiper\.cache\huggingface"
$env:TRANSFORMERS_CACHE = "C:\Users\fiper\.cache\huggingface"
```
**Impact**: Enables model loading immediately without waiting for chkdsk.
**Note**: When D: is available, move cache back and free C: space.

---

## Tier 2: Low-Effort Optimizations (hours, minor code changes)

### 2.1 Add torch.compile to TT Inference (3.14x speedup)
**Current**: Eager mode inference: 12.8ms per forward pass
**After**: `torch.compile(mode='reduce-overhead')`: 4.1ms per forward pass

One-line change:
```python
compiled_tt = torch.compile(trajectory_transformer, mode='reduce-overhead')
v_pred = compiled_tt(hidden_seq)
```

**Impact**:
- Steering eval: 62 -> 244 evaluations/s (at bs=16)
- Per-layer sweeps: 28 layers x 50 problems / 244/s = ~6s (was ~18s)
- **Note**: First forward pass includes compilation (~2-5s). Use a warmup call.

### 2.2 Pre-Compute Trajectories (Eliminate Redundant Backbone Forwards)
**Current**: Every trajectory collection re-runs the backbone forward pass.
**Fix**: Pre-compute all h_last trajectories for GSM8K once, store as .pt:
```
precomputed_trajs.pt = {h_last: (N, max_len, d_model), labels: (N,)}
```
Then training/evaluation load from disk instead of re-running the model.

**Impact**: Trajectory collection goes from ~30min per run to ~22min one-time, then instant.
**Pattern from qwen3_trm**: Pre-computation gave 260-400x speedup for training.

### 2.3 Batch Steering Evaluation
**Current**: Process one problem at a time (bs=1 for generation).
**After**: Process multiple problems with different steering policies simultaneously.

Batch scaling is near-perfect: bs=1 gives 82/s, bs=64 gives 4110/s.
Full per-layer sweep (28 layers x 3 alpha x 50 problems) goes from ~50min to ~2min.

### 2.4 Smaller TT (d_model=512) for Rapid Iteration
**Current**: TT d_model=768, L=8: 62.3M params, 119MB
**After**: TT d_model=512, L=6: 22.7M params (-64%)
**Recommendation**: Keep d_model=768 for final results, use d_model=512 for rapid iteration.
R-squared likely drops from 0.855 to ~0.82-0.83 (acceptable for sweeps).

### 2.5 Async Data Loading
Already partially implemented. Extend to overlap trajectory loading with GPU computation:
- Load next batch while current batch trains
- Pre-fetch from SSD to RAM while GPU computes
- Use `concurrent.futures.ThreadPoolExecutor`

---

## Tier 3: Medium-Effort Optimizations (days)

### 3.1 AWQ-Marlin Integration (When D: Drive Ready)
**The Game-Changer**: AWQ-Marlin provides:
1. **Qwen2.5-7B at 4-bit (~4.5GB)**: Fits alongside TT (119MB) + reading head (5MB) + KV cache in 8GB
2. **Marlin inference speed**: Near-FP16 throughput for 4-bit models
3. **AutoAWQ quantization**: Can quantize any model we need

**Integration with steering**:
- AWQ models loaded via `AutoModelForCausalLM.from_pretrained(..., device_map='auto')` support `output_hidden_states=True`
- Forward hooks work on quantized layers (confirmed via bitsandbytes test)
- KV-cache modification works identically to FP16 models

**Checklist for D: drive availability**:
- [ ] chkdsk completes
- [ ] D: drive is accessible
- [ ] AWQ-Marlin env has autoawq installed
- [ ] AWQ-quantized Qwen2.5-7B fits in VRAM alongside TT
- [ ] Hidden states are accessible from AWQ model
- [ ] Steering mechanism works with AWQ hidden states

### 3.2 Triton Custom Kernels for Fused Steering
**Opportunity**: Triton 3.3.1 is FULLY available on Windows (rare!). This enables custom GPU kernels.

**Target: Fuse the generation-steering loop into one kernel**.

Current flow per token: 6 kernel launches, 2 Python round-trips.
Fused flow: 1 kernel launch, 0 Python round-trips.

**Impact**:
- Eliminates the "50% GPU utilization" bottleneck (23ms compute + 23ms Python overhead)
- Enables per-token adaptive alpha without Python round-trips

**Concrete kernel idea**:
```python
@triton.jit
def steered_generation_kernel(
    hidden_states,    # (1, 28, 3584)
    tt_weights_ptr,   # TT parameters
    alpha,            # steering strength
    output_ptr,       # output buffer
):
    # 1. Apply TT weight matrices (chain of matmuls)
    # 2. Compute v_pred
    # 3. h_steered = h + alpha * v_pred
    # 4. Write back
```

### 3.3 CUDA Graphs for Generation Loop
**Current**: Generation loop launches separate CUDA kernels each step.
**After**: Capture the entire generation iteration as a CUDA Graph (single launch).

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    v_pred = compiled_tt(hidden)
    h_steered = hidden + alpha * v_pred

# Replay with new data (no Python overhead)
for token in range(max_tokens):
    g.replay()  # ~0ms Python overhead
```

**Impact**: Near-zero Python overhead. GPU utilization goes from ~50% to ~95%.

### 3.4 Storage Management
**Critical**: C: drive has only 8.75GB free. This will cause failures.

**Immediate cleanup**:
- Clean `C:\Users\fiper\AppData\Local\Temp` (~3.7GB)
- Move old checkpoints to E:\
- When D: available: Use D: for active models and HF cache

**Model storage strategy**:
- Active models: C: (fast NVMe)
- Archives: E: (slower HDD, more space)
- Quantized models: D: when available (fast SSD, lots of space)

---

## Tier 4: Architectural Optimizations (longer, higher impact)

### 4.1 Distilled TT
Train a smaller TT (d_model=256, L=4) to mimic the larger TT (d_model=768, L=8):
- Student: ~5.6M params, ~30MB
- Potential 10x speedup over full TT
- R-squared likely drops from 0.855 to ~0.80-0.83 (acceptable for steering sweeps)

Use for: Rapid iteration, hyperparameter sweeps, multi-layer combinatorics.

### 4.2 Pre-Computed Steering Directions
Pre-compute steering directions for all tokens at once:
```python
v_all = compiled_tt(h_all_tokens)  # one forward pass
# During generation: h_steered[t] = h[t] + alpha * v_all[t]
```
TT inference goes from per-token (82/s) to one-shot.

### 4.3 Online alpha Learning via MetaController
The MetaController (already built in StreamFusion) can learn per-token alpha:
- Input: reading head prediction, layer position, token position
- Output: alpha per (layer, token)
- Trained via REINFORCE over GSM8K accuracy

---

## AWQ-Marlin / Triton Synergy Analysis

### The Synergy

| Component | Without AWQ | With AWQ+Marlin | Benefit |
|-----------|-------------|-----------------|---------|
| Qwen2.5-7B | ~14GB (OOM) | ~4.5GB (fits) | Can run 7B steering! |
| Qwen3.5-8B | ~16GB (OOM) | ~5.1GB (fits) | Cross-architecture tests |
| Generation speed | ~60 t/s (2B) | ~180 t/s (7B Marlin) | 3x faster eval |
| Steering model | Needs separate GPU | Co-located in 8GB | No PCIe transfer |
| Custom kernels | No Triton (usually) | Triton 3.3.1 available | Fused ops possible |

### Why Triton Is the Force Multiplier

Triton 3.3.1 on Windows is unusual and valuable. Having it means:

1. **Custom attention kernels**: Fuse TT prediction + KV-cache modification
2. **Per-layer alpha without Python**: All alpha(layer, token) computation stays on GPU
3. **torch.compile + Triton**: The 3.14x TT speedup uses Triton kernels under the hood
4. **Gradient-aware steering**: Backprop through steering for RL fine-tuning

### The Bottleneck That Remains

Even with all optimizations, the **model forward pass** (~196ms for Qwen2.5-7B) dominates. AWQ+Marlin helps (4-bit forward is ~30% smaller memory, similar compute). The real breakthrough:

**With AWQ+Marlin, Qwen2.5-7B fits in 8GB alongside the TT.** Everything stays on-GPU, eliminating PCIe transfer overhead.

---

## Implementation Roadmap

### Phase 1: Today (30 min)
- [ ] Fix PYTORCH_CUDA_ALLOC_CONF
- [ ] Enable TF32 + cudnn.benchmark in all scripts
- [ ] Redirect HF_HOME to C: while D: is busy
- [ ] Run the contrastive TT per-layer sweep (P0 experiment)

### Phase 2: This Week (2-3 hours)
- [ ] Add torch.compile to TT (3.14x speedup)
- [ ] Pre-compute Qwen2.5-7B trajectories (store as .pt)
- [ ] Implement batch steering evaluation
- [ ] Test CUDA Graphs on generation loop

### Phase 3: When D: Drive Available (1-2 days)
- [ ] Activate AWQ-Marlin env
- [ ] Test AWQ model loading + hidden state access
- [ ] Verify steering works on AWQ-quantized 7B
- [ ] Run cross-architecture tests (Qwen3.5-8B at 4-bit)

### Phase 4: Advanced (1-2 weeks)
- [ ] Build Triton fused steering kernel
- [ ] Train distilled TT (d_model=256)
- [ ] Implement pre-computed steering directions
- [ ] Set up MetaController for online alpha learning

---

## Quick-Start: The Most Important Change

**Single highest-ROI change**: Add torch.compile to the TT.

```python
# These 4 lines should be in EVERY training script:
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

# This one line gives 3.14x TT speedup:
compiled_tt = torch.compile(tt, mode='reduce-overhead')
```

**Total time**: 30 seconds to add. **Result**: 3.14x faster steering evaluation, 2x faster matmuls.



