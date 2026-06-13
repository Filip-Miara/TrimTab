# Session Debrief — RankAdaptation Steering Experiments (2026-06-12/13)

## Context

This session explored **velocity-based latent steering** across multiple models and attention architectures. The goal: predict hidden state velocities during generation using a TrajectoryTransformer (TT), then modify KV cache entries to steer generation toward better outcomes (measured by GSM8K accuracy).

## Models Tested

| Model | Params | Attention Type | GSM8K Baseline | Cached |
|-------|--------|---------------|----------------|--------|
| Qwen3.5-2B | 2B | Hybrid (18× GatedDeltaNet + 6× FullAttention) | 40% | Yes |
| SmolLM2-360M | 360M | Standard MHA (15Q/5KV) | 4% | Yes |
| Qwen2.5-0.5B | 0.5B | Standard MHA (14Q/2KV) | 6% | Yes |
| TinyLlama-1.1B | 1.1B | Standard MHA (32Q/4KV) | 0-4% | No (cached later) |
| Phi-3-mini-4k-instruct | 3.8B | Standard MHA (32Q/32KV) | N/A (failed) | Partial |
| Qwen3.5-4B | 4B | Hybrid (24× GatedDeltaNet + 8× FullAttention) | N/A (OOM) | No |
| Qwen2.5-7B-Instruct | 7B | Standard MHA (28Q/4KV) | **73%** ✅ | Yes (moved to SSD) |

## What We Did

### Phase 1: PPL-Modulated Correction (Qwen3.5-2B)
- **Approach**: Use reading head (r=0.86 uncertainty predictor) to gate correction head strength: `α = σ(β · (ppl_pred - τ))`
- **Result**: Failed. Prompt-trained correction heads got 0% generation accuracy (distribution shift). Gen-trained heads matched baseline (no improvement). PPL gating at τ=1.5 gated <0.1% of tokens — model is **confidently wrong** at generation.
- **Tags**: v0.34.0

### Phase 2: Recurrent-State Steering (Qwen3.5-2B)
- **Discovery**: Qwen3.5-2B uses **hybrid attention** (3:1 GatedDeltaNet:FullAttention), NOT standard MHA. GatedDeltaNet stores `recurrent_states [1, 16, 128, 128]`, not key/value pairs.
- **Approach**: Forward hooks to modify hidden state before GatedDeltaNet computation
- **Result**: No effect on accuracy (10% = baseline)

### Phase 3: Hybrid KV-Cache Steering (Qwen3.5-2B)
- **Approach**: Modify K/V only at the 6 Full Attention layers (3, 7, 11, 15, 19, 23) using prompt-trained TT (R²=0.62)
- **Result**: 5% vs 10% baseline — slightly harmful

### Phase 4: SmolLM2-360M Proof of Concept
- **Approach**: Collect generation trajectories → train TT on gen data → steer standard MHA model
- **Key positive result**: **Generation trajectories ARE learnable** — TT achieved **R²=0.94** on generation data vs prompt TT's R²=0.62
- **Steering test**: **88% token divergence** at α=0.1 — the steering mechanism IS working
- **Limitation**: SmolLM2 gets 0-4% on GSM8K (too small for math), so we couldn't measure accuracy impact
- **Tags**: v0.35.0

### Phase 5: Qwen2.5-7B-Instruct Pipeline
- **Approach**: Download 7B model with standard MHA, collect gen trajectories, train TT, evaluate steering
- **Baseline**: 73% on GSM8K (with chat template, 400 max tokens)
- **TT Training**: R²=0.855 (converged, lower than SmolLM2 due to 100K-dim space)
- **Collection**: 50K trajectories from 500 GSM8K train problems
- **Steering eval**: Running now (α=0.1, 0.3)

### Background: TinyLlama-1.1B & Phi-3-mini attempts
- TinyLlama: 0% GSM8K — <2B base models can't do multi-step math
- Phi-3-mini: `trust_remote_code` version incompatibility with transformers 5.9.0 (`DynamicCache.from_legacy_cache`, `get_usable_length` missing)
- Qwen3.5-4B: CUDA OOM during 4-bit loading (caching allocator warmup tried to allocate 5.07GB)

## Infrastructure Issues

### Slow Model Loading
- External HDD: ~2.5 min for 7B model (339 shards at 2-3 it/s)
- Internal SSD: ~10s for same model (35-45 it/s)
- **Fix**: Move model files from HDD to `~/.cache/huggingface/hub/` on SSD

### CPU RAM Overflows
- Storing 5000 × (28×3584) float32 trajectories in Python lists: ~4GB per batch
- Combined with other processes → 16GB RAM exhaustion
- **Fix**: Reduce batch size to 1000, use float16 (`.half()`) in accumulation lists

### CUDA OOM from orphaned processes
- Background processes don't free GPU memory when killed
- Subsequent processes fail with CUDA allocation errors
- **Fix**: `kill $(lsof /dev/nvidia* | grep python | awk '{print $2}' | sort -u)` before restarting

### bitsandbytes 4-bit bias bug
- `Linear4bit.forward()` crashes with `RuntimeError: data set to a tensor that requires gradients must be floating point` when input has certain dtypes
- **Fix**: Use `.to(torch.bfloat16)` instead of `.to(k_proj.weight.dtype)`

### Phi-3 `trust_remote_code` version lock
- Custom `modeling_phi3.py` requires newer `DynamicCache` methods (`from_legacy_cache`, `get_usable_length`)
- Not worth monkey-patching; switched to Qwen2.5-7B which doesn't need custom code

### Download bandwidth
- 7-8GB models take 10-20 min on connection
- **Fix**: Use `run_background_process` for long downloads with appropriate timeout

### SSD Storage Limits
- 71GB SSD with 22GB free initially
- Qwen2.5-7B (15GB) + Qwen3.5-4B partial (8.7GB) filled it up
- **Fix**: Delete unused models (Phi-3: 2.7GB, Qwen3.5-0.8B: 1.7GB, partial Qwen3.5-4B: 8.7GB)

## What Worked

1. **Generation trajectories are learnable** (R²=0.94 SmolLM2, R²=0.855 Qwen2.5-7B) — core positive finding
2. **Steering changes tokens** (88% divergence on SmolLM2) — mechanism works
3. **Background process pattern** for long-running operations
4. **Batch writing to disk** for memory-bound data collection
5. **Moving models to SSD** for ~15x faster loading
6. **Qwen2.5-7B-Instruct chat template** with 400 max tokens — achieved 73% baseline
7. **DoRA-inspired separation** of direction (TT) from magnitude (confidence signal) — theoretical insight, not yet tested

## What Did NOT Work

1. **All steering mechanisms on Qwen3.5-2B** (logit correction, recurrent-state steering, hybrid KV-cache steering) — no improvement
2. **PPL-modulated correction** — reading head works (r=0.85) but correction heads produce wrong offsets at generation
3. **SmolLM2 math evaluation** — model too small for GSM8K (4% baseline)
4. **Phi-3-mini** — `trust_remote_code` version incompatibility
5. **Qwen3.5-4B** — CUDA OOM during 4-bit loading
6. **Standard prompting on instruct models** — raw text (Q:... A:...) gives 4%; need chat template for 73%
7. **max_new_tokens=200** on 7B model — generation gets truncated mid-reasoning; need 400+
8. **4-bit loading allocator warmup** — tries to allocate large contiguous buffers, can OOM

## Key Lessons

1. **Attention architecture matters enormously.** Hybrid attention (GatedDeltaNet + FullAttention) complicates steering — only 25% of layers are steerable. Standard MHA models (Qwen2.5, LLaMA) are preferred.
2. **Distribution shift is real and severe.** Models trained on prompt trajectories (R²=0.62) don't generalize to generation. Generation-trained models (R²=0.94) are dramatically better.
3. **Small models can't do math.** <2B base models fail at GSM8K regardless of prompting. Need >3B or instruction-tuned.
4. **Confidently wrong.** Models generate wrong answers with low perplexity — uncertainty signals don't correlate well with correctness.
5. **TT capacity matters.** SmolLM2's 30K-dim space got R²=0.94 with d_model=512. Qwen2.5-7B's 100K-dim space needed d_model=768 and only got R²=0.855.
6. **OS resource management is critical.** Batch sizes, storage paths, GPU memory cleanup, and timeout handling must be explicitly managed.
7. **The DoRA analogy suggests separating direction from magnitude** — the TT's direction predictions (cos≈0.65-0.82) are more reliable than its magnitude predictions.

## Files Created This Session

| File | Purpose |
|------|---------|
| `run_ppl_modulated_correction.py` | PPL-gated logit correction |
| `run_ppl_sweep.py` | Sweep of PPL gating configs |
| `run_ppl_targeted.py` | Targeted PPL gating evaluation |
| `run_recurrent_steering.py` | GatedDeltaNet recurrent-state steering |
| `run_kvcache_accuracy.py` | SmolLM2 KV-cache accuracy eval |
| `run_hybrid_steering.py` | Hybrid (FA-only) KV-cache steering |
| `run_collect_gen_trajs.py` | SmolLM2 trajectory collection |
| `run_train_gen_tt.py` | SmolLM2 TT training |
| `run_smolm2_steering.py` | SmolLM2 steering evaluation |
| `run_collect_gen_trajs_7b.py` | Qwen2.5-7B trajectory collection |
| `run_train_gen_tt_7b.py` | Qwen2.5-7B TT training |
| `run_7b_baseline.py` | Qwen2.5-7B GSM8K baseline |
| `run_7b_steering.py` | Qwen2.5-7B steering evaluation |
| `run_7b_dora_steering.py` | DoRA-inspired steering (not yet run) |
| `best_gen_tt.pt` | SmolLM2 gen-trained TT (R²=0.94) |
| `best_gen_tt_7b.pt` | Qwen2.5-7B gen-trained TT (R²=0.855) |

## Data on External HDD

```
/run/media/filip/B522-875D/Datasets/project_data/
  smolm2_gen_trajs/          — 10 batches × 5K = 50K SmolLM2 trajectories
  qwen25_7b_gen_trajs/       — 49 batches × ~1K = 49K Qwen2.5-7B trajectories

---

## Proposed Next Experiments (from end-of-session analysis)

### A. Contrastive Steering Refinements (currently running)
- Single-layer contrastive sweep on Math-1.5B (28 layers × 50 problems)
- Expected: identify trim-tab layers for contrastive direction

### B. Asymmetric α Steering
- Use α_c and α_i independently: h' = h + α_c · TT_correct(h) − α_i · TT_incorrect(h)
- Rationale: attraction toward correct and repulsion from incorrect may need different strengths
- Sweep α_c ∈ {0.01, 0.05, 0.1}, α_i ∈ {0.01, 0.05, 0.1} on best trim-tab layers

### C. Multi-Layer Combinations
- Test L2+L8, L5+L8, L2+L5+L8 (pairs and triplets of best layers)
- Per-layer α scaling: α[l] varies per layer, not global

### D. Multi-Head Contrastive Ensembles
- Train N (3-5) independent TT_correct/TT_incorrect pairs on bootstrapped trajectory subsets
- v_ensemble = mean(TT_correct_k) − mean(TT_incorrect_k)
- Benefits: bagging reduces variance, ensemble disagreement gives steering confidence
- Parallels Phase 3 DiMAE approach but applied to velocity prediction

### E. Layer-Specific α Vector
- Instead of global α, learn α[l] per layer via sweep or optimization
- Could be learned as a lightweight MLP from trajectory features

### F. Performance Optimization
- `torch.compile()` on generation forward pass to reduce Python overhead
- Current: 50% GPU utilization (small model: 23ms compute + 23ms Python overhead per step)
- torch.compile could cut Python overhead by ~30-50%
- Alternative: C++/Rust inference (vLLM, Candle) but requires TT reimplementation

