# Session Compaction Recovery — 2026-06-19

## Recovery Status: ✅ New Session (Linux, Session 7)

Significant advances in TT architecture, loss functions, and understanding of steering mechanisms.

## Key Checkpoints

| File | Content | R² | Notes |
|------|---------|:--:|-------|
| `best_gen_tt_7b.pt` | Qwen2.5-7B standard TT (old pipeline) | 0.855 | Original BnB TT |
| `best_gen_tt_7b_new.pt` | d_model=768, baseline (overwritten) | — | Overwritten by d1024 |
| `best_tt_d1024.pt` | d_model=1024 | **0.866** | +0.018 over baseline |
| `best_tt_d1280.pt` | d_model=1280 | **0.882** | +0.034 over baseline |
| `best_tt_d1536.pt` | d_model=1536, MSE only | **0.899** | Best R², 181M params |
| **`best_tt_d1536_cos.pt`** | d_model=1536, MSE+cosine | **0.900 / Cos=0.824** | **Best overall** |
| `best_tt_decomp.pt` | d_model=768, Huber+cosine | 0.808 | Slower convergence |
| `best_tt_plnorm.pt` | d_model=768, per-layer norm | 0.842 | Cos=0.565 (bad) |

## Key Findings

### Architecture
- **Width (d_model) is the single most impactful lever** — scaling from 768→1024→1280→1536 monotonically improves R²
- **Depth (n_layers, d_ff_ratio) hurts** — faster convergence with fewer params
- Per-layer normalization is worse than global normalization (even with variance weighting)

### Loss Functions
- MSE+cosine (λ=0.5) gives best cos (0.824) at small R² cost (-0.011)
- Huber+cosine converges too slowly
- **d_model=1536 + MSE+cosine is the best combination**: R²=0.900, Cos=0.824

### Steering Mechanism
- KV cache modification works via `DynamicCache.layers[li].keys` (Transformers 5.9.0 API)
- α=0.05 gives same effect as α=1.0 (saturation at tiny α)
- Even α=0.0 changes output (K/V replacement from hs[li+1] differs from original)
- Cross-layer: velocity source barely matters — injection target defines effect
- **L9 death is layer-specific, not velocity-specific** (L9's velocity at L10 = +10pp)
- **Retroactive steering collapses at ALL layers** due to temporal KV cache inconsistency

### Trim-Tab Results (AWQ, 30 problems)

| Layer | Acc | Δ |
|:-----:|:---:|:-:|
| Baseline | 36.7% | — |
| L2 | 40.0% | +3.3pp |
| L5 | 43.3% | +6.7pp |
| L8 | 26.7% | -10.0pp |
| L9 | 0.0% | -36.7pp 💀 |
| **L10** | **46.7%** | **+10.0pp 👑** |

L10 is the best trim-tab across both AWQ (+10pp) and BnB (+13.3pp).

### Future Directions
See ROADMAP.md for:
- Path 6: Advanced Steering Architectures (retroactive, multi-layer, iterative)
- Cross-dataset generalization tests
- Latent Briefing integration (Steer-Once-Broadcast-Many)
