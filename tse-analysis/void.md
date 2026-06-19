# Phase 0: VOID — Assumption Surfacing & Bracketing

## Subject: TrajectoryTransformer Training Pipeline
**Date**: 2026-06-18
**Mode**: rapid (phases 0-5 + 11)

---

## Explicit Assumptions

| # | Assumption | Source | Status |
|---|-----------|--------|--------|
| A1 | Hidden state velocity vectors (Δ between consecutive layers) are a meaningful and sufficient target for KV-cache steering | Pipeline design doc | Active |
| A2 | Global normalization (one mean/std per feature across ALL layers and samples) is the correct preprocessing approach | Preprocessing spec | Active |
| A3 | A 48M-parameter transformer with d_model=768, 6 layers, 8 heads can learn the velocity prediction function effectively | Architecture spec | Active |
| A4 | Bidirectional self-attention (not causal) is appropriate for velocity prediction | Architecture spec | Active |
| A5 | MSE loss with per-layer uniform weighting is the correct optimization objective | Training config | Active |
| A6 | 90K trajectories from Qwen2.5-7B-Instruct on GSM8K provide sufficient coverage for generalization | Data collection | Active |
| A7 | Float16 precision is adequate for training the TT without significant degradation | Training config | Active |
| A8 | Memory-mapped float16 storage is necessary due to dataset size | Preprocessing spec | Active |
| A9 | The validation set of 500 trajectories is representative | Pipeline design | Active |
| A10 | Gradient clipping at 1.0 and AdamW with LR 3e-4 are near-optimal hyperparameters | Training config | Active |
| A11 | Batch size 64 with VRAM double-buffering is efficient for GPU utilization | Training config | Active |
| A12 | The frozen 7B LM's hidden states are stationary (distribution doesn't shift during TT training) | Architecture spec | Active |
| A13 | Cosine loss variant (λ=0.5) adds value over pure MSE | Training config | Active |
| A14 | Layer-wise uniform weighting is optimal (all 28 layers equally important) | Training config | Active |
| A15 | TF32 matmul precision provides acceptable accuracy for training | Training config | Active |

## Implicit Assumptions

| # | Implicit Assumption | Inference Chain | Status |
|---|-------------------|-----------------|--------|
| A16 | The 28 layers of the 7B model encode velocity information at the same granularity | Architecture design presumes uniform treatment | Active |
| A17 | Velocity prediction is independent of the input text distribution | Training on GSM8K → deployment on any reasoning task | Active |
| A18 | The global normalization statistics from 90K trajectories are representative | Validity depends on coverage of the state space | Active |
| A19 | Bidirectional context helps velocity prediction (not just future-leaking artifact) | Chosen over causal despite worse empirical match | Active |
| A20 | The R²=0.85 ceiling is a fundamental limit of the current architecture, not a hyperparameter issue | Best results accepted as ceiling | Active |
| A21 | AWQ quantization changes hidden states in a way that is predictable (can be learned) | Assumption underlying transfer attempt | Active |
| A22 | Catastrophic forgetting in AWQ→BnB transfer is due to distribution shift, not capacity limitation | Implicit in fine-tuning approach | Active |
| A23 | CUDA crashes are environmental (hardware stability), not algorithmic | Issue diagnosis | Active |
| A24 | Data I/O being bottleneck on slow drives is acceptable (can engineer around) | Issue diagnosis | Active |
| A25 | The velocity representation is layer-homogeneous — same d_model=3584 across all 28 layers | Architecture treats all layers identically | Active |
| A26 | A single global mean/std captures the relevant statistics of a 28×3584×90K tensor | Preprocessing choice | Active |
| A27 | Transformer-based prediction is the right architecture (vs. MLP, CNN, Mamba, etc.) | Architecture selection | Active |

## Counter-Assumptions (What if ¬[assumption]?)

| Base | Counter-Assumption |
|------|-------------------|
| A1 | "What if velocity vectors are NOT the right steering signal — what if attention patterns, activation magnitudes, or subspace projections matter more?" |
| A2 | "What if global normalization destroys per-layer and per-sample distributional information that is critical for prediction?" |
| A3 | "What if 48M is either too few parameters (underfitting) or too many (overfitting noisy velocities)?" |
| A4 | "What if causal attention would produce better generalization because it matches inference-time conditions?" |
| A5 | "What if MSE is the wrong loss — what if cosine similarity, contrastive loss, or distribution matching would work better?" |
| A6 | "What if 90K trajectories are insufficient or that GSM8K is too narrow a distribution?" |
| A7 | "What if float16 precision is losing signal in the gradient for high-frequency velocity features?" |
| A8 | "What if the dataset can be compressed (via PCA, quantization, or online generation) to eliminate the I/O bottleneck?" |
| A9 | "What if the 500 validation trajectories systematically under represent failure modes?" |
| A10 | "What if the optimal hyperparameters are far from current values (e.g., LR 3e-5, no gradient clipping)?" |
| A11 | "What if batch size 64 is suboptimal — that smaller batches regularize better, or larger batches use hardware more efficiently?" |
| A12 | "What if the frozen 7B model's hidden states DO shift during training (e.g., due to changing KV-cache steering affecting subsequent states)?" |
| A13 | "What if cosine loss (λ=0.5) is actively harmful by creating gradient interference with MSE?" |
| A14 | "What if some layers (e.g., early vs late) should be weighted differently because they encode different velocity magnitudes?" |
| A15 | "What if TF32 is losing critical precision in the attention softmax or the prediction head?" |
| A16 | "What if velocity structure differs dramatically between early, middle, and late layers?" |
| A17 | "What if velocity prediction is highly input-dependent, and GSMM8K-trained TT fails on out-of-distribution texts?" |
| A18 | "What if the normalization statistics are dominated by a few outlier trajectories?" |
| A19 | "What if bidirectional attention is cheating — using future velocity information to predict current velocity?" |
| A20 | "What if R²=0.85 is NOT a ceiling and substantially better results are achievable with different approaches?" |
| A21 | "What if AWQ changes are fundamentally unpredictable (deterministic but chaotic relative to velocity space)?" |
| A22 | "What if catastrophic forgetting is a capacity issue — the TT simply cannot represent both distributions simultaneously?" |
| A23 | "What if CUDA crashes are algorithmic — e.g., numerical instability from the double-buffer prefetch or TF32 non-determinism?" |
| A24 | "What if data I/O is a symptom of a deeper architectural issue (need to train online or with progressive data loading)?" |
| A25 | "What if treating all 28 layers identically via global normalization destroys per-layer-specific velocity structure?" |
| A26 | "What if per-layer normalization or per-sample normalization would dramatically improve prediction?" |
| A27 | "What if a simpler architecture (e.g., 2-layer MLP per layer, or a linear probe) would match or exceed transformer performance?" |

## Bracket Statement

These assumptions are set aside for the analysis. They will be re-examined in Phase 6 (Disparity Detection) and throughout the lens cascade to check whether any assumption violation is the root cause of observed failures. The counter-assumptions are particularly valuable — they serve as alternative framings that will seed divergent thinking in Phase 4.

Key areas of assumption density:
1. **Normalization strategy** (A2, A18, A26) — Multiple implicit assumptions about one design choice
2. **Architecture adequacy** (A3, A4, A27) — Core model design choices untested against alternatives
3. **Loss function optimality** (A5, A13, A14) — Single loss function with minimal exploration
4. **Distributional stationarity** (A12, A17, A21) — Critical for transfer yet unvalidated
5. **Data representativeness** (A6, A9) — Training domain may not match deployment
