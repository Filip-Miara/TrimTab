# Phase 0: VOID — Assumption Surfacing & Bracketing

**Subject**: RankAdaptation — Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Structural Scan: Explicit Assumptions

| # | Assumption | Source | Evidence Grounding |
|---|------------|--------|-------------------|
| A1 | Hidden state velocities during generation contain useful steering information | Core thesis | R²=0.85-0.94 across models, but "useful" presumes steering signal exists |
| A2 | The KV cache is the right intervention surface for steering | Architecture design | KV-cache modification achieves 88% token divergence; logit correction failed |
| A3 | Per-layer selectivity is necessary; all-layers steering is net negative | Finding 4 | Qwen2.5-7B all-layers steering confirmed net negative |
| A4 | Steering amplifies existing capability, does not create it | Finding 3 | Models below ~40% GSM8K baseline cannot be steered |
| A5 | Standard MHA is preferred over hybrid attention for steering | Finding 6 | Qwen3.5-2B (hybrid: GDN+FA) all steering attempts failed; only 25% layers steerable |
| A6 | Generation trajectories differ from prompt trajectories | Lesson 4 | Prompt-trained TT R²=0.62 vs gen-trained TT R²=0.94 |
| A7 | Contrastive TT converts descriptive prediction into normative prediction | Lesson 5 | Standard TT faithfully reproduces errors; contrastive should correct them |
| A8 | The trim-tab/death-layer pattern generalizes across datasets and model families | Finding 2 | SVAMP replicates GSM8K pattern; cross-model SmolLM2→7B preserves L8 |
| A9 | Models require >~40% baseline capability to benefit from steering | Finding 3 | Boundary from 4 models below threshold vs 1 above |
| A10 | Velocity prediction error (R²) correlates with steering success | Implicit in design | High R² on Math-1.5B (0.892) but no trim tabs found — contradicts this |
| A11 | A single α=0.1 is near-optimal for steering strength | Experimental default | No systematic α sweep reported for best layers |
| A12 | The training pipeline (async loading, GPU caching, checkpoint resume) is mature | Lesson 6 | Infrastructure consumed ~40% of session time — contradicts "mature" label |

## Relational Scan: Implicit Assumptions

| # | Assumption | Inference Chain | Risk if Violated |
|---|------------|-----------------|------------------|
| B1 | The hidden state manifold is approximately Euclidean (linear interpolation works) | Velocity prediction uses MSE loss; steering adds linear delta | Steering could push states off-manifold, causing degradation |
| B2 | Single α per layer is sufficient (not per-token, per-head, or per-position) | Sweep scripts use per-layer α only | Per-token or per-position α could unlock finer-grained control |
| B3 | The trajectory of velocity predicts the trajectory of correctness | TT predicts v_next; steering uses v_pred to modify K/V | High R² doesn't guarantee v_pred points toward correct answer |
| B4 | Correct and incorrect trajectories occupy separable manifolds | Contrastive TT computes v_correct − v_incorrect | If manifolds overlap, contrastive signal is noise |
| B5 | What works for GSM8K will generalize to other reasoning tasks | Cross-dataset test with SVAMP only (2 datasets) | ARC, BBH, MMLU may have different velocity dynamics |
| B6 | KV-cache modification doesn't disrupt model's internal consistency | No analysis of attention patterns after steering | Steering could create conflicting representations across layers |
| B7 | Steering signal propagated through residual stream survives to final output | Signal added to K/V at layer L must influence logits | Attention mechanisms could filter out or override the modification |
| B8 | Instruct models and base models share the same velocity manifold structure | Assumed for Math-1.5B (base) vs Qwen2.5-7B-Instruct | Math-1.5B's lack of trim tabs could be architectural, not training-difference |
| B9 | L9 is a death layer due to its specific computational function | Finding attributes to "what computation layer 9 performs" | Could be random variation, noise sensitivity, or downstream effect |
| B10 | Single-layer steering is superior to multi-layer combinations | Experiments focus on per-layer, not combinatorial | Multi-layer could produce synergy beyond single-layer max |
| B11 | The trajectory data is statistically clean enough for training | Pipeline uses raw generated trajectories | Sampling noise, unreachable states, or off-manifold points in training data |
| B12 | Death layers are stable (same layers are always death layers) | Pattern from single sweep at specific α | Death layer identity could be α-dependent or problem-dependent |

## Potential Scan: Counter-Assumptions

| Assumption | Counter-Assumption "What if ¬[...]" |
|------------|--------------------------------------|
| A1 | Hidden state velocities are predictive but the steering direction derived from them does not point toward correct answers — velocity is a descriptive statistic, not a normative direction |
| A2 | The KV cache surface is too low-level; residual stream or MLP outputs may be better intervention points |
| A3 | Selective per-layer steering is a local optimum that misses combinatorial synergies across multiple layers |
| A4 | Steering below the capability threshold is possible with stronger or differently-structured steering signals — the threshold is a property of the method, not a fundamental limit |
| A5 | Hybrid attention (GDN+FA) may be steerable through the recurrent state path rather than the KV path |
| A6 | The prompt/gen trajectory gap may be bridgeable with domain adaptation (e.g., fine-tune prompt TT on gen data) |
| A7 | Contrastive TT may also be descriptive of the "decisive difference" manifold rather than normative — it describes how trajectories differ without explaining why one is correct |
| A8 | The trim-tab pattern may be dataset-specific; SVAMP is similar to GSM8K in structure |
| A9 | There is no hard capability threshold; the effective threshold is an artifact of steering strength and method |
| A10 | R² is irrelevant to steering success because prediction error is not the same as steering effectiveness |
| A11 | The optimal α varies per layer, per task, and per model; using α=0.1 everywhere masks the true potential |
| A12 | The pipeline is fragile, not mature — 40% overhead on infrastructure indicates fundamental inefficiency |
| B1 | The hidden state manifold is strongly curved/non-Euclidean; linear interpolation pushes states into low-probability regions, which is why steering often degrades output |
| B2 | Per-token α is essential because different token positions have different sensitivity to steering |
| B3 | Velocity predicts direction but not destination — the next state being predictable doesn't mean the trajectory is headed toward correctness |
| B4 | The correct/incorrect manifolds are interleaved at a fine grain; their difference is dominated by noise |
| B5 | Each reasoning domain has distinct velocity dynamics; GSM8K/SVAMP (arithmetic) are not representative |
| B6 | KV-cache modification at a single layer creates internal inconsistency that later layers must compensate for, consuming model capacity |
| B7 | The residual stream amplifies some modifications and attenuates others; steering signal may be filtered out by attention mechanisms |
| B8 | Instruct tuning fundamentally restructures the hidden state manifold; base models and instruct models are incomparable |
| B9 | L9 death is a measurement artifact — it appears dead because of interaction with downstream layers given specific α |
| B10 | Multi-layer steering is harmful because errors compound; per-layer isolates the effect |
| B11 | Trajectory data contains systematic biases from sampling strategy, quantized representations, etc. |
| B12 | Death layers are not stable; they shift with α, dataset, and sampling temperature |

## Bracket Statement

These assumptions are set aside for the analysis. They will be re-examined in Phase 6 (Disparity Detection) to check if any assumption violation is the root cause of observed failures. The most critical assumptions to stress-test are: (1) that velocity prediction points toward correctness (A1→B1 bridge), (2) that the hidden state manifold supports linear interventions (B1), and (3) that the capability threshold is fundamental rather than methodological (A4↔A9).

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Explicit Assumptions | 12 |
| Implicit Assumptions | 12 |
| Counter-Assumptions | 24 |
| Critical for Testing | A1, A4, B1, B3, B4, A11 |
