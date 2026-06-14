# Phase 10: Hyperstitional Bridge

**Subject**: RankAdaptation — Velocity-based latent steering
**Date**: 2026-06-14

---

## Overview

Hyperstitional hypotheses are precise, falsifiable claims that, if true, would transform the understanding of the steering problem. They bridge from retrospective analysis to prospective experiment.

---

## H-1: Direction Alignment Hypothesis (Structural)

| Property | Detail |
|----------|--------|
| **ID** | H-1 |
| **Type** | Structural |
| **Statement** | A layer is a trim-tab if and only if its natural computation direction (the dominant eigenvector of its output covariance) has positive cosine similarity with the steering direction (v_pred). A layer is a death layer if and only if this cosine similarity is negative. |
| **Falsification Criteria** | |
| ... Refutation | Compute cosine similarity between v_pred and layer-output PCA1 for L8 and L9. If similarity is positive for BOTH layers (not just L8), hypothesis is false. |
| ... Confirmation | L8 shows positive cosine similarity (cos>0.3); L9 shows negative (cos<−0.3). |
| **Minimum Experiment** | Collect 1000 hidden states from L8 and L9 during generation, compute PCA per layer, dot product with v_pred at corresponding token positions. |
| **Risk** | If false, the trim-tab/death-layer classification may be spurious or driven by different factors. |
| **Value** | If true, provides the first mechanistic theory of steelayer function. Enables prediction of trim-tab/death-layer identity without brute-force sweep. |
| **Channel** | `[experiment][theory]` |

---

## H-2: Off-Manifold Death Hypothesis (Relational)

| Property | Detail |
|----------|--------|
| **ID** | H-2 |
| **Type** | Relational |
| **Statement** | Steering at death layers (L9, L15+) reduces accuracy because the modified K/V values produce hidden states outside the training data manifold, causing the model to enter a region of the representation space for which later layers have no learned computation. |
| **Falsification Criteria** | |
| ... Refutation | If random perturbation of the same magnitude ALSO causes degradation at L9 (tested via H0-2), then off-manifold is confirmed. If random perturbation is benign, death is computation-specific, not manifold-related. |
| ... Confirmation | Random perturbation at L9 causes similar degradation to TT steering at L9. |
| **Minimum Experiment** | M11-1 (random direction baseline) at L9 with matched magnitude. |
| **Risk** | If false, we incorrectly believe death is a manifold problem and search for manifold-preserving steering techniques that don't help. |
| **Value** | If true, the steering problem reduces to a manifold-preservation problem, which has known solutions from manifold learning and generative modeling. |
| **Channel** | `[experiment][codebase]` |

---

## H-3: Contrastive Direction Is Normative (Potential)

| Property | Detail |
|----------|--------|
| **ID** | H-3 |
| **Type** | Potential |
| **Statement** | The direction v_correct − v_incorrect, when used as a steering signal, will produce larger accuracy improvements at trim-tab layers than the standard TT's v_all direction, because it encodes the *difference* between correct and incorrect reasoning rather than the *average* of all reasoning. |
| **Falsification Criteria** | |
| ... Refutation | Contrastive steering at L8 produces ≤ standard TT steering at L8 (no improvement). OR: contrastive steering creates trim-tabs at DIFFERENT layers than standard TT (pattern doesn't preserve). |
| ... Confirmation | Contrastive at L8 produces > standard TT at L8 (e.g., +25pp vs +20pp) AND preserves L8 as best layer. |
| **Minimum Experiment** | Run contrastive evaluation script (already exists). |
| **Risk** | If false, the project's primary next step (contrastive approach) may be a dead end, wasting significant inference and analysis time. |
| **Value** | If true, establishes normative steering as a viable paradigm, potentially doubling the improvement ceiling. |
| **Channel** | `[experiment]` |

---

## H-4: α Capability Threshold (Structural)

| Property | Detail |
|----------|--------|
| **ID** | H-4 |
| **Type** | Structural (modulatory) |
| **Statement** | The capability threshold (models below ~40% GSM8K cannot be steered) is not a fixed property of the model but is jointly determined by (model baseline, α). For any model, there exists an α > 0 such that steering improves accuracy, provided α < α_collapse (the α that destroys the model). |
| **Falsification Criteria** | |
| ... Refutation | For Math-1.5B (38% baseline), NO α in [0.001, 1.0] produces any accuracy improvement at any layer. |
| ... Confirmation | At least 1 (layer, α) pair for Math-1.5B produces accuracy > 38%. |
| **Minimum Experiment** | RECOMB-FP2: α sweep on Math-1.5B across all layers. |
| **Risk** | If false, the capability threshold is confirmed as a fundamental limitation, limiting the approach to models above ~7B with instruct tuning. |
| **Value** | If true, the approach extends to smaller and base models, dramatically expanding the usable model set. |
| **Channel** | `[experiment][theory]` |

---

## H-5: Instruct Tuning Creates Steering Manifold (Relational)

| Property | Detail |
|----------|--------|
| **ID** | H-5 |
| **Type** | Relational |
| **Statement** | Instruct tuning (not model size) is the causal factor that creates a standardized reasoning manifold, enabling the trim-tab/death-layer pattern. A 1.5B instruct-tuned model will show trim-tabs; a 7B base model will not. |
| **Falsification Criteria** | |
| ... Refutation | A 1.5B instruct-tuned model shows no trim-tabs (size dominates) OR a 7B base model shows trim-tabs (instruct isn't necessary). |
| ... Confirmation | 1.5B-instruct shows at least 1 trim-tab layer; 7B-base shows none. |
| **Minimum Experiment** | D5 resolution: test both 1.5B instruct model and 7B base model. |
| **Risk** | If false, we misattribute the cause of trim-tabs and may waste effort on instruction-tuning models unnecessarily. |
| **Value** | If true, instruct-based fine-tuning becomes a prerequisite for steering, guiding model selection. |
| **Channel** | `[experiment]` |

---

## H-6: Dual-Surface Steering (Potential)

| Property | Detail |
|----------|--------|
| **ID** | H-6 |
| **Type** | Potential |
| **Statement** | Combining KV-cache steering with residual stream addition using the same predicted velocity produces greater accuracy improvement than either surface alone, because the dual modification reinforces the steering signal through both the attention and feedforward pathways. |
| **Falsification Criteria** | |
| ... Refutation | Dual-surface ≤ max(single-surface) at L8. |
| ... Confirmation | Dual-surface > max(KV-only, residual-only) at L8. |
| **Minimum Experiment** | Add residual stream modification to existing KV-cache script; test at L8 with α=0.1 on each surface. |
| **Risk** | Low — both surfaces are well-understood independently. Dual may cause interference. |
| **Value** | If true, reveals a new dimension of steering control. |
| **Channel** | `[experiment]` |

---

## H-7: Cross-Layer Interference (Structural)

| Property | Detail |
|----------|--------|
| **ID** | H-7 |
| **Type** | Structural |
| **Statement** | Steering at two layers simultaneously produces worse-than-additive effects because the residual stream's linearity assumption breaks down: the attention mechanism at later layers receives conflicting signals from the two modified layers. |
| **Falsification Criteria** | |
| ... Refutation | Steering at L2+L8 produces accuracy ≥ L8 alone + L2 alone (additive or super-additive). |
| ... Confirmation | L2+L8 produces accuracy < L8 alone (sub-additive). |
| **Minimum Experiment** | RECOMB-FP1: L2+L8 pair sweep. |
| **Risk** | Low — informative regardless of outcome. |
| **Value** | If true, motivates single-layer as optimal and constrains future multi-layer approaches. If false, opens the door to combinatorial optimization. |
| **Channel** | `[experiment]` |

---

## H-8: Universal Velocity Invariant (Theoretical)

| Property | Detail |
|----------|--------|
| **ID** | H-8 |
| **Type** | Structural (theoretical) |
| **Statement** | The function f(h_t) → v_{t+1} is approximately invariant across transformer-based language models trained with next-token prediction, up to a linear transformation of the hidden state space. This invariance is a consequence of the universality of next-token prediction as a training objective. |
| **Falsification Criteria** | |
| ... Refutation | Cross-model transfer (A→B) fails at significantly greater rate than within-model (A→A) for 5+ diverse model pairs. |
| ... Confirmation | Cross-model transfer works across ≥3 architecture families. |
| **Minimum Experiment** | Test cross-model transfer between LLaMA-3, Mistral-7B, Gemma-7B (all MHA, similar sizes). |
| **Risk** | If false, velocity dynamics are model-specific, limiting transfer and generalization. |
| **Value** | If true, establishes velocity dynamics as a fundamental property of LLMs, with major implications for mechanistic interpretability and model merging. |
| **Channel** | `[theory]` |

---

## H-9: Steering Is Gradient Following (Potential)

| Property | Detail |
|----------|--------|
| **ID** | H-9 |
| **Type** | Potential |
| **Statement** | The predicted velocity v_pred is approximately proportional to the gradient of "answer correctness" with respect to hidden states, averaged over the training distribution. Steering is therefore a form of gradient ascent on correctness in hidden state space. |
| **Falsification Criteria** | |
| ... Refutation | v_pred has cosine similarity < 0.1 with the empirical gradient of accuracy w.r.t. hidden states (computed via finite differences). |
| ... Confirmation | v_pred aligns with the accuracy gradient (cos > 0.5). |
| **Minimum Experiment** | Compute accuracy gradient for a batch of examples at L8; compare with v_pred. |
| **Risk** | If false, guidance analogy is flawed; steering works through a different mechanism. |
| **Value** | If true, connects steering to optimization theory, enabling gradient-based techniques for improvement. |
| **Channel** | `[theory][experiment]` |

---

## H-10: Token-Position Specific Steering (Structural)

| Property | Detail |
|----------|--------|
| **ID** | H-10 |
| **Type** | Structural |
| **Statement** | The trim-tab/death-layer classification reverses at different token positions: L8 is a trim-tab for early tokens (problem understanding) and a death layer for late tokens (answer generation). Per-layer classification aggregates across positions, obscuring position-dependent effects. |
| **Falsification Criteria** | |
| ... Refutation | L8's per-token accuracy effects are uniformly positive across all token positions. |
| ... Confirmation | L8 helps early but hurts late, or vice versa. |
| **Minimum Experiment** | Track per-token accuracy during steering; compare to unsteered per-token accuracy. |
| **Risk** | Medium — requires per-token accuracy tracking infrastructure. |
| **Value** | If true, enables token-position-aware steering with higher precision. |
| **Channel** | `[experiment]` |

---

## Summary

| H | Name | Type | Key Prediction | Critical Evidence | Channel |
|---|------|------|----------------|-------------------|---------|
| H-1 | Direction Alignment | Structural | Cosine similarity predicts trim-tab/death | L8 cos>0, L9 cos<0 | `[experiment][theory]` |
| H-2 | Off-Manifold Death | Relational | Random perturbation same as TT at L9 | M11-1 experiment | `[experiment][codebase]` |
| H-3 | Contrastive Normative | Potential | Contrastive > standard TT at L8 | A1 experiment | `[experiment]` |
| H-4 | α Threshold | Structural | Sub-threshold model steerable with high α | RECOMB-FP2 | `[experiment][theory]` |
| H-5 | Instruct Manifold | Relational | Instruct creates steerable space | 1.5B-instruct test | `[experiment]` |
| H-6 | Dual-Surface | Potential | KV+Residual > max(single) | New script needed | `[experiment]` |
| H-7 | Cross-Layer Interference | Structural | L2+L8 < L8 alone | RECOMB-FP1 | `[experiment]` |
| H-8 | Universal Velocity | Theoretical | f(h_t) invariant up to linear transform | Multi-family transfer | `[theory]` |
| H-9 | Steering as Gradient | Potential | v_pred ≈ ∂accuracy/∂h | Finite-difference check | `[theory][experiment]` |
| H-10 | Position-Dependent | Structural | Trim-tab classification reverses per position | Per-token accuracy tracking | `[experiment]` |
