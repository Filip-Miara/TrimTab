# Phase 0: VOID — Assumption Surfacing & Bracketing

**Subject**: Velocity-based latent steering for language model reasoning (RankAdaptation)
**Date**: 2026-06-14

---

## Structural Scan: Explicitly Stated Assumptions

Sources: PROJECT_DEBRIEF.md, system architecture, experiment protocols.

| # | Assumption | Source | Evidence Grounding |
|---|-----------|--------|-------------------|
| A1 | "Hidden state velocities during generation ARE learnable" | Finding 1 (R²=0.85-0.94) | ✅ High confidence — replicated across 3 models |
| A2 | "Steering requires the model to already have the target capability" | Finding 3 | ✅ Supported by 5-model comparison |
| A3 | "Per-layer selectivity is mandatory — all-layers steering is net negative" | Finding 4 | ✅ Supported by per-layer sweep data |
| A4 | "Generation trajectories differ from prompt trajectories" | Lesson 4 | ✅ R²=0.62 vs 0.94 |
| A5 | "Standard MHA is preferred for steering experiments" | Finding 6 | ✅ Qwen3.5 (hybrid) all failed |
| A6 | "Cross-model transfer preserves trim-tab pattern" | Finding 5 | ✅ SmolLM2→7B confirmed |
| A7 | "L8 is a trim-tab layer for Qwen2.5-7B" | Finding 2 | ✅ +20pp on 100 problems |
| A8 | "L9 is a death layer for Qwen2.5-7B" | Finding 2 | ✅ -23pp on 100 problems |
| A9 | "Trim-tab pattern generalizes across math datasets" | Finding 2 (SVAMP) | ✅ GSM8K→SVAMP replicates |
| A10 | "Contrastive TTs convert descriptive to normative prediction" | Lesson 5 | ⚠️ Hypothesized, evaluation pending |

## Relational Scan: Implicit Assumptions

| # | Assumption | Inference Chain | Risk Level |
|---|-----------|-----------------|------------|
| B1 | **Velocity direction encodes correctness**: Moving toward the correct hidden state trajectory is both necessary and sufficient for improved reasoning | The entire steering mechanism depends on velocity predictions pointing in a direction that correlates with answer quality. If velocities encode only fluency/coherence rather than correctness, steering is a no-op for accuracy. | HIGH |
| B2 | **The KV cache is the correct intervention surface**: Modifying K/V projections at a layer is equivalent to modifying the computation at that layer | KV-cache steering assumes that the value stored in the cache is the mechanism, not a side-effect. If the cache merely records a computation that happens elsewhere, steering the cache misses the causal node. | HIGH |
| B3 | **Layer-level granularity is sufficient**: Per-layer steering can achieve near-optimal results; finer (head-level, token-level) or coarser (block-level) granularity is not needed | The 12-layer sweep found trim-tabs and death layers, but doesn't prove that within-layer mixing isn't better. Optimal steering might be head-specific or token-specific within a layer. | MEDIUM |
| B4 | **Linear extrapolation (α·v_pred) is a valid steering mechanism**: Applying a scalar multiple of predicted velocity to the hidden state moves the state in a useful direction | TT predicts velocity, but the optimal steering direction is not necessarily parallel to the predicted velocity. Curvature in hidden state manifold might mean α·v is a poor approximation. | HIGH |
| B5 | **The TrajectoryTransformer's architecture (DeltaNet/MLP) is adequate for the task**: TT can capture the relevant dynamics of the generation process | If generation dynamics are higher-order (acceleration-dependent, or context-dependent), the TT's architecture may miss critical structure. | MEDIUM |
| B6 | **GSM8K accuracy is a sufficient proxy for reasoning quality**: Steering that improves GSM8K accuracy will generalize to other reasoning tasks | Math word problems test a narrow band of reasoning. Improvements might be task-specific (surface pattern matching) rather than general reasoning improvements. | MEDIUM |
| B7 | **Correct-only and incorrect-only trajectories are separable**: The contrastive TT formulation assumes the correct/incorrect hidden state manifolds have distinct velocity dynamics | If the manifolds overlap significantly (same trajectory shapes, different outcomes due to token-level stochasticity), contrastive TT will learn noise, not signal. | HIGH |
| B8 | **The steering coefficient α is constant across tokens within a layer**: One α per layer, applied uniformly to all tokens at that layer | Different token positions may need different steering strengths. Early tokens (problem statement) vs late tokens (answer generation) might require different α. | MEDIUM |
| B9 | **Small model failure is due to capability threshold, not architecture or data**: The 4-6% baseline models cannot be steered because they lack the latent capability | Alternative: small models CAN be steered but the steering signal is too weak relative to noise, or the KV-cache mechanism operates differently at small scales. | MEDIUM |
| B10 | **The prompt format (chat template) is responsible for the 4%→73% baseline jump**: The chat template provided the missing context | This assumption is supported by evidence (the baseline jumped after applying chat template), but the mechanism is unclear — does the template activate reasoning circuits or just reduce token-level perplexity? | LOW |

## Potential Scan: Counter-Assumptions

For each assumption (explicit and implicit), formulate a counter-assumption:

| ID | Original | Counter-Assumption |
|----|----------|-------------------|
| ¬A1 | "Generation trajectories are learnable" | "What if high R² is an artifact of low-rank token embeddings being linearly predictable, and the velocity predictions capture no semantic structure?" |
| ¬A2 | "Steering requires existing capability" | "What if steering CAN create capability, but only with the right steering surface (e.g., activation steering, not KV-cache)? The threshold observation is a surface-specific artifact." |
| ¬A3 | "Per-layer selectivity is mandatory" | "What if all-layers steering is negative only because α is constant across layers, and per-layer α optimization (asymmetric α) would make all-layers steering beneficial?" |
| ¬A4 | "Generation trajectories differ from prompt" | "What if the prompt/gen R² gap is due to token count (prompts are shorter) rather than a fundamental difference in dynamics?" |
| ¬A5 | "Standard MHA is preferred" | "What if hybrid attention CAN be steered, but via a different mechanism (e.g., modifying the GatedDeltaNet's recurrent state instead of the KV cache)?" |
| ¬A6 | "Cross-model transfer preserves pattern" | "What if the preserved L8 pattern is coincidental — both models happened to have L8 as trim-tab for unrelated reasons (e.g., both are 7B-parameter English LMs)?" |
| ¬A7 | "L8 is a trim-tab" | "What if L8's +20pp is a statistical fluke on 100 problems (binomial CI: ±10pp at 95% confidence)? The true effect might be 0-10pp." |
| ¬A8 | "L9 is a death layer" | "What if L9's -23pp is actually a beneficial signal that requires negative α (push AWAY from predicted velocity) rather than positive α?" |
| ¬A9 | "Pattern generalizes across datasets" | "What if GSM8K and SVAMP are similar enough (both grade-school math) that pattern generalization is trivial, and non-math datasets would show a completely different pattern?" |
| ¬A10 | "Contrastive TT is normative" | "What if v_correct − v_incorrect learns the difference in fluency/style (longer answers for correct, shorter for incorrect) rather than the difference in reasoning quality?" |
| ¬B1 | "Velocity direction encodes correctness" | "What if velocity direction encodes syntactic/surface features (e.g., token type, position) rather than semantic correctness, and steering merely biases token distribution?" |
| ¬B2 | "KV cache is the correct intervention" | "What if modifying K/V entries creates a distribution mismatch that degrades the attention computation, and the +20pp at L8 is from chaos-based randomness rather than meaningful steering?" |
| ¬B3 | "Layer-level granularity is sufficient" | "What if optimal steering requires head-level or even attention-head-element-level granularity, and per-layer steering leaves 50%+ of potential improvement on the table?" |
| ¬B4 | "Linear α·v steering is valid" | "What if the hidden state manifold is highly curved, and extrapolating along the tangent (velocity) vector moves the state OFF the manifold entirely, creating out-of-distribution hidden states?" |
| ¬B5 | "TT architecture is adequate" | "What if generation dynamics are second-order (acceleration-dependent), requiring a model that predicts both velocity AND acceleration (delta of deltas)?" |
| ¬B6 | "GSM8K accuracy is sufficient" | "What if GSM8K improvements reflect improved arithmetic token prediction (better number generation) rather than improved reasoning? The model learns to output correct numbers, not correct reasoning chains." |
| ¬B7 | "Correct/incorrect trajectories are separable" | "What if correct and incorrect trajectories share the same manifold, differing only in the final few tokens, and contrastive TT learns only the ending divergence rather than the reasoning path?" |
| ¬B8 | "α is constant across tokens" | "What if the optimal α varies by ~5× across token positions (high at critical reasoning steps, near-zero elsewhere), and a single α washes out the effect?" |
| ¬B9 | "Small model failure = capability threshold" | "What if small models CAN be steered but the steering vector needs to be larger (higher α) because the hidden state manifold is more compressed? The threshold observation may be an α-search artifact." |
| ¬B10 | "Chat template activates reasoning" | "What if the chat template changes output length/token distribution rather than reasoning quality, and the baseline jump is a decoding artifact (longer generations have higher chance of correct answer)?" |

## Bracket Statement

These assumptions are set aside for the analysis. They will be re-examined in Phase 6 (Disparity Detection) to check if any assumption violation is the root cause of observed failures. The counter-assumptions are carried forward as seeds for divergence in Phase 4.

**Key risk vectors**:
1. The steering mechanism's validity (B1, B2, B4) — if any of these are false, the entire approach is compromised
2. The contrastive TT's effectiveness (A10, B7) — if false, the normative direction doesn't exist
3. The capability threshold hypothesis (B9) — if false, steering might work on small models via different mechanisms
4. GSM8K as proxy (B6) — if false, findings might not transfer to real reasoning tasks

**Total assumptions bracketed**: 20 (10 explicit + 10 implicit)
**Counter-assumptions generated**: 20
