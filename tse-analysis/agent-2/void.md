# Phase 0: VOID — Assumption Surfacing & Bracketing

## Subject: Velocity-based Latent Steering for Language Model Reasoning (RankAdaptation)

---

## Structural Scan — Explicit Assumptions

These are directly stated or strongly implied in PROJECT_DEBRIEF.md and the codebase:

| # | Assumption | Source | Evidence |
|---|-----------|--------|----------|
| A1 | Hidden state velocities during generation are learnable via a TrajectoryTransformer | Debrief §Finding 1 | R²=0.85-0.94 across models |
| A2 | Steering a model improves reasoning accuracy (measured by GSM8K accuracy) | Debrief §Executive Summary | L8: +20pp on Qwen2.5-7B |
| A3 | Per-layer selectivity is mandatory — all-layers steering is net negative | Debrief §Finding 4, §Key Lesson 2 | Death layers at L9, L15+ |
| A4 | Steering requires the model to already have the target capability | Debrief §Finding 3 | <38% baseline → all steering harmful |
| A5 | KV-cache modification is the correct geometric surface for steering | Debrief §Steering Mechanisms | Logit correction failed; KV-cache works |
| A6 | Generation trajectories differ fundamentally from prompt trajectories | Debrief §Key Lesson 4 | R²=0.94 gen vs R²=0.62 prompt |
| A7 | Standard MHA architectures (Qwen2.5, LLaMA, SmolLM2) are preferred | Debrief §Finding 6 | Hybrid attention (GDN+FA) limits steering |
| A8 | The trim-tab/death-layer pattern generalizes across datasets and model families | Debrief §Finding 2, §Cross-dataset gen | SVAMP replicates GSM8K pattern; cross-model transfer preserves it |
| A9 | Contrastive training converts TT from descriptive → normative | Debrief §Key Lesson 5 | Implicit: v_correct − v_incorrect should steer toward correct manifold |
| A10 | GSM8K accuracy is an adequate proxy for "reasoning ability" | Debrief §Executive Summary | Primary metric throughout |

## Relational Scan — Implicit Assumptions

| # | Assumption | Inference Chain | Why It's Implicit |
|---|-----------|----------------|-------------------|
| B1 | The hidden state manifold at generation time is approximately Euclidean near the current trajectory | Steering via h' = h + α·v assumes vector addition is valid in activation space | Never tested; linear interpolation assumed |
| B2 | A single α per layer is sufficient — per-position α is unnecessary | Only per-layer α sweeps performed; no per-token α | Sweep scripts use single α for entire generation |
| B3 | The TrajectoryTransformer's learned velocity field is sufficiently smooth that interpolation between training examples works | TT evaluated on held-out problems without explicit smoothness check | R² alone doesn't measure smoothness |
| B4 | "Correct" and "incorrect" trajectories inhabit separable manifolds | Contrastive steering (v_c − v_i) assumes these manifolds have a distinguishable mean direction | Math-1.5B showed NO trim tabs with either standard or contrastive TT |
| B5 | Layer-specific effects are additive — steering L2+L8 ≈ steering(L2) + steering(L8) | Multi-layer combinations tested without interaction term analysis | Stage 3 in autonomous sweep tests combos but doesn't model interactions |
| B6 | The steering vector α·v at step t only affects future tokens, not past | KV-cache modification at last position is a causal intervention | Implicit in the KV-cache steering mechanism |
| B7 | Cross-model transfer works because velocity dynamics are model-agnostic | SmolLM2→7B preserved L8 pattern | Only one transfer direction tested; could be coincidental |
| B8 | 73% baseline on Qwen2.5-7B is a strong starting point | Used as primary success benchmark | Could saturate; ceiling effects possible |
| B9 | TT trained on generation trajectories captures causal dynamics, not just auto-regressive noise | High R² could mean TT learns to predict the auto-regressive transformer's inherent smoothness | Not tested against a null model (e.g., linear extrapolation) |

## Potential Scan — Counter-Assumptions

For each assumption above, the negation:

| # | Counter-Assumption | Violation Symptoms |
|---|-------------------|-------------------|
| ¬A1 | Velocities are not learnable — high R² is an artifact (e.g., auto-correlation in adjacent hidden states) | TT performs no better than linear baseline |
| ¬A2 | Steering does NOT improve reasoning — observed +20pp is statistical noise or dataset artifact | Fails to replicate on held-out dataset splits |
| ¬A3 | All-layers steering could work with better α per layer — the problem is coarse α, not layer count | Per-layer α vector on all layers beats selective steering |
| ¬A4 | Steering could create capability in models with <38% baseline using different steering mechanisms (not KV-cache) | An alternative mechanism succeeds on small models |
| ¬A5 | KV-cache is NOT the correct surface — other surfaces (e.g., MLP activations, attention logits) would work better | Alternative geometric surface outperforms KV-cache |
| ¬A6 | Prompt and generation trajectories are not fundamentally different — apparent difference is due to data quality or quantity | Matching data sizes eliminates the R² gap |
| ¬A7 | Hybrid attention models CAN be steered effectively with the right mechanism | Novel steering approach succeeds on Qwen3.5-2B |
| ¬A8 | The trim-tab pattern does NOT generalize — L8 effect on SVAMP (+4pp not +20pp) shows dataset-dependent magnitude | Broader evaluation (ARC, BBH, MMLU) shows no consistent pattern |
| ¬A9 | Contrastive steering is NOT additive with standard TT — the difference vector v_c − v_i pushes the model off-manifold | Contrastive steering degrades accuracy below baseline |
| ¬A10 | GSM8K accuracy is a poor proxy for reasoning — it measures arithmetic, not reasoning | High GSM8K doesn't correlate with performance on non-math reasoning tasks |

## Bracket Statement

These assumptions are set aside for the main analysis. They will be re-examined in Phase 6 (Disparity Detection) to check if any assumption violation is the root cause of observed failures. The counter-assumptions in particular serve as "suspicion seeds" — if the analysis converges on a finding that matches a counter-assumption, that assumption should be elevated from bracketed to actively tested.

Key bracketing note: **Assumption B1 (Euclidean hidden state manifold) and B4 (separable correct/incorrect manifolds) are the most structurally fragile.** If these are false, the entire steering framework collapses. They receive special attention in Phase 8 (Mechanistic Interpretability Check).
