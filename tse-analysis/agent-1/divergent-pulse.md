# Phase 4: Divergent Pulse

**Subject**: RankAdaptation — Velocity-based latent steering
**Date**: 2026-06-14

---

## Seed Expansion

### Semantic Constellation: Functionally Analogous Entities

| Analogous Entity | Domain | Mapping to Steering |
|------------------|--------|---------------------|
| Gradient descent step | Optimization | v_pred is analogous to a gradient step in hidden state space toward "better" output |
| PID controller | Control theory | TT is the predictor, α is the gain, residual stream is the plant; integral term missing |
| Adversarial perturbation | ML security | Steering is structure-preserving perturbation (like adversarial, but for improvement) |
| Diffusion guidance (classifier-free) | Generative AI | v_correct − v_incorrect = epsilon_correct − epsilon_incorrect in diffusion space |
| LoRA adapter fine-tuning | PEFT | Steering modifies activations at specific layers; LoRA modifies weights at specific layers |
| Gradient checkpointing | ML systems | Both trade computation for memory — steering trades accuracy for directional control |
| Prompt engineering | NLP | Both are "surface-level" interventions that don't modify weights |
| RLHF reward model | RL | Contrastive TT is analogous to a reward model (distinguishes good/bad trajectories) |
| Pruning (lottery ticket) | Compression | Finding trim-tab layers is like finding winning lottery tickets: specific subnetworks matter more |
| Neurofeedback | Neuroscience | External signal fed back into the system to guide it toward desired state |

### Cryptic Analogy Mining

| Atom Function | Abstract Form | Cross-Domain Analogy |
|---------------|---------------|----------------------|
| Predict next hidden state (A2) | Given current position, predict next position | Astronomical ephemeris (predict planetary positions) |
| Select which layers to modify (A4) | Choose intervention points in a causal chain | Triage in emergency medicine (choose which patients to treat first) |
| Scale intervention magnitude (A10) | Amplify or dampen a corrective signal | Volume control in audio mixing (gain staging) |
| Transfer across models (A12) | Apply pattern from one system to another | Language acquisition (apply grammar rules from L1 to L2) |
| Detect correct/incorrect trajectories (A8) | Classify outcomes | Wine tasting (distinguish good/bold from bad/bitter vintages) |

### Hallucinatory Pre-Seed Refinements

| Atom | Pre-Seed | Refined Pre-Seed |
|------|----------|------------------|
| A1 | Oracle velocity field | Oracle velocity field discovered through contrastive pretraining on synthetic data with known ground truth |
| A5 | Computation aligned with steering | Trim-tab layers are those where the gradient of "correctness" with respect to hidden state is aligned with the natural direction of computation |
| A10 | Learned α per token | α(t) = sigmoid(w^T · h_t + b) where w is learned from data |

---

## Mutation Operators

### M1: SUBSTITUTE — Replace steering surface

| Original | Substituted | Variant | Quality Score |
|----------|-------------|---------|---------------|
| A3: KV-cache steering | **Residual stream addition** | Add v_pred directly to residual stream instead of K/V | 4/5 (bypasses attention mechanics) |
| A3: KV-cache steering | **MLP activation modification** | Scale MLP output by predicted velocity | 3/5 (more invasive) |
| A2: Transformer predictor | **Linear predictor** | Simplify TT to linear regression on current state | 2/5 (less expressive) |

### M2: INVERT — Reverse the steering direction

M2-1: Instead of steering toward v_pred, steer AWAY from v_pred (anti-steering). This would intentionally degrade output — useful for understanding death layers by testing if anti-steering at L9 improves accuracy (if death is caused by misalignment, reversing direction should fix it).

| Quality | Score |
|---------|-------|
| Novelty | 4/5 |
| Feasibility | 5/5 (just flip sign of α) |
| Coherence | 4/5 |
| Risk | 3/5 (could amplify harm) |
| Emergent Potential | 4/5 (diagnostic goldmine) |

### M3: SCALE — Modify α magnitude across regimes

| Regime | α Range | Predicted Behavior | Quality |
|--------|---------|-------------------|---------|
| Micro-steering | 0.001-0.01 | Subtle bias, high safety | 4/5 |
| Standard | 0.05-0.2 | Current regime | Baseline |
| Macro-steering | 0.5-1.0 | Strong modification, high divergence | 3/5 |
| Extreme | 2.0-10.0 | Model collapse, diagnostic only | 2/5 |

### M4: REORDER — Change intervention timing

M4-1: Apply steering only during the first N tokens (early-only steering). Hypothesis: early token steering shapes the reasoning path before it commits.
M4-2: Apply steering only during the last N tokens (late-only steering). Hypothesis: late token steering corrects only the final answer.
M4-3: Apply steering with schedule (α decays from high to low over generation).

### M5: MERGE — Combine steering mechanisms

M5-1: **Dual-surface steering**: Apply v_pred to BOTH KV-cache AND residual stream with different α values.
M5-2: **Contrastive + standard ensemble**: Average v_standard and v_contrastive.
M5-3: **Multi-layer combination**: Apply at L2, L8, L10 simultaneously with learned mixing weights.

| Quality | Score |
|---------|-------|
| Novelty | 4/5 |
| Feasibility | 3/5 (requires new infrastructure) |
| Coherence | 5/5 |
| Risk | 4/5 (error compounding) |
| Emergent Potential | 5/5 (combinatorial synergy) |

### M6: SPLIT — Decouple steering components

M6-1: Separate v_pred into magnitude and direction. Steer with direction only (unit vector), scale by token-level confidence.
M6-2: Split K and V steering: steer only K entries, or only V entries, not both simultaneously.

### M7: ABSTRACT — Generalize the steering concept

M7-1: **Task-conditional steering**: Train different TTs for different task types (math, coding, reasoning) and select based on task classifier.
M7-2: **Meta-steering**: Train a "steering policy" that decides when AND where AND how much to steer, using RL.

### M8: CONCRETIZE — Specialize to a specific component

M8-1: **Attention-head-specific steering**: Within a layer, steer only specific attention heads instead of all heads uniformly.
M8-2: **Token-position-specific steering**: Steer only at specific token positions (e.g., after "=", before "Therefore").

### M9: TRANSPOSE — Move to different architecture

M9-1: Apply steering to **Mixture-of-Experts** models — modify expert routing weights instead of K/V.
M9-2: Apply steering to **encoder-decoder** models — steer encoder outputs.

### M10: NEGATE — Use negative contrastive signal

M10-1: Instead of v_correct − v_incorrect, use v_incorrect − v_correct to intentionally degrade. Useful for ablation studies.
M10-2: Use ALL negative examples from incorrect trajectories with positive weighting as a "what NOT to do" signal.

### M11: RANDOMIZE — Replace TT with random direction

M11-1: Replace v_pred with random unit vector (same magnitude). Tests whether steering signal matters or just perturbation.
M11-2: Replace v_pred with v_random scaled by R². Tests whether direction specificity matters.

### M12: OSCILLATE — Time-varying steering

M12-1: Apply α that oscillates (sinusoidal) during generation. Tests whether the model can "ride" the oscillation to better states.
M12-2: Apply steering only every Nth token (skip connections).

---

## Forced Collisions

### Speculative Analogues (10 per Atom)

**For A5 (Trim-Tab Layer)**:
1. A task-specific attention head that fires on math tokens
2. A layer that computes numerical magnitude estimation
3. The layer with highest mutual information with answer correctness
4. The layer closest to the "concept bottleneck" for arithmetic reasoning
5. A layer that performs working-memory-like functions (maintaining intermediate results)
6. The layer with lowest functional redundancy
7. A layer that, when ablated (not steered, fully removed), causes the largest accuracy drop
8. The layer whose representation space has the clearest correct/incorrect separation
9. The layer whose velocity field is most aligned with the gradient of a math-proxy objective
10. The layer with highest intrinsic dimensionality (most expressive power)

**For A6 (Death Layer)**:
1. A layer that performs structural/template organization of the answer
2. A layer that gates between reasoning and non-reasoning modes
3. A layer with low intrinsic dimensionality (fragile to perturbation)
4. A layer whose velocity field is orthogonal to the correctness gradient
5. A layer that's a "bottleneck" where all reasoning paths converge
6. A layer implementing positional consistency (destroying it breaks token ordering)
7. A layer with highest functional redundancy (steering disrupts redundant consensus)
8. A layer that performs "reality check" — verifying internal consistency
9. A layer whose output is used directly by the LM head (near-final layers)
10. A layer that calibrates probability distributions

### Orthogonal Mechanisms per Master Regulator

**MR#1: Per-Layer Selectivity**:
1. Monte Carlo tree search over layer combinations
2. Gradient-based saliency (gradients of accuracy w.r.t. layer activations)
3. Causal mediation analysis (Pearl's do-operator on layers)
4. Active learning (choose next layer to test based on uncertainty)
5. Random sampling with Thompson sampling (explore/exploit tradeoff)

**MR#2: α Strength**:
1. α = f(perplexity) where f is learned calibration curve
2. α = α_0 / (1 + t/T) where t is token position, T is total tokens
3. α per head within layer based on head attention entropy
4. α = 0 for tokens where baseline accuracy is already high
5. α sampled from learned distribution per (layer, task, token_position)

**MR#3: Contrastive Signal**:
1. Direct preference optimization (DPO) style: train TT to prefer correct trajectories
2. Reward-modeled steering: train a reward model, use its gradient as steering signal
3. Contrastive with hard-negative mining (find trajectories near decision boundary)
4. Multi-step contrastive: v_correct trajectory − v_incorrect trajectory over multiple tokens
5. Contrastive ensemble: bootstrap N correct/incorrect TTs, average their predictions

### Paradoxical Combinations

**PC-1**: "Assume the optimal steering is to steer NO layers — the improvement from per-layer selectivity is actually the removal of noise from all-layer steering, not the addition of a useful signal. The trim-tab effect is therefore a release-from-interference effect."

**PC-2**: "Assume death layers are the MOST informative layers, not the least — their extreme sensitivity to steering means they contain the most concentrated computation. The goal should be to protect death layers from interference, not to avoid them."

**PC-3**: "Assume the contrastive direction is the WRONG direction — v_correct − v_incorrect points toward the region that the model itself identifies as 'different,' which may be a spurious difference (e.g., formatting, token length) rather than a correctness-relevant difference."
