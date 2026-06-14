# Disparity Matrix — RankAdaptation

## Disparity Catalog

### D1: High R² ≠ Good Steering (logical_contradiction — structural/fundamental)
- **Concepts**: A4 (TT, R²=0.855) × A10 (GSM8K accuracy) 
- **Conflict**: TT's high R² means it accurately predicts the model's velocity. But the model's velocity leads to both correct and incorrect answers. High R² means faithful *error reproduction* — the TT describes the model's dynamics, including its mistakes.
- **Severity**: FUNDAMENTAL
- **Resolution**: SYNTHESIS — Contrastive TT converts descriptive → normative by learning the *difference* between correct and incorrect trajectories. The hybrid approach (v_std + β·(v_c - v_i)) preserves the high R² descriptive power while adding a normative correction term.

### D2: Distribution Shift (operational_incompatibility — relational)
- **Concepts**: A16 (prompt→generation distribution shift) × C2-3 (velocity prediction during generation)
- **Conflict**: TT trained on generation data works; TT trained on prompt data doesn't (0% accuracy at generation time). The hidden state distribution during generation differs from prompt-time distribution.
- **Severity**: STRUCTURAL (resolved)
- **Resolution**: SUBSTITUTION — Already resolved by the project: train TT on generation trajectories instead of prompt trajectories. The generation-trained TT achieves R²=0.85-0.94. Further resolution via contrastive training on generation trajectories.

### D3: Capability Threshold Paradox (assumption_clash — fundamental)
- **Concepts**: A15 (steering requires ~40% baseline) × A15 implied (steering works on any model with modifications)
- **Conflict**: All models below ~40% GSM8K baseline show steering FAILURE (accuracy degrades). But why should a continuous capability measure have a discrete threshold? Is 39% fundamentally different from 41%?
- **Severity**: FUNDAMENTAL
- **Resolution**: BOUNDING — Document as a discovered limit. The threshold may reflect a phase transition in hidden state manifold separability. At R²=0.83-0.89, both "correct" and "incorrect" manifolds exist, but below the threshold they may be completely entangled. An experiment: check if v_c and v_i trajectories from below-threshold models are separable in latent space (via linear probe on TT representations). If inseparable, the threshold is a manifold property; if separable, the threshold is about steering mechanism.

### D4: Math-1.5B Anomaly (abstraction_mismatch — relational)
- **Concepts**: A15 (38% ≈ near threshold) × Observed (no trim tabs, all harmful)
- **Conflict**: Math-1.5B has 38% baseline, close to the 40% threshold. But unlike Qwen2.5-7B (73%), it shows NO trim tabs with either standard or contrastive TT.
- **Severity**: STRUCTURAL (unresolved)
- **Resolution**: ABSTRACTION — Abstract the explanation: it's not just about baseline accuracy but about *instruct-tuning*. Math-1.5B is a base model; 7B is instruct-tuned. Instruct-tuning may be necessary to create separable correct/incorrect manifolds. Test: apply steering to a similarly-sized instruct-tuned model (e.g., Qwen2.5-1.5B-Instruct).

### D5: Contrastive Cancellation (logical_contradiction — potential)
- **Concepts**: A11 (v_c - v_i) × A3 (velocity = shared dynamics)
- **Conflict**: If both correct and incorrect trajectories share fundamental language modeling dynamics (syntax, token prediction, etc.), then v_c - v_i cancels this shared structure. But the shared structure might be essential for coherent generation.
- **Severity**: FUNDAMENTAL
- **Resolution**: SEPARATION with SYNTHESIS — Replace subtraction with weighted combination: v = v_std + β·(v_c - v_i). v_std preserves language modeling dynamics; the contrastive term adds normative direction. If v_std = (v_c + v_i)/2 and v_c - v_i is the difference, then v_std + β·(v_c - v_i) = (1/2 + β)·v_c + (1/2 - β)·v_i. At β=0.5, this equals v_c (steer toward correct manifold entirely). At β=0, this equals v_std (descriptive). The hybrid interpolates between descriptive and normative steering.

### D6: Per-Layer Granularity vs Head-Level Effects (abstraction_mismatch — structural)
- **Concepts**: C3-3 (per-layer steering) × A14 (attention heads within layer)
- **Conflict**: Layer-level steering assumes all heads within a layer have the same trim-tab/death-layer polarity. But different heads perform different functions (syntactic parsing, positional encoding, semantic processing), and steering might help some heads while hurting others within the same layer.
- **Severity**: STRUCTURAL (unresolved)
- **Resolution**: ABSTRACTION + SEPARATION — Accept layer-level as the current abstraction level. Identify per-head steering as the next granularity level. In parallel, add head-level analysis to gain finer understanding.

### D7: Linear α vs Non-linear Effects (temporal_misalignment — relational)
- **Concepts**: A5 (α·v additive steering) × Observed (non-linear accuracy-α relationship)
- **Conflict**: The steering formula assumes a linear relationship: h' = h + α·v, and accuracy = f(h'). But the relationship between α and accuracy is non-monotonic (there's a goldilocks zone).
- **Severity**: OPERATIONAL (resolvable)
- **Resolution**: SYNTHESIS — Model the α-accuracy relationship as a quadratic or Gaussian function: acc(α) = acc_0 + A·exp(-(α - α_opt)²/σ²). Find α_opt per layer via Bayesian optimization or golden-section search.

### D8: Reading Head Distribution Shift (operational_incompatibility — potential)
- **Concepts**: A12 (reading head, r=0.86 on unsteered data) × C3-4 (steered generation)
- **Conflict**: The reading head was trained on Perceiver latents from unsteered forward passes. Under steering, hidden states change, so Perceiver latents differ. The reading head's accuracy degrades.
- **Severity**: OPERATIONAL (resolvable)
- **Resolution**: SEPARATION — Train a new reading head on steered-generation data. Or use a different confidence signal (logit entropy, attention entropy) that doesn't depend on the Perceiver.

### D9: First-Step Skip (goal_conflict — relational)
- **Concepts**: Code pattern `if not first_step: ... steer` × Leverage principle (first token has max influence)
- **Conflict**: All steering scripts skip steering at the first generation step. But the first token determines the entire reasoning trajectory — it's the highest-leverage point for intervention.
- **Severity**: OPERATIONAL (resolvable)
- **Resolution**: SUBSTITUTION — Remove the first_step skip. For the first token, compute velocity from the model's prompt hidden states (which are available first forward pass). The prompt trajectories have lower R² (0.62) but this is still meaningful signal.

### D10: GSM8K Statistical Significance (resource_conflict — structural)
- **Concepts**: A10 (GSM8K accuracy) × Observed (N=100-200 problems in published results)
- **Conflict**: The per-layer sweep results (L8: +20pp) used 100 problems. At 45% baseline, the 95% confidence interval for 100 problems is ±9.8pp. The +20pp result is outside this, but +4pp (SVAMP) is inside it.
- **Severity**: OPERATIONAL (resolvable)
- **Resolution**: REORDERING — Prioritize re-running L8, L2, L9 sweeps with N=500+ problems for statistical confidence. The data already exists (500 problems collected).

## Assumption Violations (from Phase 0 VOID)

| Assumption | Violated By | Evidence |
|------------|-------------|----------|
| IA3: Consistent direction | Math-1.5B contrastive failure | v_c - v_i may not be consistent across generation |
| IA5: Monotonic contrastive | Untested | Needs verification |
| IA7: Quantization preserves structure | Untested | Needs comparison of 4-bit vs full precision steering |
| IA2: Per-layer is right granularity | Head-level effects not ruled out | D6 unresolved |
| A9: Reading head gating practical | Distribution shift (D8) | Needs steered-data training |

## Summary

| Metric | Count |
|--------|-------|
| Total Disparities | 10 |
| Resolved | 3 (D2, D7, D9) |
| Bounded (irreconcilable) | 2 (D3 capability threshold, D5 contrastive cancellation) |
| Critical Unresolved | 3 (D4 Math-1.5B anomaly, D6 head-level effects, D1 high R² ≠ good steering) |
| Key Assumption Violations | 5 (IA3, IA5, IA7, IA2, A9) |
