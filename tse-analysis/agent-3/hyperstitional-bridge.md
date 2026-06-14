# Phase 10: Hyperstitional Bridge

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## H-1: Steering Effect Null Hypothesis (Structural)

**Type**: Structural  
**Statement**: "The accuracy improvement from L8 TT-predicted steering (+20pp on GSM8K) is indistinguishable from the improvement produced by random perturbation vectors of the same norm distribution."

**Falsification Criteria**:
- **Refutes hypothesis** (TT > random): TT-driven accuracy exceeds random-vector accuracy by >5pp at p<0.05 on 100 GSM8K problems
- **Confirms hypothesis** (TT ≈ random): Accuracy difference ≤5pp
- **Minimum experiment**: 100 GSM8K problems × 2 conditions (TT, random) × 3 random seeds

**Risk if false (we act on "random is as good as TT")**: We abandon velocity prediction and miss a genuine steering mechanism.

**Value if true (we confirm TT > random)**: Establishes that TT predictions capture causally meaningful structure, validating the entire approach.

**Research Call**: ML interpretability community — "Do predicted hidden-state velocities capture causal structure, or are they random perturbations with the same statistical properties?"

---

## H-2: Death Layer Direction Inversion (Relational)

**Type**: Relational  
**Statement**: "L9's death-layer effect is direction-dependent: applying NEGATIVE α (steering AWAY from predicted velocity) will produce accuracy improvement, while positive α produces accuracy degradation."

**Falsification Criteria**:
- **Refutes hypothesis**: Negative α at L9 produces accuracy > baseline (45%) + 5pp for at least one α ∈ {-0.3, -0.2, -0.1}
- **Confirms hypothesis**: All negative α produce accuracy ≤ baseline + measurement noise
- **Minimum experiment**: 7 α values × 100 problems on L9 Qwen2.5-7B

**Risk if false**: We apply negative α to death layers and amplify their harmful effect, wasting evaluation time.

**Value if true**: Transforms death layers from liabilities to assets. Could double the number of steerable layers.

**Causal chain if true**: "Velocity prediction at L9 points in the WRONG direction (toward error). Negative α is a sign flip that corrects the direction → L9 becomes a trim-tab."

**Research Call**: Mechanistic interpretability — "What computation does layer 9 perform that its velocity predictions are consistently error-pointing?"

---

## H-3: Contrastive Signal as Style, Not Reasoning (Potential)

**Type**: Potential  
**Statement**: "The contrastive signal (v_correct − v_incorrect) is dominated by differences in output style (length, token distribution, confidence) rather than differences in reasoning quality, and steering with this signal will primarily change style, not accuracy."

**Falsification Criteria**:
- **Refutes hypothesis** (contrastive is about reasoning): Contrastive-directed steering improves accuracy on GSM8K beyond standard TT with β>0 in dual-mode steering
- **Confirms hypothesis** (contrastive is style): Contrastive steering changes token distribution statistics (length, perplexity, entropy) without improving accuracy
- **Minimum experiment**: Style metrics correlation analysis (H0-2 from Phase 8) + dual-mode steering with varying β

**Risk if false**: We invest in contrastive TT improvements (multi-head ensemble, hard negative mining) that address the wrong signal.

**Value if true**: We focus on alternative normative signals (adversarial TT, preference optimization, ranking-based steering) rather than contrastive approaches.

**Research Call**: "Do correct and incorrect language model trajectories differ in reasoning content or only in surface form? A contrastive analysis."

---

## H-4: Capability Threshold as α Artifact (Structural)

**Type**: Structural  
**Statement**: "Small models (SmolLM2, Qwen2.5-0.5B) CAN be steered toward improved accuracy, but require larger α values (α > 1.0) because their hidden state manifold is more compressed."

**Falsification Criteria**:
- **Refutes hypothesis** (threshold is an artifact): Any α in {0.5, 1.0, 2.0, 5.0} improves accuracy above baseline on SmolLM2 or Qwen2.5-0.5B
- **Confirms hypothesis** (threshold is fundamental): All α values ≤ baseline for both small models
- **Minimum experiment**: 2 models × 4 α values × 100 problems

**Risk if false (threshold is NOT an artifact)**: We stop investigating small model steering, missing a major expansion opportunity.

**Value if true (threshold IS an artifact)**: Dramatically expands the range of steerable models → steering can help small/cheap models improve reasoning.

---

## H-5: Multi-Layer Synergy (Relational)

**Type**: Relational  
**Statement**: "Steering multiple layers simultaneously with optimal signed α produces greater accuracy improvement than the best single layer, due to interaction effects that are not predictable from per-layer sweeps."

**Falsification Criteria**:
- **Refutes hypothesis**: max(L8_alone, L2_alone, neg_L9_alone) + 5pp < composite (L8 + L2 + neg_L9)
- **Confirms hypothesis**: Composite ≤ best single layer + 5pp
- **Minimum experiment**: 3 single layers + all 7 combinations of (L2, L8, L9) × 100 problems

**Risk if false**: We assume multi-layer is always beneficial and create unnecessary complexity.

**Value if true**: Steering improves with model depth utilization ⇒ scaling law for steering (more steered layers → better results).

**Causal chain**: "Trim-tab layers operate on DIFFERENT aspects of reasoning (L2: problem comprehension, L8: solution planning, neg_L9: answer generation). Combined steering covers all phases."

---

## H-6: Token-Position Specificity (Potential)

**Type**: Potential  
**Statement**: "The steering effect is concentrated at specific token positions (reasoning transitions, arithmetic operations), and position-gated steering can match uniform steering with <50% of the token divergence."

**Falsification Criteria**:
- **Refutes hypothesis**: Position-gated steering (α_high at critical positions, α=0 elsewhere) achieves ≥90% of uniform-L8 accuracy with ≤50% token divergence
- **Confirms hypothesis**: Position-gated accuracy < 90% of uniform with ≤50% divergence
- **Minimum experiment**: Analyze per-token steering effect → identify critical positions → test gated steering

**Risk if false**: We assume steering must affect all tokens uniformly.

**Value if true**: Major step toward practical deployment — reduced divergence means fewer side effects.

---

## H-7: Cross-Model Universality (Structural)

**Type**: Structural  
**Statement**: "The trim-tab/death-layer pattern is universal across standard-MHA language models: layer position relative to model depth (early, middle, late decile) predicts trim-tab vs death classification regardless of model size or training data."

**Falsification Criteria**:
- **Refutes hypothesis**: For ≥3 model families (Qwen2.5, LLaMA-3, SmolLM2), the trim-tab layer in the middle-to-late decile (75-85% depth) is consistently among the best 3 layers
- **Confirms hypothesis**: Trim-tab position is model-specific and not predicted by relative depth
- **Minimum experiment**: Full per-layer sweep on 3 model families × 50 problems each

**Risk if false**: We assume universality and waste cross-model research effort.

**Value if true**: Trim-tab discovery becomes trivial — just compute model depth and steer the 75-85% decile layer.

---

## H-8: Self-Bootstrapping Convergence (Potential)

**Type**: Potential  
**Statement**: "An iteratively trained TT (trained on steered-model trajectories) will CONVERGE to a fixed point within 3 iterations, producing a TT that is more accurate on the steered model than the original TT on the unsteered model."

**Falsification Criteria**:
- **Refutes hypothesis** (converges): After 3 iterations, TT accuracy on steered validation data is higher than original TT accuracy on unsteered validation, and iteration-to-change < 5%
- **Confirms hypothesis** (diverges or stagnates): TT accuracy doesn't improve, or oscillates between iterations
- **Minimum experiment**: 3-5 iterations of: steer → collect trajectories → train new TT → compare predictions on steered data

**Risk if false**: We implement complex bootstrapping infrastructure for no gain.

**Value if true**: Self-improving steering system that gets better with use — the holy grail of adaptive control.

---

## Hypothesis Summary

| ID | Statement | Phase | Proof of Concept Cost | Impact if True |
|----|-----------|-------|----------------------|----------------|
| H-1 | Steering is random perturbation | A1 | 1 hour | Validates/invalidates entire approach |
| H-2 | Negative α on death layers works | A2 | 45 min | Doubles steerable layers |
| H-3 | Contrastive is style, not reasoning | A3, B2 | 3 hours | Redirects research direction |
| H-4 | Capability threshold is α artifact | B3 | 2 hours | Expands addressable models |
| H-5 | Multi-layer > single-layer | C1 | 1 day | Scaling law for steering |
| H-6 | Position-gated steering is efficient | C2 | 2 days | Practical deployment feasible |
| H-7 | Trim-tab pattern is universal | C3 | 1 day | Trivial trim-tab discovery |
| H-8 | Bootstrapping converges | D1 | 3-5 days | Self-improving steering |
