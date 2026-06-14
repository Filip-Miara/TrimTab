# Phase 3: Master-Regulator Identification

Ranked by Influence Centrality × Junction Leverage (composite score 0-100).

---

## #1: Per-Layer α Allocation (A05 × A17)

**Composite Score**: 94/100

**Influence Centrality**: 9/10 — Changing α per layer affects EVERY steering outcome. It directly modulates J05 (α→KV modification), which feeds into J06 (trim-tab→accuracy) and J07 (death layer→accuracy). This is the single knob controlling the entire steering profile.

**Junction Leverage**: 9/10 — The α→KV modification relationship (J05) is the only place where the system designer can intervene without changing infrastructure. It's a pure gain parameter.

**Modulation Strategies**:
1. **Current**: Uniform α=0.1 across all layers — simple but ignores differences
2. **Per-layer manual**: Set α=0 for death layers (L9, L7, L15+), α=0.1 for trim-tabs (L8, L2), α=0.05 for neutral
3. **Per-layer learned**: Train an α-predictor network: α(l, h[l], context) = σ(MLP(h[l], l_embed))
4. **Per-layer sweep grid**: Exhaustive grid search over [0.0, 0.05, 0.1, 0.2, 0.5, 1.0] for each layer

**Expected Impact**: HIGH — Proper α allocation could: (1) eliminate death layer collapse, (2) potentially double trim-tab gains through stronger α on identified trim-tabs, (3) enable multi-layer combinations.

**Risk**: Over-optimizing α on a small evaluation set (100 problems) could lead to overfitting to those specific problems.

---

## #2: Contrastive-Normative Signal Learning (C05)

**Composite Score**: 88/100

**Influence Centrality**: 8/10 — The contrastive signal (v_c - v_i) determines the DIRECTION of steering, which is more important than the magnitude. If the direction is wrong, no α tuning helps. If the direction is correct, even modest α gains.

**Junction Leverage**: 9/10 — The transition from standard TT (descriptive: "where it's going") to contrastive TT (normative: "where it should go") rewrites J02 (TT→velocity prediction) from a copying mechanism to a correcting mechanism.

**Modulation Strategies**:
1. **Current**: Train separate TTs on correct and incorrect trajectories, subtract outputs
2. **Multi-head contrastive ensemble**: Train N bootstrapped contrastive pairs, average outputs (bagging)
3. **Direct normative training**: Train a single TT with a modified loss: L = ||v_pred - v_target||² + β·||v_pred - v_harmful||² (push away from bad)
4. **RL-based**: Use policy gradient with accuracy reward to optimize steering direction directly

**Expected Impact**: HIGH — If contrastive steering works (+5-10pp over standard), it would validate the entire normative-steering paradigm. If it fails, it forces a fundamental rethinking.

**Risk**: Contrastive signal could amplify noise (v_c - v_i ≈ noise if manifolds aren't separable). Currently unvalidated.

---

## #3: Capability Threshold Diagnostic (A11 → C08)

**Composite Score**: 82/100

**Influence Centrality**: 9/10 — The capability threshold determines the ENTIRE applicability domain of steering. It gates whether any other regulator matters.

**Junction Leverage**: 7/10 — J09 (threshold→trim-tab) is a binary gate: below threshold, trim-tabs don't exist. Above it, they do. Binary decisions have high leverage but low granularity.

**Modulation Strategies**:
1. **Current**: Implicit diagnostic (run steering, observe 0% improvement → conclude below threshold)
2. **Proactive diagnostic**: Measure hidden state manifold separability (correct vs incorrect centroid distance) as a proxy metric — no steering needed
3. **Capability bootstrapping**: Fine-tune model for 100 steps on target task, then re-evaluate threshold. If fine-tuning creates steerable structure, it confirms the bottleneck
4. **Cross-model transfer**: Use a capable model's TT to steer a less capable model (already partially tested: SmolLM2→7B)

**Expected Impact**: MEDIUM-HIGH — A reliable diagnostic prevents wasted computation and provides scientific insight into what steering actually requires.

**Risk**: The diagnostic itself might be wrong (says below threshold but steering could work with different parameters), leading to premature abandonment.

---

## #4: Death Layer Identification & Exclusion (A09 → J07)

**Composite Score**: 76/100

**Influence Centrality**: 7/10 — Death layers affect accuracy with high magnitude (-23pp each) but their effect is limited to specific layers (L9, L7, L15+).

**Junction Leverage**: 7/10 — J07 (death→accuracy) is a strong negative relationship. Breaking this junction (by excluding those layers) immediately improves the system.

**Modulation Strategies**:
1. **Current**: Manual identification after per-layer sweep
2. **Automatic classifier**: Train a probe on hidden states to predict whether a layer is trim-tab or death without running a full sweep
3. **Adaptive exclusion**: Monitor steering efficacy online — if accuracy drops below a rolling baseline, exclude the most recently steered layer
4. **Death layer repurposing**: Apply NEGATIVE α to death layers (α = -0.1) — this inverts the steering direction, potentially converting death→trim-tab

**Expected Impact**: HIGH — Simply excluding death layers from multi-layer steering would eliminate the all-layers-failure mode.

**Risk**: The death layer set may be task-dependent (different death layers for different datasets).

---

## #5: Architecture-Aware Steering Surface Selection (A19 → C07)

**Composite Score**: 68/100

**Influence Centrality**: 6/10 — Architecture determines steering surface availability, but the current project only has one relevant comparison (Qwen2.5 MHA vs Qwen3.5 hybrid).

**Junction Leverage**: 7/10 — J08 (architecture→surface) is a hard constraint. Changing the architecture changes which steering mechanisms are possible.

**Modulation Strategies**:
1. **Current**: Select MHA models, avoid hybrid attention
2. **GDN-specific steering**: For GatedDeltaNet layers, steer the recurrent state instead of KV cache (the original recurrent_steering attempt was preliminary)
3. **Hybrid-aware pipeline**: Tag each layer as MHA or GDN, apply different steering per type
4. **Universal steering interface**: Project hidden state modifications to affect output without going through architecture-specific projections

**Expected Impact**: MEDIUM — Enables steering on a wider range of models (including future hybrid architectures).

**Risk**: Architecture-specific engineering is high-effort; might be obviated if hybrid architectures become rare.
