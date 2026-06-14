# Phase 3: Master-Regulator Identification

**Subject**: RankAdaptation — Velocity-based latent steering
**Date**: 2026-06-14

---

## Methodology

Master regulators are identified by their **Influence Centrality** (how many nodes are affected if this node changes) × **Junction Leverage** (how much the system responds to modulation of this relationship).

### Scoring

| Score | Meaning |
|-------|---------|
| 1-3 | Local effect, minor leverage |
| 4-6 | Cascading effect, moderate leverage |
| 7-9 | System-wide effect, high leverage |
| 10 | Transformative effect, maximum leverage |

---

## Master Regulator #1: Per-Layer Selectivity (A4/J4)

**Type**: Relational (Junction J4: Selectivity → Trim-Tab Discovery)
**Influence Centrality**: 9/10
**Junction Leverage**: 9/10
**Composite Score**: 81 (HIGHEST)

| Why | Detail |
|-----|--------|
| Scope | Controls access to ALL trim-tab and death-layer effects |
| Magnitude | Decision to select vs not-select changes accuracy by ±20-45pp |
| Downstream effects | Determines which layers get steering → affects all steering experiments → affects conclusions about steering viability |
| Number of affected nodes | A3 (Steering), A5 (Trim-Tab), A6 (Death Layer), A7 (Threshold), A10 (α), A11 (Divergence), C4 (Layer Analysis), C2 (Steering Mechanism), all composites |

**Modulation Strategies**:
- **Current**: Manual per-layer sweep with fixed α=0.1
- **Proposed 1**: Bayesian optimization over (layer, α) space
- **Proposed 2**: Learned routing policy that selects layers adaptively per token
- **Proposed 3**: Genetic algorithm for multi-layer combination optimization
- **Proposed 4**: Attention-based saliency map to predict trim-tab/death layers without brute-force sweep

**Expected Impact**: HIGH — finding the right layer(s) is the strongest known lever
**Risk**: Over-optimizing on GSM8K may not generalize to other tasks

---

## Master Regulator #2: Steering Strength α (A10/J3)

**Type**: Modulatory (Junction J3: α → Steering)
**Influence Centrality**: 7/10
**Junction Leverage**: 8/10
**Composite Score**: 56

| Why | Detail |
|-----|--------|
| Scope | Continuous parameter that controls intensity of ALL steering |
| Magnitude | At α=0, no effect; at α→∞, complete destruction; optimal somewhere between |
| Current state | α=0.1 used everywhere; no evidence this is optimal |
| Number of affected nodes | A3 (Steering), A5 (Trim-Tab via magnitude), A6 (Death Layer via magnitude), A11 (Divergence), C2 (Steering Mechanism) |

**Modulation Strategies**:
- **Current**: Fixed global α=0.1
- **Proposed 1**: Per-layer α sweep for top-5 trim-tab and death layers to find optimal operating point
- **Proposed 2**: Token-adaptive α based on per-token confidence or uncertainty
- **Proposed 3**: Clipped adaptive α that grows during answer generation (early tokens: small α, late tokens: larger α)
- **Proposed 4**: Learned α via policy gradient (RL-based α optimization)

**Expected Impact**: HIGH — the optimal α could unlock currently hidden trim-tabs on "marginal" layers
**Risk**: α that is too aggressive could destroy the model; need safety bounds

---

## Master Regulator #3: Contrastive Direction Signal (A8/J10)

**Type**: Relational (Junction J10: Contrastive TT → Predicted Trim-Tab)
**Influence Centrality**: 8/10
**Junction Leverage**: 7/10
**Composite Score**: 56

| Why | Detail |
|-----|--------|
| Scope | Converts descriptive TT into normative steering direction |
| Magnitude | If it works, it transforms the entire approach from "faithful error reproduction" to "corrective steering" |
| Current state | TTs trained, evaluation pending |
| Number of affected nodes | A2 (TT), A3 (Steering), A5 (Trim-Tab expected), A7 (Threshold), C5 (Contrastive System), all of L3-2 (Research Program) |

**Modulation Strategies**:
- **Current**: v_correct − v_incorrect from two separately trained TTs
- **Proposed 1**: Single Siamese network with contrastive loss (minimize distance between v_pred and v_correct, maximize distance from v_incorrect)
- **Proposed 2**: Multi-head contrastive ensemble (bagging bootstrapped pairs)
- **Proposed 3**: Triple loss (anchor = current state, positive = correct next state, negative = incorrect next state)
- **Proposed 4**: Adversarial contrastive where the TT learns to fool a discriminator that classifies correct/incorrect trajectories

**Expected Impact**: TRANSFORMATIVE if it works; would establish the "normative steering" paradigm
**Risk**: The correct/incorrect manifolds may not be separable (as Math-1.5B suggests)

---

## Master Regulator #4: Capability Threshold (A7/J6)

**Type**: Conditional (Junction J6: Threshold → Trim-Tab)
**Influence Centrality**: 6/10
**Junction Leverage**: 9/10
**Composite Score**: 54

| Why | Detail |
|-----|--------|
| Scope | Determines WHICH models are steerable |
| Magnitude | Binary gate: below threshold → all steering fails; above threshold → trim-tabs possible |
| Current state | Empirical boundary at ~40% GSM8K |
| Number of affected nodes | A5 (Trim-Tab), A6 (Death Layer), all steering mechanisms, all models below threshold |

**Modulation Strategies**:
- **Current**: Only use models above threshold
- **Proposed 1**: Fine-tune sub-threshold models on math data to raise baseline → then steer
- **Proposed 2**: Curriculum steering — start with easy problems, build capability, steer on harder ones
- **Proposed 3**: Discover sub-threshold steering that works by different mechanism (e.g., different α, different intervention surface)
- **Proposed 4**: Combine steering with few-shot prompting to bootstrap capability

**Expected Impact**: MEDIUM — relaxing the threshold would expand the steerable model set significantly
**Risk**: The threshold may be fundamental (models below threshold have no "correct hidden state" to steer toward)

---

## Master Regulator #5: Training Data Quality (A14/A15/A16 → A2)

**Type**: Structural (Composite C7: Training Infrastructure → TT quality)
**Influence Centrality**: 8/10
**Junction Leverage**: 5/10
**Composite Score**: 40

| Why | Detail |
|-----|--------|
| Scope | Controls TT quality which controls steering direction quality |
| Magnitude | Better data → better TT → better steering → potentially better trim-tabs |
| Current state | Standard generation data, balanced correct/incorrect for contrastive |
| Number of affected nodes | A2 (TT), A8 (Contrastive TT), A3 (Steering quality), A5/A6 discovery accuracy |

**Modulation Strategies**:
- **Current**: Raw generated trajectories, split by correctness
- **Proposed 1**: Importance-weighted trajectories (correct answers with long reasoning chains weighted higher)
- **Proposed 2**: Self-consistent trajectories (majority-vote correct answers)
- **Proposed 3**: Augmented trajectories (apply steering in loop: steer → generate → collect → re-train)
- **Proposed 4**: Hard-negative mining for contrastive (find trajectories where correct/incorrect are hardest to distinguish)

**Expected Impact**: MEDIUM-HIGH — data quality is a known lever in ML
**Risk**: Diminishing returns — R² is already 0.94; ceiling effect

---

## Summary: Master Regulator Ranking

| Rank | Regulator | Type | Score | Current State | Recommended Action |
|------|-----------|------|-------|---------------|-------------------|
| #1 | Per-Layer Selectivity (A4/J4) | Junction | 81 | Fixed manual sweep | Automate with Bayesian optimization |
| #2 | α Strength (A10/J3) | Junction | 56 | Fixed α=0.1 | Per-layer α sweep |
| #3 | Contrastive Signal (A8/J10) | Junction | 56 | TTs trained, pending eval | Complete evaluation now |
| #4 | Capability Threshold (A7/J6) | Conditional | 54 | Empirical ~40% boundary | Test with fine-tuned sub-threshold models |
| #5 | Training Data Quality (C7) | Composite | 40 | Raw generation data | Augmented/curated trajectory selection |
