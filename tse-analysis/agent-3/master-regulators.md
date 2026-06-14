# Phase 3: Master-Regulator Identification

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Methodology

Master regulators are identified by **Influence Centrality × Junction Leverage** — nodes and junctions whose modulation would propagate most broadly and deeply through the system.

### Scoring

| Score | Meaning |
|-------|---------|
| 9-10 | System-transforming modulation |
| 7-8.9 | High-impact, changes multiple downstream outcomes |
| 5-6.9 | Meaningful but bounded impact |
| 3-4.9 | Localized effect |
| 1-2.9 | Minimal leverage |

---

## Master Regulator #1: Contrastive Signal (A9 / J10)

**Influence Centrality**: 9/10  
**Junction Leverage**: 9/10  
**Combined Score**: 9.0

**Why**: The contrastive signal is the single concept with the highest potential to transform the system from descriptive (predicts what the model WILL do) to normative (predicts what the model SHOULD do). If it works, it addresses the fundamental flaw of standard TT. If it doesn't work, the entire framework hits a ceiling.

**Modulation Strategies**:
- **Existing**: Contrastive TT pair trained (v_correct, v_incorrect), evaluation pending
- **Proposed 1**: Combine standard + contrastive: v_standard + β·(v_correct − v_incorrect)
- **Proposed 2**: Multi-head contrastive ensemble — bag N bootstrapped contrastive pairs
- **Proposed 3**: Contrastive signal as a regularizer during standard TT training

**Expected Impact**: HIGH — could move from +20pp to +30pp+ on GSM8K, or reveal the theoretical ceiling

**Risk**: HIGH — contrastive signal may learn fluency/style differences rather than reasoning quality; evaluation may show no improvement over standard TT

---

## Master Regulator #2: Per-Layer Selectivity Mechanism (C2-3 / J5)

**Influence Centrality**: 9/10  
**Junction Leverage**: 8/10  
**Combined Score**: 8.5

**Why**: The per-layer sweep revealed the fundamental structure (trim tabs vs death layers) that governs the entire steering approach. Without per-layer selectivity, steering is net negative. This is the boundary condition that all other mechanisms must respect.

**Modulation Strategies**:
- **Existing**: Manual per-layer sweep with α=0.1 on 100 problems
- **Proposed 1**: **Asymmetric α** — different α per layer (positive for trim-tabs, negative for death layers)
- **Proposed 2**: **Layer selection via learned policy** — RL agent that selects which layer(s) to steer at each generation step
- **Proposed 3**: **Adaptive layer gating** — learned binary mask per layer that determines whether steering is applied
- **Proposed 4**: **Per-layer α as a function of token position** — front-load steering on early layers during reasoning, back-load during answer generation

**Expected Impact**: VERY HIGH — asymmetric α alone could double the effective steering improvement

**Risk**: MEDIUM — per-layer optimization increases search space exponentially; combinatorial (layer, α, token-position) optimization may be computationally expensive

---

## Master Regulator #3: Capability Threshold (A10 / J4)

**Influence Centrality**: 7/10  
**Junction Leverage**: 9/10  
**Combined Score**: 8.0

**Why**: The capability threshold is the most important constraint on system applicability. If it's a fundamental property (steering cannot create capability), the system is limited to capable models. If it can be bypassed, the system could help any model.

**Modulation Strategies**:
- **Existing**: Filter models by baseline accuracy >~40%
- **Proposed 1**: **Find steering surfaces for small models** — activation steering, representation engineering, or supervised fine-tuning as alternatives to KV-cache steering
- **Proposed 2**: **Two-stage approach** — fine-tune small model to ~40% baseline, then apply steering
- **Proposed 3**: **Larger α for small models** — the hidden state manifold may be more compressed in small models, requiring proportionally larger steering vectors
- **Proposed 4**: **Head-level steering on small models** — even if layer-level KV-cache doesn't work, individual attention heads might be steerable

**Expected Impact**: HIGH — bypassing the threshold would dramatically expand the addressable model range

**Risk**: VERY HIGH — if the threshold is fundamental (information-theoretic or manifold-geometric), attempts to bypass it will fail, wasting resources

---

## Master Regulator #4: Steering Coefficient α (A5 / J2)

**Influence Centrality**: 8/10  
**Junction Leverage**: 7/10  
**Combined Score**: 7.5

**Why**: α is the primary control parameter for steering strength. Currently a single scalar per layer, it could be a vector or learned function. Modulating α changes every aspect of steering: effectiveness, token divergence, side effects.

**Modulation Strategies**:
- **Existing**: Manual α=0.1 for all experiments
- **Proposed 1**: **Per-token-position α** — decay α over generation steps (strong at beginning, weak at end)
- **Proposed 2**: **Per-dimension α vector** — steer some hidden state dimensions more than others
- **Proposed 3**: **Adaptive α via RL** — learn α as a function of (layer, token_position, hidden_state_norm, PPL)
- **Proposed 4**: **α as a learned function of layer depth** — deeper layers get smaller α (assuming less perturbation tolerance)

**Expected Impact**: MEDIUM-HIGH — α optimization is likely low-hanging fruit

**Risk**: LOW-MEDIUM — α space is well-behaved (monotonic effects expected); risk is mainly computational cost of search

---

## Master Regulator #5: Trim-Tab Layer Identification (C3-3 / J11)

**Influence Centrality**: 6/10  
**Junction Leverage**: 8/10  
**Combined Score**: 7.0

**Why**: The ability to rapidly identify trim-tab and death layers for any model is the prerequisite for practical application. Currently requires 100+ forward passes for a full sweep.

**Modulation Strategies**:
- **Existing**: Full per-layer sweep (12-40 forward passes depending on model depth)
- **Proposed 1**: **Zero-shot trim-tab prediction from model weights** — use weight statistics (spectral norm, Frobenius norm, attention head entropy) to predict trim-tab layers without evaluation
- **Proposed 2**: **Intervention-internal probes** — measure the KL divergence of hidden state distributions when a layer is perturbed; trim-tab layers have low KL at high accuracy perturbation
- **Proposed 3**: **TT's own predictions as trim-tab indicator** — layers where TT's velocity norm is high and prediction direction is consistent across inputs may be trim-tabs
- **Proposed 4**: **Fine-grained sweep with early stopping** — evaluate layers in order of predicted importance, stop when diminishing returns detected

**Expected Impact**: HIGH — reduces the cost of applying steering to new models from O(days) to O(minutes)

**Risk**: MEDIUM — zero-shot prediction methods may have high error rates; false positives (predicting trim-tab for a death layer) cause catastrophic performance drops

---

## Ranked Master Regulators

| Rank | Regulator | Score | Control Type | Current State | Quick Win? |
|------|-----------|-------|-------------|---------------|------------|
| 1 | Contrastive Signal | 9.0 | Training/Architecture | Evaluation pending | ✅ (evaluation is the action) |
| 2 | Per-Layer Selectivity | 8.5 | Architecture/Policy | Manual sweeps only | ✅ (asymmetric α is easy to test) |
| 3 | Capability Threshold | 8.0 | Constraint | Accepted as fundamental | ❌ (requires new experiments) |
| 4 | Steering Coefficient α | 7.5 | Hyperparameter | Fixed at 0.1 | ✅ (α sweep is cheap) |
| 5 | Trim-Tab Identification | 7.0 | Prediction/Estimation | Full sweep required | ✅ (zero-shot prediction methods) |
| 6 | TT Architecture | 6.5 | Model Design | DeltaNet/MLP | ❌ (requires architecture changes) |
| 7 | Cross-Model Transfer | 6.0 | Projection | Works for SmolLM2→7B | ⚠️ (validated on one pair) |
| 8 | Prompt Template | 5.5 | Preprocessing | Fixed format | ✅ (template search is cheap) |
| 9 | Async Data Loading | 4.5 | Infrastructure | Working | ❌ (already mature) |
| 10 | GPU Caching | 4.0 | Infrastructure | Working | ❌ (already mature) |
