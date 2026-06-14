# Phase 3: Master-Regulator Identification

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Master Regulator Ranking

Ranked by **Influence Centrality × Junction Leverage**, where Influence Centrality measures how many nodes are affected by modifying this node, and Junction Leverage measures how many critical junctions include this node.

---

### #1: Layer Selectivity (C2_SELECT)
**Influence Centrality**: 9/10 — Affects A_KV (which layers get modified), A_B (accuracy outcome), A_P (pattern discovery), A_D (token divergence), and every downstream steering evaluation
**Junction Leverage**: 10/10 — Involved in J5, J8, J11 (all critical junctions)
**Classification**: **STRUCTURAL** — The choice of which layers to steer is the single most important decision in the pipeline

**Modulation Strategies**:
- **Existing**: Manual per-layer sweep (brute force, N=28 layers × M values of α)
- **Proposed 1**: Automated layer selection via learned gating network (gradient-based optimization over {0,1}^28)
- **Proposed 2**: Information-theoretic layer ranking (mutual information between layer output and correctness)
- **Proposed 3**: Evolutionary layer search (CMA-ES over layer subsets)

**Expected Impact**: **HIGH** — Correct layer selection is the difference between +20pp and -23pp
**Risk**: Without true layer independence, selected layers may interact destructively, requiring multi-layer combinatorial optimization

---

### #2: Steering Alpha (A_α)
**Influence Centrality**: 8/10 — Affects A_B (accuracy), A_D (divergence), A_KV (modification magnitude), C2_STEER (steering behavior)
**Junction Leverage**: 9/10 — Involved in J4, J13 (accuracy impact), J12 (divergence)
**Classification**: **MODULATORY** — α is the gain knob on the entire steering system

**Modulation Strategies**:
- **Existing**: Manual α selection (single scalar, typically 0.1)
- **Proposed 1**: Per-layer α learned via gradient descent on validation accuracy
- **Proposed 2**: Per-token α predicted by a meta-network conditioned on hidden state and position
- **Proposed 3**: Online α adaptation via reinforcement learning (policy that adjusts α based on PPL change)
- **Proposed 4**: Asymmetric α (different α for correct vs incorrect steering directions)

**Expected Impact**: **HIGH** — α determines the operating point on the accuracy-α curve; optimal α may be layer-specific
**Risk**: α too high → token divergence destroys coherence; α too low → no effect. The α-accuracy function may be non-convex and sharp.

---

### #3: TrajectoryTransformer Quality (A_TT)
**Influence Centrality**: 8/10 — Affects every downstream component: C2_VEL (velocity prediction), C2_STEER (steering direction), C3_TRAIN (training quality), A_C (contrastive direction), C4_PIPELINE (overall system)
**Junction Leverage**: 9/10 — Involved in J2, J3, J10, J19 (all critical)
**Classification**: **FOUNDATIONAL** — The TT is the source of all steering direction information

**Modulation Strategies**:
- **Existing**: MLP architecture, trained with MSE loss on velocity prediction
- **Proposed 1**: Transformer-based TT that conditions on trajectory history (not just current h_t)
- **Proposed 2**: Multi-task TT that jointly predicts velocity + steering advantage + confidence
- **Proposed 3**: Ensemble of TTs with different random seeds; steering = mean prediction, uncertainty = variance
- **Proposed 4**: TT pretrained on multiple models (foundation TT) then fine-tuned per-model

**Expected Impact**: **HIGH** — If TT captures causal dynamics (not just statistical correlations), steering quality improves proportionally
**Risk**: The fundamental limitation is that TT is trained on natural trajectories and used on steered trajectories — out-of-distribution failure is guaranteed at some α threshold. Also, TT's high R² may obscure the fact that it's learning confounded features.

---

### #4: Contrastive Direction (A_C)
**Influence Centrality**: 7/10 — Affects C2_CONTRAST (contrastive training), C3_TRAIN (training mode), and potentially C2_STEER (steering direction)
**Junction Leverage**: 7/10 — Involved in J9, J19 (moderate leverage)
**Classification**: **DIFFERENTIATING** — The contrastive signal is the proposed solution to the descriptive/normative problem

**Modulation Strategies**:
- **Existing**: Train two TTs separately (correct trajectories, incorrect trajectories); subtract predictions
- **Proposed 1**: Train a single contrastive TT with modified loss: L = MSE(v̂, v_correct) − MSE(v̂, v_incorrect)
- **Proposed 2**: Learn a contrastive direction field via Siamese network with triplet loss
- **Proposed 3**: Bootstrap ensemble — train N TTs on different trajectory subsets; contrastive direction = mean(TT_correct) − mean(TT_incorrect)
- **Proposed 4**: Online contrastive direction that updates as the model's trajectory distribution shifts during steering

**Expected Impact**: **MEDIUM-HIGH** — If correct and incorrect trajectories are separable, contrastive direction could provide the "normative" steering signal that standard TT lacks
**Risk**: On Math-1.5B, contrastive TT showed no trim tabs — suggesting the manifolds may not always be separable. Even if separable, the direction may still not be the correct one for steering (the vector between manifolds may point in a semantically unhelpful direction).

---

### #5: Model Capability Threshold (A_M / C2_CAP)
**Influence Centrality**: 6/10 — Affects model selection (which models to test), experimental design (which datasets), and theoretical framework (generalization claims)
**Junction Leverage**: 8/10 — Involved in J6, J7 (critical gating junction)
**Classification**: **CONDITIONAL** — This is a gate, not a knob: if the model is below threshold, steering is off

**Modulation Strategies**:
- **Existing**: Passive observation (we can't change the model, only select it)
- **Proposed 1**: Identify the specific sub-circuit that enables steerability; transfer it to smaller models
- **Proposed 2**: Train a "steerability probe" that predicts whether a model will respond to steering
- **Proposed 3**: Use steering on individual reasoning steps instead of full generations (bypass coherence requirements)
- **Proposed 4**: Use contrastive steering specifically on the reasoning trace (chain-of-thought) rather than token-level

**Expected Impact**: **LOW-MEDIUM** — Changing the model's baseline capability is the goal, not the input. However, understanding the threshold mechanistically is high-value.
**Risk**: The threshold may be an irreducible property of model scale, not a structural feature we can modify. The project's assumption that steering is "amplification, not creation" may be a fundamental limit.

---

## Master Regulator Interaction Map

```
                  ┌─────────────────────┐
                  │ #3: TT Quality (A_TT)│◄──────── Source of direction
                  └──────┬──────────────┘
                         │ J2, J3
                         ▼
                  ┌─────────────────────┐
           ┌─────►│ #2: Alpha (A_α)     │◄──────── Gain control
           │      └──────┬──────────────┘
           │             │ J4, J13
           │             ▼
┌──────────────────┐    ┌─────────────────────┐
│ #4: Contrastive  │    │ #1: Layer Select    │◄──────── Spatial selection
│ Direction (A_C)  │───►│ (C2_SELECT)         │
└──────────────────┘    └──────┬──────────────┘
                               │ J5, J8
                               ▼
                        ┌─────────────────────┐
                        │ Steering Effect      │
                        │ (Accuracy Change)    │
                        └─────────────────────┘
                               ▲
                               │ J7 (gate)
                        ┌─────────────────────┐
                        │ #5: Capability       │◄──────── Prerequisite
                        │ Threshold (C2_CAP)   │
                        └─────────────────────┘
```

The interaction is hierarchical: **Capability → TT → Contrastive → Layer → Alpha → Effect**. Each regulator operates at a different point in the pipeline, and the output of one constrains the input of the next.

---

## Critical Observation

The top-3 regulators (Layer Selectivity, Alpha, TT Quality) form a **triple constraint**: no amount of TT quality can compensate for wrong layer selection, no amount of layer tuning can compensate for wrong α, and no α tuning on the right layer can compensate for a TT that learned spurious correlations. These three must be optimized jointly, not sequentially.

This joint optimization is currently done manually (independent sweeps), which is the project's biggest missed opportunity. A **joint optimization loop** (e.g., Bayesian optimization over {layer, α, TT_architecture}) would likely discover better steering configurations faster.
