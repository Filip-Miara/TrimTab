# Phase 1: Atomic Decomposition & Pyramid Construction

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Atoms (Level 1 — Indecomposable Concepts)

| ID | Atom | Description | Evidence Grounding | Pre-Seed (Ideal Form) |
|----|------|-------------|-------------------|-----------------------|
| A_H | Hidden State | Layer output vector h_t at position t | Standard transformer definition | Continuously differentiable manifold with interpretable directions |
| A_KV | KV Cache | Key/Value matrices for each layer's attention | Standard transformer definition | Controllable via differentiable steering with zero overhead |
| A_V | Velocity | h_{t+1} − h_t, 1-step hidden state delta | Project definition (R²=0.85-0.94) | Multi-scale velocity: Δt = {1, 2, 4, 8} tokens capturing hierarchical dynamics |
| A_TT | TrajectoryTransformer | Small MLP predicting v̂_t from h_t | Implemented, R²=0.85-0.94 | Hypernetwork producing layer-specific, input-adaptive steering vectors |
| A_α | Steering Alpha | Scalar multiplier for steering magnitude | Experimental parameter | Learnable per-token, per-layer, per-head α predicted by a meta-network |
| A_L | Layer Selectivity | Choosing which layers to steer | Key finding (L8 vs L9) | Learned gating mechanism that activates steering only at beneficial layers |
| A_M | Model Capability | Baseline GSM8K accuracy of the unsteered model | Empirical observation (4% to 73%) | Steering should bootstrap capability, not require it |
| A_C | Contrastive Direction | v_correct − v_incorrect | Proposed architecture | Multi-head contrastive ensemble with bootstrap-aggregated direction |
| A_P | Per-Layer Pattern | Trim-tab / death-layer classification | Empirical (L8: +20pp, L9: -23pp) | Dynamic pattern that adapts per task, determined by automated probe |
| A_D | Token Divergence | Measurement of how much steered output differs from baseline | 88% measured | Bounded divergence: max benefit at minimal token change |
| A_G | Generation Trajectory | Sequence of hidden states during autoregressive generation | Data collected | Compressed latent representation of the trajectory (not raw states) |
| A_B | Baseline Accuracy | Unsteered model performance on GSM8K | 73% for Qwen2.5-7B | Steering should achieve superhuman accuracy (no upper bound) |
| A_R | Residual Stream | The sum of layer outputs plus embedding | Transformer architectural primitive | Decomposable into independently steerable subspaces |
| A_S | Steering Surface | What the steering intervention actually modifies (K/V, logits, recurrences) | Architectural constraint | Universal steering surface that works across all architectures |

---

## Composites by Level

### Level 2 — Elementary Composites

| ID | Composite | Atoms | Junctions | Description |
|----|-----------|-------|-----------|-------------|
| C2_VEL | Velocity Dynamics | A_H, A_V, A_TT | compositional, causal | The core prediction pipeline: hidden state → velocity → TT prediction |
| C2_STEER | Steering Mechanism | A_KV, A_α, A_S | compositional, dependency | How steering is applied to the model |
| C2_EVAL | Evaluation Framework | A_B, A_D, A_P | comparative, temporal | How steering effects are measured |
| C2_CONTRAST | Contrastive Signal | A_H, A_C, A_TT | compositional, synergistic | The contrastive direction approach |
| C2_SELECT | Layer Selection | A_L, A_P | hierarchical, dependency | Which layers to steer and why |
| C2_CAP | Capability Threshold | A_M, A_B | causal, conditional | The relationship between baseline and steerability |

### Level 3 — Intermediate Composites

| ID | Composite | Subcomponents | Junctions | Description |
|----|-----------|--------------|-----------|-------------|
| C3_TRAIN | Training Pipeline | C2_VEL, C2_CONTRAST, trajectory data | temporal, dependency | End-to-end training of TT models |
| C3_STEER | Steering Pipeline | C2_STEER, C2_SELECT, generation loop | temporal, dependency | End-to-end steering during generation |
| C3_PATTERN | Layer Pattern Discovery | C2_SELECT, A_P, C2_EVAL | analytical, causal | The trim-tab/death-layer discovery process |
| C3_TRANSFER | Cross-Model Transfer | C2_VEL, C2_STEER, model projection | compositional, analogical | Applying TT from one model to another |

### Level 4 — System Composites

| ID | Composite | Subcomponents | Junctions | Description |
|----|-----------|--------------|-----------|-------------|
| C4_PIPELINE | Full Steering System | C3_TRAIN, C3_STEER, C3_PATTERN | dependency, temporal | The complete end-to-end system |
| C4_KNOWLEDGE | Theoretical Framework | C3_PATTERN, C3_TRANSFER, C2_CAP | abstraction, generalization | What we know about velocity steering |

### Level 5 — Peak Composite

| ID | Composite | Definition |
|----|-----------|------------|
| PEAK | Velocity-Based Latent Steering | The proposition that modifying KV-cache entries using predicted hidden state velocities can improve LM reasoning |

---

## Junctions

| ID | Source | Target | Type | Description | Leverage |
|----|--------|--------|------|-------------|----------|
| J1 | A_H | A_V | **causal** | Hidden state at t determines velocity to t+1 | High |
| J2 | A_V | A_TT | **compositional** | TT learns to predict V from H | Critical |
| J3 | A_TT | C2_STEER | **dependency** | TT output is input to steering mechanism | Critical |
| J4 | A_α | C2_STEER | **modulation** | α controls steering strength | High |
| J5 | A_L | C2_SELECT | **hierarchical** | Layer selection depends on A_L identity | Critical |
| J6 | A_M | A_B | **causal** | Model capability determines baseline | High |
| J7 | A_B | C2_CAP | **conditional** | Baseline > threshold enables steering | Critical |
| J8 | A_P | C2_SELECT | **causal** | Layer pattern determines selection | Critical |
| J9 | A_C | C2_CONTRAST | **compositional** | Contrast builds on velocity concept | Medium |
| J10 | C2_VEL | C3_TRAIN | **temporal** | Velocity training precedes steering | Critical |
| J11 | C3_TRAIN | C3_STEER | **temporal** | Training must complete before steering | Critical |
| J12 | C2_STEER | A_D | **causal** | Steering causes token divergence | Medium |
| J13 | C2_STEER | A_B | **causal** | Steering changes accuracy | **Highest** |
| J14 | A_S | C2_STEER | **constraint** | Steering surface constrains mechanism | High |
| J15 | C4_PIPELINE | PEAK | **compositional** | Pipeline instantiates the concept | High |
| J16 | A_R | A_H | **compositional** | Hidden states are part of residual stream | Low |
| J17 | C3_TRANSFER | C4_KNOWLEDGE | **generalization** | Transfer confirms pattern generality | Medium |
| J18 | A_D | C2_EVAL | **comparative** | Divergence is a secondary metric | Low |
| J19 | C2_CONTRAST | C3_TRAIN | **compositional** | Contrastive training is a mode of TT training | Medium |
| J20 | A_G | C2_VEL | **dependency** | Generation trajectories are training data | High |

---

## Key Junction Properties

### Critical Junctions (system-collapse risks)

| Junction | Collapse Mode | Mitigation |
|----------|--------------|------------|
| J2 (V→TT) | If R² is spurious, all downstream steering is compromised | Phase 8 mechanistic check |
| J7 (Baseline→Threshold) | If capability threshold paradigm is wrong, small models may be steerable | Test with RL-based steering |
| J13 (Steering→Accuracy) | If accuracy gain is statistical noise, the entire approach is invalidated | Increase N from 100 to 1000 |
| J5 (Layer→Selection) | If layer effects are not independent, layer selection is meaningless | Test with multi-layer combinations |

### Antagonistic Junctions

| Junction | Conflict Type | Evidence |
|----------|-------------|----------|
| J13 vs J12 | Accuracy gain vs token divergence | L8: +20pp at high divergence; L9: -23pp at high divergence — same divergence, opposite effect |
| J7 vs J6 | Threshold contradicts gradual scaling | 38% baseline (Math-1.5B) should show some steerability but shows none |
| J3 vs B1 | TT output direction may be wrong for steering | Descriptive accuracy ≠ normative utility |

---

## Hallucinatory Pre-Seeds (Unconstrained Ideal Forms)

| Atom | Pre-Seed |
|------|----------|
| A_H (Hidden State) | A fully interpretable, 3D latent space where reasoning steps are distinct trajectories in semantic directions |
| A_KV (KV Cache) | Steering is as simple as gradient descent on the KV manifold — differentiable, cheap, and architecture-agnostic |
| A_V (Velocity) | Velocity decomposes into interpretable components: "reasoning direction," "confidence momentum," "topic shift" |
| A_TT (TrajectoryTransformer) | A single universal TT that works for all models, discovered via foundation-model-scale pretraining on language generation trajectories |
| A_α (Alpha) | A gating network learns α = f(layer, token_position, task, confidence), producing zero steering when no improvement is possible |
| A_L (Layer Selectivity) | Steering is automatically routed through a differentiable "steering switch" — no manual layer selection needed |
| A_M (Model Capability) | The capability threshold is eliminated — steering creates reasoning ability via latent structure amplification |
| A_C (Contrastive Direction) | Contrastive direction is the first principal component of the trajectory ensemble, automatically extracted |
| A_P (Per-Layer Pattern) | Each layer's role in the computation graph is fully mapped, and steering targets specific computational functions |
| A_S (Steering Surface) | A universal steering interface that works on all architectures — K/V, Q, O, logits, and MLP activations are all steerable through the same mechanism |
