# Phase 1: Atomic Decomposition & Pyramid Construction

## Concept Hierarchy

### Level 0 — Atoms (Indecomposable Units)

| ID | Atom | Evidence Grounding | Confidence |
|----|------|-------------------|------------|
| A01 | Hidden state h[l] at layer l | Directly observed during forward pass | 10/10 |
| A02 | Velocity v[l] = h[l+1] - h[l] | Defined operation; computed from observed h | 10/10 |
| A03 | Velocity prediction v̂[l] = TT(h[l]) | TT output, empirically measurable | 9/10 |
| A04 | KV cache entry modification | Code in kv_cache_steering.py, lines 41-95 | 10/10 |
| A05 | Steering strength α | Scalar hyperparameter | 10/10 |
| A06 | GSM8K accuracy | Empirical measurement | 10/10 |
| A07 | TrajectoryTransformer (TT) | Neural network defined in trajectory_transformer.py | 10/10 |
| A08 | Trim-tab layer | Layer where steering improves accuracy >5pp | 9/10 |
| A09 | Death layer | Layer where steering reduces accuracy >10pp | 9/10 |
| A10 | Baseline accuracy (no steering) | Empirical measurement | 10/10 |
| A11 | Capability threshold | Observed: models <~40% GSM8K cannot benefit | 8/10 |
| A12 | Generation trajectory | Recorded hidden states + velocities during generation | 10/10 |
| A13 | Prompt trajectory | Recorded hidden states during prompt processing | 10/10 |
| A14 | Contrastive velocity v_c - v_i | Computed difference of two TT outputs | 8/10 |
| A15 | Cross-model projection W_proj | Linear layer mapping source→target hidden dim | 10/10 |
| A16 | Token prediction (next token) | Standard LM head argmax | 10/10 |
| A17 | Per-layer α allocation | Scalar per layer (currently uniform α=0.1) | 10/10 |
| A18 | Training data regime | Correct vs incorrect trajectory subsets | 9/10 |
| A19 | Model architecture type | MHA vs GDN+FA hybrid | 10/10 |
| A20 | Hidden state manifold geometry | Latent structure of h[l] across tokens and layers | 7/10 |
| A21 | Attention head voting | How K/V modification changes attention distributions | 6/10 |
| A22 | Reasoning path quality | Correctness of intermediate reasoning (not just final) | 5/10 |

### Level 1 — Composites

| ID | Composite | Constituent Atoms | Junctions |
|----|-----------|-------------------|-----------|
| C01 | Trajectory collection pipeline | A01, A12, A13 | Compositional (pipeline) |
| C02 | TT training pipeline | A07, A18, A12 | Compositional + Dependency |
| C03 | KV-cache steering mechanism | A04, A05, A16, A17 | Causal (A04×A05 → A16) |
| C04 | Per-layer sweep | A08, A09, A06, A10 | Causal (A07×A04 → A06) |
| C05 | Contrastive steering | A14, A07, A04, A05 | Compositional (A14 uses A07) |
| C06 | Cross-model transfer | A15, A07, A04, A06 | Hierarchical (A15 transforms space) |
| C07 | Steering surface | A19, A04 | Constraint (A19 limits A04) |
| C08 | Capability limitation | A11, A10, A06 | Constraint (A11 gates A06) |
| C09 | "Steering is amplification, not creation" | A11, A10, A08, A20 | Causal + Abstraction |
| C10 | All-layers steering failure | A09, A08, A04, A05 | Causal (A09 dominates A04) |

### Level 2 — Subsystems

| ID | Subsystem | Composites | Function |
|----|-----------|------------|----------|
| S01 | Data generation subsystem | C01, A12, A13 | Produce training data for TT |
| S02 | Prediction subsystem | C02, A07, A03 | Learn velocity field |
| S03 | Intervention subsystem | C03, C04, C05, C06, C07 | Apply steering during generation |
| S04 | Evaluation subsystem | A06, A10, A22 | Measure effectiveness |

### Level 3 — System (Peak)

| ID | Peak Concept | All Subsystems | Integration |
|----|--------------|----------------|-------------|
| P01 | Velocity-based Latent Steering | S01→S02→S03→S04 | Full pipeline: collect → predict → steer → evaluate |

## Junction Typology

| JID | Source | Target | Type | Description |
|-----|--------|--------|------|-------------|
| J01 | A12 (gen trajectory) | A07 (TT) | DEPENDENCY | TT trained on gen trajectories |
| J02 | A07 (TT) | A03 (velocity prediction) | CAUSAL | TT produces velocity predictions |
| J03 | A03 (velocity prediction) | A04 (KV modification) | CAUSAL | Velocity used to modify KV cache |
| J04 | A04 (KV modification) | A16 (token prediction) | CAUSAL | Modified KV changes next-token logits |
| J05 | A05 (α) | A04 (KV modification) | MODULATORY | α controls steering strength |
| J06 | A08 (trim-tab) | A06 (GSM8K accuracy) | CAUSAL | Trim-tab layer improves accuracy |
| J07 | A09 (death layer) | A06 (GSM8K accuracy) | ANTAGONISTIC | Death layer reduces accuracy |
| J08 | A19 (architecture) | C07 (steering surface) | CONSTRAINT | Architecture determines what is steerable |
| J09 | A11 (capability threshold) | A08 (trim-tab) | GATING | Capability must exist for trim-tab to function |
| J10 | A18 (training regime) | A07 (TT) | MODULATORY | Correct/incorrect/all data changes TT behavior |
| J11 | A14 (contrastive velocity) | A03 (velocity prediction) | COMPOSITIONAL | Contrastive = v_correct - v_incorrect |
| J12 | A15 (projection) | A07 (TT) | TRANSFORMATIONAL | Maps source model's hidden dim to target |
| J13 | A20 (manifold geometry) | A09 (death layer) | EXPLANATORY | Manifold structure may explain death layers |
| J14 | C08 (capability limitation) | C04 (per-layer sweep) | CONSTRAINT | Limits which models can benefit from sweep |
| J15 | S01 → S02 → S03 → S04 | — | TEMPORAL | Pipeline order is fixed (sequential dependency) |

## Hallucinatory Pre-Seeds

For each atom, the ideal form if all constraints were lifted:

| Atom | Ideal Form | Key Difference from Current |
|------|------------|----------------------------|
| A01 (h[l]) | Full residual stream at every sub-layer, not just layer output | Richer signal per layer |
| A02 (v[l]) | Actual next-token hidden state derivative, not just inter-layer diff | Would capture token-level intent |
| A03 (v̂[l]) | Perfect velocity prediction (oracle) | No error propagation |
| A04 (KV modification) | Direct logit-space intervention (bypass attention entirely) | Avoids attention dilution |
| A05 (α) | Per-layer, per-token, per-head learned dynamic α(θ) | Full adaptive control surface |
| A06 (GSM8K) | Multi-metric reasoning quality: accuracy + step correctness + coherence | Richer signal than binary |
| A07 (TT) | Foundation model trained on all LMs' trajectories | Generalizable steering predictor |
| A08 (trim-tab) | All layers are trim-tabs; death layers don't exist | No harmful steering |
| A09 (death layer) | Identified and skipped automatically by a classifier | Safety mechanism |
| A10 (baseline) | Near-100% baseline (perfect model) | Steering would only need tweaks |
| A11 (capability threshold) | No threshold; any model can be steered to improve | Universal applicability |
| A12 (gen trajectory) | Real-time closed-loop trajectory collection during generation | No storage needed |
| A13 (prompt trajectory) | Same as gen trajectory (no distribution shift) | One regime for all |
| A14 (contrastive velocity) | Unbiased estimator of the "correct reasoning direction" | Pure normative signal |
| A15 (projection) | Universal hidden state space (all models share semantics) | No projection needed |
| A16 (token prediction) | Differentiable through the full generation loop | End-to-end steering optimization |
| A17 (per-layer α) | Learned by RL from final accuracy as reward | Optimal allocation |
| A18 (training regime) | Perfect separation of correct/incorrect reasoning steps (not just final answer) | Granular training signal |
| A19 (architecture) | Any architecture exposes steerable surface (universal interface) | No architecture lock-in |
| A20 (manifold geometry) | Fully understood and analytically tractable | No black box |
| A21 (attention heads) | Per-head steering (each head gets different α) | Finer-grained control |
| A22 (reasoning quality) | Perfectly measurable with automated verifiers | Ground truth for every step |
