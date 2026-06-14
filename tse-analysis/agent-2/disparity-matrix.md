# Phase 6: Disparity Detection & Reconciliation

---

## 6.1 Structural Disparities

### D1: Logical Contradiction — TT Predicts Velocity / Velocity May Not Exist as a Meaningful Quantity

**Concepts**: C01 (TT predicts velocity) × A02 (velocity = h[l+1] − h[l])
**Severity**: FUNDAMENTAL
**Description**: Velocity is defined as the finite difference between consecutive hidden states. But hidden states are not positions in a Euclidean space — they are activations in a high-dimensional neural network. The "vector" h[l+1] − h[l] is a difference of activation vectors, not a physical velocity. The assumption that this difference is semantically meaningful (B1 from VOID) is unverified.
**Status**: RESOLVED via BOUNDING — Document that "velocity" is a metaphor. The TT could be learning the auto-regressive correlation between adjacent layers, not a true dynamical variable. This is a fundamental ambiguity in the project's framing. The resolution: treat velocity as a **mathematical convenience** that works empirically, not a physical quantity.

### D2: Resource Conflict — Trajectory Data Volume vs Storage

**Concepts**: Data collection requires 10-35GB per model × 7 models tested × multiple splits (all/correct/incorrect)
**Severity**: OPERATIONAL
**Description**: 71GB SSD filled; external HDD used for overflow. Full 7B trajectory set is 35GB on HDD; training subset is 10.5GB on SSD. Data management consumes significant time.
**Status**: RESOLVED via SEPARATION — HDD for cold storage, SSD for active training subset. Pipeline already handles this.

### D3: Abstraction Mismatch — Single α vs Heterogeneous Layers

**Concepts**: A06 (single α per layer) × the fact that each layer has different dynamics
**Severity**: STRUCTURAL
**Description**: Using the same α for all tokens within a layer ignores the fact that the hidden state evolves during generation and the optimal steering magnitude may change.
**Status**: RESOLVED via SUBSTITUTION — Per-layer α vector (Stage 4 of autonomous sweep) already planned. RECOMMENDATION: extend to per-token α.

### D4: Logical Contradiction — High R² Should Mean Zero Steering Effect

**Concepts**: A10 (R²=0.94) × C02 (steering changes behavior 88% token divergence)
**Severity**: FUNDAMENTAL
**Description**: If TT predicts velocities with R²=0.94, then the predicted velocity is very close to the actual velocity. Steering modifies h by α·v_pred. If α=0.1, the steered state is h + 0.1·v_pred ≈ h + 0.1·v_actual. This is a small perturbation in a 2048-dim space. That this perturbation changes token selection 88% of the time implies the hidden state manifold is extremely sensitive to small perturbations — OR that the steering effect is coming from somewhere else (e.g., the projection through K/V weights amplifies the perturbation nonlinearly).
**Status**: UNRESOLVED — This is the central paradox of the project. Requires mechanistic investigation (Phase 8).

---

## 6.2 Relational Disparities

### D5: Temporal Misalignment — TT Trained on Prompt / Applied During Generation

**Concepts**: J14 (trajectory collected during generation) × C01 (TT trained on collected trajectories)
**Severity**: RELATIONAL
**Description**: The TT is trained on offline-collected trajectories (h[l] at each generation step) but applied online during a different generation. Distribution shift between training and inference: training data comes from a non-steered model, but during steering, the model's hidden states are being modified, creating a feedback loop where the TT sees states it wasn't trained on.
**Status**: UNRESOLVED — The TT is evaluated on held-out data from non-steered generation, but during actual steering, the hidden states deviate from the training distribution. This could explain why α > 0.5 causes quality collapse (out-of-distribution for TT). RESOLUTION via ABSTRACTION: reframe TT as learning the **local gradient** of the hidden state field, not the global dynamics. If the field is smooth enough, local gradients remain valid for small perturbations (α < 0.5).

### D6: Causal Incompatibility — L8 Trim Tab vs L9 Death Layer

**Concepts**: J24 (antagonistic junction between A07 and A08)
**Severity**: STRUCTURAL
**Description**: Adjacent layers (L8 and L9) have opposite steering effects. This is structurally incompatible with a simple "push toward correct manifold" explanation — if the correct manifold is a coherent region in hidden state space, pushing toward it at L8 should also help at L9.
**Status**: RESOLVED via ABSTRACTION — Reconceptualize L8 and L9 as serving different computational roles. L8 may be a "reasoning integration" layer where the model combines problem understanding with answer generation. L9 may be a "output preparation" layer where small perturbations disrupt the carefully assembled output representation. The disparity arises from different functional roles, not different manifold geometries.

### D7: Goal Conflict — Maximize Accuracy vs Minimize Token Divergence

**Concepts**: P01 (steering framework goal = improve accuracy) × observed 88% token divergence at α=0.1
**Severity**: OPERATIONAL
**Description**: Steering changes token selection 88% of the time. While some of these changes improve accuracy, most are neutral or harmful. The system is simultaneously trying to be "steered" (change tokens) and "stable" (don't change tokens needlessly).
**Status**: RESOLVED via SYNTHESIS — Accept that token divergence is the mechanism of action. The goal is NOT to minimize divergence but to **direct** divergence toward correct answers. The contrastive TT is designed for precisely this.

---

## 6.3 Potential Disparities

### D8: Synthetic vs Organic — Contrastive TT May Push Off-Manifold

**Concepts**: C04 (contrastive TT pair) × A18 (hidden manifold structure)
**Severity**: FUNDAMENTAL
**Description**: The contrastive difference v_correct − v_incorrect is a vector computed from two learned TTs. If the correct and incorrect manifolds are not linearly separable (as suspected for Math-1.5B), this difference vector may point in a direction that is not on either manifold — pushing the model into a region of activation space that is neither correct nor incorrect but simply degenerate.
**Status**: UNRESOLVED — Requires evaluation of contrastive TT on Qwen2.5-7B. If contrastive steering does NOT produce trim tabs (or produces worse results than standard TT), this confirms the off-manifold hypothesis.

### D9: Organic vs Synthetic — Cross-Model Transfer May Lose Signal

**Concepts**: C05 (cross-model transfer) × A15 (projection adaptation)
**Severity**: STRUCTURAL
**Description**: The SmolLM2→7B transfer requires projecting velocities from 960-dim → 3584-dim via a learned linear layer. This is a 3.7× expansion that could introduce artifacts. The 7B's own TT has R²=0.855 while SmolLM2's TT has R²=0.94. The transferred TT may benefit from higher R² but lose model-specific structure.
**Status**: RESOLVED via BOUNDING — The fact that both TTs agree L8 is the best layer (despite different dimensionalities, training data, and architectures) actually STRENGTHENS the finding. If the transferred TT disagreed with the native TT, it would be a concern. Agreement despite different sources is evidence for a model-agnostic phenomenon.

---

## 6.4 Assumption Violations (from Phase 0)

| Assumption | Status | Evidence | Affected Disparities |
|-----------|--------|----------|---------------------|
| A1 (velocities learnable) | NOT VIOLATED | R²=0.85-0.94 across models | — |
| A2 (steering improves accuracy) | NOT VIOLATED | L8: +20pp on 7B | — |
| A3 (per-layer selectivity mandatory) | PARTIALLY VIOLATED | Per-layer α vector might outperform single-layer; not tested | D3 (single α mismatch) |
| A4 (steering requires capability) | NOT VIOLATED | Consistent across 7 models | — |
| A5 (KV-cache is correct surface) | NOT VIOLATED | KV-cache works; alternatives failed | — |
| A6 (gen vs prompt trajectories differ) | NOT VIOLATED | R²=0.94 gen vs 0.62 prompt | — |
| A7 (MHA preferred) | NOT VIOLATED | Hybrid attention resists steering | — |
| A8 (pattern generalizes) | PARTIALLY VIOLATED | SVAMP shows same pattern but smaller magnitude (L8: +4pp vs +20pp on GSM8K) | Pattern may be task-specific in magnitude |
| A9 (contrastive→normative) | UNKNOWN | Evaluation pending | D8 (off-manifold risk) |
| B1 (Euclidean manifold) | POTENTIALLY VIOLATED | High sensitivity (88% divergence from small α) suggests non-Euclidean geometry | D1, D4, D5 |
| B4 (separable manifolds) | PARTIALLY VIOLATED | Math-1.5B shows no trim tabs with either standard or contrastive TT | D8 |

---

## Disparity Matrix Summary

| ID | Type | Severity | Status | Resolution |
|----|------|----------|--------|-----------|
| D1 | logical_contradiction | fundamental | RESOLVED | BOUNDING: velocity is a mathematical convenience |
| D2 | resource_conflict | operational | RESOLVED | SEPARATION: HDD/SSD tiering |
| D3 | abstraction_mismatch | structural | RESOLVED | SUBSTITUTION: per-layer α vector |
| D4 | logical_contradiction | fundamental | **UNRESOLVED** | Requires mechanistic investigation |
| D5 | temporal_misalignment | relational | **UNRESOLVED** | Needs OOD detection for TT predictions |
| D6 | causal_incompatibility | structural | RESOLVED | ABSTRACTION: different layer roles |
| D7 | goal_conflict | operational | RESOLVED | SYNTHESIS: direct, don't minimize divergence |
| D8 | off_manifold_risk | fundamental | **UNRESOLVED** | Requires contrastive evaluation results |
| D9 | signal_degradation | structural | RESOLVED | BOUNDING: agreement across models is evidence |

### Critical Unresolved Disparities

1. **D4** (High R² → zero steering paradox): This is the most intellectually critical. If the TT is almost perfectly predicting velocities, how can a small α cause large behavioral changes? The resolution of this paradox would reveal the true mechanism of steering. Hypothesis: the K/V projection parameters (W_k, W_v) amplify small hidden-state perturbations into large attention distribution changes due to the high-dimensional geometry of attention softmax.

2. **D5** (Train/test distribution shift): The TT is tested on states it was never trained on (steered states). The α > 0.5 degradation threshold may be exactly the point where the steered states leave the TT's training distribution.

3. **D8** (Contrastive off-manifold risk): Binary question — either contrastive steering works (proving separable manifolds) or it doesn't (indicating off-manifold degradation). This is the most important experimental result currently pending.
