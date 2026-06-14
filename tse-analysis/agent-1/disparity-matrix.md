# Phase 6: Disparity Detection & Reconciliation

**Subject**: RankAdaptation — Velocity-based latent steering
**Date**: 2026-06-14

---

## Disparity Inventory

### D1: High R² ↔ No Steering Improvement (logical_contradiction)

| Property | Detail |
|----------|--------|
| Nodes | A2 (TT predicts velocity R²=0.85-0.94) ↔ A3 (Steering should improve accuracy) |
| Observed | Math-1.5B: R²=0.892 but no trim tabs; Qwen2.5-7B: R²=0.855 with strong trim tabs |
| Type | logical_contradiction |
| Severity | FUNDAMENTAL — undermines the basic assumption that prediction quality → steering quality |
| Resolution | **ABSTRACTION**: Redefine R² as necessary but not sufficient for steering. Steering quality depends on direction alignment, not magnitude of prediction error. R² measures "how well" but not "where to." Accepted as bounded limitation. |
| Status | RESOLVED (bounded) |

### D2: Logit Correction Failure ↔ TT Prediction Quality (operational_incompatibility)

| Property | Detail |
|----------|--------|
| Nodes | A19 (Logit correction failed) ↔ A2 (TT predicts well) |
| Observed | Prompt-trained: 0% gen (distribution shift); Gen-trained: =baseline (no signal) |
| Type | operational_incompatibility |
| Severity | STRUCTURAL — suggests the steering surface matters as much as the signal |
| Resolution | **SUBSTITUTION**: Replace logit surface with KV-cache surface (already done). The incompatibility is resolved by changing the intervention surface, not the signal. |
| Status | RESOLVED (substitution) |

### D3: All-Layers Harmful ↔ Per-Layer Beneficial (logical_contradiction)

| Property | Detail |
|----------|--------|
| Nodes | A4 (Per-layer selectivity) ↔ "All layers steering" |
| Observed | All-layers: -45pp net; L8 only: +20pp; L9 only: -23pp; combined: less than sum of parts |
| Type | logical_contradiction |
| Severity | FUNDAMENTAL — suggests steering is not additive across layers |
| Resolution | **SYNTHESIS**: Steering signal interacts destructively when applied at multiple layers due to interference. The net effect is not the sum of per-layer effects. This is a property of the residual stream dynamics. Per-layer selectivity exploits the fact that most layers should be left unmodified. |
| Status | RESOLVED (synthesis) |

### D4: Capability Threshold ↔ α=0.1 Fixed (assumption_clash)

| Property | Detail |
|----------|--------|
| Nodes | A7 (Threshold: ~40%) ↔ A10 (α=0.1 default) |
| Observed | All sub-threshold models tested at α=0.1; none showed improvement |
| Type | assumption_clash |
| Severity | FUNDAMENTAL — the threshold observation depends on a specific α value |
| Resolution | **SUBSTITUTION**: Not a real contradiction — the threshold is a joint property of (model, α). If α changes, the effective threshold changes. The resolution is to test α sweep on a sub-threshold model to map the (baseline, α) → improvement boundary. |
| Status | **UNRESOLVED** — requires RECOMB-FP2 experiment to resolve |

### D5: Instruct Model Pattern ↔ Base Model Absence (abstraction_mismatch)

| Property | Detail |
|----------|--------|
| Nodes | A5 (Trim-tab on Qwen2.5-7B-Instruct) ↔ No trim-tab on Qwen2.5-Math-1.5B (base) |
| Observed | 7B-Instruct: L8 +20pp; 1.5B-base: no trim tabs at any layer |
| Type | abstraction_mismatch — comparing across model families, sizes, and training paradigms |
| Severity | STRUCTURAL — confounds size (7B vs 1.5B), training (instruct vs base), and architecture |
| Resolution | **SEPARATION**: Decompose into independent hypotheses: (1) instruct tuning creates steerable manifold, (2) larger models have richer velocity structure, (3) Math specialization changes dynamics. Test each separately. |
| Status | **UNRESOLVED** — requires controlled experiment (e.g., 1.5B-instruct or 7B-base) |

### D6: SVAMP Replication Weak ↔ GSM8K Strong (resource_conflict)

| Property | Detail |
|----------|--------|
| Nodes | GSM8K L8: +20pp ↔ SVAMP L8: +4pp |
| Observed | 5× weakening of trim-tab effect on different math dataset |
| Type | resource_conflict (generalization vs specificity) |
| Resolution | **ABSTRACTION**: The pattern (L8 best, L9 worst) generalizes, but the magnitude is dataset-dependent. This is expected — different datasets have different reasoning demands. The generalized finding is the *pattern*, not the *magnitude*. |
| Status | RESOLVED (bounded) |

### D7: Cross-Model Transfer Works ↔ Architecture Matters (goal_conflict)

| Property | Detail |
|----------|--------|
| Nodes | A12 (Cross-model transfer: SmolLM2→7B preserves pattern) ↔ A18 (Hybrid attention: Qwen3.5 steering fails) |
| Observed | MHA→MHA transfer works; MHA→Hybrid fails |
| Type | goal_conflict — we want universal steering but architecture constrains it |
| Resolution | **BOUNDING**: Document as fundamental constraint: cross-model transfer works within attention architecture families. MHA→MHA is feasible; MHA→Hybrid requires new steering mechanism for recurrent components. |
| Status | RESOLVED (bounded) |

### D8: PPL Not Correlating with Correctness ↔ Steering Should Target Uncertainty (relational)

| Property | Detail |
|----------|--------|
| Nodes | A20 (PPL gate <0.1% activation) ↔ "Steer when model is uncertain" |
| Observed | Reading head r=0.85 but model confidently wrong; PPL doesn't detect overconfidence |
| Type | relational — the assumed relationship (uncertainty ↔ incorrectness) doesn't hold |
| Resolution | **SEPARATION**: PPL is an uncertainty measure about next-token prediction, not about answer correctness. These are fundamentally different concepts. The incompatibility is resolved by recognizing that PPL-based gating targets the wrong uncertainty. |
| Status | RESOLVED (separation) |

### D9: High Token Divergence ↔ Accuracy Improvement (resource_conflict)

| Property | Detail |
|----------|--------|
| Nodes | A11 (88% token divergence) ↔ A9 (Accuracy: +20pp max) |
| Observed | Steering changes output dramatically (88% different tokens) but accuracy changes modestly (+20pp). Most of the divergence is noise, not improvement. |
| Type | resource_conflict — massive perturbation for modest gain |
| Resolution | **SYNTHESIS**: Token divergence is not a good proxy for steering quality. The divergence measure captures ALL changes, but only a small fraction of those changes are "steering toward correct" — most are incidental perturbations. A better metric would measure "steering efficiency" (improvement per unit divergence). |
| Status | RESOLVED (synthesis) |

### D10: Contrastive TT Trained ↔ Evaluation Not Done (temporal_misalignment)

| Property | Detail |
|----------|--------|
| Nodes | A8 (Contrastive TTs trained with R²≈0.83) ↔ "Evaluation pending" |
| Observed | TTs completed but no steering results; sweep script ready but not run |
| Type | temporal_misalignment — infrastructure ready but experiment untested |
| Severity | CRITICAL — the highest-value pending experiment is not yet executed |
| Resolution | **REORDERING**: Prioritize contrastive evaluation as the immediate next step. Move from "infrastructure building" to "hypothesis testing" phase. |
| Status | **UNRESOLVED** — requires execution priority |

### D11: Small Model Overhead (resource_conflict — infrastructure)

| Property | Detail |
|----------|--------|
| Nodes | Model inference (23ms) ↔ Python overhead (23ms) — 50% GPU utilization |
| Observed | For small models, Python generation loop = GPU compute time |
| Type | resource_conflict |
| Resolution | **SUBSTITUTION**: torch.compile (C++-level loop) or batched inference. Async loading and GPU caching already help. |
| Status | PARTIALLY RESOLVED |

### D12: High R² on All Data ↔ High R² on Correct/Incorrect Separately (logical)

| Property | Detail |
|----------|--------|
| Nodes | A2 (TT_all R²=0.855) ↔ A8 (TT_correct R²=0.832, TT_incorrect R²=0.829) |
| Observed | All three TTs have nearly identical R² (~0.83-0.86), suggesting that the trajectory dynamics are largely uniform across correct and incorrect paths |
| Type | logical_contradiction — if correct/incorrect trajectories were mechanistically different, their predictability would differ |
| Resolution | **ABSTRACTION**: The velocity structure of correct and incorrect trajectories is similar in PREDICTABILITY but different in DIRECTION. R² captures only predictability, not direction. The contrastive signal (v_correct − v_incorrect) captures direction difference, which is independent of predictability. |
| Status | RESOLVED (bounded) |

---

## Assumption Violation Check (Phase 0 Review)

| Phase 0 Assumption | Status | Evidence |
|-------------------|--------|----------|
| A1: Velocities contain steering info | PARTIALLY VIOLATED | High R² but direction alignment matters more than prediction error |
| A2: KV cache is right surface | CONFIRMED | All successful steering uses KV-cache; logit correction failed |
| A3: Per-layer is necessary | CONFIRMED | All-layers net negative by -45pp |
| A4: Amplifies, not creates | CONFIRMED | No sub-threshold model benefits |
| A5: MHA preferred | CONFIRMED | Hybrid attention unfriendly |
| A7: Contrastive → normative | UNTESTED | Evaluation pending |
| A8: Pattern generalizes | PARTIALLY CONFIRMED | Pattern generalizes, magnitude doesn't |
| A9: ~40% threshold | UNTESTED AT OTHER α | Boundary specific to α=0.1 |
| A10: R² correlates with success | VIOLATED | Math-1.5B: R²=0.892 but no trim tabs |
| A11: α=0.1 is near-optimal | UNTESTED | No α sweep performed |
| B1: Manifold is Euclidean | PARTIALLY VIOLATED | Death layers suggest steering pushes off-manifold |
| B4: Correct/incorrect separable | UNTESTED | Contrastive evaluation pending |
| B9: L9 death is functional | UNTESTED | No mechanistic analysis performed |

**Key Assumption Violations**:
1. **A10 violated**: R² is NOT a reliable predictor of steering success (Math-1.5B R²=0.892, no trim tabs)
2. **A11 untested**: The default α=0.1 may be suboptimal; threshold observations depend on it
3. **B1 partially violated**: Steering likely pushes states off the natural manifold, especially at death layers

---

## Reconciliation Summary

| Metric | Count |
|--------|-------|
| Total Disparities | 12 |
| Resolved | 7 (D1, D2, D3, D6, D7, D8, D9, D12) |
| Unresolved (Bounded) | 1 (D11 — partial) |
| **Unresolved (Critical)** | **3** (D4, D5, D10) |
| Key Assumption Violations | 3 (A10 violated, A11 untested, B1 partially violated) |

### Critical Unresolved Disparities

| Disparity | Unblocking Experiment | Impact if Unresolved |
|-----------|----------------------|----------------------|
| D4: Threshold × α | α sweep on sub-threshold model | May invalidate the capability threshold claim |
| D5: Instruct vs Base | 1.5B-Instruct or 7B-base test | May invalidate trim-tab generalization |
| D10: Contrastive untested | Run contrastive evaluation | May invalidate the core next-step hypothesis |
