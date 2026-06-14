# Phase 6: Disparity Detection & Reconciliation

## Structural Disparities

| D-ID | Concept A | Concept B | Type | Severity | Description |
|------|-----------|-----------|------|----------|-------------|
| D01 | A07 (TT, d_model=768) | A07 (need: captures all velocity structure) | operational_incompatibility | WARNING | 768-dim model processing 3584-dim hidden states has a 4.7× bottleneck. Is information lost? |
| D02 | A05 (α=0.1 constant) | A17 (per-layer α) | logical_contradiction | WARNING | "Per-layer" but not per-layer in practice — all layers use same α |
| D03 | A03 (velocity prediction) | A04 (KV modification) | abstraction_mismatch | CRITICAL | Velocity is in hidden-state space, KV modification is in attention-projected space. The projection W_k/W_v may not preserve steering direction. |
| D04 | S01 (data generation) | S02 (prediction) | operational_incompatibility | WARNING | Generation produces trajectories on Qwen2.5-7B; TT trains on 23/28 layers (missing layers 23-27). Prediction target doesn't match full architecture. |
| D05 | A12 (gen trajectory) | A13 (prompt trajectory) | abstraction_mismatch | WARNING | R²=0.94 vs 0.62 — they are structurally different kinds of trajectories, not just different distributions |

## Relational Disparities

| D-ID | Junction A | Junction B | Type | Severity | Description |
|------|-----------|-----------|------|----------|-------------|
| D06 | J02 (TT→velocity): "TT predicts velocity accurately" | J06 (trim-tab→accuracy): "Steering improves accuracy" | temporal_misalignment | CRITICAL | High R² means the TT predicts where the hidden state IS going, not where it SHOULD go. The relationship between predictive accuracy and steering efficacy is unverified. |
| D07 | J06 (L8→+20pp): "L8 improves accuracy" | J07 (L9→−23pp): "L9 destroys accuracy" | causal_incompatibility | CRITICAL | Adjacent layers (8 and 9) have OPPOSITE effects. If single-layer steering works via a local mechanism, adjacent layers shouldn't differ this dramatically. This suggests the mechanism is not local. |
| D08 | J08 (MHA→steerable): "MHA enables steering" | J08 (hybrid→not steerable): "GDN+FA prevents steering" | goal_conflict | WARNING | Architecture constraints directly contradict the project's goal of universal steering applicability. |
| D09 | J09 (threshold gates trim-tab): "Need 40% baseline" | C09 (amplification): "Steering amplifies existing capability" | logical_contradiction | FUNDAMENTAL | If steering amplifies capability, it should improve a 38% model to >38%. But Math-1.5B at 38% showed no improvement. The "amplification" model is incomplete. |

## Potential Disparities

| D-ID | Synthetic Variant | Organic Concept | Type | Severity | Description |
|------|-------------------|-----------------|------|----------|-------------|
| D10 | FP-1 (cross-model injection) | A11 (capability threshold) | assumption_clash | FUNDAMENTAL | Cross-model injection would circumvent the threshold if it works. The threshold may not be a fundamental limit but a steering-signal-quality limit. |
| D11 | M5 (merged steering) | J05 (α→KV mod is linear) | abstraction_mismatch | WARNING | Merged steering signals (v_standard + β·v_contrastive) assume linear combination works in a nonlinear system |
| D12 | PC-3 (random steering as baseline) | C09 (steering is amplification) | assumption_clash | FUNDAMENTAL | If random steering works as well as TT-predicted steering, the "amplification" story is wrong — it's just noise injection, not directed amplification. |

## Assumption Violations (from Phase 0)

| Assumption | Status | Violation Evidence | Affected Disparities |
|------------|--------|-------------------|---------------------|
| A1 (velocities learnable) | ✅ CONFIRMED | R²=0.85-0.94 across 4 models | None |
| A2 (per-layer selectivity) | ✅ CONFIRMED | L8:+20pp vs all-layers:-45pp | None |
| A3 (steering requires capability) | ⚠️ CHALLENGED | Math-1.5B at 38% baseline doesn't contradict (below 40%?) but the threshold is suspiciously close to 38% | D09, D10 |
| A4 (KV-cache is correct surface) | ⚠️ NOT VERIFIED | No comparison experiment done (steer vs other surfaces) | D03 |
| A5 (gen TT > prompt TT) | ✅ CONFIRMED | R²=0.94 vs 0.62 | D05 |
| A6 (cross-model transfer works) | ⚠️ WEAK (n=1) | Only SmolLM2→7B tested | — |
| A7 (MHA preferred) | ⚠️ UNTESTED for GDN steering | GDN-specific steering never seriously attempted | D08 |
| A8 (contrastive is normative) | ❓ UNKNOWN | Evaluation pending | D06, D11 |
| A9 (linear projection suffices) | ⚠️ UNTESTED | No nonlinear adapter tested | — |
| A10 (GSM8K is sufficient) | ⚠️ WEAK | Only math reasoning tested | — |
| A11 (α=0.1 is good) | ⚠️ UNTESTED | No α sweep per layer | D02 |
| A12 (100 problems enough) | ⚠️ WEAK | 95% CI ≈ ±10pp | — |
| B1 (R² → steering quality) | ❓ UNKNOWN | No correlation computed | D06 |
| B2 (layer independence) | ⚠️ UNTESTED | No multi-layer combinatorial sweep | D07 |
| B3 (α=0.1 optimal) | ⚠️ UNTESTED | No per-layer α sweep | D02 |
| B5 (separable manifolds) | ❓ UNKNOWN | No manifold analysis done | D11 |

## Reconciliation

### D03 (velocity ↔ KV projection mismatch) — ABSTRACTION
**Reconciliation**: Generalize the steering target. Instead of steering via K/V projections, steer the residual stream directly and observe how it propagates through ATTENTION (not just K/V). This tests whether the projection is the bottleneck.
**Resolution**: ABSTAIN (creates a new experiment proposal)

### D06 (predictive accuracy ≠ steering efficacy) — SYNTHESIS
**Reconciliation**: Compute the correlation between per-layer R² and per-layer steering Δ accuracy. If they're uncorrelated, the TT is learning structure irrelevant to steering. If correlated, R² is a useful proxy.
**Resolution**: Create a new metric: `steering_alignment = cos_sim(v_pred, v_optimal)` where v_optimal is computed via oracle (backward pass through generation loop).

### D07 (adjacent L8/L9 have opposite effects) — SUBSTITUTION
**Reconciliation**: Replace the "functionally specialized layers" hypothesis with a "boundary layer" hypothesis: L8 and L9 sit at a phase boundary in the transformer where the computation transitions from reasoning to output projection. Steering pushes states across this boundary, with direction depending on which side you're on.
**Resolution**: Measure hidden state norm, variance, and PCA projection across layers to characterize the phase boundary.

### D09 (threshold vs amplification contradiction) — BOUNDING
**Reconciliation**: The contradiction is resolved by recognizing that "amplification" amplifies the SIGNAL the model already generates. Below threshold, the signal-to-noise ratio of correct reasoning in hidden states is too low — steering amplifies noise. Above threshold, the SNR is high enough that amplification helps.
**Resolution**: Document as a fundamental limit (irreconcilable without changing the steering paradigm).

### D10 (cross-model injection vs threshold) — SYNTHESIS
**Reconciliation**: Cross-model injection doesn't contradict the threshold — it BYPASSES it by providing an external signal. The threshold applies to self-steering (model using own trajectories), not to externally-guided steering.
**Resolution**: This creates the design space for "steering parasitism" — using a capable model's steering signal to improve a less capable model.

## Summary

| Category | Count |
|----------|-------|
| **Total Disparities** | 12 |
| **Resolved** | 5 (D03, D06, D07, D09, D10) |
| **Unresolved (Bounded)** | 5 (D01, D02, D04, D08, D11) |
| **Critical/Blocking** | 2 (D06, D07) |
| **Key Assumption Violations Found** | A3, A4, A6, A8, A9, A10, A11, A12, B1, B2, B3, B5 |

The two most critical unresolved disparities are:
1. **D06**: The fundamental gap between velocity prediction accuracy and steering efficacy. Until this is resolved, the entire approach rests on an unverified assumption.
2. **D07**: The L8/L9 adjacent-layer paradox. Resolving this would provide mechanistic understanding of how steering actually works.
