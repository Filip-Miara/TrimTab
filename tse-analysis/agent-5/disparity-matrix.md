# Phase 6: Disparity Detection & Reconciliation

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Disparity Matrix

### Structural Disparities

| ID | Type | Concepts | Conflict | Severity | Resolution |
|----|------|----------|----------|----------|------------|
| D1 | **logical_contradiction** | A_V (1-step velocity) vs C3_PATTERN (per-layer pattern) | If velocity is truly local (1-step), then per-layer effects should be small (each layer modifies the hidden state by a small delta). But observed per-layer effects produce ±23pp, which is non-local — suggesting velocity is NOT the right steering surface. | **FUNDAMENTAL** | SEPARATION: Velocity predicts next state; steering amplifies or dampens that prediction. The effect is non-local because the residual stream amplifies small changes through subsequent layers. The two are compatible if we model the residual stream as an amplifier. |
| D2 | **operational_incompatibility** | A_S (steering surface = K/V) vs A_GDN (GatedDeltaNet recurrence) | KV-cache steering assumes standard attention (K/V stored per layer). GatedDeltaNet has no K/V cache — it's recurrent. On Qwen3.5, only 25% of layers have standard K/V. | **FUNDAMENTAL** | ABSTRACTION: Define steering surface abstractly as "the internal state that determines the next token's computation." For MHA it's K/V; for GDN it's the recurrent state. Implement steering as surface-agnostic. |
| D3 | **resource_conflict** | A_G (gen trajectories, 10.5GB) vs SSD storage (71GB total) | Trajectory data consumes 35GB on HDD, 10.5GB on SSD. SSD also holds 15GB model. 71GB fills quickly with model + trajectory + experiments. | **STRUCTURAL** | SUBSTITUTION: Compress trajectories (float16, delta encoding), stream from HDD, keep only current batch on SSD. |
| D4 | **abstraction_mismatch** | A_P (per-layer pattern) vs A_L (layer identity) | The pattern classification (trim-tab/death/neutral) is an abstraction over the physical layer index. But the abstraction may not capture the actual computational role of each layer. Two "neutral" layers may have different computational roles. | **MEDIUM** | ABSTRACTION: Replace "per-layer pattern" with "per-circuit pattern" — identify the computational circuit that each layer participates in (via mechanistic interpretability). |
| D5 | **logical_contradiction** | A_TT (R²=0.85-0.94) vs C4_KNOWLEDGE (steering works) | If TT predicts velocity with R²=0.94 but steering only produces at most +20pp (far below the accuracy gain from simply training on more data), then the TT is not capturing the right signal. High prediction accuracy ≠ useful steering signal. | **FUNDAMENTAL** | SYNTHESIS: The TT captures velocity dynamics faithfully (descriptive) but the steering direction (+α) may be wrong (normative). They are compatible if we separate descriptive accuracy from normative utility. |

### Relational Disparities

| ID | Type | Junctions | Conflict | Severity | Resolution |
|----|------|-----------|----------|----------|------------|
| D6 | **causal_incompatibility** | J2 (H→V) vs J3 (TT→Steer) | J2 says hidden state causes velocity. J3 says TT prediction causes steering. But if steering changes the hidden state, the causal chain becomes: H → V → TT → steer → H' → V' → ... This means J2 and J3 form a cycle where steering changes the input that J2's causal claim depends on. | **FUNDAMENTAL** | BOUNDING: This is a real feedback loop. The causal chain is not H→V→TT→Steer→Accuracy but H→V→TT→Steer→H'→V'→... The system is fundamentally cyclical; J2 and J3 are not incompatible, they form a coupled dynamical system. |
| D7 | **temporal_misalignment** | J7 (Baseline→Threshold) vs temporal timing | J7 says baseline accuracy determines whether steering works. This is a cross-sectional property (measured at time of evaluation). But baseline accuracy may change during generation as steering modifies the trajectory. | **MEDIUM** | REORDERING: The threshold applies to the model's capability at the time of steering, not at the time of evaluation. If steering creates capability during generation (controversial), the threshold is time-dependent. |
| D8 | **goal_conflict** | J13 (Steering→Accuracy) vs J12 (Steering→Divergence) | J13 seeks to maximize accuracy (increase J13). J12 seeks to minimize divergence (decrease J12). These goals conflict when large divergence is needed for large accuracy gains (L8: +20pp at 88% divergence). | **STRUCTURAL** | SYNTHESIS: The goal is accuracy improvement *bounded by* acceptable divergence. Formalize as: max accuracy s.t. divergence ≤ d_max. This is a constrained optimization problem. |
| D9 | **causal_incompatibility** | J5 (Layer identity→Selection) vs J8 (Layer pattern→Selection) | Both Layer identity and per-layer pattern claim to determine layer selection. If identity determines the pattern, then J5 ⊇ J8 (identity subsumes pattern). But the pattern may depend on more than just identity (e.g., task, α, other layers). | **MEDIUM** | ABSTRACTION: Layer selection is determined by (layer identity, task, α, other layers) jointly. J5 and J8 are incomplete specifications of the full selection function. |
| D10 | **temporal_misalignment** | J10 (Train TT) → J11 (Steer) | The sequential training-then-steering pipeline means the TT is always trained on data from the *previous* version of the model (or unsteered trajectories). By the time steering is applied, the TT is outdated. | **STRUCTURAL** | REORDERING: Implement online training: collect trajectories, train incrementally, and update steering mid-generation. |

### Potential Disparities (Organic vs Synthetic)

| ID | Type | Organic Concept | Synthetic Variant | Conflict | Severity | Resolution |
|----|------|----------------|-------------------|----------|----------|------------|
| D11 | **assumption_clash** | B1: TT sign = correct direction | V1: Negative α on death layers | The entire project assumes +α is correct. V1 contradicts this assumption for death layers. | **FUNDAMENTAL** | SYNTHESIS: The sign is layer-dependent. For some layers (L8, L2) +α is correct; for others (L9, L15+) −α is correct. The TT prediction direction is correct for trim-tabs and wrong for death layers because death layers perform different computations. |
| D12 | **goal_conflict** | A_B (baseline accuracy) | V6: Self-improving loop | Baseline measurement assumes a fixed model. Self-improving loop changes the model each iteration, invalidating the baseline. | **STRUCTURAL** | SEPARATION: Use rolling baseline — after each improvement cycle, re-evaluate unsteered baseline to account for distribution shift. |
| D13 | **abstraction_mismatch** | A_α (scalar α) | V3: Multi-scale TT | Scalar α applies uniformly to all velocity scales. Multi-scale TT predicts velocities at different scales, but α can't differentiate between them. | **STRUCTURAL** | SYNTHESIS: Multi-scale α — different α per velocity scale. α_s for velocity at scale s. |
| D14 | **operational_incompatibility** | A_KV (KV-cache steering) | PC-3: Train TT on steered trajectories | KV-cache steering modifies the generation; collecting steered trajectories requires running the generation loop, which requires steering. This creates a bootstrapping problem. | **MEDIUM** | REORDERING: Start with unsteered data, train initial TT, steer, collect steered data, train V2 TT. This is exactly the self-improving loop (V6). |

### Assumption Violations (from Phase 0 VOID)

| ID | Assumption | Evidence of Violation | Disparity Affected | Status |
|----|-----------|----------------------|-------------------|--------|
| B1 | Predicted sign is correct | Death layers exist — applying TT prediction at L9 destroys accuracy. If sign were correct for all layers, all layers would improve (or be neutral). | D11 | **VIOLATED** — Sign is layer-dependent |
| B3 | Layer independence | L15+ shows "complete collapse" (−23pp+) which is disproportionate — if layers were independent, each layer's contribution would be additive with diminishing returns. The collapse suggests cascading failure (layer interdependence). | D6, D10 | **VIOLATED** — Non-independent effects observed |
| B4 | TT learns causal dynamics | The +20pp upper bound despite R²=0.94 suggests the TT captures both causal (beneficial) and confounded (harmful) structure. High R² does not mean the TT learned the right thing. | D1, D5 | **SUSPECT** — Insufficient evidence either way |
| A3 | Trim-tab pattern is real (not noise) | SVAMP replicates GSM8K: L8=+4pp (attenuated but same direction). Cross-model transfer also replicates. Pattern is robust. | — | **CONFIRMED** — Pattern is real |
| B8 | KV-cache steering is local to layer | L15+ collapse = −23pp+. If steering were local, the effect would be comparable to L8's +20pp (same mechanism, different sign). The disproportionately large negative effect suggests non-local propagation. | D6 | **SUSPECT** — Evidence of non-local effects |

---

## Critical Disparities Summary

| ID | Description | Status | Impact | Effort to Resolve |
|----|------------|--------|--------|-------------------|
| D1 | Velocity locality vs non-local steering effect | **UNRESOLVED** (bounded) | Understanding mechanism | HIGH (mechanistic interpretability study) |
| D2 | K/V surface incompatible with hybrid attention | **RESOLVED** (Switch to MHA models) | Architectural restriction | LOW (already implemented) |
| D5 | High R² but limited accuracy gain | **UNRESOLVED** (bounded) | Core paradigm risk | MEDIUM (test V1: negative α) |
| D6 | Causal cycle H→V→TT→Steer→H' | **UNRESOLVED** (bounded) | Theoretical understanding | HIGH (formal causal model) |
| D8 | Accuracy vs divergence tradeoff | **RESOLVED** (constrained optimization) | Experimental design | LOW (add divergence constraint) |
| D11 | Sign depends on layer | **UNRESOLVED** (experiment pending) | **CRITICAL** — most important unresolved disparity | LOW (run ±α sweep) |

---

## Reconciliation Priority

1. **D11 (sign per layer)** — Minimal effort, maximum impact. A single experiment resolves whether death layers are just "flipped trim tabs."
2. **D5 (R² vs accuracy gap)** — If D11 resolves (death layers become trim tabs with −α), this disparity is partially resolved. Remaining gap may be due to TT quality or α suboptimality.
3. **D6 (causal cycle)** — Requires formal causal model; informs long-term theory.
4. **D1 (locality paradox)** — Requires mechanistic interpretability; Phase 8 target.
5. **D8 (accuracy-divergence tradeoff)** — Trivial to formalize; just add the constraint to optimization.
6. **D10 (TT is always outdated)** — Addressed by V6 (self-improving loop).
