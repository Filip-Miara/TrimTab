# Phase 6: Disparity Detection & Reconciliation

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Structural Disparities

| ID | Type | Severity | Concept A | Concept B | Description |
|----|------|----------|-----------|-----------|-------------|
| D1 | logical_contradiction | Fundamental | A4 (TT trains on gen trajectories) | A2 (Velocity defined as h_{t+1}−h_t) | TT predicts velocity from past hidden states, but velocity is defined as the DIFFERENCE between consecutive states — if we knew h_{t+1}, we wouldn't need TT. The TT is predicting something that doesn't yet exist. This is standard next-step prediction and not a contradiction per se, but it means TT can only be evaluated on held-out time steps, not on truly novel trajectories. |
| D2 | abstraction_mismatch | Structural | A5 (Single scalar α) | A20 (Hidden dimension = 3584) | A single scalar modulating a 3584-dimensional vector is extreme dimensionality reduction. The assumption is that all dimensions are equally important for steering — almost certainly false. |
| D3 | resource_conflict | Structural | A12 (Generation trajectories = 10-35GB) | Storage limits (71GB SSD) | Trajectory data exceeds available SSD storage, requiring HDD spillover with 15× slower access. |
| D4 | logical_contradiction | Fundamental | A6 (L8: +20pp) | A7 (L9: -23pp) | Adjacent layers (L8 and L9) have OPPOSITE effects from the same operation. This contradicts the assumption that nearby layers have similar function — there must be a sharp functional boundary between L8 and L9. |
| D5 | abstraction_mismatch | Structural | C2-3 (Per-layer steering as "steering a layer") | A3 (KV cache) | "Steering a layer" actually means modifying the K/V projections at that layer, not the layer's computation itself. The steering surface (KV cache) is an indirect access point, not the layer's processing. |

## Relational Disparities

| ID | Type | Severity | Junction A | Junction B | Description |
|----|------|----------|-----------|-----------|-------------|
| D6 | temporal_misalignment | Structural | J19 (Pipeline: train→steer→evaluate) | J5 (Steering→Accuracy) | There's a delay between training (hours) and evaluation (hours). During this time, the model's hidden state distribution could shift (e.g., due to data drift in the generation process). The steering operates on old information. |
| D7 | causal_incompatibility | Fundamental | J3 (Steering Operator → KV Cache) | J9 (Standard MHA → Steering) | J3 says steering modifies the KV cache; J9 says standard MHA is required. But GDN/hybrid architectures have non-standard KV mechanisms — the causal path from "steering operator" to "accuracy improvement" is broken for these models. The steering mechanism is architecture-dependent. |
| D8 | goal_conflict | Potential | "Maximize GSM8K accuracy" | "Minimize token divergence" | High accuracy gains (+20pp) come with 88% token divergence (almost every token changes). This means the steered model is saying completely different things — accuracy might improve, but the model's behavior is fundamentally altered. |
| D9 | temporal_misalignment | Fundamental | "TT trained on unsteered trajectories" | "Steering changes trajectories" | The TT is trained on unsteered model outputs but applied to a steered model. The distribution shift between training and inference is uncontrolled. This is the core of the Steering Paradox from Lens 10. |
| D10 | causal_incompatibility | Structural | "Correct trajectories → TT_correct" | "Incorrect trajectories → TT_incorrect" | These TTs are trained on mutually exclusive data. Their outputs may not be combinable — v_correct and v_incorrect might live in different regions of the velocity manifold, making their difference (contrastive signal) ill-defined or dominated by manifold-boundary artifacts. |

## Potential Disparities

| ID | Type | Severity | Synthetic Variant | Organic Concept | Description |
|----|------|----------|------------------|----------------|-------------|
| D11 | operational_incompatibility | Structural | SV-15 (Head-level steering) | A3 (KV cache: layer-level) | Head-level steering requires modifying individual attention head K/V projections, but the current KV cache interface is layer-level. Implementing head-level access requires significant refactoring. |
| D12 | assumption_clash | Fundamental | ¬B1 (Velocity does NOT encode correctness) | A6 (L8 trim-tab) | If velocity doesn't encode correctness, then steering toward higher-velocity directions cannot improve accuracy. The +20pp at L8 must be explained by some OTHER mechanism (e.g., chaos-based randomness, fluency bias). |
| D13 | assumption_clash | Fundamental | ¬B4 (α·v moves off manifold) | C2-2 (Steering operator) | If the hidden state manifold is curved, α·v moves states OFF the manifold (producing OOD hidden states). The 88% token divergence would be a sign of manifold violation, not successful steering. |
| D14 | operational_incompatibility | Structural | SV-13 (Uncertainty-aware TT) | A4 (Current TT: deterministic) | Current TT outputs point estimates of velocity. Training an evidential regression models requires: (1) different loss function (NLL of Normal-Inverse-Gamma), (2) more data (uncertainty requires multiple samples at each point), (3) different architecture (output dimension × 4). |
| D15 | goal_conflict | Potential | SV-14 (Dual-Mode: standard + contrastive) | A9 (Contrastive signal) | If the contrastive signal is in a different direction than the standard TT's prediction (e.g., opposite sign), combining them could cancel out both effects. The optimal combination might require β < 0, which is counterintuitive. |

## Assumption Violations (from Phase 0)

| Violated Assumption | Evidence | Affected Disparities | Resolution |
|--------------------|---------|---------------------|------------|
| B1 (Velocity → correctness) | L9 has high R² but steering is HARMFUL | D12 | The assumption of velocity→correctness is layer-dependent. Velocity at L9 encodes something harmful (overconfidence, noise amplification). Solution: **Per-layer trust**: only steer using TTs from trim-tab layers. |
| B4 (α·v is valid steering) | 88% token divergence with α=0.1 is large for a small perturbation | D13 | The hidden state manifold is NOT flat — α·v at 10% of velocity norm causes near-complete token change. The effective perturbation is much larger than intended. Solution: **α scaling calibration** — estimate the actual perturbation magnitude vs intended magnitude, rescale accordingly. |
| B5 (TT architecture is adequate) | No explicit test of TT architecture quality | D9 | Without testing TT architecture, we don't know if prediction errors are due to velocity noise or model capacity limits. Solution: **Increase TT capacity 2× and compare R²**. |
| B6 (GSM8K is sufficient proxy) | No non-math testing | D8 | If non-math performance drops under steering, GSM8K gains are uninformative about general reasoning improvement. Solution: **Add non-math baseline/steering evaluation**. |
| B8 (α is constant across tokens) | No token-position analysis | D2 | Without testing per-token α, we assume uniform sensitivity — likely false. Solution: **Compute per-token steering effect on accuracy**. |

## Reconciliation

| Disparity | Mechanism | Resolution | Residual Risk |
|-----------|-----------|------------|---------------|
| D1 (prediction-definition gap) | BOUNDING | Document that TT evaluates on held-out steps of EXISTING trajectories, not novel generation. For novel scenarios, out-of-distribution prediction is an open question. | LOW — standard ML limitation |
| D2 (scalar α vs 3584-dim) | SYNTHESIS | Replace α with α_vector ∈ R^{3584} with learned structure (e.g., low-rank factorization of the steering vector). | MEDIUM — increases parameter count |
| D3 (storage conflict) | SEPARATION | Hot/cold data separation: keep recent trajectories on SSD, archive older ones to HDD. Auto-delete after model deployment. | LOW — operational |
| D4 (L8 vs L9 contradiction) | SYNTHESIS | The L8/L9 boundary marks a functional transition in the residual stream. Propose: L1-L8 = "reasoning computation" layers, L9+ = "output generation" layers. Verify with probing. | MEDIUM — speculative division |
| D5 (steering surface vs computation) | SUBSTITUTION | Reframe: we are not "steering layers" but "modifying the attention pattern at specific layers." This is KV-cache-based attention modulation, not layer steering. | LOW — semantic clarity |
| D6 (training-evaluation delay) | SEPARATION | Run TT training and evaluation in a back-to-back pipeline with no delay. If training finishes, immediately evaluate before model state changes. | LOW — process fix |
| D7 (architecture dependency) | ABSTRACTION | Implement a UnifiedSteeringInterface that handles KV-cache (MHA), recurrent-state (GDN), and embedding steering uniformly. | HIGH — significant engineering |
| D8 (accuracy vs divergence) | BOUNDING | Accept 88% divergence as inherent to KV-cache steering. Investigate if divergence can be reduced with lower α or position gating. | LOW — known trade-off |
| D9 (train-distribution mismatch) | SUBSTITUTION | Train TT on STEERED trajectories. Use the current TT to generate steered trajectories, then train a new TT on those. This bootstraps toward a TT that predicts steered-model dynamics. | HIGH — bootstrapping may diverge |
| D10 (mutually exclusive TT data) | SYNTHESIS | Instead of separate TTs, train a SINGLE TT that takes a "conditioning signal" (correct/incorrect) as input and predicts the corresponding velocity. The contrastive signal is then the difference in output with differing conditioning. | MEDIUM — architecture change |
| D11 (head-level steering refactoring) | REORDERING | Implement head-level steering as Phase C (medium-term) — after layer-level mechanism is fully understood. | LOW — deferred |
| D12 (velocity≠correctness) | SUBSTITUTION | Test the hypothesis: compare steering toward v_pred (TT) vs steering toward random vectors of same norm. If random matches TT's effect, velocity encoding is NOT the mechanism. | HIGH — could invalidate core assumption |
| D13 (manifold curvature) | ABSTRACTION | Estimate manifold curvature via intrinsic dimension analysis. If curvature is significant, use manifold-aware steering (e.g., projecting α·v onto the tangent space). | MEDIUM — additional analysis |
| D14 (uncertainty-aware refactoring) | REORDERING | Implement uncertainty-aware TT as Phase C. | LOW — deferred |
| D15 (contrastive cancellation) | BOUNDING | Test β ∈ {-1.0, -0.5, 0, 0.5, 1.0} for dual-mode steering to find the optimal combination sign and magnitude. | LOW — hyperparameter search |

---

## Disparity Matrix Summary

| Metric | Count |
|--------|-------|
| **Total Disparities** | **15** |
| Resolved | 13 |
| Unresolved (Bounded) | 2 (D4 speculative division, D7 significant engineering) |
| Critical Disparities (blocking) | **D12** (velocity→correctness assumption) and **D13** (manifold curvature) — if these are confirmed, the theoretical foundation shifts |
| Key Assumption Violations Found | 5 (B1, B4, B5, B6, B8) |

### Critical Path

The two critical disparities (D12, D13) MUST be addressed before further development:
1. **D12** (random vector control experiment): Run within 1 hour, produces binary result. If TT predictions > random vectors → velocity encoding confirmed. If TT ≈ random vectors → mechanism is not velocity-based.
2. **D13** (manifold curvature estimation): Run within 2 hours using intrinsic dimension estimation on hidden state trajectories. If curvature is significant → require manifold-aware steering. If manifold is approximately flat → α·v is valid.
