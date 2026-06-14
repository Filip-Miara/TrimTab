# Phase 10: Hyperstitional Bridge

## Structural Hypotheses

### H-1: Steering-Random Equivalence
- **Type**: Structural
- **Statement**: "If we replace the TT's velocity prediction with a random vector of equal magnitude, per-layer steering patterns remain qualitatively similar (same trim-tab/death layer classification)."
- **Falsification**: L8 accuracy with random steering ≤ baseline + 5pp (95% CI)
- **Confirmation**: L8 accuracy with random steering ≥ baseline + 15pp
- **Minimum Experiment**: Phase A1 — 30-minute sweep with random vectors
- **Risk (false positive)**: Entire paradigm rests on spurious noise effect
- **Value (true positive)**: Confirms velocity direction matters; validates infrastructure
- **Research Call**: "Is latent steering just noise injection?" to mechanistic interpretability community

### H-2: Directional Death Layer
- **Type**: Structural
- **Statement**: "Death layers are layers where the sign of α determines whether the layer helps or hurts — negative α on L9 converts it to a trim-tab."
- **Falsification**: L9 accuracy with α=-0.1 ≤ baseline + 5pp
- **Confirmation**: L9 accuracy with α=-0.1 ≥ baseline + 10pp
- **Minimum Experiment**: Phase A2 — 30-minute sweep with α=-0.1
- **Risk**: If false, negative α could worsen performance (-30pp instead of -23pp)
- **Value**: Opens α-flipping as a general technique for death layer remediation

### H-3: Low-Dimensional Steering Manifold
- **Type**: Structural
- **Statement**: "The effective steering signal lives in ≤50 dimensions of the 3584-dim hidden state space. A low-rank TT (d_model=64) preserves ≥80% of the steering improvement."
- **Falsification**: Low-rank TT achieves ≤5pp improvement vs 20pp for full TT
- **Confirmation**: Low-rank TT achieves ≥16pp improvement (80% retention)
- **Minimum Experiment**: Train small TT, evaluate per-layer sweep, compare with full TT
- **Risk**: Low — small model trains faster. If false, the steering signal is genuinely high-dimensional.
- **Value**: Simpler, faster, cheaper steering pipeline

### H-4: K-Value Dominance
- **Type**: Structural
- **Statement**: "Steering the KEY projections alone produces the same accuracy improvement as steering both K and V. Value projection steering has negligible effect."
- **Falsification**: K-only steering ≤ 5pp improvement on L8
- **Confirmation**: K-only steering ≥ 15pp improvement on L8
- **Minimum Experiment**: Phase B3 — run L8 sweep with K-only and V-only variants
- **Risk**: Low — simple code change
- **Value**: Halves compute for steering, simplifies mechanism

## Relational Hypotheses

### H-5: R²-Steering Correlation
- **Type**: Relational
- **Statement**: "Per-layer R² of the TT's velocity prediction correlates positively with per-layer accuracy improvement under steering (ρ > 0.5)."
- **Falsification**: Pearson ρ < 0.2 between per-layer R² and per-layer Δ accuracy
- **Confirmation**: Pearson ρ > 0.5 (moderate-to-strong positive correlation)
- **Minimum Experiment**: Compute per-layer R² (hold-out validation data), correlate with per-layer sweep results
- **Risk**: Low — purely correlational analysis on existing data
- **Value**: If confirmed, R² becomes a cheap proxy for steering efficacy (no need for expensive per-layer sweeps)

### H-6: Multi-Layer Additivity
- **Type**: Relational
- **Statement**: "Steering multiple trim-tab layers produces accuracy improvement equal to (or exceeding) the sum of individual improvements — trim-tab effects are additive."
- **Falsification**: Multi-layer (L2+L8) accuracy ≤ max(L2, L8) + 5pp
- **Confirmation**: Multi-layer (L2+L8) accuracy ≥ L2_improvement + L8_improvement (≥37pp)
- **Minimum Experiment**: Phase B2 — 4 combinations × 100 problems
- **Risk**: Medium — if effects are sub-additive, the mechanism is more complex than expected
- **Value**: Multi-layer synergy would push accuracy toward theoretical upper bound

### H-7: All-Layer Rescue
- **Type**: Relational
- **Statement**: "If death layers are excluded from steering (α=0), steering ALL remaining layers simultaneously produces accuracy ≥ best single layer."
- **Falsification**: All-layers-with-mask accuracy ≤ best single layer
- **Confirmation**: All-layers-with-mask accuracy ≥ best single layer
- **Minimum Experiment**: Modify all-layers steering to accept a death-layer mask
- **Risk**: Low — partial experiment already done
- **Value**: Resolves the all-layers failure mode

### H-8: Contrastive Normativity
- **Type**: Relational
- **Statement**: "v_correct - v_incorrect produces a steering vector that, when applied to a neutral layer, improves accuracy proportionally to the model's capability."
- **Falsification**: Contrastive steering on any layer ≤ standard steering on that layer
- **Confirmation**: Contrastive steering on a neutral layer (e.g., L0 at +0pp) produces ≥+5pp improvement
- **Minimum Experiment**: Phase B1 — contrastive evaluation on neutral layers
- **Risk**: Medium — requires training investment already made
- **Value**: Converts the steering paradigm from descriptive to normative

## Potential Hypotheses

### H-9: Cross-Model Injection (Capability Bypass)
- **Type**: Potential
- **Statement**: "A capable model's (7B, 73% baseline) contrastive steering vector, projected into a less capable model's (0.5B, 6% baseline) hidden state space, improves the less capable model's accuracy by ≥5pp."
- **Falsification**: Small model accuracy unchanged under injected steering
- **Confirmation**: Small model accuracy improves ≥5pp (from 6% to 11%+) under injection
- **Minimum Experiment**: Phase C3 — use existing projection infrastructure
- **Risk**: Medium-high — if injected signal is meaningless in target space, wastes compute
- **Value**: If true, circumvents the capability threshold — most impactful finding possible
- **Theoretical implication**: Hidden state dynamics are model-agnostic across scales

### H-10: Death Layer as Immune Response
- **Type**: Potential
- **Statement**: "Death layers are an active 'anti-steering' mechanism — the model detects KV cache modification and adjusts its attention distribution to compensate. The harmful effect is the model defending its computation."
- **Falsification**: Removing KV modification on death layers (leaving them unsteered) while steering all other layers produces the same pattern as steering death layers
- **Confirmation**: The harm pattern disappears when death layers are excluded, and no compensation is visible in attention patterns
- **Minimum Experiment**: Compare attention patterns between steered and unsteered runs on death layers
- **Risk**: Low — observational study on existing data
- **Value**: Transforms understanding of steering from "assistance" to "adversarial perturbation"

### H-11: Steering Upper Bound
- **Type**: Potential
- **Statement**: "For a model with baseline accuracy B on GSM8K, the maximum achievable accuracy under steering is bounded by B + (1-B) * 0.75 — steering can recover at most 75% of remaining errors."
- **Falsification**: Any experiment achieving > B + (1-B)*0.75 accuracy
- **Confirmation**: All experiments below this bound
- **Minimum Experiment**: Meta-analysis of all steering results to date
- **Value**: Provides a theoretical upper bound, informs expectations, helps decide when to stop steering research

### H-12: Attention Head Specialization
- **Type**: Potential
- **Statement**: "Within a trim-tab layer, only 1-2 attention heads are responsible for the steering improvement. Within a death layer, 1-2 specific heads cause the harm. Per-head steering would amplify the signal and eliminate the noise."
- **Falsification**: Per-head steering within L8 doesn't outperform per-layer steering
- **Confirmation**: A single head in L8 produces ≥15pp improvement
- **Minimum Experiment**: Analyze attention head contribution via activation patching or head ablation
- **Risk**: Medium — requires new infrastructure for per-head KV modification
- **Value**: If true, enables extremely precise steering (head-level instead of layer-level)

## Summary Table

| H-ID | Type | Statement | Falsification | Experiment | Priority | Value |
|------|------|-----------|---------------|------------|----------|-------|
| H-1 | Structural | Random steering ≈ TT steering | Random ≤ +5pp | Phase A1 | 🔴 **CRITICAL** | Paradigm validation |
| H-2 | Structural | Negative α flips death layers | L9@α=-0.1 ≤ +5pp | Phase A2 | 🔴 **CRITICAL** | Death layer remediation |
| H-3 | Structural | Steering is low-dimensional | Small TT ≤ +5pp | Follow-up A | 🟡 HIGH | Efficient steering |
| H-4 | Structural | K-only steering suffices | K-only ≤ +5pp | Phase B3 | 🟡 HIGH | Simplified mechanism |
| H-5 | Relational | R² predicts steering quality | ρ < 0.2 | Compute from existing data | 🟡 HIGH | Cheap proxy metric |
| H-6 | Relational | Multi-layer additivity | Multi ≤ max single + 5pp | Phase B2 | 🟡 HIGH | Multi-layer protocol |
| H-7 | Relational | All-layers-with-mask rescue | Masked-all ≤ best single | Phase B2 variant | 🟡 HIGH | Fix all-layers failure |
| H-8 | Relational | Contrastive is normative | Contrastive ≤ standard | Phase B1 | 🔴 **CRITICAL** | Normative steering |
| H-9 | Potential | Cross-model injection | Small model unchanged | Phase C3 | 🟢 MEDIUM | Bypass threshold |
| H-10 | Potential | Death layer = immune response | Harm pattern from steering only | Observational | 🟢 MEDIUM | Paradigm shift |
| H-11 | Potential | Steering has upper bound | Any experiment exceeds bound | Meta-analysis | 🟢 MEDIUM | Expectation setting |
| H-12 | Potential | Per-head specialization | Per-head ≤ per-layer | New infrastructure | 🟢 MEDIUM | Ultra-precise steering |
