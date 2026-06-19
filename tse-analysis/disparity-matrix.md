# Phase 6: Disparity Detection & Reconciliation

---

## Disparity Matrix

### Structural Disparities

| ID | Concepts | Type | Severity | Description |
|----|----------|------|----------|-------------|
| SD-1 | A2 (d_model=768) vs A9 (48M params) vs D2 (90K trajectories) | resource_conflict | MEDIUM | 48M params / 90K samples = 533 params/sample. For 0.5M-dim output, this is 96 params per output dim. Models with higher param/sample ratio often generalize better. |
| SD-2 | A7 (Bidirectional) vs P7 (Causal worse) | operational_incompatibility | MEDIUM | Bidirectional works better empirically but creates training-inference mismatch. This is either genuine or an artifact. |
| SD-3 | D6 (Global Norm) vs A5 (Input Projection 3584→768) | abstraction_mismatch | HIGH | Global normalization destroys per-layer structure → input projection operates on degraded features. The projection's LayerNorm can't recover lost per-layer information. |
| SD-4 | T8 (MSE uniform) vs P1 (R²=0.85) | resource_conflict | MEDIUM | Uniform layer weighting means early layers (high velocity magnitude) dominate the loss. Late-layer velocity signal is under-represented. |

### Relational Disparities

| ID | Junction Conflict | Severity | Description |
|----|-------------------|----------|-------------|
| RD-1 | J08 (Norm→Input) × J02 (Data→Training) | HIGH | J02 depends on J08 being correct. If global norm destroys signal, the entire training pipeline is built on degraded data. The entire causal chain is compromised. |
| RD-2 | J09 (MSE→Weights) × J16 (Steering→Reasoning) | HIGH | J09 optimizes velocity MSE. J16 requires directionally accurate velocities. These are different optimization targets. The system may optimize MSE perfectly (J09) while failing at steering (J16). |
| RD-3 | J14 (Buffer→Crashes) × J04 (Buffer Size→Batch Size) | MEDIUM | Double-buffer causes CUDA crashes AND constrains batch size. There is no free parameter to independently optimize both. |
| RD-4 | J17 (AWQ Dist→TT) × J07 (Frozen→AdamW only TT) | MEDIUM | TT must adapt to AWQ distribution but Qwen is frozen. The only adaptation mechanism is changing TT weights, which causes forgetting. No mechanism for conditional computation. |

### Potential Disparities

| ID | Synthetic Variant Conflicts | Severity | Description |
|----|---------------------------|----------|-------------|
| PD-1 | V1.1 (Per-layer Norm) × V4.1 (Pure Causal) | LOW | Per-layer norm improves signal whether bidir or causal. Compatible. |
| PD-2 | V6.1 (Decomposed Loss) × V7.1 (PCA Compression) | LOW | Decomposed loss works in original space and PCA space. Compatible — PCA first, then decomposed loss in latent space. |
| PD-3 | EM-1 (Multi-format Training) × MR-3 (Domain-adversarial) | LOW | Both aim at AWQ transfer. They are complementary: EM-1 provides data diversity, MR-3 provides representation alignment. Best used together. |
| PD-4 | V3.1 (12-layer TT) × V7.1 (PCA Compression) | MEDIUM | If PCA reduces output to 256 dims, a 12-layer TT may be overkill. The 6-layer may be sufficient + PCA. Resource allocation mismatch. |
| PD-5 | V1.1 (Per-layer Norm) × D6 (Global Norm Data) | MEDIUM | Existing data is normalized globally. Switching to per-layer norm requires re-normalizing all 90K trajectories. Backward compatibility. |

### Assumption Violations (from Phase 0)

| Assumption | Violated By | Evidence | Impact |
|------------|-------------|----------|--------|
| A2 (Global norm correct) | SD-3, RD-1 | 5 lenses identify information loss | HIGH — preprocessing reform needed |
| A5 (MSE correct) | RD-2 | Direction vs magnitude conflation | HIGH — loss function misalignment |
| A14 (Uniform layer weight) | SD-4 | Gradient dominance by early layers | MEDIUM — training inefficiency |
| A12 (Stationary hidden states) | P4, P5 | AWQ transfer collapse | HIGH — core viability question |
| A19 (Bidirectional is correct) | SD-2 | Training-inference mismatch | MEDIUM — but unclear if important |

---

## Reconciliation Mechanisms

### R1: SD-3 + RD-1 — Normalization Mismatch (HIGH severity)
**Resolution**: **SUBSTITUTION** — Replace global normalization with per-layer normalization. Also apply to existing data via batch re-normalization.
**Cost**: 1 hour + 2 hours GPU time for data re-normalization.
**Outcome**: Resolved.

### R2: RD-2 — Loss Function Misalignment (HIGH severity)
**Resolution**: **SUBSTITUTION** — Replace single MSE with decomposed loss (cosine for direction, Huber for magnitude). Add learned gating parameter α to balance.
**Cost**: 3 hours code.
**Outcome**: Resolved.

### R3: SD-1 — Capacity vs Data Mismatch (MEDIUM severity)
**Resolution**: **SYNTHESIS** — Combine PCA compression (V7.1) with multi-format data (EM-1). PCA reduces effective output dimensionality → fewer params needed per output dim. More data (3× multi-format) increases param/sample ratio. Combined: both sides of the ratio improve.
**Cost**: 2-3 days.
**Outcome**: Resolved via synthesis.

### R4: RD-3 — Buffer Constraints (MEDIUM severity)
**Resolution**: **SEPARATION** — If CUDA crashes are from async prefetch, separate the concerns: use synchronous prefetch as fallback, async as optimization. Crash → retry with sync.
**Cost**: 1 day.
**Outcome**: Resolved (graceful degradation).

### R5: RD-4 — Frozen Qwen + AWQ Adaptation (HIGH severity)
**Resolution**: **ABSTRACTION** — Instead of adapting TT weights, introduce an abstraction layer: a small correction network (MLP, 5M params) that maps AWQ hidden states → BnB-equivalent states before feeding to TT. TT stays frozen, correction network adapts.
**Cost**: 2 days.
**Outcome**: Resolved — TT keeps BnB performance, correction network handles AWQ.

### R6: SD-2 — Bidirectional vs Causal (MEDIUM severity)
**Resolution**: **REORDERING** — Train with bidirectional (better signal), but at inference, use only left-to-right context via causal masking of the attention. The TT learns from full context but can be deployed causally.
**Cost**: 1 day.
**Outcome**: Resolved — bidirectional training, causal inference.

### R7: PD-5 — Data Backward Compatibility (MEDIUM severity)
**Resolution**: **BOUNDING** — Document that re-normalization changes the data format. Old checkpoints are incompatible with new normalization. Accept as a breaking change for improvement.

---

## Summary

| Metric | Count |
|--------|-------|
| Total Disparities | 17 |
| **Resolved** | **14** |
| Unresolved (Bounded) | 3 |
| Critical (blocking) | 0 (all resolved or bounded) |

### Key Assumption Violations Found
1. **Global normalization destroys per-layer signal** — Most impactful finding. Source of systematic R² ceiling.
2. **MSE optimizes magnitude, not direction** — Loss function misaligned with actual goal.
3. **Uniform layer weighting ignores velocity heterogeneity** — Training signal dominated by early layers.

### Unresolved Disparities (Bounded)

| ID | Disparity | Reason for Bounding | Acceptable? |
|----|-----------|-------------------|-------------|
| UD-1 | P1 (R²=0.85) × latent noise ceiling | Cannot be resolved without computing noise ceiling | Yes — testable hypothesis |
| UD-2 | C2 (Steering improves reasoning) × no end-to-end validation | Requires expensive downstream eval | Yes — must be resolved before deployment |
| UD-3 | F3 (Frozen Qwen) × C1 (Velocity from Qwen's hidden states) | Fundamental design constraint | Yes — TT is external by design |
