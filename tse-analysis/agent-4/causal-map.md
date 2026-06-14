# Phase 7: Causal Mapping & Counterfactual Analysis

## Causal DAG

```
[A01: Hidden state h[l]] 
    |
    |---→ [A12: Gen trajectory] ---→ [A07: TT training] ---→ [A03: Velocity prediction]
    |                                      |                        |
    |                                      |                        ↓
    |                                      |               [A04: KV modification]
    |                                      |                        |
    |                                      |                        ↓
    |---→ [A10: Baseline accuracy]      [A18: Train regime]   [A16: Token prediction]
    |                                                        (correct/incorrect/all)     |
    |                                      |                        |
    |                                      ↓                        ↓
    |                               [A14: Contrastive signal]  [A06: GSM8K accuracy]
    |                                      |                        ↑
    |                                      ↓                        |
    |                               [C05: Contrastive steering] ----+
    |                                                                
    +---→ [A19: Architecture] ---→ [C07: Steering surface] ---→ [A04: KV modification]
                                       |
                                       ↓
                                 [A08: Trim-tab layer] ---→ (+) → [A06]
                                 [A09: Death layer]    ---→ (-) → [A06]
                                       ↑
                                       |
                                  [A17: Per-layer α]
                                       |
                                  [A05: α allocation]

[A15: Cross-model projection] ---→ [C06: Cross-model transfer] ---→ [A06]

[A11: Capability threshold] ---→ [C08: Capability limitation] ---→ [gates A08 existence]
```

### Node Statistics

| Node | In-Degree | Out-Degree | Type |
|------|-----------|------------|------|
| A04 (KV modification) | 4 (A03, A05, C07, A17) | 2 (A16) | **HIGHEST IN-DEGREE** — central intervention point |
| A06 (GSM8K accuracy) | 5 (A10, A08↑, A09↓, A16, C06) | 0 | Sink node — all paths lead here |
| A03 (velocity prediction) | 1 (A07) | 2 (A04, A14) | Key mediator |
| A07 (TT) | 2 (A12, A18) | 2 (A03, A14) | Learnable function |
| A08/A09 (trim-tab/death) | 2 (C07, A17) | 1 each | **BRANCHING POINTS** |
| A11 (threshold) | 0 | 2 (C08) | Exogenous variable |
| A05 (α) | 1 | 3 (A04, A08, A09) | **HIGH OUT-DEGREE** |

### Edge Delays

| Edge | Causal Type | Estimated Delay | Notes |
|------|-------------|-----------------|-------|
| A12→A07 | Causal | 1-2 hours (training time) | Training pipeline |
| A07→A03 | Causal | <1ms (inference time) | Forward pass |
| A03→A04 | Causal | <0.1ms | Steering application |
| A04→A16 | Causal | ~10-100ms (next token generation) | Per-token delay |
| A08→A06 | Causal | 100-400ms (full generation) | Accumulates over tokens |
| A11→C08 | Constraint | N/A | Always present |
| A19→C07 | Constraint | N/A | Architecture fixed at selection time |

## Branching Points (Top-5 by Out-Degree)

| # | Node | Out-Deg. | Description | Why High Leverage |
|---|------|----------|-------------|-------------------|
| 1 | **A04 (KV modification)** | 2→A16 + feedback | Central intervention. ALL steering flows through this. Changing the mechanism here propagates everywhere. | It's THE intervention point. |
| 2 | **A07 (TT training)** | 2→A03, A14 | TT outputs feed both standard and contrastive pipelines. Changing training data changes both. | The quality of training determines everything downstream. |
| 3 | **A05 (α allocation)** | 3→A04, J05 mod | α affects trim-tab AND death layers simultaneously. It's the only tunable parameter with broad effect. | The ONLY dial the operator can turn in real-time. |
| 4 | **A18 (training regime)** | 2→A07 (mode) | Switching between correct/incorrect/all data changes what the TT learns. | Determines descriptive vs normative. |
| 5 | **A17 (per-layer α)** | 2→A08, A09 | Per-layer α decides which layers are trim-tab, death, or neutral. | Binary classification with continuous parameter. |

## Counterfactuals

### CF-1: What if α=0.0 (no steering at all)?
**Scenario**: Run the full sweep but never apply any steering. Compare with α=0.1 results.
**Predicted Outcome**: L8 accuracy would be 45% (baseline) instead of 65%. L9 would be 45% instead of 0%. The trim-tab effect disappears entirely; the death effect also disappears.
**Testability**: This is just the baseline measurement — already done (45% for baseline).
**Counterfactual Insight**: The trim-tab effect is REAL (not a statistical fluctuation) because L8's accuracy significantly exceeds baseline. But the death effect is ALSO real.

### CF-2: What if steering was applied at L8 with α=0.5 instead of α=0.1?
**Scenario**: Increase steering magnitude on the best trim-tab layer.
**Predicted Outcomes**:
- **Optimistic**: Accuracy improves further (65% → 80%) — stronger steering provides stronger signal
- **Pessimistic**: Accuracy drops (65% → 30%) — L8 becomes a death layer at higher α; the trim-tab effect is α-dependent
- **Likely**: Accuracy plateaus or drops slightly — there's a "sweet spot" α for each layer
**Testability**: Easy — modify per_layer_sweep to accept α as parameter. Cost: ~1 hour.
**Counterfactual Insight**: This would reveal whether L8 is intrinsically a trim-tab or whether the classification is α-dependent.

### CF-3: What if we steered ALL layers but with α=0 for death layers?
**Scenario**: Multi-layer steering with a "death layer mask": α=0.1 for L2, L3, L5, L8, L10; α=0 for L0, L1, L4, L7, L9, L15+.
**Predicted Outcome**: Accuracy somewhere between 65% (best single layer) and theoretical multi-layer synergy estimate (70-75%). Death layers don't contaminate the signal.
**Testability**: Medium complexity — modify per_layer_sweep to accept a mask. Cost: ~2 hours.
**Counterfactual Insight**: Would reveal whether multiple trim-tabs are additive, synergistic, or redundant.

### CF-4: What if the TT was trained on CORRECT trajectories only and applied with standard steering (not contrastive)?
**Scenario**: TT_correct (R²=0.832) is used instead of TT_all (R²=0.855) for standard steering.
**Predicted Outcome**: If the correct-only TT produces BETTER steering (despite lower R²), then steering quality depends on training data purity, not prediction accuracy. This would confirm the descriptive→normative hypothesis.
**Testability**: Easy — use the trained TT_correct.pt checkpoint. Cost: ~30 min.
**Counterfactual Insight**: Critical test of the hypothesis that steering quality and prediction accuracy are linked.

### CF-5: What if L9's steering direction was inverted (α = -0.1)?
**Scenario**: Apply negative steering to L9, pushing hidden states in the OPPOSITE direction of the TT's prediction.
**Predicted Outcomes**:
- **Death layer converted**: L9 accuracy = baseline or higher (death layer was pointing in wrong direction)
- **Death layer unaffected**: L9 accuracy = 0% (death layer effect is independent of direction)
- **Worse**: L9 accuracy even lower (both directions are harmful)
**Testability**: Easy — modify one line in per_layer_sweep. Cost: ~30 min.
**Counterfactual Insight**: Would determine whether death layers are directional or omnidirectional.

### CF-6: What if the TT architecture was significantly smaller (d_model=128, 2 layers)?
**Scenario**: Train a tiny TT on the same data. Compare prediction accuracy AND steering efficacy.
**Predicted Outcome**: If tiny TT preserves trim-tab/death pattern despite lower R², the steering-relevant structure is LOW-dimensional. If pattern disappears, high-dimensional structure is essential.
**Testability**: Medium — modify trajectory_transformer.py. Cost: ~2 hours.
**Counterfactual Insight**: Reveals the intrinsic dimensionality of steering-relevant velocity structure.

### CF-7: What if cross-model transfer went in the opposite direction (7B→SmolLM2)?
**Scenario**: Transfer Qwen2.5-7B's TT to SmolLM2-360M (3584→960 dim) via a learned projection.
**Predicted Outcome**: If transfer works bidirectionally, it confirms velocity dynamics are truly model-agnostic. If one-directional only, the "model-agnostic" claim is weakened.
**Testability**: Requires training a projection matrix for the reverse direction. Cost: ~3 hours.
**Counterfactual Insight**: Strong test of the model-agnostic velocity hypothesis.

## Intervention Points (Feasible External Modulation)

| Node | Intervention | Feasibility | Expected Effect |
|------|-------------|-------------|-----------------|
| **A05 (α)** | Change scalar value | INSTANT — command-line parameter | Direct, immediate, reversible |
| **A17 (per-layer α)** | Set per-layer mask | EASY — code modification | Direct, immediate, reversible |
| **A18 (training regime)** | Switch correct/incorrect/all | MODERATE — retraining needed | Indirect, delayed, persistent |
| **A07 (TT capacity)** | Change d_model, n_layers | MODERATE — retraining needed | Indirect, delayed, persistent |
| **A04 (steering surface)** | Switch KV→residual→logits | HARD — code redesign | Direct, immediate, design-dependent |
| **A19 (model choice)** | Switch MHA→hybrid | EASY — choice at startup | Indirect, persistent for session |
