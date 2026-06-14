# Phase 7: Causal Mapping & Counterfactual Analysis

**Subject**: RankAdaptation — Velocity-based latent steering
**Date**: 2026-06-14

---

## Causal DAG

### Node Summary

| Node | Type | In-Degree | Out-Degree | Description |
|------|------|-----------|------------|-------------|
| N1 | Input | 0 | 1 | Model Architecture (MHA vs Hybrid) |
| N2 | Input | 0 | 1 | Model Baseline Capability |
| N3 | Input | 0 | 2 | Training Data Quality |
| N4 | Process | 2 | 2 | Trajectory Collection |
| N5 | Process | 2 | 2 | TT Training |
| N6 | Process | 2 | 1 | Contrastive TT Training |
| N7 | Parameter | 1 | 2 | Layer Selection (per-layer vs all) |
| N8 | Parameter | 0 | 2 | α (Steering Strength) |
| N9 | Steering | 3 | 2 | KV-Cache Steering |
| N10 | Intermediate | 1 | 1 | Token Divergence |
| N11 | Output | 2 | 0 | Final Accuracy |
| N12 | Output | 1 | 0 | Death Layer Identification |
| N13 | Output | 1 | 0 | Trim-Tab Identification |
| N14 | Mediator | 1 | 1 | Steering Direction Alignment |
| N15 | Mediator | 1 | 1 | Off-Manifold Risk |
| N16 | Input | 0 | 1 | Task Type (GSM8K vs SVAMP) |

### Edge Summary

| Edge | From | To | Type | Estimated Delay | Notes |
|------|------|----|------|----------------|-------|
| E1 | N1 (Architecture) | N7 (Layer Selection) | Constraint | Immediate | MHA → per-layer possible; Hybrid → limited |
| E2 | N2 (Baseline) | N11 (Accuracy) | Direct | Per-generation | Higher baseline → higher absolute accuracy |
| E3 | N2 (Baseline) | N9 (Steering) | Conditional | Per-generation | Below threshold → steering fails |
| E4 | N3 (Data Quality) | N4 (Collection) | Dependency | Training-time | Better data → better trajectories |
| E5 | N3 (Data Quality) | N5 (TT) | Causal | Training-time | Data quality → TT quality |
| E6 | N4 (Collection) | N5 (TT) | Sequential | Training-time | Trajectories → TT |
| E7 | N5 (TT) | N9 (Steering) | Causal | Generation-time | TT predicts velocity for steering |
| E8 | N6 (Contrastive) | N9 (Steering) | Causal | Generation-time | Contrastive TT provides alternative direction |
| E9 | N7 (Layer Selection) | N9 (Steering) | Config | Generation-time | Which layer gets steering |
| E10 | N8 (α) | N9 (Steering) | Modulatory | Generation-time | How much steering |
| E11 | N9 (Steering) | N10 (Divergence) | Direct | Per-token | Steering causes token changes |
| E12 | N9 (Steering) | N11 (Accuracy) | Indirect | End-of-generation | Steering → accuracy (mediated by alignment) |
| E13 | N5 (TT) | N14 (Alignment) | Causal | Per-token | TT direction determines alignment |
| E14 | N14 (Alignment) | N11 (Accuracy) | Causal | End-of-generation | Good alignment → accuracy improvement |
| E15 | N14 (Alignment) | N15 (Off-Manifold) | Causal | Per-token | Poor alignment → off-manifold |
| E16 | N15 (Off-Manifold) | N11 (Accuracy) | Causal | End-of-generation | Off-manifold → degradation |
| E17 | N9 (Steering) | N12 (Death Layer) | Discovery | Per-experiment | If steering at layer L degrades → L is death |
| E18 | N9 (Steering) | N13 (Trim-Tab) | Discovery | Per-experiment | If steering at layer L improves → L is trim-tab |
| E19 | N16 (Task) | N11 (Accuracy) | Modulatory | Per-experiment | Different tasks → different accuracy |
| E20 | N2 (Baseline) | N14 (Alignment) | Causal | Per-token | Higher baseline → better intrinsic alignment |

---

## Branching Points (out-degree ≥ 2)

| Node | Out-Degree | Branches To | Criticality |
|------|-----------|-------------|-------------|
| **N9: KV-Cache Steering** | 2 | N10 (Divergence), N11 (Accuracy via N14/N15) | HIGH — central decision point that affects both divergence and accuracy, often in opposite directions |
| **N5: TT Training** | 2 | N9 (Steering), N14 (Alignment) | HIGH — TT determines both the steering signal and its alignment quality |
| **N7: Layer Selection** | 2 | N9 (Steering application), N12/N13 (Layer identity) | HIGH — single decision determines trim-tab vs death-layer classification |
| **N8: α** | 2 | N9 (Steering intensity), N11 (Accuracy magnitude) | MED — affects both activation and outcome |
| **N3: Data Quality** | 2 | N4 (Collection), N5 (TT) | MED — upstream quality cascade |

---

## Counterfactuals

### CF-1: What if L9 steering used −α instead of +α?

| Aspect | Detail |
|--------|--------|
| Scenario | Apply anti-steering (flip α sign) at L9, the strongest death layer |
| Predicted Outcome | If L9 death is caused by direction misalignment, anti-steering should improve accuracy. If death is caused by off-manifold perturbation (any modification damages this layer), anti-steering should also degrade. |
| Testability | TRIVIAL — single parameter change in existing script |
| Information Gain | HIGH — distinguishes two fundamentally different mechanisms for death layers |
| Priority | **IMMEDIATE** (Rank #2 in Convergent Pulse) |

### CF-2: What if α=1.0 at L8 instead of α=0.1?

| Aspect | Detail |
|--------|--------|
| Scenario | 10× stronger steering at the best trim-tab layer |
| Predicted Outcome | If the steering effect is monotonic in α, accuracy should increase further (potentially +40pp+). If there's an optimal α, L8 may degrade at high α (death-by-oversteering). |
| Testability | TRIVIAL — α sweep on L8 |
| Information Gain | HIGH — reveals (layer, α) optimal manifold |
| Priority | **IMMEDIATE** (Rank #1 in Convergent Pulse) |

### CF-3: What if the contrastive direction (v_correct − v_incorrect) pointed AWAY from correct?

| Aspect | Detail |
|--------|--------|
| Scenario | The difference vector actually encodes spurious differences (format, length, token choice), not correctness-relevant differences |
| Predicted Outcome | Contrastive steering would either be neutral (no effect) or harmful (steering toward wrong-answer features). |
| Testability | Run contrastive evaluation and compare to standard TT; if standard outperforms contrastive, the signal is spurious |
| Information Gain | **CRITICAL** — determines the entire contrastive approach's validity |
| Priority | **IMMEDIATE** — this is D10 (untested contrastive) |

### CF-4: What if sub-threshold model (e.g., Math-1.5B, 38%) received α=0.5?

| Aspect | Detail |
|--------|--------|
| Scenario | Strong steering on a model below the "capability threshold" |
| Predicted Outcome | If threshold is fundamental: model collapses (α too big for weak model). If threshold is α-dependent: accuracy may improve. If threshold is about S/N ratio: accuracy improvement proportional to α×baseline. |
| Testability | Run α sweep on Math-1.5B |
| Information Gain | HIGH — tests MR#4 and resolves D4 |
| Priority | SHORT-TERM |

### CF-5: What if we removed edge E1 (Architecture → Layer Selection)?

| Aspect | Detail |
|--------|--------|
| Scenario | Assume hybrid attention models have the same trim-tab/death-layer structure as MHA |
| Predicted Outcome | For Qwen3.5-2B, per-layer steering would find trim tabs among the 25% steerable layers, or the recurrent-state path would provide an alternative steering surface |
| Testability | Test per-layer steering on Qwen3.5-2B's MHA layers only (25% of layers) |
| Information Gain | MEDIUM — tests architecture constraint rigidity |
| Priority | LOW (depends on resolving D5 first) |

---

## Intervention Points (where external modulation is feasible)

| Node | Intervention | Feasibility | Expected Effect | Risk |
|------|-------------|-------------|-----------------|------|
| **N7: Layer Selection** | Manual or automated layer choice | 5/5 | Up to ±45pp accuracy swing | Accidental death layer selection |
| **N8: α** | Parameter adjustment | 5/5 | Continuous modulation from improvement to destruction | Over-steering collapse |
| **N5: TT** | Training regime, data, architecture | 4/5 | Direction quality | Training artifacts |
| **N3: Data Quality** | Curation, augmentation | 3/5 | Upstream quality cascade | Diminishing returns |
| **N16: Task** | Task selection | 5/5 | Pattern validation | Generalization limits |

---

## Delay Mapping

| Causal Chain | Total Delay | Bottleneck |
|-------------|-------------|------------|
| Collection → TT Training → Steering | Hours (1-2) | TT training epoch time |
| α Change → Steering Result | Seconds (per generation) | Forward pass time |
| Layer Change → Steering Result | Minutes (per layer sweep) | 100 generations × generation time |
| Data Quality Change → New TT → Steering | Hours | Full retraining cycle |
| Architecture Change → All | Days | Model download, adaptation, new pipeline |
