# Temporal Plan — Complex α (Cross-Pollinated)

## Phase A: Gate Check (TODAY, ≤30 min)

### A-C1: Measure Acceleration Structure
- **Cost**: 10 min compute + 20 min TT training
- **Action**: From existing trajectory data, compute a[l] = h[l+1] - 2h[l] + h[l-1], train a TT_a on it, measure R²_a
- **Gate**: If R²_a > 0.3 → proceed to phase A-C2. If R²_a < 0.3 → ABANDON complex α concept

### A-C2: L8 Phase Sweep
- **Cost**: 1 hour (100 problems, θ ∈ {0, π/6, π/4, π/3, π/2})
- **Action**: Run `run_per_layer_sweep.py` modified to use `h' = h + r·(cosθ·v + sinθ·a)` with r=0.1, sweeping θ
- **Success**: θ_opt ≠ 0 (any optimal phase differs from pure velocity)
- **Failure**: θ_opt = 0 (pure velocity is always best) → complex α not useful at L8

### A-C3: L9 Phase Sweep (if A-C2 succeeds)
- **Cost**: 1 hour (same setup, L9 only)
- **Success**: θ_opt at L9 ≠ θ_opt at L8 → phase distinguishes trim-tab from death
- **Breakthrough**: L9 at θ_opt beats baseline → death layers are phase-inverted trim-tabs

## Phase B: Expansion (1-2 days)

### B-C1: N=100 Phase Sweep at All Layers
- Full 28-layer sweep with θ ∈ {0, π/4, π/2, π} (4 phases, not 9 — coarse grid)
- Identify which layers benefit from which phase

### B-C2: Complex + Contrastive (cross-pollinated)
- Phase-rotate the contrastive vector: h' = h + r·(cosθ·(v_c-v_i) + sinθ·(a_c-a_i))
- This is THE complete second-order normative steering equation

### B-C3: Phase-Aware α Vector
- Learn (r_l, θ_l) per layer via Bayesian optimization or gradient descent

## Phase C: Synthetic Validation (within main synthesis C3)
- Extend toy transformer with known acceleration
- Verify complex α pipeline recovers known (r, θ)

## Decision Tree

```
Start → A-C1: R²_a measurement
  ├── R²_a < 0.3 → STOP. Complex α collapses to real α.
  │                  Report: "Acceleration is noise — no second-order structure"
  └── R²_a ≥ 0.3 → A-C2: L8 phase sweep
       ├── θ_opt = 0 → complex α ≈ real α at L8.
       │                Report: "Trim-tab layers are velocity-dominated"
       └── θ_opt ≠ 0 → A-C3: L9 phase sweep
            ├── θ_opt(L9) = 0 → phase doesn't distinguish polarity.
            │                  Report: "Death layers are not invertible by phase alone"
            └── θ_opt(L9) ≠ θ_opt(L8) → MAJOR FINDING.
                 Report: "Phase distinguishes trim-tab from death-layer.
                          Death layers are layers with inverted or rotated optimal phase."
                 → Phase B: full expansion
```
