# Master Regulators — Complex α (Cross-Pollinated)

## Rankings (Influence × Leverage)

---

### #1: Steering Phase θ at L8 — Score: 86

| Attribute | Value |
|-----------|-------|
| **Influence Centrality** | 9/10 — L8 is the confirmed best trim-tab (+20pp). Complex α at L8 could amplify this further or unlock orthogonal improvement. |
| **Junction Leverage** | 9.5/10 — J6 (phase layer synergy) × cross-pollination from main synthesis (L8 × Contrastive) |
| **Modulation** | Sweep θ ∈ {0, π/6, π/4, π/3, π/2} at L8 with r=0.1 |
| **Expected Impact** | HIGH — If θ_opt ≠ 0, accuracy exceeds +20pp. If θ_opt = 0, L8 is confirmed as pure-velocity trim-tab. |
| **Cross-Pollination** | **From main synthesis**: L8 is MR#1. Adding phase is the next knob. **From mechanistic check**: If norm-growth dominates L8 velocity (H0-2), then θ ≠ 0 may be NEGATIVE (acceleration even more trivial). |

### #2: Phase Parameterization of Death Layers — Score: 79

| Attribute | Value |
|-----------|-------|
| **Influence Centrality** | 8/10 — Death layers (L7, L9, L15+) currently destroy accuracy. If they have invertible phase, complex α reclaims them. |
| **Junction Leverage** | 9.8/10 — J7 (antagonistic opposing phases) — if L9's optimal θ ≠ 0, the sign-flip hypothesis (main synthesis A1) is a special case of phase rotation. |
| **Modulation** | Sweep θ ∈ {-π/2, 0, π/4, π/2, π} for L9 with r=0.05 |
| **Expected Impact** | VERY HIGH — Converts 4+ death layers to trim-tabs. Could double total accuracy gain. |
| **Cross-Pollination** | **From main synthesis A1**: death-layer sign-flip is explicitly θ = π (inverted polarity). Complex α generalizes this to ANY rotation. **From emergent discovery RECOMB-7**: The death-layer problem may be a phase singularity in the steering space. |

### #3: Acceleration Structure (R²_a) — Score: 72

| Attribute | Value |
|-----------|-------|
| **Influence Centrality** | 6/10 — If acceleration R² is too low (< 0.3), the entire complex α concept collapses (acceleration = noise). If high (> 0.5), it opens a new control dimension. |
| **Junction Leverage** | 9/10 — J2 (acceleration = change in velocity) — this is the EMPIRICAL foundation. Everything depends on it. |
| **Measurement** | Compute a[l] = h[l+1] - 2·h[l] + h[l-1] from existing trajectory data, train TT_a on it, compare R²_a vs R²_v |
| **Expected Impact** | CRITICAL GATE — Determines whether complex α is viable. |
| **Cross-Pollination** | **From mechanistic check H0-2**: The norm-growth baseline for velocity is critical. For acceleration, the baseline is even simpler: predict a[l] = 0 (zero acceleration = constant velocity). If R²_a > 0, acceleration has structure. If R²_a ≈ 0, complex α reduces to real α anyway (α₂ = 0 is optimal). |

### #4: per-Head Phase Steering — Score: 65

| Attribute | Value |
|-----------|-------|
| **Influence Centrality** | 6/10 — Within L8, each of 28 attention heads could have its own (r_h, θ_h). The trim-tab effect might come from 3-5 heads at specific phases. |
| **Junction Leverage** | 9/10 — Cross-pollination from emergent EM-1 (per-head steering). Complex α + per-head = head-specific phase. |
| **Modulation** | Requires per-head K/V access (GQA shared KV limits this). |
| **Expected Impact** | VERY HIGH (medium-term) — If 5/28 heads at θ_opt drive trim-tab, amplify those 5× and zero the rest. |
| **Cross-Pollination** | **From emergent discovery EM-1**: per-head steering is CONFIRMED EMERGENT. Complex α at the head level is a further emergent capability. |
