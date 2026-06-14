# Convergent Pulse — Complex α (Cross-Pollinated)

## Filtering

| Candidate | F1 (Feas ≥3) | F2 (Safe) | F3 (Telos ≥4) | F4 (Novelty ≥3) | F5 (Synergy ≥3) | Pass? |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| L8 phase sweep (θ ∈ {0, π/6, π/4, π/3, π/2}) | 5 | ✅ | 5 | 4 | 5 | ✅ |
| L9 phase sweep (θ ∈ {-π/2, 0, π/4, π/2, π}, r=0.05) | 5 | ✅ | 5 | 5 | 5 | ✅ |
| Acceleration R² measurement (existing data) | 5 | ✅ | 5 | 3 | 4 | ✅ |
| Complex α + contrastive (phase rotation of v_c-v_i) | 4 | ✅ | 5 | 5 | 5 | ✅ |
| Toy transformer with known a_known (extend C3) | 4 | ⚠️ | 5 | 4 | 5 | ✅ |
| Per-head complex α | 2 | ✅ | 5 | 5 | 5 | ❌ (Feas) |
| GDN complex recurrent phase | 2 | ⚠️ | 4 | 5 | 4 | ❌ (Feas) |

## Ranking

Score = (Novelty + Feas + Telos + (6-Risk) + Emerg) / 5

| # | Candidate | Nov | Feas | Telos | Risk | Emerg | **Score** | Timeline |
|---|-----------|:---:|:----:|:-----:|:----:|:-----:|:---------:|:---------|
| 1 | **L9 phase sweep (θ sweep)** | 5 | 5 | 5 | 2 | 5 | **4.6** | **TODAY** |
| 2 | **Acceleration R² measurement** | 3 | 5 | 5 | 1 | 4 | **4.2** | **TODAY** (10 min) |
| 3 | **L8 phase sweep** | 4 | 5 | 5 | 2 | 4 | **4.2** | **TODAY** |
| 4 | Complex + contrastive phase rotation | 5 | 4 | 5 | 2 | 5 | 4.2 | Short-term |
| 5 | Toy transformer with acceleration | 4 | 4 | 5 | 2 | 4 | 4.0 | Short-term |
| 6 | Per-head complex α | 5 | 2 | 5 | 3 | 5 | 3.2 | Medium |
| 7 | GDN complex phase | 5 | 2 | 4 | 3 | 5 | 3.2 | Long |

## Critical Insight (cross-pollinated)

**The acceleration R² measurement (Candidate #2) is the CRITICAL GATE for everything else.** If acceleration R² ≈ 0 (acceleration is noise), then every other candidate fails because α₂·a = noise. This is the analog of the norm-growth null hypothesis (H0-2) from the main synthesis, applied to acceleration. Measure it FIRST (10 min from existing data).
