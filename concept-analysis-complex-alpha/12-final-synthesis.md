=======================================================================
COMPLEX α — FINAL SYNTHESIS REPORT (Cross-Pollinated)
=======================================================================
Subject: Complex α as Acceleration-Phase Steering for Velocity-Based Latent Steering
Mode: Full (12 phases, cross-pollinated with main tse-analysis/)
Date: 2026-06-14

--- EXECUTIVE SUMMARY ---

The idea of making α complex (α = α₁ + iα₂) has a concrete, implementable
meaning in the hidden state space: α₁ controls steering along velocity
(first derivative), α₂ controls steering along acceleration (second derivative).
Equivalently, in polar form α = r·e^(iθ), θ = 0 gives pure velocity steering,
θ = π/2 gives pure acceleration (curve-bending), and intermediate θ gives a
controlled blend. This subsumes the death-layer sign flip (θ = π) as a
special case.

Two CRITICAL GATES determine viability:
1. **Gate 1 (10 min)**: Compute acceleration a[l] from existing trajectory data.
   If acceleration has structure (R²_a ≥ 0.3), proceed. If not, complex α ≈ real α.
2. **Gate 2 (1 hour)**: Sweep phase θ at L8. If θ_opt ≠ 0, complex α adds value.
   If θ_opt = 0, trim-tab layers are velocity-dominated.

The highest-value prediction: **the acceleration-to-velocity ratio
||a[l]||/||v[l]|| may predict layer polarity without running sweeps**
(H-C4) — death layers might have high acceleration relative to velocity,
meaning their dynamics are dominated by curvature rather than direction.

--- CROSS-POLLINATION SUMMARY ---

| Main Synthesis Finding | Complex α Implication |
|------------------------|----------------------|
| H0-2: Norm-growth null | Acceleration may be DOUBLY trivial — measure R²_a FIRST |
| A1: Death-layer sign flip | θ = π is one special case; θ could be ANY value |
| A3: Contrastive cosine | v_c and v_i may share θ=0 but diverge at θ=π/2 |
| C3: Synthetic validation | Extend toy transformer with a_known — verify pipeline |
| EM-1: Per-head steering | Each head gets (r_h, θ_h) — richer control surface |
| RECOMB-5: GDN state | GDN state update S += β(k⊗v) → S += β·e^(iφ)·(k⊗v) |
| Quadruple synergy | Complex α adds geometric dimension to ALL four components |
| Open Question 3 (L9 mystery) | Phase inversion explains why L9 kills accuracy |

--- ATOMIC CONCEPTS ---
A1: Velocity v[l] = h[l+1] - h[l] (existing, R²=0.855)
A2: Acceleration a[l] = h[l+1] - 2h[l] + h[l-1] (proposed, R² unknown)
A3: Real α₁ = coefficient on velocity
A4: Imaginary α₂ = coefficient on acceleration
A5: Steering Phase θ = atan2(α₂, α₁)
A6: Steering Magnitude r = √(α₁² + α₂²)

--- EMERGENT CAPABILITIES ---
1. **Phase-Disambiguated Steering** (CONFIRMED EMERGENT): Separates correction
   direction from correction geometric mode
2. **Phase-Locked Multi-Layer Resonance** (CONFIRMED EMERGENT): Aligned phases
   produce constructive interference across layers

--- MASTER REGULATORS ---
1. L8 Phase θ (Score: 86) — does phase add to pure velocity?
2. Death-Layer Phase Parameterization (Score: 79) — can θ invert polarity?
3. Acceleration R² (Score: 72) — is acceleration structured at all?

--- TOP RECOMMENDATIONS (sorted by priority) ---

**CRITICAL GATE — TODAY, 10 min:**
#1: Compute acceleration a[l] from existing trajectory data, measure R²_a
    If R²_a < 0.3 → STOP. Document that acceleration is noise.

**IF GATE PASSES — TODAY, 1 hour each:**
#2: L8 phase sweep θ ∈ {0, π/6, π/4, π/3, π/2} with r=0.1
#3: L9 phase sweep θ ∈ {-π/2, 0, π/4, π/2, π} with r=0.05

**SHORT-TERM:**
#4: Compute ||a[l]||/||v[l]|| ratio per layer → correlate with trim-tab/death-layer
    from main synthesis per-layer sweep results
#5: Phase-rotate contrastive vector: r·e^(iθ)·(v_c - v_i + a_c - a_i)
#6: Extend synthetic toy transformer with a_known

--- TESTABLE HYPOTHESES ---

H-C1: R²_a ≥ 0.3 (acceleration is structured) [GATE, 10 min]
  → falsified by: R²_a < 0.1

H-C2: θ_opt(L8) ≠ θ_opt(L9) (phase distinguishes polarity) [1 hour]
  → falsified by: both have same optimal θ

H-C3: ||a||/||v|| correlates with per-layer accuracy Δ [30 min]
  → if true: zero-shot polarity prediction from trajectory data

H-C4: Complex + contrastive outperforms either alone [1 day]
  → falsified by: complex α doesn't improve over real α with contrastive direction

--- RESOURCE-BUDGETED PLAN ---

Phase A (TODAY, ≤2.5h):
  A-C1: Acceleration R² measurement (10 min) — MUST DO FIRST
  A-C2: L8 phase sweep (1h) — if A-C1 passes
  A-C3: L9 phase sweep (1h) — if A-C2 finds θ_opt ≠ 0

Phase B (Day 2-3, ~4h):
  B-C1: Coarse 28-layer phase sweep (4 phases, 28 layers)
  B-C2: Complex + contrastive combination
  B-C3: ||a||/||v|| polarity prediction

Phase C (Week 2):
  C-C1: Extend synthetic toy transformer with acceleration
  C-C2: Per-head complex α (if GQA allows)

--- CRITICAL DISPARITIES ---
1. D-C1: Acceleration triviality — if R²_a < 0.3, concept collapses
2. D-C2: Boundary artifacts at L0, L27 — may explain observed polarity
3. D-C3: Phase overparameterization — 28 extra params risk overfitting

--- ANALYSIS SELF-ASSESSMENT ---
Confidence in concept viability (overall): 6/10
  - R²_a ≥ 0.3: 50% (educated guess — velocity changes between layers may be small)
  - θ_opt ≠ 0 at L8: 35% (velocity might dominate dynamics)
  - L9 phase invertible: 45% (death-layer sign flip gives some evidence)
  - ||a||/||v|| predicts polarity: 55% (plausible geometric diagnostic)

What would increase confidence: The 10-minute acceleration R² measurement.
Until that runs, this is a mathematically beautiful idea with unknown empirical basis.

=======================================================================
END OF REPORT
=======================================================================
