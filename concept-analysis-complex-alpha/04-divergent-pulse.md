# Divergent Pulse — Complex α (Cross-Pollinated)

## Seed Expansion from Prior Synthesis

| Prior Finding | Cross-Pollination to Complex α |
|--------------|--------------------------------|
| **Norm-growth null (H0-2)** | Acceleration's null baseline is even lower (predict a[l] = 0). If velocity R² is inflated by norm growth, acceleration R² will be near-zero → complex α collapses to real α. **Counter-seed**: if velocity is NOT constant (stdev(v) > 0 across layers), acceleration carries independent signal. |
| **Contrastive cosine (A3)** | The optimal phase θ may differ between correct and incorrect trajectories. cos(v_c, v_i) might be high at θ=0 but low at θ=π/2 → contrastive acceleration > contrastive velocity. |
| **GDN recurrent state (RECOMB-5)** | GDN's state update S += β·(k⊗v) has a natural complex extension: S += e^(iφ)·β·(k⊗v) where φ is a phase on the outer product. The recurrent state acts as a *complex accumulator*. |
| **Synthetic validation (C3)** | The toy transformer can be extended: give it a known acceleration a_known (in addition to v_known). Verify complex α recovers both. |

## Mutation Operators (focused on complex α)

### M1: SUBSTITUTE
| Original | Substitution | Quality |
|----------|-------------|---------|
| a = h[l+1] - 2h[l] + h[l-1] | a = h[l+2] - 2h[l+1] + h[l] (forward acceleration) | 4/5 — avoids boundary issues |
| Layer index l as "time" | Token position t as "time" → a[t] = h[t+1] - 2h[t] + h[t-1] across tokens | 4/5 — token-level acceleration |
| Real α₁ = α₁ < 0 | Phase rotation ONLY: α = i·α₂ (pure acceleration, no velocity) | 5/5 — pure bending |
| Polar (r, θ) | Complex exponential: α = r·exp(iθ) | 5/5 — standard notation |

### M2: INVERT
| Original | Inversion | Quality |
|----------|-----------|---------|
| θ = π at L9 (inverted polarity) | θ = -π/2 at L9 (pure negative acceleration) | 5/5 — tests all 4 quadrants |
| r(θ) constant | r(θ) varies with θ → elliptical steering | 4/5 — non-circular steering |

### M3: SCALE
- θ ∈ {0, π/6, π/4, π/3, π/2, 2π/3, 3π/4, 5π/6, π} — full 180° sweep
- r ∈ {0.01, 0.03, 0.05, 0.1, 0.3} — explore magnitude alongside phase
- 9 × 5 = 45 combinations per layer (manageable)

### M4: MERGE (cross-pollinated from main synthesis)
| Merged | Result |
|--------|--------|
| Complex α + Contrastive direction | Phase rotates the contrastive difference: h' = h + r·e^(iθ)·(v_c - v_i) |
| Complex α + Per-layer α vector | Per-layer (r_l, θ_l) = 56 parameters instead of 28 |
| Complex α + Adaptive α(t) | θ(t) schedule: rotate phase as generation progresses |
| Complex α + Confidence gate | θ = f(ppl_pred): when uncertain, rotate toward acceleration (curve-bending) |

### M5: SPLIT (cross-pollinated from emergent EM-1)
- Per-head complex α within L8: 28 heads × (r_h, θ_h) = 56 parameters per layer
- K-phase vs V-phase: separate θ_k and θ_v for key and value projections

## Forced Collisions (cross-pollinated)

**Collision 1: Complex α × GDN recurrent state**
- *Idea*: GDN's state S = sum(β_t · (k_t ⊗ v_t)) over tokens. Replace β_t with β_t · e^(i·φ_t) where φ_t is token-specific phase. This makes the recurrent state a *complex matrix*.
- *Paradox*: GDN states are real matrices. "Complex" GDN would double state size.

**Collision 2: Complex α × Synthetic toy transformer**
- Extend the two-layer toy (from main synthesis C3): give layer 1 a known acceleration a_known (in addition to v_known). The correct steering direction is α₁·v_known + α₂·a_known with α₁, α₂ known by construction. Verify the pipeline recovers them.

**Collision 3: Complex α × Capability threshold**
- Below-threshold models might have velocity structure that's random (R²_v ≈ 0) but *acceleration* structure that's preserved (R²_a > 0). Complex α could steer below-threshold models via acceleration alone (θ = π/2).
