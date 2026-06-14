# Lens Cascade — Complex α Acceleration-Phase Steering

---

## Lens 1: ANALOGICAL

### Structural
- **Classical mechanics**: Position x(t), velocity v = dx/dt, acceleration a = d²x/dt². Our layer index l maps to time t. Current steering only uses velocity (first-order). Acceleration adds the *force* analogy — changing the trajectory's curvature, not just its direction.
- **AC circuits**: Impedance Z = R + iX (resistance + reactance). Real power = I²R (dissipated), reactive power = I²X (stored). Real α = resistive steering (dissipates along v), imaginary α = reactive steering (stores energy in perpendicular direction). The phase θ is the power factor.
- **Quantum mechanics**: The wavefunction ψ evolves under the Schrödinger equation iℏ·∂ψ/∂t = Ĥψ. The factor i is fundamental — it couples real and imaginary parts, creating oscillatory dynamics. Acceleration in hidden states may be the classical analogue of quantum oscillation.

### Relational
- **PID control**: Proportional (P) = velocity steering (current error), Integral (I) = accumulated velocity (position correction), Derivative (D) = acceleration (predicted error). Complex α unifies P (real) and D (imaginary) gains.
- **Signal processing**: A complex coefficient acts as a *filter* — selecting which frequency components of the hidden state trajectory to amplify. θ selects the phase of the response.

### Potential
- **Lock-in amplification**: If we know the "resonant frequency" of a trim-tab layer, we can tune θ to selectively amplify that layer's dynamics while suppressing others — spectral selectivity.

### Blind Spot Alert
Physical analogies break down because transformer layers are discrete and non-homogeneous. There's no continuous "time" — only 28 discrete steps.

---

## Lens 2: DIALECTICAL

### Thesis
Complex α subsumes real α as a special case (θ=0). Adding the acceleration term provides strictly more degrees of freedom. Either:
- The optimal θ ≠ 0 for some layer → complex α outperforms real α
- The optimal θ = 0 for all layers → complex α reduces to real α (no worse)

Therefore complex α is *always at least as good* as real α.

### Antithesis
The acceleration term adds noise. If a[l] is not structured (R² for acceleration prediction is lower than for velocity), then α₂·a just injects random perturbations. Additionally:
- Boundary effects at l=0, l=L-1 introduce artifacts
- Two parameters (r, θ) per layer doubles the sweep space → harder to optimize
- The phase concept may not have a natural interpretation, leading to overfitting

### Synthesis
Complex α is Pareto-superior to real α *if* the acceleration has predictable structure. The test: compute the acceleration field from existing trajectory data and measure its R² when predicted by a TT variant. If R²_acceleration ≥ R²_velocity × 0.5, the acceleration signal is strong enough to be useful.

---

## Lens 3: BLENDING

### Structural Blends
1. **Acceleration + Contrastive TT**: Train TT to predict acceleration a[l] on correct and incorrect trajectories separately. Compute a_correct - a_incorrect = contrastive acceleration. Apply both contrastive velocity AND contrastive acceleration → full second-order normative steering.
2. **Phase + Per-layer α**: θ_l becomes the 29th parameter per layer (alongside r), enabling the 2D α-θ sweep.

### Relational Blends
1. **Phase + Confidence gating**: θ as a function of reading head uncertainty. When uncertain, rotate the steering phase toward the imaginary axis (curve-bending rather than pushing) — qualitatively different intervention for uncertain tokens.
2. **Acceleration + First-step steering**: The first token has no previous velocity, but acceleration can be defined using the prompt's last two hidden states. First-step steering with acceleration might be *more reliable* than first-step steering with velocity (which requires prompt-trained TT with lower R²).

---

## Lens 4: SYSTEMS

### Feedback Structure
- **Velocity steering only**: δh = α·v. This is a *first-order* perturbation. The effect on downstream layers is through the modified K/V entries only.
- **Acceleration + velocity**: δh = α₁·v + α₂·a. This modifies both the state AND its rate of change — like changing both position and momentum. Downstream layers see a "smoother" perturbation because acceleration compensates for discontinuities in the velocity field.

### Leverage Point
The **acceleration-to-velocity ratio** a/l · v/l at each layer may predict whether that layer is a trim-tab (low ratio — velocity dominates) or death layer (high ratio — acceleration dominates). This would give a **zero-shot polarity predictor** from the trajectory data alone, without running steering experiments.

---

## Lens 5: ABDUCTIVE

### What explains the L8 vs L9 difference if α is complex?

**Observation**: L8 steering with α = +0.1 increases accuracy by +20pp. L9 steering with α = +0.1 *decreases* accuracy by -23pp.

**Hypothesis (complex α)**: L8 and L9 have opposite optimal phases.
- L8: θ_opt ≈ 0 (steer along velocity = good)
- L9: θ_opt ≈ π (steer along negative velocity = good) OR θ_opt ≈ π/2 (steer along acceleration only = good, steer along velocity = bad)

If L9 is at phase θ = π (inverted polaritiy), then α = +0.1 at L9 is equivalent to α = -0.1 at a trim-tab. The death-layer sign flip experiment tests this.

If L9 is at phase θ = π/2 (pure acceleration), then steering with any velocity component is harmful, but steering with pure acceleration (α₁ = 0, α₂ > 0) might help. This would be a NEW result not tested by simple sign flip.

### Blind Spot Alert
Abduction here is seductive — it's easy to retroactively explain L9 with phase, but the hypothesis may not be predictive. The critical test is: can we *predict* which layers benefit from which θ *before* running the experiment?

---

## Lens 6: TRAJECTORY

### Evolution of the steering concept
1. Single α, all layers → failure (death layers dominate)
2. Single α, per layer → L8 trim-tab discovered
3. Per-layer α vector → exploring multi-layer combinations
4. **Complex α per layer (proposed)** → adding phase dimension

Each step adds a degree of freedom that addresses a specific failure mode:
- Step 2: Per-layer selectivity fixed "death layers dominate"
- Step 4: Per-layer phase fixes "L9 is anti-aligned with velocity"

### Extrapolation
If complex per-layer α works, the next step would be **complex per-head α** — 28 layers × 28 heads × (r, θ) = 1568 parameters. At that point, α optimization requires gradient-based methods, not sweeps.

---

## Lens 7: METACOGNITIVE

### Blind Spots in This Analysis
1. **Is acceleration truly the natural "imaginary" direction?** The complex analogy is one of several possible interpretations. Alternative: rotation through the attention logit space, not the hidden state space.
2. **Are we reifying the steering phase?** θ is a convenient parameterization, but it might not correspond to any physical property of the transformer computation.
3. **What about third-order effects?** Jerk (third derivative of h w.r.t. layer) might matter too. Where do we stop?
4. **Boundary condition sensitivity**: The extrapolation at l=0 and l=L-1 might dominate the signal for those layers, which happen to be L0 (neutral) and L27+ (death layers). The boundary effect might be the *cause* of their polarity, not a coincidence.

---

## Lens 8: INSPIRATION

### Foreign-Domain Structures
- **Hamiltonian mechanics**: The total energy H = T + V where kinetic T = ½mv² and potential V. The acceleration a = -∇V/m. Steering along acceleration is steering against the potential gradient — pushing the system *uphill* toward higher-energy (but possibly more correct) states.
- **Complex step differentiation**: In numerical analysis, complex step differentiation uses f(x + ih) to compute f'(x) without cancellation errors. The "imaginary" steering direction might give numerically more stable gradient information about the correctness landscape.
- **Spiral dynamics**: In optimization, the triple (position, momentum, and their coupling) creates spiral convergence paths. Complex α with optimal θ might create a *spiral ascent* toward the correct reasoning manifold.

---

## Lens 9: ADVERSARIAL

### Attack Surface
If an adversary controls θ, they could set θ = π/2 (pure acceleration) at a trim-tab layer, steering the model *sideways* instead of forward — a form of "lateral thinking" that degrades accuracy without obvious perturbation (hidden state magnitude is preserved, but direction is rotated).

### Failure Mode
The worst case for complex α: it doubles the search space and finds spurious correlations. With 28 × 2 = 56 parameters, overfitting to 100-problem evaluations becomes a real risk.

---

## Lens 10: PARADOXICAL

### The Acceleration Paradox
If acceleration a[l] = v[l] - v[l-1] (change in velocity), then steering with acceleration is equivalent to steering with *the difference between consecutive velocities*. But the TT already learns v[l] from the full trajectory context. If the TT is good enough (R²=0.855), it implicitly captures velocity differences. Adding explicit acceleration steering might be *redundant* — the TT already models second-order effects through its transformer architecture (self-attention across layers).

### The Phase Singularity Paradox
At r = 0, θ is undefined. The "no steering" configuration is a singularity in the phase representation. This means smooth optimization (e.g., gradient descent on r and θ) has a degenerate point at the origin, which is where all steering experiments start.

---

## Convergence

**HIGH confidence** (≥5 lenses agree):
- Complex α provides strictly more degrees of freedom than real α (Lens 1, 2, 3, 6, 8)
- Acceleration is computable from existing data (all lenses)
- The phase interpretation has physical meaning (Lens 1, 5, 8)

**CONTESTED**:
- Whether acceleration has sufficient structure for useful steering (Lenses 5 vs 10)
- Whether the phase concept maps to real transformer computation (Lenses 7 vs 1)
