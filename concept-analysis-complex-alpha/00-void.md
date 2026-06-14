# VOID Record — Complex α as Acceleration-Phase Steering

## Explicit Assumptions

1. **A1**: Hidden state dynamics are approximately linear across layers for small perturbations (validating the additive form h' = h + α·v).
2. **A2**: The velocity field v[l] = h[l+1] - h[l] contains the primary steering-relevant signal.
3. **A3**: A complex α = α₁ + i·α₂ can be given mathematical meaning in real vector spaces via an operator J with J² = -I.

## Implicit Assumptions

1. **IA1 (relational)**: The natural interpretation of "multiplication by i" in hidden state space is applying the layer's own transformation (attention+MLP) to the velocity — i.e., the *acceleration* or *curvature* of the trajectory.
2. **IA2 (structural)**: Acceleration a[l] = h[l+1] - 2·h[l] + h[l-1] is well-defined everywhere (requires hidden states at l-1, l, l+1 — valid for l=1..L-2, but boundary layers l=0 and l=L-1 need extrapolation).
3. **IA3 (temporal)**: The layer index l serves as a valid analogue of "time" for defining velocity and acceleration.
4. **IA4 (relational)**: The optimal steering phase θ differs between trim-tab layers (θ ≈ 0) and death layers (θ ≠ 0), and this phase difference explains their opposing effects.
5. **IA5 (potential)**: A complex α with per-layer phase enables *layer resonance* — constructive interference where trim-tab effects synchronize and death-layer effects cancel.

## Counter-Assumptions

1. **¬A1**: Hidden state dynamics are highly non-linear; the linear additive model h + α·v breaks down even at small α.
2. **¬IA1**: The layer transformation fₗ is not a good J operator because it's not norm-preserving and doesn't satisfy J² = -I.
3. **¬IA2**: Acceleration at the boundaries (l=0, l=L-1) requires extrapolation that introduces artifacts.
4. **¬IA4**: Death layers are genuinely harmful regardless of steering direction — they're not just "out of phase" trim-tabs.
5. **¬IA5**: The phase concept doesn't transfer from wave mechanics to transformer hidden states because there's no natural interference phenomenon.

## Bracket Statement

These assumptions are set aside for the analysis. The steering phase concept is a *formal analogy* — we are not claiming hidden states are complex numbers, but that the *steering dynamics* have a second-order structure (position + velocity + acceleration) that can be usefully parameterized as a complex-valued gain.
