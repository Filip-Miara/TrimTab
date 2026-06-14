# Concept Hierarchy — Complex α Acceleration-Phase Steering

## Atoms

| ID | Atom | Evidence |
|----|------|----------|
| A1 | **First-derivative velocity**: v[l] = h[l+1] - h[l] | Confirmed R²=0.85-0.94 in main project |
| A2 | **Second-derivative acceleration**: a[l] = h[l+1] - 2·h[l] + h[l-1] | Directly computable from same hidden states |
| A3 | **Boundary condition**: a[0] and a[L-1] need forward/backward extrapolation (a[0] := a[1], a[L-1] := a[L-2]) | Architectural constraint |
| A4 | **Real α₁**: Coefficient on velocity — existing steering gain | Main project α ∈ [0, 0.3] |
| A5 | **Imaginary α₂**: Coefficient on acceleration — new steering dimension | Proposed |
| A6 | **Steering phase θ**: Polar form α = r·e^(iθ) where tan θ = α₂/α₁ | Proposed |
| A7 | **Steering magnitude r**: √(α₁² + α₂²) — total perturbation strength | Proposed |
| A8 | **Layer-specific phase signature**: Each layer l has optimal (r_l, θ_l) | Hypothesis |
| A9 | **Layer resonance**: θ_l aligned across layers → constructive interference | Hypothesis |
| A10 | **J operator**: Abstract linear map with J² = -I on the hidden state manifold | Mathematical structure |

## Composites

| ID | Composition | Definition |
|----|------------|------------|
| C1 | A1 + A2 + A4 + A5 | **Complex steering**: h' = h + α₁·v + α₂·a |
| C2 | A6 + A7 + C1 | **Phase-amplitude steering**: h' = h + r(cosθ·v + sinθ·a) |
| C3 | A8 + C2 | **Per-layer phase steering**: (r_l, θ_l) per layer |
| C4 | A9 + C3 | **Resonant steering**: θ_l aligned for constructive interference |
| C5 | A10 + C1 | **Geometric complex structure**: true complex multiplication on h |

## Junctions

| ID | Type | Source → Target | Meaning |
|----|------|----------------|---------|
| J1 | compositional | A1 + A2 → v, a are independent directions | v and a span a 2D subspace of steering directions |
| J2 | temporal | A1 → A2 (acceleration = change in velocity) | a is the discrete derivative of v w.r.t. layer |
| J3 | constraint | A3 → A2 (boundary limits) | Acceleration undefined at layer 0 and L-1 |
| J4 | dependency | A4 + A5 + A6 → r = √(α₁²+α₂²), θ = atan2(α₂, α₁) | Polar ↔ Cartesian equivalence |
| J5 | hierarchical | A8 → C3 (per-layer phase defines steering) | Phase signature is a property of the layer, not the input |
| J6 | synergistic | A8 + A9 → C4 (alignment → resonance) | Aligned phases produce stronger effect than sum of individuals |
| J7 | antagonistic | θ_l vs θ_k for death/trim layers | Opposing phases produce cancellation |
| J8 | binding | A10 → valid J on h-space | Without J, complex analogy is formal only |

## Hallucinatory Pre-Seeds

| Atom | Ideal form |
|------|------------|
| A2 | Acceleration perfectly predicts the residual error of velocity-only steering for all layers and all inputs |
| A6 | θ is a *learned* function of the hidden state itself: θ(h[l]) — phase adapts to input |
| A9 | Resonant steering produces superlinear accuracy gains (2 trim-tabs in phase → +40pp, not +20pp) |
| A10 | The transformer's attention mechanism naturally satisfies J² ≈ -I on the subspace of "reasoning tokens" |
