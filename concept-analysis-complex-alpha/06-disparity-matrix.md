# Disparity Matrix — Complex α (Cross-Pollinated)

## Disparities

### D-C1: Acceleration Triviality (CRITICAL — from mechanistic check cross-pollination)
- **Concepts**: A2 (acceleration) × H0-2 (norm-growth null from main synthesis)
- **Conflict**: If velocity is largely norm-growth (trivial), then acceleration is norm-growth-curvature (doubly trivial). a[l] ≈ 0 if ||h[l]|| grows linearly. The acceleration field may be near-zero everywhere.
- **Severity**: FUNDAMENTAL — if true, α₂·a = 0 regardless of α₂, making complex α collapse to real α.
- **Resolution**: SYNTHESIS — Empirically measure std(v[l]) across layers from existing data. If std(v) >> 0, acceleration has non-zero signal. Resolution: run Acceleration R² measurement (Candidate #2).

### D-C2: Boundary Artifacts at L0, L27 (operational)
- **Concepts**: A3 (boundary extrapolation) × C1 (complex steering at all layers)
- **Conflict**: Acceleration undefined at l=0 and l=L-1 without extrapolation. These happen to be L0 (neutral layer) and L27+ (death layers). The boundary could *cause* the observed polarity.
- **Severity**: STRUCTURAL (resolvable)
- **Resolution**: SUBSTITUTION — Use forward-only acceleration at boundaries: a[0] = h[2] - 2h[1] + h[0] (same formula, just shifts window). Compare boundary vs interior phase behavior to isolate artifacts.

### D-C3: Phase Overparameterization (potential — from H0-1 cross-pollination)
- **Concepts**: A6 (phase θ) × main synthesis finding: N=100 insufficient for statistical power
- **Conflict**: Adding 28 more parameters (θ per layer) to the sweep space demands exponentially more evaluations. With 100 problems, overfitting to θ is almost guaranteed.
- **Severity**: STRUCTURAL (resolvable)
- **Resolution**: BOUNDING — Only sweep θ at 3 layers (L8, L2, L9) initially, not all 28. Restrict to coarse θ grid (9 values). Validate with main synthesis B1 (N=500) before expanding.

### D-C4: Phase ≠ Physical Property (assumption clash)
- **Concepts**: A6 (phase) × Lens 7 (metacognitive: are we reifying?)
- **Conflict**: θ is a convenient parameterization but may not correspond to any real property of transformer computation.
- **Severity**: PHILOSOPHICAL
- **Resolution**: BOUNDING — Accept that θ is a formal parameter whose utility is measured by predictive power, not physical correspondence.

### D-C5: Complex α × Per-Layer α Redundancy (abstract)
- **Concepts**: Complex α per layer × main synthesis per-layer α vector
- **Conflict**: If each layer already has its own real α, adding θ is redundant — the same effect could be achieved by redefining "velocity" to include acceleration via a different TT.
- **Resolution**: ABSTRACTION — Complex α is not redundant because it controls the *relationship* between v and a (through θ), not just their independent magnitudes.
