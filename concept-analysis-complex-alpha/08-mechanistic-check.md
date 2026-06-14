# Mechanistic Check — Complex α (Cross-Pollinated)

## Predictor Analysis for Acceleration TT

### What would an acceleration TT learn?
If we train a TT to predict a[l] = h[l+1] - 2h[l] + h[l-1] instead of v[l]:
- **If R²_a > 0.5**: Acceleration has meaningful structure. Complex α has empirical foundation.
- **If 0.1 < R²_a < 0.5**: Weak acceleration signal. May contribute marginally at high α₂.
- **If R²_a < 0.1**: Acceleration is noise. Complex α ≈ real α. Drop the idea.

### Null Baseline for Acceleration (cross-pollinated from H0-2)
- **H0-C1**: "Acceleration is trivial — a[l] ≈ 0 everywhere because velocity changes little between adjacent layers."
- **Falsification**: Compute mean(a) across all layers and tokens. If ||mean(a)|| >> 0, H0-C1 is rejected.
- **Simpler test**: Compute std(v[l] - v[l-1]) across layers. If this is << std(v[l]), acceleration is small.

## Synthetic Validation (cross-pollinated from main synthesis C3)

Extend the two-layer toy transformer to test complex α:

```python
class ToyTransformerComplex:
    def __init__(self, d_model):
        self.layer0 = nn.Identity()
        self.layer1 = nn.Linear(d_model, d_model)
        self.v_known = ...  # ground truth velocity
        self.a_known = ...  # ground truth acceleration

    def forward(self, x, r=0, theta=0):
        h0 = self.layer0(x)
        v = self.v_known
        a = self.a_known
        # complex steering: h' = h + r·(cosθ·v + sinθ·a)
        h0_steered = h0 + r * (cos(theta) * v + sin(theta) * a)
        h1 = self.layer1(h0_steered)
        return h0, h1
```

**Goal**: Verify that:
1. TT can predict v_known (R² > 0.95, already tested in main C3)
2. TT_can predict a_known (R² > 0.95) — tests acceleration structure
3. Steering at θ=0 uses only v, θ=π/2 uses only a
4. The optimal θ is the one that aligns with the target manifold's phase

## Representation Analysis (cross-pollinated)

### Key Questions for Existing Data
1. **std(v[l]) across layers**: Is velocity constant across layers (low std) or variable (high std)?
   - Constant velocity → acceleration ≈ 0 → complex α fails
   - Variable velocity → acceleration carries information → complex α viable
2. **PCA of (v, a) pairs**: Do velocity and acceleration span different subspaces?
   - If v and a are parallel (cos(v, a) ≈ 1) → they carry same direction info
   - If v and a are orthogonal (cos(v, a) ≈ 0) → they carry independent direction info
3. **Cross-model comparison**: Does Math-1.5B (no trim tabs) have lower R²_a than 7B (has trim tabs)?
   - If yes → acceleration structure predicts steerability — new diagnostic!
