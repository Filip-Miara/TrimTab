# Mechanistic Interpretability Check — RankAdaptation

## 1. Predictor Analysis

### TrajectoryTransformer Internal Analysis

**What does the TT actually learn?**
- Current understanding: The TT learns to predict v[l] = h[l+1] - h[l] from the sequence of hidden states h[0..L-1].
- Architecture: Input proj (3584→768) → 6×TransformerBlock (768, 8 heads) → Output proj (768→3584)
- 6 transformer layers with 8 heads each = 48 attention heads processing layer-position information

**Latent Space (d_model=768)**:
- The 768-dimensional space must compress 28 layers × 3584 dim = 100,352 input values
- Position embeddings indicate layer index (0-27), enabling the TT to distinguish layer-specific patterns
- The intermediate latents capture *cross-layer dependencies* — how the hidden state at layer 5 relates to layer 12, etc.

**Key Features Driving Predictions**:
Unknown from current analysis. The TT is treated as a black box. We know:
- Standard TT (all data, 7B): R²=0.855
- Contrastive TT_correct (7B): R²=0.832
- Contrastive TT_incorrect (7B): R²=0.829
- Standard TT (SmolLM2): R²=0.94

**Failure Modes**:
1. **Low hidden state norm** — if |h| is small, velocity prediction is unreliable (denominator in variance)
2. **Out-of-distribution inputs** — if steering has already changed the hidden state distribution, TT predictions degrade
3. **Non-reasoning tokens** — transition tokens ("Step", ":", numbers) may have different velocity patterns than reasoning tokens

**Causal vs Correlational**:
CRITICAL UNKNOWNS:
- Is the TT capturing *causal* dynamics of the transformer computation, or *correlational* patterns (e.g., hidden state norms tend to increase through layers)?
- The high R² could reflect the TT learning that h[l] and h[l+1] are similar (velocity is small relative to hidden state magnitude), not that it understands the transformer's computational structure.
- Proposed test: Compare TT prediction accuracy vs a baseline that predicts v[l] = mean(v across all layers). If the TT R² is only marginally better than the layer-mean baseline, it's learning trivial structure.

## 2. Representation Analysis

### Hidden State Manifold

**Intrinsic Dimensionality**:
- 7B hidden state: d=3584
- Effective rank (from SVD of trajectory matrix) is unknown
- Hypothesis: The effective dimensionality of trajectories is much lower (100-500), consistent with the TT's 768-dim bottleneck working well
- Test: Compute PCA of collected trajectories → cumulative variance explained → effective rank at 95% threshold

**Manifold Structure**:
- **Are correct/incorrect trajectories separable?** Indirect evidence:
  - Contrastive TTs achieve R²=0.83 on both correct and incorrect subsets separately → both subsets have learnable structure
  - But the TT_correct and TT_incorrect are *separate models* — we don't know if they learn the same or different dynamics
  - Critical test: Apply TT_correct to incorrect trajectories and vice versa. If R² drops significantly, the manifolds are separable. If R² stays high, they share the same dynamics.
- **Is there a "decision boundary" layer?** The trim-tab/death-layer pattern suggests some layers are critical decision points. Hypothesis: Layer 8-9 is where the model transitions from encoding to reasoning (this pattern appears in mechanistic interpretability literature as "early" vs "late" layers).

**Invariance Properties**:
- Velocity predictions vary with: prompt content, token position, model parameters
- Velocity predictions invariant to: batch size, attention mask changes (padding), token formatting

## 3. Synthetic Data Validation

### Test Design

**Synthetic Task**: Construct a minimal two-layer transformer where:
- Layer 0: Copy input to output (identity function)
- Layer 1: Compute h' = h + α_known * v_known where v_known is the vector pointing from the current hidden state to a target correct state
- Ground truth: The "correct" velocity v_known is known by construction
- The "incorrect" velocity is a random direction

**Goal**: Does the TT pipeline correctly identify that steering layer 1 with v_known improves accuracy?

**Expected Difficulty**: The TT should achieve high R² because the system is deterministic. But the *steering action* at layer 1 with v_known should produce a measurable improvement in logit alignment with the target correct answer.

**Validation Steps**:
1. Generate 1000 synthetic trajectories from this two-layer system
2. Train TT on 80% of trajectories, leaving 20% for validation
3. Apply steering with v_known at layer 1, measure alignment with correct target
4. Compare with steering at layer 0 (which should have no effect since it's identity)

**Predicted Result**: TT should achieve R² > 0.95 on synthetic data. Steering at layer 1 with v_known should show significant signal.

**If this synthetic test fails** (TT can't learn the known dynamics, or steering doesn't align with correct target):
- Then the real-model results are suspect — the entire pipeline may have a hidden bug
- Common failure: steering modifies the *input* to layer N, not the *output*, resulting in no effect

### Code Implementation Status
The synthetic test does not exist yet. It would be:
```python
class TwoLayerToyTransformer:
    def __init__(self, d_model):
        self.layer0 = nn.Identity()  # pass-through
        self.layer1 = nn.Linear(d_model, d_model)  # learnable
    def forward(self, x, steer_v=None, alpha=0):
        h0 = self.layer0(x)
        if steer_v is not None:
            h0 = h0 + alpha * steer_v  # steering at layer 0 output
        h1 = self.layer1(h0)
        return h0, h1
```

This is implementable in < 50 lines and would validate the entire steering concept.

## 4. Null Hypothesis Test

### H0-1: "The observed trim-tab effect (+20pp at L8) is due to random variation, not causal steering."
- **Falsification experiment**: Run the L8 sweep with N=500 problems (current N=100). The 95% CI for N=100 is ±9.8pp at 45% baseline. At N=500, the CI narrows to ±4.4pp. If L8 still shows +20pp with N=500, the null is rejected.
- **Verdict**: NOT_FALSIFIED yet — need larger N.

### H0-2: "The TT's high R² comes from learning trivial norm-growth patterns, not meaningful dynamics."
- **Falsification experiment**: Compare TT R² vs a baseline that predicts v[l] = (norm(h[l+1]) / norm(h[l]) - 1) * h[l] (velocity proportional to norm change). If TT R² is not significantly better than this norm-based baseline, H0-2 cannot be rejected.
- **Verdict**: NOT_FALSIFIED — this baseline has never been computed.

### H0-3: "Contrastive TT reduces to standard TT on 7B — v_c ≈ v_i, so v_c - v_i ≈ 0."
- **Falsification experiment**: Compute cosine similarity between v_c and v_i for the same input on 7B. If cos(v_c, v_i) ≈ 1.0, H0-3 is confirmed.
- **Verdict**: NOT_TESTED — cosine similarity between TT_correct and TT_incorrect hasn't been computed.

### H0-4: "Steering at L8 improves accuracy because it makes the model produce shorter (less error-prone) generations, not better reasoning."
- **Falsification experiment**: Compare generation length distribution between baseline and L8-steered answers. If L8 produces the same or longer generations but still higher accuracy, H0-4 is rejected.
- **Verdict**: NOT_TESTED — generation length hasn't been analyzed.

## Summary

| Area | Status | What's Missing |
|------|--------|----------------|
| TT interpretability | Black box analysis only | PCA of latent space, feature attribution, adversarial examples |
| Manifold analysis | Indirect evidence | Direct correct/incorrect trajectory comparison, effective rank |
| Synthetic validation | NOT IMPLEMENTED | Two-layer toy transformer test |
| Null hypothesis tests | Mostly untested | Norm-growth baseline, contrastive similarity, length analysis |
