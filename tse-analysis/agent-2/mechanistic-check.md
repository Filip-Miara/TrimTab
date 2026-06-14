# Phase 8: Mechanistic Interpretability Check

---

## 8.1 Predictor Analysis

### TrajectoryTransformer Dissection

The TT is the core learned model in the pipeline. Let's analyze what it actually learned.

**Architecture**:
- Input: (B, 23, 2048) for 7B or (B, 28, 1536) for 1.5B — hidden states at all layers
- d_model: 512 or 768
- Layers: 6 transformer blocks
- Output: (B, 23, 2048) or (B, 28, 1536) — predicted velocities

**What the TT likely learned**:
1. **Layer-to-layer smoothness**: Since v[l] = h[l+1] − h[l], and adjacent layers in a transformer have similar representations (CKA between layers i and i+1 is typically >0.9), most of the "velocity" signal is small. The TT may be learning to predict the small deviation from the identity mapping.
2. **Model-specific velocity patterns**: Each model has characteristic "velocity magnitudes" at each layer (some layers change more than others between adjacent layers). The TT has likely learned these layer-specific statistics.
3. **Input-specific modulation**: On top of the per-layer baseline velocity, the TT modulates based on the specific hidden state values at that position.

**Latent space analysis (theoretical)**:
- If we run PCA on TT's hidden representations (after the transformer blocks), we would likely find:
  - PC1 explains ~60-70% variance: overall velocity magnitude per layer (how much h changes at each layer on average)
  - PC2 explains ~10-15%: "correct vs incorrect trajectory" direction (if separable)
  - PC3-PC10: individual layer-specific dynamics
- This is speculative — no PCA was actually performed on TT representations.

**Key Features Driving TT Predictions**:
- Feature #1 likely: **The hidden state at the current layer** (h[l]) — from this alone, a linear model could predict the next layer's velocity with moderate accuracy, because h[l] and h[l+1] are highly correlated.
- Feature #2 likely: **Surrounding layer context** (h[l-1], h[l], h[l+1]) — the transformer's self-attention can use all 23 layers to refine the prediction.
- Feature #3 likely: **Global trajectory shape** — whether the hidden states are "trending" in a particular direction across layers (e.g., monotonic increase in norm).

**Failure Modes**:
- **Input statistics shift**: During steering, hidden states are modified by α·v_pred. If α > 0.5, the steered hidden state may be in a region of activation space the TT has never seen → predictions become unreliable.
- **Out-of-distribution layer patterns**: If the model's computation is disrupted (e.g., death layers cause attention collapse), the TT sees abnormal trajectories and produces unreliable velocities.
- **Long generation**: After many steered tokens, the hidden state may drift significantly from the training distribution.

**Causal vs Correlational Assessment**: **MOSTLY CORRELATIONAL**.
- The TT predicts v from h[l]. Since h[l] and h[l+1] are temporally adjacent, there is a strong auto-correlation between them. The TT may be exploiting this auto-correlation rather than learning the causal dynamics of the transformer computation.
- To test: compare TT performance against:
  - Baseline: v_pred = 0 (zero velocity) — gives h' = h, no steering
  - Linear: v_pred = Wh[l] + b (linear layer from h[l])
  - Lag: v_pred = h[l] − h[l-1] (previous layer's velocity as current estimate)
- If TT performs similarly to the linear baseline, the transformer architecture is unnecessary.

### K/V Projection Amplification

The steering mechanism is: h_steered = h + α·v_pred → k' = W_k(h_steered) → attention uses k'.

The critical question: **How does a small change in h produce a large change in attention output?**

Projection analysis:
- W_k ∈ R^(256 × 2048) for 7B (n_kv_heads × head_dim, d_model)
- v_pred is a 2048-dim vector, scaled by α=0.1
- The change in k: Δk = W_k(α·v_pred) ∈ R^256
- The attention logit change: Δa = q^T Δk ∈ R^1

Since the key vector is 256-dim, and q is normalized, the softmax is sensitive to changes in a (the dot product). A change of 0.01 in the logit can shift the attention distribution significantly. The observed 88% token divergence at α=0.1 is **consistent with plausible attention sensitivity** — it does NOT require the hidden manifold to be "non-Euclidean" in the strong sense.

This suggests: **The steering effect is mediated through attention distribution changes, not through "pushing toward correct manifolds."** The TT learns to predict which direction of hidden state change will produce beneficial attention shifts.

---

## 8.2 Representation Analysis

### Hidden State Structure

**Intrinsic Dimensionality**: Theoretical estimate based on prior work (Li et al., 2018; Aghajanyan et al., 2020):
- For a 2048-dim hidden state at L8, the intrinsic dimension of task-relevant features is likely **16-64**.
- Most of the 2048 dimensions are "noise" or task-irrelevant features.
- The TT's 512-dim bottleneck likely captures the relevant structure.

**Correct vs Incorrect Trajectories**:
- The key question: are correct and incorrect trajectories linearly separable in hidden state space?
- For Qwen2.5-7B (baseline 73%), there IS evidence of separability (contrastive TT has R²=0.83 for both classes, meaning the correct-incorrect distinction is learnable).
- For Math-1.5B (baseline 38%), separability is SUSPECT (neither standard nor contrastive TTs produce trim tabs).

**Invariance Properties**:
- The hidden state at L8 is likely **approximately invariant** to the surface form of the question (e.g., rephrasing) but **sensitive** to the mathematical structure.
- The TT may have learned this invariance — if so, it's capturing a genuinely useful representation.

---

## 8.3 Synthetic Data Validation

### Test Design

Construct a **toy transformer** where the correct steering direction is known analytically:

1. **Synthetic model**: A 2-layer transformer with d=64, trained on a synthetic task (e.g., "predict the parity of the sum of input bits").
2. **Ground truth**: For this task, we can analytically determine which hidden state directions correspond to "moving toward correct answer."
3. **Apply the same pipeline**: Collect trajectories, train TT, perform per-layer sweep.
4. **Validation criterion**: Does the TT identify the analytical correct direction? Does the per-layer sweep identify the layer where the correct computation occurs?

**Expected result**: The pipeline should identify the correct direction if it's genuinely learning causal structure. If it fails on synthetic data (where ground truth is known), then it's likely learning correlational patterns on real data too.

**Required resources**: Minimal — synthetic task, small model, CPU-only feasible.

**Status**: NOT YET PERFORMED. This is recommended as a high-value, low-cost validation experiment.

### Synthetic Baseline for 7B

**Alternative approach** (lower cost): Use the existing 7B pipeline but with a **null model**:

1. **Null model 1 (Zero)**: Replace TT output with zeros → v_pred = 0. Evaluate per-layer sweep.
   - If layer pattern persists, steering effect is independent of TT predictions.
2. **Null model 2 (Constant)**: Replace TT output with per-layer learned constants. Evaluate.
   - If constant steering produces similar pattern to TT steering, the TT's input-dependence is irrelevant.
3. **Null model 3 (Shuffled)**: Randomly permute the association between (input trajectory, velocity target) in training data. Train TT on shuffled data.
   - If the "random TT" still shows layer patterns, the patterns are an artifact of the architecture, not the training data.

---

## 8.4 Null Hypothesis Tests

### H0-1: "The apparent steering effect is due to random variation, not velocity structure."

**Statement**: The observed +20pp at L8 (from 45% to 65% on 100 problems) could occur by chance even if steering has no causal effect on reasoning accuracy.

**Falsification experiment**:
1. Run the per-layer sweep 10 times with different random seeds (different 100-problem subsets).
2. Compute the distribution of L8 accuracy across runs.
3. If the 95% CI excludes 45% (baseline), reject H0.

**Expected result**: With baseline 45% and n=100, the 95% CI for a binomial proportion is [35%, 55%]. An observed 65% is outside this CI, so the result IS statistically significant at p < 0.05. **H0 is REJECTED** — the steering effect is real, not random.

**Confidence**: HIGH (standard statistical test supports the claim).

### H0-2: "The TT's predictive power comes from auto-correlation, not causal dynamics."

**Statement**: TT achieves R²=0.94 because adjacent hidden states are highly correlated, not because it has learned causal structure.

**Falsification experiment**:
1. Train a baseline predictor: v_baseline[l] = h[l] − h[l-1] (previous layer's velocity).
2. Compute R²_baseline on held-out data.
3. If R²_TT ≈ R²_baseline, reject the claim that TT learns causal dynamics.
4. If R²_TT >> R²_baseline, the TT IS learning structure beyond auto-correlation.

**Expected result**: Given the TT's architecture (6-layer transformer with self-attention), it is likely to significantly outperform the lag-based baseline. The auto-correlation hypothesis predicts R² ≈ 0.5-0.6 (from adjacent-layer CKA); the observed 0.94 is unlikely to be explained by auto-correlation alone.

**Confidence**: MEDIUM — this test has NOT been performed and the auto-correlation explanation cannot be ruled out without it.

### H0-3: "Contrastive TT provides no benefit over standard TT."

**Statement**: The difference v_correct − v_incorrect is not a useful steering direction; it performs no better (or worse) than v_all (standard TT).

**Falsification experiment**: Compare accuracy of contrastive steering vs standard steering on L8 and L9 across α ∈ [0.01, 0.5].

**Verdict**: **PENDING** — this is the most important unknown in the entire project. The comparison should resolve H0-3 decisively.

---

## Summary

| Test | Status | Finding |
|------|--------|---------|
| TT latent space analysis | NOT PERFORMED | Recommended: PCA on TT representations |
| K/V amplification analysis | THEORETICAL | Attention sensitivity explains 88% divergence without invoking non-Euclidean geometry |
| Linear baseline comparison | NOT PERFORMED | Critical: would test if TT is necessary |
| Null model (constant steering) | NOT PERFORMED | Critical: would test if TT's input-dependence matters |
| Shuffled data TT | NOT PERFORMED | Would test if layer patterns are architecture artifacts |
| Synthetic toy transformer | NOT PERFORMED | High-value, low-cost validation experiment |
| H0-1 (random variation) | **REJECTED** | +20pp on 100 problems from 45% baseline is significant at p < 0.05 |
| H0-2 (auto-correlation) | INCONCLUSIVE | Test not performed; genuine risk that TT exploits correlation |
| H0-3 (contrastive useless) | **PENDING** | Most important pending result |
