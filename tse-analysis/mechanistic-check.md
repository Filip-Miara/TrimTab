# Phase 8: Mechanistic Interpretability Check

---

## Predictor Dissection

### Latent Space Analysis (of TT)

**Current state**: Unknown — no PCA/UMAP has been run on TT's d_model=768 internal representations.

**Predicted structure** (based on lens cascade):
- **High-dimensional manifold**: The 768-dim latent space likely encodes 28×3584→768 compression, creating a "velocity lexicon" organized by direction similarity
- **Feature clusters**: Expect clusters corresponding to: (a) early-layer high-magnitude velocities, (b) mid-layer refinement velocities, (c) late-layer low-magnitude velocities
- **Anisotropy**: Some dimensions likely dominate (high variance in velocity magnitude direction), while others are near-zero (noise)

**Recommended analysis**:
1. Collect TT hidden states on 1000 validation trajectories
2. Run PCA on the 768-dim representations
3. Plot explained variance — expect 80% in <100 dimensions
4. Color by layer index — expect trajectory structure along layer dimension

### Feature Attribution

**Question**: Which of the 3584 input features drive velocity prediction most?

**Hypothesis**: Velocity prediction relies heavily on a subset of features that correspond to:
- **Attention head outputs**: ~12% of features may drive 80% of prediction
- **MLP intermediate activations**: Possibly lower predictive value

**Recommended analysis**:
1. Integrated gradients: attribute velocity prediction to input features
2. Identify top-100 features by attribution magnitude
3. Cross-reference with mechanistic roles in Qwen (which attention heads do these correspond to?)
4. **Expectation**: Top features concentrate in specific Qwen attention heads (e.g., heads 5, 12, 23 of early layers)

### Failure Mode Analysis

**When does TT fail worst?**

Based on current data (R² average 0.848, but likely high variance):
- **Long trajectories** (>2000 tokens): Likely higher error due to compounding distribution shift
- **Rare tokens/patterns**: The 90K GSM8K trajectories may miss diverse linguistic patterns
- **Edge case reasoning**: Multi-step math problems with unusual intermediate steps

**Recommended analysis**:
1. Compute per-trajectory R²
2. Identify bottom 5% of trajectories (R² < 0.65)
3. Characterize them: input length, topic, token types, velocity magnitude
4. **Key question**: Are bad predictions random or systematic?

### Causal vs Correlational

**Critical assessment**: The TT may be learning correlational patterns between hidden states and their deltas, not causal velocity dynamics.

| Evidence for Causal | Evidence for Correlational |
|---------------------|---------------------------|
| Velocity = delta between consecutive layers — this IS a causal quantity by definition | TT trained offline on frozen model — doesn't see causal effects of its own predictions |
| R²=0.85 suggests genuine predictive structure | Distribution shift (AWQ 0.85→0.45) suggests reliance on surface features |
| Cosine loss improves direction | Simple feedforward may match the smoothness bias |

**Verdict**: The TT likely captures MIXED causal+correlational patterns. It may have learned true velocity dynamics for common patterns (high signal) while relying on correlational shortcuts for rare patterns.

**Test**: Add adversarial noise to input hidden states. If predictions degrade catastrophically, TT relies on correlations. If graceful, TT captures causal dynamics.

---

## Representation Analysis

### Intrinsic Dimensionality of Velocity Targets

| Metric | Expected Value | How to Measure |
|--------|---------------|----------------|
| PCA explained variance (90%) | <200 dimensions | PCA on flattened velocity targets [90K×28, 3584] |
| Participation ratio | <0.10 | PR = (∑λ_i)² / ∑λ_i² where λ are eigenvalues |
| Correlation dimension | <50 | Grassberger-Procaccia algorithm |

**Prediction**: The 3584-dim velocity space is likely **highly redundant**. Attention head outputs within a layer are correlated; layer-to-layer velocities are smooth → low intrinsic dimensionality (~50-200).

**If confirmed**: This is THE key insight. It means:
1. PCA compression (256 dims) will lose <5% information while reducing output by 14×
2. The 48M model capacity is far more effective in compressed space
3. The noise component is largely in the high-variance tail → PCA acts as denoiser

### Manifold Structure

**Correct vs incorrect velocity predictions**:

| Property | Expectation | Test |
|----------|-------------|------|
| Separability | Partial — high-error predictions may cluster | t-SNE of velocity predictions colored by error |
| Curvature | Likely low (velocity is locally smooth) | Hessian of predicted velocity w.r.t. layer index |

### Invariance Properties

**What transformations leave velocity invariant?**
1. **Global translation**: Adding constant to hidden state → no change in velocity (delta)
2. **Layer reordering**: Not invariant (velocity is layer-to-layer)
3. **Feature rescaling**: NOT invariant — velocity magnitude scales with feature magnitude
4. **Quantization**: NOT invariant — this IS the AWQ transfer problem

**Key insight**: The TT should ideally be invariant to quantization format (same velocity for BnB and AWQ hidden states). Current architecture has no mechanism for this. Multi-format training (EM-1) would force this invariance.

---

## Synthetic Data Validation

### Test Design

**Objective**: Verify that the pipeline can predict velocities in a simplified system with known ground truth.

**Synthetic system**:
1. Create a "toy Qwen" — 3 layers, hidden_dim=64
2. Define a known velocity function: v_l = sin(π·h_l) + ε where ε∼N(0, 0.01)
3. Generate 10K trajectories
4. Train a small TT (3-layer transformer, d_model=32)
5. Measure: can the TT recover the known velocity function?

**Expected result**: Synthetic TT should achieve R² > 0.95 on known function, validating that the architecture CAN learn velocities in principle.

**Failure mode**: If synthetic TT achieves R² < 0.80, the architecture has a fundamental limitation (e.g., transformer not suited for this prediction task).

**Recommendation**: Run this test before ANY other change. It costs <2 GPU hours and answers: "Is the TT architecture capable of learning velocity functions?"

### Null Hypothesis Test

**H0**: "The R²=0.85 ceiling is caused by irreducible noise in velocity targets, not by architectural or algorithmic limitations."

| Attribute | Value |
|-----------|-------|
| **Statement** | The observed R² ceiling of 0.85 is due to inherent stochasticity in Qwen's hidden state evolution, not to suboptimal normalization, loss, capacity, or attention |
| **Falsification Experiment** | Compute noise ceiling: run Qwen twice on same input (with disabled dropout, deterministic attention). Compute velocity MSE between runs. If noise MSE < 0.10 · total MSE → H0 rejected (architectural limitation dominates). If noise MSE > 0.50 · total MSE → H0 supported (noise ceiling near). |
| **Verdict** | **Pending experiment** — this is the single most informative diagnostic experiment to run |

**If H0 is supported** (noise ceiling):
- Further architecture improvements won't help
- Focus shifts to: (a) denoising input hidden states, (b) averaging multiple velocity estimates, (c) accepting R²=0.85 as fundamental

**If H0 is rejected** (architectural limitation):
- Normalization, loss, capacity changes have headroom
- Pursue top-5 recommendations from Phase 5
- Target R² = 0.90+
