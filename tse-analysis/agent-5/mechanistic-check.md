# Phase 8: Mechanistic Interpretability Check

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## 1. Predictor Dissection

### TrajectoryTransformer (TT) Internal Analysis

**Architecture**: Small MLP (likely 1-2 hidden layers, ~few million params) taking h_t → v̂_t

**What does the TT actually learn?** — This is currently unknown and is the single biggest gap in the project's understanding.

**Hypothesis 1: Frequency Predictor**
- The TT learns positional frequency patterns — at position t, the hidden state tends to change in a characteristic way it has seen during training
- **Evidence for**: Positional encoding in transformers creates regular hidden state patterns; TT may learn these patterns rather than semantic velocity
- **Test**: Shuffle token positions in test trajectories; if R² drops, TT was using positional information

**Hypothesis 2: Token Identity Predictor**
- The TT learns that certain tokens (".", "Step", numbers) are followed by particular hidden state changes
- **Evidence for**: High R² (0.94) on SmolLM2 could come from simple token-level regularities
- **Test**: Mask token embeddings; if R² stays high, TT uses positional/hidden state dynamics, not token identity

**Hypothesis 3: Smoothness Exploiter**
- The TT exploits the fact that hidden states change smoothly (h_{t+1} ≈ h_t + ε) — R² comes from predicting "near-identity" not velocity
- **Evidence for**: Any smooth trajectory has high predictability; R² may be inflated
- **Test**: Compare TT error to naive baseline (predict v̂_t = 0, i.e., h_{t+1} = h_t); if TT only slightly beats this, it's exploiting smoothness

**Hypothesis 4: Error Amplifier**
- The TT is actually learning error patterns — where the model is going wrong — and faithfully reproducing them
- **Evidence for**: High R² on both correct (0.832) and incorrect (0.829) trajectories separately; the TT doesn't distinguish quality
- **Test**: Compare TT prediction error on correct vs incorrect trajectories; if similar, the TT is quality-agnostic

### Proposed TT Dissection Experiments

| Experiment | Method | Expected Insight |
|------------|--------|-----------------|
| **E1: Position shuffle** | Shuffle token positions, recompute R² | Frequency vs dynamics |
| **E2: Ablate token embedding** | Zero out token embedding in hidden state, predict velocity | Token vs hidden state contribution |
| **E3: Naive baseline** | Predict h_{t+1} = h_t, compute R² | Smoothness contribution |
| **E4: Residual analysis** | Compute velocity = TT_prediction vs actual; analyze residual structure | What the TT misses |
| **E5: Layer-wise TT probes** | Train separate TT per layer, compare R² and prediction directions | Which layers have learnable dynamics |
| **E6: PCA projection** | Project hidden states to 50D, train TT on projected states | What dimensionality of dynamics? |

---

## 2. Representation Analysis

### Hidden State Manifold

**Intrinsic Dimensionality**:
- Qwen2.5-7B hidden state: 3584D
- Estimated intrinsic dimension of trajectory manifold: **unknown** — this should be measured
- Method: PCA of all hidden states across all tokens and problems; count components needed to explain 95% variance
- **Prediction**: The trajectory manifold has much lower intrinsic dimensionality (50-200) than the full hidden state (3584)

**Manifold Structure**:
- Do correct and incorrect trajectories occupy separable manifolds?
- **Known**: R² for correct-only TT = 0.832, for incorrect-only TT = 0.829 (almost identical)
- This suggests both trajectories have equally learnable structure, BUT says nothing about separability
- **Test**: Train a classifier (linear probe) to distinguish correct vs incorrect hidden states; if >80% accuracy, manifolds are separable

**Invariance Properties**:
- What transformations leave the hidden state representation unchanged?
- **Token position**: Known to be encoded (positional embeddings) — velocity may be partially position-invariant
- **Problem difficulty**: Unknown — does the hidden state trajectory change qualitatively for harder problems?
- **Token type**: Numbers vs operators vs reasoning tokens — may have different velocity dynamics

### Proposed Representation Experiments

| Experiment | Method | Expected Insight |
|------------|--------|-----------------|
| **R1: PCA trajectory** | PCA on all hidden states across all generations | Intrinsic dimensionality |
| **R2: Linear probe** | Classify correct vs incorrect from hidden state | Manifold separability |
| **R3: UMAP visualization** | 2D UMAP of correct/incorrect trajectories | Qualitative separability |
| **R4: Trajectory clustering** | K-means on trajectories, examine cluster centers | Types of reasoning paths |
| **R5: Cross-problem generalization** | Do trajectories from problem A predict behavior on problem B? | Task specificity of dynamics |

---

## 3. Synthetic Data Validation

### Test Design
Construct a simplified system with known ground truth for steering:

**Synthetic Model**: A 4-layer transformer (hidden_dim=64, 2 heads) trained to perform binary addition of 8-bit numbers (easy, verifiable task)
- **Ground truth**: The model either computes correctly (target output) or makes specific error types
- **Steering target**: The "carry bit" layer — we know that layer 3 computes the carry bit; steering this layer toward correct carry computation should improve accuracy
- **Synthetic velocity**: We know the correct hidden state trajectory because we can run the model on problems where we know the answer

**Test Protocol**:
1. Train tiny transformer on binary addition (100% accuracy on training distribution)
2. Collect correct and incorrect trajectories (induce errors by adding noise to specific layers)
3. Train TT on these trajectories
4. Verify: does TT identify the carry-bit layer (layer 3) as the trim-tab?
5. Verify: does steering at layer 3 recover accuracy?

### Result
(Not yet conducted — this is a proposed validation)

**Expected result**: If the pipeline can identify the known-correct steering target (layer 3) in a system where we know the mechanism, then it validates the approach. If it fails (identifies a different layer or no layer), there is a fundamental flaw in the steering methodology.

**If pipeline fails on synthetic data**: The entire real-data analysis is compromised. TT is not learning causal dynamics, and per-layer patterns are artifacts.

**If pipeline succeeds on synthetic data**: Confirms that the methodology is sound; real-model results are likely valid.

### Implications

| Synthetic Result | Implication for Real Data |
|-----------------|---------------------------|
| ✅ Identifies correct layer | Methodology validated; proceed |
| ❌ Identifies wrong layer | Per-layer patterns are confounded |
| ❌ No layer found (all neutral) | Steering methodology has hidden flaw |
| ✅ Finds trim-tab + death layer pattern | Pattern is fundamental to transformers |

---

## 4. Null Hypothesis Test

### H0: The +20pp Effect Is Due to Random Variation, Not Steering

**Statement**: "The observed accuracy improvement at L8 (+20pp from 45% to 65% on 100 problems) is within the sampling noise of the evaluation procedure and does not reflect a genuine steering effect."

**Falsification Experiment**:
1. Run 1000 baseline evaluations (no steering) on Qwen2.5-7B with different random seeds
2. Compute the 99.9th percentile of accuracy differences between any two runs
3. If +20pp exceeds this threshold, reject H0 (p < 0.001)
4. If +20pp is below threshold, fail to reject H0 (effect is noise)

**Minimum detectable effect**: With 100 problems, the standard error of accuracy difference is approximately sqrt(p(1-p)/n) = sqrt(0.73*0.27/100) ≈ 4.4pp. A 20pp effect is 4.5 standard errors, which is statistically significant by conventional measures (p < 0.00001). However, this assumes independent trials, which may not hold if there are systematic correlations.

**Confounders to Control**:
- Random seed differences in model sampling (temperature, top-k)
- Prompt formatting variations
- Evaluation order effects
- GPU nondeterminism

### Verdict (Expected)

| Scenario | Verdict | Confidence |
|----------|---------|------------|
| +20pp exceeds 99.9th percentile of null | **REJECT H0** (effect is real) | 8/10 |
| +20pp within 99th percentile | **INCONCLUSIVE** (more data needed) | 2/10 |
| +20pp within 95th percentile | **NOT REJECTED** (effect is noise) | 1/10 |

**Current confidence**: Given that the +20pp effect is 4.5 standard errors above baseline, and the pattern replicates on SVAMP (+4pp) and via cross-model transfer, H0 is likely false. However, the formal null hypothesis test should still be conducted for rigor.

---

## Summary of Gaps

| Gap | Criticality | Resolution |
|-----|-------------|------------|
| TT internals unknown (what does it learn?) | **CRITICAL** | E1-E6 experiments |
| Hidden state manifold structure unknown | **HIGH** | R1-R5 experiments |
| No synthetic data validation | **CRITICAL** | Build synthetic binary-addition model |
| Null hypothesis not formally tested | **HIGH** | Run 1000 seed baseline |
| Steering mechanism (K/V add) never validated at attention level | **MEDIUM** | Attention pattern visualization |
| Residual stream propagation of steering effect unknown | **HIGH** | Measure hidden states before/after steering per layer |
