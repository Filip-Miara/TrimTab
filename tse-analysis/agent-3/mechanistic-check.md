# Phase 8: Mechanistic Interpretability Check

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## 1. Predictor Dissection

### Latent Space Analysis (TT Internal Representations)

**Current TT Architecture**: DeltaNet (recurrent) + MLP head. Input is sequence of past hidden states at a given layer; output is predicted velocity (Δh) for the next step.

**What we know**:
- TT achieves R²=0.85-0.94 on held-out generation steps
- TT trained on ALL layers separately or jointly? → Per-layer check needed
- TT's internal hidden state (DeltaNet recurrence) compresses ~50 past tokens into a single vector

**Hypothesized Latent Structure**:
- The DeltaNet's recurrent state likely captures "trajectory style" — rate of change, smoothness, direction persistence
- These features may correlate with token type (arithmetic tokens vs natural language tokens) rather than reasoning quality
- The R² gap between prompt (0.62) and gen (0.94) suggests generation trajectories are more structured (repetitive parts of reasoning: "Step 1:", "Therefore," etc.)

**Recommended Analysis**:
- PCA/UMAP of TT's recurrent hidden states during generation
- Color by: (a) token type (digit vs word vs operator), (b) position in generation (early vs late), (c) correct vs incorrect trajectory
- **Prediction**: Tokens will cluster by TYPE rather than by CORRECTNESS — supporting the surface-feature hypothesis (¬B1)

### Feature Attribution (What Drives TT Predictions?)

**Key Features**:
1. **Recent velocity**: h_t − h_{t-1} — the immediate past velocity is the strongest predictor of next velocity (momentum)
2. **Velocity norm**: ||h_t − h_{t-1}|| — magnitude of recent change (high when switching topics)
3. **Hidden state norm**: ||h_t|| — overall activation level
4. **Average velocity over window**: Mean of v_{t-5}, ..., v_{t-1}

**Hypothesized Importance**:
- Feature 1 (momentum) likely dominates — time-series prediction typically relies on autoregressive components
- If the TT is essentially an autoregressive momentum model (predict next velocity = last velocity + small correction), the "prediction" is trivial and the high R² is expected
- This would explain why contrastive signal is needed — momentum-based prediction is descriptive by nature (things continue as they were)

**Recommended Experiment**:
- **Compare TT vs naive baseline**: How does TT R² compare to a simple "velocity stays the same" predictor (v_{t+1} ≈ v_t)?
- If TT R² is close to the naive baseline, the TT is learning momentum, not complex dynamics
- If TT significantly outperforms the naive baseline, it's learning genuine trajectory structure

### Failure Modes (When Does TT Fail?)

| Condition | Expected R² | Explanation |
|-----------|-------------|-------------|
| High-velocity transitions (answer change) | LOW | TT trained on smooth trajectories fails at discontinuities |
| First token after a newline/step marker | LOW | Generation structure changes abruptly |
| Long deductions (>50 tokens) | LOW | Recurrent state may wash out early context |
| Correct trajectories | ~0.83 | Slightly lower than incorrect (~0.89) — more structured incorrect paths? |
| Incorrect trajectories | ~0.89 | Higher R² — incorrect reasoning may follow more predictable (stuck) patterns |

**Recommended Experiment**:
- Compute per-token prediction error as a function of token position, token type, and trajectory correctness
- If errors cluster at "reasoning transitions" (e.g., between arithmetic steps), TT is failing at the most important moments

### Causal vs Correlational Assessment

**Current Status**: The TT's predictions are CORRELATIONAL — it learns patterns in the trajectory data. The leap to causal interpretation (steering toward these velocities causes accuracy improvement) is unproven.

**Evidence for Causal**:
- Steering at L8 improves accuracy (+20pp)
- Steering at L9 degrades accuracy (-23pp)
- Steering direction matters (different layers → different results)

**Evidence for Correlational Only**:
- No control experiment (random vectors vs TT predictions)
- 88% token divergence suggests broad, non-specific effect
- Capability threshold suggests steering is amplifying existing noise-correlated structure

**Verdict**: INCONCLUSIVE — the L8 effect is reproducible but the MECHANISM is unknown. The following experiment would distinguish causal from correlational:

**Definitive Experiment**: Compare steering with TT predictions vs steering with matched random vectors (same norm, random direction) at L8:
- If TT > random → TT is capturing something meaningful (causal mechanism)
- If TT ≈ random → TT is just providing random perturbation, and L8 happens to be sensitive to any perturbation (correlational mechanism)
- If random > TT → something deeply unexpected

---

## 2. Representation Analysis

### Intrinsic Dimensionality

**Hidden State Trajectories**: 3584-dimensional (Qwen2.5-7B)

**Expected Intrinsic Dimension**: 
- 50-200 — much lower than ambient dimension (common for transformer hidden states)
- Generation trajectories may have lower intrinsic dimension than prompt trajectories (generation is more structured/repetitive)

**Recommended Analysis**: PCA on the full trajectory matrix (n_tokens × 3584) and compute explained variance ratio:
- If 90% variance is captured by <50 dimensions, the effective steering space is much smaller than 3584
- This would support per-dimension α (SV-17) — most dimensions are noise for steering purposes
- Could identify which dimensions are "critical" across layers (shared reasoning subspace)

### Manifold Structure

**Hypothesized Structure**: The hidden state manifold during generation is approximately a "reasoning path" — a low-dimensional curve through the high-dimensional space, parameterized by generation progress.

**Correct vs Incorrect Separability**:
- If correct and incorrect trajectories are in DISTINCT regions of the manifold → steering can push toward correct region
- If correct and incorrect trajectories OVERLAP → steering cannot separate them

**Recommended Analysis**:
- UMAP projection of trajectories (2000 tokens × 3584 → 2D)
- Color by: correctness, layer, generation position
- Measure: distance between correct and incorrect trajectory clusters per layer

**Prediction**: 
- At L8 (trim-tab), correct/incorrect trajectories are MORE separable than at other layers → explains why L8 steering works
- At L9 (death), correct/incorrect trajectories may be INVERTED or NON-SEPARABLE → explains why L9 steering is harmful
- This would be the mechanistic explanation for the trim-tab/death phenomenon

### Invariance Properties

**What transformations leave hidden states unchanged?**
- Token position translations (shifting the entire sequence by 1 token) → approximate invariance
- Scaling (multiplying all states by a constant) → NOT invariant (attention is scale-sensitive)
- Rotation within attention-orthogonal subspaces → partially invariant

**Relevance to Steering**:
- If hidden states are invariant to certain transformations, steering in those invariant directions is wasted
- The optimal steering direction lies in the NON-invariant subspace — where changing the hidden state actually changes model behavior

---

## 3. Synthetic Data Validation

### Test Design

**Goal**: Verify that the analysis pipeline can detect a known ground-truth steering effect.

**Construction**:
1. Create a SIMPLIFIED model where the "correct" steering direction is known:
   - Use a 2-layer MLP as a proxy for the language model
   - Define a binary classification task (spiral dataset or moons dataset)
   - Design the MLP such that the hidden state at layer 1 has a known "correct manifold" (linearly separable)
2. Apply the same trajectory prediction + steering pipeline:
   - Collect "generation trajectories" from forward passes
   - Train TT on these trajectories
   - Apply steering at layer 1 (the hidden layer)
   - Measure accuracy improvement

**Known Ground Truth**: The correct steering direction is toward the class centroid of the correct class.

**Verification Criteria**:
- Pipeline passes: TT achieves high R², and steering toward the correct direction improves accuracy
- If the pipeline identifies the correct steering direction → validation passes
- If the pipeline fails → the approach has fundamental issues that are NOT specific to LLMs

### Expected Result

**Pass**: The pipeline should work on the synthetic model — the task is simpler, velocity dynamics are more structured, and the correct direction is known.

**Failure Mode (if pipeline fails on synthetic data)**:
- Implication: The steering framework has a fundamental flaw not related to LLM complexity
- Possible root causes: TT architecture is inadequate even for simple dynamics, α·v steering is invalid even in flat manifolds, trajectory prediction is fundamentally not useful for classification

### Status

**Not yet performed** — this is a proposed experiment. Priority: HIGH.

---

## 4. Null Hypothesis Test

### H0-1: Steering Effect Is Random Perturbation

**Null Hypothesis**: The accuracy improvement from steering at L8 is due to random perturbation of the hidden state, not to velocity-specific structure.

**Observation**: L8 steering with TT predictions gives +20pp.

**Falsification Experiment**: Compare L8 steering with TT predictions vs L8 steering with random vectors of the same norm distribution. If random vectors also produce +20pp (or some positive improvement), the null hypothesis is confirmed — TT's predictions are not necessary.

**Required Setup**: 
- Generate random perturbation vectors with the same mean norm as TT predictions at L8
- Apply to KV cache at L8 with same α=0.1
- Evaluate on same 100 GSM8K problems

**Minimum Effect to Reject Null**: TT > random by >5pp (statistically significant at p<0.05)

**Verdict**: **NOT YET TESTED** — this is the highest-priority experiment.

### H0-2: Contrastive Signal Is Style Difference

**Null Hypothesis**: The contrastive signal (v_correct − v_incorrect) captures differences in output style (length, confidence, token distribution) rather than differences in reasoning quality.

**Observation**: Contrastive TTs achieve R²=0.83 on correct and incorrect trajectories separately.

**Falsification Experiment**: Measure the correlation between contrastive signal magnitude and non-reasoning features (output length, token entropy, perplexity). If correlation > 0.5 with any style feature, the null is partially confirmed.

**Required Setup**: 
- Generate 100 correct and 100 incorrect trajectories
- Compute contrastive signal at each token
- Regress contrastive signal magnitude against: (1) next-token perplexity, (2) output position, (3) answer length, (4) token type (digit vs word)
- If any regression R² > 0.25, style dominates

**Verdict**: **NOT YET TESTED** — recommended before interpreting contrastive evaluation results.

### H0-3: Capability Threshold Is α-Search Artifact

**Null Hypothesis**: Below-threshold models CAN be steered but only with a different α than the default 0.1.

**Observation**: SmolLM2 (4% baseline) shows all harmful steering with α=0.1.

**Falsification Experiment**: Sweep α over a 100× range {0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0} on SmolLM2 at L8 (or the equivalent layer position). If any α > 0 gives positive improvement, the threshold is an artifact of fixed α.

**Minimum Effect to Reject Null**: Any improvement ≥2pp (above the 4% baseline) at p<0.1.

**Verdict**: **NOT YET TESTED** — RECOMB-FP4 from Phase 4b recommends this experiment.

---

## Summary

| Check | Status | Key Finding |
|-------|--------|-------------|
| Latent Space Analysis | Not performed | Likely clusters by token type, not correctness |
| Feature Attribution | Not performed | Likely dominated by momentum (autoregressive) |
| Failure Modes | Not performed | Likely highest at reasoning transitions |
| Causal vs Correlational | **INCONCLUSIVE** | L8 effect is real but mechanism unknown |
| Intrinsic Dimensionality | Not performed | Estimated 50-200 for generation trajectories |
| Manifold Separability | Not performed | L8 may have higher correct/incorrect separability |
| Invariance Properties | Not performed | Unknown |
| **Synthetic Validation** | **Not performed** | Highest priority — validates entire pipeline |
| **H0-1 (Random vs TT)** | **Not performed** | Highest priority — 1 hour experiment |
| H0-2 (Style vs Reasoning) | Not performed | Recommended before contrastive eval |
| H0-3 (α-Search Artifact) | Not performed | 2 hour experiment |

**Critical Path**: Perform H0-1 (random vector control) and synthetic validation before any further development. These two experiments determine whether the observed results are meaningful or artifactual.
