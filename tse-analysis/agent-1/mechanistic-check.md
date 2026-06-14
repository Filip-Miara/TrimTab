# Phase 8: Mechanistic Interpretability Check

**Subject**: RankAdaptation — Velocity-based latent steering
**Date**: 2026-06-14

---

## 1. Predictor Analysis

### 1.1 TT (TrajectoryTransformer) Latent Space

| Property | Assessment |
|----------|------------|
| **Architecture** | Unknown (spec not in debrief). Likely MLP or small transformer predicting 4-8K dim vector. |
| **Latent Space Structure** | Not analyzed. No PCA/UMAP projection performed on TT's internal representations. |
| **What does the TT learn?** | Based on R²=0.85-0.94 across models, TT learns the dominant velocity mode per position. Given the high R² on both correct and incorrect trajectories (~0.83 each), it likely learns the *average* velocity field rather than the *conditional* velocity field. |
| **Key Features Driving Predictions** | Unknown. No feature attribution (gradient-based or otherwise) performed on TT. |
| **Causal vs Correlational** | The TT is purely correlational. It predicts v_{t+1} from h_t, but there's no causal identification. The high R² could come from: (1) genuine velocity dynamics, (2) positional encoding bias (velocity is similar at same token positions), (3) auto-correlation in hidden states. |

### 1.2 Failure Mode Analysis

| Condition | Predicted Behavior | Evidence |
|-----------|-------------------|----------|
| **Distribution shift** (prompt→gen) | R² drops from 0.62→? | Prompt-trained TT fails at generation time → confirms distribution sensitivity |
| **Low confidence tokens** | May produce high-variance predictions | Not tested |
| **Long generations** (>200 tokens) | Velocity statistics may drift | Not tested |
| **Adversarial/degenerate inputs** | Unknown | Not tested |
| **Cross-model (domain shifted)** | SmolLM2 → 7B preserves L8 pattern | Works surprisingly well → suggests robust features |

### 1.3 Reading Head Analysis

The reading head (r=0.85) in the PPL-modulated correction was accurate at predicting the steering offset but the *application failed*. This suggests the bottleneck is not prediction quality but the *intervention mechanism* — the reading head predicted the right offset, but applying that offset at generation time didn't improve the output. This is consistent with the steering-direction-alignment hypothesis.

---

## 2. Representation Analysis

### 2.1 Hidden State Manifold

| Property | Assessment |
|----------|------------|
| **Intrinsic Dimensionality** | Not computed. For a 7B model with 3584-dim hidden states, intrinsic dimensionality likely 50-500 (based on literature: typical LLM intrinsic dim is 1-10% of full dim). |
| **Correct vs Incorrect Separability** | The existence of a contrastive TT with comparable R² to standard TT suggests correct/incorrect trajectories ARE distinguishable at the velocity level. The key question is whether they're distinguishable at the *individual trajectory* level or only at the *ensemble average* level. |
| **Manifold Invariance** | The cross-model transfer result (SmolLM2→7B preserving L8 pattern) suggests the velocity-encoding manifold is partially invariant to model scale — a surprising and important invariance property. |

### 2.2 Potential Manifold Issues

| Issue | Assessment | Severity |
|-------|------------|----------|
| **Off-manifold steering** | Adding α·v_pred to K/V may push representations off the natural data manifold, especially for large α or death layers. This would explain the catastrophic degradation pattern. | HIGH |
| **Manifold curvature** | Linear interpolation (α·v_pred) assumes locally flat manifold. If curvature is high, steering is suboptimal. Death layers may be regions of high curvature. | MED-HIGH |
| **Token-position manifold collapse** | Later tokens may have lower intrinsic dimensionality (more constrained by prior context), making them more sensitive to perturbation. Explains why degradation appears in later layers. | MED |

---

## 3. Synthetic Data Validation

### 3.1 Test Design

**Goal**: Verify that the steering pipeline can detect trim-tab layers when ground truth is known.

**Synthetic Construction**:
1. Take a small model (SmolLM2-360M, 4% baseline)
2. Artificially modify weights of ONE specific layer to produce "correct" outputs on a subset of problems
3. This creates a known "trim-tab layer" — the artificially modified layer
4. Run the standard steering pipeline: collect trajectories → train TT → per-layer sweep
5. Check: does the pipeline identify the artificially modified layer as a trim-tab?

**Expected Result**: If the pipeline works mechanistically, it should identify the forged trim-tab layer. If it fails, the pipeline is capturing correlation, not causal structure.

### 3.2 Result

**Status**: NOT PERFORMED. This test would require modifying model weights or creating a controlled environment, which was outside project scope.

**Implications**: Without synthetic validation, all steering results are correlational. We know steering L8 improves accuracy, but we don't know if L8 is *causally* the right layer or just *correlationally* the right layer. The trim-tab/death-layer classification remains an empirical observation without causal grounding.

### 3.3 Simplified Alternative

**Simpler synthetic test**: 
1. Generate trajectories from a model with known ground-truth answers
2. For each token, compute the *direction toward the correct final answer's hidden state* (this is the "true" steering direction if we had an oracle)
3. Train TT to predict this direction instead of the actual next-state velocity
4. Compare accuracy of TT predicting true direction vs actual direction
5. If TT can predict the true direction with R²>0, the steering signal is latent in the velocity field

**If this test passes**: The velocity field contains information about correctness direction, and steering can recover it.
**If this test fails**: Steering works through a different mechanism (e.g., general perturbation, not direction-specific).

---

## 4. Null Hypothesis Tests

### H0-1: The trim-tab effect is caused by layer position in the residual stream, not layer function.

| Property | Detail |
|----------|--------|
| **Statement** | L8 is a trim-tab not because of its specific computation, but because of its position in the 32-layer stack. Any layer at position 8 would show trim-tab properties. |
| **Falsification Experiment** | Ablate L8 (replace with identity) and check if the model still functions. If L8 is critical, it's functional. If not, it's positional. |
| **Alternative** | Shuffle layer order without changing weights. If trim-tab follows function (not position), it will change. |
| **Verdict** | **NOT TESTED**. Cannot distinguish positional from functional without intervening on layer identity. |

### H0-2: The death layer (L9) effect is caused by off-manifold perturbation, not computation disruption.

| Property | Detail |
|----------|--------|
| **Statement** | Steering at L9 pushes hidden states off the natural manifold, causing model collapse. The same collapse would occur with ANY perturbation at L9, not just the TT-predicted direction. |
| **Falsification Experiment** | Apply random perturbation (same magnitude as steering) at L9. If random also causes collapse → off-manifold effect. If random doesn't cause collapse → computation disruption. |
| **Also test** | Anti-steering at L9 (−α). If improvement → direction misalignment. If continued collapse → off-manifold perturbation. |
| **Verdict** | **PARTIALLY TESTABLE** — M2-1 (anti-steering) and M11-1 (random baseline) from Convergent Pulse would address this. |

### H0-3: The TT is just learning the autoregressive prior.

| Property | Detail |
|----------|--------|
| **Statement** | The TT's high R² (0.94) comes from learning that the next hidden state is similar to the current one (smoothness prior), not from learning reasoning-specific velocity dynamics. |
| **Falsification Experiment** | Compare TT prediction to a baseline that predicts v=0 (no change). If TT significantly outperforms v=0 baseline, it's learning real dynamics. |
| **Alternative** | Compare TT to naive baseline: h_{t+1} = h_t (identity prediction). The velocity is the residual, so R² of 0.94 means 94% of velocity is predictable — far above the identity baseline. So H0-3 is likely FALSE. |
| **Verdict** | **REJECTED** (from existing evidence). R²=0.94 significantly exceeds identity baseline. |

### H0-4: The capability threshold (40%) is actually an attention-head capacity threshold.

| Property | Detail |
|----------|--------|
| **Statement** | Models below 40% GSM8K lack the attention head capacity to have "correct" and "incorrect" paths diverge. Their hidden states are too constrained by low-capacity attention to support separable velocity manifolds. |
| **Falsification Experiment** | For a sub-threshold model (e.g., Math-1.5B), compute the attention head entropy and effective head count. Compare to threshold model (7B). Check if head-level capacity correlates with steerability. |
| **Verdict** | **NOT TESTED**. Requires attention analysis infrastructure. |

---

## Summary

| Component | Status | Key Finding |
|-----------|--------|-------------|
| TT Latent Space Analysis | ❌ NOT PERFORMED | Lack of interpretability tools limits causal understanding |
| Correct/Incorrect Separability | ⚠️ PARTIAL | R² similar for both → dynamics are similar; direction difference untested |
| Off-Manifold Assessment | ❌ NOT PERFORMED | Hypothesized but not measured |
| Synthetic Data Validation | ❌ NOT PERFORMED | All findings are correlational without it |
| H0-1: Layer Position vs Function | ❌ NOT TESTED | Cannot distinguish without layer interventions |
| H0-2: Off-Manifold vs Computation | ⚠️ TESTABLE | M2-1 + M11-1 would resolve this |
| H0-3: Autoregressive Prior | ✅ REJECTED | TT clearly learns real dynamics beyond inertial prior |
| H0-4: Head Capacity Threshold | ❌ NOT TESTED | Attention analysis infrastructure needed |
