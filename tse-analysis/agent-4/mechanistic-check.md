# Phase 8: Mechanistic Interpretability Check

## 1. Predictor Dissection

### TrajectoryTransformer Analysis

**What does the TT's latent space look like?**

The TT is a 6-layer transformer (d_model=768, 8 heads) processing 23 position embeddings (one per hidden layer). Its latent representations are at position-indexed token level. We can analyze:

- **Internal representations**: After input projection, 3584-dim h[l] → 768-dim. This 4.7× compression likely preserves directions with highest variance (i.e., dimensions where hidden states differ most across layers and tokens).
- **Attention patterns**: With 23 positions (one per layer), self-attention can learn inter-layer dependencies. The causal/positional relationships between adjacent layers (L3→L4) are likely the strongest attention patterns.
- **Output projection**: 768→3584 expands back. The nullspace of this projection determines which velocity components CANNOT be predicted.

**What features drive its predictions?**

Hypothesized (not empirically verified):
1. **Hidden state norm**: Norm of h[l] likely correlates with velocity magnitude
2. **Hidden state direction**: The angular position in the 3584-dim space
3. **Layer index**: Position embedding provides layer identity
4. **Cross-layer differences**: Attention over (h[l-1], h[l], h[l+1]) patterns
5. **Token position within sequence**: Not used (TT only sees one time step)

**Feature attribution needed**: Apply integrated gradients or attention rollout to determine which of these actually matter.

**Under what conditions does it fail?**

- **Abrupt generation shifts**: If the model suddenly changes reasoning direction (e.g., self-correction), the velocity prediction based on the current state will be wrong.
- **First generated token**: No previous steered state to condition on (the bootstrap problem).
- **Low-probability tokens**: When the model assigns low probability to all alternatives (high entropy), velocity dynamics may be erratic.

**Is it capturing causal structure or spurious correlations?**

Likely spurious: The TT is trained to predict v[l] from h[l] using MSE loss. High R² could arise from:
1. **Smoothness artifact**: h[l] and h[l+1] are nearby in space, so their difference is small and predictable from local geometry
2. **Linear predictability**: If the transformer's residual stream dynamics are approximately linear (h[l+1] ≈ W·h[l]), the TT learns the effective W
3. **Genuine structure**: If the TT captures which directions in hidden state space correspond to "improving reasoning"

**Diagnostic experiment**: Train a LINEAR predictor (ridge regression) on h[l]→v[l]. If R² > 0.8, the velocity structure IS approximately linear and the TT's capacity is wasted.

## 2. Representation Analysis

### Hidden State Manifold

**Intrinsic dimensionality** (estimated, not measured):
- 3584-dim hidden states likely live on a low-dimensional manifold (intrinsic dim ~50-200)
- If true, steering is only effective in ~50-200 directions, and the remaining 3384 dimensions are noise or task-irrelevant

**Manifold structure hypotheses**:
- **H1**: Correct and incorrect trajectories occupy SEPARABLE regions of the manifold (supports contrastive approach)
- **H2**: Correct and incorrect trajectories INTERLEAVE on the manifold (contrastive approach will fail)
- **H3**: The distinction is not correct/incorrect but "confident reasoning" vs "confused guessing" — and the TT is really predicting confidence, not correctness

**Invariance properties** (untested):
- Is the hidden state representation invariant to surface-form changes (e.g., "2+2=4" vs "two plus two equals four")?
- Is it invariant to different GSM8K problem phrasings that require the same reasoning?
- If invariant, steering improves reasoning; if not, steering memorizes surface patterns.

### Synthetic Trajectory Test

**Design**: Create a synthetic 2-layer "reasoning" problem:
- Layer 1 has d=10, truth: h[1] = W·h[0] where W is a known rotation matrix
- Layer 2 has d=10, truth: h[2] = h[1] + g where g is a "good reasoning direction" (known ground truth)
- Generate 1000 trajectories: 500 with g added (correct), 500 with -g added (incorrect)
- Train TT on these synthetic trajectories
- Test: does the TT learn the known g direction? Does contrastive TT recover g?

**Prediction**:
- Standard TT: predicts h[2]-h[1] accurately (R² close to 1.0) but doesn't distinguish g from -g
- Contrastive TT: v_c - v_i ≈ 2g (the correct direction), confirming the approach works in principle

**Ground truth**: g is known by construction. If the pipeline can't recover g from synthetic data, it will certainly fail on real data.

**Synthetic data requirements**:
- Simple enough: 10-dim, 2 layers, linear dynamics — ground truth is unambiguous
- Complex enough to exercise: TT input, attention over positions, output projection
- Independent of GSM8K: avoids overfitting to the test

## 3. Null Hypothesis Test

### H0-1: "The apparent steering effect is due to random perturbation, not directional velocity prediction"

**Statement**: Steering improves accuracy by injecting RANDOM noise into the KV cache, which happens to help on some problems. The TT's velocity prediction is irrelevant — any vector of appropriate scale would produce similar results.

**Falsification Experiment**:
1. Run per-layer sweep with v = TT(h) (standard method)
2. Run per-layer sweep with v = random_normal(0, σ²) matched to ||TT(h)|| (same magnitude, random direction)
3. Compare: if random steering achieves ≥L8's +20pp, H0 is confirmed (steering is just noise)
4. If random steering achieves ≤5pp for all layers, H0 is rejected (direction matters)

**Success criterion**: Random steering produces ≤5pp improvement on the best layer.
**Required compute**: ~2 hours (2 sweeps × 28 layers × 100 problems).

### H0-2: "The contrastive signal is not normative — it's just amplified noise from dataset imbalance"

**Statement**: v_c - v_i mainly captures differences in trajectory statistics (e.g., correct trajectories have longer/shorter generations, different token lengths) rather than meaningful steering directions.

**Falsification Experiment**:
1. Create a PAIRED dataset: for each problem, one correct trajectory and one incorrect trajectory with EQUIVALENT generation length (pad/truncate to match)
2. Train contrastive TT on paired data
3. If steering still works, the signal is about reasoning quality, not surface statistics
4. If steering fails, the original contrastive signal was partially spurious

**Success criterion**: Paired-data contrastive steering preserves ≥80% of the unpaired steering improvement.
**Required compute**: Requires data preprocessing + new TT training (~4-6 hours).

### H0-3: "Death layers are not protective — they're just layers where the steering direction happens to be wrong"

**Statement**: There's nothing special about death layers. If we flipped α to -0.1, L9 would become a trim-tab and L8 would become a death layer. The classification is α-dependent.

**Falsification Experiment**:
1. Run per-layer sweep with α = -0.1 (negative steering on all layers)
2. If L9 improves (L9:+N pp) and L8 worsens (L8:-M pp), H0 is confirmed — death layers are just layers with wrong steering direction
3. If L9 is still harmful (L9:-N pp) and L8 is still helpful (L8:+M pp), H0 is rejected — death layers are intrinsically harmful

**Success criterion**: Clear pattern shift under α inversion.
**Required compute**: ~1 hour (1 sweep × 28 layers × 100 problems with α=-0.1).

## Summary

| Test | Status | Finding |
|------|--------|---------|
| TT latent space analysis | NOT PERFORMED | Would reveal what TT actually learned |
| Linear baseline comparison | NOT PERFORMED | Would test if velocity is linearly predictable |
| Manifold intrinsic dimensionality | NOT PERFORMED | Would inform required TT capacity |
| Synthetic trajectory validation | NOT PERFORMED | Critical ground-truth test |
| H0-1: Random steering control | NOT PERFORMED | **MOST IMPORTANT** — tests the entire paradigm |
| H0-2: Paired contrastive control | NOT PERFORMED | Tests contrastive signal purity |
| H0-3: Negative α inversion | NOT PERFORMED | Tests death layer nature |

None of the mechanistic interpretability tests have been performed. The project currently lacks a verified causal understanding of why steering works. The top priority is H0-1 (random steering control) — if that fails (random steering works), the entire velocity-prediction approach needs fundamental reexamination.
