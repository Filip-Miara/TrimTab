# Phase 10: Hyperstitional Bridge

---

## H-1: Frequency-Specific Steering (Structural)

- **Type**: STRUCTURAL
- **Statement**: L8 functions as a trim-tab layer because it modulates the **mid-frequency components** of the hidden state trajectory — components that correspond to mathematical reasoning structure — while L9 modulates **high-frequency noise** that disrupts token selection.

- **Evidence chain**: 
  1. PCA of TT velocity predictions across the training set will reveal ~5-10 significant components
  2. The per-layer projection onto these components will differ: L8 projects strongly onto components 2-4 (mid-rank), L9 projects onto components 8+ (high-rank)
  3. Steering using only the mid-rank components at L8 preserves the +20pp effect
  4. Steering using only the high-rank components at L9 preserves the −23pp effect

- **Falsification criteria**:
  - **Refuted by**: PCA reveals no consistent component structure across layers, OR L8 and L9 project onto the same components with similar magnitudes.
  - **Confirmed by**: L8 and L9 have statistically significantly different projections onto PCA components (p < 0.01), and component-filtered steering matches unfiltered steering.

- **Minimum experiment**: 
  1. Collect TT predictions on 1000 training examples
  2. Stack predictions into matrix (1000 × 2048) per layer
  3. Run PCA; compute per-layer component loadings
  4. Compare L8 and L9 loading vectors via cosine similarity

- **Risk**: False positive — component structure may be an artifact of PCA on high-dimensional data.
- **Value**: If confirmed, provides the first mechanistic explanation for L8's special role. Opens the door to frequency-specific steering as a general technique.

---

## H-2: K/V Amplification Hypothesis (Relational)

- **Type**: RELATIONAL
- **Statement**: The steering effect is mediated by **nonlinear amplification through the K/V projection matrices**, not by "pushing hidden states toward correct manifolds." Specifically, the key projection W_k maps small hidden-state perturbations into large changes in attention logits due to the high-dimensional geometry of the key space, where quasi-orthogonal key vectors produce sharp softmax sensitivity.

- **Evidence chain**:
  1. The distribution of attention logit changes Δa = q^T W_k(α·v) has heavy tails — most changes are small, but a few (at tokens attended to by the modified key) are large
  2. The large changes occur at tokens whose key vectors are nearly orthogonal to most queries, making them highly sensitive to perturbation
  3. The difference between L8 and L9 is that L8's K/V projections have higher "amplification factor" (ratio of ‖W_k(α·v)‖ to ‖α·v‖)

- **Falsification criteria**:
  - **Refuted by**: The distribution of attention logit changes is Gaussian (not heavy-tailed), OR L8 and L9 have similar K/V amplification factors, OR the attention distribution before/after steering is identical.
  - **Confirmed by**: Heavy-tailed Δa distribution, significant amplification factor difference between L8 and L9.

- **Minimum experiment**: 
  1. During steering, capture W_k(α·v) and compute ‖W_k(α·v)‖ / ‖α·v‖ per layer
  2. Collect Δa values across all attention heads and positions
  3. Compare amplification factors and Δa distributions between L8 and L9

- **Risk**: Medium — requires attention capture infrastructure.
- **Value**: If confirmed, reframes the entire project from "velocity dynamics" to "attention sensitivity engineering."

---

## H-3: Capability Threshold as Manifold Phase Transition (Potential)

- **Type**: POTENTIAL
- **Statement**: The capability threshold (~40% GSM8K) is not a smooth function of model size but a **phase transition** in hidden state manifold topology. Below threshold, correct and incorrect trajectories occupy the same manifold region (non-separable). Above threshold, a symmetry-breaking occurs: the correct-trajectory manifold bifurcates from the incorrect-trajectory manifold, creating the geometric separation that steering exploits.

- **Evidence chain**:
  1. The R² of contrastive vs standard TT changes discontinuously at ~40% baseline
  2. Below threshold, the distribution of cosine similarities between correct and incorrect trajectories (for the same problem) is unimodal
  3. Above threshold, the distribution becomes bimodal (two clusters: similar and dissimilar)
  4. This bifurcation occurs at a specific model size (~3B params) or specific baseline accuracy (~40%), whichever is reached first

- **Falsification criteria**:
  - **Refuted by**: Qwen2.5-Math-1.5B (38% baseline, just below threshold) shows NO bimodality, while Qwen2.5-7B (73%) shows bimodality. But if a 3B model with 40% baseline also shows no bimodality, the phase transition is a myth.
  - **Confirmed by**: Multiple models tested at various baselines show a clear threshold effect, and within-model scaling (e.g., using fewer CoT steps to reduce baseline) shows the same threshold.

- **Minimum experiment**: 
  1. Compute cosine similarity between correct and incorrect trajectory hidden states for each problem at each layer
  2. Collect these similarities across all layers, models, and problems
  3. Fit a mixture model (1-component vs 2-component Gaussian) to the similarity distribution
  4. Correlate the best-fitting model (1 vs 2 components) with baseline accuracy

- **Risk**: High — requires trajectory data from multiple models with varying baselines (currently: 7B, Math-1.5B, SmolLM2-360M available; need 3-4 more models for proper phase transition analysis).
- **Value**: High — would provide a theoretical foundation for "steering requires capability," elevating it from empirical observation to predicted phenomenon. If correct, suggests that **steering is impossible for models below a critical capability threshold, regardless of architecture or steering mechanism.**

---

## H-4: Self-Application Projection (Structural-Meta)

- **Type**: STRUCTURAL
- **Statement**: The project's own research trajectory follows the same velocity-prediction dynamics as the models it studies. Specifically, the "velocity" of research progress (findings per session) is decreasing, and the "correct direction" (contrastive evaluation → publication) is known but requires a steering intervention at the "meta-layer" corresponding to experimental prioritization.

- **Evidence**: The project's discovery velocity was highest in sessions 1-3 (exponential learning: R², K/V method, per-layer effect) and has slowed in sessions 4-5 (diminishing returns: contrastive training, cross-dataset results). Without explicit "steering" toward the contrastive evaluation, the project may stall.

- **Falsification**: If completing the contrastive evaluation (Phase A1) does NOT produce a significant result that advances the project, the self-application analogy is false.

- **Value**: Framework for reasoning about research velocity as a steerable quantity.

---

## Summary Table

| ID | Type | Confidence | Value if True | Risk if False |
|----|------|-----------|---------------|---------------|
| H-1 | Structural | 4/10 | HIGH (mechanistic explanation) | Low (PCA is cheap) |
| H-2 | Relational | 6/10 | HIGH (reframes entire mechanism) | Medium (attention capture cost) |
| H-3 | Potential | 3/10 | VERY HIGH (theoretical foundation) | High (requires 3+ more models) |
| H-4 | Structural-Meta | 5/10 | Medium (project planning) | Low (no cost to test) |

### Research Call

**Target audience**: Mechanistic interpretability community, LLM alignment researchers.

**One-sentence pitch**: "We hypothesize that the effectiveness of per-layer KV-cache steering in LLMs is mediated by frequency-specific attention sensitivity, not by manifold pushing — and that a phase transition in hidden state topology explains why steering requires baseline capability."
