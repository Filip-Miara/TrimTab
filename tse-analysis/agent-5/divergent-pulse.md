# Phase 4: Divergent Pulse

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Seed Expansion

### Semantic Constellation

| Concept | Analogous Structure | Domain | Mapping |
|---------|-------------------|--------|---------|
| Velocity prediction | Optical flow estimation | Computer vision | Predict pixel motion from local context (same as TT predicting state motion) |
| KV-cache steering | Targeted drug delivery | Pharmacology | Deliver modifier to specific cell (layer) without affecting others |
| Death layer | Cortical spreading depression | Neuroscience | Local disruption that propagates globally |
| Trim-tab layer | Leverage point in martial arts | Combat sports | Small correct force at the right moment produces disproportionate effect |
| Capability threshold | Activation energy in chemistry | Chemistry | Below threshold, reaction doesn't proceed; above, it's self-sustaining |
| Per-layer sweep | Gene knockout experiment | Genetics | Disable one gene (layer) at a time and observe phenotype (accuracy) |
| Contrastive direction | Antibody-antigen binding | Immunology | Difference vector between harmful and benign determines therapeutic direction |
| Token divergence | Genetic drift | Evolutionary biology | Random change in population (token distribution) over time |

### Pre-Seed Refinements

| Atom | Original Pre-Seed | Refined Pre-Seed |
|------|-------------------|------------------|
| A_H (Hidden State) | Interpretable 3D latent space | Hidden state decomposes into 5 interpretable components via sparse autoencoder: {reasoning_vector, position_code, token_type, confidence, uncertainty} |
| A_V (Velocity) | Multi-scale velocity | Velocity at scale s is (h_{t+s} − h_t)/s; TT predicts velocity at multiple scales simultaneously; steering at scale 1 produces scale-s effects predicted by scale-s TT |
| A_TT (TT) | Universal foundation TT | TT is a 2-layer transformer that takes (h_t, h_{t-1}, ..., h_{t-k}) and predicts (v_t, v_{t+1}, ..., v_{t+k}) with uncertainty |
| A_α (Alpha) | Learned gating network | α(layer, position) = sigmoid(MLP([h_layer; pos_embedding; confidence])) × α_max |
| A_L (Layer) | Differentiable steering switch | g(layer) = sigmoid(θ_layer) ∈ {0,1}; trained end-to-end via straight-through estimator |
| A_M (Capability) | Steering bootstraps capability | Use self-distillation: model generates correct answers on easy problems → trains implicit capability → steering amplifies it |

### Cryptic Analogy Mining

| Atom Function (Abstract) | Analogy Domain | Found Mechanism |
|--------------------------|---------------|-----------------|
| "Predict where a dynamic system will be next" | Weather forecasting (4D-Var data assimilation) | Use ensemble Kalman filter: run N forward simulations (at low compute), compute covariance of prediction error, steer based on Kalman gain |
| "Modify one part of a coupled system" | Guitar string damping | Touch a string at the harmonic node to damp only that frequency — steer only the component of the hidden state that affects the target behavior |
| "Find which component is responsible for output" | Fault isolation in electronics | Signal injection: inject a known perturbation at each layer, measure output change, build transfer function → layer output relevance |
| "Improve reasoning without changing weights" | Cognitive behavioral therapy | Change the model's "thought patterns" by modifying the internal monologue (KV cache is the internal monologue state) |

---

## Mutation Operators (M1-M12)

### Applied to Atoms

| Mutation | Atom | Variant | Quality |
|----------|------|---------|---------|
| M1_SUBSTITUTE | A_KV | Replace KV-cache steering with **activation steering** (modify MLP activations instead) | Novelty: 4, Feasibility: 3, Risk: 2 |
| M1_SUBSTITUTE | A_V | Replace 1-step velocity with **2-step acceleration** (h_{t+2} − 2h_{t+1} + h_t) | Novelty: 3, Feasibility: 5, Risk: 1 |
| M2_INVERT | A_α | Use **negative α** (steer opposite to TT prediction) on death layers | Novelty: 5, Feasibility: 5, Risk: 3 |
| M2_INVERT | A_C | Use **v_incorrect − v_correct** instead of v_correct − v_incorrect | Novelty: 4, Feasibility: 5, Risk: 2 |
| M3_SCALE | A_TT | Scale TT from MLP → **2-layer transformer** with 4 attention heads | Novelty: 2, Feasibility: 4, Risk: 2 |
| M3_SCALE | A_G | Scale trajectory from 1 model → **100-model trajectory dataset** from diverse models | Novelty: 5, Feasibility: 1, Risk: 4 |
| M4_REORDER | C3_TRAIN | **Online training**: train TT while collecting trajectories (no separate collection phase) | Novelty: 4, Feasibility: 3, Risk: 3 |
| M5_MERGE | A_TT + A_α | **Steering-ware TT**: TT predicts both velocity AND optimal α for that prediction | Novelty: 5, Feasibility: 2, Risk: 4 |
| M6_SPLIT | A_TT | Split TT into **layer-specific TTs** (one per layer, each 1/28 the size) | Novelty: 3, Feasibility: 4, Risk: 2 |
| M7_ABSTRACT | A_P | Abstract "per-layer pattern" to **"per-circuit pattern"** (steer specific computational circuits, not layers) | Novelty: 5, Feasibility: 1, Risk: 3 |
| M8_CONCRETIZE | A_V | Replace velocity with **explicit reasoning step detection** (only steer at reasoning token boundaries) | Novelty: 4, Feasibility: 4, Risk: 2 |
| M9_TRANSPOSE | C2_STEER | Transpose steering from **attention** to **MLP layers** (modify MLP weights for one forward pass) | Novelty: 5, Feasibility: 3, Risk: 4 |
| M10_NEGATE | A_C | **Don't use contrastive direction**; use TT uncertainty as steering signal (steer where TT is most uncertain) | Novelty: 5, Feasibility: 3, Risk: 3 |
| M11_RANDOMIZE | A_α | Randomize α per layer from distribution N(μ_layer, σ_layer); learn μ, σ via RL | Novelty: 4, Feasibility: 3, Risk: 3 |
| M12_OSCILLATE | A_α | Oscillate α between +α and −α at 2-token frequency (explore both directions) | Novelty: 5, Feasibility: 4, Risk: 2 |

### Applied to Junctions

| Mutation | Junction | Variant | Quality |
|----------|----------|---------|---------|
| M1_SUBSTITUTE | J3 (TT→Steer) | Replace: don't steer based on TT; steer based on **contrastive PCA of trajectory ensemble** | Novelty: 5, Feasibility: 2, Risk: 3 |
| M2_INVERT | J7 (Baseline→Threshold) | Invert: **higher baseline → LESS steering needed** (inverse relationship) | Novelty: 5, Feasibility: 3, Risk: 3 |
| M5_MERGE | J2 + J13 | Merge: TT jointly predicts velocity AND accuracy improvement of steering | Novelty: 5, Feasibility: 1, Risk: 5 |
| M7_ABSTRACT | J1 (H→V) | Abstract: not just H→V but **H→(V, next_token_logits, confidence)** | Novelty: 4, Feasibility: 3, Risk: 2 |
| M9_TRANSPOSE | J13 (Steer→Accuracy) | Transpose: steering → accuracy is mediated by **coherence** (steering → coherence → accuracy) | Novelty: 4, Feasibility: 4, Risk: 2 |

---

## Forced Collisions

### Speculative Analogues (10 per atom, top-3 shown)

**For A_TT (TrajectoryTransformer)**:
1. **Reservoir computing**: Instead of training TT, use a fixed random projection (echo state network) that captures velocity features → steering direction = random projection of recent trajectory
2. **Physics-informed neural network**: TT loss includes physical constraints (e.g., hidden state norm should be bounded, velocity should be smooth)
3. **Neural ODE**: Model the hidden state trajectory as a continuous ODE; steering = modifying ODE parameters

**For A_L (Layer Selectivity)**:
1. **Randomized layer selection**: Instead of per-layer sweep, select layers at random and aggregate results (Monte Carlo layer selection)
2. **Information bottleneck**: Rank layers by mutual information between their output and the correct answer; steer top-K informative layers
3. **Causal tracing (Wang et al.)**: Use causal intervention to identify layers most responsible for correct reasoning; these are the trim tabs

**For A_α (Steering Alpha)**:
1. **Bayesian α**: Treat α as a random variable; integrate over its posterior: accuracy = ∫ accuracy(α) p(α | data) dα
2. **Adaptive α via PPL**: α = α_max × (1 − sigmoid(PPL − threshold)), so α decreases when perplexity rises
3. **Discrete α**: Only 2 values: {0, α_max}; simplifies optimization and prevents intermediate harmful regimes

### Orthogonal Mechanisms (5 per master regulator)

**For MR#1 (Layer Selection)**:
1. Learned gating via Gumbel-Softmax
2. Mutual information ranking (I(Layer output; Correctness))
3. Causal tracing intervention
4. Gradient-based importance (layer-wise gradient × activation)
5. Evolutionary search over layer subsets

**For MR#2 (Alpha)**:
1. Per-token α from meta-network
2. α = f(PPL_shift) — adaptive to model confidence
3. α annealed over generation (start high, decay)
4. Asymmetric α per steering direction (positive vs negative)
5. α optimized by Bayesian optimization per layer

**For MR#3 (TT Quality)**:
1. Transformer-based TT (trajectory context)
2. Multi-task TT (velocity + accuracy prediction)
3. Variational TT (predicts distribution over velocities)
4. Fourier features TT (learns in frequency domain)
5. Online TT (updated via streaming gradient descent)

### Paradoxical Combinations

**PC-1: "Steer by doing nothing."** The best steering is zero steering on all layers — because the TT is a confounded predictor, and any intervention reduces accuracy. The +20pp result is a statistical artifact. Test this by running 1000 random seeds of the same experiment and computing the null distribution.

**PC-2: "Steer the opposite direction on death layers."** If L9 is -23pp at +α, it should be +23pp at −α. The TT prediction is symmetric in quality (R² ≈ same), so the direction must be wrong, not the magnitude. Test by running a per-layer sweep with both +α and −α.

**PC-3: "The TT should be trained on steered trajectories."** The distribution shift between training (natural trajectories) and deployment (steered trajectories) is the root cause of limited improvement. If we train a TT on steered data from the current steering policy, the next generation of TTs will be better. This creates a self-improving loop: steer → collect new trajectories → train new TT → steer better.

---

## Variant Quality Summary

| Variant | Novelty | Feasibility | Coherence | Risk | Emergent Potential | Quality Index |
|---------|---------|-------------|-----------|------|-------------------|--------------|
| V1: Negative α on death layers | 5 | 5 | 5 | 3 | 4 | **4.2** |
| V2: Activation steering (MLP) | 4 | 3 | 4 | 2 | 4 | **3.8** |
| V3: Multi-scale TT | 3 | 5 | 5 | 1 | 3 | **3.8** |
| V4: Oscillating α | 5 | 4 | 4 | 2 | 3 | **3.8** |
| V5: Online TT training | 4 | 3 | 5 | 3 | 5 | **3.6** |
| V6: Self-improving steering loop | 5 | 3 | 5 | 4 | 5 | **3.6** |
| V7: Per-circuit steering | 5 | 1 | 4 | 3 | 5 | **3.4** |
| V8: Null hypothesis test | 3 | 5 | 5 | 1 | 3 | **3.8** |
| V9: Steering-aware TT | 5 | 2 | 4 | 4 | 5 | **3.4** |
| V10: Randomized per-layer α | 4 | 3 | 3 | 3 | 3 | **3.2** |
