# Phase 4: Divergent Pulse

---

## 4.1 Seed Expansion

### Semantic Constellation

**Analogous concepts by domain**:
- **Control theory**: Model Predictive Control, Linear-Quadratic Regulator, adaptive cruise control, fly-by-wire systems, PID controller tuning
- **Neuroscience**: Neurofeedback (real-time brain state modification), deep brain stimulation (electrode placement = layer selection), optogenetics (cell-type-specific = layer-specific)
- **Education**: Tutoring systems (targeted intervention at the right "layer" of understanding), scaffolding theory
- **Software engineering**: Hot patching (modify running system at specific module), aspect-oriented programming (cross-cutting concern injection at specific join points)
- **Evolutionary biology**: Developmental regulatory networks (Hox genes = master regulators), phenotypic plasticity

### Hallucinatory Pre-Seed Refinements

| Atom | Phase 1 Pre-Seed | Refined with Real-World Knowledge |
|------|-------------------|----------------------------------|
| A01 hidden_state | Continuous differentiable manifold | **Learned via SimCLR on trajectories**: train a contrastive model to embed hidden states such that correct answers cluster |
| A02 velocity | Exact ground-truth velocity | **Oracle from next-token future**: after generation completes, compute actual h[l+1] − h[l] and use as training target (already done — this IS the method) |
| A05 steering_vector | RL-optimized per context | **Meta-learned**: train a hyper-network that predicts α given the problem context embedding |
| A06 alpha | Dynamic per-token α | **Bayesian α**: treat α as a random variable; maintain belief over optimal α; update based on generation progress (e.g., confidence calibration) |
| A07 trim_tab_layer | Mechanistically explained | **Attention-flow analysis**: compute how much information flows through each layer using attention rollout; trim tabs = highest information-flow layers |
| A08 death_layer | Mechanistically explained | **Gradient signal-to-noise**: death layers = layers where gradient of accuracy w.r.t. hidden state has lowest SNR |

### Cryptic Analogy Mining

- **Function**: "Predict future state of a dynamical system and apply corrective perturbation to guide it toward a target."
- **Domain 1** (Astrodynamics): **Station-keeping** — satellites need periodic thruster burns to counteract orbital drift. The burns must be applied at specific points in the orbit (analogous to specific layers at specific token positions). **Layer 8 = perigee burn.**
- **Domain 2** (Audio engineering): **Compressor/gate** — an audio compressor applies gain reduction only when signal exceeds a threshold. The "trim-tab" is the frequency band where compression is most effective. **Death layer = feedback loop at resonant frequency that causes oscillation.**
- **Domain 3** (Fermentation): **Starter culture** — adding a small amount of a desired microbial culture at the right growth phase determines the entire fermentation outcome. **Steering = adding the right culture at the right layer.**

---

## 4.2 Mutation Operators

### M1: SUBSTITUTE — Replace steering surface

| Atom/Junction | Variant | Description | Quality Score |
|--------------|---------|-------------|---------------|
| C02 (KV-steering) | **MLP-output steering** | Instead of K/V projections, modify the MLP output at specific layers | Novelty: 4, Feasibility: 3, Risk: 4 |
| C02 (KV-steering) | **Attention logit biasing** | Directly modify the pre-softmax attention logits rather than modifying KV cache entries | Novelty: 4, Feasibility: 4, Risk: 3 |
| C02 (KV-steering) | **Embedding steering** | Modify the input token embeddings before the first layer | Novelty: 3, Feasibility: 5, Risk: 2 |

### M2: INVERT — Opposite direction steering

| Variant | Description | Quality Score |
|---------|-------------|---------------|
| **Negative α only** | Steer in opposite direction of velocity: h' = h − α·v | Novelty: 3, Feasibility: 5, Risk: 3 |
| **Contrastive with negative α** | v_contrastive = v_incorrect − v_correct (reverse direction) | Novelty: 4, Feasibility: 4, Risk: 4 |
| **Per-layer sign optimization** | For each layer, learn whether α should be positive or negative | Novelty: 5, Feasibility: 4, Risk: 3 |

### M3: SCALE — Vary granularity

| Variant | Description | Quality Score |
|---------|-------------|---------------|
| **Per-head steering** | Compute velocity per attention head and steer each head independently | Novelty: 5, Feasibility: 2, Risk: 4 |
| **Per-position steering** | Different α for each generated token (not uniform across generation) | Novelty: 4, Feasibility: 4, Risk: 3 |
| **Continuous α over positions** | α as a learned function of token position (e.g., sinusoidal schedule) | Novelty: 5, Feasibility: 3, Risk: 3 |
| **Per-layer α vector** | Learn a vector α ∈ R^L with L = number of layers | Novelty: 3, Feasibility: 5, Risk: 2 |

### M4: REORDER — Reverse pipeline dependencies

| Variant | Description | Quality Score |
|---------|-------------|---------------|
| **Steer THEN generate** | Modify KV cache for prompt tokens before generation begins, not during | Novelty: 4, Feasibility: 5, Risk: 3 |
| **Iterative steer-collect** | Interleave steering and trajectory collection in a closed loop | Novelty: 5, Feasibility: 2, Risk: 5 |
| **Apply TT at prompt only** | Use TT on prompt hidden states to compute steering for entire generation | Novelty: 4, Feasibility: 4, Risk: 3 |

### M5: MERGE — Combine with other techniques

| Variant | Description | Quality Score |
|---------|-------------|---------------|
| **Steering + Chain-of-Thought prompting** | Apply steering on top of CoT prompting | Novelty: 3, Feasibility: 5, Risk: 2 |
| **Steering + Few-shot examples** | Steering KV cache + few-shot context | Novelty: 3, Feasibility: 5, Risk: 2 |
| **Steering + Self-consistency** | Generate N steered trajectories, take majority vote | Novelty: 4, Feasibility: 4, Risk: 2 |
| **Steering + Adapter fusion** | Combine steering with LoRA/DoRA weight updates via StreamFusion | Novelty: 5, Feasibility: 3, Risk: 4 |

### M6: SPLIT — Decompose steering

| Variant | Description | Quality Score |
|---------|-------------|---------------|
| **Multi-stage steering** | Phase 1: steer at L2 (encoding) to improve understanding → Phase 2: steer at L8 (reasoning) to improve output | Novelty: 5, Feasibility: 3, Risk: 4 |
| **Decompose by attention head role** | Key-steering and value-steering independently | Novelty: 4, Feasibility: 2, Risk: 5 |
| **Spatial-frequency decomposition** | Steer low-frequency components of hidden state vs high-frequency | Novelty: 5, Feasibility: 2, Risk: 5 |

### M7: ABSTRACT — Generalize mechanism

| Variant | Description | Quality Score |
|---------|-------------|---------------|
| **Graph-based steering** | Represent layers as nodes in a computation graph; compute optimal intervention via influence propagation | Novelty: 5, Feasibility: 1, Risk: 5 |
| **Information-theoretic steering** | Maximize mutual information between steered state and correct answer | Novelty: 5, Feasibility: 2, Risk: 4 |
| **Energy-based steering** | Define energy function over hidden states; steer toward lower energy states | Novelty: 5, Feasibility: 2, Risk: 4 |

### M8: CONCRETIZE — Implementation variants

| Variant | Description | Quality Score |
|---------|-------------|---------------|
| **torch.compile for TT** | Compile TT for faster inference during generation | Novelty: 2, Feasibility: 5, Risk: 1 |
| **CUDA Graph for steering loop** | Capture entire steering loop as a CUDA graph | Novelty: 3, Feasibility: 4, Risk: 2 |
| **Sparse α (most layers α=0)** | Skip computation for zero-α layers | Novelty: 2, Feasibility: 5, Risk: 1 |

### M9: TRANSPOSE — Cross-domain application

| Variant | Description | Quality Score |
|---------|-------------|---------------|
| **Apply to decoder-only vision models** | Steering VLM (vision-language model) hidden states | Novelty: 4, Feasibility: 3, Risk: 4 |
| **Steer embedding models for RAG** | Improve retrieval quality by steering encoder hidden states | Novelty: 4, Feasibility: 3, Risk: 3 |
| **Steer diffusion model U-Net** | Apply velocity-based steering to denoising U-Net hidden states | Novelty: 5, Feasibility: 2, Risk: 5 |

### M10: NEGATE — Remove component

| Variant | Description | Quality Score |
|---------|-------------|---------------|
| **No TT — constant steering** | Replace TT with a learned constant vector per layer (treat as parameter) | Novelty: 3, Feasibility: 5, Risk: 3 |
| **No TT — random steering** | Steer with random Gaussian vectors; compare to TT-based steering | Novelty: 4, Feasibility: 5, Risk: 2 |
| **Remove steering for confident tokens** | Only steer when model confidence < threshold | Novelty: 3, Feasibility: 4, Risk: 3 |

### M11: RANDOMIZE — Stochastic steering

| Variant | Description | Quality Score |
|---------|-------------|---------------|
| **Annealed noise steering** | Add decreasing Gaussian noise to steering vector during generation | Novelty: 4, Feasibility: 4, Risk: 3 |
| **Monte Carlo steering** | Sample multiple steering directions, aggregate effects | Novelty: 4, Feasibility: 3, Risk: 4 |
| **Dropout on steering layers** | Randomly select different layer subsets each generation step | Novelty: 5, Feasibility: 4, Risk: 4 |

### M12: OSCILLATE — Time-varying steering

| Variant | Description | Quality Score |
|---------|-------------|---------------|
| **Cyclic α** | α oscillates between layers during generation | Novelty: 5, Feasibility: 3, Risk: 4 |
| **Steering annealing** | Stronger steering at early generation steps, tapering to zero | Novelty: 4, Feasibility: 4, Risk: 3 |
| **Attention coordinated steering** | Schedule layer steering based on model's current attention distribution | Novelty: 5, Feasibility: 2, Risk: 5 |

---

## 4.3 Forced Collisions

### Speculative Analogues (10 per atom, top-3 shown)

| Atom | Analogue 1 | Analogue 2 | Analogue 3 |
|------|-----------|-----------|-----------|
| A01 hidden_state | Concentration gradient in morphogenesis | Latent code in VAE | Wave function collapse |
| A02 velocity | Optical flow in video | Momentum in SGD | Drift velocity in semiconductors |
| A06 alpha | Learning rate in optimization | Mutation rate in EA | Spring constant in Hooke's law |
| A07 trim_tab | Rudder chord on aircraft | Local oscillator in radio | Mute button on mixer |
| A08 death_layer | Destructive interference | Negative control in CRISPR | Short circuit in electronics |
| A16 contrastive | Adversarial training (GANs) | Triplet loss (FaceNet) | Contrastive predictive coding |

### Orthogonal Mechanisms (5 per Master Regulator)

**MR1 (Layer Selection)**:
1. **Gradient-based**: ∂(accuracy)/∂(layer_steering_strength) via REINFORCE
2. **Attention-based**: Identify layers with highest attention entropy → these are most malleable
3. **Mutual-information-based**: I(hidden_state(L); answer) → steer layers with highest MI with answer
4. **Causal tracing**: Intervene at each layer using Pearl's do-operator → find causally relevant layers
5. **Representation-similarity-based**: CKA between layers of correct/incorrect trajectories → find divergent layers

**MR2 (Contrastive TT)**:
1. **Single TT with contrastive loss**: Train one TT with ℓ = MSE(v_pred, v_correct) − MSE(v_pred, v_incorrect)
2. **Direct difference without TTs**: Compute v_contrastive = h_correct[l+1] − h_incorrect[l+1] directly from data averages
3. **Prototypical steering**: Cluster trajectories; compute centroid of correct cluster; steer toward centroid
4. **Ensemble contrastive**: Bootstrap 50 random correct/incorrect splits; average steering directions
5. **Adversarial contrastive**: Train TT to predict velocity, discriminator to classify correct vs incorrect from steered states

**MR3 (Architecture)**:
1. **GDN-specific steering**: Directly modify GatedDeltaNet's recurrent state (s[l]) instead of K/V
2. **Universal steering surface**: Steer LayerNorm output instead of K/V → works regardless of attention type
3. **Adaptive steering surface detection**: Automatically detect which activations have most steering leverage for any architecture
4. **Quantization-aware steering**: Compensate for 4-bit quantization errors in the steering direction
5. **Sparse expert steering**: Route steering through different pathways based on input (MoE + steering)

**MR4 (Alpha Optimization)**:
1. **Confidence-scaled α**: α = α₀ × (1 − model_confidence)
2. **Uncertainty-aware α**: α = α₀ × TT_prediction_uncertainty (steer more when TT is more sure)
3. **RL-policy α**: Train a small Q-network to select α given (layer, token_position, attention_entropy)
4. **Genetic algorithm α**: Evolve α values via simple EA over multiple generation steps
5. **Bayesian optimization α**: Model α→accuracy as GP; maximize expected improvement

**MR5 (Data Split)**:
1. **Confidence-thresholded**: Use model's own confidence to create three-way split (confident correct, uncertain, confident wrong)
2. **Diffiсulty-stratified**: Split by problem difficulty (number of reasoning steps) instead of correctness
3. **Answer-distance-weighted**: Weight trajectories by |predicted_answer − correct_answer| (continuous, not binary)
4. **Clustering-based**: K-means on trajectories; compute per-cluster centroid velocities
5. **Magnitude-based**: Use velocity magnitude as signal — unusual velocities = moments of "insight" or "confusion"

### Paradoxical Combinations

**PC-1: "The steering direction that works best is the direction the model would naturally go, but with different magnitude."**
- Variation on the Phase 2 paradox: if the model naturally goes toward correct (73% baseline), steering at L8 amplifies an *existing* tendency, not a *corrective* one. This predicts that contrastive TT (v_c − v_i) would perform WORSE than standard TT at L8, because it pushes the model toward a direction it DOESN'T naturally go.
- **Test**: Compare standard TT vs contrastive TT on L8 specifically.

**PC-2: "The trim-tab layer is actually a death layer when α is large enough."**
- If steering at L8 with α=0.1 gives +20pp, but α=0.5 at L8 gives −30pp, then L8 is simultaneously a trim tab and a death layer — it's a layer of HIGH sensitivity, not of inherent goodness. This reframes trim tabs as "high-gain amplifiers" that need careful gain control.
- **Test**: Run α sweep on L8 from 0.001 to 2.0 and find the sign-flip point.

**PC-3: "The optimal strategy is to steer no layers and instead train the TT differently."**
- If contrastive TT + no steering(α=0) produces higher accuracy than any steered configuration, this means the mere act of *computing contrastive differences* during generation changes the model's behavior (through gradient flow during training, not inference-time steering).
- **Test**: Train contrastive TT, evaluate with α=0 as a baseline. If improved, the steering pipeline is unnecessary — the improved TT training is the real contribution.
