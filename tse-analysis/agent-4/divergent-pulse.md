# Phase 4: Divergent Pulse

## Seed Expansion

### Semantic Constellation — Analogous Entities

| Atom | Analogous Entity | Domain | Key Difference |
|------|-----------------|--------|----------------|
| A07 (TT) | Kalman filter | Control theory | TT is feedforward, Kalman is recursive |
| A04 (KV mod) | Synaptic weight modification | Neuroscience | KV mod is per-inference, not persistent |
| A05 (α) | Learning rate | ML | α is inference-time, not training-time |
| A08 (trim-tab) | Gain stage | Audio engineering | Gain stages amplify signal AND noise |
| A09 (death layer) | Feedback loop with wrong polarity | Control theory | Pole-zero cancellation could fix it |
| C09 (amplification) | Transistor amplifier | Electronics | Amplification is bounded by supply voltage (hidden state norm) |
| A11 (threshold) | Critical mass | Nuclear physics | Chain reaction doesn't start below critical mass |
| A20 (manifold) | Fitness landscape | Evolutionary biology | Steering = gradient ascent on latent fitness |

### Hallucinatory Pre-Seed Refinements

| Atom | Refined Pre-Seed | Real-World Approximation |
|------|-----------------|-------------------------|
| A05 (α) | Per-token, per-head learned α via meta-learning | A small α-predictor network (2-layer MLP) using h[l] and attention pattern features |
| A07 (TT) | Foundation TT trained on 100+ LMs | Start with 5 diverse LMs (LLaMA, Mistral, Gemma, Qwen, Phi) |
| A09 (death layer) | Automatically detected and skipped by a classifier | Linear probe on h[l] → predict trim-tab/death using activation patterns |
| A15 (projection) | Universal hidden state manifold via contrastive learning | Align hidden states from different models using CCA |
| A17 (α alloc) | RL-optimized α(θ) | PPO with accuracy reward, α weights as action space |

### Cryptic Analogy Mining

| Atom Function (Abstracted) | Cryptic Search Query | Analogous Domain |
|---------------------------|---------------------|------------------|
| "Predict future state from current state" | "predictor-corrector methods differential equations" | Numerical analysis |
| "Steer composite system by modifying one component" | "trim tab control surface reversal at high Mach" | Aerospace engineering |
| "Some modifications help, some harm — identify which" | "gain scheduling for time-varying systems" | Control theory |
| "Self-referential improvement loop" | "bootstrapped reinforcement learning with learned reward" | RL theory |
| "Different architectures allow different interventions" | "instruction set architecture determines exploit surface" | Computer security |

## Mutation Operators (M1-M12)

### M1: SUBSTITUTE — Replace steering surface

| Target | Original | Substitute | Quality Score |
|--------|----------|------------|---------------|
| C07 (steering surface) | KV cache entries | Residual stream activations at attention output | 4.2/5 |
| A04 (KV mod) | K and V both | Only K, or only V, or Q | 3.8/5 |
| A07 (TT) | Transformer encoder | Mamba state-space model for velocity prediction | 3.5/5 |

### M2: INVERT — Reverse steering direction

| Target | Inversion | Quality Score |
|--------|-----------|---------------|
| A05 (α sign) | α = -0.1 on death layers → converts death→potential trim-tab | 4.0/5 |
| A14 (contrastive) | v_incorrect - v_correct (steer toward wrong answers) → control experiment | 3.0/5 |
| J09 (threshold) | Low-capability model can benefit MORE (steering fills in missing capability) | 2.5/5 (counterintuitive but testable) |

### M3: SCALE — Change magnitude regime

| Target | Scaled Variant | Quality Score |
|--------|---------------|---------------|
| A05 (α) | α ∈ {-1.0, -0.5, -0.2, -0.1, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0} | 4.5/5 |
| A12 (trajectory count) | 5000 → 50000 trajectories | 3.0/5 |
| A07 (TT capacity) | d_model: 256, 512, 768, 1024, 2048 | 3.5/5 |

### M4: REORDER — Change pipeline sequence

| Target | Reordered Variant | Quality Score |
|--------|-------------------|---------------|
| Steering timing | Steer BEFORE token generation (on initial prompt), not during | 3.8/5 |
| TT application | Apply TT every N-th token instead of every token (save compute) | 3.5/5 |
| Evaluation order | Run baseline AFTER steering sweep (control for model state drift) | 2.5/5 |

### M5: MERGE — Combine steering signals

| Target | Merged Variant | Quality Score |
|--------|---------------|---------------|
| C05 (contrastive) + A03 (velocity) | v = v_standard + β * (v_c - v_i) | 4.8/5 |
| A08 (trim-tab) + A09 (death) × A17 (α) | α_layer = α_base * (1 + trimtab_score - death_score) | 4.2/5 |
| S02 (prediction) + S03 (intervention) | End-to-end steering policy: TT output → directly modulate logits | 3.5/5 |

### M6: SPLIT — Separate combined processes

| Target | Split Variant | Quality Score |
|--------|---------------|---------------|
| A04 (KV mod) | Modify K and V with DIFFERENT α (α_k ≠ α_v per layer) | 4.5/5 |
| A07 (TT) | Per-layer TTs: train 28 small TTs instead of 1 big one | 3.5/5 |
| C05 (contrastive) | Three-way: v_correct, v_incorrect, v_all → ensemble voting | 3.2/5 |

### M7: ABSTRACT — Generalize the mechanism

| Target | Abstraction | Quality Score |
|--------|-------------|---------------|
| A04 (KV mod) | General activation steering: modify ANY intermediate activation | 4.0/5 |
| A05 (α) | Steering strength as a function of prediction uncertainty (α = f(entropy)) | 4.5/5 |
| A08 (trim-tab) | Any layer where gradient of accuracy w.r.t. hidden state > 0 | 3.5/5 |

### M8: CONCRETIZE — Specialize

| Target | Concretization | Quality Score |
|--------|---------------|---------------|
| A07 (TT) | Tiny TT: d_model=64, 2 layers, predicts only L8 velocity | 3.5/5 |
| A03 (velocity) | Binary velocity: steer up or down per component (ternary: +α, 0, -α) | 2.8/5 |
| A20 (manifold) | 2D PCA projection of hidden states for visualization | 2.5/5 |

### M9: TRANSPOSE — Change coordinate system

| Target | Transposed Variant | Quality Score |
|--------|-------------------|---------------|
| A03 (velocity) | Fourier-domain velocity: predict frequency components, steer in freq space | 3.0/5 |
| A20 (manifold) | Steering in PCA space → project back to original | 3.5/5 |
| C05 (contrastive) | Contrast in logit space (logit diff) instead of hidden state space | 2.5/5 |

### M10: NEGATE — Remove component

| Target | Negated Variant | Impact |
|--------|----------------|--------|
| A09 (death layer exclusion) | Remove death layers from steering set entirely | Eliminates main failure mode |
| J03 (steering skip) | Skip KV modification when TT prediction is low-confidence | Prevents harmful steering |
| A08 (trim-tab only) | Only steer trim-tab layers, ignore neutral layers | Simplifies + focuses |

### M11: RANDOMIZE — Stochastic variants

| Target | Randomized Variant | Quality Score |
|--------|-------------------|---------------|
| A05 (α) | α ~ Uniform(0, 0.2) per step (exploration) | 2.0/5 |
| J03 (steering timing) | Steer with probability p instead of always | 2.5/5 |
| A03 (velocity) | v = TT(x) + ε where ε ~ N(0, σ²) (noise injection for robustness) | 3.0/5 |

### M12: OSCILLATE — Alternating patterns

| Target | Oscillating Variant | Quality Score |
|--------|---------------------|---------------|
| A04 (KV mod) | Alternate: steer L8 on odd tokens, L2 on even tokens | 3.0/5 |
| A05 (α) | α = α_base * sin(π * t / T) (scheduled strength) | 3.5/5 |
| J07 (death layer) | Steer death layer every other token (test if damage is cumulative) | 3.2/5 |

## Forced Collisions

### Speculative Analogues (10 per key atom)

**For A04 (KV modification)**:
1. "What if steering modified the attention logits directly (before softmax) instead of K/V values?"
2. "What if steering added a bias term to the attention output?"
3. "What if steering scaled existing KV cache entries (multiplication instead of addition)?"
4. "What if steering was applied to a COPY of the KV cache (clean + steered branch, merge at output)?"
5. "What if steering modified the position embeddings instead of content embeddings?"
6. "What if steering was applied to the FFN intermediate activations?"
7. "What if steering modified the LayerNorm parameters?"
8. "What if steering was applied to the logits directly (logit perturbation)?"
9. "What if steering modified the sampling temperature at steered positions?"
10. "What if steering replaced the KV cache entry with a retrieved exemplar from a memory bank?"

**For A08 (trim-tab)**:
1-10: Different layer(s), datasets, α values, model sizes, steering surfaces, training regimes, etc.

**For A09 (death layer)**:
1-10: Different ways to detect/avoid/convert death layers.

### Orthogonal Mechanisms (5 per Master Regulator)

**For MR#1 (Per-layer α)**:
1. Evolutionary strategy: CMA-ES to optimize 28 α values
2. Bayesian optimization: Gaussian process over α-space
3. Gradient-based: α = α - η * ∇_α L (requires differentiable generation loop)
4. Zero-order: SPSA (simultaneous perturbation) for α optimization
5. CEM (cross-entropy method): Sample α distributions, keep top-k

**For MR#2 (Contrastive signal)**:
1. Train TT on confidence-weighted: v_hat = Σ w_i * v_i where w_i = P(correct|trajectory)
2. Direct normative loss: L = ||v_pred||² - λ * accuracy(v_pred) (maximize steering impact)
3. Use a critic network: train a "steering quality evaluator" separately
4. Contrastive in vector-quantized space: classify trajectory mode, steer toward desired mode prototype
5. RL with accuracy reward: treat steering as actions, accuracy as reward

**For MR#3 (Capability threshold)**:
1. Synthetic capability injection: insert correct answer template into KV cache
2. Chain-of-thought bootstrapping: steer reasoning steps before answer
3. Multi-turn steering: iteratively refine over multiple generation passes
4. Prompt engineering: reformulate problem to match model's capability profile
5. Mixture-of-experts: route to model that can handle the problem

### Paradoxical Combinations

**PC-1**: "Assume the optimal steering strategy is to NOT steer any layer, but instead to modify the INPUT to the model (prompt engineering)." This tests whether hidden-state steering is fundamentally more effective than input optimization. If true, all velocity prediction is noise.

**PC-2**: "Assume death layers are actually the most informative layers — they show where the model is most sensitive to perturbation, and the negative signal is a protective mechanism." This reframes death layers as "canaries in the coal mine" and suggests steering should be designed to NOT disturb these layers.

**PC-3**: "Assume the TT is unnecessary — a random vector with the right α applied to the right layer produces similar improvements." This is the most dangerous counter-assumption: if true, the entire velocity prediction infrastructure is wasted, and a cheaper baseline (random steering) would be just as effective.
