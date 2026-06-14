# Phase 4: Divergent Pulse

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Part 1: Seed Expansion

### Semantic Constellation — Structurally/Functionally Analogous Entities

| Analog | Domain | Mapping to Steering System |
|--------|--------|---------------------------|
| Cruise Control | Automotive | TT is the speed sensor, α is the throttle, trim-tab layers are the accelerator pedal |
| Cochlear Implant | Medical | Stimulating specific electrodes (layers) to improve hearing (reasoning); some electrodes are trim-tabs, some cause discomfort (death layers) |
| Gradient Descent | Optimization | Steering is taking a gradient step in hidden state space; α is the learning rate; trim-tab layers have high gradient signal |
| Deep Brain Stimulation | Neuroscience | Electrical stimulation at specific brain regions (layers) modulates neural activity; DBS targets are identified empirically (like trim-tab sweeps) |
| Active Noise Cancellation | Audio | Measuring ambient noise (current hidden state), generating anti-noise (predicted velocity), canceling at specific frequencies (layers) |
| Autotune | Audio | Pitch correction (velocity steering) tuned per frequency band (per layer); over-correction creates artifacts (token divergence, death layers) |
| Genetic Engineering | Biology | CRISPR targets specific genes (layers) for modification; some genes (trim-tabs) have consistent beneficial effects; knockouts (death layers) are catastrophic |
| Magnetic Resonance Imaging | Medical | Different gradient coils (layers) produce different spatial encoding; optimizing coil currents (α) improves image quality (accuracy) |

### Hallucinatory Pre-Seed Refinements

| Atom | Refined Pre-Seed | Real-World Approximation |
|------|-----------------|------------------------|
| A1 (Hidden State) | Hidden states are structured as a Riemannian manifold with known curvature | Currently unmeasured — could compute via intrinsic dimension estimation |
| A2 (Velocity) | Velocity is the gradient of a latent "correctness" potential field | Can test by computing if v points toward higher-probability correct tokens |
| A5 (α) | Learned α has a characteristic "activation function" shape per layer: sigmoid over token position | Could fit a 3-parameter sigmoid: α(t) = α_max / (1 + exp(-k(t - t₀))) |
| A6 (Trim-Tab) | Trim-tab layers are those with high "circuit importance" — they participate in many downstream computations | Could approximate with activation patching — measure effect of zeroing each layer on accuracy |
| A9 (Contrastive) | Contrastive signal is the "reasoning direction" in latent space — it points toward correct reasoning regardless of task | Test by computing v_correct − v_incorrect on non-math tasks to check task-specificity |

### Cryptic Analogy Mining

Abstract restatement of the problem: **"Given a high-dimensional dynamical system with a trained forward model of local derivatives, apply scalar-gated perturbation at specific spatial locations to shift the system's trajectory toward a goal state, subject to the constraint that some locations amplify noise when perturbed."**

| Abstract Form | Domain Match | Transferred Insight |
|--------------|--------------|-------------------|
| "Locally-adaptive perturbation of high-D dynamical systems" | Plasma physics — magnetic confinement fusion | Steering is analogous to applying localized magnetic fields to control plasma instabilities; death layers are "tearing modes" where perturbation destabilizes the plasma |
| "Scalar-gated neural trajectory modulation with location-specific sensitivity" | Robotics — joint-space control | Each layer is a joint; α is the torque; trim-tabs are joints near the end effector (high leverage); death layers are base joints (small perturbation causes large, uncontrolled movement) |
| "Conditional information routing with selective amplification" | Telecommunications — beamforming | Steering at L8 is like aligning antenna phase for maximum signal; death layers are interferers requiring null steering |

---

## Part 2: Synthetic Variants (M1-M12 Mutation Operators)

### M1: SUBSTITUTE

| ID | Original | Substitution | Synthetic Variant | Novelty | Feasibility | Coherence | Risk | Emergent Potential |
|----|----------|-------------|-------------------|---------|-------------|-----------|------|-------------------|
| SV-1 | Linear α·v steering | Non-linear steering: v·tanh(α·||v||) | **Tanh-Clamped Steering**: Clamp large velocity predictions to prevent OOD hidden states | 3 | 5 | 5 | 1 | 2 |
| SV-2 | Single TT | Ensemble of N bootstrapped TTs | **Bagged TT Steering**: Average predictions from TTs trained on different data subsets | 4 | 4 | 5 | 2 | 4 |
| SV-3 | KV-cache steering | Activation steering on residual stream | **Residual Activation Steering**: Modify hidden states directly instead of KV cache | 4 | 3 | 4 | 3 | 5 |
| SV-4 | GSM8K evaluation | Multi-task evaluation suite | **Multi-Task Steering Validation**: Test on ARC, BBH, MMLU, GSM8K simultaneously | 3 | 5 | 5 | 1 | 3 |

### M2: INVERT

| ID | Original | Inversion | Synthetic Variant | Novelty | Feasibility | Coherence | Risk | Emergent Potential |
|----|----------|-----------|-------------------|---------|-------------|-----------|------|-------------------|
| SV-5 | Steer toward v_pred | Steer AWAY from v_pred for death layers | **Negative α Steering**: Apply α < 0 to death layers to push hidden states away from harmful trajectory | 5 | 5 | 4 | 2 | 5 |
| SV-6 | α uniform across layers | α as learned vector per layer | **Learned Layerwise α**: Train a small network to predict optimal α for each layer given context | 4 | 4 | 4 | 2 | 4 |
| SV-7 | Train on correct+incorrect | Train only on high-confidence trajectories | **High-Confidence TT**: Train on trajectories where model was most confident (correct or incorrect) | 3 | 4 | 4 | 2 | 3 |

### M3: SCALE

| ID | Original | Scaling | Synthetic Variant | Novelty | Feasibility | Coherence | Risk | Emergent Potential |
|----|----------|---------|-------------------|---------|-------------|-----------|------|-------------------|
| SV-8 | Fixed α=0.1 | α sweep over 0.001-10.0 | **Multi-Scale α Sweep**: 100× range to find optimal steering magnitude per layer | 2 | 5 | 5 | 1 | 2 |
| SV-9 | 100 problems/layer | 1000+ problems/layer | **High-Statistics Validation**: Reduce binomial noise on per-layer accuracy estimates | 2 | 4 | 5 | 1 | 1 |
| SV-10 | Single layer | All pairs of layers | **Pairwise Steering**: Evaluate all (L_i, L_j) combinations for synergistic effects | 3 | 3 | 4 | 2 | 5 |

### M4: REORDER

| ID | Original | Reordering | Synthetic Variant | Novelty | Feasibility | Coherence | Risk | Emergent Potential |
|----|----------|-----------|-------------------|---------|-------------|-----------|------|-------------------|
| SV-11 | Steer at all generation steps | Steer only at specific token positions | **Position-Gated Steering**: Only apply steering at "reasoning tokens" (e.g., before arithmetic operations) | 4 | 4 | 4 | 2 | 5 |
| SV-12 | Independent layer sweeps | Sequential layer sweeps | **Sequential Layer Discovery**: Find first trim-tab, add layers one at a time in order of predicted importance | 3 | 5 | 4 | 1 | 3 |

### M5: MERGE

| ID | Original | Merge | Synthetic Variant | Novelty | Feasibility | Coherence | Risk | Emergent Potential |
|----|----------|-------|-------------------|---------|-------------|-----------|------|-------------------|
| SV-13 | TT + PPL gate | TT with integrated uncertainty | **Uncertainty-Aware TT**: Train TT to predict both velocity AND prediction uncertainty (evidential regression) | 5 | 3 | 4 | 3 | 5 |
| SV-14 | Standard + Contrastive TT | Combined steering vector | **Dual-Mode Steering**: v_combined = v_standard + β·(v_correct − v_incorrect) with β as learned parameter | 4 | 5 | 5 | 2 | 4 |

### M6-M12: Additional Variants

| ID | Operator | Variant | Novelty | Feasibility | Coherence | Risk | Emergent Potential |
|----|----------|---------|---------|-------------|-----------|------|-------------------|
| SV-15 | SPLIT (head-level) | **Head-Level Steering**: Apply different α to different attention heads within a layer | 4 | 3 | 4 | 3 | 5 |
| SV-16 | ABSTRACT (functional) | **Functional Steering Layer**: Instead of per-layer, define steering targets by functional role (attention heads that attend to reasoning tokens) | 5 | 2 | 3 | 4 | 5 |
| SV-17 | CONCRETIZE (element-wise) | **Element-Wise α**: Different steering magnitude for each dimension of the hidden state | 3 | 4 | 3 | 3 | 3 |
| SV-18 | TRANSPOSE (generation↔embedding) | **Embedding Steering**: Apply steering at the embedding layer instead of during generation | 3 | 4 | 4 | 3 | 3 |
| SV-19 | NEGATE (noise) | **Stochastic Steering**: Add noise sampled from TT's error distribution instead of predicted velocity | 4 | 5 | 3 | 2 | 2 |
| SV-20 | RANDOMIZE (α) | **Random α Per Step**: Sample α from a distribution at each generation step = stochastic control | 3 | 5 | 3 | 2 | 3 |
| SV-21 | OSCILLATE (periodic) | **Cyclic Steering**: Switch between steering and no-steering in blocks of tokens | 3 | 5 | 4 | 1 | 3 |

---

## Part 3: Forced Collisions

### Speculative Analogues (10 per Atom)

**For A5 (Steering Coefficient α)**:
1. α as a temperature parameter for hidden state softmax
2. α as a Kalman gain — adaptive per-step
3. α as a synaptic weight in a neural circuit model
4. α as a PID controller gain with history-dependent scheduling
5. α as a risk parameter in portfolio optimization
6. α as a mutation rate in evolutionary strategies
7. α as a learning rate with cosine decay over generation steps
8. α as a forgetting factor in online learning
9. α as a confidence-weighted update in Bayesian inference
10. α as a gate opening in attention (like AFT)

**For A6 (Trim-Tab Layer)**:
1. Trim-tab as a "critical point" in the hidden state phase space
2. Trim-tab as a layer with high "betweenness centrality" in the computational graph
3. Trim-tab as a layer near an "attention sink" where model stores global context
4. Trim-tab as a layer performing "next-token-prediction head" function
5. Trim-tab as a layer with low "functional redundancy" — no backup circuits
6. Trim-tab as a layer with high "mutual information" with correct answer tokens
7. Trim-tab as a layer that "routes" information from early to late layers
8. Trim-tab as a layer with specific "semantic head" (e.g., arithmetic operations)
9. Trim-tab as a layer where the "velocity norm" is consistently high across inputs
10. Trim-tab as a layer where the "attention pattern" distinguishes correct/incorrect paths

**For A9 (Contrastive Signal)**:
1. Contrastive signal as the "Helmholtz free energy" difference between correct and incorrect trajectories
2. Contrastive signal as the "log-odds" of correct vs incorrect reasoning
3. Contrastive signal as a "policy gradient" in RL — direction to increase probability of correct tokens
4. Contrastive signal as the "Fisher information matrix" direction — most informative perturbation
5. Contrastive signal as a "counterfactual" — what should have been different
6. Contrastive signal as "adversarial perturbation" toward correct answers
7. Contrastive signal as "direct preference optimization (DPO)" gradient
8. Contrastive signal as "contrastive divergence" in energy-based models
9. Contrastive signal as "prototypical network" centroid direction
10. Contrastive signal as "canonical correlation" between correct and incorrect trajectories

### Orthogonal Mechanisms per Master Regulator

**MR1: Contrastive Signal — 5 orthogonal mechanisms**:
1. **Ranking-based steering**: Instead of subtractive contrastive, train a ranker that orders trajectories by quality, then steer toward higher-ranked trajectories
2. **Adversarial TT**: Train a discriminator to distinguish correct/incorrect trajectories; use its gradient as the steering direction
3. **Direct preference optimization for steering**: Formulate steering as a preference optimization problem — maximize probability of correct tokens under steered hidden states
4. **Contrastive with hard negative mining**: Only use trajectories where the model was highly confident but wrong (hard negatives) for the contrastive signal
5. **Multi-step contrastive**: Instead of single-velocity difference, compute the difference in multi-step trajectory predictions

**MR2: Per-Layer Selectivity — 5 orthogonal mechanisms**:
1. **Head-level selector**: Learn binary mask per attention head instead of per layer
2. **Attention-pattern-based gating**: Only steer at layers where attention pattern changes significantly between correct/incorrect generations
3. **Functional cluster steering**: Group layers by functional similarity (via probing or activation similarity), steer the cluster centroid
4. **Learnable layer mask via Gumbel-Softmax**: Differentiable binary mask that selects layers to steer
5. **Sparse steering**: Use L1 regularization to select a sparse subset of layers for steering

**MR3: Capability Threshold — 5 orthogonal mechanisms**:
1. **Fine-tune first, steer second**: Quick fine-tuning to bring model above threshold, then apply steering
2. **LoRA steering**: Instead of KV-cache steering, train LoRA adapters that simulate the steering effect during training
3. **Cross-capability steering**: Use a larger model's TT to steer a smaller model (extending cross-model transfer)
4. **Curriculum steering**: Start with easy problems, gradually increase difficulty as steering kicks in
5. **Multiplicative steering**: Instead of additive (α·v), use multiplicative gating that amplifies existing signal rather than adding new signal

### Paradoxical Combinations

**Paradox 1: "The Death Layer Amplifier"**
- **Concept**: What if death layers are not harmful but are actually the KEY to large improvements — we just need to apply NEGATIVE α? Instead of steering L9 toward its predicted trajectory, steer it AWAY. The -23pp at +α might become +23pp at -α.
- **Risk**: If wrong, performance collapses completely
- **Novelty**: 5/5

**Paradox 2: "The Anti-Steering Steering"**
- **Concept**: What if the best steering is NOT to modify the KV cache but to PREVENT the LM from changing certain hidden states? The TT predicts where the state will go; instead of pushing it FURTHER in that direction, we DAMPEN the predicted movement (apply negative velocity to counteract overconfident trajectory).
- **Risk**: Low risk because dampening is conservative
- **Novelty**: 5/5

**Paradox 3: "The Contradiction Steering"**
- **Concept**: What if we steer L8 using the CONTRASTIVE signal from L9's predictions? Using a death layer's error signal to guide a trim-tab layer's steering direction. The idea: L9 knows what's wrong (even if it can't fix it), and L8 knows what to do about it.
- **Risk**: High — indirect and untested
- **Novelty**: 5/5
