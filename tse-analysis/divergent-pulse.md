# Divergent Pulse — RankAdaptation

## 1. Seed Expansion

### Analogues Found
| Atom | Analogous Concept | Domain | Relevance |
|------|------------------|--------|-----------|
| A3 (Velocity) | Momentum in optimization | ML (SGD with momentum) | Velocity = direction of state change, momentum = direction of gradient update |
| A4 (TT) | Forward model in control theory | Robotics | Predicts next state given current state and action (steering = action) |
| A6 (Trim-tab) | Gain scheduling | Control theory | Different α for different operating conditions (problem types) |
| A7 (Death layer) | Mode collapse | GAN training | Single intervention destroys all useful computation |
| A11 (Contrastive) | Siamese networks | ML | Learning by comparing pairs; NT-Xent loss could train TTs directly |
| A14 (Attention head) | Cortical column | Neuroscience | Specialized computational unit within a larger processing region |

### Cryptic Analogies
| Atom | Abstract Function | Analogous Domain | Analogous Entity |
|------|-------------------|------------------|------------------|
| A5 (α·v) | Scale a direction vector | Economics | Leverage (financial) — small multiplier with outsized effect |
| J2 (TT→h') | Map observation to correction | Medicine | Diagnosis → prescription |
| J4 (Cache→token) | Modify memory to change future inference | Law | Precedent-setting — modify past record to influence future rulings |
| C3-3 (per-layer) | Selectively modify components | Engineering | Circuit breaker — isolate faulty components |

### Hallucinatory Pre-Seed Refinements
- **A4 ideal**: TT that jointly predicts velocity AND classifies correct/incorrect → single model for both descriptive and normative
- **A5 ideal**: α is not a scalar but a learned function α(h[l], context, problem_difficulty)
- **A15 ideal**: No capability threshold — even random models have steerable hidden states via different mechanisms

## 2. Mutation Operators

### M1: SUBSTITUTE
| Original | Substitution | Quality | Rationale |
|----------|-------------|---------|-----------|
| A3: h[l+1] - h[l] | h[l+2] - h[l] (skip-layer velocity) | 4/5 | Captures longer-range dynamics |
| A4: Transformer | Mamba (SSM) for velocity prediction | 3/5 | Linear-time alternative |
| A5: Additive steering | Multiplicative steering: h' = h ⊙ (1 + α·v) | 3/5 | Different geometric operation |
| A10: Binary accuracy | Continuous: expected correct tokens | 4/5 | Finer gradient signal |
| A15: 40% threshold | Per-task threshold (math=40%, code=20%) | 3/5 | Task-specific capability |

### M2: INVERT
| Original | Inversion | Quality | Risks |
|----------|-----------|---------|-------|
| α > 0 (steer toward v) | α < 0 (steer away from v) | 4/5 | May convert trim-tab to death-layer |
| Steer at L8 (+20pp) | Steer at L9 (death layer) with α < 0 | 5/5 | Flipping death layer sign may make it trim-tab |
| v = standard TT prediction | v = -standard TT prediction | 2/5 | Probably catastrophic |
| Controllable: steer known layers | Uncontrollable: steer all layers | 1/5 | Death layers dominate |

### M3: SCALE
| Variant | Scale Factor | Expected Effect |
|---------|-------------|-----------------|
| Whole-model α | α ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0} | Goldilocks zone 0.01-0.3 |
| Per-layer α | 28 independent α values | High-dimensional optimization |
| Per-token α(t) | α decays exponentially from t=0 | Exploit first-token leverage then stabilize |
| Per-head α | 28 layers × 28/4 heads | Very high-dim but precise |

### M4: MERGE
| Merge | Result | Quality |
|-------|--------|---------|
| A4 (TT) + A12 (Reading head) | TT that outputs (v, ppl_pred) jointly | 4/5 — shared representation benefits both |
| A6 (Trim-tab) + A8 (α) | Learned mapping: layer → optimal α | 4/5 — eliminates α sweep |
| C3-2 (Steering) + C3-5 (Confidence gating) | Steer only when uncertain | 5/5 — PPL-modulated steering |
| A11 (Contrastive) + A5 (α) | β·(v_c - v_i) combined with standard steered α·v_std | 4/5 — hybrid descriptive-normative |

### M5: SPLIT
| Original | Split | Rationale |
|----------|-------|-----------|
| A3: velocity v[l] | v_pos[l] + v_neg[l] (positive and negative components) | Directional steering |
| C3-2: KV steering | K-steering vs V-steering separately | Different effects |
| A4: single TT | TT_ensemble = {TT_1, ..., TT_K} with bagging | Reduced variance |
| A10: GSM8K accuracy | Per-step + per-problem accuracy | Finer evaluation |

### M6: ABSTRACT
| Abstract | Concrete(s) |
|----------|-------------|
| Hidden state modifier | KV-cache modification, logit correction, MLP activation perturbation, residual stream editing |
| Velocity predictor | TrajectoryTransformer, Perceiver, Mamba, linear probe |
| Layer selector | Per-layer sweep, per-head sweep, attention-block sweep, MLP-only sweep |

### M7: CONCRETIZE
| Abstract | Concrete Implementation |
|----------|------------------------|
| Confidence gate | α(t) = α_max · σ(γ · (τ - ppl_pred(t))) where τ is threshold |
| Multi-layer combo | Bayesian optimization over 28-dimensional α space |
| Per-token α | α(t) = min(α_max, α_max · (1 - t/T)) where T = max tokens |

### M8: NEGATE
| Statement | Negation | Validity |
|-----------|----------|----------|
| Steering requires capability | Steering is possible on any model with different mechanism | Partially true — different mechanism (e.g., noise injection for random models) |
| Trim tabs are layer-specific | Trim tabs are function-specific (same layer, different function per input) | Could be true — L8 handles certain reasoning patterns |
| Contrastive improves steering | Contrastive cancels shared structure and reduces signal | Testable — compare v_c - v_i vs v_std alone |

## 3. Forced Collisions

### Speculative Analogues (10 per atom for key atoms)

**For A3 (Velocity)**:
1. Curriculum learning velocity: v changes as generation progresses (early tokens = exploration, late = exploitation)
2. Quantized velocity: only top-K velocity dimensions matter for steering
3. Frequency-decomposed velocity: low-freq = global direction, high-freq = token-specific noise
4. Momentum velocity: exponential moving average of v across steps
5. Gradient velocity: v = ∇_h P(correct|h) (steepest ascent toward correctness)
6. Normalized velocity: v/||v|| for pure direction regardless of magnitude
7. Clipped velocity: clamp v to [-σ, σ] per dimension for stability
8. Sparse velocity: only modify K dimensions of h (most influential)
9. Adversarial velocity: v = argmax_{||δ||<ε} P(wrong|h+δ) — steer toward errors (for testing)
10. Residual velocity: velocity of the residual stream (h - h_prev) instead of layer-to-layer

### Orthogonal Mechanisms (for Master Regulators)

**MR #1 (L8 steering)**:
1. Per-attention-head steering within L8 (28 heads, only some trim-tabs)
2. L8 MLP-only steering (bypass attention modification)
3. L8 steering with annealed α over generation steps
4. L8 combined with L7 (upstream) and L9 (downstream) with phased scheduling
5. L8 inverse steering (α < 0) to test symmetry

**MR #2 (Contrastive direction)**:
1. Weighted contrastive: 0.7·v_c - 0.3·v_i (asymmetric push-pull)
2. Normalized contrastive: normalize both v_c and v_i to unit vectors before subtraction
3. Ensemble contrastive: v_ensemble = mean(v_c_i - v_i_i) over bootstrapped TT pairs
4. Soft contrastive: v = v_std + β·(v_c - v_i) with β optimized
5. Adversarial contrastive: train TT_incorrect adversarially against TT_correct

### Paradoxical Combinations (3 total)

**Combination 1: "Steer the model toward the answer it would have given anyway"**
- Paradox: Standard TT predicts the velocity the model would naturally follow. Steering with this velocity amplifies the existing trajectory — both correct and incorrect ones. The improvement from L8 suggests the model already has the correct answer latent in its hidden states, and steering "helps it get there." The ultimate paradox: **the steering vector that most improves accuracy is the one that least changes the model's behavior** — it just removes obstacles (death-layer interference) from the model's own correct trajectory.

**Combination 2: "Train TT on steered data"**
- Paradox: To have the best TT for steering, you need to train it on data from steered generations. But to generate good steered data, you need a good TT. This chicken-egg problem suggests an iterative co-adaptation process: TT → steer → collect data → train TT → steer → ...

**Combination 3: "The reading head gates steering, but reading head accuracy depends on Perceiver latents, which are computed from unsteered hidden states"**
- Paradox: The confidence gate (reading head) uses frozen Perceiver latents trained on unsteered data. Under steering, hidden states change, making the Perceiver latents less reliable. The confidence gate's accuracy degrades exactly when steering makes it most needed.
