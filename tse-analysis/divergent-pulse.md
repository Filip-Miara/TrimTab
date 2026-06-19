# Phase 4: Divergent Pulse

## Seed Expansion + Mutation Operators + Forced Collisions

---

## Seed Expansion

### Semantic Constellation

| Atom | Analogous Concept | Domain | Quality |
|------|-------------------|--------|---------|
| D6 (Global Norm) | Batch normalization in vision | Computer Vision | High — same math structure |
| A1 (6-layer Transformer) | Shallow decoder in image generation | Generative AI | Medium — different task |
| T8 (MSE Loss) | L2 loss in optical flow | Video Analysis | High — identical math |
| C1 (Velocity) | Optical flow in video | Computer Vision | High — delta between frames/layers |
| F4 (AWQ Shift) | Domain shift in medical imaging | Medical AI | High — covariate shift |
| T5 (Double Buffer) | CPU→GPU texture streaming | Graphics | Medium — pipeline structure |

### Cryptic Analogy Mining

| Atom | Abstract Restatement | Analogous Domain | Candidate |
|------|---------------------|------------------|-----------|
| D6 | "One summary statistic for a heterogeneous population" | Economics | Gini coefficient / Lorenz curve as normalization |
| A7 (Bidir) | "Seeing past and future to predict present" | Time-series forecasting | Kalman smoother (bidirectional filtering) |
| T8 (MSE) | "Punish deviations quadratically regardless of direction" | Control theory | LQR cost function |
| C1 (Velocity) | "Rate of change of internal state per processing step" | Dynamical systems | Lyapunov exponent of hidden state trajectory |
| F4 (AWQ) | "Measurement error changes the observable but not the underlying state" | Sensor fusion | Kalman filter innovation signal |

### Pre-Seed Refinements

| Original Pre-Seed | Refined Version |
|-------------------|-----------------|
| Per-layer adaptive normalization | LayerNorm applied to each layer's hidden states separately + learned affine transform per layer |
| Multi-objective loss | Directional loss (cosine) + magnitude loss (Huber, not MSE) + domain confusion loss (gradient reversal) |
| Online trajectory generation | Qwen forward pass at training time with gradient checkpointing; store only velocity targets |
| LoRA-style adapters | Instead of external steering, train low-rank perturbations to Qwen's attention outputs |
| Depth-adaptive architecture | Router network predicts per-sample compute budget; allocate TT layers dynamically |

---

## Mutation Operators

### M1: SUBSTITUTE — Replace global norm with per-layer norm
| Variant | Quality |
|---------|---------|
| V1.1: Per-layer z-score normalization (mean=0, std=1 per layer) | Novelty: 3/5, Feasibility: 5/5, Coherence: 5/5, Risk: 1/5, Emergent: 2/5 |

### M2: INVERT — Use unnormalized data instead of normalized
| Variant | Quality |
|---------|---------|
| V2.1: Raw hidden states + learned input normalization via LayerNorm | Novelty: 2/5, Feasibility: 4/5, Coherence: 4/5, Risk: 3/5, Emergent: 2/5 |

### M3: SCALE — Change TT capacity
| Variant | Quality |
|---------|---------|
| V3.1: 12-layer TT (2× depth), d_model=1024 | Novelty: 1/5, Feasibility: 4/5, Coherence: 4/5, Risk: 2/5, Emergent: 3/5 |
| V3.2: 3-layer TT (½ depth), d_model=1536 (wide) | Novelty: 2/5, Feasibility: 5/5, Coherence: 3/5, Risk: 2/5, Emergent: 1/5 |
| V3.3: Depth-adaptive: 2-12 layers depending on velocity complexity | Novelty: 4/5, Feasibility: 2/5, Coherence: 3/5, Risk: 4/5, Emergent: 4/5 |

### M4: REORDER — Change attention mechanism order
| Variant | Quality |
|---------|---------|
| V4.1: Causal attention only (as tested, worse) | Novelty: 1/5, Feasibility: 5/5, Coherence: 5/5, Risk: 2/5, Emergent: 1/5 |
| V4.2: Bidirectional encoder → causal decoder (hybrid) | Novelty: 4/5, Feasibility: 3/5, Coherence: 4/5, Risk: 2/5, Emergent: 3/5 |
| V4.3: Mamba (SSM) replaces attention entirely | Novelty: 4/5, Feasibility: 4/5, Coherence: 4/5, Risk: 3/5, Emergent: 4/5 |

### M5: MERGE — Combine input projection and output prediction
| Variant | Quality |
|---------|---------|
| V5.1: Skip the bottleneck: direct 3584→3584 via single linear layer + residual | Novelty: 3/5, Feasibility: 5/5, Coherence: 4/5, Risk: 2/5, Emergent: 1/5 |
| V5.2: U-Net style: down-project 3584→768, process, up-project 768→3584 with skip connection | Novelty: 3/5, Feasibility: 4/5, Coherence: 5/5, Risk: 2/5, Emergent: 3/5 |

### M6: SPLIT — Decompose velocity prediction into sub-tasks
| Variant | Quality |
|---------|---------|
| V6.1: Predict direction (cosine) and magnitude separately with two heads | Novelty: 4/5, Feasibility: 4/5, Coherence: 5/5, Risk: 2/5, Emergent: 3/5 |
| V6.2: Group layers into 3-4 blocks; predict per-block velocities | Novelty: 3/5, Feasibility: 4/5, Coherence: 4/5, Risk: 2/5, Emergent: 2/5 |
| V6.3: Predict per-head attention differences instead of per-layer | Novelty: 5/5, Feasibility: 2/5, Coherence: 3/5, Risk: 4/5, Emergent: 4/5 |

### M7: ABSTRACT — Predict at higher level
| Variant | Quality |
|---------|---------|
| V7.1: PCA on velocity targets → predict top-256 components | Novelty: 4/5, Feasibility: 4/5, Coherence: 5/5, Risk: 2/5, Emergent: 3/5 |
| V7.2: Predict velocity in QK/OV circuit space instead of residual stream | Novelty: 5/5, Feasibility: 1/5, Coherence: 3/5, Risk: 5/5, Emergent: 5/5 |

### M8: CONCRETIZE — Add explicit layer-specific features
| Variant | Quality |
|---------|---------|
| V8.1: Add layer index as a learnable embedding concatenated to input | Novelty: 3/5, Feasibility: 5/5, Coherence: 5/5, Risk: 1/5, Emergent: 2/5 |
| V8.2: Add attention pattern statistics (entropy, max attn) as features | Novelty: 4/5, Feasibility: 3/5, Coherence: 4/5, Risk: 2/5, Emergent: 3/5 |

### M9: TRANSPOSE — Use different model family
| Variant | Quality |
|---------|---------|
| V9.1: Replace TransformerEncoder with MambaMixer (Mamba + MLP) | Novelty: 4/5, Feasibility: 3/5, Coherence: 4/5, Risk: 4/5, Emergent: 4/5 |
| V9.2: Use Graph Neural Network — layers are nodes, attention is edges | Novelty: 5/5, Feasibility: 2/5, Coherence: 3/5, Risk: 5/5, Emergent: 5/5 |

### M10: NEGATE — What makes velocity prediction HARDER
| Variant | Quality |
|---------|---------|
| V10.1: Remove normalization entirely; force TT to learn scale internally | Novelty: 3/5, Feasibility: 4/5, Coherence: 3/5, Risk: 4/5, Emergent: 2/5 |
| V10.2: Add random noise to training velocities (regularization) | Novelty: 2/5, Feasibility: 5/5, Coherence: 4/5, Risk: 2/5, Emergent: 1/5 |

### M11: RANDOMIZE — Bayesian approach
| Variant | Quality |
|---------|---------|
| V11.1: Predict velocity distribution (mean + variance) instead of point estimate | Novelty: 4/5, Feasibility: 3/5, Coherence: 4/5, Risk: 3/5, Emergent: 4/5 |
| V11.2: Bayesian TT with Monte Carlo dropout for uncertainty | Novelty: 3/5, Feasibility: 4/5, Coherence: 4/5, Risk: 3/5, Emergent: 3/5 |

### M12: OSCILLATE — Alternating training objectives
| Variant | Quality |
|---------|---------|
| V12.1: Interleave MSE and cosine loss training epochs | Novelty: 3/5, Feasibility: 5/5, Coherence: 4/5, Risk: 2/5, Emergent: 2/5 |
| V12.2: Alternate velocity prediction with actual steering in a closed loop | Novelty: 5/5, Feasibility: 2/5, Coherence: 3/5, Risk: 5/5, Emergent: 5/5 |

---

## Forced Collisions

### Speculative Analogues (10 per key atom)

**For D6 (Global Norm)**:
1. Group normalization (divide features into groups)
2. Instance normalization (per-sample)
3. Layer normalization (per-sample per-layer)
4. Adaptive instance normalization (AdaIN) from style transfer
5. Whitening transform (PCA-based decorrelation)
6. Quantile normalization (non-parametric alignment)
7. Power transform (Box-Cox for non-Gaussian distributions)
8. Robust normalization (median/IQR instead of mean/std)
9. Spectral normalization (constrain Lipschitz constant)
10. Gradient normalization (GradNorm for multi-task balance)

**For T8 (MSE Loss)**:
1. Cosine proximity loss
2. Huber loss (L1-L2 hybrid)
3. Correlation coefficient loss (Pearson)
4. Spearman rank correlation loss
5. InfoNCE / contrastive loss
6. Energy-based loss (margin ranking)
7. Earth mover's distance (Wasserstein)
8. Structural similarity (SSIM) adapted to 1D
9. Focal loss (down-weight easy examples)
10. Triplet loss (anchor-positive-negative)

### Orthogonal Mechanisms per Master Regulator

**For MR-1 (Normalization)**:
1. Adaptive: small network that predicts optimal normalization per layer
2. Competitive: train with/without normalization, ensemble
3. Residual: keep global norm but add per-layer residual features
4. Hierarchical: normalize layers in groups (early/mid/late)
5. Dynamic: different normalization per training phase (warmup vs fine-tune)

**For MR-2 (Loss)**:
1. Curriculum: start with MSE, transition to cosine
2. Adversarial: discriminator that detects unrealistic velocities
3. Multi-scale: predict coarse then fine velocities with different losses
4. Student-teacher: MSE vs soft targets from exponential moving average
5. Meta-loss: learn the loss function via hypernetwork

### Paradoxical Combinations (3)

**PC-1**: "Assume the solution is the EXACT OPPOSITE of the accepted approach"
- Accepted: Train TT on BnB Qwen with MSE, normalize globally
- Opposite: Train TT on ADVERSARIAL (worst-case) trajectories designed to maximize velocity error; use contrastive loss; anti-normalize (amplify differences)
- Rationale: Training on the hardest cases may build robustness; TT may learn velocity invariants instead of surface patterns

**PC-2**: "Assume the TT should NOT predict velocity"
- Accepted: Predict velocity (delta between consecutive layers)
- Opposite: Predict hidden state directly at layer L using only layers 1..L-1 (next-step prediction), then compute velocity from two predictions
- Rationale: TT becomes a forward model of Qwen's next hidden state. Velocity is derived, not predicted. This removes the exposure bias.

**PC-3**: "Assume Qwen doesn't need steering at all; the TT should predict WHEN to steer, not HOW"
- Accepted: TT predicts continuous velocity values
- Opposite: TT predicts a binary "steer/no-steer" flag per layer per token; velocity is predefined template
- Rationale: Perhaps steering is not a continuous function but a sparsely triggered correction. 95% of tokens may not need steering.
