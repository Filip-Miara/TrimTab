# Phase 2: Multi-Lens Analysis Cascade

## 10-Stacked Lens Analysis of TrajectoryTransformer Pipeline

---

## Lens 1: ANALOGICAL — Structural Isomorphisms

**Blind Spot from Prior**: None (first lens).

### Structural Findings
- **TT : Qwen :: Autoencoder : Decoder** — TT compresses 3584-dim velocity space → 768-dim latent → reconstructs 3584. Homologous to autoencoder bottleneck.
- **Global normalization : BatchNorm** — One mean/std across all layers/samples = BatchNorm with population statistics. Mirror of batch-dependent normalization in vision models.
- **Velocity prediction : Optical flow** — Predicting deltas between consecutive layers is structurally identical to optical flow in video (predict pixel displacement between frames).
- **Double-buffer prefetch : CPU->GPU texture streaming** — Standard pipelining in graphics pipelines.

### Relational Findings
- **Steering via velocity : Gradient descent on hidden states** — KV-cache modification via predicted velocity is analogous to taking a gradient step in hidden state space toward "better reasoning."
- **48M parameter predictor for 7B : Small world model for large environment** — Matches predictive coding hierarchy (small model predicts big model's internals).

### Potential Findings
- **Composite mirroring**: What if TT is structured to mirror the Qwen layer hierarchy (6 vs 28 layers)? Thinner mirror with down-projection.
- **Physics simulation analogy**: Velocity + position (hidden state) = Hamiltonian dynamics. Could leverage symplectic integration (Verlet integration preserves energy).
- **Transfer learning challenge : Domain adaptation in vision** — AWQ shift = covariate shift. Solution space from domain adaptation literature directly applies.

### Blind Spot Alert
- **Missing analogy**: TT training is supervised but inference is autoregressive. Supervised velocity prediction ≠ free-running steering. This maps to **exposure bias** in seq2seq literature — the fundamental blind spot.

---

## Lens 2: DIALECTICAL — Thesis/Antithesis/Synthesis

**Blind Spot from Lens 1**: Exposure bias (train vs inference mismatch).

### Finding 1: Normalization Strategy
| Thesis | Antithesis | Synthesis |
|--------|------------|-----------|
| Global normalization (one mean/std across all layers/samples) normalizes the entire distribution uniformly | Layers have systematically different velocity magnitudes (early layers: large activations; late layers: fine-grained); global norm destroys this signal | **Per-layer normalization with global residual**: normalize each layer independently, then project residual = layer_stat - global_stat as additional feature |

### Finding 2: Attention Type
| Thesis | Antithesis | Synthesis |
|--------|------------|-----------|
| Bidirectional attention gives better R², so use it | Causal attention matches inference-time conditions; bidir leaks future information | **Dual-mode training**: bidir for representation learning (encoder), causal for prediction head (decoder). Or: train with bidir, fine-tune with causal. |

### Finding 3: AWQ Transfer
| Thesis | Antithesis | Synthesis |
|--------|------------|-----------|
| Train on BnB → works on BnB; fine-tune on AWQ → forgets BnB | AWQ hidden states are systematically different; no shared representation | **Domain-invariant representation learning**: train on mixed BnB+AWQ data with domain confusion loss (gradient reversal layer) to force shared velocity representations |

### Finding 4: Single Global Model
| Thesis | Antithesis | Synthesis |
|--------|------------|-----------|
| One TT model for all 28 layers | Velocity structure differs dramatically across layers | **Layer-group experts**: 3-4 TT or heads, each specializing in layer groups (1-9, 10-19, 20-28) with soft routing |

### Finding 5: MSE Sufficiency
| Thesis | Antithesis | Synthesis |
|--------|------------|-----------|
| MSE captures both magnitude and direction of velocity | MSE treats all error directions equally; small-angle-high-magnitude errors dominate | **Angular + magnitude decomposition**: predict direction (cosine loss) and magnitude (LogCosh) separately with learned gating |

### Blind Spot Alert
- **Hidden synthesis**: The deepest contradiction is TT as externally learned correction vs Qwen as internally adaptive system. If Qwen already steers its own hidden states during generation, TT is redundant. Need to measure whether Qwen's hidden states already contain "optimal" velocity information.

---

## Lens 3: BLENDING — Conceptual Integration

**Blend 1: TT + Differentiable Neural Computer (DNC)**
- **Input 1**: TT predicts layer-wise velocities from hidden states
- **Input 2**: DNC writes to external memory via attention
- **Blend**: TT predicts *write vectors* for an external memory that stores "useful velocity corrections" keyed by hidden state signatures
- **Emergent**: Memory-augmented TT can retrieve correction patterns seen during training rather than recomputing from scratch

**Blend 2: Velocity Prediction + Mixture of Experts (MoE)**
- **Input 1**: Single TT predicts all 28×3584 velocities
- **Input 2**: Top-k routing in MoE — only relevant experts activate
- **Blend**: A routing network predicts which *layer group* needs velocity correction, then specialized sub-TT predicts velocities for that group
- **Emergent**: Sparse activation = lower compute, better specialization, less interference

**Blend 3: Velocity + Adapters (LoRA-style)**
- **Input 1**: TT is external predictor attached to Qwen
- **Input 2**: LoRA adapters modify Qwen's weights directly
- **Blend**: Instead of KV-cache modification, TT predicts *adapter activations* that get injected into Qwen's forward pass
- **Emergent**: Internal perturbation vs external steering — potentially more expressive

**Blend 4: Velocity + Contrastive Predictive Coding (CPC)**
- **Input 1**: MSE prediction of absolute velocity values
- **Input 2**: CPC learns representations by predicting future in latent space
- **Blend**: Train TT to maximize mutual information between hidden state sequence and velocity, not minimize MSE
- **Emergent**: TT learns velocity *directions* (which are what matter for steering) rather than velocity *values*

**Blend 5: TT + Mamba/State Space Models**
- **Input 1**: Transformer with 48M params for velocity prediction
- **Input 2**: SSMs have linear-time inference and state propagation
- **Blend**: Replace transformer with Mamba block trained on the same data; maintain or exceed R² at fraction of compute
- **Emergent**: State-tracking interpretation: velocity = derivative of SSM state, not of position

### Blind Spot Alert
- **Missing blend**: TT + Reinforcement Learning. If we can differentiate through the KV-cache steering effect, we could train TT with RL (maximize downstream accuracy) rather than supervision (minimize velocity error).

---

## Lens 4: SYSTEMS — Feedback Loops & Leverage Points

**Blind Spot from Lens 3**: RL training signal not considered.

### Variables

| Variable | Symbol | Measurable? | Current Value |
|----------|--------|-------------|---------------|
| Velocity prediction accuracy | V_acc | Yes (R², Cos) | R²=0.85 |
| Steering magnitude | S_mag | Yes (norm of applied velocity) | Unknown |
| Reasoning accuracy | R_acc | Yes (GSM8K score) | Unknown (not measured) |
| Distribution shift magnitude | D_shift | Yes (MMD, Wasserstein) | Unknown between BnB/AWQ |
| Normalization fidelity | N_fid | Yes (per-layer variance ratio) | Not measured |
| Input trajectory diversity | T_div | Yes (coverage metrics) | Unknown |
| Training stability | Stab | Yes (loss variance, gradient norms) | Known unstable (CUDA crashes) |

### Feedback Loops

| Loop ID | Type | Structure | Description |
|---------|------|-----------|-------------|
| R1 | Reinforcing | *Better velocity pred → Better steering → Better reasoning → More informative hidden states → Better velocity pred* | Virtuous cycle: if accurate steering leads to states that are easier to predict |
| B1 | Balancing | *Poor velocity pred → Noisy steering → Degraded hidden states → Different distribution → Even worse pred* | Vicious cycle: prediction errors compound through feedback |
| B2 | Balancing | *Global norm → Shrinks per-layer variance → Reduces signal for prediction → More aggressive norm needed* | Normalization destroying signal |
| R2 | Reinforcing | *More layers in TT → Better capacity → Better R² → But more overfitting to BnB distribution* | Capacity-diversity tradeoff |
| B3 | Balancing | *TF32 → Faster training → More batches → But precision noise accumulates* | Speed vs stability tradeoff |

### Leverage Points (Meadows' 12 Places)

| Rank | Point | Type (Meadows Level) | Intervention | Impact | Effort |
|------|-------|---------------------|-------------|--------|--------|
| 1 | Loss function | 6 (Structure of info flows) | Replace MSE with directional loss that ignores magnitude | HIGH | LOW |
| 2 | Normalization | 5 (Rules of system) | Per-layer normalization instead of global | HIGH | LOW |
| 3 | Training data composition | 5 (Rules of system) | Mix BnB+AWQ trajectories during training | HIGH | MED |
| 4 | Attention type | 6 (Structure of info flows) | Hybrid bidir-causal with regularization | MED | MED |
| 5 | Multi-objective training | 6 (Structure of info flows) | Add contrastive/domain-confusion loss | MED | MED |
| 6 | Expert specialization | 4 (Self-organization) | Layer-group routing + sub-experts | HIGH | HIGH |

### Blind Spot Alert
- **Missing leverage**: *Goal alignment* (Meadows Level 2). The system's goal is "minimize velocity MSE" but the actual goal should be "maximize reasoning accuracy." If minimizing MSE doesn't correlate with steering success, the entire system is optimizing the wrong objective.

---

## Lens 5: ABDUCTIVE — Best Explanations for Observed Failures

**Blind Spot from Lens 4**: Goal mismatch between training objective and actual system purpose.

### Failure 1: R²=0.85 ceiling

| Candidate Explanation | Explanatory Power | Parsimony | Combined | What Would Disprove |
|---------------------|-------------------|-----------|----------|---------------------|
| **H1**: Global normalization destroys per-layer velocity structure | 0.85 | 0.90 | 0.875 | Train with per-layer norm, show no improvement |
| **H2**: Transformer capacity insufficient (48M too small for 0.5M-dim prediction) | 0.75 | 0.80 | 0.775 | Scale to 100M+ params, show no improvement |
| **H3**: Velocity prediction has irreducible noise component | 0.80 | 0.85 | 0.825 | Compute upper bound from noise ceiling (repeated trajectories) |
| **H4**: MSE loss is wrong objective (minimizes magnitude error, not direction) | 0.90 | 0.75 | 0.825 | Train with cosine-only loss, show no improvement |
| **H5**: Bidirectional attention overfits to training-time patterns | 0.70 | 0.70 | 0.700 | Use causal attention with equal capacity, show no degradation |

**Best explanation**: **H1 + H4 combined** — Global normalization mixes layer statistics (destroying signal) AND MSE optimizes the wrong thing (rewarding magnitude accuracy over directional correctness).

### Failure 2: AWQ Transfer Collapse (R² 0.85→0.45)

| Candidate | Exp Power | Parsimony | Combined | Disproof |
|-----------|-----------|-----------|----------|----------|
| **H1**: AWQ changes hidden state distribution globally (affine shift) | 0.80 | 0.95 | 0.875 | Normalize AWQ states to BnB distribution, check if R² recovers |
| **H2**: AWQ changes hidden states in a layer-specific, non-linear way | 0.75 | 0.60 | 0.675 | Fine-tune last 2 TT layers only on AWQ, check if BnB performance is retained |
| **H3**: AWQ amplifies noise in certain features that TT relied on | 0.70 | 0.70 | 0.700 | Feature ablation: remove top-5 features TT weights most, measure R² drop |

**Best explanation**: **H1** — Most likely an affine distribution shift. Quick test: compute per-feature mean/std of AWQ and BnB hidden states, measure Wasserstein distance. If close to affine, align via simple normalization.

### Failure 3: Catastrophic Forgetting

| Candidate | Exp Power | Parsimony | Combined | Disproof |
|-----------|-----------|-----------|----------|----------|
| **H1**: TT capacity insufficient for multi-distribution representation | 0.85 | 0.90 | 0.875 | Add LoRA adapters for AWQ, keeping BnB weights frozen |
| **H2**: Gradient interference: AWQ gradient direction conflicts with BnB | 0.80 | 0.85 | 0.825 | Compute gradient cosine similarity between BnB and AWQ losses |
| **H3**: Elastic weight consolidation needed | 0.70 | 0.75 | 0.725 | Apply EWC with BnB as anchor, measure forgetting reduction |

**Best explanation**: **H1** — Most parsimonious. Capacity limitation is the classic cause of forgetting.

### Failure 4: CUDA Crashes

| Candidate | Exp Power | Parsimony | Combined | Disproof |
|-----------|-----------|-----------|----------|----------|
| **H1**: Double-buffer prefetch + TF32 creates race condition | 0.75 | 0.80 | 0.775 | Disable async prefetch, check if crashes stop |
| **H2**: float16 gradient overflow in output projection | 0.70 | 0.75 | 0.725 | Use mixed precision (bf16 for output head), check stability |
| **H3**: Memory fragmentation from 4200×28×3584 buffer allocations | 0.65 | 0.70 | 0.675 | Allocate buffers once at startup, reuse |

**Best explanation**: **H1** — Most likely a concurrency issue with async CUDA streams.

### Blind Spot Alert
- **Missing candidate**: What if R²=0.85 IS the noise ceiling — meaning the velocity *cannot* be predicted more accurately from hidden states alone, and improvement requires additional input features (e.g., attention patterns, token embeddings, layer-specific metadata)?

---

## Lens 6: TRAJECTORY — Temporal Projection

**Blind Spot from Lens 5**: Noise ceiling hypothesis not explored.

### If NO changes (current trajectory):

| Timescale | State Description | Most Likely Failure | Probability |
|-----------|------------------|-------------------|-------------|
| 1 more session | R² stuck at ~0.85, CUDA crashes frustrate scaling attempts | Hardware crash during long training run | 0.7 |
| 5 sessions | Research dead-end: incremental tuning yields <0.01 R² improvement | Abandonment of velocity prediction approach | 0.6 |
| 20 sessions | Without resolving AWQ transfer, TT is pinned to one model variant | Fork: one TT per quantization format (maintenance burden) | 0.8 |

### Next Opportunity
- **Immediate**: Change normalization → per-layer. Expected R² gain: +0.02-0.05. Cost: 1 hour of coding. Highest ROI by far.
- **Short-term**: Multi-objective loss (direction + magnitude + domain confusion). Expected gain: +0.03-0.08 R² + AWQ transfer.
- **Medium-term**: Online trajectory generation from Qwen during training. Eliminates I/O bottleneck, increases data diversity naturally.

### Inflection Points

| Point | Trigger | Consequence |
|-------|---------|-------------|
| Normalization change shows 0% improvement | Assumption: normalization is not the bottleneck → shift focus to architecture | Save weeks of tuning |
| R² exceeds 0.90 | Validation that velocity prediction has headroom | Opens deployment path |
| AWQ R² after normalization > 0.70 | AWQ transfer is primarily distribution shift | Domain adaptation approach validated |
| Cosine loss alone beats MSE | Entire loss paradigm shifted | Abandon MSE |

### Key Assumptions
- GSM8K trajectories generalize to other reasoning tasks
- Velocity prediction accuracy correlates with KV-cache steering quality
- R²=0.85 is not a noise ceiling

### Early Warning Signs
- Per-layer normalization doesn't improve R² → noise ceiling
- AWQ/BnB distribution difference is huge (Wasserstein > 1.0) → need domain adaptation
- Gradient conflict score close to -1.0 → fine-tuning impossible without EWC

### Blind Spot Alert
- **Missing trajectory**: What if the field moves to different architectures (Mamba, GatedDeltaNet, etc.) and Qwen becomes obsolete before TT matures? The TT is tightly coupled to Qwen's specific hidden state structure.

---

## Lens 7: METACOGNITIVE — What Are We Missing?

**Blind Spot from Lens 6**: Technology risk (TT tightly coupled to Qwen architecture).

### Embedded Assumptions

| Assumption | How It Shapes Findings | Alternative |
|------------|----------------------|-------------|
| Velocity is the right target | All analysis focused on predicting velocity better; never questioned | Attention pattern deltas, activation magnitude changes, subspace rotations |
| 28 layers are equally important | All analysis treats layers uniformly | 80% of steering value may come from 3-5 critical layers |
| GSM8K is representative | Findings implicitly generalize beyond GSM8K | Velocity patterns may be task-specific |
| R² is the right metric | "Improving R²" drives all recommendations | R² improvement may not translate to better reasoning |

### Systematic Gaps

| Gap | Why Missed | How to Fill |
|-----|-----------|-------------|
| **Ground truth**: What is the "correct" velocity? | Velocity is defined as layer-to-layer delta of the frozen model; this is just one path | **Counterfactual experiment**: For input pairs (correct answer, wrong answer), compute velocity difference. The difference IS the ground truth steering direction. |
| **Downstream validation**: Does velocity prediction improve reasoning? | Pipeline measured only R², not end-to-end accuracy | Run end-to-end: TT → Steering → GSM8K accuracy. Measure correlation. |
| **Layer importance distribution**: Which layers matter for steering? | All layers treated equally | Ablation experiment: mask velocity for each layer, measure R² drop per layer |
| **Velocity manifold structure**: Do velocities lie on a low-dimensional manifold? | Assumed full-rank 3584-dim space | PCA on velocity targets → measure explained variance ratio |
| **Adversarial/stress testing**: When does TT fail worst? | Only average metrics reported | Identify trajectories where R² < 0.5; characterize them |

### Confidence Calibration

| Area | Status | Bias Source |
|------|--------|-------------|
| Overconfident: MSE is "good enough" | High R² overrides concerns | Outcome bias (R²=0.85 looks good) |
| Underconfident: Architecture capacity | "Maybe 48M is enough" without evidence | Anchoring on current architecture |
| Blind: Downstream validation | Not measured at all | Assumption that better R² → better steering |

### Unasked Questions

1. **Why 28 layers?** Qwen has 28 layers. What if velocity prediction only needs the last 10?
2. **What does the TT latent space look like?** d_model=768 space — is it structured?
3. **Can we train WITHOUT frozen Qwen?** Joint training: TT + LoRA on Qwen for steering?
4. **What is the velocity distribution?** Heavy-tailed? Gaussian? Bimodal?
5. **Are velocities sparse?** Could we predict only top-k velocity components by magnitude?

### Blind Spot Alert
- **Meta-blind spot**: The entire analysis framework assumes velocity prediction is the correct path. What if velocity is NOT the right steering signal and something else (attention entropy, hidden state curvature, subspace angles) works better?

---

## Lens 8: INSPIRATION — Cross-Domain Adaptations

### Source: Computational Fluid Dynamics (CFD)
- **Mechanism**: In CFD, velocity fields are predicted on coarse grids and refined via learned super-resolution
- **Adaptation**: Predict velocities on a *subsampled* layer grid (e.g., every 3rd layer → 10 layers) then upsample to 28 layers via learned interpolation
- **Constraints**: 7B model has 28 layers, not 10; but layers are sequential → interpolation is plausible
- **Novel adaptation**: 4× compute reduction, potential regularization benefit

### Source: Neural Radiance Fields (NeRF)
- **Mechanism**: NeRF predicts color+density at continuous 3D positions using positional encoding
- **Adaptation**: Replace learned position embeddings with sinusoidal positional encoding (NeRF-style) for layer index
- **Constraints**: Layers are discrete (1-28) but velocity as a function of layer index may be smooth
- **Novel adaptation**: Better generalization because position encoding captures layer-to-layer continuity

### Source: Momentum Contrast (MoCo) / BYOL
- **Mechanism**: Self-supervised learning uses a momentum encoder to generate stable training targets
- **Adaptation**: Create a momentum-Qwen that maintains a moving average of hidden states; TT predicts velocity toward momentum states rather than raw deltas
- **Constraints**: Requires two forward passes through 7B; compute doubles
- **Novel adaptation**: Momentum targets are smoother, less noisy → higher R² achievable

### Source: Meta-Learning (MAML)
- **Mechanism**: Train a model that can quickly adapt to new tasks with few gradient steps
- **Adaptation**: Meta-train TT across multiple quantization formats (BnB, AWQ, GPTQ) so it learns a shared initialization that adapts rapidly
- **Constraints**: Requires trajectories from multiple quantized variants; data generation cost
- **Novel adaptation**: "One TT to rule them all" — load and adapt in one gradient step

### Source: Neural ODEs / Continuous Normalizing Flows
- **Mechanism**: Model hidden state evolution as an ODE: dh/dt = f(h, t)
- **Adaptation**: Velocity prediction IS dh/dt. Model as continuous-time ODE where the TT learns the vector field across the layer dimension
- **Constraints**: ODE solvers require multiple function evaluations; higher inference cost
- **Novel adaptation**: Smooth interpolation: TT can predict velocity at *any* layer position, not just integer indices

### Blind Spot Alert
- **Missing inspiration**: Model-based RL / World Models. The TT is a world model of Qwen's internal dynamics. Dreamer-style training (imagine trajectories, train on imagined data) could apply.

---

## Lens 9: ADVERSARIAL — Attacks & Defenses

### Finding 1: R² Ceiling Attack
- **Target**: MSE loss minimization (P1: R²=0.85)
- **Vector**: (c) No-free-lunch: optimizing MSE does not guarantee optimal steering
- **Severity**: 0.85 — if R² doesn't correlate with reasoning, the entire enterprise is optimizing a proxy
- **Defense**: End-to-end validation experiment linking R² to reasoning accuracy

### Finding 2: AWQ Transfer Attack
- **Target**: P4 (R² drop from 0.85→0.45)
- **Vector**: (d) Empirical counter-evidence: the distribution shift is so large that velocity is not a transferable quantity
- **Argument**: If AWQ changes hidden states such that velocity vectors undergo non-linear transformation, predicting velocity on BnB and applying to AWQ is fundamentally unsound
- **Severity**: 0.90 — questions core viability
- **Defense**: Learn a *correction function*: small MLP that maps AWQ hidden states to BnB-equivalent states before velocity prediction

### Finding 3: Architecture Attack
- **Target**: A1 (6-layer Transformer)
- **Vector**: (b) Capacity mismatch: 48M params / 3584 output = 13,393 parameters per output dimension. This is very low for 0.5M-dimensional prediction
- **Argument**: Predicted output has 0.5M dimensions (28×3584×2 = velocity magnitude+direction). With 48M params, each output dim gets ~100 params. This is severely bottlenecked
- **Severity**: 0.80 — architecture is under-parameterized for the task
- **Defense**: Increase capacity OR reduce output dimensionality (PCA, group prediction)

### Finding 4: Normalization Attack
- **Target**: D6 (Global normalization)
- **Vector**: (a) Info-theoretic bounds: compressing 28×3584 distribution into 3584 mean/std discards per-layer information
- **Argument**: Mutual information I(hidden_layer, normalization_param) is reduced by 28×. Per-layer variance information is destroyed
- **Severity**: 0.85 — fundamental information loss
- **Defense**: Per-layer normalization (trivial change, high impact)

### Finding 5: Gradient Interference Attack
- **Target**: T8 (MSE loss) applied to all 28 layers uniformly
- **Vector**: (a) Info-theoretic bounds: gradients from early and late layers can conflict
- **Argument**: If early-layer velocities are 10× larger in magnitude than late-layer, MSE gradients are dominated by early layers. Weight updates ignore late-layer signal
- **Severity**: 0.75 — gradient imbalance
- **Defense**: Per-layer loss normalization (scale MSE by inverse variance per layer) or per-layer prediction heads

### Finding 6: CUDA Crash Attack
- **Target**: P6 (CUDA crashes)
- **Vector**: (e) Overfitting trap: the double-buffer prefetch creates a hidden dependency on async stream ordering
- **Argument**: The system appears to work but fails under sustained load because async prefetch + TF32 creates implicit ordering assumptions that break under heavy GPU utilization
- **Severity**: 0.70 — reliability issue
- **Defense**: Synchronous prefetch (2× slower but stable) or CUDA graph recording

### Findings Ranked by Severity

1. **AWQ transfer (0.90)** — Core viability question
2. **Global normalization info loss (0.85)** — Fundamental
3. **MSE≠steering-correlation (0.85)** — Proxy optimization
4. **Under-parameterization (0.80)** — Capacity bottleneck
5. **Gradient imbalance (0.75)** — Training inefficiency
6. **CUDA instability (0.70)** — Reliability

### Blind Spot Alert
- **Critical missed attack**: What if a small perturbation to input hidden states (adversarial noise) causes TT to make catastrophically wrong velocity predictions? Adversarial robustness of velocity prediction is untested.

---

## Lens 10: PARADOXICAL — Self-Reference & Inversion

### Paradox 1: The Steering Contradiction
- **Statement**: "TT predicts velocities to steer Qwen's KV-cache toward better reasoning"
- **Self-reference**: If TT successfully steers Qwen toward correct reasoning, then the steered hidden states are different from the training distribution (which was from unsteered Qwen). TT must predict velocities for states it has never seen.
- **Gödel sentence**: "This velocity prediction is only valid for models that have not been steered by velocity predictions."
- **Resolution**: Online adaptation loop — TT must be fine-tuned on steered trajectories, creating a moving target

### Paradox 2: The Normalization Paradox
- **Statement**: "Normalization improves training by centering and scaling hidden states"
- **Inversion**: What if normalization removes the exact signal TT needs to predict velocity? Velocity IS the difference between consecutive layers. Global normalization can amplify or suppress these differences non-uniformly.
- **Resolution**: Normalize after computing velocity targets, not before. Or: normalize input hidden states but compute velocity targets BEFORE normalization.

### Paradox 3: The Capacity Paradox
- **Statement**: "48M parameters should be enough for velocity prediction"
- **Inversion**: The 7B model has 28×3584 = 100,352 hidden state dimensions. Velocity has the same number. A 48M predictor for a 100K-dim target ratio = 480×. But the Qwen itself is 7B and can't predict its own velocities. How can a 0.7%-scale model predict the full-dynamics of its parent?
- **Gödel sentence**: "This 48M model cannot completely model the 7B model's internal dynamics because if it could, it could be used to compress the 7B model."
- **Resolution**: TT doesn't need to model all 100K dims perfectly — it only needs to model the *steerable subspace*.

### Paradox 4: The Training-Inference Paradox
- **Statement**: "Train with bidirectional attention for best prediction accuracy"
- **Inversion**: At inference time, future layers don't exist yet (autoregressive). The TT sees hidden states up to the current layer only. Bidirectional training creates dependency on future layers that doesn't exist at inference.
- **Resolution**: Causal attention during training, or masking: train bidirectional but mask the loss so each layer's velocity only uses information from previous layers.

### Paradox 5: The Transfer Paradox
- **Statement**: "Fine-tune on AWQ to adapt the TT"
- **Inversion**: Fine-tuning on AWQ changes weights to fit AWQ patterns. But AWQ patterns are close to BnB patterns (same model, different precision). If they are close, why does R² drop from 0.85→0.45? If they are far, fine-tuning on AWQ will destroy BnB performance.
- **Resolution**: Neither fine-tuning nor static training works. Need *domain-invariant representations* from the start, or *hypernetwork* that takes quantization info as input.

### Persistent Blind Spots (After All 10 Lenses)

| # | Blind Spot | Which Lenses Missed It | Why Persistent |
|---|-----------|----------------------|----------------|
| 1 | **End-to-end validation missing** | 1-6, 8-10 | Requires expensive downstream eval; easier to measure R² |
| 2 | **Velocity manifold structure** | 1-10 | Assumes full-rank; PCA never run on targets |
| 3 | **Layer importance heterogeneity** | 1-3, 5-6, 8-9 | Convenient to treat uniformly |
| 4 | **Adversarial robustness** | 1-8, 10 | Not standard practice in velocity prediction |
| 5 | **Noise ceiling** | 1-4, 6, 8-9 | Would be demoralizing to confirm; psychology of research |
| 6 | **Goal alignment: velocity R² vs reasoning quality** | 1-10 | Entire field assumes proxy metrics suffice |
