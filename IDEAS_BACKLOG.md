# Undocumented Ideas, Proposals & Experiments

This document catalogs every concept, proposal, and experiment discussed during development that was **not implemented or fully tested**. Each entry includes the idea, rationale, status, and a concrete next step for implementation.

---

## 1. SelectiveLoRA — Importance-Driven Element Freezing

**Status**: Not implemented
**Priority**: ★★★ (next in queue)

**Concept**: Train all adapter parameters for K warmup steps, compute per-element importance scores, freeze the bottom X% (set gradients to zero), continue training only the top (100-X)%. Periodically re-score and update the mask every N steps.

**Importance metrics to compare** (not implemented):
- Gradient magnitude: `|∂L/∂θ|` averaged over recent steps
- Parameter magnitude × gradient: `|θ · ∂L/∂θ|` (sensitivity-weighted)
- Fisher information: `(∂L/∂θ)²` (empirical Fisher)
- Integrated gradients: path-integrated sensitivity (IGU-LoRA style)
- Movement: `|θₜ − θ₀|` over training (how far from initialization)

**Implementation sketch**:
```python
class SelectiveLoRA(LowRankAdapter):
    def __init__(self, ...):
        self.lora_A, self.lora_B = nn.Parameter(...), nn.Parameter(...)
        self.warmup_steps = 50
        self.sparsity = 0.5  # freeze bottom 50%
        self.rescore_interval = 100
        self.register_buffer("mask_A", torch.ones(r, in_features))
        self.register_buffer("mask_B", torch.ones(out_features, r))
    
    def forward(self, x, base_weight):
        # Apply masks
        eff_A = self.lora_A * self.mask_A
        eff_B = self.lora_B * self.mask_B
        BA = (eff_B @ eff_A) * self.scaling
        ...
```

**Needs investigation**:
- What sparsity ratio works best? (10%, 50%, 90%?)
- Does periodic rescoring help or hurt?
- Should scoring be per-element or per-rank? (rank-level = SMoRA-style)
- How does this interact with cycling variants?

---

## 2. Direction-Aware Routing — Separate Experts Per Fwd/Bwd Direction

**Status**: Not implemented
**Priority**: ★★★ (unique research gap)

**Concept**: In a bidirectional architecture, use completely separate adapter parameters for the forward direction vs backward direction, each with their own A, B, magnitude. A learned router decides per-token whether to use the fwd expert, bwd expert, or both.

**Why it's novel**: No paper explores direction-aware routing. The closest work (SMoRA, rank-as-expert) and MoLoRA (per-token routing) both route between different adapters, but none routes between information-flow directions within a single adapter.

**Architecture**:
```python
class DirectionalMoE(LowRankAdapter):
    # Expert 1: forward-path adaptation
    self.fwd_A, self.fwd_B, self.fwd_mag
    # Expert 2: backward-path adaptation  
    self.bwd_A, self.bwd_B, self.bwd_mag
    # Learned router: per-token gating
    self.router = nn.Linear(in_features, 2)  # 2 experts
```

**Questions**:
- Should routing be per-token, per-sequence, or per-layer?
- Soft routing (weighted combination) vs hard routing (top-1)?
- Fwd-expert only, bwd-expert only, or both?

---

## 3. Mixture-of-Direction-Experts (Full MoE)

**Status**: Not implemented
**Priority**: ★★

**Extension of #2**: Instead of just 2 experts (fwd/bwd), train K experts per transformer layer, each with its own A, B, magnitude. A router selects which expert(s) to use per token. Experts can specialize in different directions, frequency bands, or task types.

**Related**: SMoRA (rank-as-expert), MoLoRA (per-token routing), MoLA (layer-wise expert allocation)

**Key design choices**:
- Number of experts per layer (fixed or MoLA-style varying)
- Whether experts share the base frozen_weight
- Router architecture (linear projection, small MLP)
- Load balancing loss to prevent expert collapse

---

## 4. Cycled Variants × All Technique Flags

**Status**: Not implemented
**Priority**: ★★

**Concept**: The `gen_dora.py` generator currently supports flags (AFA, GA, SR, EVA, Knit, norm). Add a `has_cycle` flag to generate cycled variants of every flag combination. The cycle alternates between the two branches/axes of the chosen base architecture.

**Variants to generate** (example):
- `gend_cycled_afa_doran` — AFA + cycling + norm
- `gend_cycled_sr_ga_doran` — SR + GA + cycling + norm
- `gend_cycled_afa_eva_doran` — AFA + EVA + cycling + norm

**Rationale**: Cycling might interact differently with technique flags. E.g., GA (gradient alignment) might make cycling unnecessary, or AFA's annealed activation might complement cycling's alternating structure.

---

## 5. Multi-Angle DiagLoRA — Angle Set Sweep

**Status**: Not implemented (MultiAngleLoRAN exists but untested)
**Priority**: ★★

**Concept**: The MultiAngleLoRA class accepts configurable angles. Test different angle sets to understand which diagonal directions matter most.

**Angle sets to test**:
- `[0, 45, 90, 135]` — cardinal + main diag + anti diag (current default)
- `[0, 90]` — row + col only (axis-aligned, ≈ DoRA+BoRA)
- `[45, 135]` — diag + anti only
- `[0, 30, 60, 90, 120, 150]` — 6 evenly spaced angles
- `[22.5, 67.5, 112.5, 157.5]` — 45° rotated from cardinal
- `[0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]` — all 8

**Per-parameter cost**: N_angles × (out+in-1) mag params per angle

---

## 6. Rank Sweep — Does Ranking Change With r?

**Status**: Not implemented
**Priority**: ★★

**Test**: Run top-10 variants at r=2, 4, 8, 16, 32 to see if relative rankings shift.

**Hypothesis**: Magnitude decomposition might matter more at low ranks (where the BA update is more constrained) than high ranks.

**Variants to test**: plain_lora, edoran (gs=128), doran, cycled_axial_boran, cycled_bvoran, cycled_diag_lora, diag_loran

---

## 7. Training Depth — Does Ranking Converge at 500+ Steps?

**Status**: Not implemented
**Priority**: ★★

**Test**: Run top-10 variants for 500+ steps. Check if rankings at step 100 correlate with rankings at step 500.

**Motivation**: Cycling variants might need more steps to converge (since each branch only trains every other step). Static variants might plateau earlier.

---

## 8. Cross-Dataset Validation

**Status**: Not implemented
**Priority**: ★★

**Datasets available on HDD**:
- arxiv-summarization (3.5GB, currently used)
- OpenCodeInstruct (6.4GB, code instructions)
- OpenMathInstruct-2 (12GB, math reasoning)
- OpenThoughts3-1.2M (27GB, reasoning traces)

**Test**: Run top-10 variants on 2-3 different datasets. Magnitude decomposition might favor structured text (code, math) more than free-form scientific text.

---

## 9. Gradient Conflict Measurement

**Status**: Not implemented
**Priority**: ★★

**Concept**: Quantify the gradient conflict in bidirectional adapters by computing cosine similarity between fwd and bwd branch gradients:

```python
grad_fwd = [p.grad for p in [lora_A_fwd, lora_B_fwd, mag_fwd]]
grad_bwd = [p.grad for p in [lora_A_bwd, lora_B_bwd, mag_bwd]]
cos_sim = F.cosine_similarity(
    torch.cat([g.flatten() for g in grad_fwd]),
    torch.cat([g.flatten() for g in grad_bwd]),
    dim=0
)
```

**Expected**: BoRA/BVoRAN show negative or near-zero cosine similarity (= conflicting signals). CycledBoRAN shows alternating positive/negative as it cycles. DoRA (single direction) has no conflict by definition.

---

## 10. Adapter Merging Investigation

**Status**: Not implemented
**Priority**: ★★

**Question**: Can cycling variants be merged into W₀ for zero-overhead inference? (Linear operations like magnitude scaling, column normalization, diagonal scaling can theoretically be merged.)

**For each variant**:
- DoRA/EDoRA: merge `magnitude · column_norm(W₀ + BA)` into `W̃₀`
- CycledBVoRAN: merge depends on which branch was active — need to track or average
- DiagLoRA: diagonal scaling is element-wise (less merge-friendly)

**Relevance**: Variants that can't merge incur inference overhead, which may be unacceptable for deployment.

---

## 11. KnitLoRA Memory Optimization

**Status**: OOM on all model sizes
**Priority**: ★

**Problem**: VE parameters create ve_B @ (ve_lambda * ve_A) in the forward pass — 4 additional matmuls per layer (2 directions × fwd/bwd). On a 30-layer SmolLM with 4 target modules per layer = 120 × 4 = 480 extra matmuls per forward pass. With 100 training steps, this is 48,000 extra matmuls. The intermediate VE tensors consume ~0.5GB extra peak memory.

**Potential fixes**:
1. Use CPU offloading for VE buffers (move ve_A, ve_B to CPU, compute on CPU, copy result to GPU)
2. Reduce ve_rank (currently r//2 = 4; try 1 or 2)
3. Share VE across layers (one global VE per model instead of per-layer)
4. Fuse VE into the magnitude computation (eliminate separate VE matmul)

**Priority low** because VE's benefit is unclear compared to the memory cost.

---

## 12. BoRA Family With LayerNorm

**Status**: Not implemented
**Priority**: ★

**Currently**: Only DoRA variants have norm versions (`doran`, `edoran`, `gend_*doran*`). The bidirectional BoRA variants (`bora`, `bvoran`, etc.) lack norm equivalents.

**Add**: `boran`, `eboran` (with LayerNorm), and norm versions of gen bidirectional variants.

**Note**: The standalone `bvoran.py` already has LayerNorm but also has VE parameters, making it ≈ KnitBVoRAN. Need a clean BoRA+norm without VE.

---

## 13. Toy Model With Structured Data

**Status**: Not implemented
**Priority**: ★

**Current**: Toy transformer trained on random token sequences. Random data means all variants converge to the same entropy — no signal for relative rankings.

**Fix**: Generate structured synthetic data:
- Simple grammar: `S → NP VP`, `NP → Det N`, etc.
- Repeating n-gram patterns
- Synthetic arithmetic (2+3=5 style)
- Copy task (input repeated at output)

**Benefit**: Would give meaningful relative rankings in seconds per variant, allowing 50+ variant sweeps in ~5 minutes.

---

## 14. Switching Interval Sweep

**Status**: Not implemented
**Priority**: ★

**Cycling hyperparameter**: The `cycle_interval` (currently hardcoded default=10) determines how many steps before switching branches/axes. Test values: 1, 5, 10, 20, 50.

**Hypothesis**: Interval 1 (switch every step) is effectively random. Interval 50 (switch once in 100 steps) is almost like training only one direction. The optimal is somewhere in between where each branch gets enough contiguous gradient steps to converge, but not so many that the other branch atrophies.

---

## 15. DoRAN vs EDoRA Full Factorial

**Status**: Partially done (EDoRA group sweep complete)
**Priority**: ★

**Complete the grid**: The EDoRA group size sweep tested `edoran` (EDoRA + norm) with gs=64-1024. Missing:
- `edora` (no norm) × same group sizes
- `doran` (DoRA + norm) — already at gs=1024 by definition (per-output)
- `dora` (DoRA, no norm)
- Compare whether the optimal group size differs with vs without norm

---

## 16. Quantization Behavior of Cycling Variants

**Status**: Not implemented
**Priority**: ★

**Question**: Do cycling variants train differently under quantization (NF4, int8)? Cycling's alternating structure might be more or less compatible with quantized base weights.

**Related**: QA-LoRA (QADoRA) applies fake quantization to merged weights. Cycling variants' merged weight changes every cycle_interval steps — does this make quantization harder?

---

## 17. Diagonal Dropout for DiagLoRA

**Status**: Not implemented
**Priority**: ★

**Concept**: In DiagLoRA, some diagonal bands may be naturally more important than others (near-diagonals in attention, for example). Apply structured dropout to entire diagonal groups, forcing the model to distribute information across bands.

**Implementation**: During each forward pass, randomly zero out P% of the diagonal magnitudes. This is analogous to Dropout but applied at the diagonal-group level rather than the neuron level.

---

## 18. Frequency-Domain Analysis

**Status**: Not implemented
**Priority**: ★

**Concept**: Compute 2D DFT of weight updates (ΔW) for each variant and analyze the frequency structure. DiagLoRA should concentrate energy along diagonal frequency lines. DoRA/EDoRA should concentrate along axis-aligned lines.

**Method**:
```python
dft = torch.fft.fft2(delta_W)
magnitude_spectrum = dft.abs().log()
```

**Hypothesis**: Cycling variants should have more uniform frequency distribution (less concentrated in specific directions) because they alternate between different axes.

---

## 19. SelectiveLoRA × Cycling Combo

**Status**: Not implemented
**Priority**: ★

**Concept**: Apply selective importance-based freezing to cycling variants. The active branch's parameters get scored and frozen independently from the inactive branch.

**Motivation**: Cycling already reduces gradient conflict by separating fwd/bwd updates. Adding selective freezing within each active branch might further improve efficiency by focusing the update on the most important parameters of the currently active branch.

---

## 20. Sparsity Ratio and Rescoring Schedule for SelectiveLoRA

**Status**: Not implemented
**Priority**: ★

**Independent sweep**: For SelectiveLoRA, two key hyperparameters:
- Sparsity ratio: 0.1, 0.25, 0.5, 0.75, 0.9 (fraction of params frozen)
- Rescoring interval: never, 25, 50, 100, 200 steps

**Hypothesis**: Higher sparsity (90% frozen) works with frequent rescoring (every 25 steps). Lower sparsity (25% frozen) works with infrequent rescoring.

---

## 21. Importance Metric Comparison

**Status**: Not implemented
**Priority**: ★

**Compare scoring methods** for SelectiveLoRA:
1. Gradient magnitude: `|g|`
2. Sensitivity: `|w · g|`
3. Empirical Fisher: `g²`
4. Movement: `|wₜ − w₀|`
5. Random baseline
6. Uniform (no selection, full training)

**Metric**: Does the scoring method improve final loss over random selection? If random selection works as well as importance-based, then the structure of the adapter doesn't matter — only the reduced parameter count.

---

## 22. Per-Token vs Per-Sequence Routing

**Status**: Not implemented
**Priority**: ★

**For DirectionalMoE (#2)**: Should the router decide per-token (different tokens in the same sequence can use different directions) or per-sequence (all tokens use the same direction)?

**Expected**: Per-token routing is more expressive (MoLoRA-style) but requires more compute. Per-sequence is simpler and may work better for homogeneous sequences (scientific text).

---

## 23. Fwd/Bwd Parameter Count Asymmetry

**Status**: Not implemented
**Priority**: ★

**Question**: In bidirectional variants, should the fwd and bwd paths have the same rank/parameters? Or can they be asymmetric?

**Rationale**: The forward path processes the main signal, while the backward path processes the transposed signal. These may have different information capacity requirements. Trial r_fwd = 8, r_bwd = 4 (or vice versa).

---

## 24. Priority Summary Table

| # | Idea | Effort | Expected Impact | Novelty |
|---|------|--------|----------------|---------|
| 1 | SelectiveLoRA | 1 day | High | High (gap) |
| 2 | Direction-Aware Routing | 2 days | High | Very High (unique) |
| 4 | Cycled × Flag Combos | 2 hr | Medium | Low (extension) |
| 6 | Rank Sweep | 3 hr | Medium | Low (baseline) |
| 7 | Training Depth | 3 hr | High | Low (validation) |
| 9 | Gradient Conflict Measurement | 4 hr | High (insight) | Medium |
| 5 | Multi-Angle Sweep | 2 hr | Low | Medium |
| 14 | Switching Interval Sweep | 1 hr | Medium | Low |
| 20 | Sparsity Ratio Sweep | 2 hr | Medium | Low |
| 21 | Importance Metric Comparison | 3 hr | Medium | Medium |

**Recommendation**: Pursue #1 (SelectiveLoRA) first, then #9 (gradient conflict measurement) to generate publishable insights, then #2 (direction-aware routing) as the novel contribution.

---

## Phase 3 Additions: StreamFusion / Flow Matching / Diffusion

These ideas emerged from the StreamFusion phase (v0.1–v0.18) and were not in the original backlog.

### 25. Flow Matching Over Adapter Weights

**Status**: Implemented, evaluated, hits generalization wall
**Priority**: ★★★

**Concept**: Train a velocity field v_θ(W_t, t, data_ctx) → δW that predicts the next weight from the current weight + conditioning. Train via MSE(v_pred, W_{t+1} - W_t) on observed SGD trajectories. At inference, integrate from W_0=0 to generate trained weights without SGD.

**Key finding**: Perfect training fit (MSE ≈ 0) but predicts zero on test data. Conditioning (mean hidden state + gradient) is information-theoretically insufficient.

**Potential fixes**:
- Richer conditioning (full hidden state sequence, not mean)
- Closed-form SVD as primary target (well-defined for any input)
- Higher-rank model (2B instead of 0.8B — richer hidden states)

### 26. Closed-Form SVD Optimal Target

**Status**: Implemented, not yet primary training objective
**Priority**: ★★★

**Concept**: The optimal rank-r LoRA update has closed form: SVD of R·X⁺ where R = Y - WX is the residual. Achieves 29.8% loss reduction in ONE step (vs SGD's ~37% over 20 steps).

**Key insight**: The velocity field should be trained to predict the SVD-optimal update, not noisy SGD trajectories. The SVD target is meaningful for ANY input (unlike SGD trajectories which are zero for unseen data).

**Needs investigation**:
- Does training solely on the SVD target (no flow matching) produce better generalization?
- How does the SVD target interact with the stagnation penalty?
- Can the SVD be computed efficiently enough for online use?

### 27. Stagnation Penalty for Velocity Field

**Status**: Implemented (in training, not yet evaluated)
**Priority**: ★★★

**Concept**: Add loss term L_stag = -λ_stag · ||v||² to penalize zero velocity predictions. Forces the model to output non-zero directions even when uncertain about the correct update.

**Result**: Increases velocity norm 1615× compared to baseline. Combined with flow + optimal targets, should produce meaningful predictions on test data.

**Needs investigation**:
- What λ_stag value balances exploration vs. correctness?
- Does stagnation + SVD target generalize to test data?

### 28. Dynamic K Regulation for Perceiver Latents

**Status**: Implemented, not yet evaluated end-to-end
**Priority**: ★★

**Two approaches**:
1. **Entropy-thresholded** (Approach A): Drop latents with attention entropy > τ·H_max
2. **Adaptive growth** (Approach F): Start with K_min, expand when thought change stalls

**Needs investigation**:
- Does dynamic K improve reasoning quality over fixed K?
- Which approach works better for which type of input?
- Can both be combined? (entropy-based pruning + growth-based expansion)

### 29. Soft Flag Morphing for Architecture Adaptation

**Status**: Implemented
**Priority**: ★★

**Concept**: Replace discrete boolean adapter flags with continuous [0,1] values. Flags move toward targets via exponential moving average each training step. Enables smooth architectural transitions within a segment.

**Implementation**: HybridStreamExpert with soft_flags dict, set_targets(), morph_step()

**Needs investigation**:
- Optimal morph_rate schedule (fast for exploration, slow for convergence?)
- Can the MetaController learn to set optimal flag trajectories?

### 30. Latent Reasoning Engine

**Status**: Designed (thoughts/latent_reasoning_analysis.md), not implemented
**Priority**: ★★★

**Concept**: Replace "weights" as the flow matching object with "thoughts" (hidden states of a language model). Thoughts HAVE structure → denoising should work. The same PerceiverFusion + WeightDiffusion architecture applies, just over thought trajectories instead of weight trajectories.

**Key advantage**: Thoughts are structured (unlike random weights). Attention patterns, token representations, and layer activations live on a low-dimensional manifold. This is where diffusion actually works.

**Design**: See `thoughts/latent_reasoning_analysis.md` for full architecture. Key components:
- PerceiverFusion cross-attends thoughts to input
- WeightDiffusion denoises thought trajectories
- StreamFusion trains experts on reasoning step gradients
- MetaController decides reasoning mode per step
- Dynamic K regulates number of thought latents

### 31. MetaController End-to-End Learning

**Status**: Implemented (ES optimization) but not evaluated
**Priority**: ★★

**Concept**: A transformer that reads adapter history → suggests next configuration (flags, poly_order, morph_rate). Can be trained end-to-end via the unified differentiable lifecycle optimizer (v0.9).

**Needs investigation**:
- Does ES produce better lifecycles than random search?
- Can the differentiable optimizer (v0.9) train the controller directly?
- How many lifecycles are needed for the controller to learn meaningful schedules?

### 32. Cross-Domain Lifecycle Transfer

**Status**: Not implemented
**Priority**: ★★

**Question**: Does a MetaController trained on arxiv text learn lifecycles that transfer to code (HumanEval) or math (GSM8K)?

**Test**: Train controller on 3 domains, test on 2 held-out domains. If lifecycles transfer, the controller has learned domain-agnostic reasoning patterns.

### Priority Summary (Phase 3)

| # | Idea | Effort | Expected Impact | Novelty |
|---|------|--------|----------------|---------|
| 25 | Flow Matching Over Weights | Complete (evaluate) | High | High |
| 26 | Closed-Form SVD Target | 1 day | Very High | High |
| 27 | Stagnation Penalty | 1 day (training running) | Medium | Medium |
| 30 | Latent Reasoning Engine | 1 week | Very High | Very High |
| 28 | Dynamic K | 2 days | Medium | High |
| 29 | Soft Flag Morphing | Complete | Medium | High |
| 31 | MetaController E2E | 3 days | High | High |
| 32 | Cross-Domain Transfer | 4 days | Medium | Medium
