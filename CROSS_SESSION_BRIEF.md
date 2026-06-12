# Cross-Session Brief — RankAdaptation / Latent Reasoning
## Sessions: 2026-06-10 → 2026-06-12

## Narrative Summary

This project evolved from LoRA adapter comparison (Phase 1-2) through online
continual learning (Phase 3) into **latent reasoning via flow matching over
thought trajectories** (Phase 4-5). The current session (June 12) produced
8 new tags and definitive proof for three steering mechanisms.

### Phase 4: Flow Matching Over Thoughts (Tags: v0.21-26)

**Finding**: Thoughts HAVE structure (R²=0.62 TrajectoryTransformer, R²=0.75
reasoning-step trajectories), weights DON'T (R²≈0).

Key results:
- **TrajectoryTransformer** (direct self-attention over 23 hidden states) beats
  the Perceiver bottleneck (R²=0.62 vs 0.42) — removing the compression
  bottleneck recovers lost signal
- **Reasoning-step trajectories** (token-to-token) achieve R²=0.75 — higher than
  layer-to-layer (0.62) because they capture reasoning evolution, not just
  layer computation
- **Reading heads**: r=0.86 correlation with token perplexity from frozen
  Perceiver latents — latents strongly encode uncertainty

### Phase 5: Steering Mechanisms (Tags: v0.27-33)

Three steering mechanisms validated:

| Mechanism | Metric | When It Works | When It Fails |
|-----------|--------|--------------|---------------|
| **Logit correction** | +13.8pp per-token accuracy | Prompt data | Generation (distribution shift → 0%) |
| **Layer-0 injection** | 100% divergence at α=2.0 | Always (24× amplification) | Never tested on accuracy |
| **KV-cache steering** | 85-90% divergence at α=0.1 | Geometrically viable (cos_v=0.94) | Signal = noise at binary metric |

**Critical discovery**: Last-layer steering is geometrically impossible
(<1% divergence at α=100). Adding velocity to the last hidden state and
projecting through the LM head doesn't change token selection because the
velocity lives in a subspace orthogonal to the LM head's projection directions.
KV-cache steering avoids this by modifying what FUTURE tokens attend to.

## Key Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-10 | Use 0.8B model for weight flow experiments | Faster iteration |
| 2026-06-11 | Add gradient conditioning to weight flow | Richer context signal |
| 2026-06-11 | Switch to 2B model trajectories | 2× larger hidden state for better conditioning |
| 2026-06-11 | Drop denoising for weight flow | MSE stuck at 1.0 — weights are unstructured |
| 2026-06-11 | Add stagnation penalty | Force non-zero predictions |
| 2026-06-11 | Add alignment-based data filtering | Remove noisy SGD steps |
| 2026-06-11 | Pre-compute trajectories to disk | Save ~60min per training run |
| 2026-06-11 | Shift to latent reasoning | Weight flow hits information-theoretic wall |

## Survivors (What to Keep)

- **StreamFusionLoRA** (`src/adapters/stream_fusion.py`): PerceiverFusion,
  HybridStreamExpert, soft flags, expert pool — core infrastructure for
  online adaptation.

- **WeightDiffusion** (`src/adapters/diffusion_weight_flow.py`): Perceiver
  bottleneck, separate noise/flow heads, weight normalizer — reusable
  for thought-space flow matching.

- **DynamicPerceiverWrapper** (`src/adapters/dynamic_k.py`): Entropy-based
  and growth-based dynamic K — directly applicable to thought latents.

- **MetaController** (`src/adapters/adapter_evolution.py`): Transformer
  over history → next config — reusable as reasoning mode arbiter.

- **Pre-computed 2B trajectories** (`trajectories_2B/`): 50 trajectory
  files with weights, contexts, gradients, and text — can be used to
  bootstrap the latent reasoning system's training.

- **Closed-form SVD** (`src/adapters/flow_weight_expert.py`):
  `compute_closed_form_lora` — the optimal rank-r update direction.
  Theoretically grounded target for any flow matching system.

- **Gradient decomposition** (`src/adapters/gradient_decomposition.py`):
  TaylorContribution, AlternatingTrainer, OverlapConsistency — for
  measuring per-component importance in reasoning trajectories.

- **FULL_SESSION_RECOVERY.md**: Complete 3-phase project state with
  experiment results, key numbers, and architecture maps.

- **IDEAS_BACKLOG.md**: 32 documented ideas (24 Phase 2 + 8 Phase 3)
  with priority, effort, and impact assessments.

## Graveyard (What to Abandon)

- **Diffusion denoising on weights**: MSE stuck at 1.0 — weights are
  unstructured. **Don't retry** unless the object of diffusion changes
  (e.g., to thought trajectories).

- **Weight flow generalization**: After stagnation penalty + alignment
  filtering + 2B trajectories, flow still = zero on held-out text.
  **Don't retry** with the current conditioning approach. Requires
  fundamentally richer conditioning (full hidden state sequence) or
  a different object to flow-match (thoughts).

- **KnitLoRA**: OOM on all model scales. **Forsaken.**

- **DDIM denoising**: Produces worse results than zero initialization
  due to error amplification. **Don't use for weight generation.**

## Phase 4 Executed: Flow Matching Over Thought Trajectories

### Core Finding: ✅ Thoughts HAVE Structure — R² = 0.29

**Weight flow**: R² ≈ 0.0, cos ≈ 0 (failed — weights are random)
**Thought flow**: R² = 0.29, cos = 0.54 (works — hidden states are structured)

### What Worked

| Approach | R² | Cos | Key Insight |
|----------|-----|------|-------------|
| Per-step MLP (no norm) | 0.00 | 0.03 | Can't learn per-layer functions |
| Per-step MLP (normed) | 0.05 | 0.25 | Small but real signal |
| **Perceiver (full trajectory)** | **0.29** | **0.54** | **Cross-layer modeling works** |

### What Was Built (Phase 4)

| Component | File | Role |
|-----------|------|------|
| **ThoughtDiffusion** | `src/adapters/thought_diffusion.py` | Perceiver over full trajectory → predicts all velocities |
| **ThoughtTrajectoryDataset** | `run_thought_flow_train.py` | Loads 500 trajectories, (25, 2048) each |
| **Layer-traj generator** | `generate_thought_trajectories.py` | Records hidden states at each layer for last token |
| **Perceiver trainer** | `run_perceiver_flow_train.py` | MSE loss on (B, 23, 2048) → (B, 23, 2048) |
| **MLP trainer** | `run_thought_flow_train.py` | Per-step prediction (for comparison) |
| **Eval script** | `run_thought_flow_eval.py` | Held-out R², cos, per-layer breakdown |

### Why Perceiver > Per-Step MLP

Per-step prediction sees **one** hidden state → predicts one velocity. Each layer is a different function (different attention/MLP weights). The model must learn 23 functions from ~40 examples each.

The Perceiver sees **all 23 hidden states** simultaneously → predicts all 23 velocities. Latents can attend to cross-layer patterns (how attention evolves, MLP nonlinearities compound). This provides 23× the conditioning signal and enables pattern-matching.

### Data Generated

| Dataset | Location | Size | Content |
|---------|----------|------|---------|
| Layer trajectories (50) | `thought_trajectories/` | 6 MB | Qwen3.5-2B layer-wise hidden states |
| Layer trajectories (500) | `thought_trajectories_500/` | 60 MB | 500 trajectories, 11500 training pairs |
| Reasoning-step trajs (156) | `reasoning_trajectories/` | — | Token-by-token trajectories (slow to generate) |

### Key Commands (Phase 4)

```bash
# Train Perceiver on layer trajectories:
python3 run_perceiver_flow_train.py --trajectories ./thought_trajectories_500/

# Train per-step MLP (comparison):
python3 run_thought_flow_train.py --trajectories ./thought_trajectories_500/

# Generate more layer trajectories:
python3 generate_thought_trajectories.py --model 2B --n-traj 500 --output ./new_trajs/

# Evaluate:
python3 run_thought_flow_eval.py --model-path best_perceiver.pt
```

### Phase 4c: Reading Heads + Correlation Experiments (v0.23.0-readingheads)

| Experiment | Result | Interpretation |
|------------|--------|----------------|
| Flow-correctness correlation | p=0.12, no sig | R²=0.42 insufficient for per-answer detection |
| Reading heads (perplexity) | **r=0.862** | Perceiver latents STRONGLY encode uncertainty |

The reading head is a linear probe (524K params) on frozen Perceiver latents (32×128=4096-dim).
Trained on 90K tokens from GSM8K to predict token-level perplexity. Achieves r=0.86 on held-out data.

**Key files**:
- `run_flow_correctness.py` — Flow-correctness correlation on GSM8K
- `run_reading_heads.py` — Linear probe: latents → perplexity prediction
- `src/adapters/thought_diffusion.py` — Added `return_latents=True` option

### Phase 5: MetaController Uncertainty Analysis (v0.24.0-metacontroller)

| Finding | Result |
|---------|--------|
| Per-token uncertainty gap (correct vs incorrect) | 0.004 (weak) |
| Top-10% high-uncertainty tokens: % from incorrect | 74% (vs 70% baseline, +4pp) |

The reading head's uncertainty signal IS real (r=0.86) but per-token predictive
power for correctness is modest. Suggests need for windowed aggregation or
delta-uncertainty (change from previous token).

**Key file**: `run_metacontroller.py` — Two-pass analysis pipeline:
  Pass 1: `model.generate()` (fast, uses KV cache)
  Pass 2: single full forward pass → Perceiver → reading head per token

## Phase 5b: Three Steering Mechanisms (v0.27-33)

### Mechanism 1: Logit Correction (latents → logit offsets)

| Variant | Per-token accuracy | Generation accuracy |
|---------|-------------------|---------------------|
| 1 head (prompt-trained) | +12.6pp | 0% (distribution shift) |
| 3 heads ensemble | +13.8pp | 0% |
| **3 heads (gen-trained)** | **-3.4pp** | **20%** |

### Mechanism 2: Layer-0 Velocity Injection

| α | Token change rate | Notes |
|---|-------------------|-------|
| 0.0 | 0.35% | Baseline (no injection) |
| 0.5 | ~60% | Changes most tokens |
| 2.0 | 100% | Catastrophic divergence |

### Mechanism 3: KV-Cache Steering (Phase 1)

| Metric | Value |
|--------|-------|
| Geometric alignment (cos_v, layer 23) | 0.94 |
| Divergence rate (α=0.1) | 85-90% |
| Noise control | 95% (signal ≈ noise at binary metric) |
| Next step | Phase 3: downstream accuracy metric |

### The Three Delegations (Conceptual Diffuser Analysis)

All available via `delegation_read(<id>)`:
- **maximum-amethyst-crow**: Structured 3-phase KV-cache plan with noise controls
- **adequate-emerald-wildfowl**: Reading head as central gateway controller, multi-scale velocity fusion
- **gradual-azure-wasp**: 22 assumptions audited (7 wrong), RL correction, SVD-targeted flow, attention diffusion

### Three Delegation Consensus

1. **Reading head (r=0.86) should be the central controller** — gate all interventions by ppl_pred
2. **Distribution shift is fixable** — train on generation data, use RL, or confidence-weighted interpolation
3. **KV-cache steering is geometrically viable** — pre-flight confirmed, needs accuracy metric

## Open Questions (Ranked by Impact)

| # | Question | Why It Matters | What Would Answer It |
|---|----------|---------------|---------------------|
| 1 | Can PPL-modulated correction (α = f(ppl_pred)) fix the distribution shift? | Highest-ROI next experiment | Compare uniform vs ppl-gated correction on GSM8K generation |
| 2 | Does KV-cache steering improve downstream accuracy? | Need better metric than "does token change" | Phase 3: GSM8K accuracy with KV steering |
| 3 | Does multi-scale velocity fusion (TT + Perceiver ensemble) achieve R² > 0.78? | Independent errors improve prediction | Compute inverse-variance weighted ensemble |
| 4 | Can RL-trained correction heads (REINFORCE) learn to steer at generation? | Directly optimizes for what we care about | Implement REINFORCE training loop |
| 5 | What's the optimal uncertainty threshold τ for PPL gating? | Too low → unnecessary compute. Too high → missed opportunities. | Sweep τ on GSM8K validation |

### Key Commands to Resume

```bash
cd /home/filip/Projects/Personal/AI/RankAdaptation

# Check model checkpoints exist:
ls -la best_*.pt best_head_*.pt

# Three delegation outputs:
delegation_read("maximum-amethyst-crow")   # KV-cache plan
delegation_read("adequate-emerald-wildfowl") # Reading head controller  
delegation_read("gradual-azure-wasp")      # Assumption audit, RL/SVD

# KV-cache results:
cat kv_cache_phase1.log | tail -15

# Distribution shift evidence:
cat e2e_results.json

# Run correction head (Phase 1):
python3 run_logit_correction.py --n-test 50

# Run KV-cache Phase 1:
python3 kv_cache_phase1.py

# Train reasoning-step TT:
python3 train_reasoning_transformer.py \
  --trajectories /run/media/filip/B522-875D/Datasets/project_data/reasoning_trajs_5k/all_trajs.pt
```

---

*Recovery completed at: 2026-06-12 19:00 UTC*
*System health: healthy — 34 tags, 54 commits, all changes committed*
