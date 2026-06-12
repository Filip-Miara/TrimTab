# Cross-Session Brief — RankAdaptation / Latent Reasoning
## Sessions: 2026-06-10 → 2026-06-12

## Narrative Summary

This project began as a comparison of 47 LoRA adapter variants (Phase 1-2),
then evolved into a system for **online continual learning via flow matching
over adapter weights** (Phase 3), and is now transitioning into **latent
reasoning via flow matching over thought trajectories** (Phase 4).

### What Was Tried (Phase 3 — Flow Matching Over Weights)

1. **StreamFusion-LoRA** (v0.1): PerceiverFusion bottleneck + TAF routing +
   expert pool for online continual learning. Worked well — eval PPL dropped
   78% across 9 segments on Qwen3.5-2B.

2. **Expert variants** (v0.2-0.3): 20+ architectural variants tested via
   HybridStreamExpert. BVA (bidirectional+vectors+norm) best overall. AFA
   (annealed tanh) best unorthodox variant.

3. **Weight flow matching** (v0.10-0.12): Trained a velocity field v_θ(W_t, t, ctx)
   to predict weight changes from SGD trajectories. **Perfect training fit
   (MSE ≈ 0), but predicts zero on held-out data.** The conditioning
   (mean hidden state + gradient) is information-theoretically insufficient.

4. **Diffusion denoising** (v0.13-0.18): Added noise prediction head + separate
   decode layers. **Denoising stuck at MSE=1.0** — clean adapter weights are
   random, no manifold to learn. Flow matching works; denoising doesn't.

5. **Stagnation penalty + alignment filtering** (v0.19-0.20): Penalized zero
   velocity predictions. Filtered training data to only keep trajectory steps
   where SGD update aligned with gradient (cos > 0.3). **Prevented collapse,
   but flow still ≈ zero on held-out real text** (Flow: 4.10, SGD: 1.52).

6. **2B model trajectories** (v0.20): Generated 50 persistent trajectories
   from Qwen3.5-2B (2048-dim hidden state, 6144-dim context). Reusable
   across all training approaches.

### What Worked

| Component | Status |
|-----------|--------|
| StreamFusion online continual learning | ✅ Works (78% PPL reduction) |
| Flow matching training fit | ✅ MSE ≈ 0 on training data |
| Closed-form SVD optimal update | ✅ 29.8% loss reduction in 1 step |
| Hyperparameter sweep infrastructure | ✅ Batch_size=20 optimal, 21 sps |
| Pre-computed trajectory storage | ✅ 50 trajectories, reusable |
| Dynamic K (entropy + growth) | ✅ Implemented |
| HybridStreamExpert soft flags | ✅ Implemented |
| MetaController + ES evolution | ✅ Implemented |
| BVA expert variant | ✅ Best overall (1269 PPL) |

### What Didn't Work

| Component | Status | Why |
|-----------|--------|-----|
| **Weight generalization to test data** | ❌ Flow ≈ zero on held-out | Conditioning insufficiency |
| **Diffusion denoising on weights** | ❌ MSE=1.0 | Clean weights are unstructured |
| **DDIM generation** | ❌ Worse than zero | Error amplification |
| **ES evolution** | ⚠️ Insufficient | Population variance too low |

### Current Best Hypothesis

The weight manifold is fundamentally unstructured — adapter weights are
initialized randomly and SGD moves them slightly. **Flow matching over
weights fails to generalize because there's no structure to learn.**

The solution is to shift from **weight-space flow matching** to
**latent-space flow matching** — apply the same PerceiverFusion +
WeightDiffusion architecture to thought trajectories (hidden states of
a language model) instead of weight trajectories. Thoughts HAVE structure
(attention patterns, layer activations, token representations) — diffusion
and flow matching should work on this domain.

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

### Next: Phase 5 — MetaController (Uncertainty-Guided Reasoning)

Build on two confirmed capabilities:
1. **R²=0.42** flow matching on thought trajectories
2. **r=0.86** reading heads for uncertainty detection

The MetaController architecture:
```
At each token generation step:
  1. Forward pass → capture hidden states
  2. Perceiver → predict velocity + extract latents
  3. Reading head → compute uncertainty score
  4. If uncertainty > threshold τ:
     - Apply α × velocity to hidden states (steering)
     - Increase temperature for broader exploration
     - Or re-generate with modified state
  5. If uncertainty < threshold τ:
     - Pass-through (default generation)
```

Implementation plan:
1. **Week 1**: Build MetaController class + inference pipeline
   - Load Perceiver + reading head
   - Hook into model forward pass
   - Per-token: read hidden states → Perceiver → reading head → gate decision
2. **Week 2**: Implement adaptive compute
   - If uncertainty > τ: latent CoT (extra iterations in latent space)
   - If uncertainty < τ: continue normally
   - Test on GSM8K: does accuracy improve?
3. **Week 3**: Integrate flow steering
   - Apply α × Perceiver velocity when uncertainty is high
   - Measure: accuracy delta, generation quality, computational cost
4. **Week 4**: End-to-end evaluation
   - Compare: base model vs MetaController vs MetaController+steering
   - Metrics: accuracy, tokens per answer, variance across seeds

## Open Questions (Ranked by Impact)

| # | Question | Why It Matters | What Would Answer It |
|---|----------|---------------|---------------------|
| 1 | Can MetaController uncertainty threshold improve reasoning accuracy? | Direct test of adaptive compute. | Compare GSM8K accuracy with/without MetaController. |
| 2 | What's the optimal uncertainty threshold τ? | Too low → unnecessary compute. Too high → missed opportunities. | Sweep τ=0.1-0.9 on GSM8K validation. |
| 3 | Does velocity steering + uncertainty gating outperform either alone? | Tests if both components provide additive benefit. | Ablation study on GSM8K. |
| 4 | What is the optimal latent dimensionality K for a "thought"? | Too few → collapse. Too many → noise. | Sweep K=4,8,16,32,64 with DynamicPerceiverWrapper. |
| 5 | Does reasoning-step flow matching work better than layer-to-layer? | Which formulation for latent reasoning? | Generate 1000+ reasoning-step trajectories, compare R². |

### Key Commands to Start

```bash
# The current project state is fully committed.
# To resume:
cd /home/filip/Projects/Personal/AI/RankAdaptation
git log --oneline -5
cat FULL_SESSION_RECOVERY.md    # complete project state
cat IDEAS_BACKLOG.md            # all open ideas

# For the latent reasoning pivot, read:
cat thoughts/latent_reasoning_analysis.md
cat thoughts/self_modulating_transformer_architecture.md

# Existing pre-computed trajectories:
ls trajectories_2B/             # 50 trajectories from Qwen3.5-2B

# To verify the stagnation-trained model works:
python3 run_diffusion_flow_eval.py

# To generate more 2B trajectories:
python3 generate_trajectories.py --model 2B --n-traj 50 --output ./trajectories_2B/
```

---

*Recovery completed at: 2026-06-12 00:30 UTC*
*System health: healthy — 20 tags, 53 commits, all changes committed*
