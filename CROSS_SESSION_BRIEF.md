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

## Open Questions (Ranked by Impact)

| # | Question | Why It Matters | What Would Answer It |
|---|----------|---------------|---------------------|
| 1 | Do thought trajectories have enough structure for flow matching to generalize? | If yes, our entire infrastructure transfers. If no, need new approach. | Train WeightDiffusion on thought trajectories from a reasoning dataset. Measure held-out MSE. |
| 2 | What is the optimal latent dimensionality K for a "thought"? | Too few → collapse. Too many → noise. | Sweep K=4,8,16,32,64 on a small reasoning task with DynamicPerceiverWrapper. |
| 3 | Should reasoning be monotonic (each step improves) or exploratory (can get worse)? | Determines whether diffusion (monotonic) or flow matching (any trajectory) is correct formulation. | Compare DDIM denoising vs. flow integration on thought trajectories. |
| 4 | Can reading heads extract useful concepts (uncertainty, contradiction) from thought latents? | Would provide conditioning signal for MetaController to decide reasoning mode. | Train linear probes on thought latents with contrastive pairs. |
| 5 | Does the closed-form SVD optimal update work for thoughts as well as weights? | If yes, training signal is always meaningful (not zero). | Compute SVD of thought residual. Compare to observed thought trajectories. |

## Recommended Next Phase

### Phase 4: Latent Reasoning Engine

Build on the existing infrastructure to create a system that performs
iterative reasoning in latent space:

1. **Week 1**: Adapt PerceiverFusion to process thought trajectories
   (hidden states from a language model) instead of adapter weights.
   - Replace weight-space tensors with thought-space tensors
   - Keep the same Perceiver bottleneck + cross/self-attention
   - Initial K fixed at 16 (from DynamicPerceiverWrapper)

2. **Week 2**: Train WeightDiffusion on thought trajectories from a
   reasoning dataset (e.g., GSM8K chain-of-thought traces).
   - Collect thought trajectories = hidden states at each reasoning step
   - Train flow matching objective (predict next thought from current)
   - Evaluate: does flow matching generalize to held-out reasoning data?

3. **Week 3**: Add structured reading heads + mode arbitration.
   - Linear probes to extract uncertainty, contradiction from thought latents
   - MetaController decides: pass-through, self-correct, or latent CoT mode
   - Train via RL (task accuracy improvement as reward)

4. **Week 4**: End-to-end evaluation on reasoning benchmarks.
   - Correctness, efficiency (steps to convergence), consistency,
     coherence, adaptivity, causal density
   - Compare against chain-of-thought, self-consistency, tree-of-thought

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
