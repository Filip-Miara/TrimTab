
---

## Avenue 1: Complete the Contrastive Steering Loop
**ROI: ★★★★★** | **Effort: 1-2 days** | **Novelty: High**

### Status
Contrastive TTs (v_correct, v_incorrect) are **already trained** on Qwen2.5-7B (R²≈0.83 each) with checkpoint files \est_tt_correct.pt\, \est_tt_incorrect.pt\. Per-layer sweeps have NOT been run with these -- only the standard TT (all data) was used for the L8→+20pp result.

### Key Experiments

**1A. Contrastive Per-Layer Sweep on Qwen2.5-7B**
Sweep α ∈ {0.01, 0.05, 0.1, 0.3}, β ∈ {0, 0.01, 0.05, 0.1} on the top 6 trim-tab layers and death layers as controls. Test four steering modes:

1. Attraction only: h' = h + α · v_correct(h)
2. Repulsion only: h' = h - β · v_incorrect(h)
3. Combined: h' = h + α·v_correct(h) - β·v_incorrect(h)
4. Normalized combined: h' = h + α·(v_correct - v_incorrect)(h)

**Hypothesis**: Contrastive direction outperforms standard TT because it directly models the difference between correct and incorrect trajectories. Expected: L8 contrastive → +25-30pp (vs +20pp standard).

**1B. Asymmetric α (Critical Variant)**
Use independent α_c (attraction toward correct) and α_i (repulsion from incorrect):
h' = h + α_c · TT_correct(h) - α_i · TT_incorrect(h)
Sweep α_c ∈ {0.01, 0.05, 0.1, 0.3}, α_i ∈ {0.01, 0.05, 0.1, 0.3}.
**Rationale**: Attraction and repulsion may need different strengths. Too much repulsion could push the hidden state into territory mapped only by incorrect trajectories.

**1C. Multi-Head Contrastive Ensemble**
Train N=5 independent TT pairs on bootstrapped trajectory subsets:
v_ensemble = mean(TT_correct_k) - mean(TT_incorrect_k)
**Benefits**: Bagging reduces variance. Ensemble disagreement gives per-token steering confidence.

### Expected Outcomes
- Best case: L8 contrastive + asymmetric α → **+30pp** (73% → 95-100% on GSM8K subset)
- Minimal: +5-10pp over standard TT (still validates the contrastive approach)
- Failure mode: α_c and α_i compete destructively

---

## Avenue 2: Multi-Layer Steering Combinatorics
**ROI: ★★★★** | **Effort: 2-3 days** | **Novelty: High**

### Status
All experiments used single-layer steering. The best layers are:
- L2: +17pp (early layer, shapes computation globally)
- L8: +20pp (mid-layer, the strongest trim tab)
- L10: +17pp (late-mid, also strong)
- L3/L5: +13pp (supporting trim tabs)

### Key Experiments

**2A. Best-Pair Combinations**
Test L2+L8, L5+L8, L8+L10 with per-layer α[l]:
h'[L2] = h[L2] + α_2 · v_steer(h[L2])
h'[L8] = h[L8] + α_8 · v_steer(h[L8])
with independent α_2, α_8 ∈ {0.01, 0.05, 0.1}.

**Hypothesis**: L2 + L8 could be additive (+17 + +20 → potential +30-35pp) if they affect different computational subspaces. L2 shapes early representations; L8 refines mid-layer reasoning.

**2B. Best-Triplet Combination**: L2+L5+L8 with global α vs independent α[l].

**2C. Trim-Tab + Anti-Death-Layer Combination**
Steer L8 toward correct while neutralizing L9:
h'[L9] = h[L9] - γ · v_steer(h[L9]) (repel from death direction)
**Hypothesis**: If L9 amplifies error patterns, steering AWAY from its trajectory at L9 while steering TOWARD correct at L8 yields compound improvement.

**2D. Sequential Layer Steering**: Steer different layers at different tokens during generation. Tests whether layers contribute at different reasoning stages.

### Expected Outcomes
- Additive: +30-35pp from L2+L8
- Synergistic: >35pp if subspaces are independent
- Sequential: Reveals reasoning-stage-specific layer roles

---

## Avenue 3: Reading-Head Gated Adaptive Steering (The Triple Controller)
**ROI: ★★★★★** | **Effort: 3-5 days** | **Novelty: Very High**

### Status
The reading head achieves r=0.86 correlation with token perplexity. Delegation consensus identified it as the ideal central controller. But the model is "confidently wrong" -- perplexity doesn't correlate well with correctness.

### Proposal: Per-Token Adaptive α(layer, token)
Instead of constant α, compute α per (layer, token) via the reading head:
α(l, t) = σ(β_l · (τ_l - reading_head(h[l][t])))
Where τ_l is a per-layer uncertainty threshold and β_l scales sensitivity.

**Intuition**: When the model is uncertain (high predicted perplexity), steering should be stronger. When confident (low perplexity), steering should be weak.

### Key Experiments

**3A. Gating Functions Comparison**: Hard gate vs Soft gate (sigmoid) vs Linear modulation.

**3B. Per-Layer τ Sweep**: Sweep τ_l ∈ {0.5, 1.0, 1.5, 2.0} per trim-tab layer.

**3C. Correctness-Adaptive Switch**: Use the *difference* between TT_correct and TT_incorrect velocity magnitudes as the gating signal: α = σ(||v_correct(h)|| - ||v_incorrect(h)||). When TT_correct predicts a stronger direction, the model is in "steerable territory."

**3D. DoRA-Inspired Steering (from SESSION_DEBRIEF)**
The TT predicts velocity (direction), but the reading head provides magnitude confidence:
steering = direction(TT_pred) × magnitude(reading_head(h))
Separating direction from magnitude mirrors DoRA's weight decomposition.

### Expected Outcomes
- Adaptive α should outperform constant α by 3-8pp
- Triple-control system (reading head + contrastive TT + per-layer α) is unmatched in literature
- Key risk: the model IS confidently wrong -- reading head may predict high confidence for wrong tokens

---

## Avenue 4: The Steering Fragility Profile -- Capability Diagnostics
**ROI: ★★★★** | **Effort: 1 week** | **Novelty: Very High (Potentially Publishable)**

### Core Insight
The trim-tab/death-layer pattern is a **fingerprint of capability**. Models without steerable capability (SmolLM2-360M, Qwen2.5-0.5B, TinyLlama, Math-1.5B) show NO trim tabs -- all layers are neutral or harmful. Models WITH capability (Qwen2.5-7B at 73%) show a rich trim-tab landscape.

**This means**: The per-layer steering sensitivity profile IS a diagnostic of whether the model has a capability.

### Key Experiments

**4A. Full Model Family Trim-Tab Mapping**
Per-layer sweeps on Qwen2.5-0.5B → 1.5B → 3B → 7B → 14B → 32B.
Hypothesis: Trim-tab amplitude increases with model size. Death layers emerge at a specific threshold.

**4B. Task-Specific Trim-Tab Maps**
Test Qwen2.5-7B on GSM8K (math), ARC (science), BBH (reasoning), MMLU (knowledge), HumanEval (code).
Hypothesis: Different tasks have different trim-tab layers. Math → L8, Code → L2.

**4C. Training Checkpoint Trim-Tab Evolution**
Track trim-tab emergence across model training. Do they appear gradually or suddenly?

**4D. Universal Position Hypothesis**
Test if trim tabs always appear at ~28% depth (L8 of 28) across LLaMA-3.2, Gemma-2, Mistral-7B.

### Expected Outcomes
- **Publishable insight**: Trim-tab maps as a new class of capability diagnostic for LLMs
- **Emergent capability detection**: Per-layer sweep on a new model → instantly know what capabilities exist
- **Training guidance**: If trim tabs don't emerge, the model hasn't internalized the capability

---

## Avenue 5: Steering + RL Fine-Tuning Synergy
**ROI: ★★★★★** | **Effort: 1-2 weeks** | **Novelty: Very High**

### Status
"Steering cannot create capability" -- models below ~40% GSM8K cannot be steered toward better answers. Steering only amplifies existing capability.

### Proposal: Use Steering as the Action Space for RL
Instead of the model generating tokens and hoping for reward, **the RL agent steers the model** at trim-tab layers. Action space = (layer, α, direction). Reward = GSM8K accuracy.

### Key Experiments

**5A. REINFORCE Over Steering Parameters**
Learn α_c[l], α_i[l] per layer via policy gradient. Action: (α_c for L2, α_i for L2, α_c for L8, α_i for L8, ...). Reward: GSM8K accuracy.

**5B. PPO for Multi-Token Steering**: Handle per-token credit assignment -- which steering action at which token caused correctness?

**5C. Curriculum Steering**: Start with steering-heavy policy on easy problems, gradually reduce α as model learns. The hope: the model **internalizes the steering direction** and eventually doesn't need it. This would break the "steering can't create capability" barrier.

**5D. SVD-Optimal Steering Targets**: Instead of MSE-trained TT, compute the closed-form SVD of (h_correct - h_incorrect) × h^+ -- the optimal rank-1 steering direction per layer.

### Expected Outcomes
- First demonstration of **learned steering policies** via RL
- Curriculum steering potentially **breaks the fundamental limitation** "steering can't create"
- SVD targets provide **theoretical grounding** for optimal steering direction

---

## Avenue 6: Flow Matching Over Steering Trajectories
**ROI: ★★★★** | **Effort: 1 week** | **Novelty: High**

### Status
- Weight flow: MSE≈0 train, ≈0 test (conditioning insufficiency)
- Thought flow: R²=0.29-0.75 (structure exists but limited)
- Steering flow: Not attempted

### Proposal
Flow match over **steering trajectories** -- sequences of (hidden_state, steering_action, correctness). Steering trajectories are doubly structured:
1. Track the model's internal computation (structured by architecture)
2. Track how steering modifies that computation (structured by the steering mechanism)

Training data: For each problem and steering policy π, record [(h_0, π_0, r), ..., (h_T, π_T, r)] and train v_θ(h_t, π_t, context) → Δh.

At inference: start with no steering, predict unsteered trajectory, choose steering that pushes toward correct manifold, apply, observe new h, repeat.

### Why This Is Different From Current TT
Current TT predicts velocity from hidden_states alone. This predicts velocity from (hidden_states, steering_policy) -- enabling what-if analysis, optimal steering policy selection, and counterfactual reasoning about unsteered trajectories.

### Expected Outcomes
- R² > 0.8 for steering-conditioned velocity prediction
- Model-predictive control for LLM steering (plan before executing)
- First integration of flow matching with latent steering

---

## Avenue 7: Continuous Expert Manifold -- Flow-Routed Steering
**ROI: ★★★★** | **Effort: 2-3 weeks** | **Novelty: Very High (Unique Research Gap)**

### Core Concept (From DiMAE)
Experts (LoRA adapters, steering policies, attention heads) are NOT discrete -- they are points on a continuous manifold parameterized by a latent code z. The router outputs z, and the "expert" is the point on the manifold at z. Flow matching over z navigates the expert manifold.

### Application to Steering
Learn a **continuous steering manifold**: z ∈ ℝ^d → steering policy π_z = (α_c[l], α_i[l] for each layer).
Flow: dz/dt = v_θ(z, t, context).
Train v_θ to navigate toward high-reward regions of z-space.
At inference: flow from random z → optimal steering → apply π_z per token.

### Key Advantage Over Discrete Steering
Discrete: 6 layers × 3 α values = 18 configurations. Continuous: explores full manifold.
Flow field learns that L2+L8 are complementary (increase both), L8+L9 are conflicting (decrease L9).

### Existing Infrastructure Ready For This
- **PerceiverFusion**: Input → latent codes (built)
- **FlowWeightExpert**: Velocity fields over latent spaces (built)
- **MetaController**: Latent codes → actions (built)
- **Soft flags**: Continuous [0,1] blending (built)

---

## Avenue 8: Emergent Capability Detection via Geometric Analysis
**ROI: ★★★** | **Effort: 1 week** | **Novelty: Very High (Potentially Publishable)**

### Core Question
What changes geometrically in the hidden state manifold when a model crosses the capability threshold? Models below ~40% GSM8K show no trim tabs. What unlocks steerability?

### Key Experiments

**8A. Manifold Dimensionality vs Capability**: Estimate intrinsic dimensionality of the hidden state manifold at each layer for Qwen2.5-0.5B (6%), Qwen2.5-1.5B (?), Qwen2.5-7B (73%). Hypothesis: Capable models have higher effective rank in mid-layers.

**8B. Correct vs Incorrect Trajectory Divergence**: Measure cosine similarity between correct and incorrect trajectories per layer. Hypothesis: At L8 they strongly diverge; at L9 they are nearly identical; in non-capable models they are randomly related.

**8C. Velocity Field Curvature**: curvature(l) = ||v(h[l+1]) - v(h[l])|| / ||h[l+1] - h[l]||. Hypothesis: High curvature at trim-tab layers, low curvature at death layers.

### Expected Outcomes
- **Geometric capability signature**: Assess capability from manifold geometry alone, no downstream task needed
- **Understanding why steering works**: It exploits curvature in the velocity field

---

## Avenue 9: Cross-Architecture Steering Transfer
**ROI: ★★★★** | **Effort: 3-5 days** | **Novelty: High**

### Status
SmolLM2-360M (MHA) TT → Qwen2.5-7B (MHA): L8 preserved as best trim tab.

### Key Experiments

**9A. MHA → Hybrid Attention Transfer**: Qwen2.5-7B TT → Qwen3.5-2B. If velocity dynamics are universal, pattern should transfer despite different attention.

**9B. Cross-Language Transfer**: Train TT on English math (GSM8K) → steer Chinese/French math. Hypothesis: hidden states are language-agnostic at mid-layers.

**9C. Cross-Task Transfer Expansion**: GSM8K-trained TT → ARC, BBH, MMLU, HumanEval. SVAMP already confirmed L8:+4pp, L9:-14pp.

### Expected Outcomes
- Universal trim-tab position (~28% depth) would reveal a **fundamental property of transformer residual stream**
- Steering may work across languages without retraining

---

## Avenue 10: The Negative Space -- What Steering Cannot Do
**ROI: ★★★** | **Effort: 2-3 days** | **Novelty: Medium (but important)**

### Known Negative Results
1. Cannot create capability (SmolLM2, TinyLlama)
2. Hybrid attention resists steering (Qwen3.5-2B)
3. Math-1.5B has no trim tabs despite 38% baseline
4. Model is confidently wrong (reading head can't distinguish correct/incorrect)
5. Weight flow predicts zero on test data

### Key Experiments

**10A. Capability Threshold Sweep**: Find the precise baseline accuracy where steering starts working using Qwen2.5-1.5B and Qwen2.5-3B.

**10B. Steering Decay vs Problem Difficulty**: Hypothesis: steering helps most on medium-difficulty problems. On easy (already correct) and hard (beyond capability), no effect.

**10C. Steering as Model Confidence Test**: If a model has trim tabs → it genuinely has the capability. No trim tabs → memorized patterns without geometric representation.

---

## Avenue 11: KV-Cache Steering Phase 3 -- Downstream Accuracy
**ROI: ★★★★** | **Effort: 3-5 days** | **Novelty: Medium (completion of existing work)**

### Status
- Phase 1 (geometric): cos_v=0.94 at L23, divergence 85-90% ✅
- Phase 2 (multi-layer): Scaling across layers ⚠️ Partially done
- Phase 3 (accuracy): Not yet measured ❌

### Key Experiments

**11A. KV-Cache Steering on Trim-Tab Layers**: Previous experiments used Qwen3.5-2B at L23. Run on Qwen2.5-7B at L8 (the confirmed trim tab).

**11B. KV-Cache vs Direct Hidden-State Steering Comparison**: Direct modifies current hidden state; KV-cache modifies future attention. Which is more effective at the same layer?

**11C. Attention Weight Biasing Contingency**: If KV steering fails at L8, fallback to directly modifying attention distributions.

---

## Avenue 12: The DiMAE Architecture -- Implementation Roadmap
**ROI: ★★★★★** | **Effort: 2-3 weeks** | **Novelty: Very High (Published-Gap Level)**

### Overview
DiMAE (Diffusive Mixture of Attentive Experts) synthesizes:
- PerceiverFusion (latent encoding) ✅ Built
- Flow-corrected latent queries (velocity field over latents) ✅ Infrastructure exists
- Diffusive refinement of expert outputs (structured diffusion) ❌ New
- Dynamic K + MetaController ✅ Built
- Reading head uncertainty gating ✅ Built

### Implementation Phases

**Phase A (Week 1): Flow-Corrected Latents**
Modify ThoughtDiffusion to predict latent velocity dL/dt. Add a velocity head v_θ(Z, t, context) → dZ/dt. Train via flow matching. Tests whether flow over latent queries achieves R² > 0.4 (vs 0.29 for static Perceiver).

**Phase B (Week 1): Entropy-Based Dynamic K + Flow**
Connect entropy computation (dynamic_k.py) with flow-corrected latents. Prune high-entropy latents, grow when all are specialized.

**Phase C (Week 2): Diffusive Expert Output Refinement**
Add DDPM-style denoising over expert outputs E_k. The key: expert outputs are structured (unlike random weights), so diffusion should converge (unlike the WeightDiffusion failure).

**Phase D (Week 2-3): Reading Head + MetaController Integration**
Gate diffusive refinement by reading head uncertainty. MetaController decides denoising steps and pruning thresholds per input.

### Key Testable Predictions
1. Flow over latent queries: R² > 0.4 (vs 0.29 static)
2. Diffusion over expert outputs: MSE < 0.1 (vs 1.0 for weights)
3. Full DiMAE: >5pp accuracy improvement over static Perceiver baseline

---

## Priority Summary

| Priority | Avenue | ROI | Effort | Why Now |
|----------|--------|:---:|:------:|---------|
| **P0** | 1. Contrastive Steering | ★★★★★ | 1-2 days | TTs already trained, evaluation is the bottleneck |
| **P0** | 3. Reading-Head Gated Steering | ★★★★★ | 3-5 days | Reading head and TTs both exist; integration is the gap |
| **P0** | 5. Steering + RL Synergy | ★★★★★ | 1-2 weeks | Could break the fundamental "can't create capability" limit |
| **P1** | 2. Multi-Layer Combinatorics | ★★★★ | 2-3 days | Single-layer ceiling is +20pp; pairs could push to +35pp |
| **P1** | 11. KV-Cache Phase 3 | ★★★★ | 3-5 days | Completes existing pre-flighted work |
| **P1** | 12. DiMAE Architecture | ★★★★★ | 2-3 weeks | Grand synthesis of all components |
| **P2** | 4. Fragility Profile | ★★★★ | 1 week | Publishable, but requires multiple models |
| **P2** | 6. Flow Over Steering Trajs | ★★★★ | 1 week | Novel direction, needs infrastructure |
| **P2** | 7. Continuous Expert Manifold | ★★★★ | 2-3 weeks | High novelty, high effort |
| **P2** | 9. Cross-Architecture Transfer | ★★★★ | 3-5 days | Builds on existing transfer result |
| **P3** | 8. Geometric Analysis | ★★★ | 1 week | Foundational understanding, lower direct impact |
| **P3** | 10. Negative Space | ★★★ | 2-3 days | Important but descriptive, not generative |

## Quick-Start Commands

`ash
# === P0: Contrastive Sweep on Qwen2.5-7B ===
python run_contrastive_eval.py --n-test 100 --alpha 0.1 --layers 0 1 2 3 4 5 6 7 8 9 10
python run_contrastive_eval.py --n-test 100 --alpha_c 0.1 --alpha_i 0.05 --asymmetric --layers 2 8

# === P0: Adaptive Reading Head Steering ===
# (New script needed: run_adaptive_steering.py)
# Combines reading head + contrastive TT + per-layer gating

# === P1: Multi-Layer Steering ===
python run_7b_steering.py --layers 2 8 --alpha 0.05 0.1
python run_7b_steering.py --layers 2 5 8 --per-layer-alpha 0.05 0.1 0.1

# === P1: KV-Cache Phase 3 ===
python kv_cache_phase1.py --model Qwen2.5-7B --layer 8 --alpha 0.1

# === P2: Steering Fragility Profile ===
python run_per_layer_sweep.py --model Qwen2.5-1.5B --dataset GSM8K --n-test 100
python run_per_layer_sweep.py --model Qwen2.5-3B --dataset GSM8K --n-test 100
`

## Cross-Pollination Opportunities

| Project | Connection | What TrimTab Offers | What External Offers |
|---------|-----------|---------------------|---------------------|
| **qwen3_trm** (distilled student) | Frozen student ceiling | Steering can amplify student features | Student provides 260× faster training |
| **EGGROLL_QLoRA** | LoRA adaptation | Trim-tab maps identify which layers to LoRA-tune | QLoRA enables efficient fine-tuning of steerable layers |
| **QA-DVoRA** | Quantization | Steering fragility profile under quantization | Quantization-aware training for steerable models |
| **efficient-dlm** | Diffusion LMs | Steering + diffusion could control generation quality | Diffusion provides alternative generation paradigm |
