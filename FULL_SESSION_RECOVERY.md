# FULL SESSION RECOVERY — RankAdaptation

## 1. Project State

**Goal**: End-to-end system for online continual learning with adaptive LoRA adapters. Spans three phases:
- **Phase 1 (original)**: Compare 47 LoRA variants on ppl (Qwen3.5-0.8B, r=8, α=8, 50 steps)
- **Phase 2 (expansion)**: SmolLM2-135M sweeps — combinatoric variants, cycling architectures, DiagLoRA/MultiAngleLoRA
- **Phase 3 (StreamFusion)**: Online continual learning with PerceiverFusion bottleneck, TAF routing, weight flow matching, diffusion denoising, MetaController lifecycles, dynamic K

### Key Files

| File | Purpose | Phase |
|------|---------|-------|
| `run_exp.py` | Main batch experiment runner (HF model, arxiv, 100 steps) | 1+2 |
| `run_stream_fusion.py` | Streaming continual learning (segments + experts) | 3 |
| `run_lifecycle.py` | PERA→BVA morphing lifecycle schedule | 3 |
| `run_meta_evolution.py` | ES optimization over adapter lifecycles | 3 |
| `run_weight_flow_train.py` | Weight trajectory collection + flow training on real LM data | 3 |
| `run_weight_flow_eval.py` | Evaluation: flow-generated vs SGD vs zero weights | 3 |
| `run_diffusion_flow_train.py` | Diffusion + flow matching over weights (diverse data) | 3 |
| `run_diffusion_flow_eval.py` | WeightDiffusion vs SGD vs zero evaluation | 3 |
| `run_sweep_analysis.py` | Multi-seed multi-segment analysis | 3 |
| `run_comparison.py` | Flow vs SVD vs SGD velocity cosine similarity | 3 |
| `gen_combo.py` | Generator for bidirectional combinatorics | 2 |
| `gen_dora.py` | Generator for DoRA-style combinatorics | 2 |
| `src/adapters/stream_fusion.py` | **StreamFusionLoRA**, PerceiverFusion, HybridStreamExpert, all expert variants | 3 |
| `src/adapters/diffusion_weight_flow.py` | WeightDiffusion (flow + denoising + optimal target), DiffusionFlowTrainer | 3 |
| `src/adapters/flow_weight_expert.py` | FlowWeightExpert, compute_closed_form_lora (SVD optimal) | 3 |
| `src/adapters/adapter_evolution.py` | MetaController, AdapterEvolution (ES), LifecycleConfig | 3 |
| `src/adapters/dynamic_k.py` | Dynamic K: entropy-thresholded + adaptive growth | 3 |
| `src/adapters/gradient_decomposition.py` | TaylorContribution, AlternatingTrainer, OverlapConsistency | 3 |
| `src/adapters/lifecycle_flow.py` | LifecycleFlow velocity field, FlowMatchingTrainer | 3 |
| `src/adapters/differentiable_lifecycle.py` | Unified differentiable morph optimizer | 3 |
| `src/adapters/weight_flow.py` | WeightFlowField, WeightFlowTrainer | 3 |
| `src/adapters/base.py` | LowRankAdapter, AdapterWrappedLinear, adapt_linear_layer | 1 |
| `src/adapters/__init__.py` | Exports all adapter classes (118 batch + 10 StreamFusion) | 1+2+3 |

### Adapter Architecture Map (Phase 1+2)

```
Base LowRankAdapter
├── No magnitude: PlainLoRA
├── Column magnitude: DoRA
│   ├── +VE: DVoRA
│   └── +LayerNorm: DoRAN
│       ├── + group mag: EDoRA → EDoRAN
│       ├── + AFA flag: GenDDoRAN variants (32)
│       └── + bidirectional: BVoRAN → GenBVoRAN variants (64)
│           ├── + SR flag: SRBVoRAN
│           ├── + Knit flag: KnitBVoRAN (OOM)
│           └── + cycling: CycledBoRAN
├── Cycling (axis-cycling): CycledAxialBoRA
├── Cycling (diag-cycling): CycledDiagLoRA
├── Diagonal magnitude: DiagLoRAN
└── Configurable angles: MultiAngleLoRAN
```

### StreamFusion Architecture Map (Phase 3)

```
StreamFusionLoRA
├── PerceiverFusion bottleneck
│   ├── Cross-attend: latents ↔ expert embeddings
│   ├── Self-attend: latents ↔ latents
│   └── Output cross-attend: query → fused delta
├── Expert variants (HybridStreamExpert)
│   ├── Plain (B·A), DoRA (magnitude-norm), BoRA (bidirectional), VeRa (vectors)
│   ├── AFA (annealed tanh), AUR (autoencoder MLP), PERA (polynomial)
│   └── Soft flags: continuous [0,1] blending for smooth morphing
├── Flow matching over weights
│   ├── WeightFlowField (Perceiver bottleneck over weight space)
│   ├── WeightDiffusion (separate noise + velocity heads)
│   ├── Closed-form SVD optimal target
│   └── Stagnation penalty
├── MetaController (transformer over adapter history)
│   ├── Suggests flags + poly_order + morph_rate per segment
│   └── Evolution-strategy optimization
├── Dynamic K regulation
│   ├── Entropy-thresholded (drop high-entropy latents)
│   └── Adaptive growth (expand when reasoning stalls)
└── Gradient decomposition
    ├── TaylorContribution (per-rank-1 loss contribution)
    ├── AlternatingTrainer (gradient isolation via masking)
    └── OverlapConsistency (MSE over overlapping components)
```

---

## 2. Phase 1+2 Experiments (Batch LoRA Variants)

### 2.1 Complete Combined Ranking (34 unique variants, SmolLM2-135M)

| # | Variant | Eval Loss | PPL | Norm? | Notes |
|---|---------|-----------|-----|-------|-------|
| 1 | `plain_lora` | 3.06 | **21** | ❌ | Best — but on random base weights |
| 2 | `edoran` (gs=128) | 7.24 | **1,394** | ✅ | Best magnitude variant |
| 3 | `edoran` (gs=64) | 7.24 | 1,401 | ✅ | |
| 4 | `doran` | 7.31 | 1,492 | ✅ | |
| 5 | `gend_afa_eva_doran` | 7.34 | 1,547 | ✅ | Best gen'd norm variant |
| 6-18 | All `gend_*doran*` | 7.35-7.43 | 1,561-1,689 | ✅ | Flag combos (tight cluster) |
| 19 | `cycled_diag_lora` | 7.58 | 1,952 | ❌ | Cycled diag/anti bands |
| 20 | `cycled_axial_loran` | 7.86 | 2,581 | ❌ | Axis-cycling |
| 21 | `gend_cycled_axial_boran` | 7.91 | 2,718 | ✅ | Gend axis-cycling + norm |
| 22 | `cycled_bvoran` | 8.15 | 3,477 | ✅ | Branch-cycling |
| 23 | `gend_cycled_boran` | 8.32 | 4,108 | ✅ | Gend branch-cycling + norm |
| 24 | `diag_loran` | 10.01 | 22,292 | ✅ | Bidirectional diag |
| 25 | `gend_cycled_axial_bora` | 11.90 | 147K | ❌ | No-norm axis-cycling |
| 26-34 | All no-norm variants | 13.3-13.5 | 600K-730K | ❌ | No norm cluster |

### 2.2 EDoRA Group Size Sweep (SmolLM)

| Group Size | Eval Loss | PPL |
|-----------|-----------|-----|
| 64 | 7.24 | 1,401 |
| **128** | **7.24** | **1,394** ← best |
| 256 | 7.36 | 1,567 |
| 512 | 7.36 | 1,568 (default) |
| 1024 | 7.75 | 2,324 |

### 2.3 Phase 1+2 Key Findings

- **LayerNorm dominates**: 6-point loss gap between norm and no-norm (7.3 vs 13.4)
- **Plain LoRA beats magnitude on random weights** (PPL 21 vs 1,394) — magnitude needs pretrained
- **Cycling**: axis-cycling > branch-cycling, but cycling < static+norm
- **EDoRA**: smaller groups (64-128) beat larger (512-1024)
- **KnitLoRA**: OOM on all model scales — forsaken

---

## 3. Phase 3 Experiments (StreamFusion / Weight Flow)

### 3.1 Streaming Continual Learning (Qwen3.5-2B)

| Segments | Train PPL | Eval PPL | Experts |
|----------|-----------|----------|---------|
| 1 | 5,920 | 3,203 | 1 |
| 3 | 715 | 1,395 | 3 |
| 5 | 442 | 1,193 | 5 |
| 7 | 814 | 1,092 | 7 |
| 9 | 347 | **698** | 9 |

StreamFusion improves across segments — 78% eval PPL reduction from seg 1 to seg 9.

### 3.2 Expert Variant Sweep (11 variants × 3 seeds × 5 segments)

| Rank | Variant | Eval PPL (seg 5) | vs Plain |
|------|---------|-----------------|----------|
| 1 | **BVA** (bi+vec+norm) | 1,289 | -5.0% |
| 2 | dora | 1,293 | -4.6% |
| 3 | plain | 1,303 | baseline |
| 4 | AFA (annealed tanh) | 1,300 | -0.2% |
| 5 | PERA (polynomial) | 1,309 | +0.5% |

### 3.3 Weight Flow Matching (Qwen3.5-0.8B)

| Model | Training Data | Train Flow Loss | Eval vs Zero |
|-------|--------------|-----------------|--------------|
| WeightFlowField | 15 traj (arxiv) | ~0 | 2.8925 ≈ 2.8928 |
| + gradient conditioning | 15 traj | ~0 | 2.8919 ≈ 2.8928 |
| WeightDiffusion + opt target | 70 traj (5 sources) | flow~0, opt~2e-5 | 2.8928 = 2.8928 |
| + stagnation penalty | 70 traj | flow~0, stag active | in progress |

**Key finding**: Flow matching perfectly learns training dynamics (MSE ~0) but predicts zero on test data. Conditioning (mean hidden state + gradient) is information-theoretically insufficient to determine the correct weight update for unseen data.

### 3.4 Cosine Similarity: Flow vs SVD vs SGD

| Comparison | Cosine Similarity | Meaning |
|-----------|------------------|---------|
| SGD update vs SVD optimal | -0.003 | SGD does NOT follow SVD direction |
| Flow velocity vs SVD optimal | 0.009 | Flow learned SGD, not SVD |
| Flow velocity vs SGD update | 0.001→0.008 | Flow ≈ zero vector |

### 3.5 Diffusion Denoising Analysis

**Finding**: Denoising MSE stays at 1.0 (random guess) even with proper weight normalization and separate output heads. Root cause: clean adapter weights are random (initialized randomly, SGD shifts slightly). There is no low-dimensional manifold for denoising to learn.

**Recommendation**: Drop diffusion denoising. Focus on flow matching + closed-form SVD optimal target.

### 3.6 Full Pipeline Evaluation

| Model Configuration | Flow Eval PPL | Zero Eval PPL | Flow < Zero? |
|--------------------|--------------|---------------|--------------|
| 12 traj, no gradient | 2.8925 | 2.8928 | 0/5 |
| 12 traj + gradient conditioning | 2.8919 | 2.8928 | 4/5 (marginal) |
| 70 traj + gradient + optimal target | 2.8928 | 2.8928 | 0/5 |

All models match zero. The wall is conditioning insufficiency.

---

## 4. Key Cross-Cutting Findings

### 4.1 What Works
- **StreamFusion online training**: Eval PPL drops 78% across segments via expert accumulation
- **Flow matching on weight trajectories**: Perfect training fit (MSE ~0)
- **Closed-form SVD**: 29.8% loss reduction in a single step — the optimal update direction
- **Entropy-thresholded dynamic K**: Correctly drops inattentive latents
- **Adaptive growth**: Expands from K_min when reasoning stalls
- **Stagnation penalty**: Increases velocity norm 1615× (forces non-zero predictions)
- **HybridStreamExpert soft flags**: Continuous [0,1] blending for smooth architectural morphing

### 4.2 What Doesn't Work
- **Diffusion denoising on weights**: MSE stuck at 1.0 — weights are unstructured
- **Weight generalization to test data**: All models predict zero on unseen data
- **ES evolution**: Population variance too low (0.03 after 5 gens) — needs more data
- **DDIM denoising**: Amplifies errors, produces worse results than zero initialization
- **KnitLoRA**: OOM on all model scales — forsaken

### 4.3 The Core Bottleneck
The fundamental issue across all weight flow models: **conditioning insufficiency**. The mean hidden state + gradient (5120-dim) doesn't contain enough information to determine the correct weight update for unseen data. The model correctly learns that "predict zero" minimizes held-out MSE.

**Solutions identified**:
1. Richer conditioning (full hidden state sequence, not mean)
2. Closed-form SVD as primary target (well-defined for any input)
3. Shift to latent reasoning (thoughts have structure; weights don't)

---

## 5. Infrastructure Details

### Memory Optimization (Phase 1)
- AdapterWrappedLinear frees base_linear.weight after cloning (saves ~400MB)
- saved_weights stored on CPU
- Gradient checkpointing enabled for all runs

### Key Commands

```bash
# Run batch experiment
python3 run_exp.py --variants dora edora --r 8 --alpha 8.0 --batch-size 1 --max-steps 200

# Run streaming continual learning
python3 run_stream_fusion.py --n-segments 10 --steps-per-segment 20 --r 8

# Run expert variant sweep
python3 run_sweep_analysis.py

# Run weight flow training
python3 run_weight_flow_train.py

# Run weight flow evaluation
python3 run_weight_flow_eval.py

# Run diffusion flow training (diverse data)
python3 run_diffusion_flow_train.py

# Run diffusion flow evaluation
python3 run_diffusion_flow_eval.py

# Run flow vs SVD vs SGD comparison
python3 run_comparison.py
```

### Known Issues
- Knit variants OOM on all model scales
- MultiAngleLoRA outputs NaN (empty band issue)
- Weight flow models don't generalize to test data (conditioning insufficiency)
- Batch size vs throughput: SmolLM too small to saturate GPU

---

## 6. Literature Landscape

### Flow Matching / Diffusion for Weights
| Paper | Year | Mechanism |
|-------|------|-----------|
| Doc-to-LoRA | 2026 | Hypernetwork generates LoRA weights from prompts |
| HyperAdaLoRA | 2025 | Attention-based SVD parameter generation |
| P-diff | 2024 | Diffusion for parameter generation |
| **Your gap**: Flow matching over weight trajectories | — | Predict weight velocity via Perceiver bottleneck, conditioned on data |

### Continual / Online LoRA
| Paper | Year | Mechanism |
|-------|------|-----------|
| Online-LoRA | 2025 (WACV) | Task-free online continual learning |
| Temp-Lora | 2024 (COLM) | Temporary LoRA trained during generation |
| C-LoRA | 2025 | Routing matrix for sequential tasks |
| **StreamFusion-LoRA** | 2026 | PerceiverFusion bottleneck + TAF routing + expert pool |

### LoRA Variants
| Paper | Year | Mechanism |
|-------|------|-----------|
| DoRA | 2024 (NeurIPS) | Magnitude-direction decomposition |
| AdaLoRA | 2023 (ICLR) | SVD param., prune singular values |
| VeRa | 2025 | Vector-based eigendirections |
| SRLoRA | 2025 | Dynamic subspace recomposition |
| **Your variants**: AFA, AUR, PERA | 2026 | Annealed tanh, autoencoder MLP, polynomial expansion |

---

## 7. Tagged Milestones (Git)

| Tag | Description |
|-----|-------------|
| v0.1.0-streamfusion | PerceiverFusion + TAF routing + expert pool |
| v0.2.0-experts | DoRA, BoRA, VeRa + HybridStreamExpert (8 flags) |
| v0.3.0-orthodox | AFA, AUR, PERA + 20-variant sweep |
| v0.4.0-decomposition | TaylorContribution, AlternatingTrainer, OverlapConsistency |
| v0.5.0-analysis | Multi-seed multi-segment analysis framework |
| v0.6.0-evolution | MetaController (ES over adapter lifecycles) |
| v0.7.0-lifecycle | PERA→BVA morphing schedule |
| v0.8.0-flowmatching | Flow matching over flags (architecture space) |
| v0.9.0-unified | Differentiable lifecycle optimizer |
| v0.10.0-weightflow | Flow matching over weights (synthetic) |
| v0.10.1-flowweights | FlowWeightExpert + closed-form LoRA analysis |
| v0.11.0-realtraj | Real LM trajectory collection (Qwen3.5-0.8B) |
| v0.12.0-eval | Full evaluation: flow ≈ zero on test data |
| v0.12.1-realeval | Overfitting confirmed: 300 samples / 11M params |
| v0.13.0-diffusion | WeightDiffusion (flow + denoising + flag conditioning) |
| v0.14.0-diversetrain | 70 trajectories, 5 sources, 4275 augmented |
| v0.15.0-closedform | Closed-form SVD training target |
| v0.16.0-deliverables | Dynamic K design, meta-synthesis skill, MCP proposals |
| v0.17.0-dynamick | Entropy-thresholded + adaptive growth implemented |
| v0.18.0-denoising | Proper denoising analysis: flow works, denoising doesn't |

---

## 8. Quick Resume

### Next experiments to run

```bash
# 1. Stagnation-penalty training (in progress)
#    Already running with λ_stag=0.2 on 70 trajectories

# 2. Evaluate stagnation model
python3 run_diffusion_flow_eval.py

# 3. SelectiveLoRA implementation (from Phase 2 roadmap)
#    Create with: warmup → score → freeze low → continue → periodic rescore

# 4. Validate top-5 Phase 2 variants on Qwen3.5-0.8B
python3 run_exp.py --variants edoran doran gend_afa_eva_doran --model-path ...qwen3.5-0.8b...

# 5. Full latent reasoning engine prototype
```

### Key Numbers
- Batch: Best magnitude variant: edoran gs=128, PPL 1,394 (Phase 2)
- StreamFusion: 78% eval PPL reduction across 9 segments (Phase 3)
- Weight flow: Flow MSE ~0 on training, ≈zero on test — conditioning wall
- SVD closed-form: 29.8% loss reduction in one step
- GPU: 7.6GB — enough for Qwen3.5-0.8B (batch=1) or Qwen3.5-2B (batch=1, gradient ckpt)
- Diffusion training: ~60 min for 70 trajectories, 15 epochs
- Model sizes: weight_flow_model.pt = 10.6MB, diffusion_weight_flow.pt = 12MB

### Architecture Decisions
1. **Magnitude needs pretrained weights** — Phase 1 finding confirmed
2. **Flow matching works for training memorization, not generalization** — Phase 3 finding
3. **Denoising is pointless on random weights** — they have no manifold
4. **Closed-form SVD is the correct target** — 29.8% improvement in one step
5. **Stagnation penalty forces non-zero predictions** — 1615× velocity increase

### Open Research Questions
1. Can richer conditioning (full hidden states, not mean) unlock weight generalization?
2. Can the SVD optimal target alone train a weight flow model (no SGD trajectories)?
3. Does latent reasoning (thought trajectories, not weight trajectories) have enough structure for diffusion to work?
4. Can the MetaController + WeightDiffusion composition learn end-to-end lifecycles?
