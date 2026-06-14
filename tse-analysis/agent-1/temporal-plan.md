# Phase 9: Resource-Budgeted Temporal Phasing

**Subject**: RankAdaptation — Velocity-based latent steering
**Date**: 2026-06-14

---

## Available Resources

| Resource | Capacity | Constraint |
|----------|----------|------------|
| GPU | 1× GPU (model-dependent VRAM) | Shared; OOM risk with prior processes |
| Internal SSD | ~15GB free (project data moved to HDD) | Limit: 71GB total |
| External HDD | ~1TB free | Slow I/O (but acceptable for storage) |
| CPU RAM | ~16-32GB | OOM at >5000 trajectories in float32 |
| Models | 7B, 1.5B, 360M, 135M, 0.5B, 0.8B | 7B takes ~15GB SSD for storage |
| Time | Continuous (dedicated machine) | Limited by patience/interest |
| Existing checkpoints | 7 TTs + trajectories | Immediate availability |

---

## Phase A: Diagnostic Sprint (0-2 hours)

**Budget**: ≤2 hours, ≤10% of total resources
**Goal**: Resolve the 3 critical unresolved disparities (D4, D5, D10) and run the top-3 experiments

### Experiment A1: Contrastive Evaluation (D10 resolution)

| Parameter | Value |
|-----------|-------|
| Script | `run_contrastive_eval.py` (already exists) |
| Model | Qwen2.5-7B-Instruct |
| TTs | best_tt_correct.pt, best_tt_incorrect.pt |
| α | 0.1 |
| Layers | All 32 layers (sweep) |
| n_test | 50 problems |
| Expected time | ~30-45 min |
| GPU memory | ~15GB |
| **Success criterion** | At least 1 layer shows α > baseline (trim-tab exists) |
| **Failure criterion** | All layers ≤ baseline (contrastive doesn't produce trim tabs) |

### Experiment A2: Anti-Steering at L9 (H0-2, CF-1)

| Parameter | Value |
|-----------|-------|
| Script | Modify `run_math15_sweep.py` to accept negative α |
| Model | Qwen2.5-7B-Instruct |
| α | −0.1 |
| Layers | L9 (primary death layer) + L8 control |
| n_test | 100 problems |
| Expected time | ~30 min |
| GPU memory | ~15GB |
| **Success criterion** | −α at L9 improves accuracy (direction misalignment) |
| **Failure criterion** | −α at L9 also degrades (off-manifold perturbation) |

### Experiment A3: α Sweep on L8 (CF-2)

| Parameter | Value |
|-----------|-------|
| Script | Modify sweep to vary α across range |
| Model | Qwen2.5-7B-Instruct |
| α range | [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0] |
| Layers | L8 (best trim-tab) + L9 (death control) |
| n_test | 50 problems per α (total: 8×50 = 400 problems) |
| Expected time | ~1 hour |
| GPU memory | ~15GB |
| **Success criterion** | Monotonic improvement (no optimal α found) or U-shaped (optimal α exists) |
| **Failure criterion** | α=0.1 is the unique optimum (current default is already optimal) |

### Phase A Go/No-Go Criteria

| Experiment | Go Signal | No-Go Signal | Consequence |
|-----------|-----------|--------------|-------------|
| A1 | Any trim-tab found | No trim-tabs | Contrastive approach invalid; revert to standard TT refinement |
| A2 | −α improves L9 | −α also degrades | Death layers are off-manifold sensitive; need protective measures |
| A3 | U-shaped α curve | Monotonic degradation | α is not a meaningful dial; other mechanisms needed |

**Overall Phase A Go**: ≥2 of 3 experiments produce positive signals. Otherwise → **reassess** rather than proceed to Phase B.

---

## Phase B: Short-Term Targeted Experiments (2-12 hours)

**Budget**: ≤10 hours, ≤30% of total resources
**Prerequisite**: Phase A Go

### Experiment B1: Random Direction Baseline (M11-1)

| Parameter | Value |
|-----------|-------|
| Script | Modify steering script to accept random vector input |
| Model | Qwen2.5-7B-Instruct |
| Layers | L8, L9 |
| α | Same magnitude as TT steering |
| Noise types | (1) isotropic Gaussian, (2) uniform on sphere, (3) smoothed noise |
| n_test | 100 per condition |
| **Expected finding** | If random = TT → steering is just perturbation; if random < TT → direction matters |

### Experiment B2: Multi-Layer L2+L8 (RECOMB-FP1)

| Parameter | Value |
|-----------|-------|
| Script | Extend per-layer to per-pair sweep |
| Model | Qwen2.5-7B-Instruct |
| Layer pairs | (L2, L8), (L8, L10), (L2, L10), (L8, L9) |
| α | 0.1 per layer |
| n_test | 100 per pair |
| **Expected finding** | Test whether trim-tab effects are additive, synergistic, or competitive |

### Experiment B3: High α on Sub-Threshold (RECOMB-FP2, D4 resolution)

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-Math-1.5B (38% baseline) |
| α range | [0.05, 0.1, 0.2, 0.5, 1.0] |
| Layers | All layers (or top-10 candidates) |
| n_test | 100 per condition |
| **Expected finding** | If threshold is real: all improvements near zero. If α-dependent: some positive at high α. |

### Experiment B4: Early-Only Steering (M4-1)

| Parameter | Value |
|-----------|-------|
| Script | Modify steering to apply only during first N tokens |
| Model | Qwen2.5-7B-Instruct |
| N | [10%, 25%, 50%, 75%] of generation |
| Layer | L8 |
| α | 0.1 |
| **Expected finding** | If reasoning direction is set early → early steering dominates. If correction needed throughout → later steering matters. |

### Phase B Success Criteria

| Experiment | Success | Partial | Failure |
|-----------|---------|---------|---------|
| B1 | TT > random at L8 | TT > random but same pattern | TT = random |
| B2 | Pair > single (synergy) | Pair = single (additive) | Pair < single (interference) |
| B3 | Some positive at high α | All α negative but not catastrophic | All α catastrophic |
| B4 | Clear early/late difference | Weak difference | No difference |

---

## Phase C: Medium-Term Architectural Changes (1-3 days)

**Budget**: ≤2 days, ≤50% of remaining resources
**Prerequisite**: Phase B success in ≥3 of 4 experiments

### C1: Contrastive TT with Improved Architecture

| Component | Detail |
|-----------|--------|
| **Goal** | Build single Siamese contrastive TT instead of two separate TTs |
| **Approach** | Train a single model with triplet loss: anchor = current state, positive = correct next state, negative = incorrect next state |
| **Expected improvement** | Shared representation may capture richer contrastive structure |
| **Cost** | ~4 hours training, ~$0 (local GPU) |
| **Risk** | May not outperform two-TT baseline |

### C2: Per-Layer α Optimization via Bayesian Search

| Component | Detail |
|-----------|--------|
| **Goal** | Learn optimal α for each layer simultaneously |
| **Approach** | Bayesian optimization over 32-dimensional α space |
| **Expected improvement** | Discover non-obvious combinations (e.g., small α on trim-tabs, zero on death layers) |
| **Cost** | ~500 evaluations × 1 min = ~8 hours |
| **Risk** | 32-dim space is sparse; optimization may not converge |

### C3: Cross-Dataset Evaluation

| Component | Detail |
|-----------|--------|
| **Goal** | Test steering on ARC (reasoning), BBH (big-bench), MMLU (knowledge) |
| **Approach** | Collect trajectories, train TT, run per-layer sweep on each dataset |
| **Expected finding** | Trim-tab pattern may be task-specific or task-general |
| **Cost** | ~2 hours per dataset × 3 = ~6 hours |
| **Risk** | Trajectory structure may differ significantly per task type |

### C4: Inoculation Training (EM-5)

| Component | Detail |
|-----------|--------|
| **Goal** | Make death layers steering-amenable by adding noise during training |
| **Approach** | Apply small perturbations to death-layer activations during forward pass → model learns to be robust |
| **Expected finding** | Death layers become neutral or weakly positive for steering |
| **Cost** | Requires fine-tuning (hours); risk of catastrophic forgetting |
| **Risk** | HIGH — fine-tuning may degrade baseline accuracy |

---

## Phase D: Long-Term Research Program (1-2 weeks)

**Budget**: Remaining resources
**Prerequisite**: Phase C success in ≥2 of 4 experiments

### D1: Self-Correcting Steering Loop (EM-1)

| Component | Detail |
|-----------|--------|
| **Goal** | Build closed-loop steering controller |
| **Approach** | Real-time health monitor (perplexity, attention entropy, token divergence) + adaptive α |
| **Components** | (1) Health signal predictor, (2) α(t) = f(health(t)), (3) online learning |
| **Expected impact** | Transformative: steering becomes adaptive rather than static |
| **Cost** | 1-2 weeks, significant infra development |

### D2: Universal Velocity Manifold (EM-2) Verification

| Component | Detail |
|-----------|--------|
| **Goal** | Confirm velocity dynamics are universal across LLM families |
| **Approach** | Train TT on 5+ diverse model families, test cross-model transfer |
| **Models** | LLaMA-3, Mistral-7B, Gemma-7B, Phi-3, GPT2-XL |
| **Expected impact** | Foundational theoretical contribution; universal steerability |
| **Cost** | 1-2 weeks, ~50GB+ trajectory storage |

### D3: RL-Based α Optimization (Open Question #7)

| Component | Detail |
|-----------|--------|
| **Goal** | Learn α per (layer, token) via reinforcement learning |
| **Approach** | Policy gradient: action = (layer, α), reward = accuracy improvement |
| **Expected impact** | Maximum possible steering improvement |
| **Cost** | 1-2 weeks, complex RL infrastructure |
| **Risk** | Credit assignment over 200+ tokens × 32 layers is extremely sparse |

---

## Decision Tree

```
START
│
├──→ Phase A (Diagnostic): 0-2 hours
│   ├──→ A1: Contrastive eval (30-45min)
│   ├──→ A2: Anti-steering L9 (30min)  
│   └──→ A3: α sweep L8 (1 hour)
│   │
│   ├──→ ≥2/3 positive ──→ PHASE B GO
│   │                      │
│   │                      ├──→ B1: Random baseline (1 hour)
│   │                      ├──→ B2: Multi-layer pair (2 hours)
│   │                      ├──→ B3: High-α sub-threshold (2 hours)
│   │                      └──→ B4: Early-only (1 hour)
│   │                      │
│   │                      ├──→ ≥3/4 success ──→ PHASE C GO
│   │                      │                      │
│   │                      │                      ├──→ C1: Siamese contrastive TT (4 hours)
│   │                      │                      ├──→ C2: Bayesian α optimization (8 hours)
│   │                      │                      ├──→ C3: Cross-dataset eval (6 hours)
│   │                      │                      └──→ C4: Death layer inoculation (risky)
│   │                      │                      │
│   │                      │                      ├──→ ≥2/4 success ──→ PHASE D GO
│   │                      │                      │                      │
│   │                      │                      │                      ├──→ D1: Self-correcting loop (1-2 weeks)
│   │                      │                      │                      ├──→ D2: Universal manifold (1-2 weeks)
│   │                      │                      │                      └──→ D3: RL α optimization (1-2 weeks)
│   │                      │                      │
│   │                      │                      └──→ <2/4 success ──→ [Report negative results]
│   │                      │                                              │
│   │                      └──→ <3/4 success ──→ [Focus on contrastive improvement]
│   │                                              Refine contrastive direction
│   │
│   └──→ <2/3 positive ──→ [Reassess fundamental assumptions]
│                           ├──→ Reconsider steering surface (residual stream?)
│                           ├──→ Reconsider direction signal (not velocity?)
│                           └──→ Reconsider project viability
```

---

## Contingency Plans

| Unforeseen Event | Branch | Action |
|-----------------|--------|--------|
| GPU OOM during Phase A | Skip A3 (least critical) | Move to A1, A2 only |
| No trim-tabs in contrastive evaluation | Revert to standard TT | Use standard TT for all Phase B experiments as comparison |
| Anti-steering ALSO destroys L9 | Conclude death is perturbation-sensitive | Focus on off-manifold hypotheses; add synthetic data test |
| Novel result (e.g., anti-steering at L9 IMPROVES) | Pivot entire direction | Investigate direction reversal as primary mechanism |
| Cross-dataset shows opposite pattern | Revise generality claims | Document as task-specific phenomenon; focus on math domain |
