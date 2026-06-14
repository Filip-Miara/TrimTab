# Phase 9: Resource-Budgeted Temporal Phasing

---

## Available Resources

| Resource | Quantity | Constraints |
|----------|----------|-------------|
| GPU | 1× NVIDIA (likely RTX 3090/4090, 24GB VRAM) | Shared with other processes |
| Internal SSD | ~60GB free (after model + active data) | 71GB total, models ~15GB |
| External HDD | ~200GB+ free | 50MB/s read, 2-3 it/s for model loading |
| CPU RAM | Unknown (estimated 32-64GB) | OOM with >5000 trajectories |
| Time per experiment | ~2-4 hours for full per-layer sweep | Must fit in session window |
| Models available | Qwen2.5-7B (SSD), Math-1.5B (SSD), SmolLM2, others on HDD | Loading time: 10s (SSD) vs 2.5 min (HDD) |
| Trained TTs | 7 checkpoints (standard + contrastive pairs) | In current directory |
| Codebase | ~80 Python scripts, modular pipeline | Mature; new experiments = new script |

**Budget inference**: `auto` → approximately 40 GPU-hours remaining (estimated until project conclusion). Budget is tight — each full per-layer sweep costs ~4 GPU-hours.

---

## Phase A — IMMEDIATE: Diagnostic Experiments

**Cost**: ≤2 hours wall time, ≤4 GPU-hours

### A1: Contrastive Evaluation on Qwen2.5-7B (URGENT — already set up)

- **Script**: `run_contrastive_eval.py --n-test 100 --alpha 0.1 --layers 0 1 2 3 4 5 6 7 8 9`
- **GPU cost**: ~2 hours (100 problems × 10 layers with steering)
- **Success criterion**: At least one layer shows positive Δ (contrastive > standard TT at same layer)
- **Failure criterion**: ALL layers show contrastive ≤ baseline (confirms off-manifold hypothesis D8)
- **Go/No-Go decision**: If PASS → proceed to A2. If FAIL → pivot to Non-Contrastive Path (A2-alt).
- **Risk**: LOW — already set up, just needs to be run.

### A2: Alpha Asymmetry Sweep (if A1 passes)

- **Script**: Modify `run_autonomous_sweep.py` to test both positive AND negative α for L8 and L9
- **Parameters**: α ∈ {−0.5, −0.3, −0.1, −0.05, −0.01, 0, 0.01, 0.05, 0.1, 0.3, 0.5} on L8 and L9 separately
- **GPU cost**: ~30 min (2 layers × 11 α values × 50 problems)
- **Success criterion**: Negative α at L9 improves accuracy (L9 is "invertible")
- **Failure criterion**: All negative α values harm accuracy
- **Go/No-Go decision**: If negative α at L9 works → L9 is a "phase-inverted" trim tab, not a death layer.

### A2-alt: Non-Contrastive Enhancement Path (if A1 fails)

- **Objective**: Improve standard TT steering since contrastive direction doesn't exist
- **Actions**: 
  - Per-layer α vector on ALL layers (not just top-3) — cost: 1 GPU-hour
  - Test TT with shuffled training labels (null model) — cost: 30 min TT training + 1 hour eval
  - Negative α on L9 only — cost: 30 min
- **GPU cost**: ~3 hours
- **Success criterion**: Per-layer α vector outperforms single-layer L8 steering
- **Failure criterion**: No configuration improves over L8 α=0.1

### A3: L8 Ablation (Keystone Test)

- **Script**: Modify steering to set α=0 at L8 only (steer all other layers normally or don't steer at all)
- **Parameters**: 
  - Test 1: α=0 at L8, α=0.1 elsewhere → does removing L8 steering collapse accuracy?
  - Test 2: α=0 at L8, α=0 everywhere → baseline with specific L8 zeroing
- **GPU cost**: ~20 min (2 evaluations × 100 problems)
- **Success criterion**: L8 ablation significantly reduces accuracy below baseline (confirms keystone hypothesis)
- **Failure criterion**: L8 ablation has no effect → L8 is not special; it just happens to have a positive effect

---

## Phase B — SHORT-TERM: Targeted Optimization

**Cost**: ≤1 day wall time, ≤12 GPU-hours
**Prerequisite**: At least one Phase A experiment produces a go decision.

### B1: TT Null Model Comparison

- **Objective**: Determine whether TT architecture is necessary
- **Experiments**:
  1. Zero TT (v=0): Confirm steering effect actually requires non-zero velocities
  2. Constant TT (v=constant per layer): Test if layer-specific constants match TT performance
  3. Linear TT (v = Wh + b): Compare against full transformer TT
  4. Layer-specific linear TT: One small linear model per layer
- **GPU cost**: ~6 hours (training 4 null models + evaluating each on per-layer sweep)
- **Success criterion**: Transformer TT significantly outperforms all null models at L8
- **Failure criterion**: Constant TT matches transformer TT performance → TT's complexity is unnecessary
- **Go/No-Go decision**: If TT is unnecessary → the discovery is about layer-specific steering, not velocity prediction.

### B2: Frequency-Domain Analysis (EM-2)

- **Objective**: Determine if L8's effectiveness is related to specific frequency components of velocity
- **Experiments**:
  1. PCA on TT's velocity predictions across training data
  2. Steer using only top-k PCA components (k = 1, 2, 5, 10, 50, 100)
  3. Compute per-layer frequency signature: which PCA components dominate at each layer?
  4. Compare L8 and L9 frequency signatures
- **GPU cost**: ~2 hours (PCA analysis is fast; 6 steering evaluations × 50 problems)
- **Success criterion**: L8 and L9 have qualitatively different frequency signatures
- **Failure criterion**: All layers have similar frequency distributions

### B3: Attention Pattern Analysis

- **Objective**: Determine if steering works by shifting attention distributions
- **Experiments**:
  1. Capture attention distributions at L8 and L9 during generation (with and without steering)
  2. Compute attention entropy, attention-to-special-tokens (e.g., separator tokens)
  3. Correlate attention changes with accuracy improvement
- **GPU cost**: ~2 hours (requires attention capture, slower generation)
- **Success criterion**: Steering at L8 consistently shifts attention toward answer-relevant tokens
- **Failure criterion**: Attention distributions are unchanged by steering

---

## Phase C — MEDIUM-TERM: Architectural Changes

**Cost**: ≤1 week wall time, ≤24 GPU-hours
**Prerequisite**: Phase B confirms TT is useful AND contrastive direction exists.

### C1: RL-Based Per-Token Alpha

- **Objective**: Learn optimal α per (layer, token) via reinforcement learning
- **Design**: Small Q-network (2-layer MLP, 256 hidden) taking (layer, token_position, attention_entropy, TT_uncertainty) → α prediction
- **Training**: REINFORCE with accuracy as reward; ~500 problems for training
- **GPU cost**: ~8 hours (RL training is sample-inefficient)
- **Success criterion**: RL-based α outperforms best fixed α at L8
- **Failure criterion**: RL fails to learn (α collapses to constant) → fixed α is sufficient

### C2: Dual-Surface Steering (EM-3)

- **Objective**: Simultaneously steer KV-cache and weight-flow
- **Design**: Combine TT (for KV-cache) with learned weight-flow expert (for MLP weights at same layer)
- **Implementation**: Integrate src/adapters/flow_weight_expert.py with kv_cache_steering.py
- **GPU cost**: ~12 hours (training weight-flow expert + joint evaluation)
- **Success criterion**: Combined steering outperforms either surface alone
- **Failure criterion**: Combined steering is no better than KV-cache alone

### C3: Multi-Head Contrastive Ensemble

- **Objective**: Improve contrastive signal via bootstrapping
- **Design**: Create 50 random correct/incorrect splits of training data; train 50 contrastive TT pairs; ensemble via average of v_c − v_i
- **GPU cost**: ~8 hours (50 × 10 min training + evaluation)
- **Success criterion**: Ensemble outperforms single contrastive pair
- **Failure criterion**: Ensemble is equivalent to single pair → variance is not the issue

---

## Phase D — LONG-TERM: Fundamental Research

**Cost**: ≥1 week, unlimited GPU-hours (as available)
**Prerequisite**: Phase C confirms steering is robust and mechanistically understood.

### D1: Steering as a General Capability Amplifier

- **Experiments**: Apply steering to non-math tasks (ARC, BBH, MMLU, code generation)
- **Cost**: HIGH — each dataset requires separate evaluation pipeline
- **Success criterion**: Trim-tab pattern (specific layer improves accuracy) holds across tasks

### D2: Automated Trim-Tab Discovery

- **Objective**: Eliminate the expensive per-layer sweep by predicting trim-tab layers from model architecture
- **Design**: Train a meta-predictor that takes model architecture parameters (n_layers, d_model, attention_type, baseline_accuracy) → predicts trim-tab layer indices
- **Cost**: HIGH — requires data from many more models (10+)

### D3: Self-Adaptive Steering (Online System ID)

- **Objective**: TT adapts during generation based on which steering directions produce correct answers
- **Design**: Online learning loop: TT predicts → steering applied → observe outcome → update TT
- **Cost**: VERY HIGH — requires real-time training infrastructure

---

## Decision Tree

```
START
  │
  ├── Phase A1: Contrastive Eval
  │   ├── SUCCESS (≥1 layer positive Δ)
  │   │   └── Phase A2: Asymmetric α sweep
  │   │       ├── NEGATIVE α at L9 works
  │   │       │   └── Phase B1: TT null models
  │   │       └── Negative α at L9 fails
  │   │           └── Phase B1: TT null models
  │   └── FAILURE (all layers ≤ baseline)
  │       └── Phase A2-alt: Non-contrastive path
  │           ├── Per-layer α vector works
  │           │   └── Phase B2: Frequency analysis
  │           └── Nothing works beyond L8
  │               └── Phase B1: TT null models
  │                   ├── TT is necessary
  │                   │   └── Phase C1: RL α
  │                   └── TT is unnecessary (constant works)
  │                       └── Publish "layer-specific steering" finding
  │
  └── All paths converge to publishing decision at Week 2

Phase Budget Allocation:
  A: 4 GPU-hours (10% budget) — MANDATORY
  B: 12 GPU-hours (30% budget) — CONDITIONAL on Phase A
  C: 24 GPU-hours (60% budget) — CONDITIONAL on Phase B
  D: Unbounded — only if C succeeds
```

---

## Contingency Plan

**If contrastive evaluation FAILS** (most likely single point of failure):
- Pivot to: "Why contrastive steering doesn't work" as a research question
- Run TT ablation experiments to understand the contrastive failure mechanism
- Publish negative result: "Contrastive latent steering fails due to non-separable hidden manifolds in capable LMs" — this is a publication-worthy finding even if negative

**If L8 pattern doesn't replicate on new models**:
- Investigate what makes L8 special by training sparse probes on L8 hidden states
- Compare L8 representations across models using CKA
- Determine if there's an "L8-analogous" layer in other models based on representation similarity

**If the project is resource-constrained (GPU needed for other work)**:
- Run Phase A1 only (contrastive eval) — this is the highest-information experiment per GPU-hour
- Results from A1 determine whether the project should continue or wrap up
