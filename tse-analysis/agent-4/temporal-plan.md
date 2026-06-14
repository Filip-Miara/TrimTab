# Phase 9: Resource-Budgeted Temporal Phasing

## Available Resources

| Resource | Details | Constraint |
|----------|---------|------------|
| **GPU** | Single NVIDIA GPU (likely RTX 3090/4090, 24GB VRAM) | 24GB limits batch size, model size ≤ 7B with quantization |
| **CPU RAM** | ~32GB (previously OOM at 10GB for 5000 trajectories) | Effective limit ~1000 trajectories in memory |
| **Storage (SSD)** | ~56GB free (71GB total, ~15GB used by 7B model + data) | Severe constraint — limits trajectory storage |
| **Storage (HDD)** | External 500GB+ HDD | Slower access (15× slower model loading) but sufficient |
| **Time** | Active project timeline: 5 sessions over 4 days | Assume ~8-12 more sessions available |
| **Models** | Qwen2.5-7B-Instruct (loaded), checkpoints for SmolLM2, Math-1.5B, Qwen2.5-0.5B | Limited to models already downloaded (7-8GB each, slow downloads) |

## Phase A — Diagnostic Foundation (Immediate, ≤2 hours)

**Cost**: <2 hours wall time, no new infrastructure, <500MB storage

### A1: Random Steering Control (H0-1 from Phase 8) — CRITICAL

Run per-layer sweep with random vectors matched to the TT's output norm. If random steering works, the entire paradigm is noise injection.
- **Script**: Modify `run_per_layer_sweep.py` to accept a `--random` flag
- **Parameters**: Same as standard sweep (28 layers, 100 problems, α=0.1)
- **Time**: ~30 min
- **Success criterion**: Random steering produces ≤5pp improvement on best layer
- **Failure criterion**: Random steering produces ≥10pp improvement on any layer (paradigm challenged)

### A2: Negative α Inversion (H0-3 from Phase 8)

Run per-layer sweep with α = -0.1. Test if death layers become trim-tabs.
- **Script**: Modify `run_per_layer_sweep.py` to accept `--alpha=-0.1`
- **Parameters**: 28 layers, 100 problems, α=-0.1
- **Time**: ~30 min
- **Success criterion**: L9 pattern reverses (L9 accuracy increases)
- **Failure criterion**: L9 still harmful (death is intrinsic)

### A3: α Sweep on L8

Find the optimal α for the best trim-tab layer.
- **Parameters**: α ∈ {0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0} on L8 only, 100 problems
- **Time**: ~30 min
- **Success criterion**: Identify the α that maximizes L8 accuracy
- **Go/No-Go**: If L8 accuracy peaks at α=0.1, the current setting is optimal. If it peaks elsewhere, all prior measurements are contingent.

### A4: Linear TT Baseline

Train a linear regression (ridge, no attention) on h[l]→v[l]. Compare R².
- **Implementation**: Use sklearn.linear_model.Ridge or equivalent
- **Data**: Same training data as TT
- **Time**: ~15 min
- **Success criterion**: Linear R² < 0.8 (nonlinear structure exists)
- **Failure criterion**: Linear R² > 0.8 (velocity is linear — TT capacity wasted)

### Phase A Go/No-Go Decision

- **GO to Phase B**: If random steering (A1) fails AND at least one of L8, L9, or α behavior is informative
- **NO-GO**: If random steering works (paradigm invalid) OR all results are flat/uninformative

## Phase B — Short-Term Targeted (≤1 day, ≤30% budget)

**Cost**: <1 day wall time, existing infrastructure, <5GB storage

### B1: Contrastive Evaluation (PENDING)

Evaluate the already-trained contrastive TTs on Qwen2.5-7B.
- **Script**: `run_contrastive_eval.py` (already written)
- **Parameters**: 28 layers, 50-100 problems, α=0.1, both v_c and v_c - v_i
- **Time**: ~2 hours
- **Success criterion**: Contrastive steering shows +5pp improvement over standard steering
- **Failure criterion**: Contrastive ≤ standard steering

### B2: Multi-Layer Steering

Steer multiple trim-tab layers simultaneously (L2 + L8 + L10) with α=0.1 each.
- **Implementation**: Modify per_layer_sweep to accept a list of layers
- **Parameters**: Combinations: {L8}, {L2, L8}, {L8, L10}, {L2, L8, L10}
- **Time**: ~2 hours
- **Success criterion**: Multi-layer is at least additive (≥+37pp = 17+20)
- **Failure criterion**: Multi-layer < best single layer (interference)

### B3: K/V Split Steering

Steer keys and values with DIFFERENT α (α_k = 0.2, α_v = 0.0; α_k = 0.0, α_v = 0.2; etc.)
- **Implementation**: Split `steer_kv_cache` into key-only and value-only variants
- **Parameters**: L8, 100 problems
- **Time**: ~1 hour
- **Success criterion**: One of K-only or V-only steering matches or exceeds both
- **Failure criterion**: Neither works — both must be steered together

### B4: Per-Layer α Tuning (CMA-ES)

Optimize 28 α values using CMA-ES with accuracy as reward.
- **Implementation**: Use library (e.g., `cma` package) with 28-dim search space
- **Parameters**: Population size = 6, generations = 20 (120 evaluations × 100 problems)
- **Compute**: ~8 hours (but can be parallelized)
- **Success criterion**: CMA-ES finds α vector that outperforms uniform α=0.1
- **Failure criterion**: CMA-ES converges to uniform α or worse

### Phase B Go/No-Go Decision

- **GO to Phase C**: If B1 or B4 shows significant improvement over baseline
- **NO-GO**: If all targeted interventions fail, revisit fundamentals

## Phase C — Medium-Term Architectural (≤1 week, ≤50% budget)

**Cost**: <1 week, new infrastructure development, <20GB storage

### C1: Steering-Regime Classifier

Train a classifier on (h[l], layer_id, context) to predict if steering will help.
- **Data**: Generate labeled data from existing per-layer sweep results
- **Architecture**: Lightweight classifier (2-layer MLP, d_hidden=256)
- **Compute**: ~2 hours training, <1 hour evaluation
- **Success criterion**: Classifier F1 > 0.8 on held-out steering decisions
- **Integration**: Output gates the steering mechanism in real-time

### C2: Adaptive α = f(entropy)

Implement α as a function of prediction uncertainty (entropy of next-token distribution).
- **Implementation**: α(h, entropy) = α_base * sigmoid((threshold - entropy) / temperature)
- **Effect**: Steer less when model is uncertain (avoid harmful steering)
- **Compute**: ~1 day evaluation across conditions
- **Success criterion**: Adaptive α outperforms fixed α on accuracy

### C3: Cross-Model Injection

Apply Qwen2.5-7B's steering signal to Qwen2.5-0.5B (or another small model).
- **Implementation**: Use existing transfer infrastructure with 7B TT as source
- **Compute**: ~4 hours
- **Success criterion**: Small model shows improved accuracy under 7B steering (circumventing capability threshold)
- **Failure criterion**: Small model still unaffected (threshold is intrinsic, not signal-quality)

## Phase D — Long-Term Fundamental (≥1 week, remaining budget)

**Cost**: >1 week, could require additional compute, <100GB storage

### D1: Multi-Head Contrastive Ensemble

Train N bootstrapped contrastive TT pairs, ensemble via averaging or voting.
- **N = 10** bootstrapped samples of trajectory data
- **Expected**: Variance reduction in steering signal, potentially improved accuracy
- **Risk**: Compute cost (10× TT training)

### D2: Per-Head Steering

Steer individual attention heads instead of entire layers.
- Requires identifying which heads are "reasoning heads" vs "output heads"
- Could reveal that death layers have mostly harmful heads and trim-tab layers have mostly helpful heads

### D3: Non-Math Task Generalization

Test on ARC (reasoning), BBH (challenging), MMLU (knowledge).
- Tests whether trim-tab/death pattern generalizes beyond math
- Could reveal task-specific patterns

### D4: RL-Based α Optimization

Treat α as actions in an RL framework with accuracy as delayed reward.
- Requires differentiable or approximate generation loop
- Could discover non-trivial α schedules (early tokens vs late tokens)

## Decision Tree

```
START
  |
  v
Phase A: Diagnostic Foundation (<2 hours)
  |
  ├── A1: Random steering control ──── FAIL (random works) → STOP paradigm invalid; write paper as cautionary tale
  |                                       |
  |                                       v
  |                                   REPORT: "Velocity steering is noise injection"
  |
  ├── A2: Negative α inversion ────── L9 improves → death layers are directional
  |                                       |
  |                                       v
  |                                   α flipping becomes standard technique
  |
  ├── A3: α sweep on L8 ──────────── Peak found → calibrate all subsequent experiments
  |
  ├── A4: Linear TT baseline ──────── Linear R² > 0.8 → simplify TT architecture
  |
  └── GO criteria met? ────── NO → STOP; revisit data collection or model choice
                                  |
                                  YES
                                  v
Phase B: Targeted Intervention (<1 day)
  |
  ├── B1: Contrastive eval ──────── Works (+5pp+) → normative steering confirmed
  |                                     |
  |                                     v
  |                                 Confirms EM-2 (cross-model injection potential)
  |
  ├── B2: Multi-layer combo ──────── Additive → multiple trim-tabs are independent
  |                                       |
  |                                       v
  |                                   Synergy → emergent multi-layer effect
  |
  ├── B3: K/V split ──────────────── K-only ≈ both → simplify steering mechanism
  |                                       |
  |                                       v
  |                                   V-only ≈ better → values drive attention
  |
  ├── B4: CMA-ES α ───────────────── Better α found → per-layer optimization pays off
  |
  └── GO criteria met? ────── NO → STOP; steering plateaus at +20pp
                                  |
                                  YES
                                  v
Phase C: Architectural (<1 week)
  |
  ├── C1: Steering classifier ────── F1 > 0.8 → automated safe steering
  ├── C2: Adaptive α ──────────────── Works → uncertainty-aware steering
  ├── C3: Cross-model injection ──── Works → capability threshold bypassed
  |
  └── GO criteria met? ────── NO → STOP; infrastructure insufficient
                                  |
                                  YES
                                  v
Phase D: Fundamental (long-term)
  |
  ├── D1: Multi-head ensemble
  ├── D2: Per-head steering
  ├── D3: Task generalization
  └── D4: RL α optimization
         |
         v
    FINAL SYNTHESIS
```

## Budget Summary

| Phase | Cost (GPU-hours) | Storage | Wall Time | Decisions |
|-------|-----------------|---------|-----------|-----------|
| A | ~2 | <1GB | <2 hours | Go/No-Go on paradigm |
| B | ~15 | <5GB | <1 day | Go/No-Go on specific mechanisms |
| C | ~40 | <20GB | <1 week | Go/No-Go on advanced features |
| D | ~100+ | <100GB | >1 week | Open-ended research |

**Total estimated budget**: ~157 GPU-hours + 126GB storage (funding permitting)
