# Phase 9: Resource-Budgeted Temporal Phasing

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14
**Budget**: auto (inferred from project context — single researcher, single GPU, ~71GB SSD)

---

## Available Resources

| Resource | Specification | Constraint |
|----------|--------------|------------|
| **GPU** | Single NVIDIA GPU (unspecified VRAM, likely 8-24GB) | Must fit model + TT + trajectories |
| **Compute** | Consumer-grade workstation | No cluster access |
| **Storage (SSD)** | ~71GB (post-model: ~41GB free) | 15GB for Qwen2.5-7B, ~10.5GB for trajectories |
| **Storage (HDD)** | External 1TB+ | 15× slower access, good for archives |
| **Models on SSD** | Qwen2.5-7B-Instruct (15GB) | Active experiments |
| **Models on HDD** | Qwen3.5-2B, Qwen2.5-0.5B, Math-1.5B | Archived but accessible |
| **Trajectories on SSD** | ~10.5GB (25 files) | 26% of free SSD |
| **Trajectories on HDD** | ~54GB (130 files across 3 datasets) | Full archive |
| **TT Checkpoints** | 5 total (see PROJECT_DEBRIEF): 1×SmolLM2, 3×Qwen2.5-7B, 2×Math-1.5B | ~1GB total |
| **Time per session** | ~2-4 hours (typical) | Longer experiments need checkpoint resume |
| **Bandwidth** | 7-8GB models take 15-30 min to download | Use background processes |
| **Expertise** | Single researcher, ML engineering background | No dedicated infrastructure team |

---

## Phase A: Immediate — Diagnostic (≤2 hours)

### A1: Random Vector Control Experiment

**Goal**: Resolve whether the L8 steering effect is due to TT predictions or random perturbation.

**Cost**: 1 hour wall-time (model loading + 2 evals × 100 problems)

**Experiments**:
- Generate 100 random vectors W ∈ R^{3584} with ||W|| ≈ ||v_pred|| (matched norm distribution)
- Apply L8 steering with α=0.1 using W instead of v_pred
- Evaluate on GSM8K (same 100 problems as TT condition)

**Success Criterion**: TT_prediction accuracy > random_vector_accuracy by >5pp (statistically significant)

**Failure Criterion**: TT_accuracy - random_accuracy ≤ 5pp → steering effect is NOT specific to velocity predictions

**Go/No-Go Decision**:
- If TT > random: Continue to Phase B (TT captures meaningful structure)
- If TT ≈ random: **[STOP]** — pivot investigation to "why L8 is sensitive to any perturbation" (mechanistic question, not steering question)

### A2: Negative L9 Steering

**Goal**: Test whether L9's death-layer effect reverses with negative α.

**Cost**: 45 minutes

**Experiments**: Apply L9 steering with α ∈ {-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3} on 100 problems

**Success Criterion**: Any negative α produces accuracy > baseline (45%)

**Failure Criterion**: All negative α produce accuracy ≤ baseline + measurement noise (±5pp)

### A3: Contrastive TT Evaluation (Existing Checkpoints)

**Goal**: Evaluate the already-trained contrastive TTs on Qwen2.5-7B.

**Cost**: 2 hours (sweep over layers 0-15 with α=0.1 per layer)

**Experiments**:
- Layer sweep with v_correct TT only
- Layer sweep with v_incorrect TT only
- Layer sweep with contrastive signal (v_correct − v_incorrect)

**Success Criterion**: At least one layer shows >baseline improvement with contrastive signal

**Failure Criterion**: All layers ≤ baseline with contrastive

---

## Phase B: Short-Term — Targeted (≤1 day, requires Phase A success)

### B1: Signed Per-Layer Sweep

**Prerequisite**: A2 completed (L9 negative result)
**Cost**: 4 hours compute, 1 hour analysis

**Experiments**: For all 28 layers of Qwen2.5-7B, test α ∈ {-0.2, -0.1, 0, 0.1, 0.2} on 50 problems each

**Success Criterion**: Discover 3+ layers where optimal α is negative (inverse trim-tabs)
**Failure Criterion**: All layers optimal at α ≥ 0 (negative α never helps)

**Budget Allocation**: 
- Compute: ~28 layers × 5 α × 50 problems × 200 tokens = 1.4M tokens ≈ 1 hour GPU
- Rest is analysis overhead

### B2: Dual-Mode Steering (Standard + Contrastive)

**Prerequisite**: A3 completed (contrastive TT evaluated individually)
**Cost**: 3 hours

**Experiments**: Combine standard TT with contrastive signal: v_combined = v_standard + β·(v_correct − v_incorrect) with β ∈ {0, 0.2, 0.5, 1.0, 2.0} on L8 and L2

**Success Criterion**: Some β > 0 improves accuracy over standard TT alone
**Failure Criterion**: Best accuracy occurs at β=0 (contrastive doesn't help)

### B3: Over-Steering Small Models

**Prerequisite**: Phase A complete (confidence in steering methodology)
**Cost**: 2 hours

**Experiments**: Sweep α ∈ {0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0} on SmolLM2 and Qwen2.5-0.5B at layer 8 (or equivalent)

**Success Criterion**: Any α > baseline for either model
**Failure Criterion**: All α ≤ baseline for both models (strengthens capability threshold confirmation)

### B4: Multi-Task Validation

**Prerequisite**: Phase A complete
**Cost**: 3 hours (download datasets + evaluation)

**Experiments**: Run L8-steered Qwen2.5-7B (α=0.1) on ARC (Easy), BBH (subset), and GSM8K (full 1319 problems)

**Success Criterion**: Steering improves accuracy on ≥2 of 3 tasks
**Failure Criterion**: Steering improves GSM8K only (task-specific effect)

---

## Phase C: Medium-Term — Architectural (≤1 week, requires Phase B success)

### C1: Asymmetric Multi-Layer Steering

**Prerequisite**: B1 completed (signed α values known per layer)
**Cost**: 1 day compute + analysis

**Experiments**: Apply optimal α to ALL layers simultaneously (positive on trim-tabs, negative on inverse trim-tabs, zero on neutral). Search for layer interaction effects: (L8+L2), (L8+neg_L9), (L2+neg_L9), all three.

**Success Criterion**: Multi-layer > best single-layer by >5pp
**Failure Criterion**: Multi-layer ≤ best single-layer (layer interaction is detrimental)

### C2: Position-Gated Steering

**Prerequisite**: Phase A confidence established
**Cost**: 2 days (needs token-position analysis infrastructure)

**Experiments**:
1. Analyze per-token steering effect: for each token position, measure accuracy improvement from steering at that token alone
2. Test position-dependent α: high α at reasoning-critical positions, zero elsewhere
3. Compare position-gated vs uniform steering

**Success Criterion**: Position-gated matches or exceeds uniform steering with <50% token divergence
**Failure Criterion**: Position-gated is worse or doesn't reduce divergence

### C3: Cross-Model Transfer Extension

**Prerequisite**: Phase A complete
**Cost**: 1 day (need to train TT on another model)

**Experiments**: Train TT on LLaMA-3.2-3B (or another MHA model). Transfer to Qwen2.5-7B and vice versa. Test whether trim-tab pattern (L8 dominance) is preserved.

**Success Criterion**: Cross-model transfer preserves trim-tab structure for at least one model pair beyond SmolLM2→7B
**Failure Criterion**: Pattern doesn't generalize to other model families

---

## Phase D: Long-Term — Fundamental (≥1 week, requires Phase C success)

### D1: Self-Bootstrapping TT

**Prerequisite**: Phase B + C demonstrate stable steering improvements
**Cost**: 3-5 days

**Experiments**:
1. Generate steered trajectories using current best TT
2. Train new TT on steered trajectories (resolving D9 distribution shift)
3. Compare new TT accuracy on steered-model predictions
4. Iterate 3-5 times

**Success Criterion**: TT accuracy on steered trajectories improves with each iteration (converges)
**Failure Criterion**: TT accuracy degrades or oscillates (diverges)

### D2: Head-Level Steering

**Prerequisite**: Phase C architectural confidence
**Cost**: 5-10 days (significant code changes)

**Experiments**:
1. Implement per-head KV-cache modification
2. Per-head sweep on L8: identify trim-tab heads and death-heads
3. Compare head-level vs layer-level steering at L8

**Success Criterion**: Head-level achieves L8-level improvement with <50% token divergence
**Failure Criterion**: Head-level doesn't isolate beneficial computation

### D3: Manifold-Aware Steering

**Prerequisite**: Phase C confirms α·v is valid for standard steering
**Cost**: 3-5 days

**Experiments**:
1. Estimate intrinsic dimensionality of hidden state trajectories (PCA)
2. Compute manifold curvature via local PCA / geodesic distance
3. Implement manifold projection: modify steering along geodesic instead of tangent
4. Compare manifold-aware vs Euclidean steering

**Success Criterion**: Manifold-aware > Euclidean steering by >5pp
**Failure Criterion**: No improvement (manifold is approximately flat)

---

## Decision Tree

```
START
  │
  ▼
PHASE A
  │
  ├── A1: Random Vector Control ──[TT > random]──→ CONTINUE
  │                              ──[TT ≈ random]──→ PIVOT to mechanistic study
  │
  ├── A2: Negative L9 ──[α<0 helps]──→ Include in multi-layer plan
  │                    ──[no effect]──→ L9 remains death layer only
  │
  └── A3: Contrastive Eval ──[works]──→ Proceed to B2
                           ──[fails]──→ Abandon contrastive, focus on standard TT
  │
  ▼
PHASE B (requires A1 positive)
  │
  ├── B1: Signed Sweep ──[3+ inverse trim-tabs]──→ Phase C1 viable
  │                    ──[all α≥0 optimal]──→ C1 is standard multi-layer only
  │
  ├── B2: Dual-Mode ──[β>0 helps]──→ D1 (bootstrap) gets better signal
  │                 ──[β=0 optimal]──→ Standard TT is sufficient
  │
  ├── B3: Small Models ──[works]──→ Capability threshold is NOT fundamental
  │                     ──[fails]──→ Confirms threshold (finding 3)
  │
  └── B4: Multi-Task ──[2/3 tasks]──→ Steering is general
                     ──[GSM8K only]──→ Task-specific, require per-task tuning
  │
  ▼
PHASE C (requires B1-4 positive direction)
  │
  ├── C1: Multi-Layer ──[>5pp vs single]──→ Steering improves with complexity
  │                   ──[≤single]──→ Single layer is optimal
  │
  ├── C2: Position Gate ──[match uniform, <50% div]──→ Practical deployment feasible
  │                      ──[worse]──→ Position doesn't matter
  │
  └── C3: Cross-Model ──[preserved]──→ Velocity dynamics are universal
                      ──[not preserved]──→ Model-specific
  │
  ▼
PHASE D (requires C1-3 positive)
  │
  ├── D1: Bootstrap ──[converges]──→ Self-improving steering
  │                 ──[diverges]──→ Distribution shift is fundamental
  │
  ├── D2: Head Level ──[less divergence]──→ More precise steering
  │                  ──[no improvement]──→ Layer is the right granularity
  │
  └── D3: Manifold ──[>5pp]──→ Steering geometry matters
                   ──[no effect]──→ Euclidean is fine
```

---

## Budget Summary

| Phase | Name | Cost (GPU-hours) | Wall Time | SSD Space | Go/No-Go After |
|-------|------|-----------------|-----------|-----------|-----------------|
| A1 | Random Vector Control | 0.5 | 1 hour | Negligible | 1 hour |
| A2 | Negative L9 | 0.3 | 45 min | Negligible | 2 hours |
| A3 | Contrastive Eval | 1.0 | 2 hours | Negligible | 3 hours |
| **A Total** | | **1.8** | **~3 hours** | **Negligible** | |
| B1 | Signed Sweep | 2.0 | 4 hours | 2GB temp | 1 day |
| B2 | Dual-Mode | 1.5 | 3 hours | Negligible | 1 day |
| B3 | Small Model Sweep | 1.0 | 2 hours | 1GB model | 1 day |
| B4 | Multi-Task Validation | 1.5 | 3 hours | 1GB datasets | 1 day |
| **B Total** | | **6.0** | **~1 day** | **~4GB** | |
| C1 | Asymmetric Multi-Layer | 4.0 | 1 day | Negligible | 1 week |
| C2 | Position-Gated Steering | 8.0 | 2 days | 5GB temp | 1 week |
| C3 | Cross-Model Transfer | 4.0 | 1 day | 4GB model | 1 week |
| **C Total** | | **16.0** | **~4 days** | **~9GB** | |
| D1 | Self-Bootstrapping TT | 12.0 | 3-5 days | 10GB+ trajectories | 2 weeks |
| D2 | Head-Level Steering | 20.0 | 5-10 days | Negligible | 2 weeks |
| D3 | Manifold-Aware | 8.0 | 3-5 days | 2GB | 2 weeks |
| **D Total** | | **40.0** | **~2 weeks** | **~12GB** | |

**Total Budget**: ~64 GPU-hours, ~3 weeks wall time (sequential), ~25GB storage
