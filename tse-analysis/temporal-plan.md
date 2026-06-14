# Resource-Budgeted Temporal Plan — RankAdaptation

## Available Resources

| Resource | Status | Notes |
|----------|--------|-------|
| GPU | RTX 4060 Ti 16GB VRAM | Can run 7B with 4-bit quantization |
| Internal SSD | ~56GB free (/home) | For models + active data |
| External HDD | ~100GB+ free | For trajectory storage |
| CPU RAM | Unknown, ~16-32GB likely | Sufficient for 500 trajectories |
| Existing TTs | 5 checkpoints (4 on SSD, 2 on HDD) | All trained and ready |
| Existing trajectories | 7B (83 files, 35GB), Math-1.5B (37, 13GB), SmolLM2 (10, 6GB) | Sufficient for most experiments |

## Phase A: Immediate — Diagnostic (≤2 hours, no new infrastructure)

### A1: Death-Layer Sign Flip on L9
- **Cost**: ~20 min (30 problems, 4-bit 7B, single layer per configuration)
- **Experiment**: `run_per_layer_sweep.py` with `--alpha -0.1 --layers 9` (modify sign)
- **Success**: L9(-α) accuracy > L9(+α) accuracy — confirms L9 is an inverted trim-tab
- **Success+**: L9(-α) accuracy ≥ baseline — L9 is reclaimable
- **Failure**: L9(-α) accuracy still 0% — L9 is genuinely harmful regardless of direction
- **Go/No-Go**: If success, expand to negative α sweep across all death layers (L7, L9, L15+)

### A2: Remove First-Step Gate
- **Cost**: ~15 min code change + ~30 min evaluation
- **Experiment**: Modify all steering scripts to steer at t=0 (remove `first_step` guard)
- **Success**: Accuracy at least matches steering without first-step, ideally exceeds it
- **Failure**: Accuracy degrades significantly — first-step steering hurts
- **Go/No-Go**: If success, adopt as new default. If failure, investigate using prompt-trained TT for t=0.

### A3: Contrastive Similarity Analysis (H0-3 from Phase 8)
- **Cost**: ~10 min (compute cosine similarity between v_c and v_i for 50 examples)
- **Experiment**: Modify `run_contrastive_eval.py` to compute and print cos(v_c, v_i) per example
- **Success**: cos(v_c, v_i) < 0.5 — TTs learn meaningfully different dynamics
- **Failure**: cos(v_c, v_i) > 0.9 — TTs are nearly identical, contrastive direction ≈ 0
- **Go/No-Go**: If success, proceed with contrastive evaluation. If failure, contrastive TT approach is likely invalid.

### A4: λ Interpolation Steering (CF-6 from Phase 7)
- **Cost**: ~10 min code change + ~30 min evaluation
- **Experiment**: `v_steer = λ·v_c + (1-λ)·v_i` for λ ∈ {0, 0.25, 0.5, 0.75, 1.0} at L8
- **Success**: Accuracy peaks at λ > 0.5 — confirms normative direction helps
- **Failure**: Flat accuracy across λ — contrastive signal is noise
- **Go/No-Go**: If λ=1.0 gives best results, steer directly with v_c.

### Total Phase A Cost: ~2 hours compute, 0 new infrastructure

## Phase B: Short-term — Targeted (≤1 day, builds on Phase A)

### B1: Statistical Validation (H0-1 from Phase 8)
- **Requires**: Phase A decisions on best steering configuration
- **Cost**: ~2 hours (500 problems, 3-5 layer configurations)
- **Experiment**: Run L8, L2, L9 (+ best A1 result) with N=500 problems
- **Success**: +20pp L8 confirmed at N=500 (±4.4pp CI)
- **Failure**: Effect shrinks to <10pp at N=500 — original result was statistical fluctuation

### B2: Norm-Growth Baseline (H0-2 from Phase 8)
- **Requires**: None
- **Cost**: ~30 min
- **Experiment**: Compute null baseline velocity = norm-ratio-based, compare R² to TT
- **Success**: TT significantly outperforms norm baseline — TT learns meaningful dynamics
- **Failure**: TT ≈ norm baseline — TT is learning trivial norm growth

### B3: Hybrid Steering (v_std + β·(v_c - v_i))
- **Requires**: A4 results
- **Cost**: ~1 hour
- **Experiment**: Sweep β ∈ {0, 0.1, 0.3, 0.5, 1.0} with best α from per-layer sweep
- **Success**: β > 0 gives better accuracy than β = 0 (v_std alone)
- **Failure**: β = 0 gives best result — contrastive signal doesn't add to standard

### B4: GDN Recurrent State Steering (RECOMB-5)
- **Requires**: Working GDN state modification code
- **Cost**: ~2 hours (implement + test on 5 problems + scale to 50)
- **Experiment**: Modify Qwen3.5-2B's GDN recurrent states instead of K/V cache
- **Success**: Any accuracy improvement on Qwen3.5-2B (which was 0% with K/V approach)
- **Failure**: No improvement — GDN recurrent states not steerable

### Total Phase B Cost: ~5-6 hours compute

## Phase C: Medium-term — Architectural (≤1 week, systemic changes)

### C1: Per-Head Steering (EM-1)
- **Requires**: Understanding of L8 per-head effects
- **Cost**: ~2 days (code + analysis)
- **Experiment**: Within L8, steer each attention head independently (28 heads)
- **Success**: Identify 3-5 "trim-tab heads" that drive the L8 effect; steer them at 2× α and others at 0

### C2: Adaptive α(t) via Bayesian Optimization
- **Requires**: B1 results for reliable accuracy measurement
- **Cost**: ~3 days (Bayesian optimization setup)
- **Experiment**: Optimize α(t) as a function of token position (e.g., Gaussian process with position as input)
- **Success**: α(t) schedule that outperforms best constant α

### C3: Synthetic Toy Transformer Validation (Phase 8)
- **Requires**: None (purely synthetic)
- **Cost**: ~4 hours
- **Experiment**: Build two-layer synthetic transformer, generate trajectories with known correct velocity, verify pipeline can recover it
- **Success**: TT achieves R² > 0.95 on synthetic data; steering with ground-truth velocity improves accuracy
- **Failure**: Pipeline fails on known ground truth → fundamental bug in steering mechanism

### Total Phase C Cost: ~5 days, significant code development

## Phase D: Long-term — Fundamental (research program)

### D1: Cross-Task Polarity Map (EM-3)
- **Requires**: C1 completion, capability to run multiple benchmarks
- **Cost**: ~2 weeks
- **Experiment**: Run per-layer sweeps on ARC, BBH, MMLU (3-5 tasks × 28 layers × 100 problems)
- **Goal**: Compute polarity correlation matrix across tasks. Are L8 trim-tab effects consistent?

### D2: Self-Supervised Contrastive Steering (EM-4)
- **Requires**: Understanding of trajectory clustering (from C3 synthetic analysis)
- **Cost**: ~1 month
- **Experiment**: Cluster trajectories by convergence property without labels, train TT on clusters
- **Goal**: Label-free normative steering

### D3: RL-Optimized Steering Policy
- **Requires**: B1 (reliable metric), B3 (hybrid steering)
- **Cost**: ~2 weeks
- **Experiment**: Train RL policy (PPO/REINFORCE) that selects α per (layer, token) to maximize GSM8K accuracy
- **Goal**: Maximize theoretical upper bound on steering improvement

### Total Phase D Cost: ~2 months

## Decision Tree

```
Phase A Start
├── A1 (Death sign flip) → success → B1 includes negative-α layers
│                       → failure → continue with positive α only
├── A2 (First-step) → success → adopt as default
│                  → failure → keep first-step skip, investigate separately
├── A3 (Contrastive similarity) → cos < 0.5 → proceed with contrastive
│                               → cos > 0.9 → ABANDON contrastive approach
└── A4 (λ interpolation) → λ peak > 0.5 → contrastive viable
                         → flat → contrastive may be noise

Phase B (if Phase A mostly successful)
├── B1 (N=500 validation) → +20pp confirmed → high confidence in steering
│                         → effect shrinks → need more robust approach
├── B2 (Norm baseline) → TT better → dynamics are meaningful
│                      → TT ≈ norm → need new TT architecture
├── B3 (Hybrid steering) → β > 0 helps → adopt hybrid steering
│                        → β=0 best → use standard TT only
└── B4 (GDN steering) → works → unlock Qwen3.5 hybrid models
                      → fails → accept hybrid architecture limitation

Phase C (if Phase B shows robust effects)
├── C1 (Per-head) → identify trim-tab heads → precision steering
├── C2 (Adaptive α) → α(t) beats constant → temporal optimization
└── C3 (Synthetic validation) → PASS → pipeline validated → confidence high
                              → FAIL → STOP AND INVESTIGATE FUNDAMENTAL ISSUE

Phase D (if Phase C successful)
├── D1 (Cross-task) → polarity consistent → generalizable mechanism
│                   → polarity task-specific → task-adaptive steering needed
├── D2 (Self-supervised) → works → label-free steering, major advance
└── D3 (RL policy) → beats manual α → automated steering optimization
```
