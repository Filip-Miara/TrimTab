=======================================================================
TRIADIC SYNTHESIS ENGINE — FINAL REPORT
=======================================================================
Subject: RankAdaptation — Velocity-Based Latent Steering
Mode: full
Date: 2026-06-13

--- EXECUTIVE SUMMARY ---
The project has proven that (1) hidden state velocities during generation ARE learnable (R²=0.94), (2) per-layer steering reveals "trim tabs" (+20pp at L8) and "death layers" (-23pp at L9), and (3) the layer-specific pattern transfers across datasets (SVAMP) and model families (SmolLM2→Qwen2.5-7B). The central failure of all-layers steering is explained by death-layer noise drowning out trim-tab signal. The contrastive TT (v_correct − v_incorrect) is the highest-leverage next step, with asymmetric α and multi-layer combinations as refinements.

--- CORE FINDINGS ---
1. **Trim-tab layers exist** — L8 = +20pp, L2 = +17pp, L9 = -23pp [confidence: 9/10, evidence: per-layer sweep]
2. **Gen trajectories are learnable** — R²=0.94 (SmolLM2), 0.873/0.909 (Math-1.5B contrastive) [9/10]
3. **Cross-domain generalization** — SVAMP replicates L8/L9 pattern [8/10]
4. **Cross-model transfer** — SmolLM2 TT → Qwen2.5-7B preserves L8 as best [7/10]
5. **Contrastive TTs trained** — TT_correct R²=0.873, TT_incorrect R²=0.909 [8/10]
6. **All-layers steering is net negative** — bad layers dominate [9/10]

--- MASTER REGULATORS ---
1. **Contrastive TT training** — solves descriptive/normative flaw [HIGH]
2. **Layer-selective steering** — eliminates death-layer noise [HIGH]
3. **Per-layer asymmetric α** — independent attraction/repulsion strengths [MED-HIGH]

--- TOP RECOMMENDATIONS (sorted by expected value) ---

#1: Evaluate contrastive steering sweep (currently running)
   Confidence: 8/10 | P(true improvement): 70% | Cost: ~2 GPU-hours
   Wait for contrastive_results.json to identify Math-1.5B trim tabs

#2: Asymmetric α sweep on best trim-tab layers
   α_c ∈ {0.01, 0.05, 0.1}, α_i ∈ {0.01, 0.05, 0.1}
   Confidence: 7/10 | P(improvement): 65% | Cost: ~3 GPU-hours

#3: Multi-layer combinations (pairs and triplets of best layers)
   L2+L8, L5+L8, L2+L5+L8, etc.
   Confidence: 7/10 | P(improvement): 60% | Cost: ~4 GPU-hours

#4: Multi-head contrastive ensembles (N=3-5 bootstrapped pairs)
   Confidence: 6/10 | P(improvement): 55% | Cost: ~6 GPU-hours (training)

#5: torch.compile() for 2× training speed
   Confidence: 8/10 | P(improvement): 95% for speed | Cost: <1 hour

--- RESOURCE-BUDGETED PLAN ---

Phase A (Immediate — ~4 GPU-hours):
  ├── A1: Wait for contrastive sweep results [running]
  ├── A2: Asymmetric α sweep on top-3 layers
  ├── A3: Multi-layer combination sweep (pairs, triplets)
  └── A4: Report best configuration

Phase B (Short-term — ~8 GPU-hours):
  ├── B1: Multi-head contrastive ensemble (N=3)
  ├── B2: Apply best config to SVAMP + MATH datasets
  ├── B3: Per-layer α vector optimization
  └── B4: Compare with prompt-trained TT baseline

Phase C (Medium-term):
  ├── C1: Standard-attention 7B model with contrastive steering
  ├── C2: Online evaluation (early-exit correctness proxy)
  └── C3: torch.compile for 2× iteration speed

--- TESTABLE HYPOTHESES ---

H-1: [Contrastive Steering] Contrastive TT (v_correct − v_incorrect) at L8 with α=0.05 improves GSM8K accuracy by ≥5pp over baseline on Math-1.5B
  Falsified by: ≤0pp improvement at 95% confidence interval

H-2: [Asymmetric α] α_c ≠ α_i improves accuracy over α_c = α_i on the same layer
  Falsified by: α_c = α_i sweep dominates α_c ≠ α_i sweep

H-3: [Multi-Layer] Steering L2+L8 simultaneously produces ≥1.5× the improvement of either alone
  Falsified by: combined improvement ≤ max(individual)

--- SKILL SELF-ASSESSMENT ---
Weaknesses: Heavy on analysis, light on implementation details for each recommendation. The contrastive sweep is already running, which is good — the analysis is keeping pace with execution.

=======================================================================
