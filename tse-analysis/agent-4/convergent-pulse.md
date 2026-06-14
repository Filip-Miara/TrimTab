# Phase 5: Convergent Pulse

## Candidate Pool

Total candidates generated from Phase 4 and Phase 4b: **112** (32 mutation variants + 40 forced collisions + 40 unconventional recombinations)

## Filter Results

### F1: Feasibility (≥3/5)

Evaluated by structural mode: does the variant respect atom/junction constraints?

| Result | Count | Examples |
|--------|-------|----------|
| **Passed** | 67 | Adaptive α, death layer exclusion, cross-model injection, per-layer α sweep, K/V split steering |
| **Failed** | 45 | Foundation TT on 100+ LMs (no access), continuous gradient-based α optimization (non-differentiable loop), Fourier-domain steering (no infrastructure) |

### F2: Safety (No catastrophic failure modes)

Evaluated by relational mode: does it create dangerous feedback loops?

| Result | Count | Examples |
|--------|-------|----------|
| **Passed** | 58 | Most variants pass (steering safety is naturally bounded — worst case = baseline accuracy) |
| **Failed** | 9 | α = 5.0 on death layers (model collapse risk), 50K trajectory collection (SSD overflow), random steering baseline could be misleading if published as "steering works" |
| **Warning** | 12 | Adversarial steering (could be weaponized), RL-based α optimization (reward hacking) |

### F3: Telos Alignment (≥4/5)

Evaluated by potential mode: does it move toward the desired state (improving reasoning)?

| Result | Count | Rationale |
|--------|-------|-----------|
| **Passed** | 41 | Direct accuracy improvement, better understanding of steering mechanism |
| **Marginal** | 26 | Infrastructure improvements (async loading, GPU caching — enable future progress but don't directly improve steering) |
| **Failed** | 8 | Toy experiments, negative control studies, misapplication risk |

### F4: Novelty (≥3/5)

Cross-check: genuinely different from existing approaches?

| Result | Count | Notes |
|--------|-------|-------|
| **Passed** | 53 | Steering regime classifier, anti-steering defense, cross-model injection, adaptive policy |
| **Failed** | 22 | Random steering baseline (trivial), larger dataset (obvious), more TT capacity (obvious) |

### F5: Synergistic Potential (≥3/5)

Pairwise check: combines well with others?

| Result | Count | Notes |
|--------|-------|-------|
| **Passed** | 48 | Adaptive α + contrastive steering + death layer exclusion = full pipeline improvement |
| **Failed** | 19 | Standalone infrastructure improvements with no interaction |

## Final Survivors

**Total passed all 5 filters**: 36 candidates

## Top-10 Ranked

Score = `(Novelty + Feasibility + Telos_Alignment + (6 - Risk)) / 4`

| # | Candidate | Novelty (1-5) | Feasibility (1-5) | Telos (1-5) | Risk (1-5) | Score | Rationale |
|---|-----------|---------------|-------------------|-------------|------------|-------|-----------|
| 1 | **Adaptive per-layer α optimized via CMA-ES** | 5 | 4 | 5 | 2 | 4.75 | Highest impact per effort. CMA-ES is off-the-shelf. 28-dimensional search. Directly addresses death layers. |
| 2 | **Death layer exclusion + negative α on identified death layers** | 4 | 5 | 5 | 1 | 4.75 | Immediate safety improvement. Zero new infrastructure. Can be deployed today. |
| 3 | **K/V split steering (α_k ≠ α_v)** | 5 | 4 | 4 | 2 | 4.50 | Tests a fundamental assumption. Could reveal that keys and values play different roles in steering. |
| 4 | **Steering-regime classifier for safe/unsafe token-layer prediction** | 5 | 3 | 5 | 2 | 4.50 | Enables automated safe steering. Needed for production deployment. |
| 5 | **Contrastive ensemble (bagging N bootstrapped pairs)** | 4 | 4 | 5 | 2 | 4.50 | Parallelizable. Directly addresses contrastive signal quality. |
| 6 | **α = f(entropy): steering strength as function of prediction uncertainty** | 4 | 4 | 4 | 2 | 4.25 | Intuitive connection between uncertainty and steering need. Easy to implement. |
| 7 | **Cross-model injection (capable→incapable via shared steering prior)** | 5 | 3 | 5 | 3 | 4.00 | If it works, removes the capability threshold limitation. High upside. |
| 8 | **Per-token asymmetric α schedule (low early, ramp late)** | 3 | 5 | 4 | 2 | 4.00 | Simple modification. Could significantly improve generation quality. |
| 9 | **Multi-head contrastive ensemble** | 4 | 3 | 5 | 2 | 4.00 | Bagging reduces variance. Requires training multiple TTs. |
| 10 | **Anti-steering defense measurement protocol** | 5 | 4 | 3 | 3 | 3.75 | Primarily diagnostic, not directly improving accuracy. But provides crucial understanding. |

## Key Survivors from Phase 4b Emergent Discovery

| Candidate | Classification | Why Bypassed F1 |
|-----------|---------------|-----------------|
| **Adaptive Steering Policy (EM-1)** | CONFIRMED EMERGENT | Bypasses F1 (emergence is by definition not subject to feasibility constraints) — included as aspirational target |
| **Cross-Model Steering Injection (EM-2)** | CONFIRMED EMERGENT (conditional) | Bypasses F1 — included as speculative high-value target |
| **Anti-Steering Defense (EM-3)** | CONFIRMED EMERGENT (speculative) | Bypasses F1 — included as research question |
| Steering-Regime Classifier (EM-4) | QUANTITATIVE ENHANCEMENT | Passes standard filters — ranked #4 above |
| Curriculum Steering (EM-5) | COMPOSITIONAL | Passes standard filters — ranked #8 above |

## Rejected Candidates of Note

| Candidate | Failed Filter | Why Notable |
|-----------|--------------|-------------|
| Foundation TT (100+ LMs) | F1 (feasibility) | Good idea but requires resources beyond scope |
| Random steering baseline | F4 (novelty) | Important control but trivial to implement |
| Fourier-domain steering | F1 (infrastructure) | Interesting but requires new code |
| Direct logit-space intervention | F3 (telos: tangential) | Tested in Session 1 — already known to fail |
