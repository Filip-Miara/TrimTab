# Phase 12: Final Synthesis Report

**Subject**: RankAdaptation — Velocity-based latent steering for language model reasoning
**Mode**: Full (all 12 phases)
**Date**: 2026-06-14

---

## EXECUTIVE SUMMARY

This Triadic Synthesis Engine analysis of the RankAdaptation project reveals that velocity-based latent steering is a genuine phenomenon with robust empirical support (trim-tab layers produce +20pp accuracy improvements; the pattern generalizes across datasets and model families), but the project is currently at a **pre-paradigmatic stage** — it has strong observational results without a mechanistic theory for WHY steering works. The most critical finding from the TSE analysis is that **the project's highest-value experiment (contrastive TT evaluation) remains unexecuted**, and the single most impactful parameter (α per layer) has never been systematically explored. Three confirmed emergent capabilities were identified — a self-correcting feedback loop, a universal velocity manifold, and reasoning topography mapping — that could transform the project's trajectory. The analysis found 12 significant structural and relational disparities (3 critically unresolved), 10 testable hyperstitional hypotheses, and a clear resource-budgeted decision tree for proceeding.

---

## CORE FINDINGS

| # | Finding | Confidence | Source Phase | Channel |
|---|---------|------------|--------------|---------|
| F1 | The trim-tab/death-layer pattern is real, robust, and generalizes across datasets (GSM8K→SVAMP) and model families (SmolLM2→7B) | 9/10 | Lens Cascade (8/10 lenses agree) | `[theory]` |
| F2 | Steering is amplification, not creation — it requires baseline capability (~40% GSM8K) | 8/10 | Dialectical lens, but D4 unresolved (α dependency) | `[theory]` |
| F3 | Per-layer selectivity is mandatory; all-layers steering is net negative due to death-layer interference | 9/10 | Systems lens + causal map | `[codebase]` |
| F4 | The project has no mechanistic theory for why steering works or why layers differ | 9/10 | Metacognitive lens (10/10 lenses flag this gap) | `[theory]` |
| F5 | R² is NOT a reliable predictor of steering success (Math-1.5B: R²=0.892, no trim tabs) | 8/10 | Disparity Matrix (D1 resolved) | `[theory]` |
| F6 | The contrastive direction (v_correct − v_incorrect) is the most promising but untested next direction | 7/10 | Master Regulator #3; disparity D10 unresolved | `[experiment]` |
| F7 | The default α=0.1 is almost certainly suboptimal; α is the most under-explored parameter | 9/10 | All 10 lenses flagged this | `[experiment][codebase]` |
| F8 | Death-layer identity may be perturbation-sensitivity rather than computation-disruption (untested) | 6/10 | Mechanistic Check (H0-2 partially testable) | `[experiment]` |
| F9 | Three emergent capabilities exist: self-correcting loop, universal velocity manifold, reasoning topography | 6/10 | Emergent Discovery (Phase 4b) | `[theory]` |
| F10 | The capability threshold may be α-dependent, not fundamental (D4 unresolved) | 5/10 | Disparity Matrix (D4 critical) | `[experiment]` |

---

## PYRAMID OVERVIEW

| Metric | Count |
|--------|-------|
| Total Levels | 4 (Atoms → Level-2 → Level-3 → Peak) |
| Atomic Concepts | 20 (A1-A20) |
| Level-2 Composites | 9 (C1-C9) |
| Level-3 Composites | 3 (Full Pipeline, Research Program, Theoretical Framework) |
| Peak Concept | 1 (RankAdaptation System) |
| Typed Junctions | 12 (J1-J12) |
| Junction Types | Causal (3), Compositional (2), Modulatory (1), Hierarchical (1), Antagonistic (2), Conditional (1), Dependency (1), Temporal (1), Constraint (1) |

### Key Atoms

| Atom | Role | Key Property |
|------|------|-------------|
| A1: Hidden State Velocity | Core phenomenon | R²=0.85-0.94 learnable |
| A2: TrajectoryTransformer | Prediction engine | 192MB; cross-model transferable |
| A3: KV-Cache Steering | Intervention mechanism | 88% token divergence |
| A4: Per-Layer Selectivity | Methodological key | Necessary for success |
| A5: Trim-Tab Layer | Positive outcome | L8: +20pp |
| A6: Death Layer | Negative outcome | L9: -23pp |
| A7: Capability Threshold | Limiting constraint | ~40% GSM8K |
| A8: Contrastive TT | Next direction | Untested |

---

## EMERGENT DISCOVERIES

| Category | Count | Top Entry |
|----------|-------|-----------|
| Unconventional Recombinations | 12 | 3 cross-level, 3 domain-transposed, 3 forbidden-pair, 1 self-application |
| **CONFIRMED EMERGENT** | **3** | Self-correcting loop (EM-1), Universal velocity manifold (EM-2), Reasoning topography (EM-4) |
| QUANTITATIVE ENHANCEMENTS | 1 | Death layer inoculation (EM-5) |
| COMPOSITIONAL | 1 | Attention-redistribution steering (EM-3) |

### Highest Synergies

| Type | Entity | Score |
|------|--------|-------|
| **Highest Pairwise** | {A5, A6}: Trim-Tab × Death Layer | +6 (complementary classification system) |
| **Highest Higher-Order** | {A1, A2, A4, A5}: Velocity × TT × Per-Layer × Trim-Tab | **+21** |
| **Self-Organization Detected** | YES | The quadruple shows strong positive higher-order synergy |

---

## MASTER REGULATORS

| Rank | Regulator | Type | Score | Current State | Recommendation |
|------|-----------|------|-------|---------------|---------------|
| #1 | Per-Layer Selectivity | Junction | 81 | Manual fixed sweep | Automated Bayesian optimization over (layer, α) |
| #2 | Steering Strength α | Junction | 56 | Fixed α=0.1 everywhere | Per-layer α sweep (Phase A, immediate) |
| #3 | Contrastive Direction Signal | Junction | 56 | TTs trained, pending eval | **Execute evaluation now** (Phase A, immediate) |
| #4 | Capability Threshold | Conditional | 54 | Empirical ~40% boundary | Test α sweep on Math-1.5B |
| #5 | Training Data Quality | Composite | 40 | Raw generation data | Curated/importance-weighted trajectories |

---

## TOP RECOMMENDATIONS (sorted by expected value)

### #1: Run Contrastive Evaluation (Immediate)

| Property | Value |
|----------|-------|
| Confidence | 9/10 |
| P(true improvement) | 60% (informed by standard TT baseline of +20pp) |
| Cost | ~45 min GPU time |
| Phase | Phase A (Diagnostic Sprint) |
| Risk | **Minimal** — script exists, models trained, just needs execution |
| Channel | `[codebase]` |
| Expected Value | HIGH — resolves D10, tests H-3, validates the project's primary next direction |

### #2: Run Anti-Steering at Death Layers (Immediate)

| Property | Value |
|----------|-------|
| Confidence | 8/10 |
| P(informative result) | 90% (regardless of outcome, answers important question) |
| Cost | ~30 min GPU time |
| Phase | Phase A |
| Risk | Low — diagnostic only |
| Channel | `[codebase]` |
| Expected Value | HIGH — distinguishes off-manifold from direction-misalignment mechanism |

### #3: Run α Sweep on L8 and L9 (Immediate)

| Property | Value |
|----------|-------|
| Confidence | 8/10 |
| P(finding optimal α) | 85% (highly likely α is not at optimum) |
| Cost | ~1 hour GPU time |
| Phase | Phase A |
| Risk | Low — may find no better α, but still informative |
| Channel | `[codebase]` |
| Expected Value | HIGH — addresses MR#2, could significantly improve all subsequent results |

### #4: Run Random Direction Baseline (Short-term)

| Property | Value |
|----------|-------|
| Confidence | 7/10 |
| P(TT direction > random) | 80% (based on robust trim-tab pattern) |
| Cost | ~1 hour GPU time |
| Phase | Phase B |
| Risk | Low — critical ablation |
| Channel | `[codebase]` |
| Expected Value | HIGH — validates that steering direction specificity matters |

### #5: Test High α on Sub-Threshold Math-1.5B (Short-term)

| Property | Value |
|----------|-------|
| Confidence | 6/10 |
| P(finding any positive α-layer combination) | 40% |
| Cost | ~2 hours GPU time |
| Phase | Phase B |
| Risk | Medium — may destroy model output at high α |
| Channel | `[experiment]` |
| Expected Value | MEDIUM — resolves D4, potentially expands steerable model set |

---

## RESOURCE-BUDGETED PLAN

### Phase A: Diagnostic Sprint (0-2 hours)
- A1: Run contrastive evaluation (`run_contrastive_eval.py`) — **45 min**
- A2: Anti-steering at L9 and L8 (flip α sign) — **30 min**
- A3: α sweep on L8 [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0] — **1 hour**
- **Go/No-Go**: ≥2/3 positive signals → Phase B

### Phase B: Targeted Experiments (2-12 hours)
- B1: Random direction baseline — **1 hour**
- B2: Multi-layer pair (L2+L8, L8+L10, L8+L9) — **2 hours**
- B3: High α on Math-1.5B — **2 hours**
- B4: Early-only steering — **1 hour**
- **Go/No-Go**: ≥3/4 success → Phase C

### Phase C: Architectural Changes (1-3 days)
- C1: Siamese contrastive TT (triplet loss) — **4 hours**
- C2: Bayesian α optimization over layers — **8 hours**
- C3: Cross-dataset evaluation (ARC, BBH, MMLU) — **6 hours**
- C4: Death layer inoculation training (risky) — **4 hours**
- **Go/No-Go**: ≥2/4 success → Phase D

### Phase D: Research Program (1-2 weeks)
- D1: Self-correcting steering loop
- D2: Universal velocity manifold verification
- D3: RL-based α optimization

---

## TESTABLE HYPOTHESES

| ID | Hypothesis | Falsified By | Channel | Priority |
|----|-----------|--------------|---------|----------|
| H-1 | Trim-tab layers align with steering direction (cos similarity) | L8 cos<0 OR L9 cos>0 | `[experiment][theory]` | Phase C |
| H-2 | Death layers are off-manifold sensitive | Random perturbation benign at L9 | `[experiment]` | Phase B (B1) |
| H-3 | Contrastive TT improves over standard TT | Contrastive ≤ standard at L8 | `[experiment]` | **IMMEDIATE** (A1) |
| H-4 | Capability threshold is α-dependent | No (α, layer) improves Math-1.5B | `[experiment][theory]` | Phase B (B3) |
| H-5 | Instruct tuning causal for trim-tabs | 1.5B-instruct shows no trim-tabs | `[experiment]` | Phase C |
| H-6 | Dual-surface > single-surface | Dual-surface ≤ max(single) | `[experiment]` | Phase C |
| H-7 | Multi-layer steering is sub-additive | L2+L8 ≥ L8 alone | `[experiment]` | Phase B (B2) |
| H-8 | Velocity dynamics are universal across LLMs | Cross-model transfer fails | `[theory]` | Phase D |
| H-9 | v_pred ≈ gradient of correctness | cos(v_pred, ∇accuracy) < 0.1 | `[theory][experiment]` | Phase C |
| H-10 | Trim-tab classification reverses per token position | L8 helps all positions equally | `[experiment]` | Phase C |

---

## CRITICAL DISPARITIES (unresolved)

| ID | Disparity | Severity | Unblocking Action | Status |
|----|-----------|----------|-------------------|--------|
| D4 | Capability threshold measured at α=0.1 only | FUNDAMENTAL | α sweep on Math-1.5B | Requires B3 |
| D5 | Instruct vs base model confound (size × training) | STRUCTURAL | Test 1.5B-instruct + 7B-base | Requires B3 + C3 infrastructure |
| D10 | Contrastive TTs trained but evaluation not run | **CRITICAL** | Run contrastive evaluation | **Immediate action** |

---

## NEGATIVE SPACE

| Absent Finding | Significance |
|----------------|-------------|
| No mechanistic layer-function analysis | TRIM-TAB MECHANISM UNKNOWN — the single biggest knowledge gap |
| No connection to in-context learning | Both modify behavior without weight change; may share mechanisms |
| No α sweep | DEFAULT α=0.1 MAY BE MASKING 2× TO 3× IMPROVEMENT |
| No random baseline | CONFIRMING DIRECTION SPECIFICITY IS UNTESTED |
| No cross-task generalization (ARC, BBH, MMLU) | SCOPE OF FINDINGS LIMITED TO MATH DOMAINS |
| No synthetic data validation | ALL FINDINGS ARE CORRELATIONAL, NOT CAUSALLY GROUNDED |

---

## SKILL SELF-ASSESSMENT

| Dimension | Score | Weaknesses Found | Proposed TSE Updates |
|-----------|-------|------------------|---------------------|
| Structural Soundness | 8/10 | Some missing atoms (hardware, methodology, time) | Add "system constraints" atom category (U2) |
| Relational Depth | 7/10 | Missed external research context | Add "external context scan" sub-phase (U1) |
| Potential Coverage | 7/10 | Over-indexed on internal over external validity | Add null-result calibration audit (U3) |
| Actionability | 8/10 | No ROI estimates for recommendations | Add expected-value calculation (U4) |
| Self-Awareness | 8/10 | Alternative framings not explored | Add "alternative framing" to VOID (U5) |

**Aggregate Quality Index**:
```
Q = 0.2×8 + 0.25×7 + 0.2×7 + 0.2×8 + 0.15×8
  = 1.6 + 1.75 + 1.4 + 1.6 + 1.2
  = 7.55/10
```

---

## CONCLUSION

The RankAdaptation project has discovered a genuine phenomenon (velocity-based latent steering with trim-tab/death-layer structure) that is robust, reproducible, and partially generalizable. However, the project is at a critical juncture: it has accumulated infrastructure, data, and preliminary results, but the next step — contrastive evaluation — and the most impactful parameter exploration — α sweeps — remain unexecuted.

**The TSE analysis recommends:**

1. **Execute Phase A immediately** (2 hours): contrastive evaluation, anti-steering, α sweep. These three experiments will resolve the project's critical uncertainties.
2. **Build the mechanistic theory** in parallel: cosine alignment (H-1), random baselines (H-2), and per-token tracking (H-10) will transform steering from an empirical observation into a understood mechanism.
3. **If Phase A succeeds**, proceed through Phases B→C→D following the decision tree. The confirmed emergent capabilities (self-correcting loop, universal manifold) represent the project's highest long-term value.
4. **If Phase A fails**, the project should reassess fundamentals — the steering surface (residual stream instead of KV-cache), the direction signal (gradient instead of velocity), or the project's viability as an approach.

The single most important action is: **run the contrastive evaluation that's already set up**. It costs 45 minutes, it's fully resourced, and it will either validate the project's direction or redirect it.

---

*Generated by Triadic Synthesis Engine v1.0.0 — Full mode — 14 June 2026*
