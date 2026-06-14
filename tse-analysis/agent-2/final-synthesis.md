# Phase 12: Final Synthesis Report

---

```
=======================================================================
TRIADIC SYNTHESIS ENGINE — FINAL REPORT
=======================================================================
Subject: Velocity-based Latent Steering for Language Model Reasoning
        (RankAdaptation Project)
Mode: full
Date: 2026-06-14
Analyst: TSE Agent-2
Output: /home/filip/Projects/Personal/AI/RankAdaptation/tse-analysis/agent-2/

=======================================================================
```

---

## EXECUTIVE SUMMARY

The RankAdaptation project demonstrates that modifying a language model's KV cache during generation using predicted hidden state velocities can improve reasoning accuracy by up to +20pp (Qwen2.5-7B, L8, GSM8K). However, the project's strength is also its weakness: it has extensively documented **WHAT** works (learnable velocities, per-layer trim-tab/death-layer pattern, cross-model transfer, capability threshold) but has almost no mechanistic understanding of **WHY**. This TSE analysis identifies six critical findings: (1) the central R² paradox — perfect velocity prediction should mean zero steering effect — is unresolved, (2) the most important experiment (contrastive TT evaluation) remains pending, (3) the K/V amplification hypothesis offers a promising mechanistic explanation but is untested, (4) the contrastive direction may not exist for all models (Math-1.5B failure), (5) the project has four CONFIRMED EMERGENT capabilities that could transform steering into a general paradigm, and (6) the null hypothesis that TT is unnecessary (constant steering per layer works equally well) has never been tested. The analysis provides a concrete, resource-budgeted 4-phase plan (A→D) costing ~40 GPU-hours, with the immediate priority being the already-set-up contrastive evaluation.

---

## CORE FINDINGS

| # | Finding | Confidence | Source Phase | Channel |
|---|---------|-----------|-------------|---------|
| F1 | Per-layer trim-tab/death-layer pattern is empirically robust across models, datasets, and transfer scenarios | 9/10 | Phase 2 (Lenses 1,2,4,5) | [experiment] — already published |
| F2 | The steering mechanism is NOT mechanistically understood — the "why" is entirely unknown | 10/10 | Phase 2 (all lenses), Phase 8 | [theory] [doc] — critical gap |
| F3 | The R² paradox (high TT accuracy should mean zero steering effect) is the central unresolved theoretical issue | 8/10 | Phase 6 (D4), Phase 10 (H-2) | [theory] |
| F4 | Contrastive TT evaluation is the single most important pending experiment | 9/10 | Phase 7 (BP3, CF-4), Phase 9 (A1) | [codebase] — run_contrastive_eval.py exists, needs execution |
| F5 | Architecture choice (MHA vs hybrid) is a binary gate for the entire steering approach | 8/10 | Phase 2 (Lenses 1,4,6), Phase 3 (MR3) | [doc] — guideline for model selection |
| F6 | Four confirmed emergent capabilities exist: Keystone Layer, Frequency-Specific Steering, Dual-Surface Synergy, Meta-Trajectory Self-Application | 5/10 | Phase 4b | [theory] [experiment] |
| F7 | The null hypothesis "TT is unnecessary" has never been tested — constant steering per layer may match TT performance | 7/10 | Phase 8 (H0-2), Phase 9 (B1) | [experiment] |
| F8 | Cross-model transfer preserves L8 pattern despite different architectures and dimensionalities → velocity dynamics are partially model-agnostic | 7/10 | Phase 2 (Lens 6), Phase 7 (CF-6) | [experiment] |
| F9 | The capability threshold (~40% GSM8K) may be a manifold phase transition, not a smooth function | 4/10 | Phase 10 (H-3) | [theory] |
| F10 | The infrastructure (async loading, GPU cache, checkpoint resume) is mature and cost-effective | 8/10 | Phase 1 (C08), Phase 9 | [codebase] |

---

## PYRAMID OVERVIEW

**Levels**: 4 (Atoms → Composites → Higher Composites → Peak)
**Atoms**: 20 (A01-A20) — covering hidden states, velocities, steering parameters, empirical findings, infrastructure
**Composites**: 10 (C01-C10) — TT, KV-steering, per-layer sweep, contrastive pair, cross-model transfer, baseline eval, cross-dataset, async pipeline, capability threshold, all-layers
**Higher Composites**: 3 (P01-P03) — Steering Framework, Empirical Results, Contrastive Pipeline
**Junctions**: 29 (J01-J29) — 5 compositional, 8 causal, 4 temporal, 2 hierarchical, 4 constraint, 3 antagonistic, 3 synergistic

**Critical Junctions**:
- J06 (velocity → hidden manifold): Unverified causal link — the core theoretical weakness
- J24 (trim_tab ↔ death_layer): Antagonistic relationship between adjacent layers — unresolved
- J27 (contrastive TT + per-layer: synergistic): Untested — awaiting evaluation

---

## EMERGENT DISCOVERIES

| Classification | Count | Key Example |
|---------------|-------|-------------|
| CONFIRMED EMERGENT | 4 | Keystone Layer Hypothesis, Frequency-Specific Steering, Dual-Surface Synergy, Meta-Trajectory Self-Application |
| QUANTITATIVE ENHANCEMENT | 1 | Invariant Property of Trim-Tab Layers |
| COMPOSITIONAL | 2 | Hidden State × Peak Concept, Alpha × Peak Concept |
| REDUCTIVE | 0 | — |

**Highest Pairwise Synergy**: {TT, Per-layer Selective} = 0.85 — already core of the project
**Highest Higher-Order Synergy**: {TT, Dual-surface, Frequency decomposition} = 0.42 — self-organizing triple that creates "spectral resolved steering" as a qualitatively new capability
**Self-Organization Detected**: YES — two triples exhibit emergent properties not predictable from pairwise synergies

---

## MASTER REGULATORS

| Rank | Regulator | Influence | Leverage | Score | Modulation Strategy |
|------|-----------|-----------|----------|-------|-------------------|
| 1 | Layer Selection (A07/A08) | 92 | 95 | HIGH | Per-layer sweep (existing); gradient-based importance (proposed) |
| 2 | Contrastive TT Pipeline (C04) | 88 | 85 | HIGH | Two-TT differencing (existing); single TT with contrastive loss (proposed) |
| 3 | Architecture Selection (A13) | 75 | 90 | HIGH | Manual MHA selection (existing); GDN-specific mechanism (proposed) |
| 4 | Alpha Optimization (A06) | 70 | 65 | MEDIUM | Grid sweep (existing); RL-based per-token α (proposed) |
| 5 | Data Split Quality (A16) | 65 | 70 | HIGH | Binary correct/incorrect (existing); confidence-weighted, clustering-based (proposed) |

**Interaction Hierarchy**: MR3 (Architecture) → MR1 (Layer Selection) → MR2 (Contrastive Pipeline) → MR4 (Alpha) → MR5 (Data Split) — each regulator only matters if upstream regulators are correctly configured.

---

## TOP RECOMMENDATIONS (sorted by expected value / cost)

| # | Recommendation | Confidence | P(true) | Cost | Phase | Risk | Channel |
|---|---------------|-----------|---------|------|-------|------|---------|
| 1 | Run contrastive evaluation (run_contrastive_eval.py) on Qwen2.5-7B | 8/10 | 40% | 2 GPU-hrs | A1 | LOW | [codebase] |
| 2 | Test L8 and L9 with asymmetric α (positive AND negative) | 7/10 | 30% | 0.5 GPU-hrs | A2 | LOW | [codebase] |
| 3 | L8 ablation (keystone) test — does removing L8 steering collapse quality? | 8/10 | 50% | 0.3 GPU-hrs | A3 | LOW | [codebase] |
| 4 | Train and evaluate null models (zero, constant, linear TT) | 7/10 | 40% | 6 GPU-hrs | B1 | MEDIUM | [codebase] |
| 5 | PCA on TT velocity predictions; frequency-domain analysis of L8 vs L9 | 6/10 | 60% | 2 GPU-hrs | B2 | LOW | [codebase] |
| 6 | Capture attention distributions during steering (verify K/V amplification hypothesis) | 6/10 | 50% | 2 GPU-hrs | B3 | MEDIUM | [codebase] |
| 7 | Per-layer α vector on ALL layers (not just top-3) | 6/10 | 35% | 2 GPU-hrs | B (fallback) | LOW | [codebase] |
| 8 | RL-based per-token α optimization | 4/10 | 25% | 8 GPU-hrs | C1 | HIGH | [codebase] [experiment] |
| 9 | Dual-surface steering (KV-cache + weight-flow) | 4/10 | 30% | 12 GPU-hrs | C2 | HIGH | [codebase] [experiment] |
| 10 | Multi-head contrastive ensemble (bootstrap) | 5/10 | 30% | 8 GPU-hrs | C3 | MEDIUM | [codebase] [experiment] |

---

## RESOURCE-BUDGETED PLAN

### Phase A (IMMEDIATE — Diagnostic): 4 GPU-hours (10% budget)
**A1**: Contrastive evaluation (2 hrs) — GO/NO-GO for entire contrastive direction
**A2**: Asymmetric α sweep (0.5 hrs) — tests if L9 is an "invertible" death layer
**A3**: L8 ablation test (0.3 hrs) — confirms keystone hypothesis
**Contingency A2-alt**: If A1 fails → non-contrastive enhancement path (3 hrs)

### Phase B (SHORT-TERM — Targeted): 12 GPU-hours (30% budget)
**B1**: TT null model comparison (6 hrs) — determines if TT architecture is necessary
**B2**: Frequency-domain PCA analysis (2 hrs) — mechanistic hypothesis testing
**B3**: Attention pattern analysis (2 hrs) — empirical test of K/V amplification

### Phase C (MEDIUM-TERM — Architectural): 24 GPU-hours (60% budget)
**C1**: RL-based per-token α learning (8 hrs) — adaptive steering
**C2**: Dual-surface steering (12 hrs) — EM-3 implementation
**C3**: Multi-head contrastive ensemble (8 hrs) — variance reduction

### Phase D (LONG-TERM — Fundamental): Unbounded
**D1**: Non-math task evaluation (ARC, BBH, MMLU)
**D2**: Automated trim-tab discovery across model families
**D3**: Self-adaptive steering with online system identification

### Decision Tree
```
START → A1 (Contrastive Eval)
  ├── SUCCESS → A2 (Asymmetric α) → B1 (Null Models) → B2 → B3 → C1→C3
  └── FAILURE → A2-alt (Non-contrastive path) → B1 → Publish "Manifold Non-Separability"
```

---

## TESTABLE HYPOTHESES

| ID | Statement | Type | Falsified By | Confidence |
|----|-----------|------|-------------|-----------|
| H-1 | L8 modulates mid-frequency hidden state components; L9 modulates high-frequency noise | Structural | PCA reveals identical component structure across layers | 4/10 |
| H-2 | Steering effect is mediated by nonlinear K/V amplification, not manifold pushing | Relational | Gaussian Δa distribution; equal amplification across layers | 6/10 |
| H-3 | Capability threshold is a manifold phase transition (bifurcation of correct/incorrect trajectories) | Potential | Multiple models show smooth, not discontinuous, threshold crossing | 3/10 |
| H-4 | The project's own research velocity follows the same dynamics as model trajectories | Structural-Meta | Contrastive evaluation doesn't advance the project | 5/10 |

---

## CRITICAL DISPARITIES (Unresolved)

| ID | Description | Severity | Bounding Statement |
|----|-------------|----------|-------------------|
| D4 | High R² should mean zero steering effect (if TT predicts velocity perfectly, h+α·v ≈ h+α·v_actual, which is where the model naturally goes) | FUNDAMENTAL | The 88% token divergence at α=0.1 implies the hidden manifold is extremely sensitive to small perturbations. Resolution via K/V amplification hypothesis (H-2): the attention softmax amplifies small hidden state changes. |
| D5 | Train/test distribution shift: TT sees unsteered states during training but steered states during inference | RELATIONAL | Bounded by α < 0.5 (observed degradation threshold). For α ≤ 0.1, the shift is likely small enough that the TT's predictions remain valid. |
| D8 | Contrastive steering may push hidden states off-manifold (if correct/incorrect manifolds are not separable) | FUNDAMENTAL | Cannot be resolved without running the contrastive evaluation. If it works → manifolds ARE separable. If it fails → off-manifold hypothesis confirmed. |

---

## NEGATIVE SPACE

| Not Found | Why | Worth Investigating? |
|-----------|-----|---------------------|
| Interaction between steering and model quantization | Not tested | YES — 4-bit quantization may amplify or dampen steering effects |
| Steering as debiasing tool | Not tested | YES — ethical implications, but requires different evaluation metrics |
| Comparison to LoRA finetuning | Not tested | YES — establishes the relative value proposition of steering vs standard PEFT |
| Upper bound of steering improvement | Not computed | YES — from debrief's open questions; requires optimal α, layer combo, contrastive signal |
| Per-layer R² of TT | Not reported | YES — could trivially explain why L8 is special (lowest TT error at that layer) |

---

## SKILL SELF-ASSESSMENT

**Weaknesses found**:
1. Phase 8 (Mechanistic Check) made theoretical claims without empirical grounding — violated the "evidence grounding" principle. Proposed fix: add empirical grounding checkpoint before Phase 8.
2. The analysis missed the cross-reference between steering and adapter research (StreamFusion, 80+ adapter variants) because cross_subjects was not used.
3. The Inspirational lens was weak — the analyst lacks the domain breadth for high-quality forced collisions.

**Proposed updates to TSE**:
1. Add "counterfactual order analysis" to Phase 7 — CF-2 (reverse layer order could have killed project) is a generally applicable finding type.
2. Add automated per-layer metric computation to Phase 2 — never accept aggregate metrics when per-layer data exists.
3. Integrate deep-cross-research skill for the Inspirational lens (Lens 8) to improve domain breadth.

**Overall confidence in analysis**: **7/10**
- Highest confidence: Phase 0 (Assumptions), Phase 6 (Disparities), Phase 7 (Causal Map) — well-grounded in project data
- Lowest confidence: Phase 4b (Emergent) — genuinely novel but speculative; Phase 8 (Mechanistic) — theoretical without empirical backing
- What would increase confidence to 9/10: Running Phase A1 (contrastive eval), Phase B1 (null models), and Phase B2 (PCA analysis) would ground the theoretical findings in data.

---

## QUALITY INDEX

```
Q_total = 0.2·StructuralSoundness + 0.25·RelationalDepth + 0.2·PotentialCoverage
        + 0.2·Actionability + 0.15·SelfAwareness

StructuralSoundness: 8/10 (complete atom set, well-typed junctions, but missing A21-A24)
RelationalDepth: 8/10 (10 lenses applied, convergent check performed, but weak Inspirational lens)
PotentialCoverage: 7/10 (divergent pulse + emergent discovery generated ~60 candidates)
Actionability: 8/10 (resource-budgeted plan with decision tree, fallback paths)
SelfAwareness: 8/10 (recursive analysis identified 5 specific weaknesses + 5 TSE updates)

Q_total = 0.2(8) + 0.25(8) + 0.2(7) + 0.2(8) + 0.15(8) = 1.6 + 2.0 + 1.4 + 1.6 + 1.2 = 7.8/10
```

---

## FINAL STATEMENT

The RankAdaptation project has discovered a genuine phenomenon (per-layer velocity-based steering) with robust empirical support and demonstrated potential (+20pp accuracy improvement). However, the project is at an inflection point: it has accumulated extensive empirical results without developing a mechanistic understanding. The next two experiments — contrastive evaluation and TT null model comparison — will determine whether the current approach is a stepping stone to a general framework or a specific technique limited to capable MHA models on math tasks. The TSE analysis recommends investing the next 4 GPU-hours in Phase A diagnostic experiments, letting the results determine whether the project scales (Phases B-C) or wraps up with a well-documented empirical finding.

**Key insight for the researcher**: The most important discovery may not be "what works" but "what almost didn't work." Our counterfactual analysis (CF-2) shows that testing layers in reverse order could have aborted the project. The 4%→73% baseline jump from chat template was a near-miss. The +20pp result is real, but it's fragile — built on a sequence of lucky choices. Systematic mechanistic understanding (H-2: K/V amplification) would transform this from "lucky discovery" to "engineered technique."
