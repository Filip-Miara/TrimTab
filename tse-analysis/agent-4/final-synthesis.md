# Phase 12: Final Synthesis Report

=======================================================================
TRIADIC SYNTHESIS ENGINE — FINAL REPORT
=======================================================================

**Subject**: RankAdaptation — Velocity-based latent steering for language model reasoning
**Mode**: full (all 12 phases executed)
**Date**: 2026-06-14
**Analyst**: TSE Agent-4

=======================================================================

## EXECUTIVE SUMMARY

This Triadic Synthesis Engine analysis of the RankAdaptation project reveals a project with strong empirical discoveries (R²=0.85-0.94 velocity learnability, L8:+20pp trim-tab effect, cross-model transfer) that lacks mechanistic understanding of WHY these effects occur. The most critical unmet need is a **random steering control** (H0-1): if random vectors of matching magnitude produce similar per-layer patterns, the entire velocity-prediction paradigm is noise injection. The project's next experiment should be this control, not the contrastive evaluation (which is also valuable but downstream). The contrastive TT direction is promising but unvalidated; the capability threshold is real but might be bypassable via cross-model injection. The true emergent discovery is the **trim-tab/death layer duality** itself — a phenomenon that arises from the interaction of per-layer steering with the model's internal functional organization, not predictable from any component alone.

=======================================================================

## CORE FINDINGS

| # | Finding | Confidence | Source Phase | Channel |
|---|---------|------------|-------------|---------|
| 1 | Hidden state velocities during generation are learnable with high accuracy (R²=0.85-0.94) | HIGH (9/10) | Phase 1, 6 | [theory] |
| 2 | Per-layer selectivity is non-negotiable — all-layers steering compounds death layer noise | HIGH (9/10) | Phase 2, 7 | [codebase] |
| 3 | The capability threshold (~40% GSM8K baseline) gates steering efficacy | HIGH (9/10) | Phase 2, 6 | [theory] |
| 4 | The contrastive signal (v_c - v_i) has NOT been validated — its normative value is an assumption | HIGH (unvalidated) | Phase 6, 8 | [experiment] — TEST IMMEDIATELY |
| 5 | The R²→steering quality link is unverified — prediction accuracy may not correlate with steering efficacy | MEDIUM (suspected) | Phase 6, 7, 8 | [experiment] — compute correlation |
| 6 | Adjacent layers L8 (trim-tab, +20pp) and L9 (death, -23pp) have opposite effects — this is the key mechanistic mystery | HIGH (empirical) | Phase 2, 6, 7 | [experiment] |
| 7 | Architecture (MHA vs hybrid) determines steering surface; Qwen3.5 failures are architecture-driven, not paradigm failures | HIGH (8/10) | Phase 2, 5 | [doc] |
| 8 | The trim-tab/death layer duality is a GENUINE EMERGENT property — not predictable from individual components | CONFIRMED | Phase 4b | [theory] |
| 9 | Adaptive steering policy (per-token, per-layer α via CMA-ES or RL) is the highest-value architectural upgrade | HIGH (8/10) | Phase 3, 5, 9 | [codebase] |
| 10 | The random steering control (H0-1) is the single most important experiment — it validates or invalidates the paradigm | CRITICAL | Phase 8, 10 | [experiment] — RUN THIS FIRST |

=======================================================================

## PYRAMID OVERVIEW

**Levels**: 4 (Atoms → Composites → Subsystems → System)
**Atoms**: 22 identified (A01-A22)
**Composites**: 10 identified (C01-C10)
**Subsystems**: 4 identified (S01-S04)
**Junctions**: 15 identified (J01-J15), typed as: Causal (6), Dependency (1), Modulatory (2), Constraint (3), Compositional (1), Antagonistic (1), Temporal (1)

**Key structural insight**: The central intervention point is A04 (KV modification) with in-degree 4 and out-degree 2. The most underdetermined junction is J02 (TT→velocity) → J06 (trim-tab→accuracy) — the link between prediction quality and steering efficacy is entirely uncharacterized.

=======================================================================

## EMERGENT DISCOVERIES

**CONFIRMED EMERGENT**: 3
| ID | Capability | Source |
|----|-----------|--------|
| EM-1 | Adaptive Steering Policy — meta-controller that decides WHEN and HOW to steer | CL-2 × CL-4 × FP-2 |
| EM-2 | Cross-Model Steering Injection — capability transfer without fine-tuning | FP-1 × CL-3 × SA-1 |
| EM-3 | Anti-Steering Defense — model's implicit resistance to KV modification | SA-2 × PC-2 × Domain Transpose |

**QUANTITATIVE ENHANCEMENTS**: 1 (EM-4: Steering-Regime Classifier)
**COMPOSITIONAL**: 1 (EM-5: Curriculum Steering)

**Highest Pairwise Synergy**: A08 (trim-tab) × A09 (death layer) — score: 4.9/5, QUALITATIVE. The trim-tab/death distinction itself emerges from comparing per-layer steering outcomes.

**Highest Higher-Order Synergy**: A05 × A17 × A20 × A07 (α × per-layer α × manifold × TT) — score: 4.5/5, QUALITATIVE. The adaptive steering policy emerges from the interaction of these four components.

**Self-Organization Detected**: YES. The system self-organizes into distinct steering regimes (safe, risky, death) that cross-cut layers, models, and tasks.

=======================================================================

## MASTER REGULATORS

| Rank | Regulator | Composite Score | Modulation Strategy |
|------|-----------|-----------------|---------------------|
| #1 | **Per-Layer α Allocation** (A05 × A17) | 94/100 | CMA-ES optimization, adaptive α=f(entropy) |
| #2 | **Contrastive-Normative Signal** (C05) | 88/100 | Multi-head ensemble, direct normative loss |
| #3 | **Capability Threshold Diagnostic** (A11→C08) | 82/100 | Proactive manifold separability test |
| #4 | **Death Layer Exclusion** (A09→J07) | 76/100 | Automatic classifier, negative α flipping |
| #5 | **Architecture-Aware Steering** (A19→C07) | 68/100 | Hybrid GDN-specific steering mechanisms |

=======================================================================

## TOP RECOMMENDATIONS (sorted by expected value)

### #1: Run Random Steering Control (H0-1)
- **Confidence**: 9/10 that this is the right experiment to run first
- **P(true that random steering does NOT work)**: 65% (educated guess)
- **Cost**: 30 min GPU time, trivial code change
- **Phase**: Phase A1 — IMMEDIATE
- **Risk**: LOW — if random works, paradigm invalidated (scientific progress). If not, paradigm confirmed.
- **Channel**: [experiment]

### #2: Run Contrastive TT Evaluation (H-8)
- **Confidence**: 8/10 that TTs are correctly trained and ready
- **P(true that contrastive improves over standard)**: 55% (genuinely uncertain)
- **Cost**: 2 hours GPU time, script already written
- **Phase**: Phase B1 — AFTER Phase A
- **Risk**: MEDIUM — could show no improvement
- **Channel**: [experiment]

### #3: Run Negative α Inversion on Death Layers (H0-3)
- **Confidence**: 7/10 that death layers are directional
- **P(true that L9@α=-0.1 improves)**: 40%
- **Cost**: 30 min GPU time
- **Phase**: Phase A2 — IMMEDIATE
- **Risk**: LOW — quick test
- **Channel**: [experiment]

### #4: Compute Per-Layer R² vs Δ Accuracy Correlation (H-5)
- **Confidence**: 8/10 that correlation exists (but magnitude unknown)
- **P(true that ρ > 0.5)**: 50%
- **Cost**: 15 min compute from existing data
- **Phase**: Immediate — no experiments needed
- **Risk**: VERY LOW — purely analytical
- **Channel**: [theory][codebase]

### #5: Implement Per-Layer α Optimization via CMA-ES
- **Confidence**: 7/10 that CMA-ES finds better α than uniform
- **Expected improvement**: +5-15pp over L8's +20pp
- **Cost**: ~8 GPU-hours
- **Phase**: Phase B4 — AFTER confirming steering works (H0-1 negative)
- **Risk**: MEDIUM — gradient-free optimization may not converge
- **Channel**: [codebase]

=======================================================================

## RESOURCE-BUDGETED PLAN

### Phase A — Diagnostic Foundation (<2 hours, <1GB storage)
- **A1**: Random steering control (30 min) — **DO THIS FIRST**
- **A2**: Negative α inversion (30 min)
- **A3**: α sweep on L8 (30 min)
- **A4**: Linear TT baseline (15 min)
- **Decision**: GO if A1 fails (random ≤ baseline) AND either A2 or A3 informative

### Phase B — Targeted Intervention (<1 day, <5GB storage)
- **B1**: Contrastive evaluation (2 hours)
- **B2**: Multi-layer combination (2 hours)
- **B3**: K/V split steering (1 hour)
- **B4**: CMA-ES α optimization (8 hours)
- **Decision**: GO if B1 or B4 shows improvement over baseline

### Phase C — Architectural (<1 week, <20GB storage)
- **C1**: Steering-regime classifier
- **C2**: Adaptive α = f(entropy)
- **C3**: Cross-model injection

### Phase D — Fundamental (long-term, <100GB storage)
- **D1-D4**: Multi-head ensemble, per-head steering, task generalization, RL α optimization

=======================================================================

## TESTABLE HYPOTHESES

| ID | Statement | Falsified By | Priority | Value |
|----|-----------|-------------|----------|-------|
| H-1 | Random steering ≈ TT steering | Random ≤ +5pp | 🔴 CRITICAL | Paradigm validation |
| H-2 | Negative α flips death layers | L9@α=-0.1 ≤ +5pp | 🔴 CRITICAL | Death layer remediation |
| H-3 | Steering ≤ 50 intrinsic dimensions | Small TT ≤ +5pp | 🟡 HIGH | Efficient steering |
| H-4 | K-only steering suffices | K-only ≤ +5pp | 🟡 HIGH | Simplified mechanism |
| H-5 | R² predicts steering quality | ρ < 0.2 | 🟡 HIGH | Cheap proxy metric |
| H-6 | Multi-layer improvements additive | Multi ≤ max single + 5pp | 🟡 HIGH | Multi-layer protocol |
| H-7 | All-layers-with-mask works | Masked-all ≤ best single | 🟡 HIGH | Fix all-layers failure |
| H-8 | Contrastive is normative | Contrastive ≤ standard | 🔴 CRITICAL | Normative steering |
| H-9 | Cross-model injection bypasses threshold | Small model unchanged | 🟢 MEDIUM | Capability bypass |
| H-10 | Death layer = immune response | Harm from steering itself | 🟢 MEDIUM | Paradigm shift |
| H-11 | Steering has upper bound | Any experiment exceeds bound | 🟢 MEDIUM | Expectation setting |
| H-12 | Per-head specialization | Per-head ≤ per-layer | 🟢 MEDIUM | Precise steering |

=======================================================================

## CRITICAL DISPARITIES (unresolved)

| D-ID | Disparity | Severity | Bound |
|------|-----------|----------|-------|
| **D06** | Velocity prediction accuracy (R²) ≠ steering efficacy (Δ accuracy) — the link is unverified | CRITICAL | The entire paradigm rests on this assumption. Must be resolved by computing per-layer correlation. |
| **D07** | Adjacent layers L8 (+20pp) and L9 (-23pp) have opposite effects — mechanism is unknown | CRITICAL | Must be resolved via α inversion and attention pattern analysis. |
| D01 | TT d_model=768 processes 3584-dim hidden states through 4.7× bottleneck | WARNING | Informational — may or may not be a problem. |
| D03 | Velocity predictions are in hidden-state space but applied in K/V-projected space | WARNING | Tests whether projection preserves steering direction. |
| D08 | Hybrid attention architectures resist KV-cache steering | WARNING | Not all models are comfortable. |

=======================================================================

## NEGATIVE SPACE

**What was NOT found in this analysis and why it matters**:

1. **No data scaling analysis**: The effect of trajectory count on TT quality is unknown. 25 files may be overkill or insufficient.
2. **No difficulty-stratified analysis**: Steering might only work on medium-difficulty problems (not easy ones—already correct, not hard ones—too far from model's manifold).
3. **No per-problem-type analysis**: GSM8K includes multiple problem types (arithmetic, multi-step word problems, etc.). Steering effects may vary.
4. **No calibration analysis**: What is the model's irreducible error on GSM8K? The theoretical maximum steering improvement may be bounded by this.
5. **No intrinsic trim-tab analysis**: Do layers that are trim-tabs under steering naturally contribute more to correct answers without steering?
6. **No comparison to baselines**: How does +20pp from steering compare to +20pp from 100 steps of fine-tuning? Is steering competitive with standard methods?

These gaps are addressable with existing data (no new experiments needed for #1-3, #5).

=======================================================================

## SKILL SELF-ASSESSMENT

### Analysis Weaknesses
- **Structural**: Missing atoms (infrastructure, time, seeds), some causal junctions may be correlational
- **Relational**: Heavy reliance on the project's own weak mechanistic data; Phase 8 is thin
- **Potential**: Did not explore alternative steering paradigms (e.g., weight-space steering, architecture-level changes) in enough depth

### Blind Spots Discovered
- **The researcher's constraints**: The analysis assumes unlimited researcher interest; doesn't model opportunity cost
- **Publication incentives**: The "random steering control" is scientifically critical but publication-risk-increasing
- **Cost of analysis**: ~14,000 words of analysis may exceed the marginal benefit over running the actual experiments

### Proposed TSE Updates
1. Add evidence level classification to Phase 2 findings
2. Add researcher model to Phase 9 (time, skills, preferences)
3. Add expected value computation to Phase 10 hypothesis ranking
4. Add "cost of analysis" estimator to Phase 11
5. Reorder: Phase 8 (Mechanistic Check) should precede Phase 7 (Causal Mapping)

=======================================================================

## CHANNEL SUMMARY

| Channel | Count | Key Items |
|---------|-------|-----------|
| [codebase] | 5 | Per-layer α via CMA-ES, K/V split, α=f(entropy), steering classifier, all-layers mask |
| [experiment] | 12 | H-1 through H-12 — all hypotheses need experimental validation |
| [theory] | 3 | Trim-tab/death duality as emergent, capability threshold as SNR limit, steering as adversarial perturbation |
| [doc] | 1 | Architecture compatibility documentation |

=======================================================================
