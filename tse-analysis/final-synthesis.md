=======================================================================
TRIADIC SYNTHESIS ENGINE — FINAL REPORT
=======================================================================
Subject: RankAdaptation — Velocity-based Latent Steering for Language Model Reasoning
Mode: Full (12 phases)
Date: 2026-06-14
Output Dir: tse-analysis/

--- EXECUTIVE SUMMARY ---

RankAdaptation demonstrates that LLM hidden state velocities during generation
are learnable (R²=0.85-0.94), and per-layer selective steering can amplify
correct reasoning by up to +20pp (L8 on Qwen2.5-7B). The project has progressed
through 6 phases of increasing sophistication — from failed weight-flow prediction
to successful generation-trained TrajectoryTransformers with contrastive normative
direction. However, critical gaps remain: (1) the TT's high R² may partially reflect
trivial norm-growth patterns rather than meaningful dynamics, (2) the contrastive
TT evaluation is pending and vital to the framework's validity, (3) all results use
N≤200 problems with insufficient statistical power, and (4) the fundamental question
of whether steering creates or merely amplifies capability requires the synthetic
validation that hasn't been built. The next 2 hours can resolve several of these
gaps through zero-new-infrastructure experiments (death-layer sign flip, first-step
steering removal, contrastive similarity analysis, λ interpolation sweep).

--- CORE FINDINGS ---

1. **Velocity learnability**: R²=0.85-0.94 across models [confidence: 9/10, Phase 1 A4, Phase 2 Lens 1/4/6]
   - `[experiment]` MUST validate with norm-growth baseline (H-2) — top critical gap

2. **Per-layer trim-tab/death-layer**: L8: +20pp, L2: +17pp, L9: -23pp [confidence: 7/10, Phase 1 A6/A7, Phase 2 Lens 1/2/9]
   - `[experiment]` MUST validate with N=500 (B1) and test sign-flip hypothesis (A1)

3. **Capability threshold**: Models <40% GSM8K cannot be steered [confidence: 7/10, Phase 1 A15, Phase 2 Lens 2/5/9]
   - `[experiment]` MUST separate base-model vs instruct-tuning confound (H-3)

4. **Contrastive direction**: v_c - v_i theoretically converts descriptive→normative [confidence: 5/10, Phase 1 A11, Phase 2 Lens 2/3]
   - `[experiment]` CRITICAL UNKNOWN — cosine similarity (A3) and λ interpolation (A4) needed

5. **Cross-model transfer**: SmolLM2 TT preserves L8 pattern on 7B [confidence: 6/10, Phase 2 Lens 6, Phase 4b EMPTY]
   - `[codebase]` Robust validation with more model pairs needed

6. **Architecture constraint**: Hybrid attention (GDN+FA) resists KV steering [confidence: 8/10, Phase 2 Lens 5]
   - `[codebase]` GDN recurrent state steering (RECOMB-5, B4) offers alternative

--- PYRAMID OVERVIEW ---
Levels: 5 | Atoms: 20 | Composites: 16 (4 L2 + 4 L3 + 5 L4 + 1 L5) | Junctions: 13

--- EMERGENT DISCOVERIES ---
CONFIRMED EMERGENT: 3 (per-head steering, adaptive α(t), self-supervised contrastive)
QUANTITATIVE ENHANCEMENTS: 1 (cross-task polarity)
Highest Synergy: {L8 Steering × Contrastive Direction × Confidence Gate × Adaptive α(t)} — 9.5/10
Self-Organization Detected: YES — quadruple combination enables resource-efficient reasoning amplification

--- MASTER REGULATORS ---
1. **L8 Trim-tab Layer** (Score: 84) — Modulation: KV-cache steering at α=0.05-0.3
2. **Contrastive Direction** (Score: 78) — Modulation: v_c - v_i, λ interpolation, weighted combination
3. **Per-layer α Vector** (Score: 65) — Modulation: 28 independent coefficients, Bayesian optimization
4. **First Generation Step** (Score: 60) — Modulation: Remove first_step skip, steer at t=0
5. **Layer Polarity Signature** (Score: 55) — Modulation: Per-layer sweep identifying trim-tab vs death-layer

--- TOP RECOMMENDATIONS (sorted by expected value / cost ratio) ---

**IMMEDIATE (Phase A — today, ≤2 hours)**:

#1: Death-layer sign flip (A1) — cost: 20 min, expected value: VERY HIGH
    Flip α sign on L9, test with α=-0.1. If L9 becomes a trim-tab (+20pp), it doubles the improvement.
    P(true): 40% — high-risk, high-reward. Zero new code needed.

#2: Contrastive similarity analysis (A3) — cost: 10 min, expected value: CRITICAL
    Compute cos(v_c, v_i) on 50 examples. If >0.9, contrastive approach is invalid.
    P(true): 60% (that cos > 0.9). If confirmed, saves weeks of failed contrastive experiments.

#3: Remove first-step gate (A2) + λ interpolation (A4) — cost: 40 min, expected value: HIGH
    Two trivial code changes with potential for significant accuracy gain or fundamental insight.
    P(true) for either working: 30%.

**SHORT-TERM (Phase B — 1 day)**:

#4: N=500 statistical validation (B1) — cost: 2h, expected value: HIGH
    Confirms or refutes the +20pp L8 result with 95% CI = ±4.4pp.
    P(true that +20pp holds): 70%.

#5: Norm-growth baseline (B2) — cost: 30 min, expected value: CRITICAL
    Determines whether TT learns meaningful dynamics or trivial norm patterns.
    P(true that TT > norm): 60%.

#6: Hybrid steering (v_std + β·(v_c - v_i)) (B3) — cost: 1h, expected value: HIGH
    Combines descriptive and normative steering for smooth interpolation.
    P(true that β>0 helps): 50%.

**MEDIUM-TERM (Phase C — 1 week)**:

#7: Per-head steering (C1) — cost: 2 days, expected value: VERY HIGH
    Identify the 3-5 attention heads within L8 that drive the trim-tab effect.
    P(true that head-level > layer-level): 50%.

#8: Synthetic toy transformer validation (C3) — cost: 4h, expected value: CRITICAL
    Build two-layer toy with known ground truth. If pipeline fails here, stop everything.
    P(true that pipeline works on synthetic): 80%.

**LONG-TERM (Phase D — 2 months)**:

#9: Cross-task polarity map (D1) — cost: 2 weeks, expected value: HIGH (if polarity generalizes)
#10: RL-optimized steering policy (D3) — cost: 2 weeks, expected value: VERY HIGH (if RL beats manual)

--- RESOURCE-BUDGETED PLAN ---

Phase A (Today, ≤2h):
  A1: Death sign flip (20min) → if success → expand to all death layers
  A3: Cosine similarity (10min) → if cos>0.9 → ABANDON contrastive, use standard only
  A2+A4: First-step + λ (40min) → if success → adopt as defaults
  Total: ~1.5h, 0 new code

Phase B (Day 2, ~5-6h):
  B1: N=500 validation (2h) → if effect shrinks → reassess confidence
  B2: Norm baseline (30min) → if TT > norm → confirmed meaningful
  B3: Hybrid steering (1h) → if β>0 helps → adopt hybrid
  B4: GDN steering (2h) → if works → unlock hybrid models

Phase C (Week 2-3):
  C1: Per-head steering (2 days)
  C2: Adaptive α(t) (3 days)
  C3: Synthetic validation (4h) — CRITICAL GATE

Phase D (Month 2-3):
  D1: Cross-task polarity (2 weeks)
  D2: Self-supervised contrastive (1 month)
  D3: RL policy (2 weeks)

--- TESTABLE HYPOTHESES (ranked by falsifiability × value) ---

H-1: Death-layer sign inversion (VERIFIABLE TODAY, HIGH VALUE)
  → falsified by: L9(α=-0.1) accuracy ≤ baseline
  → confirmed by: L9(α=-0.1) accuracy ≥ 65%

H-2: Velocity prediction is trivial norm-growth (VERIFIABLE TODAY, CRITICAL VALUE)
  → falsified by: norm-baseline R² < 0.5 vs TT's 0.855
  → confirmed by: norm-baseline R² ≥ 0.7

H-3: Instruct-tuning separates manifolds (VERIFIABLE WITH EXISTING MODELS, HIGH VALUE)
  → falsified by: base model similar to instruct-tuned shows trim tabs
  → confirmed by: Math-1.5B (base) shows none, 7B-Instruct shows +20pp

H-4: λ interpolation > subtraction (VERIFIABLE TODAY, MEDIUM VALUE)
  → falsified by: λ=0.5 gives best accuracy
  → confirmed by: λ=0.7-0.9 outperforms λ=1 and λ=0.5

H-5: First-step steering amplifies accuracy (VERIFIABLE TODAY, MEDIUM VALUE)
  → falsified by: first-step accuracy ≤ no-first-step
  → confirmed by: first-step accuracy > no-first-step by >5pp

H-6: Layer polarity generalizes across tasks (1-2 WEEKS, VERY HIGH VALUE)
  → falsified by: L9 improves ARC accuracy
  → confirmed by: L8 trim-tab, L9 death-layer across 3+ tasks

H-7: Residual stream steering > KV-cache (1-2 WEEKS, HIGH VALUE IF CONFIRMED)
  → falsified by: residual steering ≤ KV steering at same layer/α
  → confirmed by: residual steering > KV steering by >5pp

--- CRITICAL DISPARITIES (unresolved) ---

1. **Math-1.5B anomaly** (D4): 38% baseline but NO trim tabs with any steering mechanism.
   - Bounded by: hypothesize base-model vs instruct-tuning confound.
   - Resolution: test instruct-tuned small model.

2. **High R² ≠ Good Steering** (D1): Descriptive accuracy doesn't guarantee normative improvement.
   - Bounded by: contrastive TT theoretically resolves, but untested.
   - Resolution: cosine similarity + λ sweep.

3. **Per-layer granularity vs head-level** (D6): Layer-level analysis may miss finer structure.
   - Bounded by: no head-level access in current codebase.
   - Resolution: per-head steering (C1) — medium-term.

--- NEGATIVE SPACE ---

1. **No RL-optimized steering**: Despite Open Question 7, no RL code exists in the codebase.
2. **No per-token accuracy analysis**: All evaluations are generation-level binary (correct/incorrect).
3. **No per-token steering impact**: We don't know if L8 steering helps at all tokens or only some.
4. **No TT architecture ablation**: 6 layers, 8 heads, 768 d_model are never varied.
5. **No GDN state analysis**: The recurrent state in GatedDeltaNet might be steerable (RECOMB-5) but this is untested.

--- SKILL SELF-ASSESSMENT ---

1. Proposed updates to TSE:
   - Phase 2.5: Retrospective inconsistency check between late-lens findings and Phase 0 assumptions.
   - Phase 4b: Minimum 12 recombinations (3 per class) in full mode.
   - Phase 8: Systematic null hypothesis generator for top-3 convergence findings.

2. Analysis quality: 7.5/10
   - Comprehensive but missed some codebase depth.
   - Strong on identifying critical unknowns and actionable experiments.
   - Weak on quantitative re-analysis and literature cross-referencing.

=======================================================================
END OF REPORT
=======================================================================
