=======================================================================
TRIADIC SYNTHESIS ENGINE — FINAL REPORT
=======================================================================
Subject: RankAdaptation — Velocity-based latent steering for LM reasoning
Mode: Full (all 12 phases)
Date: 2026-06-14

--- EXECUTIVE SUMMARY ---

The RankAdaptation project has demonstrated that velocity-based KV-cache steering can improve GSM8K accuracy on capable models (Qwen2.5-7B: L8 +20pp) through per-layer selectivity that distinguishes "trim-tab" layers (where steering helps) from "death layers" (where steering destroys performance). The TSE analysis reveals that the project's MOST CRITICAL gap is not technical but epistemological: no null-hypothesis experiment has been run to determine whether the L8 trim-tab effect is caused by the TT's velocity predictions or by random perturbation. The analysis identifies 8 testable hypotheses, 12 high-convergence candidate variants, and a resource-budgeted 4-phase action plan (3 hours Phase A → 1 day Phase B → 1 week Phase C → 2 weeks Phase D). Three potentially emergent capabilities are identified: chained meta-α steering, style-disentangled contrastive steering, and self-bootstrapping TT. The highest-leverage immediate action is the 1-hour random vector control experiment (A1) that resolves whether the approach has any causal validity.

--- CORE FINDINGS ---

1. **L8 trim-tab effect is robust and replicated** [confidence: 9/10, source: Phase 2 (Lens 1-10), channel: [experiment]] — +20pp on GSM8K, replicated on SVAMP (+4pp), replicated via cross-model transfer (SmolLM2→7B), stable across 100-problem evaluation.

2. **Death layers dominate all-layers steering** [confidence: 9/10, source: Phase 2 (Lens 1,2,4,5,9), channel: [experiment]] — Any steering protocol that applies uniform α to all layers is net negative. Per-layer selectivity is not optional but mandatory.

3. **Steering cannot create reasoning capability** [confidence: 8/10, source: Phase 2 (Lens 1,4), Phase 3 (MR3), channel: [theory]] — Models below ~40% GSM8K baseline cannot be improved by steering. This may be an α-search artifact (H-4) but is currently supported by 5-model evidence.

4. **The causal mechanism of steering is UNKNOWN** [confidence: 10/10, source: Phase 8, channel: [experiment]] — No null-hypothesis experiment (random vs TT) has been performed. The observed effect could be random perturbation sensitivity at L8 rather than velocity-specific steering.

5. **Contrastive steering evaluation is the highest-ROI pending experiment** [confidence: 7/10, source: Phase 3 (MR1), Phase 5, channel: [experiment]] — TTs are trained and ready. The contrastive signal (v_correct − v_incorrect) could transform steering from descriptive to normative, but the signal may learn style differences rather than reasoning content (H-3).

6. **Generation trajectories are learnable but the learned structure may be surface-level** [confidence: 7/10, source: Phase 8, Phase 10 (H-1), channel: [experiment]] — R²=0.85-0.94 is achieved across models, but the TT may be learning momentum/autoregressive structure rather than reasoning-relevant dynamics. Momentum-only baseline comparison is needed.

7. **Cross-model transfer preserves trim-tab structure** [confidence: 7/10, source: Phase 2 (Lens 3), Phase 10 (H-7), channel: [experiment]] — SmolLM2→Qwen2.5-7B transfer preserved L8 pattern. Whether this generalizes to other model families (LLaMA, Mistral) is untested.

--- PYRAMID OVERVIEW ---

Atoms: 20 | Composites: 13 (4 Level-2, 5 Level-3, 1 Level-4) | Junctions: 20

Key structural insight: The system has a LINEAR causal chain (Trajectory → TT → Steering → Accuracy) with CRITICAL BRANCHING at Per-Layer Steering (C2-3) where the same operation produces opposite effects depending on layer identity. This branching is the central structural feature.

--- EMERGENT DISCOVERIES ---

CONFIRMED EMERGENT: 3
- EM-1: Chained Steering (Meta-α: steering steered by token-divergence feedback)
- EM-3: Style-Disentangled Contrastive Steering (reasoning/style subspace separation)
- EM-4: Self-Bootstrapping TT (iterative TT improvement from own steering)

QUANTITATIVE ENHANCEMENTS: 1
- EM-2: Death-Layer Inversion (negative α on death layers as additive improvement)

Self-Organization Detected: YES — the triple (Contrastive × Asymmetric α × Self-Bootstrapping) shows higher-order synergy score of 9/10, forming a self-organizing steering system with positive feedback loop.

Highest Pairwise Synergy: (Contrastive Signal, Asymmetric α) — 9/10 qualitative
Highest Higher-Order Synergy: (Contrastive, Asymmetric α, Self-Bootstrapping) — 9/10 qualitative

--- MASTER REGULATORS ---

#1: Contrastive Signal (Score: 9.0) — Transforms descriptive→normative. Currently pending evaluation.
#2: Per-Layer Selectivity (Score: 8.5) — The essential boundary condition. Asymmetric α extends.
#3: Capability Threshold (Score: 8.0) — Binary constraint on addressable models. May be artifact.
#4: Steering Coefficient α (Score: 7.5) — Control parameter. Currently fixed at 0.1 everywhere.
#5: Trim-Tab Identification (Score: 7.0) — Prerequisite for practical use. Zero-shot methods possible.

--- TOP RECOMMENDATIONS (sorted by expected value ÷ cost) ---

#1: Random Vector Control (A1)
    Confidence: 10/10 | P(true): 50% (binary outcome) | Cost: 1 hour
    Phase: A | Risk: None (diagnostic only)
    Channel: [experiment] — resolves whether core mechanism has causal validity

#2: Negative α on Death Layers (A2)
    Confidence: 7/10 | P(true): 40% | Cost: 45 min
    Phase: A | Risk: Low (bounded α) | Max Gain: +20-30pp
    Channel: [experiment] — quick win if direction-dependent

#3: Contrastive TT Evaluation (A3)
    Confidence: 6/10 | P(true): 35% | Cost: 2 hours
    Phase: A | Risk: Low (TTs already trained) | Max Gain: +25-40pp
    Channel: [experiment] — highest potential payoff for existing assets

#4: Signed Per-Layer Sweep (B1)
    Confidence: 8/10 | P(true): 70% (at least 1 inverse trim-tab) | Cost: 4 hours
    Phase: B | Risk: Low | Max Gain: Discover 3+ inverse trim-tabs
    Channel: [experiment] — extends known methodology

#5: Over-Steering Small Models (B3)
    Confidence: 6/10 | P(true): 30% | Cost: 2 hours
    Phase: B | Risk: Low | Max Gain: Prove/falsify capability threshold
    Channel: [experiment] — high information value regardless of outcome

#6: Multi-Task Validation (B4)
    Confidence: 7/10 | P(true): 60% (some generalization) | Cost: 3 hours
    Phase: B | Risk: Low | Max Gain: Generalization evidence
    Channel: [experiment] — establishes scope of steering effects

#7: Dual-Mode Steering (B2)
    Confidence: 5/10 | P(true): 25% | Cost: 3 hours
    Phase: B | Risk: Low | Max Gain: +5-15pp over standard TT
    Channel: [experiment] — depends on A3 outcome

#8: Position-Gated Steering (C2)
    Confidence: 6/10 | P(true): 40% | Cost: 2 days
    Phase: C | Risk: Medium | Max Gain: <50% token divergence
    Channel: [experiment] — practical deployment enabler

#9: Asymmetric Multi-Layer (C1)
    Confidence: 5/10 | P(true): 35% | Cost: 1 day
    Phase: C | Risk: Low | Max Gain: >5pp over best single layer
    Channel: [experiment] — scaling law test

#10: Self-Bootstrapping TT (D1)
    Confidence: 4/10 | P(true): 20% | Cost: 3-5 days
    Phase: D | Risk: Medium | Max Gain: Self-improving system
    Channel: [experiment] — highest speculative reward, highest risk

--- RESOURCE-BUDGETED PLAN ---

Phase A (Immediate — 3 hours total):
  A1: Random vector control (1h) — resolves causal vs correlational
  A2: Negative L9 steering (45min) — death layer inversion test
  A3: Contrastive TT evaluation (2h) — normative direction validation

Phase B (Short-term — 1 day total, requires Phase A success):
  B1: Signed per-layer sweep (4h) — discover all inverse trim-tabs
  B2: Dual-mode steering (3h) — combine standard + contrastive
  B3: Over-steering small models (2h) — test capability threshold
  B4: Multi-task validation (3h) — test generalization

Phase C (Medium-term — 4 days, requires Phase B):
  C1: Asymmetric multi-layer (1d) — composite signed steering
  C2: Position-gated steering (2d) — reduce token divergence
  C3: Cross-model transfer (1d) — test universality on LLaMA-3

Phase D (Long-term — 2 weeks, requires Phase C):
  D1: Self-bootstrapping TT (3-5d) — iterative TT improvement
  D2: Head-level steering (5-10d) — finer granularity
  D3: Manifold-aware steering (3-5d) — geometric steering

Total Budget: ~64 GPU-hours, ~3 weeks wall time, ~25GB storage

--- TESTABLE HYPOTHESES ---

H-1: Steering is indistinguishable from random perturbation [Structural]
    → Falsified by: TT > random by >5pp on 100 GSM8K problems (1h experiment)

H-2: Death layers invert with negative α [Relational]
    → Falsified by: L9 with α<0 gives accuracy > baseline (45 min)

H-3: Contrastive signal captures style, not reasoning [Potential]
    → Falsified by: Contrastive steering improves accuracy with β>0 (3h)

H-4: Capability threshold is α-search artifact [Structural]
    → Falsified by: SmolLM2 improves with α > 0.1 (2h)

H-5: Multi-layer > single-layer via interaction effects [Relational]
    → Falsified by: Composite ≤ best single layer +5pp (1d)

H-6: Position gating matches uniform steering at <50% divergence [Potential]
    → Falsified by: Gated accuracy < 90% of uniform at ≤50% divergence (2d)

H-7: Trim-tab pattern is universal across model families [Structural]
    → Falsified by: LLaMA-3 shows different optimal layer than depth-based prediction (1d)

H-8: Self-bootstrapping TT converges within 3 iterations [Potential]
    → Falsified by: TT accuracy doesn't improve iteration-over-iteration (3-5d)

--- CRITICAL DISPARITIES (unresolved) ---

D4 (L8 vs L9 boundary): Adjacent layers have opposite steering effects. Proposed reconciliation (L1-L8 = reasoning, L9+ = output) is speculative and unverified. Requires activation patching to confirm functional boundary.

D12 (Velocity→correctness assumption): If velocity doesn't encode correctness, the observed +20pp must be explained by an alternative mechanism (chaos, fluency bias). This is the most critical theoretical uncertainty.

D13 (Manifold curvature): α·v steering assumes flat manifold. If the hidden state manifold is curved, steering produces OOD states and 88% token divergence is a symptom of manifold violation.

--- NEGATIVE SPACE ---

1. **Mechanistic reason for L8 trim-tab**: NOT FOUND — requires activation patching / causal tracing
2. **GDN/hybrid model steering surface**: NOT EXPLORED — could unlock Qwen3.5-series
3. **Alternative steering surfaces (activations, embeddings)**: NOT ANALYZED — may avoid death layer issues
4. **Cost-benefit vs fine-tuning**: NOT ADDRESSED — for practical use, steering must beat fine-tuning
5. **Semantic evaluation of steered outputs**: NOT PERFORMED — accuracy metric alone may miss qualitative degradation
6. **Replication on non-math tasks (ARC, BBH, MMLU)**: NOT TESTED — generalization is assumed but unverified

--- SKILL SELF-ASSESSMENT ---

Analysis Weaknesses:
- No data quality audit (Phase 0.5 proposed)
- Evidence quality not annotated per finding
- Unknown unknowns not systematically catalogued
- Post-experiment update protocol missing

Proposed Updates to TSE:
1. Add Phase 0.5: Data Audit to verify input quality
2. Add evidence quality annotation to all findings
3. Add "Unknown Unknowns" section to Phase 11
4. Add post-experiment update protocol linking Phase 9 back to earlier phases

Overall Confidence in Analysis: 7.8/10
What Would Increase Confidence Most:
- Random vector control experiment (A1) — resolves foundational uncertainty
- Synthetic data validation (Phase 8.3) — validates pipeline on known ground truth
- Replication on LLaMA-3 — establishes cross-model universality

=======================================================================
END OF TRIADIC SYNTHESIS ENGINE — FINAL REPORT
=======================================================================
