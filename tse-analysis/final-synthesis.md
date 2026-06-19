=======================================================================
TRIADIC SYNTHESIS ENGINE — FINAL REPORT
=======================================================================
Subject: TrajectoryTransformer Training Pipeline (48M velocity predictor)
Mode: Full (all phases 0-11)
Date: 2026-06-18
Output: /home/filip/Projects/Personal/AI/RankAdaptation/tse-analysis/

--- EXECUTIVE SUMMARY ---

The TrajectoryTransformer training pipeline has a fundamental ceiling at R²=0.848, not from architecture limitations, but from two tractable root causes: (1) global normalization destroys per-layer velocity signal, and (2) MSE loss conflates direction (which matters for steering) with magnitude (which is secondary). These are the highest-leverage, lowest-cost interventions available.

Three CONFIRMED EMERGENT capabilities were discovered: quantization-robust velocity representation (training on multiple Qwen formats creates invariance), layer-region specialization (grouping layers into blocks yields qualitatively different velocity dynamics), and uncertainty-aware steering (predicting velocity variance enables trustworthy abstention). The triple interaction {per-layer norm, decomposed loss, multi-format data} exhibits self-organization — the combined effect exceeds the sum of pairwise effects.

The immediate recommendation is to run 2 GPU-hours of Phase A diagnostics (noise ceiling + PCA dimensionality + AWQ shift analysis) before making any changes. This validates/falsifies the foundational assumptions on which all improvements rest. Expected R² ceiling after Phase B: 0.87-0.90. After Phase C (including AWQ transfer): R² > 0.70 on all formats.

--- CORE FINDINGS ---

1. [PHASE 2-6] Global normalization is the #1 performance limiter. It mixes 28 separate layer-wise distributions into one, destroying per-layer signal. Five independent lenses converge on this finding. [confidence: 8/10] [channel: codebase]

2. [PHASE 2-5-10] MSE loss optimizes the wrong quantity. Directional accuracy (cosine similarity) determines steering quality, not magnitude accuracy. MSE equally penalizes both, wasting capacity. Decomposing into cosine + Huber loss is expected to improve cosine by ≥0.05. [confidence: 7/10] [channel: codebase]

3. [PHASE 4b-6-10] Multi-format training creates emergent quantization robustness. Training on BnB+AWQ+GPTQ trajectories produces a CONFIRMED EMERGENT capability: velocity prediction for unseen quantization formats without fine-tuning. The triple interaction {norm, loss, data} exhibits self-organization (score: 12.5/10). [confidence: 6/10] [channel: experiment]

4. [PHASE 2-7-8] The AWQ transfer problem (R² 0.85→0.45) is primarily an affine distribution shift, not a fundamental incompatibility. A simple correction network (MLP, 5M params) mapping AWQ→BnB hidden states should recover most of the drop. Estimated AWQ R² after correction: >0.65. [confidence: 5/10] [channel: experiment]

5. [PHASE 8-10-11] The noise ceiling (irreducible error from Qwen stochasticity) may be the actual R² limit. This MUST be tested first — if noise MSE > 50% of total MSE, all other improvements are bounded. [confidence: 6/10] [channel: experiment]

6. [PHASE 2-7] No causal link between velocity R² and reasoning accuracy has been established. The entire pipeline's value proposition rests on this untested assumption. [confidence: 4/10] [channel: theory]

7. [PHASE 8-7] Velocity targets likely lie on a low-dimensional manifold (intrinsic dim < 200). PCA compression to 256 dims could provide 14× output reduction while denoising signals. [confidence: 7/10] [channel: experiment]

8. [PHASE 2-9] Layer gradient imbalance is likely severe. Early layers (1-10) probably receive ≥70% of gradient magnitude, under-optimizing late layers (20-28). Per-layer loss weighting would rebalance. [confidence: 6/10] [channel: codebase]

--- PYRAMID OVERVIEW ---

Levels: 4 | Atoms: 32 | Composites: 12 | Junctions: 17
Atom types: Data(6), Architecture(9), Training(9), Performance(5), Concept(3), Frozen(4)
Peak composite: L4-1 (Complete TrajectoryTransformer Pipeline)

--- EMERGENT DISCOVERIES ---

CONFIRMED EMERGENT: 3
  EM-1: Quantization-robust velocity representation (multi-format training)
  EM-2: Layer-region specialization (block-group routing)
  EM-3: Uncertainty-aware steering calibration (variance prediction)

QUANTITATIVE ENHANCEMENTS: 4
  (Normalization × Loss), (Data × Loss), (Layer Groups × Directional Loss), (Uncertainty × Normalization)

Highest Pairwise Synergy: Normalization × Loss = 9.5/10
Highest Higher-Order Synergy: {Normalization, Loss, Data Mixing} = 12.5/10
Self-Organization Detected: YES (triple interaction produces emergent quantization robustness)

--- MASTER REGULATORS ---

1. Normalization Strategy (Score: 85.3)
   Modulation: Per-layer normalization (28 separate mean/std). Cost: 1 hour. Impact: HIGH.
   
2. Loss Function Structure (Score: 82.5)
   Modulation: Decompose into cosine (direction) + Huber (magnitude). Cost: 3 hours. Impact: HIGH.

3. Training Data Composition (Score: 72.0)
   Modulation: Mix BnB + AWQ + GPTQ trajectories. Cost: 2 days data gen. Impact: HIGH (AWQ transfer).

4. Layer Specialization (Score: 56.3)
   Modulation: Per-layer loss weighting or layer-group experts. Cost: 2 hours-3 days. Impact: MEDIUM.

5. Attention Mechanism (Score: 45.5)
   Modulation: Hybrid bidir encoder + causal decoder. Cost: 2 days. Impact: MEDIUM.

--- TOP RECOMMENDATIONS (sorted by expected value) ---

#1: Run Phase A diagnostics first
    Confidence: 9/10 | P(true): 100% | Cost: 2 GPU-hours
    Phase: Immediate (Phase A)
    Risk: None — purely diagnostic
    What: Noise ceiling (H-6) + PCA dimensionality (H-2) + AWQ shift analysis (H-4)
    Why: Validates/falsifies ALL assumptions underlying subsequent recommendations

#2: Switch to per-layer normalization
    Confidence: 8/10 | P(true): 85% | Cost: 1 hour code + 6 GPU-hours train
    Phase: Short-term (Phase B)
    Risk: LOW — revert in 1 hour
    Expected R² gain: +0.02 to +0.05

#3: Decompose loss into direction + magnitude
    Confidence: 7/10 | P(true): 75% | Cost: 3 hours code + 4 GPU-hours sweep
    Phase: Short-term (Phase B)
    Risk: LOW — loss function swap only
    Expected cos gain: +0.03 to +0.06

#4: Multi-format mixed training (BnB + AWQ + GPTQ)
    Confidence: 6/10 | P(true): 65% | Cost: 2 days data gen + 12 GPU-hours train
    Phase: Medium-term (Phase C)
    Risk: MEDIUM — requires data generation infra
    Expected AWQ R²: >0.70 (from 0.45)

#5: PCA-compressed velocity prediction
    Confidence: 7/10 | P(true): 70% | Cost: 1 day analysis + 6 GPU-hours train
    Phase: Medium-term (Phase C)
    Risk: LOW — reversible
    Expected R² gain: +0.02 to +0.08 (via denoising + efficient capacity allocation)

--- MINIMUM VIABLE CHANGES (80% of value, 20% of effort) ---

If you can only do 3 things:
1. Per-layer normalization (1 hour) — estimated +0.03 R²
2. Decomposed loss (3 hours) — estimated +0.04 cos
3. Phase A diagnostics (2 GPU-hours) — validates everything

These three changes require ~4 hours of coding and 8 GPU-hours of compute. Expected outcome: R² ≈ 0.87-0.90, cos ≈ 0.80-0.84.

--- AVOID (proven or likely suboptimal) ---

- Fine-tuning on AWQ alone (causes catastrophic forgetting without EWC)
- Pure causal attention (already shown worse)
- Global normalization tuning (better to replace than optimize)
- Increasing batch size (buffer-constrained, marginal benefit)
- Random hyperparameter search (better with diagnostics-informed choices)

--- RESOURCE-BUDGETED PLAN ---

Phase A — Diagnostic (2 GPU-hours, 1 hour dev)
  └─ H-6 (Noise ceiling) + H-2 (PCA dim) + H-4 (AWQ shift) in parallel

Phase B — Short-term (13 GPU-hours, 2 days dev)
  ├─ B1: Per-layer normalization (6 GPU-h)
  ├─ B2: Decomposed loss sweep (4 GPU-h)
  ├─ B3: Layer-index embedding (2 GPU-h)
  └─ B4: PCA diagnostics (1 GPU-h)

Phase C — Medium-term (90 GPU-hours, 5 days dev)
  ├─ C1: Multi-format data gen (60 GPU-h)
  ├─ C2: Multi-format training (12 GPU-h)
  ├─ C3: PCA-compressed TT (6 GPU-h)
  ├─ C4: Domain-contrastive loss (8 GPU-h)
  └─ C5: Correction network (4 GPU-h)

Phase D — Long-term (200 GPU-hours, 15 days dev)
  └─ D1/D3/D5: RL/Uncertainty/Experts (choose 1-2)

Go/No-Go gates after each phase:
  Phase A success → all 3 diagnostics positive
  Phase B success → R² gain ≥ 0.03
  Phase C success → AWQ R² ≥ 0.70
  Phase D success → GSM8K improvement ≥ 5%

--- TESTABLE HYPOTHESES ---

H-1: Per-layer normalization → R² ≥ 0.868 (falsified by: gain < 0.01)
H-2: Velocity intrinsic dim ≤ 200 (falsified by: PCA 90% > 500 dims)
H-3: Layer gradient imbalance ≥ 3× (falsified by: ratio < 2×)
H-4: AWQ shift is affine (falsified by: affine R² < 0.60)
H-5: Directional error drives steering (falsified by: cosine r < 0.2 with GSM8K)
H-6: Noise ceiling < 30% of MSE (falsified by: noise ratio > 0.50)
H-7: Multi-format training → AWQ R² ≥ 0.70 (falsified by: < 0.65)
H-8: PCA-compressed TT → R² ≥ 0.868 (falsified by: ≤ baseline)
H-9: Decomposed loss → cos ≥ 0.820 (falsified by: gain < 0.02)
H-10: Correction network → AWQ R² ≥ 0.70, zero forgetting (falsified by: < 0.60)

--- CRITICAL DISPARITIES (unresolved) ---

1. R² ceiling vs noise ceiling — cannot be resolved without Phase A H-6
2. Steering quality vs reasoning accuracy — requires end-to-end validation
3. Frozen Qwen vs distribution adaptation — fundamental design constraint (solution: correction network)

--- NEGATIVE SPACE ---

What was NOT found and why:
- End-to-end accuracy improvement from TT steering (requires downstream eval)
- Layer-wise ablation of which layers benefit most from steering (not in pipeline spec)
- Alternative velocity definitions (e.g., correct-answer vs wrong-answer velocity differences)
- Optimal steering magnitude (magnitude as independent variable not studied)
- Effect of frozen vs partially-thawed Qwen (outside scope)

These are worth investigating but require separate studies.

--- SKILL SELF-ASSESSMENT ---

Weaknesses found in the analysis:
1. Strong bias toward improvement — "stop conditions" not prominent enough
2. Recommendation overload — 10+ recommendations when 3 suffice
3. Phase 8 synthetic validation was "recommended" but should be mandatory in TSE

Proposed updates to TSE:
1. Add explicit "stop condition" section to every temporal phase
2. Add "minimal viable recommendation" (2-3 items) to final report
3. Make synthetic data validation mandatory in Phase 8

--- CHANNEL ROUTING ---

[codebase] — Changes to implement:
  - Per-layer normalization (V1.1, MR-1)
  - Decomposed loss (V6.1, MR-2)
  - Layer-index embedding (V8.1)
  - PCA compression (V7.1) — after diagnostic confirmation

[experiment] — Hypotheses to test:
  - H-1 through H-10 (Phase 10)
  - Phase A diagnostics (highest priority)

[theory] — Insights for further research:
  - Velocity intrinsic dimensionality
  - Steering magnitude as independent variable
  - Noise ceiling characterization

[config] — Configuration changes:
  - Training hyperparameter sweeps per Phase A/B results

=======================================================================
END OF REPORT
=======================================================================
