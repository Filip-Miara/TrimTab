=======================================================================
CONCEPTUAL DIFFUSER 4: OPTIMAL EXPERIMENTAL PROTOCOL
=======================================================================
Subject: RankAdaptation — Velocity-based latent steering for LM reasoning
Date: 2026-06-14
Source analyses: final-meta-synthesis.md, agent-1..5/final-synthesis.md,
                 agent-4/hyperstitional-bridge.md,
                 concept-analysis-complex-alpha/09-temporal-plan.md
=======================================================================

--- PREAMBLE ---

This document fuses ~35+ distinct proposed experiments from 6 independent
analyses into a single unified, resource-optimal protocol. Every experiment
is assigned a canonical ID, all dependencies are made explicit, redundant
experiments are merged, and the ordering maximizes information gain per
GPU-hour. The protocol is organized as an adaptive decision tree with 8
epochs, each producing publishable output regardless of outcome.

Design principles:
1. MAXIMIZE INFORMATION GAIN PER GPU-HOUR — no wasted compute
2. ADAPTIVE — results determine branching, not prior preference
3. PUBLISH AT EVERY NODE — no all-or-nothing risk
4. STATISTICALLY PRINCIPLED — multiple comparisons correction,
   power analysis, randomization throughout
5. RESOURCE-AWARE — respects ~8GB VRAM, 71GB SSD, researcher time
6. MINIMAL REDUNDANCY — every experiment tests a unique hypothesis

=======================================================================

--- 0. MASTER EXPERIMENT REGISTRY ---

All ~35+ proposed experiments from 6 source analyses are cataloged with
canonical IDs. Each entry shows: source, cost, redundancy status, and
fate in this unified protocol.

Legend:
  [MERGED] → subsumed into another experiment
  [SUBSET] → subset of a larger sweep
  [CONFLICT] → logically contradicts another proposal (resolved by DAG)
  [ADOPTED] → included as-is
  [DEFERRED] → conditional on earlier results
  [REJECTED] → insufficient value/cost ratio at this stage

| Unified ID | Source | Description | Cost | Status |
|------------|--------|-------------|------|--------|
| U00 | Meta A1 | 4-condition × 28-layer (baseline, random, stdTT, contrastiveTT) | 3.7 GPU-hrs | **ADOPTED** → Epoch 1 |
| U01 | Agent4 H-5 | R²-Δ accuracy correlation | 0 GPU-hrs | **ADOPTED** → Epoch 0 |
| U02 | Agent4 H-11 | Steering upper bound meta-analysis | 0 GPU-hrs | **ADOPTED** → Epoch 0 |
| U03 | Meta A2 | Velocity-norm distribution | 0 GPU-hrs | **ADOPTED** → Epoch 0 |
| U04 | Meta A2 | Intrinsic dimensionality (PCA) | 0 GPU-hrs | **ADOPTED** → Epoch 0 |
| U05 | Agent4 H-5 | Per-layer R² computation | 0 GPU-hrs | [MERGED] into U01 |
| U06 | Agent4 H-3 | Low-rank TT vs full TT | 1 GPU-hr | [DEFERRED] → Epoch 4 |
| U07 | Agent1 A3 | α sweep on L8 [0.001-2.0] | 1 GPU-hr | [SUBSET] of U08 |
| U08 | Meta B1 | Signed α sweep (8α × 28 layers) | 4 GPU-hrs | **ADOPTED** → Epoch 3 |
| U09 | Agent1 A2 | Anti-steering at death layers (flip sign) | 0.5 GPU-hrs | [SUBSET] of U08 |
| U10 | Agent2 A2 | Asymmetric α sweep (+/-) | 0.5 GPU-hrs | [SUBSET] of U08 |
| U11 | Agent5 A2 | Negative α on death layers | 1 GPU-hr | [SUBSET] of U08 |
| U12 | Agent2 A3 | L8 ablation (keystone) test | 0.3 GPU-hrs | [ADOPTED] → Epoch 2 |
| U13 | Agent5 A1 | Null hypothesis significance test | 2 GPU-hrs | [MERGED] into U00 |
| U14 | Agent2 A1 | Contrastive evaluation | 2 GPU-hrs | [MERGED] into U00 |
| U15 | Agent1 A1 | Contrastive evaluation | 0.75 GPU-hrs | [MERGED] into U00 |
| U16 | Agent4 H-8 | Contrastive on neutral layers | — | [MERGED] into U00 |
| U17 | Agent5 B2 | Contrastive TT evaluation | 1 GPU-hr | [MERGED] into U00 |
| U18 | Agent4 H-1 | Steering-random equivalence | 0.5 GPU-hrs | [MERGED] into U00 |
| U19 | Agent1 B1 | Random direction baseline | 1 GPU-hr | [MERGED] into U00 |
| U20 | Agent5 B1 | Random steering baseline | — | [MERGED] into U00 |
| U21 | Meta A3 | TT dissection: position shuffle (E1) | 0.5 GPU-hrs | **ADOPTED** → Epoch 2 |
| U22 | Meta A3 | TT dissection: token ablation (E2) | 0.5 GPU-hrs | **ADOPTED** → Epoch 2 |
| U23 | Meta A3 | TT dissection: naive baseline (E3) | 0.5 GPU-hrs | **ADOPTED** → Epoch 2 |
| U24 | Meta A3 | TT dissection: per-layer R² (E4) | 0.5 GPU-hrs | [MERGED] into U01 |
| U25 | Agent5 A4 | TT internal representation | 2 GPU-hrs | [MERGED] into U21+U22 |
| U26 | Agent2 B1 | TT null model comparison | 6 GPU-hrs | [MERGED] into U21+U22+U23 |
| U27 | Agent2 B3 | Attention pattern analysis (K/V amplification) | 2 GPU-hrs | **ADOPTED** → Epoch 4 |
| U28 | Agent4 H-4 | K-value dominance (K-only, V-only) | 1 GPU-hr | [ADOPTED] → Epoch 4 |
| U29 | Meta B4 | K/V split steering | 1 GPU-hr | [MERGED] with U28 |
| U30 | Meta B2 | Multi-layer combination (L2+L8, L8+L10, L8+L9−α, L0+L2+L8+L10) | 3 GPU-hrs | **ADOPTED** → Epoch 4 |
| U31 | Agent1 B2 | Multi-layer pair (L2+L8, L8+L10, L8+L9) | 2 GPU-hrs | [SUBSET] of U30 |
| U32 | Agent4 H-6 | Multi-layer additivity | — | [MERGED] into U30 |
| U33 | Agent4 H-7 | All-layers-with-mask rescue | — | [SUBSET] of U30 |
| U34 | Agent5 H-2 | L8(+α) + L9(−α) combo | — | [MERGED] into U30 |
| U35 | Meta B3 | Best layer × 500 problems | 2 GPU-hrs | **ADOPTED** → Epoch 4 |
| U36 | Agent1 B3 | High α on Math-1.5B | 2 GPU-hrs | [DEFERRED] → Epoch 5 |
| U37 | Meta B5 | Cross-task (SVAMP, ARC subset) | 2 GPU-hrs | **ADOPTED** → Epoch 5 |
| U38 | Agent1 B4 | Early-only steering | 1 GPU-hr | **ADOPTED** → Epoch 5 |
| U39 | Agent2 B2 | Frequency-domain PCA analysis | 2 GPU-hrs | **ADOPTED** → Epoch 5 |
| U40 | Agent4 H-2 | Directional death layer | 0.5 GPU-hrs | [MERGED] into U08 |
| U41 | Agent5 B4 | Oscillating α | — | [DEFERRED] → Epoch 6 |
| U42 | Agent5 C | Multi-scale TT | — | [DEFERRED] → Epoch 6 |
| U43 | Agent4 H-12 | Per-head specialization | 10 GPU-hrs | [DEFERRED] → Epoch 6 |
| U44 | Meta C5 | Per-head steering | 10 GPU-hrs | [MERGED] with U43 |
| U45 | Agent4 H-9 | Cross-model injection | — | [DEFERRED] → Epoch 6 |
| U46 | Meta C4 | Cross-model transfer (LLaMA-3, Mistral) | 12 GPU-hrs | [DEFERRED] → Epoch 6 |
| U47 | Agent4 H-10 | Death layer = immune response (observational) | 0 GPU-hrs | [DEFERRED] → Epoch 5 |
| U48 | Meta C1 | Combined std + contrastive (β sweep) | 6 GPU-hrs | [DEFERRED] → Epoch 6 |
| U49 | Meta C2 | Siamese contrastive TT | 8 GPU-hrs | [DEFERRED] → Epoch 6 |
| U50 | Meta C3 | Per-position α | 4 GPU-hrs | [DEFERRED] → Epoch 6 |
| U51 | Agent1 C1 | Siamese contrastive TT (triplet loss) | 4 GPU-hrs | [MERGED] with U49 |
| U52 | Agent1 C2 | Bayesian α optimization | 8 GPU-hrs | [DEFERRED] → Epoch 7 |
| U53 | Agent1 C3 | Cross-dataset (ARC, BBH, MMLU) | 6 GPU-hrs | [MERGED] with U37+U46 |
| U54 | Agent1 C4 | Death layer inoculation training | 4 GPU-hrs | [DEFERRED] → Epoch 7 |
| U55 | Agent2 C1 | RL-based per-token α | 8 GPU-hrs | [DEFERRED] → Epoch 7 |
| U56 | Agent2 C2 | Dual-surface steering | 12 GPU-hrs | [DEFERRED] → Epoch 7 |
| U57 | Agent2 C3 | Multi-head contrastive ensemble | 8 GPU-hrs | [DEFERRED] → Epoch 7 |
| U58 | Agent1 D1 | Self-correcting steering loop | — | [DEFERRED] → Epoch 7 |
| U59 | Meta D1 | Self-bootstrapping TT loop | 15 GPU-hrs | [MERGED] with U58 |
| U60 | Meta D2 | RL-based per-token α optimization | 20 GPU-hrs | [MERGED] with U55 |
| U61 | Meta D3 | Dual-surface steering | 25 GPU-hrs | [MERGED] with U56 |
| U62 | Meta D4 | Multi-head contrastive ensemble | 15 GPU-hrs | [MERGED] with U57 |
| U63 | Complex A-C1 | Acceleration structure (R²_a) | 0.5 GPU-hrs | **ADOPTED** → Epoch 2a |
| U64 | Complex A-C2 | L8 phase sweep (θ ∈ {0, π/6, π/4, π/3, π/2}) | 1 GPU-hr | **ADOPTED** → Epoch 2a |
| U65 | Complex A-C3 | L9 phase sweep | 1 GPU-hr | [CONDITIONAL] on U64 |
| U66 | Complex B-C1 | Full 28-layer phase sweep (4 phases) | 4 GPU-hrs | [DEFERRED] → Epoch 5 |
| U67 | Complex B-C2 | Complex + contrastive cross-pollination | — | [DEFERRED] → Epoch 6 |
| U68 | Complex B-C3 | Phase-aware α vector | — | [DEFERRED] → Epoch 7 |
| U69 | Complex C | Synthetic toy transformer validation | — | [DEFERRED] → Epoch 7 |
| U70 | Agent1 D2 | Universal velocity manifold verification | — | [DEFERRED] → Epoch 7 |
| U71 | Agent1 D3 | RL-based α optimization | — | [DEFERRED] → Epoch 7 |
| U72 | Agent2 D1 | Non-math task evaluation (ARC, BBH, MMLU) | — | [MERGED] with U53 |
| U73 | Agent2 D2 | Automated trim-tab discovery | — | [DEFERRED] → Epoch 8 |
| U74 | Agent2 D3 | Self-adaptive steering with system ID | — | [DEFERRED] → Epoch 8 |
| U75 | Agent4 H-9 | Cross-model injection (capability bypass) | — | [DEFERRED] → Epoch 6 |
| U76 | Agent5 D | Self-improving loop | ~75 GPU-hrs | [DEFERRED] → Epoch 8 |

=======================================================================

--- 1. DEPENDENCY DAG ---

The logical dependency structure between experimental questions:

```
DEPTH 0 (no dependencies):
  [U01] R²-Δ accuracy correlation         [U02] Upper bound meta-analysis
  [U03] Velocity-norm distribution         [U04] Intrinsic dimensionality

DEPTH 1 (depends on U00 results):
  [U00] 4-condition × 28-layer protocol ---------->| ALL downstream
      |
      ├── [U12] L8 ablation (keystone test)
      ├── [U21] TT dissection: position shuffle
      ├── [U22] TT dissection: token ablation
      ├── [U23] TT dissection: naive baseline
      ├── [U63] Acceleration structure (complex α)
      └── [U64] L8 phase sweep (complex α)

DEPTH 2 (depends on U00 + Depth-1 results):
  [U08] Signed α sweep (8α × 28 layers)
  [U27] Attention pattern analysis
  [U28] K/V split steering
  [U30] Multi-layer combination
  [U35] Best layer × 500 problems (power analysis)
  [U36] High α on Math-1.5B (capability threshold)

DEPTH 3 (depends on Depth-2 results):
  [U37] Cross-task evaluation
  [U38] Early-only steering
  [U39] Frequency-domain PCA
  [U65] L9 phase sweep (complex α)

DEPTH 4 (depends on Depth-3):
  U41-U76 (architectural, long-term)

CRITICAL INSIGHT: The parent meta-analysis identified U00 as the
foundational experiment. This protocol confirms that U00 resolves
7 hypotheses simultaneously and feeds EVERY downstream experiment.
No experiment before U00 produces more information per GPU-hour.
```

=======================================================================

--- 2. STATISTICAL FRAMEWORK ---

Applied to ALL experiments. Every reported result must include these.

2.1 Multiple Comparisons Correction

The project tests 28 layers × multiple conditions. The corrected
significance thresholds:

| Test Family | N_tests | Bonferroni α | Required z-score | Required Δ accuracy (at σ=5pp) |
|-------------|---------|--------------|-------------------|-------------------------------|
| Single cond × 28 layers | 28 | 0.00179 | 2.91σ | 14.5pp |
| 4 cond × 28 layers (U00) | 112 | 0.000446 | 3.33σ | 16.7pp |
| 8α × 28 layers (U08) | 224 | 0.000223 | 3.51σ | 17.5pp |
| All Epoch 1-4 | ~400 | 0.000125 | 3.66σ | 18.3pp |

L8 current result: +20pp at σ=4.5pp → z=4.44 → survives all thresholds.
L9 current result: -23pp at σ=4.5pp → z=5.11 → survives all thresholds.

BUT: These are RETROSPECTIVE. Prospective thresholds for new experiments
must use the test-family-adjusted threshold.

Implementation: report both raw p-values and Bonferroni-corrected q-values.
For all claims in publication: use q < 0.05 (FDR, not FWER) as the
discovery threshold, with Bonferroni as the conservative bound.

2.2 Power Analysis

| Experiment | Effect size (Δ) | σ (estimated) | N problems | Power (β=0.8) |
|------------|-----------------|---------------|------------|----------------|
| U00 (4-cond sweep) | 10pp | 4.5pp | 100 | 0.85 |
| U00 critical test | 5pp | 4.5pp | 100 | 0.35 |
| U35 (precision eval) | 5pp | 2.0pp | 500 | 0.95 |
| U08 (α sweep) | 15pp | 5.0pp | 100 | 0.72 |

Action: U00 uses 100 problems (sufficient for 10pp effects, underpowered
for 5pp effects). If U00 results are suggestive (5-10pp range), proceed
to U35 for precision estimation before making publishable claims.

2.3 Randomization Protocol

For ALL experiments:
1. Fix random seed per condition per layer (reproducibility)
2. Shuffle problem order per condition (avoid ordering bias)
3. Use identical temperature (T=0, greedy) for all conditions
4. Interleave conditions within each layer run (not run all baseline,
   then all steering — avoids time-of-day effects)
5. For U00 specifically: randomize the 4 conditions within each layer;
   randomize layer order (not L0→L27 sequential)

2.4 Bootstrap Confidence Intervals

All accuracy measurements report:
- Point estimate (mean over N problems)
- 95% CI (percentile bootstrap, 10,000 resamples)
- Bonferroni-adjusted CI for family-wise inference

2.5 Variance Decomposition

Report σ² components:
- Between-problem variance (different problems have different difficulty)
- Between-generation variance (same problem, different seeds)
- Between-layer variance (if multiple layers)

This enables principled sample-size planning for all downstream exps.

=======================================================================

--- 3. EPOCH 0: ZERO-COMPUTE ANALYSIS (TODAY, 30 min) ---

Run while GPU is idle or before Epoch 1. Requires only existing
trajectory data and TT predictions. No new model evaluations needed.

3.1 [U01] R²-Δ Accuracy Correlation

Compute per-layer R² (TT prediction accuracy on hold-out validation data)
and correlate with per-layer Δ accuracy from existing L0-L27 sweep.
Report Pearson ρ and 95% CI.
- If ρ > 0.4: R² is a cheap proxy for steering efficacy — use for
  preliminary layer screening in downstream experiments.
- If ρ < 0.2: R² does NOT predict steering quality — theoretical
  foundation requires revision (consistent with R² paradox).

3.2 [U02] Steering Upper Bound Estimation

For each layer, compute: accuracy_with_steering, baseline_accuracy.
Empirical upper bound: max accuracy achieved across all existing runs.
Theoretical bound: B + (1-B) × 0.75 (Agent 4 H-11).
Test whether ANY experiment has exceeded this bound.
- If yes: bound is falsified, revise model.
- If no: bound is plausible, use for expectation setting.

3.3 [U03] Velocity-Norm Distribution

Compute |v| per layer across all tokens and problems.
This determines whether α=0.1 is a reasonable global scaling.
- If |v| varies by >10× across layers: α must be layer-normalized.
- If |v| is approximately uniform: global α is defensible.
Report: mean|v|, std|v|, min|v|, max|v| per layer.

3.4 [U04] Intrinsic Dimensionality (PCA)

Run PCA on held-out hidden states (1000 random states from existing
trajectories). Report number of components needed to explain 90% variance.
- If dim_90 < 100: steering manifold is low-dimensional → low-rank TT (U06)
  is promising.
- If dim_90 > 500: manifold is high-dimensional → low-rank TT unlikely
  to work.

3.5 Publishability Gate

Output: Brief technical note (2-3 pages) correlating R², velocity norms,
and dimensionality with existing per-layer accuracy results.
- Minimum publishable unit: "Empirical characterization of the hidden
  state velocity manifold in Qwen2.5-7B-Instruct"
- Can be published as a standalone result OR as supplementary material
  for the main findings.

=======================================================================

--- 4. EPOCH 1: FOUNDATIONAL VALIDATION (3.7 GPU-hrs, ~4 hrs wall) ---

THE critical experiment. All downstream work depends on this.

4.1 [U00] 4-Condition × 28-Layer Sweep

Conditions per layer:
  C0: No steering (re-run baseline for controlled comparison)
  C1: Random vector (same norm as TT prediction, resampled per layer)
  C2: Standard TT prediction (v_pred)
  C3: Contrastive TT (v_correct − v_incorrect)

Implementation:
  - Uses existing `run_per_layer_sweep.py` with 4 config variants
  - ~20 lines new code for random vector generation
  - 28 layers × 4 conditions × 100 problems = 11,200 evaluations
  - At ~20 sec/eval: 3.7 GPU-hours (parallelizable to ~2 hrs wall time)

4.2 Primary Analyses (pre-registered)

A1: Is any steering condition better than baseline?
    H0: max(C1, C2, C3) ≤ baseline for all layers
    H1: at least one condition > baseline for at least one layer
    Test: per-layer z-test, Bonferroni across 28 layers × 3 conditions = 84
    Decision threshold: q < 0.05 (FDR)

A2: Is TT direction specific?
    H0: C2 ≤ C1 for all layers (random matches TT)
    H1: C2 > C1 for at least one layer
    Test: paired comparison C2-C1 per layer
    Interpretation: If H0 not rejected → paradigm invalidated. Steering
    is noise injection/smoothness exploitation.

A3: Is contrastive better than standard?
    H0: C3 ≤ C2 for all layers
    H1: C3 > C2 for at least one layer
    Test: paired comparison C3-C2 per layer
    Interpretation: If H1 confirmed → normative steering works. Major
    result. If H0 not rejected → contrastive direction adds nothing.

A4: Does the trim-tab/death layer pattern replicate?
    For each layer: classify as trim-tab (C2 > baseline), death layer
    (C2 < baseline), neutral (C2 ≈ baseline)
    Compare classification to existing sweep results.

4.3 Secondary Analyses (exploratory, flagged as such)

S1: Which layers show random > baseline? (identifies layers where ANY
    perturbation helps — suggests noise-tolerance, not steering)
S2: Per-layer correlation between C1, C2, C3 effects
    (high correlation → all perturbation methods are equivalent)
S3: Effect of layer order on performance (if layer order was randomized,
    test for order effects)

4.4 Decision Tree

```
U00 results:

  ├── C2 > C1 ≥ baseline for ≥2 layers (TT direction matters)
  │     └── Proceed to Epoch 2 (mechanistic probing)
  │
  ├── C2 ≈ C1 > baseline for ≥2 layers (direction doesn't matter,
  │     but perturbation does)
  │     └── Proceed to Epoch 2, but with reduced scope.
  │         Publish: "KV-cache perturbation improves LM reasoning
  │         regardless of steering direction" — weaker claim but
  │         still publishable (novel finding).
  │
  ├── C1 ≈ C2 ≈ baseline for all layers (no steering effect)
  │     └── STOP. Publish negative result.
  │         "Velocity-based KV-cache steering does not improve
  │         reasoning accuracy on GSM8K." Value: saves field from
  │         pursuing dead end. Replication crisis prevention.
  │
  └── C3 > C2 for ≥1 layer (contrastive works)
        └── ACCELERATED path: proceed to Epoch 3 directly
            (skip Epoch 2 mechanistic probing; contrastive success
             implies direction specificity is already established)
```

4.5 Publishability Gate

Regardless of outcome, Epoch 1 produces a complete publication:
- Title: "Per-layer KV-cache steering for language model reasoning:
  A controlled comparison of velocity prediction, contrastive steering,
  and random perturbation"
- Core result: For each of 28 layers, 4-condition accuracy comparison
- If positive: "TT direction matters; contrastive steering works"
- If negative: "All steering conditions indistinguishable from each other"
- If null: "No steering condition improves accuracy over baseline"
- ALL three outcomes are publishable. The experiment is designed so
  there is no "bad" outcome — only informative ones.

=======================================================================

--- 5. EPOCH 2: MECHANISTIC PROBING (2.5 GPU-hrs, ~3 hrs wall) ---

Conditional on U00 showing C2 > C1 for ≥2 layers. Run after Epoch 1.

5.1 [U12] L8 Ablation (Keystone Test) — 0.3 GPU-hrs

Test whether L8 steering is necessary for the overall effect.
Method: Run full generation with steering at ALL layers EXCEPT L8
(α=0 at L8, standard α elsewhere).
Compare accuracy to all-layers steering and L8-only steering.
- If all-except-L8 ≈ baseline → L8 is a keystone layer.
- If all-except-L8 ≈ all-layers → L8 is expendable.

5.2 [U21, U22, U23] TT Dissection — 1.5 GPU-hrs total

Three experiments that open the TT black box:

E1: Position Shuffle (0.5 GPU-hr)
    Shuffle token positions in velocity prediction input.
    Compare R²(shuffled) to R²(original).
    - If R²(shuffled) ≈ R²(original): TT learns frequency/content
      patterns, NOT causal dynamics.
    - If R²(shuffled) ≪ R²(original): TT learns positional dynamics.
      Publish: "TT learns causal token dynamics, not surface statistics."

E2: Token Ablation (0.5 GPU-hr)
    Zero out specific token embeddings, measure effect on velocity
    prediction at those positions.
    Tests whether TT relies on token identity or hidden state patterns.
    - If ablation drops R² significantly at ablated positions: TT
      uses token identity (weaker finding).
    - If R² unchanged: TT uses context-independent dynamics (stronger).

E3: Naive Baseline (0.5 GPU-hr)
    v̂ = 0 (predict h_{t+1} = h_t). Compare to TT prediction accuracy.
    - If TT R² - naive R² < 0.1: TT is barely better than doing nothing.
      Steering with TT ≈ steering with zero vector (no effect).
    - If TT R² - naive R² > 0.3: TT genuinely learns dynamics.

5.3 [U63-U64] Complex α Phase Sweep — 1.5 GPU-hrs
(Conditional on U00 showing steering is real; Epoch 2a if complex α gate
passes, otherwise skip)

U63: Acceleration Structure (0.5 GPU-hr)
    Compute a[l] = h[l+1] - 2h[l] + h[l-1] from existing trajectories.
    Train TT_a on acceleration. Measure R²_a.
    Gate: If R²_a > 0.3 → acceleration is learnable. Proceed to U64.
    If R²_a < 0.3 → complex α concept is not viable at current resolution.
    Report: "Acceleration is noise — velocity-only steering is sufficient."

U64: L8 Phase Sweep (1 GPU-hr, conditional on U63)
    h' = h + r·(cosθ·v + sinθ·a) with r=0.1, θ ∈ {0, π/6, π/4, π/3, π/2}
    - If θ_opt = 0: pure velocity is optimal at L8. Complex α not useful.
    - If θ_opt ≠ 0: phase matters. Trim-tab layers respond to velocity
      + acceleration mixtures.
    Publish: "Steering optimal phase differs from pure velocity at L8"

5.4 Publishability Gate

Epoch 2 produces:
- "TT internal representation analysis for KV-cache velocity steering"
  (position shuffle + token ablation + naive baseline)
- "Keystone layer hypothesis: L8 ablation study"
- Optional: "Complex steering: acceleration structure in hidden state
  dynamics" (if U63 gate passes)

=======================================================================

--- 6. EPOCH 3: SIGNED α MAPPING (4 GPU-hrs, ~5 hrs wall) ---

Conditional on Epoch 1 showing positive steering. Highest-value
map-making experiment.

6.1 [U08] Full Signed α Sweep

α ∈ {-2.0, -1.0, -0.5, -0.1, 0, 0.1, 0.5, 1.0, 2.0}
All 28 layers × 9 α values × 100 problems = 25,200 evaluations
~4 GPU-hours (parallelizable to ~3 hrs wall time)

Pre-registered hypotheses:
  H_α1: At least one layer has optimal |α| ≠ 0.1
  H_α2: At least one death layer (C2 < baseline at α=0.1) becomes a
         trim-tab (C2 > baseline) at α < 0
  H_α3: The trim-tab/death classification reverses for some layer(s)
         when sign is flipped
  H_α4: The optimal α for L8 is NOT 0.1

Statistical approach:
  - For each layer, fit α-accuracy function (quadratic or spline)
  - Report α_opt per layer with 95% CI
  - Multiple comparisons: Bonferroni across 28 layers × 9 α = 252 tests
    Threshold: q < 0.05 (FDR)

Expected outputs:
  - Per-layer steering response curve: accuracy = f(α)
  - Optimal α vector: α_opt[l] for each layer
  - Layer classification: trim-tab (positive optimum), death (negative
    optimum), resilient (high magnitude optimum), fragile (low magnitude)
  - Best single-layer accuracy (potentially >> L8 +20pp if α was wrong)

6.2 Secondary Analysis: α Inversion Map

Compute for each layer: sign(optimal α).
If layer has sign opposite to current α=0.1, it's an "invertible"
layer — its performance improves when α sign is flipped.

Publishable finding: "Layer-specific sign inversion in KV-cache steering:
most death layers are trim-tabs with wrong steering sign"

6.3 Decision Tree

```
U08 results:

  ├── H_α1 AND H_α2 confirmed (α ≠ 0.1 matters; inversion exists)
  │     └── Proceed to Epoch 4 (multi-layer + precision)
  │
  ├── α=0.1 is near-optimal for all layers (H_α1 not confirmed)
  │     └── Publish α sweep as negative result, then decide:
  │         Either proceed to Epoch 4 with reduced scope, or
  │         publish current findings + move to Epoch 5 (generalization)
  │
  └── Multiple layers show α-accuracy cliff (sharp drop at |α| > 0.5)
        └── Publish robustness finding, restrict α range for all
            downstream experiments
```

6.4 Publishability Gate

Epoch 3 produces: "Per-layer signed α mapping for KV-cache steering:
discovering the steering response function" — a complete map of
how accuracy depends on steering strength and direction per layer.
This is a standalone publication regardless of what α values are optimal.

=======================================================================

--- 7. EPOCH 4: MECHANISM + PRECISION (6 GPU-hrs, ~1 day wall) ---

Conditional on Epoch 3 showing α inversion or direction specificity.

7.1 [U27] Attention Pattern Analysis — 2 GPU-hrs

Capture attention distributions during steered vs unsteered generation
at the best layer (identified by U08).
Compare: attention weights with and without steering.
- If attention shifts measurably (Δ > 5% of total mass): K/V
  amplification hypothesis is supported.
- If attention is unchanged: steering works through a different mechanism
  (manifold pushing, not attention modulation).

Publishable: "Attention dynamics under KV-cache steering: evidence for
K/V amplification" or "Steering without attention modulation"

7.2 [U28] K/V Split Steering — 1 GPU-hr

At the best layer (from U08), compare:
  C_K: steering KEY projections only
  C_V: steering VALUE projections only
  C_KV: steering both (standard)
- If C_K ≈ C_KV: K-only is sufficient → halve compute for future.
- If C_K ≈ C_V ≈ baseline: both are necessary → mechanism is joint.
- If C_K > C_KV: K-only outperforms joint (interesting but unlikely).

7.3 [U30] Multi-Layer Combination — 3 GPU-hrs

Test combinations at optimal α per layer (from U08):
  L8 alone (best trim-tab)
  L9 with inverted α (best death → trim-tab candidate)
  L8 + L9(−α) (trim-tab + inverted death layer — the highest-upside combo)
  L2 + L8 (early + mid trim-tab)
  L0 + L2 + L8 + L10 (all positive layers)
  All layers with death-layer mask (α=0 at death layers)

Pre-registered: Multi-layer additivity hypothesis (H-6 from Agent 4).
- If multi-layer ≈ sum(individual): effects are additive → simple model.
- If multi-layer > sum(individual): synergy → publish "multi-layer synergy
  exceeds individual contributions."
- If multi-layer < max(individual): interference → steering is
  competitive, not cooperative.

Breakthrough scenario: L8 + L9(−α) achieves > +40pp (target: from 73%
to > 93% on GSM8K). This would be a landmark result.

7.4 [U35] Precision Eval at Best Configuration — 2 GPU-hrs

Take the best {layer, α, condition} from U00+U08+U30.
Run 500 problems (up from 100) at this configuration.
Expected: σ drops from ~4.5pp to ~2.0pp.
Purpose: Provide the precise effect size needed for publication.
Report: accuracy at 500 problems with 95% CI.

7.5 Decision Tree

```
Epoch 4 results:

  ├── Best config > baseline + 30pp (e.g., L8+L9(−α) > 95%)
  │     └── PRIORITY SHIFT: Publish immediately. This is the headline
  │         result. Do NOT wait for Epochs 5-8.
  │
  ├── Best config > baseline + 15-30pp
  │     └── Proceed to Epoch 5 (generalization + cross-task)
  │
  ├── Best config > baseline + 5-15pp
  │     └── Proceed to Epoch 5, but focus on robustness (more problems,
  │         more datasets) rather than magnitude improvement.
  │
  └── Best config < baseline + 5pp
        └── Proceed to Epoch 5 for cross-validation, then publish
            comprehensive negative result.
```

7.6 Publishability Gate

Epoch 4 produces:
- "Mechanism of KV-cache steering: attention dynamics and K/V split
  analysis"
- "Multi-layer synergistic steering: combining trim-tab layers"
- "Precise effect size estimation: steering accuracy at 500 problems"
Combined with Epochs 0-3, the researcher now has 4-5 publication-quality
findings, regardless of whether the overall effect is positive or negative.

=======================================================================

--- 8. EPOCH 5: GENERALIZATION + ROBUSTNESS (7 GPU-hrs, ~2 days wall) ---

Conditional on Epochs 1-4 showing positive steering. Tests how far the
findings generalize.

8.1 [U37] Cross-Task Evaluation — 2 GPU-hrs

At the optimal {layer, α} from U08+U30:
  SVAMP (math, already set up)
  ARC-Easy (science MC, ~200 problems subset)
  BBH subset (3 selected tasks: date understanding, sports understanding,
    tracking shuffled objects)
- If steering improves ALL tasks: general reasoning improvement — major.
- If steering only improves math: domain-specific — publish as such.

8.2 [U38] Early-Only Steering — 1 GPU-hr

Steer only the first K tokens (K ∈ {1, 2, 4, 8, 16}) of generation at
the optimal {layer, α}. Tests whether early steering is sufficient.
- If early-only ≈ full-generation: steering sets trajectory then
  auto-completes — important mechanistic insight.
- If early-only ≪ full-generation: steering must persist throughout.

8.3 [U39] Frequency-Domain PCA — 2 GPU-hrs

PCA on TT velocity predictions across all layers.
Decompose velocity signal into frequency components.
Test: do trim-tab layers cluster in a specific frequency regime?
- If yes: "frequency-specific steering" (Agent 2 EM-2) — layers process
  different frequency components.
- If no: velocity content is uniform across layers.

8.4 [U36] Capability Threshold Probe — 2 GPU-hrs

Run optimal {layer, α} from U08 on Math-1.5B (sub-threshold model).
Test: Does high α (> 0.1) reveal trim-tabs on a model below 40%?
- If yes: capability threshold is α-dependent (Agent 1 D4 resolution).
- If no: capability threshold is fundamental — steering cannot create
  capability.

8.5 [U66] Full Phase Sweep — 4 GPU-hrs (complex α, conditional on U64)

If U64 (L8 phase sweep) showed θ_opt ≠ 0, run full 28-layer sweep
with θ ∈ {0, π/4, π/2, π} at 4 phases.
- Maps which layers benefit from which phase.
- If trim-tab/death classification aligns with optimal phase: major
  mechanistic insight ("death layers are π-phase shifted trim-tabs").

8.6 [U47] Death Layer Immune Response — 0 GPU-hrs (observational)

Compare attention patterns on death layers between:
  (a) death layer steered, other layers unsteered
  (b) death layer UNsteered, other layers steered
If pattern (a) shows attention compensation not present in (b): death
layers are an active anti-steering response (Agent 4 H-10).

8.7 Publishability Gate

Epoch 5 produces:
- "Cross-task generalization of KV-cache steering: from math to science
  and reasoning"
- "The capability threshold in KV-cache steering: α-dependent or
  fundamental?"
- "Frequency-domain analysis of steering velocity signals"
These are standalone publications that contextualize the main findings.

=======================================================================

--- 9. EPOCH 6: ADVANCED STEERING (20-30 GPU-hrs, ~1 week) ---

Conditional on Epochs 1-5 showing robust, generalizable steering.
Requires code base changes (new infrastructure).

9.1 [U48] Combined Standard + Contrastive Steering — 6 GPU-hrs

β coefficient sweep: h' = h + (β·v_std + (1-β)·v_contrastive) · α
β ∈ {0, 0.25, 0.5, 0.75, 1}
Tests whether the two steering signals are complementary or redundant.

9.2 [U49] Siamese Contrastive TT — 8 GPU-hrs

Train a single TT with contrastive loss (triplet: anchor vs positive vs
negative trajectories). Compare to dual-TT approach.
- If siamese ≥ dual-TT: simpler architecture suffices — drop dual-TT.
- If siamese < dual-TT: both velocities are needed (architecture matters).

9.3 [U50] Per-Position α — 4 GPU-hrs

α = f(token_position). Test whether early tokens need different α than
late tokens. At optimal layer from U08.
- If position matters: position-aware steering is more efficient.
- If uniform α is optimal: steering effect is position-independent.

9.4 [U43] Per-Head Steering — 10 GPU-hrs

Modify KV-cache per attention head within the best layer.
Identify which heads respond to steering.
Goal: head-level precision instead of layer-level sledgehammer.
- If 1-2 heads account for all steering effect: ultra-precise steering
  is viable (Agent 4 H-12).

9.5 [U46] Cross-Model Transfer — 12 GPU-hrs

Apply best Qwen2.5-7B steering configuration to:
  LLaMA-3-8B (MHA, similar size)
  Mistral-7B (MHA, sliding window)
  SmolLM2-360M (confirm transfer at small scale)
- If pattern transfers: velocity dynamics are model-agnostic.
- If pattern fails on LLaMA/Mistral: architecture-specific.

9.6 Publishability Gate

Epoch 6 produces: "Advanced KV-cache steering: combined standard +
contrastive signals, per-position α, and per-head steering resolution"

=======================================================================

--- 10. EPOCH 7: ARCHITECTURAL INNOVATION (40-50 GPU-hrs, ~2 weeks) ---

Conditional on Epoch 6 confirming steering is robust and general.
Requires significant code base changes.

10.1 [U52] Bayesian α Optimization — 8 GPU-hrs

Instead of grid sweep, use Bayesian optimization (GP-UCB) to find
optimal α per layer with fewer evaluations.
- If Bayesian finds same optimum in half the evaluations: adopt for
  all future experiments.

10.2 [U54] Death Layer Inoculation Training — 4 GPU-hrs

Fine-tune model (or TT) to reduce death layer sensitivity.
If successful: death layers are learnable artifacts, not fundamental
architecture constraints.

10.3 [U55] RL-Based Per-Token α — 20 GPU-hrs

Train a policy network that outputs α(t) per token position.
Reward: accuracy at end of generation.
If successful: fully adaptive steering. Major architectural result.

10.4 [U56] Dual-Surface Steering — 12 GPU-hrs

Steer both KV-cache AND residual stream (weight-flow modulation).
Vector combination: determine if surfaces are complementary.

10.5 [U57] Multi-Head Contrastive Ensemble — 8 GPU-hrs

Train multiple contrastive TTs with different data splits.
Ensemble their predictions. Tests whether variance reduction improves
steering quality.

10.6 Publishability Gate

Epoch 7 produces: "Architectural innovations in KV-cache steering:
Bayesian optimization, death layer inoculation, RL-based policy, and
dual-surface modulation"

=======================================================================

--- 11. EPOCH 8: FUNDAMENTAL RESEARCH (75+ GPU-hrs, 2-4 weeks) ---

The "grand vision" experiments. Only if all previous epochs confirm
the paradigm.

11.1 [U58+U59] Self-Bootstrapping TT Loop — 15 GPU-hrs

Iterative process: steer → collect new trajectories → retrain TT on
steered states → steer again. Tests whether the TT improves when
trained on its own steering outputs.
- If yes: positive feedback loop, potentially unbounded improvement.
- If no: TT can only learn from natural trajectories.

11.2 [U70] Universal Velocity Manifold — 10 GPU-hrs

Test whether velocity dynamics are universal across: model sizes,
architectures, training stages. Use projection infrastructure.

11.3 [U71+U60] RL-Based α Optimization — 20 GPU-hrs

Full RL policy: state = hidden state at position t; action = α;
reward = next-token prediction accuracy. This is the complete
closed-loop steering optimization.

11.4 [U61] Dual-Surface (Weight-Flow + KV-Cache) — 25 GPU-hrs

The most ambitious experiment: modulate both KV-cache and residual
stream simultaneously. Requires significant new infrastructure.

11.5 [U62] Multi-Head Contrastive Ensemble — 15 GPU-hrs

Ensemble of contrastive TTs for variance reduction. Tests whether
steering quality scales with ensemble size.

11.6 Publishability Gate

Epoch 8 produces the capstone publication:
"Autonomous self-improving KV-cache steering: from hand-tuned α to
RL-optimized adaptive steering policy"

=======================================================================

--- 12. SUMMARY DECISION TREE (FULL) ---

```
START
  │
  ├── Epoch 0 (0 GPU-hrs, 30 min): R² corr, velocity norms, PCA, upper bound
  │   └── → Epoch 1
  │
  ├── Epoch 1 (3.7 GPU-hrs): FOUNDATIONAL — 4-condition × 28-layer
  │   │
  │   ├── C2 > C1 > baseline (TT direction matters) ──────────────────→ Epoch 2
  │   ├── C2 ≈ C1 > baseline (direction irrelevant, perturbation works) → Epoch 2*
  │   ├── C3 > C2 (contrastive works) ─────────────────────────────────→ Epoch 3 (skip 2)
  │   └── All ≈ baseline ──────────────────────────────────────────────→ PUBLISH: null result
  │
  ├── Epoch 2 (2.5 GPU-hrs): MECHANISTIC PROBING + COMPLEX α
  │   │
  │   ├── TT learns causal dynamics (shuffle drops R², naive base > 0.3 below TT)
  │   │     └── → Epoch 3
  │   ├── TT learns surface statistics (shuffle preserves R²)
  │   │     └── → Epoch 3, but with caveat: "steering may be exploiting patterns"
  │   └── Complex α gate passes (R²_a > 0.3, θ_opt ≠ 0 at L8)
  │         └── → Epoch 5 (full phase sweep) in parallel
  │
  ├── Epoch 3 (4 GPU-hrs): SIGNED α MAPPING
  │   │
  │   ├── α inversion confirmed (death layers flip at negative α)
  │   │     └── → Epoch 4
  │   ├── α ≈ 0.1 optimal everywhere
  │   │     └── → Epoch 4 (reduced scope) OR PUBLISH α sweep + move to Epoch 5
  │   └── Steep α-accuracy cliffs detected
  │         └── → Restrict α range, → Epoch 4
  │
  ├── Epoch 4 (6 GPU-hrs): MECHANISM + PRECISION + MULTI-LAYER
  │   │
  │   ├── Best config > +30pp (e.g., L8+L9(−α) > 95%)
  │   │     └── PRIORITY SHIFT: PUBLISH IMMEDIATELY
  │   ├── Best config > +15pp
  │   │     └── → Epoch 5
  │   ├── Best config +5-15pp
  │   │     └── → Epoch 5 (robustness focus)
  │   └── Best config < +5pp
  │         └── → Epoch 5 (cross-validation) → PUBLISH comprehensive
  │
  ├── Epoch 5 (7 GPU-hrs): GENERALIZATION + ROBUSTNESS
  │   │
  │   ├── Steering generalizes to ARC, BBH, SVAMP
  │   │     └── → Epoch 6 (confident investment)
  │   ├── Steering only works on math
  │   │     └── → PUBLISH domain-specific finding → Epoch 6 (reduced)
  │   ├── Threshold probe finds α-dependent threshold
  │   │     └── → Major finding: "capability threshold is not fundamental"
  │   └── Full phase sweep confirms complex α
  │         └── → Complex α → Epoch 6
  │
  ├── Epoch 6 (20-30 GPU-hrs): ADVANCED STEERING
  │   │
  │   ├── Combined std+contrastive shows synergy
  │   │   └── → Epoch 7
  │   └── Per-head steering isolates 1-2 heads
  │         └── → Major: "head-level steering suffices"
  │
  ├── Epoch 7 (40-50 GPU-hrs): ARCHITECTURAL INNOVATION
  │   │
  │   ├── Bayesian optimization succeeds
  │   ├── Death layer inoculation works
  │   ├── RL-based per-token α works
  │   └── Dual-surface shows complementarity
  │       └── → Epoch 8
  │
  └── Epoch 8 (75+ GPU-hrs): FUNDAMENTAL RESEARCH
      │
      └── Self-bootstrapping loop + RL α = COMPLETE STEERING PARADIGM
          └── PUBLISH: "Autonomous self-improving KV-cache steering"
```

=======================================================================

--- 13. BUDGET SUMMARY ---

| Epoch | Cumulative GPU-hrs | Cumulative Wall Time | Publishable Outputs |
|-------|-------------------|---------------------|---------------------|
| 0     | 0                 | 30 min              | Velocity manifold characterization |
| 1     | 3.7               | 4 hrs               | **Foundational validation** (all outcomes) |
| 2     | 6.2               | 7 hrs               | TT mechanism + complex α gate |
| 3     | 10.2              | 12 hrs              | **Signed α map** (standalone) |
| 4     | 16.2              | 2 days              | Mechanism + precision + multilayer |
| 5     | 23.2              | 4 days              | **Generalization + capability threshold** |
| 6     | 43-53             | 1.5 weeks           | Advanced steering methods |
| 7     | 83-103            | 3 weeks             | **Architectural innovations** |
| 8     | 158-178           | 4-5 weeks           | **Complete paradigm** |

Minimum viable publication: Epoch 0 + Epoch 1 = 3.7 GPU-hrs, 1 day.
Maximum program: All epochs = ~175 GPU-hrs, ~1 month.

=======================================================================

--- 14. PUBLICATION STRATEGY ---

Each epoch is designed as a standalone publication:

| Paper | Content | Epochs | Timeline | Journal Venue |
|-------|---------|--------|----------|---------------|
| P1 | "Foundational validation of KV-cache steering" | 0, 1 | Week 1 | EMNLP 2026 (short) |
| P2 | "Signed α mapping for per-layer steering optimization" | 3 | Week 1-2 | ACL 2027 (short) |
| P3 | "Mechanistic understanding of KV-cache steering" | 2, 4 | Week 2 | ICLR 2027 |
| P4 | "Cross-task generalization of KV-cache steering" | 5 | Week 2-3 | NeurIPS 2027 |
| P5 | "Advanced KV-cache steering" | 6, 7 | Week 3-4 | JMLR or ACL 2027 |
| P6 | "Autonomous self-improving steering" | 8 | Week 4-5 | Nature MI or NeurIPS |

KEY INSIGHT: Papers P1-P4 are GUARANTEED regardless of results. Even
negative findings produce publishable content (null results with proper
statistical methodology are valuable for the field). Papers P5-P6 depend
on positive findings.

=======================================================================

--- 15. RISK MANAGEMENT ---

15.1 Risk: Epoch 1 fails (all conditions ≈ baseline)
  - Impact: LOSS of 3.7 GPU-hrs, 1 day
  - Mitigation: Publish null result (Paper P1). Save $300+ GPU budget.
  - Safeguard: All infrastructure needed for Epoch 1 already exists.
    No sunk cost in code development.

15.2 Risk: Epoch 3 shows α doesn't matter
  - Impact: Missed opportunity for improvement, but existing results stand.
  - Mitigation: The α sweep itself is publishable. Epochs 1-2 already
    provide P1-P2 content.

15.3 Risk: Hard drive space (71GB total, max ~76GB at full program)
  - Impact: Epoch 8 requires ~50GB intermediate storage.
  - Mitigation: Delete intermediate checkpoints after analysis. Keep
    only: (a) raw accuracy results, (b) optimal configs, (c) final
    model weights. Estimated peak storage: ~40GB.

15.4 Risk: GPU availability and VRAM (8GB shared)
  - Impact: Qwen2.5-7B uses ~7GB at 16-bit. Marginal headroom.
  - Mitigation: Use 8-bit quantization for evaluation if OOM occurs.
    All experiments run at batch_size=1, which is VRAM-efficient.

15.5 Risk: Researcher burnout/time constraints
  - Impact: Total program is ~1 month.
  - Mitigation: Epoch 1 takes 1 day. That's the commitment. Every epoch
    after is optional. No "sunk cost" obligation.

=======================================================================

--- 16. COMPARISON TO SOURCE ANALYSES ---

This protocol resolves every disagreement between the 5 agents:

| Disagreement | Agents | Resolution in This Protocol |
|-------------|--------|---------------------------|
| Contrastive vs Random first | Agent 1 vs Agents 3,4 | Both are in U00. Random and contrastive are tested simultaneously, not sequentially. This subsumes the debate. |
| R² paradox meaningful? | Agent 2 vs others | U01 (zero-cost) + U21-23 (TT dissection) resolve empirically. No need to debate. |
| Death layer mechanism | Agents 1,5 vs Agent 2 | U08 resolves whether negative α works. U27 resolves if mechanism is K/V amplification. |
| Complex α relevance | Complex α analysis | U63-64 test this directly. Gate determines follow-up. |
| Capability threshold fundamental? | All agents | U36 tests α-dependency directly. |
| TT architecture adequate? | Agent 5 vs Agent 2 | U21-23 (TT dissection) resolve this empirically. |
| Per-layer independence | All agents | U30 (multi-layer combination) tests directly. |

=======================================================================

--- 17. REDUNDANCY ELIMINATION ---

The ~35+ proposed experiments reduce to 14 ADOPTED experiments in
8 epochs, achieving ~60% reduction in total experiment count while
INCREASING information coverage. No proposed experiment tests a
hypothesis not covered by this protocol.

The key insight enabling this reduction:
1. MULTIPLEXING: U00 (4 conditions × 28 layers) replaces 7 separate
   experiments (random baseline, contrastive eval, standard sweep, etc.)
2. GATING: U63-64 gate determines whether complex α deserves full
   exploration — no need to propose both paths.
3. ZERO-COST: U01-04 extract information from existing data — no GPU
   needed for insights that would otherwise require experiments.
4. PROGRESSIVE PRECISION: U35 (500 problems) only runs after U08
   identifies the optimal configuration — no need for brute-force
   high-N at every layer.

=======================================================================

--- 18. IMMEDIATE ACTION CHECKLIST ---

TODAY (30 min, 0 GPU):
  [ ] U01: Compute R² vs Δ accuracy correlation
  [ ] U02: Compute steering upper bound estimate
  [ ] U03: Compute velocity-norm distribution
  [ ] U04: Run PCA on held-out hidden states
  [ ] Pre-register U00 analysis plan (4 conditions, 28 layers, Bonferroni)

NEXT GPU SESSION (3.7 GPU-hrs, ~4 hrs):
  [ ] Write ~20 lines of code for random vector generation
  [ ] U00: Run 4-condition × 28-layer sweep
  [ ] Run randomization protocol (shuffle conditions, layer order)
  [ ] Bootstrap CIs for all conditions
  [ ] Apply Bonferroni correction
  [ ] Publish results on GitHub + archive

AFTER EPOCH 1 (decision-dependent):
  [ ] If positive: Proceed to Epoch 2 (mechanistic probing)
  [ ] If null: Publish negative results
  [ ] Update pre-registration for Epoch 3 before starting

=======================================================================

END OF CONCEPTUAL DIFFUSER 4 — OPTIMAL EXPERIMENTAL PROTOCOL
=======================================================================
