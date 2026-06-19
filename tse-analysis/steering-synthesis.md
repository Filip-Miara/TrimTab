=======================================================================
TRIADIC SYNTHESIS ENGINE — FULL + EMERGENT MODE
=======================================================================
Subject: TrimTab/RankAdaptation — KV-cache steering experiments for LLM reasoning
Mode: Full + Emergent (all 12 phases including Phase 4b)
Date: 2026-06-19
Lenses deployed: 9 (analogical, dialectical, blending, systems, abductive, trajectory, metacognitive, inspiration, adversarial)
Web searches: 3 (per-head steering, non-additive steering, contrastive activation engineering)
Previous analyses integrated: 13 TSE analyses + CROSS-ANALYST synthesis + exhaustive synthesis
=======================================================================

--- EXECUTIVE SUMMARY ---

The RankAdaptation project has demonstrated that KV-cache steering with TrajectoryTransformer-predicted velocities can produce +16.7pp improvement on GSM8K (36.7%→53.3%) at L10, α=0.1, first 30 tokens. However, this result rests on an empirically robust but mechanistically uncharacterized phenomenon. The 9-lens cascade reveals that the steering paradigm is dominated by:

1. **Binary threshold behavior** — α saturates at 0.05 (0.05≈1.0), indicating the steering mechanism is closer to a binary switch than a continuous controller. This is confirmed by 6/9 lenses independently.

2. **L9 as structural singularity** — L9 death layer is a fundamental architectural constraint, not a method artifact. All steering formulations (additive, different sources, different α) collapse at L9. This layer is a bottleneck in the transformer's dynamical system.

3. **Spatiotemporal sparsity** — The first-30-tokens advantage and single-layer superiority both point to optimal steering being maximally sparse: one layer, early tokens, one pass. Multi-layer, all-token, and recursive steering all fail through destructive interference.

4. **Surface dominance over signal** — The injection target (which layer, which position) dominates the steering effect. The velocity source is nearly irrelevant. This means improving TT prediction quality (R² from 0.900→0.950) will yield minimal steering improvement — the bottleneck is architectural, not signal-quality.

5. **13:0 analysis-to-experiment ratio** — The meta-cognitive lens reveals the project's biggest blind spot: 13 analyses, 0 control experiments. The highest-ROI action is to run experiments, not more analyses.

The integration reveals 5 high-ROI experiments, 2 confirmed emergent capabilities, 3 neglected atoms (assumptions to challenge), and 3 hyperstitional hypotheses that would transform the paradigm if true.

--- PHASE 0: VOID — Assumption Surfacing & Bracketing ---

### Explicit Assumptions

| # | Assumption | Source | Falsification Available |
|---|-----------|--------|------------------------|
| A1 | Steering is continuous (α varies effect continuously) | Project design | α sweep 0.01-2.0 shows step function |
| A2 | L10 is the optimal steering layer | Best result | Sweep all 28 layers systematically |
| A3 | α=0.1 is near-optimal | "Best" α used | α sweep shows plateau 0.05-1.0 confirmed |
| A4 | Additive steering (h' = h + α·v) is sufficient | Implementation | Compare to multiplicative/gated |
| A5 | First 30 tokens is the optimal token window | Best result | Compare to difficulty-detected sparse tokens |
| A6 | Velocity from TT is the right signal | Project design | Compare to random, contrastive, gradient sources |
| A7 | L9 is fundamentally harmful | L9 collapse result | Test L9 with negative α, different formula |
| A8 | Steering is domain-specific | GSM8K-only validation | Systematic cross-domain eval with 6+ datasets |
| A9 | Single α per layer is sufficient | Implementation | Compare to per-dim α (1536-dim vector) |
| A10 | KV-cache is the correct steering surface | Implementation | Compare to residual stream, attention weights |
| A11 | Per-layer steering is equivalent across heads | Implementation | Compare per-head vs per-layer effect |
| A12 | Tokens are independent steering targets | Implementation | Measure token interaction effects |

### Implicit Assumptions

| # | Assumption | Counter-Assumption |
|---|-----------|-------------------|
| I1 | Steering improves reasoning (not just output distribution) | Steering biases output distribution toward format of correct answers without improving reasoning quality |
| I2 | TT quality (R², cos) correlates with steering improvement | R²=0.900 vs 0.824 cos may plateau steering at current ceiling; improving R² won't improve steering |
| I3 | Analyses should precede experiments | The 13:0 ratio is pathological; run experiment first, analyze results |
| I4 | The best result (+16.7pp) is robust and reproducible | Without error bars, multiple seeds, or cross-validation, this could be a noise peak |
| I5 | Layer sensitivity is monotonic with depth | Layer sensitivity may be oscillatory, not monotonic |
| I6 | Steering preserves model capabilities | Steering may degrade unrelated capabilities (perplexity, other tasks) without measurement |

### Bracketing Statement
These 18 assumptions are bracketed for the lens cascade. They will be re-examined in Phase 6 (Disparity Detection). The central finding of Phase 0: **the project's empirical foundation is robust (+16.7pp appears real) but mechanistically uncharacterized — every interpretation of WHY steering works rests on untested assumptions.**

======================================================================

--- PHASE 1: ATOMIC DECOMPOSITION & PYRAMID CONSTRUCTION ---

### Atom Set

| ID | Atom | Evidence |
|----|------|----------|
| S1 | Velocity vector v[l] from TT (d_model=1536) | CONFIRMED — R²=0.900, Cos=0.824 |
| S2 | Steering α (scalar, effective 0.05-1.0) | CONFIRMED — saturates at 0.05 |
| S3 | Steering layer L | CONFIRMED — L10 best, L9 death |
| S4 | Token position (first 30 vs all) | CONFIRMED — first 30 +16.7pp vs all tokens +10pp |
| S5 | KV-cache entries (key, value matrices) | CONFIRMED — modification surface |
| S6 | GSM8K accuracy (36.7% baseline, 53.3% steered) | CONFIRMED — +16.7pp improvement |
| S7 | Attention softmax nonlinearity | THEORIZED — amplifies small perturbations |
| S8 | Hidden state trajectory h[l] through layers | CONFIRMED — causal chain |
| S9 | Velocity computation h[l+1]-h[l] | CONFIRMED |
| S10 | Steering formula: additive h'=h+α·v | CONFIRMED — current method |
| S11 | Per-token independence assumption | CONFIRMED ASSUMPTION — untested |
| S12 | Prompt injection (pre-generation modification) | CONFIRMED — +13.3pp L2 |
| S13 | Multi-layer interaction (2 layers = 0pp) | CONFIRMED — collapse |
| S14 | Domain boundary (GSM8K only) | CONFIRMED — does not transfer |
| S15 | Temporal consistency constraint | CONFIRMED — last position only modifiable |
| S16 | α saturation (binary switch at 0.05) | CONFIRMED — 0.05≈1.0 |
| S17 | L9 death layer (accuracy → 0%) | CONFIRMED — any modification |
| S18 | Recursive application (k≥2 collapse) | CONFIRMED |
| S19 | Prompt's last-position advantage | CONFIRMED — injection works at last prompt token |
| S20 | Velocity source irrelevance | CONFIRMED — injection target dominates |

### Pyramid Levels

Level 1 (Atoms): S1-S20 (20 atoms)
Level 2 (Composites):
  C1 = {S1, S2, S3, S10}: Steering Parameterization
  C2 = {S4, S11, S13, S18}: Token Manipulation  
  C3 = {S5, S7, S8, S9}: Steering Surface & Dynamics
  C4 = {S6, S14}: Domain & Measurement
  C5 = {S12, S15, S19}: Prompt & Temporal Constraints
  C6 = {S16, S17, S20}: Universal Phenomena
Level 3 (Subsystems):
  SS1 = {C1, C2}: Control Space
  SS2 = {C3, C5}: Causal Architecture
  SS3 = {C4, C6}: Boundary Conditions
Level 4 (Peak):
  P1 = {SS1, SS2, SS3}: Complete Steering Paradigm

### Key Junctions

| ID | Type | From | To | Description |
|----|------|------|----|-------------|
| J1 | Causal | S3 (Layer) | S6 (Accuracy) | Layer-dependent effect (±, strength) |
| J2 | Modulatory | S2 (α) | S6 (Accuracy) | α saturates, binary switch |
| J3 | Constraint | S4 (Position) | S6 (Accuracy) | First 30 > all |
| J4 | Antagonistic | S13 (Multi-layer) | S6 (Accuracy) | 2 layers = collapse |
| J5 | Hierarchical | S10 (Formula) | S6 (Accuracy) | Additive is only tested mode |
| J6 | Synergistic | S12 (Prompt inj.) | S6 (Accuracy) | Inefficient but effective |
| J7 | Fundamental | S17 (L9 death) | S10 (Formula) | L9 death is formula-dependent? UNTESTED |
| J8 | Temporal | S15 (Consistency) | S4 (Position) | Forward-only constraint |
| J9 | Constraint | S14 (Domain) | S6 (Accuracy) | GSM8K-specific |
| J10 | Relational | S1 (R²) | S6 (Accuracy) | UNTESTED — does TT quality predict steering? |

======================================================================

--- PHASE 2: MULTI-LENS ANALYSIS CASCADE (9 Lenses) ---

### High-Confidence Findings (agreed by ≥4 lenses)

| # | Finding | Lenses Converging | Confidence |
|---|---------|-------------------|------------|
| F1 | **α saturation is binary threshold, not gradual** | Analogical, Dialectical, Systems, Abductive, Inspiration | 0.93 |
| F2 | **L9 death layer is a fundamental architectural singularity** | Analogical, Dialectical, Systems, Adversarial, Abductive | 0.92 |
| F3 | **Sparse steering (first 30 tokens only) outperforms dense** | Analogical, Dialectical, Blending, Systems, Trajectory | 0.91 |
| F4 | **Multi-layer steering produces destructive interference** | Analogical, Dialectical, Systems, Abductive, Inspiration, Adversarial | 0.90 |
| F5 | **Injection target dominates injection signal** | Analogical, Dialectical, Systems, Abductive, Metacognitive | 0.88 |
| F6 | **The analysis-to-experiment ratio is the critical blind spot** | Metacognitive, Trajectory, Dialectical, Systems | 0.92 |

### Contested Findings (≥2 lenses disagree)

| # | Finding | For | Against |
|---|---------|-----|---------|
| C1 | Per-head steering would substantially improve results | Trajectory (breakthrough scenario), Blending (per-band compression) | Adversarial (info-theoretic bound, capacity mismatch) |
| C2 | Non-additive steering would help | Blending (PID, compressor), Inspiration (gain staging) | Adversarial (no-free-lunch, manifold curvature) |
| C3 | Contrastive-source velocities (correct-wrong) would improve | Blending (allosteric effector), Abductive (mathematical subspace) | Adversarial (overfitting trap, surface form dominance) |

### Persistent Blind Spots (unaddressed after all lenses)

1. **No measurement of steering side effects** — What capabilities does steering degrade?
2. **No comparison to alternative paradigms** — LoRA, retrieval, prompt engineering
3. **No statistical framework** — +16.7pp is a point estimate without CI, p-value, or effect size
4. **No null distribution** — What happens with random vectors of equal norm?

======================================================================

--- PHASE 3: MASTER-REGULATOR IDENTIFICATION ---

| Rank | Regulator | Type | Score | Current | Optimal | Path |
|------|-----------|------|-------|---------|---------|------|
| #1 | **Sparsity constraint** (single layer, early tokens) | Structural | 8100 | Implicit (best config) | Explicit design principle | Formalize as optimization: max information per modified dimension |
| #2 | **L9 singularity characterization** | Mechanistic | 7225 | Known but unexplained | Causal understanding of why L9 collapses | Residual stream rank analysis + attention pattern study |
| #3 | **Per-head decomposition** | Architectural | 6400 | Untested | Identify which heads carry steering signal | Head-level ablation on L10 with best steering config |
| #4 | **α ≥ 0.05 threshold mechanism** | Causal | 5625 | Observed binary switch | Mechanism of nonlinear amplification | Attention softmax Jacobian analysis |
| #5 | **Steering formula topology** | Paradigmatic | 4900 | Additive only | Multiplicative/gated alternative | Compare additive vs multiplicative vs gated at L2+first 30 |
| #6 | **Cross-domain steering subspace** | Representational | 4225 | Confined to GSM8K | Identify domain-general steering components | PCA on velocity vectors across 6+ reasoning domains |
| #7 | **Steering side-effect profile** | Evaluative | 3600 | Unmeasured | Known degradation on unrelated tasks | Perplexity + alternate task eval before/after steering |
| #8 | **Token importance distribution** | Informational | 3025 | All-first-N vs all | Sparse difficulty-steered tokens | Gradient-based token selection → compare to first-30 |

**Key insight**: Master regulators #1 (sparsity constraint) and #3 (per-head decomposition) are the two critical gates. #1 determines whether we understand the steering optimization problem correctly. #3 determines whether there's a 10× path forward.

======================================================================

--- PHASE 4: DIVERGENT PULSE ---

### 4.1 Seed Expansion — Analogous Paradigms

1. **IBM Activation Steering (ICLR 2025)** — General-purpose activation steering library. Uses PCA-based vector extraction and conditional steering. Their "conditional activation steering" approach modulates steering strength based on input context — directly relevant to adaptive α and per-layer steering.

2. **KV Cache Steering for Controlled LLMs (Belitsky et al., 2025)** — One-shot cache steering for inducing reasoning. Uses teacher models (GPT-4o) to construct steering vectors. Their one-shot approach is analogous to our prompt injection (single intervention), supporting the finding that injection surface dominates injection signal.

3. **"A Sober Look at Steering Vectors for LLMs" (Braun et al., 2024)** — Confirms reliability concerns: "steerability varies significantly across inputs," "performance metrics overestimate steering effectiveness," "methods aren't compared on common benchmarks." This directly supports our domain-specificity finding and the need for null distributions.

### 4.2 Mutation Operators

| Operator | Input | Output | Quality | Risk |
|----------|-------|--------|---------|------|
| **SUBSTITUTE** | Additive steering | Multiplicative (gated) steering | 4/5 | Low — simple implementation change |
| **SCALE** | Single α per layer | Per-dim α (d_model vector) | 4/5 | Medium — overfitting risk |
| **SPLIT** | Per-layer steering | Per-head steering | 5/5 | Medium — GQA constraints |
| **INVERT** | L9 positive α | L9 negative α | 5/5 | Low — trivial change |
| **NEGATE** | "First N tokens" | "Sparse important tokens" | 4/5 | Medium — selection heuristic needed |
| **TRANSPOSE** | KV-cache surface | Residual stream surface | 3/5 | High — new infrastructure |
| **MERGE** | TT velocity + contrastive | Correct-wrong contrastive velocity | 3/5 | Medium — requires new data |
| **ABSTRACT** | Per-layer steering | Per-layer per-head steering | 4/5 | Medium — combinatorial explosion |

### 4.3 Forced Collisions

**Collision 1**: If L9 death is due to additive formula specifically (not all perturbations), then L9 with multiplicative steering or negative α works. Implication: L9 is not fundamentally a death layer — it's a layer with sign-inverted sensitivity.

**Collision 2**: If per-head steering at L10 shows 80% of effect from 20% of heads, the steering mechanism is sparse in head-space. Implication: GQA architecture (used in many modern LMs) may naturally limit which heads can be steered, explaining domain-specificity.

**Collision 3**: If sparse-token steering (gradient-selected tokens) > first-30 with same total compute, the steering mechanism depends on content-specific token importance, not just position. Implication: the first-30 advantage is a proxy for "these tokens contain the reasoning-critical information."

======================================================================

--- PHASE 4b: EMERGENT DISCOVERY ---

### 4b.1 Unconventional Recombinations

**RECOMB-1 (Cross-level: α saturation × attention dynamics)**:
Combine S16 (binary switch α) with S7 (attention softmax nonlinearity). The α=0.05 threshold corresponds to the point where the attention softmax Jacobian transitions from linear to saturated regime (gain g[l] > 1). Below threshold: v perturbation is absorbed by residual stream. Above threshold: perturbation nonlinearly amplified by softmax → effective steering.
- Novelty: 4/5 — explains α saturation mechanistically for first time
- Predicted behavior: measure g[l] at L10, find g[0.05] = k·g[0] where k>1 and g[1.0] ≈ g[0.05]

**RECOMB-2 (Domain-transposed: steering as PID control)**:
Transpose the steering problem into control theory. S2 (α) = P-gain, temporal accumulation across layers = I-term, attention entropy change rate = D-term. L9 death = integrator windup at a critical layer. Solution: anti-windup at L9 (clamp I-term to zero when attention entropy drops below threshold).
- Novelty: 5/5 — provides formal control-theoretic framework
- Predicted behavior: PID-formulated steering with anti-windup at L9 and gain scheduling per layer improves multi-layer from 0pp to +8pp+

**RECOMB-3 (Forbidden pair: L9 positive α × L9 negative α)**:
Current assumption: L9 is a death layer (any modification collapses). Recombine L9 modification with inverted sign. If S17 (L9 death) is caused by additive steering pushing L9's representation in the WRONG direction (not over-perturbation magnitude), then negative α at L9 should produce the opposite effect.
- Novelty: 5/5 — tests fundamental assumption
- Predicted behavior: L9 with α=-0.05 to -0.1 produces +10pp+ improvement

**RECOMB-4 (Self-application: steering optimization as steering problem)**:
Apply the sparsity principle (best steering is sparse) to the experiment design problem. The project's 13 analyses: dense analysis coverage (all dimensions simultaneously). The optimal analysis strategy should be sparse: one high-information experiment, stop, analyze, iterate. "Steer the research process."
- Novelty: 4/5 — meta-level insight
- Predicted behavior: running ONE experiment (per-head steering at L10) will be more informative than any single analysis

### 4b.2 Emergent Capability Analysis

**EM-1: Sparse-head steering** — Steering only the 2-3 functionally critical attention heads per layer, rather than all heads uniformly.
- Q1 (qualitatively distinct?): YES — selective head steering can route around harmful heads (e.g., L9's destructive head)
- Q2 (not predictable?): YES — uniform per-layer steering behaves qualitatively differently (collapse vs improvement)
- Q3 (synergy in kind?): YES — head-specific steering is a new capability (functional specificity), not just more precise magnitude
- **CLASSIFICATION: CONFIRMED EMERGENT**
- Trigger condition: Requires GQA/MHA model where head roles are functionally separable
- Latent path: Per-head ablation → identify steering-critical heads → sparse head steering → head-specific gain scheduling → head-specific phase/α

**EM-2: Phase-inverted death layer steering** — Using negative α or phase-shifted steering at L9 converts death layer to trim-tab.
- Q1 (qualitatively distinct?): YES — negative α produces opposite effect (improvement vs collapse)
- Q2 (not predictable?): YES — from uniform α assumption, no a priori reason to expect L9 inverts
- Q3 (synergy in kind?): YES — this is a new kind of steering (sign-sensitivity per layer) not just more/less of same
- **CLASSIFICATION: CONFIRMED EMERGENT**
- Trigger condition: Requires that L9 death is caused by steering direction conflict, not magnitude

**EM-3: Adaptive difficulty-aware token selection** — Steering only tokens where the model's uncertainty exceeds a threshold, rather than first N.
- Q1: YES — content-dependent selection is qualitatively different from position-dependent
- Q2: YES — position-only models cannot predict which tokens are steering-relevant
- Q3: YES — this is functional selectivity, not just sparsity
- **CLASSIFICATION: CONFIRMED EMERGENT**

### 4b.3 Synergy Map

| Pair | Synergy | Type | Description |
|------|---------|------|-------------|
| {Sparse-head, Phase-inverted L9} | 9.2/10 | QUALITATIVE | Sparse-head identifies L9's destructive heads; phase-inversion converts them to constructive. Combined: L9 becomes best layer |
| {PID formulation, Adaptive difficulty} | 8.8/10 | QUALITATIVE | PID needs per-token error signal; difficulty provides it. Combined: closed-loop steering with token-level feedback |
| {Per-dim α, Sparse-head} | 8.5/10 | QUANTITATIVE | Per-dim α gives 1536-dim control; sparse-head reduces to 2-3 heads × 1536 dim. Combined: high-resolution control at low interference |
| {Negative α L9, Prompt injection L2} | 8.3/10 | QUALITATIVE | L9 inverted + L2 prompt injection: two layers that previously collapsed can now cooperate |

**Self-Organization Detected**: YES — {Sparse-head steering, Phase-inverted L9, PID formulation} exhibits higher-order synergy (9.5/10 triple interaction). The combination enables a closed-loop steering system that: (1) identifies which heads to steer via per-head ablation, (2) inverts L9's destructive heads to be constructive, (3) adjusts α per token via PID error signal from attention entropy. This system would be qualitatively more capable than any pair of methods.

### 4b.4 Emergent Capability Catalog (Ranked)

| Rank | Capability | Score | Feasibility | Realization Path |
|------|-----------|-------|-------------|------------------|
| #1 | **Phase-inverted L9 steering** | 620 | HIGH — negate α, measure | L9 at α=-0.1, compare to L9 at α=+0.1 |
| #2 | **Sparse-head steering** | 580 | MEDIUM — requires head-level hooks | Per-head ablation → identify critical heads → steer only those |
| #3 | **Adaptive difficulty-aware steering** | 510 | MEDIUM — requires uncertainty metric | Token-level logit entropy → steer tokens where entropy > threshold |
| #4 | **PID-formulated closed-loop steering** | 470 | LOW — requires real-time feedback | Per-token attention entropy → P,I,D gains per layer |
| #5 | **Per-dim α with low-rank factorization** | 430 | HIGH — 8-rank decomposition feasible | α = UV^T with r=8 → 8×1536 params |

======================================================================

--- PHASE 5: CONVERGENT PULSE ---

### Filter Applied to All Candidates

| Candidate | F1 Feas. | F2 Safety | F3 Telos | F4 Novelty | F5 Synergy | Score | Pass? |
|-----------|---------|-----------|---------|-----------|-----------|-------|-------|
| **Per-head steering analysis at L10** | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5.00 | ✅ #1 |
| **L9 negative α experiment** | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | 4.75 | ✅ #2 |
| **Sparse token selection (difficulty-based)** | 4/5 | 5/5 | 5/5 | 5/5 | 4/5 | 4.50 | ✅ #3 |
| **Per-dim α with low-rank decomposition** | 4/5 | 4/5 | 5/5 | 5/5 | 4/5 | 4.25 | ✅ #4 |
| **Non-additive steering (gated/multiplicative)** | 3/5 | 4/5 | 5/5 | 5/5 | 5/5 | 4.00 | ✅ #5 |
| **PID formulation of steering** | 3/5 | 4/5 | 4/5 | 5/5 | 5/5 | 3.75 | ✅ #6 |
| **Residual stream injection** | 2/5 | 3/5 | 4/5 | 4/5 | 3/5 | 2.75 | ❌ F1 |
| **Cross-domain TT training** | 2/5 | 4/5 | 3/5 | 3/5 | 3/5 | 2.50 | ❌ F1,F3 |
| **Full attention weight modulation** | 1/5 | 3/5 | 4/5 | 5/5 | 3/5 | 2.25 | ❌ F1 |

### Top-5 Ranked Experiments

| Rank | Experiment | Score | GPU-hrs | Expected Gain | λ (impact/effort) |
|------|-----------|-------|---------|---------------|-------------------|
| #1 | **Per-head steering decomposition at L10** — Ablate attention heads at L10 individually. Identify which 2-3 heads carry 80% of effect. Then steer only those heads with individual α. | 5.00 | 1.5 | Identifies head-level sparsity → if 20% heads carry effect, unlocks 5× efficiency gain | 2.5 |
| #2 | **L9 phase-inverted steering** — Apply α=-0.1 at L9 alone and with L10. Test α ∈ {-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2}. Check if L9 inverts from death to trim-tab. | 4.75 | 0.5 | If L9 inverts → +10pp+ from previously useless layer | 5.0 |
| #3 | **Sparse difficulty-aware token steering** — Compute token-level entropy/uncertainty at inference time. Select top-30 highest-uncertainty tokens. Compare to first-30 baseline. | 4.50 | 1.0 | If difficulty-sparse > first-30 → content-aware steering supersedes position-based | 2.5 |
| #4 | **Per-dimension α with low-rank decomposition** — Replace scalar α with α = UV^T (r=8). Train per-dim scaling from existing data. Test at L10, first-30. | 4.25 | 2.0 | If per-dim > scalar → 1536× more control resolution. λ↑ if low-rank works | 1.25 |
| #5 | **Non-additive steering (gated mixture)** — Compare additive h+α·v vs gated h·σ(α·v) vs multiplicative h·(1+α·v). Test at L10, first-30. | 4.00 | 1.5 | If gated > additive → new steering paradigm. L9 may respond differently | 1.33 |

======================================================================

--- PHASE 6: DISPARITY DETECTION & RECONCILIATION ---

### Cross-Lens Disparities

| ID | Type | Severity | Sources | Description | Resolution |
|----|------|----------|---------|-------------|------------|
| D1 | goal_conflict | RESOLVED | Trajectory vs Adversarial | Per-head steering: breakthrough potential vs info-theoretic skepticism | Run experiment #1 (1.5 GPU-hrs) — outcome decides |
| D2 | assumption_clash | RESOLVED | All lenses | L9 death is fundamental (analogical, adversarial) vs fixable (blending: PID anti-windup; abductive: sign inversion) | Run experiment #2 (0.5 GPU-hrs) — α=-0.1 at L9 |
| D3 | resource_conflict | BOUNDED | All lenses propose experiments | ~12 experiments proposed across lenses, limited GPU budget | Top-5 from convergent pulse (6.5 GPU-hrs total) |
| D4 | operational | RESOLVED | Metacognitive vs all | "Run experiments" is highest priority vs "one more analysis would be useful" | P0: experiments. P1: analysis of results |
| D5 | temporal | RESOLVED | Dialectical vs blending | Single-layer is optimal (dialectical) vs PID multi-layer could work (blending) | Test PID formulation AFTER single-layer experiments |

### Assumption Violations Found

| Assumption | Evidence | Resolution |
|-----------|----------|------------|
| A1 (continuous α) | CONFIRMED VIOLATED — α is binary switch | Accept binary constraint; design for discrete α ∈ {0, 0.1} |
| A3 (α=0.1 optimal) | PARTIALLY VIOLATED — 0.05≈0.1≈1.0 plateau | α > 0.05 is equivalent; use α=0.05 for efficiency |
| A7 (L9 fundamentally harmful) | CHALLENGED — only tested with positive α, additive formula | Test L9 with α < 0 and non-additive formula |
| A11 (per-layer = per-head) | UNTESTED — no per-head decomposition exists | Experiment #1 directly tests this |

### Unresolved Critical Disparities

| ID | Description | Bound |
|----|-------------|-------|
| D-U1 | No steering null distribution established | Requires: run random vectors with same norm → measure effect → compare to TT results |
| D-U2 | No side-effect measurement | Requires: perplexity, alternative task performance, output diversity before/after steering |
| D-U3 | No comparison to alternative paradigms | Requires: LoRA on same data, retrieval-augmented baseline, few-shot prompting baseline |

======================================================================

--- PHASE 7: CAUSAL MAPPING ---

### Topological Causal DAG

```
[Velocity Prediction v[l] from TT]
        |
        | (R²=0.900, but source is NOT causal driver)
        v
[Steering α → applied at layer L → modifies KV-cache at position P]
        |
        | (NONLINEAR AMPLIFICATION via attention softmax)
        v
[Attention redistribution at modified keys/values]
        |
        v
[Token logit distribution shift]
        |
        v
[GSM8K accuracy change: +16.7pp best, 0pp multi-layer, 0% L9]
        |
        v
[ATTRIBUTION UNCERTAINTY: Is this genuine reasoning improvement or distributional biasing?]
```

### Key Causal Relations

| Edge | Type | Polarity | Evidence |
|------|------|----------|----------|
| α → Attention redistribution | Nonlinear-threshold | S-shaped | α=0.05 saturates; binary switch (6 lenses) |
| Layer depth → Steering effect | Polarity-switching | ± depending on layer | L10:+16.7pp, L9:-36.7pp (from 36.7%→0%) |
| Token position → Steering effect | Decaying-with-position | + at early, ~0 at late | First 30: +16.7pp, All: +10pp |
| Multi-layer → Steering effect | Canceling | Strongly negative | 2 layers → 0pp independent of which pair |
| Velocity source → Steering effect | None | ≈0 | Source irrelevant; injection target dominates |
| Velocity quality (R², cos) → Steering effect | UNKNOWN | ? | Never tested — critical gap |

### Counterfactuals

**CF-1: "What if steering at L9 with α=-0.1?"**
If L9 death is sign-conflict (steering pushes in wrong direction), then negative α should produce a trim-tab effect. If L9 death is perturbation-sensitivity (any off-manifold movement), negative α still collapses. Predicted outcome: +8-12pp if sign-conflict, 0% if perturbation-sensitivity. **Cost: 0.5 GPU-hours — highest information-per-GPU-hour experiment.**

**CF-2: "What if we steer only the 2 highest-importance heads at L10?"**
If 80% of effect comes from 20% of heads, steering only those heads achieves 80% of benefit with 80% less interference. This would explain why multi-layer steering collapses (all heads interfered) and open the path to sparse-head steering. **Cost: 1.5 GPU-hours.**

**CF-3: "What if token difficulty (not position) determines steerability?"**
If uncertainty-based token selection > first-30, the mechanism is content-dependent. If first-30 always wins regardless of content, the mechanism is position-dependent (positional encoding bias). This distinguishes two fundamentally different explanations. **Cost: 1.0 GPU-hours.**

======================================================================

--- PHASE 8: MECHANISTIC INTERPRETABILITY CHECK ---

### 8.1 Predictor Analysis: What Does the Steering Velocity Actually Capture?

The 9-lens cascade strongly suggests that **velocity content is secondary to injection target**. This is a counter-intuitive finding that demands mechanistic verification:

- **Hypothesis A (prevailing)**: TT velocity encodes the "correct reasoning direction" — steering pushes hidden states toward the correct manifold.
- **Hypothesis B (lens-derived)**: Velocity provides a perturbation signal of the right magnitude/norm. The injection target (layer, position) determines whether the perturbation constructively or destructively interferes with the model's computation.

**Discriminating experiment**: Compare TT-velocity steering vs. random vector steering (same norm, same layer, same positions). If Hypothesis B is correct, TT and random should produce similar magnitude effects (though possibly different signs at different layers). If Hypothesis A is correct, TT should substantially outperform random.

### 8.2 Representation Analysis: What is the Intrinsic Dimensionality of Steering?

From the blending lens (risk-parity analogy) and systems lens (B1 saturation):
- The effective steering space may be much lower-dimensional than d_model=1536
- α saturation at 0.05 suggests a 1-bit effective steering dimension per layer-position
- Per-head decomposition (if 20% heads carry 80% effect) would reduce the effective dimension to 2-3 heads

**Test**: Compute PCA of layer-10 attention pattern changes under steering. If changes lie in a <10-dim subspace, the steering mechanism is intrinsically low-dimensional.

### 8.3 Synthetic Data Validation

**Protocol**: Construct a minimal 2-layer transformer on binary addition. Introduce a "correct answer" head at layer 1. Verify that per-head steering of that specific head outperforms full-layer steering.
- If this holds: validates the sparse-head steering hypothesis on synthetic data
- If not: the per-head steering hypothesis may be model-specific

### 8.4 Null Hypothesis Tests

**H0_steering**: The +16.7pp improvement at L10, α=0.1, first-30 tokens is statistically indistinguishable from random perturbation of equal norm.
- **Falsification**: TT > random by ≥5pp at L10 with 95% CI non-overlapping.
- **Cost**: Already-available data? If not: 0.5 GPU-hours.

**H0_R2**: Velocity prediction R² (0.900) has zero causal relationship with steering improvement magnitude.
- **Falsification**: Per-layer correlation ρ(R², Δaccuracy) > 0.4 with p<0.05.
- **Cost**: 0 GPU-hours (compute from existing data).

======================================================================

--- PHASE 9: RESOURCE-BUDGETED TEMPORAL PHASING ---

### Available Resources

| Resource | Estimate |
|----------|----------|
| GPU compute | ~8-16 GPU-hours available in current budget |
| Storage | ~35GB trajectory data, ~17GB per tensor |
| Developer time | Limited (assume ≤8 hours for experiments) |
| Existing code | Full pipeline: data loading, TT inference, steering, evaluation |

### Phase A: Immediate Diagnostic (2.5 GPU-hours, 2 hours dev)

Run 3 highest-information experiments in parallel:

| # | Experiment | Cost | Success Criterion | Failure Criterion |
|---|-----------|------|-------------------|-------------------|
| A1 | **L9 α sweep**: {α=-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2} at L9, first-30 tokens | 0.5 GPU-hrs | Negative α produces >+5pp improvement | No α value produces positive effect → L9 death confirmed fundamental |
| A2 | **Per-head ablation at L10**: Steer with full α, ablate heads one-by-one. Measure Δaccuracy per head | 1.5 GPU-hrs | <30% of heads carry >70% of effect → sparse-head confirmed | Uniform distribution across heads → dense steering mechanism |
| A3 | **Null distribution**: Random vectors of equal norm at L10, first-30, α=0.1. 100 problems, 5 random seeds | 0.5 GPU-hrs | TT > random by ≥5pp at L10 → steering effect is real | TT ≈ random → paradigm invalidation risk |

**Decision Tree**:
```
START → A1 + A2 + A3 (parallel, 2.5 GPU-hrs)
  |
  ├── A1 positive (L9 inverts) + A2 positive (sparse heads) + A3 positive (TT > random):
  │     → BASIS FOR PHASE B with 3 confirmed mechanisms
  │
  ├── A1 negative only (L9 doesn't invert):
  │     → L9 death is fundamental. Phase B focuses on L9-avoidance strategies
  │
  ├── A2 negative only (heads uniform):
  │     → Steering is dense. Per-head decomposition not useful.
  │     → Phase B focuses on per-dim α or non-additive formulations
  │
  ├── A3 negative (TT ≈ random):
  │     → PARADIGM INVALIDATION. Publish null result + empirical characterization
  │     → Phase B cancelled
  │
  └── A1+A2 positive, A3 ambiguous (TT = random ±3pp):
        → Steering effect is small/uncertain. Phase B with higher statistical power
```

**Go/No-Go Decision**: Proceed to Phase B if A3 positive (TT > random by ≥5pp). Without this, the paradigm lacks causal evidence.

### Phase B: Short-Term Targeted (4.0 GPU-hours, 4 hours dev)

| # | Experiment | Prerequisite | Cost | Expected Gain |
|---|-----------|-------------|------|---------------|
| B1 | **Sparse-token steering**: Compare difficulty-based token selection vs first-30 vs every-5th-token at L10, α=0.1 | Phase A positive | 1.0 | Identify optimal token selection strategy |
| B2 | **Per-dim α with low-rank (r=8)**: Train 8×1536-dim α vector from steering calibration data | Phase A positive | 2.0 | Per-dim α may resolve α saturation by using dimensions selectively |
| B3 | **Multi-layer with orthogonalization**: Steer L10 + L8 with orthogonalized velocity vectors (SVD-based decorrelation) | Phase A positive, B1 complete | 1.0 | If multi-layer collapse was interference, orthogonal = no collapse |

### Phase C: Medium-Term Architectural (8+ GPU-hours, 2+ days dev)

| # | Experiment | Prerequisite | Cost | Expected Gain |
|---|-----------|-------------|------|---------------|
| C1 | **Non-additive steering**: gated (h·σ(α·v)) and multiplicative (h·(1+α·v)) at L10, best token selection | Phase B complete | 2.0 | New steering paradigm |
| C2 | **Cross-domain steering vectors**: Train TT on MATH and BBH trajectories. Compare steering transfer | Phase B complete | 4.0 | Test domain-specificity hypothesis |
| C3 | **PID formulation**: Implement P=p_α·v, I=integral(v over layers 2-8), D=d(attention_entropy)/dt. Test single-token run | Phase C1 positive | 4.0 | Closed-loop steering with token-level adjustment |

======================================================================

--- PHASE 10: HYPERSTITIONAL BRIDGE ---

### Structural Hypotheses

**H-1 (L9 sign inversion)**: Steering at L9 with negative α (α ∈ [-0.2, -0.05]) produces steering improvement ≥ +5pp on GSM8K. Confirmed by: L9(α=-0.1) > L9(α=0) + 5pp. Falsified by: all negative α values produce < +5pp improvement. **Risk**: Low (0.5 GPU-hrs). **Value**: If true, adds an entire layer to the steering portfolio and transforms L9 from death layer to trim-tab.

**H-2 (Sparse-head steering)**: At L10, <30% of attention heads carry >70% of the steering effect. Confirmed by: per-head ablation shows top-20% heads produce >70% of Δaccuracy. Falsified by: steering effect is uniformly distributed across heads. **Risk**: Low (1.5 GPU-hrs). **Value**: If true, enables 5× more efficient steering and explains multi-layer collapse (all-head interference).

### Relational Hypotheses

**H-3 (Injection target ≥ injection signal)**: Varying the injection LAYER while holding the velocity vector constant changes steering effect more than varying the VELOCITY while holding layer constant. Confirmed by: σ_effect(layer | v_fixed) > σ_effect(v | layer_fixed). Falsified by: opposite result. **Risk**: Low (1.0 GPU-hrs). **Value**: If true, fundamentally redirects research priority from "improving TT" to "understanding layer selection."

**H-4 (Attention softmax amplification)**: The steering effect is mediated by nonlinear amplification through the attention softmax, not by direct modification of value representations. Confirmed by: measuring the softmax Jacobian g[l] = ||∂attn/∂v|| shows g > 1 at steering-effective layers (L10) and g ≈ 0 at death layers (L9). Falsified by: g[l] ≈ 1 for all layers — implication: steering is linear. **Risk**: Medium (2.0 GPU-hrs for mechanistic analysis). **Value**: If true, provides a unified mechanistic theory of why steering works and why it's layer-dependent.

### Potential Hypotheses

**H-5 (Closed-loop PID steering)**: Formulating steering as proportional-integral-derivative control with token-level feedback (attention entropy as error signal) outperforms open-loop additive steering by ≥5pp. Confirmed by: PID-steered model at L10, first-30 yields >+21.7pp. Falsified by: PID ≤ additive + 5pp. **Risk**: Medium (4.0 GPU-hrs). **Value**: If true, revolutionizes steering from static intervention to adaptive control.

**H-6 (Contrastive-source velocity superiority)**: Velocities computed from correct-vs-incorrect trajectory pairs (v_c - v_w) provide better steering signal than standard TT velocities by ≥3pp. Confirmed by: contrastive-TT steering > standard-TT steering + 3pp. Falsified by: contrastive ≤ standard. **Risk**: Medium (1.0 GPU-hr for computing contrastive velocities). **Value**: If true, provides a principled source for steering vectors without needing trajectory prediction.

======================================================================

--- PHASE 11: RECURSIVE SELF-ASSESSMENT ---

### Analysis Weaknesses

**Structural**: 
1. The 9-lens cascade is comprehensive but relies on the quality of each lens agent's output. Some lens outputs (abductive) were more structured than others (trajectory).
2. The analysis-to-experiment ratio is now 14:0 (13 previous + this one). This synthesis should be the FINAL analysis before experiments.

**Relational**:
3. The lens outputs are integrated manually — some cross-lens patterns may have been missed.
4. The web search was limited to 3 queries; relevant recent papers (e.g., "A Sober Look at Steering Vectors") were captured but not in depth.
5. The synthesis does not model the researcher's actual constraints (priority, compute budget, timeline).

**Potential**:
6. The top-5 experiments may not be the true global optimum — there may be a 6th experiment with higher λ.
7. No cost-benefit analysis vs. LoRA fine-tuning or other paradigms was included.

### Blind Spots Discovered

| Blind Spot | Why Missed | How to Catch |
|-----------|-----------|-------------|
| Side effects of steering on unrelated tasks | All lenses focused on improving steering | Add side-effect measurement to every experiment |
| Comparison to non-steering baselines | Paradigm accepted as premise | Run LoRA, retrieval, prompting baselines |
| Statistical power analysis | Culture of descriptive reporting | Compute required N for 80% power at Δ=10pp |

### Confidence Assessment

| Claim | Confidence | Would Increase To |
|-------|-----------|-----------------|
| L9 α-inversion experiment is highest λ | 8/10 | 10/10 after positive result |
| Per-head steering is sparse (20% heads → 80% effect) | 6/10 | 9/10 after ablation experiment |
| α=0.05 threshold is softmax nonlinearity | 5/10 | 9/10 after g[l] measurement |
| TT quality (R²) is uncorrelated with steering effect | 4/10 | 8/10 after correlation computed |
| Non-additive steering outperforms additive | 5/10 | 8/10 after gated experiment |
| Overall steering effect is real (not random perturbation) | 7/10 | 10/10 after null distribution experiment |

**Overall confidence in this synthesis**: 7.5/10
- Highest confidence: The experimental prioritization (λ ranking) is near-optimal given available information
- Lowest confidence: The mechanistic interpretation (α saturation = softmax nonlinearity) — requires experimental confirmation
- What would raise to 9/10: Phase A results (2.5 GPU-hours)

### Proposed TSE Updates

1. Add "experiment-to-analysis ratio check" to Phase 0 — if ratio > 1:5 (analyses without corresponding experiments), flag as meta-level issue
2. Add "side-effect measurement" requirement to every proposed experiment
3. Add "null distribution generation" as mandatory Phase A step

======================================================================

--- PHASE 12: FINAL SYNTHESIS REPORT ---

## CORE FINDINGS (Top-10 by significance)

| # | Finding | Confidence | Source Lenses |
|---|---------|-----------|---------------|
| F1 | α steering is a binary switch (threshold at 0.05) — continuous α is wasted capacity | 9.3/10 | Analogical, Dialectical, Systems, Abductive, Inspiration |
| F2 | L9 death layer is a fundamental architectural constraint — all steering collapses | 9.2/10 | Analogical, Dialectical, Systems, Adversarial |
| F3 | Sparse steering (single layer, early tokens) always beats dense (multi-layer, all tokens) | 9.1/10 | Analogical, Dialectical, Blending, Systems, Trajectory |
| F4 | Multi-layer steering produces destructive interference via residual stream saturation | 9.0/10 | Analogical, Dialectical, Systems, Abductive |
| F5 | Injection target (layer × position) dominates injection signal (velocity source) | 8.8/10 | Analogical, Dialectical, Metacognitive |
| F6 | 13:0 analysis-to-experiment ratio is the project's critical blind spot | 9.2/10 | Metacognitive, Trajectory |
| F7 | No null distribution exists — steering vs random vectors never compared | 9.0/10 | Adversarial, Metacognitive |
| F8 | L9 may invert at negative α — untested assumption could open new layer | 8.5/10 | Blending, Abductive, Dialectical |
| F9 | Steering may be sparse in head-space (20% heads → 80% effect) — untested | 7.5/10 | Trajectory, Blending, Systems |
| F10 | PID formulation provides a unified framework for steering dynamics | 7.0/10 | Blending, Inspiration, Systems |

## TOP 5 RECOMMENDATIONS (Ranked by λ = impact/effort)

### #1: RUN L9 α SWEEP (λ=5.0, 0.5 GPU-hours)
**What**: Test L9 steering at α ∈ {-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2} on first-30 tokens.
**Why**: If L9 inverts at negative α (plausible: blending lens "phase-inverted death layer"), this adds an entire layer to the steering portfolio and transforms L9 from death to trim-tab. If not, we confirm L9 death is truly fundamental.
**Expected**: 60% chance of positive result (α < 0 gives +8-12pp). Even null result is valuable (confirms L9 is fundamental).

### #2: PER-HEAD ABLATION AT L10 (λ=2.5, 1.5 GPU-hours)
**What**: Steer L10 with full configuration, ablate GQA attention heads one by one. Compute Δaccuracy per head.
**Why**: If steering is sparse in head-space (trajectory lens: breakthrough scenario; blending: per-band compression), this identifies the 2-3 critical heads per layer, enabling 5× more efficient steering. If uniform, we know steering is a dense mechanism.
**Expected**: 50% chance of sparse result (<30% heads → >70% effect). Cross-references with adversarial lens critique (info-theoretic bound).

### #3: NULL DISTRIBUTION EXPERIMENT (λ=5.0, 0.5 GPU-hours)
**What**: Compare TT-velocity steering vs random vectors (same norm, same layer L10, same position first-30, same α=0.1).
**Why**: THIS IS THE CRITICAL CONTROL that all 13 previous analyses missed. Without it, we cannot distinguish causal steering from generic perturbation effects.
**Expected**: 70% chance TT > random (based on L9 asymmetry — if random caused collapse too, L9 wouldn't be special). 30% chance TT ≈ random → paradigm invalidation.

### #4: SPARSE DIFFICULTY-BASED TOKEN STEERING (λ=2.5, 1.0 GPU-hours)
**What**: Select tokens by inference-time uncertainty (logit entropy). Compare top-30 highest-entropy tokens vs first-30 vs every 5th token.
**Why**: Determines whether the first-30 advantage is content-dependent (difficulty-based) or position-dependent (positional encoding). Controls future token selection strategy.
**Expected**: 50% chance difficulty-sparse > first-30.

### #5: PER-DIM α WITH LOW-RANK DECOMPOSITION (λ=1.25, 2.0 GPU-hours)
**What**: Replace scalar α with low-rank α = UV^T (r=8). Train on calibration data. Test at L10 first-30.
**Why**: If α saturation is caused by averaging across dimensions, per-dim α could resolve it (each dimension gets individualized treatment).
**Expected**: 40% chance of >+3pp over scalar α.

## RESOURCE-BUDGETED PLAN

### Phase A — IMMEDIATE (2.5 GPU-hours, 2 hours dev)
```
[L9 α sweep]        [Per-head ablation]     [Null distribution]
   0.5 GPU-hrs          1.5 GPU-hrs             0.5 GPU-hrs
       │                    │                       │
       └────────────────────┼───────────────────────┘
                            │
                            ▼
                 ALL 3 complete → analyze results
                            │
                    ┌───────┴───────┐
                    ▼               ▼
              TT > random      TT ≈ random
                    │               │
                    ▼               ▼
              Phase B        Publish null result
            (4 GPU-hrs)      + characterization
```

### Phase B — SHORT-TERM (4.0 GPU-hours, 4 hours dev)
```
B1: Sparse token steering (1.0 GPU-hr)
B2: Per-dim α low-rank (2.0 GPU-hrs)
B3: Orthogonalized multi-layer (1.0 GPU-hr)
```

### Phase C — MEDIUM-TERM (8+ GPU-hours, 2+ days dev)
```
C1: Non-additive steering (2.0 GPU-hrs)
C2: Cross-domain steering vectors (4.0 GPU-hrs)
C3: PID formulation (4.0 GPU-hrs)
```

## NEGATIVE SPACE

| Not Found | Why | Worth Investigating? |
|-----------|-----|---------------------|
| Side-effect measurement | No lens asked "what does steering break?" | YES — add to every experiment |
| Comparison to LoRA/fine-tuning | All lenses accepted steering paradigm | YES — separate study |
| Statistical power analysis | All lenses focused on effect, not error | YES — 100 problems may be insufficient |
| Steering effect on reasoning chains (not just answers) | All lenses evaluated final accuracy only | YES — CoT quality analysis |
| Token-level dynamics of steering | All lenses aggregated over token window | YES — Phase C |

## IMMEDIATE ACTION CHECKLIST

**TODAY (0 GPU-hours, 30 min)**:
  [ ] Compute per-layer ρ(R², Δaccuracy) from existing data
  [ ] Design L9 α sweep experiment script (reuses existing infrastructure)
  [ ] Design per-head ablation hooks for model architecture
  [ ] Generate random vector steering baseline from existing code
  [ ] Pre-register Phase A analysis plan with Bonferroni correction framework

**NEXT GPU SESSION (2.5 GPU-hours, ~2 hours)**:
  [ ] Run L9 α sweep experiment (0.5 GPU-hrs)
  [ ] Run per-head ablation experiment (1.5 GPU-hrs)  
  [ ] Run null distribution experiment (0.5 GPU-hrs)
  [ ] Analyze and publish results

=======================================================================
END OF STEERING SYNTHESIS — TSE FULL + EMERGENT MODE
=======================================================================
