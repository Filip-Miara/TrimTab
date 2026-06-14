# Phase 4b: Emergent Discovery

**Subject**: RankAdaptation — Velocity-based latent steering
**Date**: 2026-06-14

---

## Unconventional Recombinations

### Class 1: Cross-Level (Atom ↔ Peak)

| ID | Atom | Peak Concept | Cross-Level Insight |
|----|------|-------------|-------------------|
| RECOMB-CL1 | A10 (α) | P (RankAdaptation) | α is not just a steering parameter — it's the *epistemic attitude* of the system toward the steering signal. α=0 is "ignore," α→1 is "fully trust." At P-level, the question is: what α should the system have toward its own belief revision threshold? |
| RECOMB-CL2 | A6 (Death Layer) | P (RankAdaptation) | Death layers may be performing *critical organizational computation* whose disruption cascades. At P-level, this suggests the RankAdaptation system should have *guardrails* that detect when the model enters an "organizational crisis" and abort. |
| RECOMB-CL3 | A7 (Capability Threshold) | P (RankAdaptation) | The capability threshold may be a *meta-property* of the RankAdaptation system itself: systems below a certain "understanding" of their own steering mechanism cannot be steered. Self-awareness of steering dynamics may be a prerequisite for effective steering. |

### Class 2: Domain-Transposed

**Domain: Architecture**

Map the entire RankAdaptation pyramid into architecture. Steering = structural reinforcement; trim-tab layers = shear walls; death layers = load-bearing columns (must not be modified).

| Architecture Element | Steering Analogue |
|---------------------|-------------------|
| Shear wall | Trim-tab layer (horizontal force resistance) |
| Load-bearing column | Death layer (must remain unmodified) |
| Reinforcement bar (rebar) | Steering signal (embedded in concrete/activations) |
| Foundation | Baseline model capability |
| Expansion joint | α=0 layer (allows movement without stress) |

**Domain: Sports Strategy**

Map steering into basketball. Trim-tab layers = star player; death layers = point guard (controls all plays); α = minutes played; contrastive TT = film study of opponent.

| Sports Element | Steering Analogue |
|---------------|-------------------|
| Star player isolation | Single-layer steering (L8 gets the ball) |
| Team chemistry disruption | Steering a point guard (L9=death) breaks system |
| Timeout/rest | α=0 for certain tokens (reset) |
| Scouting report | Contrastive TT (study of opponent/correct trajectories) |

**Domain: Cooking**

Map steering into recipe development. TT = taste predictor; α = seasoning amount; trim-tab = ingredient that lifts dish; death layer = ingredient that ruins dish if overdone.

### Class 3: Forbidden Pairs

| ID | Concept A | Concept B | Original Incompatibility | Recombinant Insight |
|----|-----------|-----------|-------------------------|---------------------|
| RECOMB-FP1 | A4: Per-layer selectivity (single layer) | A5/A6: Multi-layer effects | Project only tested single-layer; assumed combinatorial is too complex | *Forbidden combination*: Apply steering at L8 AND L2 simultaneously with different α. The assumption that these are independent might be wrong — they might interact constructively. |
| RECOMB-FP2 | A10: Fixed α=0.1 | A7: Capability threshold (~40%) | Sub-threshold models tested at α=0.1 only | *Forbidden combination*: Sub-threshold model with much larger α. The assumption that threshold is fundamental relies on α=0.1 being the right α. |
| RECOMB-FP3 | A2: Standard TT (descriptive) | A19: Logit correction (failed) | Different surfaces (K/V vs logits) | *Forbidden combination*: Use standard TT prediction to modulate logits instead of K/V. The failure of logit correction used a different training scheme, not the TT prediction. |

### Class 4: Self-Application

**Feed P (RankAdaptation) back through itself as an atom:**

Self-application question: "What does the RankAdaptation system reveal about itself when it treats itself as the subject of its own analysis?"

| Layer of Self-Analysis | Result |
|------------------------|--------|
| RankAdaptation's own "hidden states" | The output files, analysis, and accumulated knowledge are the hidden states of the research project. Their "velocity" is the rate of insight generation. |
| RankAdaptation's "trim-tab layers" | The critical experiments that provide the most insight per hour: per-layer sweep (gave ±20pp signal), cross-model transfer (gave generalization evidence). These are the "trim-tab experiments." |
| RankAdaptation's "death layers" | The experiments that consumed time without producing useful signal: PPL-modulated correction, Qwen3.5-2B steering attempts, Phi-3 deployment. |
| RankAdaptation's "α" | The resource allocation intensity per experiment. Some experiments (contrastive eval) are at α=0 (not yet done); the per-layer sweep was at α=1 (full intensity). |
| RankAdaptation's "capability threshold" | The project needed ~2 sessions of infrastructure building before it reached the "capability threshold" for generating useful scientific insights. Prior to that, all results were infrastructure failures. |
| RankAdaptation's "contrastive signal" | The difference between successful and unsuccessful experiments (v_success − v_failure) reveals: successful experiments test mechanistic hypotheses; failed experiments test infrastructure. |

---

## Emergent Capability Analysis

### EM-1: Self-Correcting Steering Loop

**Source Recombination**: RECOMB-CL1 (α as epistemic attitude) × RECOMB-FP1 (multi-layer interaction)

**Description**: A steering system that monitors its own effect on the model and adjusts α in real-time based on a "health signal" (perplexity, token divergence, or a learned predictor of generation quality). The system steers, observes the effect, and corrects the steering.

**Qualification**:
| Criterion | Answer | Evidence |
|-----------|--------|----------|
| Q1: Qualitatively distinct from constituents? | Y | Neither per-layer selectivity nor α adjustment alone creates a feedback loop; the loop is a new structure |
| Q2: Not predictable from constituent properties alone? | Y | Given knowledge of L8 effect (+20pp) and α range (0.01-1.0), one cannot predict that a closed-loop controller would emerge |
| Q3: Produces synergy > sum in *kind*? | Y | The feedback loop is a *different control paradigm* (closed-loop vs open-loop), not just a larger version |

**Classification**: CONFIRMED EMERGENT

**Trigger Conditions**:
- Requires: real-time health signal monitor
- Requires: adaptive α mechanism
- Threshold: α adjustment latency < 1 token generation time
- Phase transition: from "open-loop experiment" to "closed-loop controller"

**Latent Paths**:
1. Build health monitor → collect health data during steering → train α-predictor
2. Implement as PID controller with neural gain scheduling
3. Minimal viable set: health monitor + α(t) = α_0 × health_signal(t)

### EM-2: Universal Velocity Manifold

**Source Recombination**: RECOMB-CL3 (self-awareness prerequisite) × cross-model transfer knowledge

**Description**: A foundational finding that all transformer language models share a common velocity structure in their hidden states, determined solely by the autoregressive next-token-prediction objective. If true, this would mean a single universal TT could steer any model, and that velocity dynamics are as fundamental to LLMs as gradients are to optimization.

**Qualification**:
| Criterion | Answer | Evidence |
|-----------|--------|----------|
| Q1: Qualitatively distinct from constituents? | Y | Universal manifold is not just cross-model transfer — it's a claim about a shared representation space underlying all LLMs |
| Q2: Not predictable from constituent properties alone? | Y | Cross-model transfer works (SmolLM2→7B), but this doesn't predict a universal manifold spanning ALL architectures and sizes |
| Q3: Produces synergy > sum in *kind*? | Y | A universal manifold would change LLM theory (velocity as fundamental), not just improve steering |

**Classification**: CONFIRMED EMERGENT (conditional on verification)

**Trigger Conditions**:
- Requires: Successful cross-model transfer across ≥5 diverse model families (LLaMA, Mistral, GPT2, Gemma, Phi)
- Threshold: Universal TT works on models never seen during training
- Phase transition: Gradual (more models → stronger evidence)

### EM-3: Attention-Redistribution Steering

**Source Recombination**: RECOMB-FP3 (TT → logit correction) × M6 (Split K/V)

**Description**: Instead of modifying K/V values, steer by modifying the *attention distribution* directly — redistributing attention weights to emphasize tokens that lead toward correct answers. This bypasses the off-manifold problem of K/V modification.

**Qualification**:
| Criterion | Answer | Evidence |
|-----------|--------|----------|
| Q1: Qualitatively distinct from constituents? | Y | Attention redistribution is a different intervention mechanism than value modification |
| Q2: Not predictable from constituent properties alone? | N | Given knowledge of attention mechanisms and steering, one could predict this approach — it's compositional |
| Q3: Produces synergy > sum in *kind*? | N | It's a different kind of steering, not necessarily emergent from constituents |

**Classification**: COMPOSITIONAL

### EM-4: Reasoning Topography Mapping

**Source Recombination**: Domain-transposed (Architecture) × A5/A6 (Trim-Tab/Death Layer classification)

**Description**: Treat each layer's function as a "topographical feature" on a reasoning landscape. Trim-tab layers = peaks (high utility), death layers = fault lines (dangerous to disturb). The map would show the "critical path" of reasoning through the model — which layers must fire in sequence.

**Qualification**:
| Criterion | Answer | Evidence |
|-----------|--------|----------|
| Q1: Qualitatively distinct from constituents? | Y | A topographical map of reasoning is a new representation, not just a collection of layer classifications |
| Q2: Not predictable from constituent properties alone? | Y | Knowing individual layer effects doesn't reveal the path structure between them |
| Q3: Produces synergy > sum in *kind*? | Y | The map enables path-finding (which layers to steer in sequence), which is qualitatively different from layer-at-a-time steering |

**Classification**: CONFIRMED EMERGENT

### EM-5: Anti-Fragile Steering via Death Layer Inoculation

**Source Recombination**: PC-2 (death layers as informative) × M2 (anti-steering)

**Description**: If death layers are the most computationally concentrated, then "inoculating" the model by gently perturbing death layers during training (increasing their robustness) could make them steering-amenable. The hypothesis: death layers become death layers because they're NEVER perturbed during training; they've never needed to develop robustness.

**Qualification**:
| Criterion | Answer | Evidence |
|-----------|--------|----------|
| Q1: Qualitatively distinct from constituents? | Y | Inoculation is a training-time intervention; steering is a generation-time intervention |
| Q2: Not predictable from constituent properties alone? | Y | The brittleness-from-never-perturbed hypothesis is not derivable from the observed death-layer effect alone |
| Q3: Produces synergy > sum in *kind*? | N | Improvement would be quantitative (more layers become steerable), not qualitative |

**Classification**: QUANTITATIVE ENHANCEMENT

---

## Synergy Mapping

### Pairwise Synergy Scores

| Pair | Independent Score Sum | Interaction Score | Synergy | Classification |
|------|----------------------|-------------------|---------|----------------|
| {A4, A10}: Per-layer × α | 5 + 4 = 9 | 13 | +4 | Qualitative (enables α-per-layer) |
| {A8, A5}: Contrastive × Trim-Tab | 6 + 6 = 12 | 10 | -2 | Antagonistic (contrastive may not produce same trim-tabs) |
| {A2, A12}: TT × Cross-Model | 5 + 5 = 10 | 15 | +5 | Qualitative (universal manifold) |
| {A5, A6}: Trim-Tab × Death Layer | 6 + 4 = 10 | 16 | +6 | Qualitative (complementary classification system) |
| {A3, A19}: KV-Cache × Logit | 4 + 2 = 6 | 8 | +2 | Quantitative (dual-surface) |
| {A7, A10}: Threshold × α | 4 + 4 = 8 | 11 | +3 | Qualitative (threshold may be α-dependent) |
| {A6, A2}: Death Layer × TT | 4 + 5 = 9 | 7 | -2 | Antagonistic (TT predicts death-layer velocity faithfully, which includes harmful direction) |
| {A15, A16}: Async × GPU Cache | 3 + 3 = 6 | 6 | 0 | Additive (independent mechanisms) |
| {A14, A17}: Collection × Checkpoint | 3 + 3 = 6 | 6 | 0 | Additive (independent mechanisms) |
| {A1, A8}: Velocity × Contrastive | 5 + 6 = 11 | 14 | +3 | Qualitative (contrastive reinterprets velocity as normative) |

### Higher-Order Synergy: {A4, A10, A8}

| Order | Components | Score |
|-------|-----------|-------|
| Pairwise | {A4, A10} | 13 |
| Pairwise | {A4, A8} | 9 |
| Pairwise | {A10, A8} | 10 |
| Higher (3-way) | {A4, A10, A8} | 18 |
| Higher-Order Score | 18 − (13 + 9 + 10) = | **-14** |

**Interpretation**: Negative higher-order synergy indicates that combining per-layer selectivity, α, and contrastive direction is LESS than the sum of their pairwise interactions. This makes sense — they are somewhat redundant: choosing the right layer and the right α already captures most of the benefit; adding contrastive direction may add little on top.

### Higher-Order Synergy: {A1, A2, A4, A5}

| Order | Components | Score |
|-------|-----------|-------|
| All pairwise | {A1,A2}=8, {A1,A4}=6, {A1,A5}=7, {A2,A4}=5, {A2,A5}=6, {A4,A5}=10 | Sum=42 |
| 3-way | {A1,A2,A4}=9, {A1,A2,A5}=11, {A1,A4,A5}=13, {A2,A4,A5}=10 | Sum=43 |
| 4-way | {A1,A2,A4,A5} | 22 |
| Higher-Order Score | 22 − (43 − 42) = **21** |

**Interpretation**: HIGH positive higher-order synergy. The quadruple {Velocity, TT, Per-Layer, Trim-Tab} produces more than the sum of its pairwise and 3-way interactions. This is the core emergent finding: the full steering-discovery pipeline (collect velocities → train TT → select layer → discover trim-tab) is a self-amplifying system where each component makes the others more valuable. **Self-organization detected**.

---

## Summary

| Category | Count | Top Entries |
|----------|-------|-------------|
| Cross-Level Recombinations | 3 | CL1: α as epistemic attitude, CL2: death layer guardrails, CL3: self-awareness prerequisite |
| Domain-Transposed | 3 | Architecture, Sports, Cooking |
| Forbidden Pairs | 3 | FP1: multi-layer interaction, FP2: sub-threshold × high α, FP3: TT → logits |
| Self-Application | 6 | Project-level trim-tabs, death layers, α, capability threshold |
| **CONFIRMED EMERGENT** | **3** | Self-correcting loop (EM-1), Universal velocity manifold (EM-2), Reasoning topography (EM-4) |
| QUANTITATIVE ENHANCEMENT | 1 | Death layer inoculation (EM-5) |
| COMPOSITIONAL | 1 | Attention-redistribution steering (EM-3) |
| REDUCTIVE | 0 | — |
| **Highest Pairwise Synergy** | **+6** | {A5, A6}: Trim-Tab × Death Layer (complementary classification) |
| **Highest Higher-Order** | **+21** | {A1, A2, A4, A5}: Velocity × TT × Per-Layer × Trim-Tab |
| **Self-Organization** | **YES** | Quadruple {A1,A2,A4,A5} shows strong positive higher-order synergy |
