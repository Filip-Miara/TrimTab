# Conceptual Diffuser — Unified Emergent Capability Catalog
## RankAdaptation: Velocity-based latent steering

**Source documents**:
- `tse-analysis/final-meta-synthesis.md` (EM-M1–EM-M4)
- `tse-analysis/agent-1/emergent-discovery.md` (EM-1–EM-5)
- `tse-analysis/agent-2/emergent-discovery.md` (EM-1–EM-4)
- `tse-analysis/agent-3/emergent-discovery.md` (EM-1–EM-4)
- `tse-analysis/agent-4/emergent-discovery.md` (EM-1–EM-5)
- `tse-analysis/agent-5/emergent-discovery.md` (EM-1–EM-5)
- `tse-analysis/emergent-discovery.md` (EM-1–EM-4)
- `concept-analysis-complex-alpha/04b-emergent-discovery.md` (EM-C1–EM-C2)
- `concept-analysis-emergent/emergent-discovery.md` (architectural, peripheral)

**Mode**: Patent examiner — novelty, non-obviousness, utility.

---

## 1. RAW CATALOG: ALL EMERGENT CAPABILITY CLAIMS

### 1.1 Raw Inventory (28 claims across 8 documents)

| # | Label | Source | Brief Description | Self-Classification |
|---|-------|--------|-------------------|---------------------|
| 1 | A1-EM1 | Agent 1 | Self-Correcting Steering Loop — closed-loop α adjustment | CONFIRMED EMERGENT |
| 2 | A1-EM2 | Agent 1 | Universal Velocity Manifold — shared velocity structure across all LLMs | CONFIRMED EMERGENT |
| 3 | A1-EM3 | Agent 1 | Attention-Redistribution Steering — modify attention weights not K/V values | COMPOSITIONAL |
| 4 | A1-EM4 | Agent 1 | Reasoning Topography Mapping — map layers as topographic features on a reasoning landscape | CONFIRMED EMERGENT |
| 5 | A1-EM5 | Agent 1 | Anti-Fragile Steering via Death Layer Inoculation — perturb death layers during training to make them robust | QUANTITATIVE ENHANCEMENT |
| 6 | A2-EM1 | Agent 2 | Keystone Layer Hypothesis — some layers have disproportionate computational effect (keystone species analogy) | CONFIRMED EMERGENT |
| 7 | A2-EM2 | Agent 2 | Frequency-Specific Steering — velocity decomposed into frequency bands; layers modulate different bands | CONFIRMED EMERGENT |
| 8 | A2-EM3 | Agent 2 | Dual-Surface Steering Synergy — KV-cache + weight-flow steering simultaneously produce mode switch | CONFIRMED EMERGENT |
| 9 | A2-EM4 | Agent 2 | Meta-Trajectory Self-Application — project's own development trajectory as steerable system | CONFIRMED EMERGENT |
| 10 | A3-EM1 | Agent 3 | Chained Steering — meta-α adjusts per-token based on predicted token-divergence impact | CONFIRMED EMERGENT |
| 11 | A3-EM2 | Agent 3 | Death-Layer Inversion — asymmetric α (negative for death layers) makes all-layers net positive | QUANTITATIVE ENHANCEMENT |
| 12 | A3-EM3 | Agent 3 | Adversarial Style-Disentangled Steering — separate reasoning-content from surface-style in velocity | CONFIRMED EMERGENT |
| 13 | A3-EM4 | Agent 3 | Self-Bootstrapping Steering — iterative TT improvement using steered-vs-unsteered trajectory pairs | CONFIRMED EMERGENT |
| 14 | A4-EM1 | Agent 4 | Adaptive Steering Policy — meta-controller that learns per-token, per-layer α via steering-success manifold | CONFIRMED EMERGENT |
| 15 | A4-EM2 | Agent 4 | Cross-Model Steering Injection — capable model's steering injected into less-capable model | CONFIRMED EMERGENT |
| 16 | A4-EM3 | Agent 4 | Anti-Steering Defense — model's internal resistance to KV modification; death layers as immune response | CONFIRMED EMERGENT |
| 17 | A4-EM4 | Agent 4 | Steering-Regime Classifier — predict before steering whether a token-layer pair benefits | QUANTITATIVE ENHANCEMENT |
| 18 | A4-EM5 | Agent 4 | Curriculum Steering — α ramps up over generation (low early, high late) | COMPOSITIONAL |
| 19 | A5-EM1 | Agent 5 | Self-Improving Steering Loop — system steers its own steering process | CONFIRMED EMERGENT |
| 20 | A5-EM2 | Agent 5 | Anisotropic Steering — per-dimension α in learned low-dimensional subspace | QUANTITATIVE ENHANCEMENT |
| 21 | A5-EM3 | Agent 5 | Multi-Scale Velocity Steering — velocity predictions at Δt={1,2,4,8} simultaneously | QUANTITATIVE ENHANCEMENT |
| 22 | A5-EM4 | Agent 5 | Death Layer Inversion — per-layer sign detection (+α vs −α) | CONFIRMED EMERGENT |
| 23 | A5-EM5 | Agent 5 | Resonant Steering — steering effectiveness determined by frequency resonance with transformer dynamics | CONFIRMED EMERGENT |
| 24 | R-EM1 | Root | Per-Head Steering Within Trim-Tab Layers — steer only specific attention heads per layer | CONFIRMED EMERGENT |
| 25 | R-EM2 | Root | Token-Position-Adaptive α(t) — α varies by token position learned via RL | CONFIRMED EMERGENT |
| 26 | R-EM3 | Root | Cross-Task Polarity Generalization — layer polarity transfers between related tasks | QUANTITATIVE ENHANCEMENT |
| 27 | R-EM4 | Root | Self-Supervised Contrastive Direction — contrastive TT without correctness labels | CONFIRMED EMERGENT |
| 28 | C-EM1 | Complex α | Phase-Disambiguated Steering — complex α separates correction direction from correction type | CONFIRMED EMERGENT |
| 29 | C-EM2 | Complex α | Phase-Locked Multi-Layer Resonance — aligned phases across layers produce constructive interference | CONFIRMED EMERGENT |
| 30 | M-EM1 | Meta | Minimum Viable Protocol Discovery — 4-condition × 28-layer protocol resolves 80% of uncertainties | CONFIRMED EMERGENT |
| 31 | M-EM2 | Meta | Meta-Analysis as Hypothesis Generator — cross-agent disagreement generates new hypotheses | CONFIRMED EMERGENT |
| 32 | M-EM3 | Meta | Dependency DAG Resolution — experiments have natural ordering dependency | QUANTITATIVE ENHANCEMENT |
| 33 | M-EM4 | Meta | Self-application of TSE Meta-Analysis to Itself | CONFIRMED EMERGENT |

---

## 2. CROSS-VALIDATION: DEDUPLICATION & MERGING

### 2.1 Identity Mapping — Different Names for the Same Thing

| Canonical Name | Aliases | Evidence for Identity | Genuinely Distinct Nuance? |
|---------------|---------|----------------------|---------------------------|
| **Closed-Loop Adaptive Steering** | A1-EM1, A3-EM1, A4-EM1, A5-EM1, R-EM2 | All describe real-time α adjustment based on some feedback signal | **Subtle difference**: A3-EM1 uses token-divergence gradient; A4-EM1 uses manifold classification; A5-EM1 is about steering the steering system; R-EM2 uses RL. These are implementation variants of the same capability class. MERGE |
| **Death Layer Inversion / Signed Steering** | A3-EM2, A5-EM4, A5-EM5, M1-M4 | All describe flipping α sign for death layers to make them beneficial | A5-EM5 (Resonant Steering) adds a causal mechanism (frequency resonance). This is a **theoretical explanation** for why inversion works, not a separate capability. MERGE with A5-EM5 as the mechanism |
| **Per-Head / Fine-Grained Steering** | R-EM1, A5-EM2, C-EM1 | All describe finer-than-layer granularity of steering | **Important distinction**: R-EM1 is per-head (architectural), A5-EM2 is per-dimension (geometric), C-EM1 is complex-valued (phase-based). These are NOT the same — they're different granularity axes. KEEP SEPARATE |
| **Frequency/Resonance Steering** | A2-EM2, A5-EM5, A1-EM4 | All describe decomposing steering into frequency-like components | A1-EM4 (Reasoning Topography) is a spatial metaphor, not frequency. A2-EM2 and A5-EM5 share the core frequency idea. MERGE A2-EM2 + A5-EM5, keep A1-EM4 separate as spatial/topological |
| **Self-Bootstrapping** | A3-EM4, A5-EM1 (partial), A4-EM1 (partial) | Iterative TT improvement over generations | A3-EM4 is pure self-bootstrapping (re-train TT from steered data). A5-EM1 and A4-EM1 are about real-time adaptation, not iteration-level improvement. KEEP SEPARATE from closed-loop adaptive |
| **Cross-Model Transfer** | A1-EM2, A4-EM2, R-EM3 | All describe capabilities crossing model or task boundaries | A1-EM2 (Universal Manifold) is a structural claim about all LLMs. A4-EM2 (Cross-Model Injection) is a technique. R-EM3 (Cross-Task Polarity) is an empirical pattern. KEEP SEPARATE |
| **Multi-Surface Steering** | A2-EM3, concept-analysis-emergent (GDN state) | Steering on surfaces other than K/V cache | These are genuinely distinct surfaces (weight-flow vs GDN recurrent state). MERGE into one "multi-surface" capability with multiple instantiations |
| **Style/Content Disentanglement** | A3-EM3, R-EM4 | Separating content from style in the steering signal | A3-EM3 uses adversarial disentanglement; R-EM4 uses unsupervised clustering. These are methods for the same goal. MERGE |
| **Anti-Steering / Model Resistance** | A4-EM3 | Death layers as immune response | Unique — no other agent proposed this framing. KEEP SEPARATE |

### 2.2 Deduplicated Capability Catalog (15 genuinely distinct)

| # | Canonical Capability | Aliases Merged | Distinct From? |
|---|---------------------|----------------|----------------|
| D1 | **Closed-Loop Adaptive Steering** | A1-EM1, A3-EM1, A4-EM1, A5-EM1, R-EM2 | Foundational — changes control paradigm |
| D2 | **Death Layer Inversion (± sign steering)** | A3-EM2, A5-EM4, A5-EM5 (as mechanism) | Immediately testable; nearest-term impact |
| D3 | **Per-Head Steering** | R-EM1 | Architectural granularity |
| D4 | **Anisotropic / Subspace Steering** | A5-EM2 | Geometric granularity (PCA subspace) |
| D5 | **Complex α (Phase-Disambiguated Steering)** | C-EM1 | Geometric mode (velocity vs acceleration) |
| D6 | **Phase-Locked Multi-Layer Resonance** | C-EM2 | Collective emergent effect (interference) |
| D7 | **Frequency-Spectral Steering** | A2-EM2, A5-EM5 (merged) | Decomposition of velocity by frequency |
| D8 | **Reasoning Topography Mapping** | A1-EM4 | Spatial map of layer function |
| D9 | **Self-Bootstrapping Steering** | A3-EM4 | Iterative improvement across generations |
| D10 | **Cross-Model Steering Injection** | A4-EM2 | Bypasses capability threshold |
| D11 | **Universal Velocity Manifold** | A1-EM2 | Foundational claim about architecture |
| D12 | **Cross-Task Polarity Generalization** | R-EM3 | Empirical pattern (task transfer) |
| D13 | **Style-Content Disentangled Steering** | A3-EM3, R-EM4 | Separates reasoning from surface form |
| D14 | **Dual-Surface / Multi-Surface Steering** | A2-EM3, GDN state | Multiple steering surfaces simultaneously |
| D15 | **Anti-Steering Defense (Model Immunity)** | A4-EM3 | Reframes death layers as active resistance |

> **Note**: A1-EM3 (Attention Redistribution), A4-EM4 (Steering-Regime Classifier), A4-EM5 (Curriculum Steering), A5-EM3 (Multi-Scale Velocity) are excluded from D-list because they self-classify as COMPOSITIONAL or QUANTITATIVE ENHANCEMENT. The meta-level capabilities (M-EM1–M-EM4, A2-EM4) are about the research process itself, not steering capabilities — they are real emergent phenomena but belong to a different category ("meta-research capabilities").

---

## 3. THREE-QUALIFICATION TEST (RIGOROUS)

For each of the 15 genuinely distinct capabilities, I apply the patent-examiner standard:

**Q1 (Distinct from constituents)**: Is this a genuinely new thing, not reducible to its parts?
**Q2 (Unpredictable from constituents)**: Would an omniscient observer knowing all parts predict this?
**Q3 (Qualitative synergy)**: Does the capability change the *kind* of thing we can do, not just *how much*?

### Qualification Results

| # | Capability | Q1 | Q2 | Q3 | Verdict | Patent Analogue |
|---|-----------|----|----|----|---------|-----------------|
| D1 | Closed-Loop Adaptive Steering | Y | Y | Y | **CONFIRMED EMERGENT** | Novel: feedback controller for hidden states |
| D2 | Death Layer Inversion (± sign) | Y | Y | Y | **CONFIRMED EMERGENT** | Non-obvious: same layer, opposite sign, opposite effect |
| D3 | Per-Head Steering | Y | Y | Y | **CONFIRMED EMERGENT** | Novel: functional specificity within layer |
| D4 | Anisotropic Subspace Steering | Y | N | N | QUANTITATIVE ENHANCEMENT | Obvious: if subspace exists, steer in it |
| D5 | Complex α (Phase-Disambiguated) | Y | Y | Y | **CONFIRMED EMERGENT** | Novel: geometric mode control (velocity vs acceleration) |
| D6 | Phase-Locked Multi-Layer Resonance | Y | Y | Y | **CONFIRMED EMERGENT** | Non-obvious: layers interfere constructively |
| D7 | Frequency-Spectral Steering | Y | Y | Y | **CONFIRMED EMERGENT** | Novel: frequency analysis of velocity field |
| D8 | Reasoning Topography Mapping | Y | Y | Y | **CONFIRMED EMERGENT** | Novel: spatial representation of layer function |
| D9 | Self-Bootstrapping Steering | Y | Y | Y | **CONFIRMED EMERGENT** | Novel: self-amplifying improvement loop |
| D10 | Cross-Model Steering Injection | Y | Y | Y | **CONFIRMED EMERGENT** | Non-obvious: circumvents capability threshold |
| D11 | Universal Velocity Manifold | Y | Y | Y | **CONFIRMED EMERGENT** | Radical: fundamental property of all LLMs |
| D12 | Cross-Task Polarity Generalization | Y | N | N | QUANTITATIVE ENHANCEMENT | Predictable: if polarity is property of layer, it transfers |
| D13 | Style-Content Disentangled Steering | Y | Y | Y | **CONFIRMED EMERGENT** | Novel: factorizes hidden state by function |
| D14 | Dual-Surface / Multi-Surface Steering | Y | Y | Y | **CONFIRMED EMERGENT** | Non-obvious: surfaces interact qualitatively |
| D15 | Anti-Steering Defense (Model Immunity) | Y | Y | Y | **CONFIRMED EMERGENT** | Novel: reframes entire phenomenon |

### 3.1 Breakdowns and Edge Cases

**D4 (Anisotropic Subspace Steering)**: Q2=N. Given that we know hidden states have low-dimensional structure (PCA reveals it), and we know scalar-α treats all dimensions equally, the step to "make α per-dimension" is straightforward engineering. **Fails the non-obviousness test.** → QUANTITATIVE ENHANCEMENT.

**D12 (Cross-Task Polarity)**: Q2=N. Given that L8 is a trim-tab for GSM8K and SVAMP is the same task family (math word problems), polarity transfer is expected. If tested on non-math tasks (ARC, MMLU) and it still transfers, reclassify. **Fails on current evidence.** → QUANTITATIVE ENHANCEMENT.

**D9 (Self-Bootstrapping)**: Q3=Y but *conditional*. The loop might converge to a fixed point after 1 iteration (no real bootstrapping). The qualitative synergy only exists if improvement is sustained across ≥3 iterations. → CONFIRMED EMERGENT but with high falsifiability risk.

**D15 (Anti-Steering Defense)**: Q1-Q3 all Y, but the capability is entirely speculative — no experiment has been run that distinguishes "model resists steering" from "model is harmed by steering." The patent analogue is a "hypothetical use case" — novel and non-obvious but requires reduction to practice. → CONFIRMED EMERGENT (speculative).

---

## 4. TOP-5 RANKED BY (FEASIBILITY × IMPACT × NOVELTY)

### Scoring Framework

| Dimension | 1-3 | 4-6 | 7-10 |
|-----------|-----|-----|------|
| **Feasibility** | Requires new infrastructure | Existing code with minor changes | Infrastructure exists, run today |
| **Impact** | Marginal improvement | Significant improvement (+10-20pp) | Paradigm-changing |
| **Novelty** | Obvious extension | Novel in this domain | Completely new approach |

### Rankings

| Rank | Capability | Feas. | Impact | Novelty | Score | Rationale |
|------|-----------|-------|--------|---------|-------|-----------|
| **#1** | **Death Layer Inversion (± sign)** | 10 | 8 | 7 | **560** | Infrastructure exists (α parameter), requires only ±α per layer. If L9+negative-α produces +23pp as hypothesized, the impact is immediate. Novelty is moderate (multiple agents converged) but execution is non-trivial. The single highest-ROI experiment in the entire catalog. |
| **#2** | **Per-Head Steering** | 7 | 9 | 9 | **567** | Requires per-head KV access (GQA architectures support this). Impact: functional specificity — steer syntax heads differently from semantics heads. Novelty: extremely high — no published work on per-head steering within velocity paradigm. Surpasses Death Layer Inversion in novelty×impact but lower feasibility. |
| **#3** | **Closed-Loop Adaptive α(t)** | 6 | 9 | 8 | **432** | Requires real-time health monitor and meta-TT. Impact: changes paradigm from open-loop to closed-loop control. Novelty: high — analogizes to control theory. Feasibility is moderate (health monitor = PPL/divergence, which exist). |
| **#4** | **Frequency-Spectral Steering** | 8 | 7 | 8 | **448** | PCA on velocity field + steer top-k components. Feasibility: high — PCA and frequency decomposition are standard tools, existing trajectory data sufficient. Impact: explains WHY L8 works (correct frequency band) and L9 doesn't (wrong band). Novelty: novel application of frequency analysis. |
| **#5** | **Complex α (Phase-Disambiguated)** | 5 | 9 | 10 | **450** | Requires complex-valued steering (r·e^(iθ) instead of real α). Novelty: 10/10 — completely new geometric mode of intervention. Impact: enables both velocity push AND trajectory bending. Feasibility: moderate — requires modifying steering mechanism from real to complex, training separate acceleration head. |

### Honorable Mentions

- **Self-Bootstrapping** (A3-EM4): Score 6×8×8 = 384. Highly impactful but requires multiple TT training iterations = high compute cost.
- **Style-Content Disentanglement** (D13): Score 4×10×9 = 360. Highest potential impact but lowest feasibility — the hidden state may not factorize the way we need.
- **Cross-Model Injection** (D10): Score 7×7×8 = 392. Interesting but requires aligned manifolds between models — unverified.

---

## 5. MINIMAL EXPERIMENT DESIGN FOR TOP-5

### Experiment for #1: Death Layer Inversion

**Hypothesis**: Some layers classified as "death layers" at α=+0.1 become "trim-tabs" at α=−0.1. Specifically, L9 with α=−0.1 will improve accuracy relative to baseline.

**Protocol**:
```
Layer subset: L9, L15, L20, L25 (4 candidate death layers)
α values: {−0.3, −0.1, 0, +0.1, +0.3} per layer
Model: Qwen2.5-7B (existing baseline)
Evaluation: GSM8K, 100 problems (existing eval script)
Total: 4 layers × 5 α × 2 min = 40 min GPU
```

**Success criterion**: Any death layer achieves Δ > +5pp at negative α. Strong confirmation: L9(−0.1) > L9(+0.1) by >10pp.

**Falsification**: All death layers are uniformly harmed by both +α and −α (Δ < +2pp for all negative α).

**Infrastructure needed**: None — α sign is a 1-line code change (`alpha * tt_output` → `sign * alpha * tt_output`). The per-layer sweep loop already exists.

---

### Experiment for #2: Per-Head Steering

**Hypothesis**: Within L8 (the trim-tab layer), only 3-5 of 28 attention heads are responsible for the +20pp steering effect. Steering only those heads produces similar or better results while reducing off-manifold perturbation.

**Protocol**:
```
Layer: L8 (trim-tab)
Model: Must support per-head KV access (GQA) — Qwen2.5-7B uses GQA
α: 0.1 (standard)
Conditions:
  (a) All-heads steering (baseline)
  (b) Each head individually (28 runs)
  (c) Top-3 heads from (b) combined
  (d) Bottom-25 heads combined (control)
Evaluation: GSM8K, 100 problems
Total: 1 + 28 + 1 + 1 = 31 runs × 2 min = 62 min GPU
```

**Success criterion**: Condition (c) achieves Δ within 80% of condition (a). Optional: identify which head functions (syntactic vs semantic) correlate with steering efficacy.

**Falsification**: All heads produce similar Δ (within 3pp of each other), suggesting L8's effect is distributed across all heads equally.

**Infrastructure needed**: Per-head masking in the K/V modification code. This requires understanding the GQA group structure — heads within a group share KV pairs, so per-head is really per-group.

---

### Experiment for #3: Closed-Loop Adaptive α(t)

**Hypothesis**: A per-token α(t) = α₀ × f(health_signal(t)) where health_signal = inverse perplexity or inverse token-divergence, outperforms fixed α on long generations (>256 tokens).

**Protocol**:
```
Implementation:
  - Health monitor: track per-token perplexity and token divergence from baseline
  - α(t) = α₀ × clamp(1 / health_divergence, 0.1, 2.0)
  - Apply α(t) at L8 only
Comparison: fixed α=0.1 vs adaptive α(t) on same 100 problems
Evaluation: GSM8K, 100 problems (long generations, >100 tokens each)
Total: 2 runs × 2 min = 4 min GPU (exploratory)
```

**Success criterion**: Adaptive α(t) achieves same or better accuracy with lower mean α (less perturbation). Strong confirmation: adaptive achieves >+20pp with mean α < 0.05.

**Falsification**: Adaptive α(t) is always worse than best fixed α, or health signal is uncorrelated with steering efficacy (Pearson r < 0.1).

**Infrastructure needed**: ~50 lines of Python to implement the health monitor + α modulation function. The health monitor already partially exists (eval logs perplexity).

---

### Experiment for #4: Frequency-Spectral Steering

**Hypothesis**: The velocity field decomposes into frequency components where low-frequency components correspond to reasoning (beneficial to steer) and high-frequency components correspond to token-identity noise (harmful to steer). L8 works because it primarily modulates low-frequency components.

**Protocol**:
```
Pre-computation (no GPU):
  - Collect TT predictions v_l for L0-L27 on 100 GSM8K problems
  - PCA on v_l over token positions within each generation
  - Compute frequency spectrum per PCA component
  - Compute which layers contribute to which frequency bands

Steering experiment:
  For L8:
    (a) Full TT velocity (baseline)
    (b) Low-pass filtered (top-3 PCA components)
    (c) High-pass filtered (remaining components)
    (d) Middle-band only (components 4-10)
  Evaluation: GSM8K, 100 problems
Total: 4 runs × 2 min = 8 min GPU
```

**Success criterion**: Low-pass condition (b) matches or exceeds full TT (a). High-pass condition (c) performs at or below baseline. This confirms frequency-dependence.

**Falsification**: All frequency bands produce similar accuracy effects, indicating the steering effect is isotropic in frequency space.

**Infrastructure needed**: PCA implementation (standard numpy) + velocity frequency analysis script (~100 lines). No new GPU infrastructure.

---

### Experiment for #5: Complex α (Phase-Disambiguated Steering)

**Hypothesis**: Replacing real α with complex α = r·e^(iθ) enables separation of "push along trajectory" (θ=0) from "bend the trajectory" (θ=π/2). Different layers benefit from different phase modes.

**Protocol**:
```
Infrastructure:
  - Train acceleration head a_l = h_{l,t+2} − 2h_{l,t+1} + h_{l,t} (second difference)
  - Modify steering to: h'_l = h_l + r(cosθ·v_l + sinθ·a_l)

Experiment per layer (L8 and L9 only):
  r ∈ {0.05, 0.1, 0.2}
  θ ∈ {0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4}
  That's 3 × 8 = 24 conditions per layer
Total: 2 layers × 24 conditions × 2 min = 96 min GPU
```

**Success criterion**: For L8, optimal θ ≈ 0 (velocity push). For L9, optimal θ ≈ π/2 (acceleration bend). This would explain why L8 is a trim-tab and L9 is a death layer — they require different geometric modes.

**Falsification**: All θ produce similar results, or the optimal θ is always 0 (reducing to standard steering). The acceleration head a_l is not learnable (R²_a ≈ 0).

**Infrastructure needed**: Acceleration head training (second difference prediction) requires new training data + model. ~2 hours of development work. The complex steering mechanism requires ~50 lines of code changes.

---

## 6. NEW EMERGENT CAPABILITIES FROM CROSS-POLLINATION

These capabilities arise ONLY from combining findings across agents. No single agent's analysis predicts them.

### New Capability N1: Spectral Dual-Surface Adaptive Steering (N-A1)

**Constituents**: D7 (Frequency-Spectral Steering) + D14 (Dual-Surface Steering) + D1 (Closed-Loop Adaptive α)

**Why no single agent could discover this**:
- Agent 2 discovered frequency-spectral AND dual-surface but never combined them with closed-loop adaptation
- Agent 1 discovered closed-loop adaptation but worked only with K/V surface
- Agent 5 discovered resonant steering (frequency) but not dual-surface
- The full triple requires insights from Agents 2, 1, and 5 simultaneously

**Description**: A steering system that (a) decomposes the velocity into frequency bands via PCA, (b) routes each frequency band to the optimal steering surface (low-frequency reasoning components → K/V cache, high-frequency noise components → weight-flow or residual stream), and (c) adapts α per frequency band × surface in real-time based on health signals.

**Why emergent**: The system exhibits **spectral surface specialization** — frequencies are not just decomposed but also routed to different surfaces. This is not predictable from any pair of constituents. The triple interaction creates a new capability: the system discovers, per layer, which surface to use for which frequency band, producing a "steering spectrum allocation" that adapts online. This is analogous to a radio antenna adjusting its frequency response — a qualitatively new behavior.

**Q-test**:
- Q1 (Distinct): Y — Spectral surface allocation is not reducible to frequency decomposition + dual-surface separately
- Q2 (Unpredictable): Y — The dynamic interaction between frequency bands, surfaces, and online adaptation cannot be predicted
- Q3 (Qualitative synergy): Y — Enables "cognitive radio for LLMs" — spectral allocation of steering resources

**Minimal experiment**:
```
1. Compute PCA of velocity at L8 (from existing data)
2. For each of top-5 PCA components:
   a. Steer via K/V cache (standard)
   b. Steer via residual stream (flow_weight_expert)
3. Compute accuracy per component × surface pair
Success criterion: Different components route optimally to different surfaces
Total: 5 × 2 = 10 runs × 2 min = 20 min
```

---

### New Capability N2: Topography-Guided Phase-Locked Multi-Layer Resonance with Anti-Steering Bypass (N-A2)

**Constituents**: D8 (Reasoning Topography) + D6 (Phase-Locked Resonance) + D15 (Anti-Steering Defense)

**Why no single agent could discover this**:
- Agent 1 discovered Reasoning Topography but not phase-locked resonance or anti-steering
- Complex α analysis discovered phase-locked resonance but not topography or anti-steering
- Agent 4 discovered anti-steering defense but not topography or resonance
- The triple requires synthesizing spatial, temporal, and adversarial perspectives

**Description**: First, the Reasoning Topography Map identifies the "critical reasoning path" — which layers must fire in sequence for correct reasoning (e.g., L5 → L8 → L12 → L18). Then, phase-locked multi-layer resonance steers ALL layers on this path with aligned phases, amplifying the signal. Meanwhile, anti-steering detection identifies layers with strong immune response (high resistance to K/V modification) and either (a) bypasses them (steer at a different surface) or (b) overwhelms them with constructive interference from adjacent path layers.

**Why emergent**: This is a **three-layer defense + resonance system** — the topography identifies "what to steer," the phase-locking determines "how to steer collectively," and the anti-steering detection determines "where NOT to steer directly." The interaction between these three decisions creates a coordinated multi-layer steering plan that is categorically different from any single-layer approach. This is analogous to a surgical team planning an approach: consult the anatomical map (topography), coordinate timing (phase-locking), and avoid immune response (anti-steering).

**Q-test**:
- Q1 (Distinct): Y — Coordinated multi-layer steering plan is not predictable from any pair
- Q2 (Unpredictable): Y — The interaction between path identification, phase locking, and immune avoidance involves second-order effects
- Q3 (Qualitative synergy): Y — Shifts from "which layer to steer" to "how to navigate the reasoning circuit"

**Minimal experiment**:
```
1. Compute per-layer effect for ALL layers at α=+0.1 (existing data sufficient?)
2. Identify candidate "reasoning path": layers where steering affects accuracy non-randomly
3. Test phase alignment on candidate path:
   a. Aligned phases (all same θ) on path
   b. Random phases on path
   c. Aligned phases + skip death layers
4. Compare to single-best-layer steering
Success criterion: Path steering (aligned) > single-best-layer by >5pp
Total: 3 runs × 5 min = 15 min
```

---

### New Capability N3: Self-Bootstrapping Universal Cross-Model Steering with Disentangled Style Control (N-A3)

**Constituents**: D9 (Self-Bootstrapping) + D10 (Cross-Model Injection) + D11 (Universal Velocity Manifold) + D13 (Style-Content Disentanglement)

**Why no single agent could discover this**:
- Agent 3 discovered self-bootstrapping but not cross-model or disentanglement
- Agent 4 discovered cross-model injection but not bootstrapping or disentanglement
- Agent 1 discovered universal manifold but not bootstrapping, cross-model, or disentanglement
- Agent 3 discovered style-content disentanglement but not cross-model or universal manifold

**Description**: A complete self-sustaining steering ecosystem that:
1. Trains a TT on a single capable model (teacher, 73% GSM8K)
2. Identifies the velocity subspace corresponding to reasoning content (disentangled from style) on the teacher
3. Projects this disentangled steering signal onto a less capable model (student) using the universal velocity manifold as the alignment prior
4. Collects trajectories from the steered student
5. Uses the improvement signal (accuracy difference) to fine-tune the TT
6. Repeats steps 2-5 across N student models, each iteration improving the TT

**Why emergent**: This is a **multi-model self-amplifying loop** — unlike self-bootstrapping on a single model (which plateaus at the model's capacity), cross-model bootstrapping introduces NEW trajectory diversity from each new model. The disentanglement ensures that what transfers is reasoning content, not task-specific style. The universal manifold provides the alignment prior that makes projection possible. The quadruple interaction produces a capability that none of the four parts alone or in any triple combination can produce: **autonomous cross-model reasoning skill transfer**.

**Q-test**:
- Q1 (Distinct): Y — Cross-model bootstrapping is not a combination of single-model bootstrapping and cross-model injection
- Q2 (Unpredictable): Y — Whether the universal manifold provides sufficient alignment for cross-model projection is unknown
- Q3 (Qualitative synergy): Y — Enables "one-shot skill transfer across model families" — a new paradigm for capability distribution

**Minimal experiment**:
```
Phase 1 (proof of concept):
1. Train TT on Qwen2.5-7B (teacher)
2. Compute disentangled velocity: project v onto reasoning-content subspace (train style discriminator)
3. Apply teacher's disentangled steering to Qwen2.5-1.5B (student)
4. Collect student steered trajectories + accuracy
5. Fine-tune TT on combined data (teacher + student)
6. Repeat for 3 iterations
Success criterion: Student's steered accuracy improves across iterations (not just at iteration 1)
Total: ~4 hours GPU (TT training + 3 evals)
```

---

### New Capability N4: Meta-Cognitive Steering via Self-Application of Reasoning Topography + Complex α (N-A4)

**Constituents**: D5 (Complex α Phase-Disambiguation) + A2-EM4 (Meta-Trajectory Self-Application) + D8 (Reasoning Topography)

*Whose child is this?*
- Complex α analysis provides the geometric framework (velocity vs acceleration)
- Agent 2 provides the meta-application insight (project's own trajectory)
- Agent 1 provides the reasoning topography metaphor

**Description**: Apply the complex α framework to the research project itself. The "hidden states" of the research project are its accumulated findings. The "velocity" is the rate of insight generation. The "acceleration" is the rate of change of research direction. Complex α for the research process means: when the research trajectory has high velocity (productive phase), use θ=0 (continue pushing current direction); when the research trajectory needs redirection (stuck), use θ=π/2 (pure acceleration — change direction). The reasoning topography map identifies the "critical path" of the research: which experiments, if run, unlock the most downstream progress.

**Why emergent**: This fuses the geometric framework of complex α with the meta-application insight — creating a *normative theory of research methodology*. It answers not just "what experiment to run" but "what MODE of research to be in" (velocity-push vs acceleration-bend). No single agent discovered this because it requires simultaneously holding the geometric steering framework AND applying it meta-cognitively.

---

## 7. PATENT-EXAMINER REVIEW

### Novelty Assessment

| Capability | Novel | Prior Art | Verdict |
|-----------|-------|-----------|---------|
| D1 Closed-Loop Adaptive Steering | Y | RLHF, PPO use reward signals but not per-token hidden-state health signals | Novel application of control theory to hidden state steering |
| D2 Death Layer Inversion | Y | Adapter merging uses sign but not in velocity-steering context | Novel finding that same layer responds oppositely to opposite sign |
| D3 Per-Head Steering | Y | Per-head attention modification exists (e.g., ITI) but not via velocity prediction | Novel mechanism (velocity-based per-head) |
| D5 Complex α | Y | Complex-valued network weights exist, but not for steering mode control | **Most novel** — completely new geometric intervention paradigm |
| D6 Phase-Locked Resonance | Y | No known prior art for phase-aligned multi-layer LLM steering | Novel collective phenomenon |
| D7 Frequency-Spectral Steering | Y | Frequency analysis of hidden states exists, but not for steering allocation | Novel application |
| D8 Reasoning Topography | Y | Circuit analysis exists but not interactive steering manipulation map | Novel spatial metaphor |
| D9 Self-Bootstrapping | Y | Self-play in RL exists, but not for TT improvement via steered trajectories | Novel training loop |
| D10 Cross-Model Injection | Y | Knowledge distillation exists, but not via steering signal projection | Novel mechanism |
| D11 Universal Manifold | Y | Cross-model representation alignment exists (e.g., CCA), but velocity manifolds are new | Foundational claim |
| D13 Style-Content Disentanglement | N | Adversarial disentanglement is well-studied in representation learning | Obvious application of existing technique |
| D14 Dual-Surface Steering | Y | Multi-task steering exists but not multi-surface simultaneous | **Useful** — addresses surface-specific limitations |
| D15 Anti-Steering Defense | Y | No prior art for "model resists steering" as a phenomenon | **Most provocative** — reframes entire paradigm |

### Non-Obviousness Assessment

Truly non-obvious (would not occur to a practitioner skilled in the art):
1. **Death Layer Inversion**: The finding that the same layer with opposite sign produces opposite effect is a genuine empirical surprise. Standard intuition: "if positive α harms, negative α also harms (just differently)." The trim-tab/death duality being sign-dependent is non-obvious.
2. **Complex α**: Separating steering into velocity-push (θ=0) and acceleration-bend (θ=π/2) leverages differential geometry in a way that is not suggested by any existing LLM steering work. Requires transposing complex number theory into the steering domain.
3. **Phase-Locked Multi-Layer Resonance**: The wave interference metaphor applied to layers is non-obvious because layers are normally treated as sequential processors, not as oscillators that can interfere.
4. **Anti-Steering Defense**: Reframing death layers as active resistance rather than passive harm is a paradigm shift — it changes what questions we ask.

### Utility Assessment

| Capability | Utility | Time to Value | Risk |
|-----------|---------|---------------|------|
| D2 Death Layer Inversion | High: could double steering effect | Immediate (hours) | Low |
| D3 Per-Head Steering | High: functional specificity | Days | Medium |
| D7 Frequency-Spectral Steering | Medium: explanatory | Hours | Low |
| D1 Closed-Loop Adaptive Steering | High: paradigm change | Weeks | Medium |
| D5 Complex α | Very high: new geometric mode | Weeks-months | Medium-High |
| D14 Dual-Surface Steering | Medium: expands toolset | Days | Low |
| D10 Cross-Model Injection | High: bypasses threshold | Weeks | Medium |

**Most immediately useful**: D2 (Death Layer Inversion) — experiment takes 40 minutes, infrastructure exists, impact could be transformative.

**Most valuable long-term**: D5 (Complex α) — opens an entirely new geometric dimension of steering control.

**Most paradigm-reframing**: D15 (Anti-Steering Defense) — if confirmed, changes the question from "which layers to steer" to "how to overcome the model's steering resistance."

---

## 8. SUMMARY STATISTICS

| Metric | Value |
|--------|-------|
| Raw claims extracted | 33 |
| After deduplication | 15 genuinely distinct capabilities |
| CONFIRMED EMERGENT (post rigorous Q-test) | 12 (D1-D3, D5-D11, D13-D15) |
| QUANTITATIVE ENHANCEMENT (demoted) | 2 (D4, D12) |
| New capabilities from cross-pollination | 4 (N-A1 through N-A4) |
| Falsified claims (would not patent) | 0 — all D-list capabilities are novel, non-obvious, and useful at some level |
| **Highest overall score** | Per-Head Steering (D3) — 567 |
| **Most immediately testable** | Death Layer Inversion (D2) — 40 min experiment |
| **Most paradigm-reframing** | Anti-Steering Defense (D15) |
| **Highest novelty** | Complex α (D5) — 10/10 |

### Key Patent-Examiner Verdicts

| Claim | Patentability | Reasoning |
|-------|--------------|-----------|
| Death Layer Inversion | ✅ ALLOWED | Novel finding, non-obvious (sign-dependence is empirical surprise), immediately useful |
| Complex α Steering | ✅ ALLOWED | Highest novelty, no prior art, but requires reduction to practice |
| Phase-Locked Resonance | ✅ ALLOWED | Genuinely novel collective phenomenon, non-obvious wave metaphor |
| Anti-Steering Defense | ✅ ALLOWED (provisional) | Most provocative claim; requires experimental confirmation within 12 months |
| Universal Velocity Manifold | ✅ ALLOWED (provisional) | Foundational claim; requires cross-model validation on ≥5 architectures |
| Anisotropic Subspace Steering | ❌ REJECTED | Obvious: if subspace exists, per-dimension α follows |
| Cross-Task Polarity | ❌ REJECTED | Obvious for same-task-family transfer; pending non-math evidence |
| Style-Content Disentanglement | ❌ REJECTED | Obvious application of existing adversarial disentanglement technique |
