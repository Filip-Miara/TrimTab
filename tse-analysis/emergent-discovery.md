# Emergent Discovery — RankAdaptation

## 1. Unconventional Recombinations

### Class A: Cross-Level Recombinations (Atom ↔ Peak)

**RECOMB-1: A1 (KV Cache) × P (Velocity Steering)**
- *Protocol*: KV cache is usually the *mechanism* of steering. What if it's the *target* of analysis? Use TT to predict KV cache dynamics directly (not hidden state velocities).
- *Novelty*: 4/5 — Current TT predicts h-velocities, not K/V-velocities
- *Prediction*: K/V entries may have different dynamics than hidden states. The velocity of K entries through layers may be more structured than hidden state velocity.

**RECOMB-2: A15 (Capability Threshold) × P (Steering)**
- *Protocol*: What if the capability threshold is not a property of the model but a property of the *steering mechanism*? Different steering mechanisms may have different capability thresholds.
- *Novelty*: 5/5 — Challenges the fundamental framing
- *Prediction*: A different steering mechanism (e.g., MLP activation perturbation instead of KV-cache) might work on models below the current threshold.

### Class B: Domain-Transposed Pyramid

**RECOMB-3: Domain = Neuroscience (Neural Coding)**
- Mapping: Layer → Brain region, Velocity → Neural firing rate change, Trim-tab → LTP (long-term potentiation) site, Death layer → LTD (long-term depression) site
- *Emergent insight*: Steering = injecting current into a specific brain region. Different regions respond differently (some amplify, some suppress). The trim-tab/death-layer pattern mirrors cortical columns' differential response to TMS (transcranial magnetic stimulation).

**RECOMB-4: Domain = Economics (Market Dynamics)**
- Mapping: Hidden state → Asset price, Velocity → Price momentum, Steering → Market intervention, Layer → Market sector
- *Emergent insight*: Central bank intervention (steering) in one sector ripples through all interconnected sectors. The trim-tab layer is the "systemically important sector." The death layer is the "highly leveraged sector" where intervention causes cascading failures. The α is the "stimulus magnitude."

### Class C: Forbidden Pairs (from VOID counter-assumptions)

**RECOMB-5: A1 (KV Cache) × GDN Recurrent State (from ¬A6)**
- *Protocol*: Qwen3.5's GatedDeltaNet uses a recurrent state (S) that accumulates k⊗v outer products. Instead of modifying K/V entries, modify the recurrent state S directly at GDN layers.
- *Novelty*: 5/5 — Requires different implementation (GDN state modification, not K/V)
- *Prediction*: GDN steering via recurrent state may work where K/V steering fails

**RECOMB-6: A16 (Distribution Shift) × C3-2 (Steering)**
- *Protocol*: ¬A9 states that distribution shift corrupts confidence gating. Solution: train the reading head/Perceiver on *steered* generation data, where labels are available (steering outcome known).
- *Novelty*: 4/5 — Online adaptation of the gating mechanism
- *Prediction*: A reading head trained on steered data will maintain r > 0.8 correlation with steering accuracy

### Class D: Self-Application

**RECOMB-7: Apply the steering framework to the steering framework itself**
- *Protocol*: What is the "velocity" of the steering research program? Where is it heading? Can we steer the research process using learned velocity?
- *Result*: The research velocity points toward (1) larger models, (2) finer granularity (per-head), (3) online adaptation, (4) task diversity. The "trim-tab" of the research program is the contrastive direction — it's the small change with outsized impact. The "death layer" is the capability threshold — it's the constraint that invalidates all small-model experiments.

## 2. Emergent Capability Analysis

### EM-1: Per-Head Steering Within Trim-Tab Layers
- **Source**: RECOMB-2 (cross-level: attention heads × layer steering)
- **Description**: Instead of modifying the entire K/V at a layer, modify only specific attention heads' K/V. If L8 has 28 attention heads, only 4 might be "trim-tab heads."
- **Q1 — Qualitatively distinct?**: Y — Per-head steering is NOT the same as per-layer steering with reduced α. Different heads compute different functions (syntactic, semantic, positional), and modifying them selectively enables *functional specificity* that per-layer steering cannot achieve.
- **Q2 — Not predictable from constituents?**: Y — The interaction between per-head effects within a layer is not predictable from independent head analysis. Heads have cooperative/suppressive dynamics.
- **Q3 — Synergy in kind?**: Y — Per-head steering enables *selective reasoning pathway amplification*, which is a qualitatively different capability from "push all heads toward correctness."
- **Classification**: **CONFIRMED EMERGENT**
- **Trigger**: Requires per-head access to K/V cache, which some architectures support (GQA with shared KV across heads).

### EM-2: Token-Position-Adaptive α(t)
- **Source**: RECOMB-1 (cross-level: temporal position × steering magnitude)
- **Description**: α varies as a function of token position within the generation, learned via RL on validation accuracy.
- **Q1 — Qualitatively distinct?**: Y — Static α is a single knob; α(t) is a sequence of decisions. The temporal structure cannot be reduced to any single α value.
- **Q2 — Not predictable from constituents?**: Y — The optimal α schedule depends on the interaction between position and content, not just position alone.
- **Q3 — Synergy in kind?**: Y — Enables *sequential decision-making* about when to intervene, which is categorically different from binary "steer/don't steer."
- **Classification**: **CONFIRMED EMERGENT**
- **Trigger**: Requires an RL training loop (REINFORCE or PPO) over generation sequences, with validation accuracy as reward.

### EM-3: Cross-Task Polarity Generalization
- **Source**: RECOMB-4 (domain-transposed: economics)
- **Description**: The layer polarity pattern (which layers are trim-tabs vs death-layers) for one task predicts polarity for related tasks. The L8 polarity for GSM8K (+20pp) partially transfers to SVAMP (+4pp).
- **Q1 — Qualitatively distinct?**: Y — Polarity transfer is NOT the same as velocity transfer. Polarity is a second-order property (which layers matter) distinct from first-order dynamics (how to steer).
- **Q2 — Not predictable from constituents?**: Y — Whether polarity transfers depends on task similarity in a *functional* space that isn't captured by surface form overlap.
- **Q3 — Synergy in kind?**: N — This is a quantitative enhancement (faster identification of trim-tabs on new tasks).
- **Classification**: **QUANTITATIVE ENHANCEMENT**
- **Trigger**: Requires steering evaluation on ≥3 tasks (GSM8K, SVAMP, ARC, BBH, MMLU) to compute polarity correlation matrix.

### EM-4: Self-Supervised Contrastive Direction
- **Source**: RECOMB-6 (forbidden pair: distribution shift × steering)
- **Description**: Train TT_correct and TT_incorrect without correctness labels by clustering hidden state trajectories based on their *convergence* properties (trajectories that converge to high-confidence tokens vs trajectories that oscillate).
- **Q1 — Qualitatively distinct?**: Y — Current TTs require labeled correct/incorrect answers. Self-supervised clustering of trajectory properties would enable contrastive steering WITHOUT labels.
- **Q2 — Not predictable from constituents?**: Y — The convergence property of a trajectory emerges from the full generation; it's not a property of individual hidden states.
- **Q3 — Synergy in kind?**: Y — Enables *unsupervised* contrastive steering, which is a qualitatively different capability from supervised contrastive.
- **Classification**: **CONFIRMED EMERGENT**
- **Trigger**: Requires unsupervised clustering of trajectories (e.g., by spectral clustering of trajectory autocorrelation).

## 3. Synergy Mapping

### Highest Pairwise Synergy
- **{L8 Steering, Contrastive Direction}**: Score = 9.2/10 — Combining L8 trim-tab with contrastive velocity (v_c - v_i) may produce the strongest single-layer steering result. The contrastive signal amplifies the L8 trim-tab effect by pointing toward correct-answer trajectories directly.
- **{Per-layer α, Adaptive α(t)}**: Score = 8.5/10 — Per-layer α + per-token α gives a 28×T dimensional steering surface, enabling spatiotemporally precise intervention.
- **{Confidence Gate (Reading Head), Contrastive Steering}**: Score = 8.0/10 — Gate steering by uncertainty, but use contrastive direction when steering. The combination ensures steering only occurs when needed AND in the right direction.

### Highest Higher-Order Synergy
- **{L8 Steering, Contrastive Direction, Confidence Gate, Adaptive α(t)}**: Score = 9.5/10 — The quadruple combination is qualitatively distinct: it enables *precise, minimally invasive, normatively directed, temporally adaptive intervention*. No pairwise subset captures this capability.
- **Self-Organization Detected**: YES — The interaction of all four mechanisms (trim-tab selection + contrastive direction + confidence gating + temporal scheduling) creates a higher-order capability that emerges from the specific configuration: *resource-efficient reasoning amplification*.

### Individual Quality Scores (for Phase 5 filtering)

| Variant | Novelty | Feasibility | Coherence | Risk (1=low) | Emergent Potential | Quality Index |
|---------|---------|-------------|-----------|-------------|-------------------|--------------|
| Per-head steering (EM-1) | 5 | 3 | 4 | 3 | 5 | 4.2 |
| Adaptive α(t) (EM-2) | 4 | 3 | 4 | 3 | 5 | 4.0 |
| Cross-task polarity (EM-3) | 3 | 4 | 5 | 2 | 3 | 3.6 |
| Self-supervised contrastive (EM-4) | 5 | 2 | 3 | 4 | 5 | 3.6 |
| GDN recurrent state steering (RECOMB-5) | 5 | 4 | 4 | 3 | 4 | 4.0 |
| Steered-data reading head (RECOMB-6) | 3 | 4 | 5 | 2 | 4 | 4.0 |
| Skip-layer velocity (M1 variant) | 3 | 5 | 5 | 1 | 2 | 3.6 |
| Death-layer sign flip (M2 variant) | 4 | 5 | 4 | 2 | 5 | 4.4 |
| Hybrid standard + contrastive (M4 variant) | 4 | 5 | 4 | 2 | 4 | 4.2 |
| Per-token α (M3 variant) | 4 | 3 | 4 | 3 | 5 | 4.0 |
