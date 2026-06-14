# Phase 3: Master-Regulator Identification

---

## Methodology

Influence Centrality = number of other nodes/junctions affected by modifying this node
Junction Leverage = estimated impact of changing this relationship
Score = Influence Centrality × Junction Leverage (normalized 0-100)

---

## Ranked Master Regulators

### #1: Layer Selection (A07/A08 — Trim Tab & Death Layer Identification)

- **Influence Centrality**: 92/100
  - Affects: C02 (steering mechanism), C03 (per-layer sweep), P01 (full framework), C09 (capability threshold hypothesis), J08, J09, J18
- **Junction Leverage**: 95/100
  - Changing which layer is selected changes accuracy by up to **43pp** (L8: +20pp vs L9: −23pp)
- **Modulation Strategies**:
  - Existing: Manual per-layer sweep (28 layers × 100 problems ≈ 4 hrs)
  - Proposed: Gradient-based layer importance estimation (compute ∂accuracy/∂layer via finite differences)
  - Proposed: Attention-based probing (find layers where attention distribution correlates with correctness)
- **Expected Impact**: HIGH — layer selection is the single highest-leverage decision
- **Risk**: Sweep results may not generalize across datasets; selecting the wrong layer active on a new task could harm accuracy

### #2: Contrastive TT Training Pipeline (C04 — Contrastive TrajectoryTransformer Pair)

- **Influence Centrality**: 88/100
  - Affects: C02 (normative steering), P03 (contrastive pipeline), J11 (causal junction), J27 (synergy with per-layer), P01 (full framework evolution)
- **Junction Leverage**: 85/100
  - Converting from descriptive (standard TT: predicts what will happen) to normative (contrastive TT: predicts what *should* happen) is the crucial architectural insight
  - The contrastive signal determines whether steering amplifies correct reasoning or faithfully reproduces errors
- **Modulation Strategies**:
  - Existing: Two independently trained TTs (correct + incorrect) → differenced for inference
  - Proposed: Single TT with contrastive objective (maximize v_correct · v_z − v_incorrect · v_z)
  - Proposed: Multi-head contrastive (bootstrap N pairs → ensemble)
  - Proposed: Online contrastive learning (TT updated based on steering success)
- **Expected Impact**: HIGH — determines whether the entire steering approach achieves its goal
- **Risk**: Contrastive difference may push model off-manifold (especially if correct/incorrect manifolds are not separable — as suspected for Math-1.5B)

### #3: Architecture Selection for Steering Surface (A13 — Attention Computation Type)

- **Influence Centrality**: 75/100
  - Affects: C02 (KV-steering viability), J26 (antagonistic with hybrid), P01 (model choice), A04 (KV cache type), A13 (attention mechanism)
- **Junction Leverage**: 90/100
  - Incorrect architecture choice (hybrid GDN+FA) makes the entire steering approach impossible — 0% success across all attempts
  - Correct choice (standard MHA) enables the entire line of research
- **Modulation Strategies**:
  - Existing: Manual selection (Qwen2.5, LLaMA, SmolLM2 — all MHA)
  - Proposed: Architecture-agnostic steering surface (if hybrid can't be steered via KV-cache, find alternative surface)
  - Proposed: GDN-specific steering mechanism (recurrent states in GatedDeltaNet could be steered directly)
- **Expected Impact**: HIGH — binary gate (success vs failure) depending on choice
- **Risk**: Limiting to MHA excludes newer efficient architectures; may reduce relevance over time

### #4: Alpha Optimization (A06 — Steering Strength)

- **Influence Centrality**: 70/100
  - Affects: C02 (steering magnitude), J10 (causal junction), steering effect size, generation quality (token divergence)
- **Junction Leverage**: 65/100
  - Within optimal range (0.01-0.3), alpha has moderate effect (±5pp)
  - Outside optimal range (α > 0.5), alpha causes token divergence and quality collapse
  - Per-layer α vector could be significantly more powerful than single α
- **Modulation Strategies**:
  - Existing: Sweep over fixed grid [0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
  - Proposed: RL-based per-token α learning
  - Proposed: Uncertainty-based α (use TT's prediction uncertainty to modulate strength)
  - Proposed: Adaptive α inversely proportional to baseline confidence
- **Expected Impact**: MEDIUM — significant but secondary to layer selection
- **Risk**: Over-optimization of α on test set leads to brittle results that don't generalize

### #5: Training Data Split for Contrastive TT (A16 — Contrastive Signal Quality)

- **Influence Centrality**: 65/100
  - Affects: C04 (contrastive pair quality), J11 (causal junction strength), J27 (synergy with per-layer), P03 (pipeline viability)
- **Junction Leverage**: 70/100
  - If correct/incorrect trajectories are not separable (as seen in Math-1.5B), the contrastive signal is noise
  - Quality of the split determines whether v_c − v_i points toward the correct manifold or toward a degenerate direction
- **Modulation Strategies**:
  - Existing: Binary split by answer correctness on training data
  - Proposed: Confidence-weighted split (use baseline model's confidence to create a spectrum from correct to incorrect)
  - Proposed: Difficulty-stratified split (easy vs hard problems, not just correct vs incorrect)
  - Proposed: Clustering-based split (cluster trajectories, then compute within/between cluster velocities)
- **Expected Impact**: HIGH — determines whether contrastive steering works at all
- **Risk**: Binary correct/incorrect split is inherently noisy — some "correct" answers are lucky guesses, some "incorrect" are close calls

---

## Master-Regulator Interaction Map

```
Layer Selection (MR1)
  ↑ directs
  ↓
Contrastive Pipeline (MR2)
  ↑ feeds into
  ↓
Alpha Optimization (MR4)
  ↑ modulates
  ↓
Steering Surface (MR3) ← conditions whether any of this is possible
  ↑
Data Split (MR5) ← determines contrastive quality
```

The hierarchy shows a dependency chain: **MR3 (Architecture) → MR1 (Layer Selection) → MR2 (Contrastive Pipeline) → MR4 (Alpha) → MR5 (Data Split)**. Each regulator only matters if the upstream regulator is correctly configured.

---

## High-Leverage Intervention Points Beyond Top-5

| # | Candidate | Rationale | Why Not Top-5 |
|---|-----------|-----------|---------------|
| 6 | Training data volume (number of trajectories) | More data reduces TT error | Data collection cost is high (10GB+ per model) but impact is diminishing (R² already 0.85+) |
| 7 | TT architecture (d_model, n_layers) | Larger TT could capture more complex velocity dynamics | Current architecture (d_model=512, 6 layers) achieves R²=0.94; diminishing returns likely |
| 8 | Chat template implementation (A14) | 4%→73% baseline jump | Already fixed; no further optimization needed |
| 9 | Batch size / GPU caching (C08) | Infrastructure optimization | Improves iteration speed but doesn't affect steering results |
| 10 | Multi-layer combination strategy | Could outperform best single layer | Requires interaction terms to be understood first; high risk of compounding noise |
