# Phase 3: Master-Regulator Identification

## Ranked by Influence Centrality × Junction Leverage

---

## MR-1: Normalization Strategy (J08 — D6→A1)

| Attribute | Value |
|-----------|-------|
| **Type** | Junction (Dependency) — D6 normalization → A1 Transformer input |
| **Influence Centrality** | 8.5/10 — Affects every sample, every layer, every gradient update |
| **Junction Leverage** | 9.0/10 — If changed, alters the entire input distribution; affects all downstream representations |
| **Combined Score** | 85.3 (highest) |

### Why It's #1
Global normalization (one mean/std per feature across ALL 28 layers and 90K samples) simultaneously:
1. **Mixes layer statistics**: early layers (high activation variance) and late layers (fine-grained) are normalized identically
2. **Removes per-layer velocity signal**: mean velocity per layer is subtracted out
3. **Amplifies noise in low-variance features**: features with near-zero global variance become random noise amplifiers

### Modulation Strategies
| Strategy | Cost | Expected Impact | Risk |
|----------|------|-----------------|------|
| Change to per-layer normalization (28 separate mean/std) | 1 hour code | +0.02-0.05 R² | Low — trivial change |
| Add per-layer normalization as additional input features to TT | 2 hours code | +0.03-0.08 R² | Low |
| Per-sample, per-layer normalization (normalize each trajectory independently) | 3 hours code | +0.01-0.03 R² | Medium — removes trajectory-level signal |

### Leverage Explanation
This is the **cheapest change with the highest potential impact**. It accesses systematic information loss behavior identified by Lenses 2 (Dialectical), 4 (Systems — B2 loop), 5 (Abductive — H1), 9 (Adversarial — info-theoretic bounds), and 10 (Paradox 2 — normalization paradox).

---

## MR-2: Loss Function Structure (J09 — T8→A1)

| Attribute | Value |
|-----------|-------|
| **Type** | Junction (Causal feedback) — MSE loss → weight updates |
| **Influence Centrality** | 9.0/10 — Defines the optimization landscape entirely |
| **Junction Leverage** | 8.5/10 — Small changes in loss can produce large changes in learned representations |
| **Combined Score** | 82.5 |

### Why It's #2
MSE equally penalizes magnitude error and directional error. For velocity prediction, directional accuracy matters more than magnitude accuracy for steering. Current MSE wastes capacity on magnitude.

### Modulation Strategies
| Strategy | Cost | Expected Impact | Risk |
|----------|------|-----------------|------|
| Decomposed loss: cosine direction + LogCosh magnitude | 3 hours | +0.03-0.06 R² + better directional alignment | Low |
| Per-layer adaptive weighting (inverse variance weighting) | 4 hours | +0.01-0.03 R² | Low |
| Contrastive loss: maximize similarity with correct velocity, minimize with wrong | 8 hours | +0.05-0.10 R² | Medium — harder to tune |
| Add domain-contrastive loss for AWQ/BnB alignment | 6 hours | AWQ transfer improvement | Medium |

### Leverage Explanation
Lenses 2 (Dialectical — angular+magnitude decomposition), 5 (Abductive — H4), 9 (Adversarial — gradient imbalance) all converge on loss function as key lever.

---

## MR-3: Training Data Composition (L2-6 Data Source)

| Attribute | Value |
|-----------|-------|
| **Type** | Composite (Data Source + Preprocessing) |
| **Influence Centrality** | 8.0/10 — Determines what the TT can learn |
| **Junction Leverage** | 8.5/10 — Changing data composition changes learned distribution |
| **Combined Score** | 72.0 |

### Why It's #3
Current data (100% BnB Qwen2.5-7B on GSM8K) is homogeneous. The AWQ transfer collapse (0.85→0.45) is primarily a distribution mismatch. Changing what data the TT sees during training can solve the transfer problem at the root.

### Modulation Strategies
| Strategy | Cost | Expected Impact | Risk |
|----------|------|-----------------|------|
| Mix 50% BnB + 50% AWQ trajectories during initial training | 2 days data gen | AWQ R² > 0.70, BnB R² > 0.80 | Low — just needs data |
| Add domain-adversarial loss (gradient reversal) | 2 days code | Both >0.80 | Medium — tricky to balance |
| Online trajectory generation from both BnB and AWQ Qwen | 1 week infra | Unlimited data, eliminates I/O bottleneck | High — infrastructure |
| Curriculum: start with BnB, gradually mix AWQ | 4 hours code | Smooth adaptation | Low |

### Leverage Explanation
Identified by Lenses 2 (Dialectical — domain alignment), 4 (Systems — R2 loop), 5 (Abductive — H1), 8 (Inspiration — MAML).

---

## MR-4: Layer Specialization (Layer-level prediction structure)

| Attribute | Value |
|-----------|-------|
| **Type** | Structural concept (implicit in architecture) |
| **Influence Centrality** | 7.5/10 — Affects gradient allocation across layers |
| **Junction Leverage** | 7.5/10 — Weighting layers differently changes where model capacity goes |
| **Combined Score** | 56.25 |

### Why It's #4
The system treats all 28 layers as equally important, but velocity structure likely varies dramatically: early layers process token-level features, middle layers build representations, late layers prepare output.

### Modulation Strategies
| Strategy | Cost | Expected Impact | Risk |
|----------|------|-----------------|------|
| Per-layer loss weighting by velocity variance | 2 hours | +0.01-0.02 R² | Low |
| Layer-group experts (3 heads for early/mid/late) | 3 days code | +0.03-0.05 R² | Medium |
| Layer-specific prediction heads (28 small heads) | 2 days code | +0.02-0.04 R² | Medium |

---

## MR-5: Attention Mechanism (A7 → bidirectional)

| Attribute | Value |
|-----------|-------|
| **Type** | Atomic concept (A7) |
| **Influence Centrality** | 7.0/10 — Affects all TT forward passes |
| **Junction Leverage** | 6.5/10 — Changing attention changes representation quality |
| **Combined Score** | 45.5 |

### Why It's #5
Bidirectional vs causal attention affects not just R² but the fundamental validity of the training-inference match. Paradox 4 identifies this as a potential training-inference mismatch.

### Modulation Strategies
| Strategy | Cost | Expected Impact | Risk |
|----------|------|-----------------|------|
| Causal mode with same capacity (rerun experiment) | 1 day compute | Possibly -0.02 R² but valid inference | Low — already tested |
| Hybrid: bidirectional encoding, causal prediction head | 2 days code | Possibly +0.01 R² with valid inference | Medium |
| Causal with future masking of loss (model can see forward but loss only backward) | 1 day code | Novel approach | Medium |

---

## Summary: Modulation Priority

| Rank | Regulator | Score | Cost | Expected R² Impact | AWQ Impact |
|------|-----------|-------|------|-------------------|------------|
| 1 | Normalization strategy | 85.3 | Very Low | +0.02-0.05 | Indirect |
| 2 | Loss function | 82.5 | Low | +0.03-0.08 | Medium |
| 3 | Training data composition | 72.0 | Medium-High | +0.01-0.03 | HIGH |
| 4 | Layer specialization | 56.3 | Low-Medium | +0.02-0.05 | Low |
| 5 | Attention mechanism | 45.5 | Medium | +/-0.02 | None |
