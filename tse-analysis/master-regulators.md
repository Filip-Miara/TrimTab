# Phase 3: Master-Regulator Identification

**Ranked by Influence Centrality × Junction Leverage**

---

| # | Master Regulator | Current State | Modulation Strategy | Impact | Risk |
|---|-----------------|---------------|-------------------|--------|------|
| M1 | **TT training target** (A1→A4) | Descriptive (single TT) → Contrastive (v_correct − v_incorrect) | Switch from single TT to contrastive pair | HIGH — fixes descriptive/normative flaw | LOW — data already labeled |
| M2 | **Layer selection** (A3) | All layers steered indiscriminately | Steer ONLY trim-tab layers (L8, L2, L5) | HIGH — eliminates death-layer noise | LOW — known from sweep |
| M3 | **α scaling** (A8→A2) | Global α per experiment | Per-layer α vector + asymmetric (α_c, α_i) | MED-HIGH — adapts strength to each layer | MED — search space grows |
| M4 | **Confidence gating** (A7) | Unused (r=0.85 reading head idle) | Modulate α by token-level perplexity or ensemble disagreement | MED — uncertainty-weighted steering | MED — requires additional model |
| M5 | **Multi-head ensemble** (A12) | Single contrastive pair | N bootstrapped contrastive pairs, bagged average | MED — reduces variance, gives uncertainty | LOW — parallelizable |
| M6 | **Training data quality** (A5) | 40-48% correct trajectories | Use correctness-verified labels via chat-template prompting | MED — cleaner labels reduce noise | LOW — already fixed |

## Top-3 Master Regulators

### M1: Contrastive TT Training ← HIGHEST LEVERAGE
- **Modulation**: Train TT_correct (R²=0.873) and TT_incorrect (R²=0.909) separately
- **Current state**: Accomplished (best_tt_correct.pt, best_tt_incorrect.pt)
- **Next step**: Add asymmetric α (α_c, α_i) for independent attraction/repulsion
- **Expected impact**: +5-15pp on trim-tab layers

### M2: Layer-Selective Steering
- **Modulation**: Steer only L2, L5, L8 (Qwen2.5-7B) or equivalent trim tabs on other models
- **Current state**: Confirmed on Qwen2.5-7B (+20pp L8, +17pp L2)
- **Next step**: Find Math-1.5B trim tabs via contrastive sweep (running)
- **Expected impact**: Eliminates noise from death layers

### M3: Per-Layer Asymmetric α
- **Modulation**: h' = h + Σ α_c[l] · TT_correct(h)[l] − α_i[l] · TT_incorrect(h)[l]
- **Current state**: Not yet tested
- **Next step**: Sweep α_c ∈ {0.01, 0.05, 0.1}, α_i ∈ {0.01, 0.05, 0.1} on best layers
- **Expected impact**: Prevents over-steering on sensitive layers
