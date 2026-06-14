# Phase 7: Causal Mapping & Counterfactual Analysis

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Causal DAG

```
                    ┌──────────────┐
                    │ Model Scale   │──►│ Architecture  │
                    │ (params)      │    │ (MHA vs Hyb)  │
                    └──────┬───────┘    └──────┬────────┘
                           │                   │
                           ▼                   ▼
                    ┌──────────────┐    ┌──────────────┐
                    │ Capability    │◄───│ Steer Surface │
                    │ (baseline %)  │    │ (K/V exists?) │
                    └──────┬───────┘    └──────┬────────┘
                           │                   │
                           ▼                   ▼
                    ┌──────────────┐    ┌──────────────┐
                    │ Trajectory    │    │ Alpha (α)    │
                    │ Collection    │    │ (magnitude)  │
                    └──────┬───────┘    └──────┬────────┘
                           │                   │
                           ▼                   ▼
                    ┌──────────────┐    ┌──────────────┐
                    │ TT Training   │───►│ TT Prediction │
                    │ (R² quality)  │    │ (v̂ direction) │
                    └──────┬───────┘    └──────┬────────┘
                           │                   │
                           │                   ▼
                           │            ┌──────────────┐
                           │            │ Layer Sel.    │
                           │            │ (which layer) │
                           │            └──────┬────────┘
                           │                   │
                           ▼                   ▼
                    ┌────────────────────────────────────┐
                    │         Steering Intervention       │
                    │  (modify K/V at layer L by α × v̂)  │
                    └────────────────┬───────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              ┌──────────┐    ┌──────────┐    ┌──────────┐
              │ Token     │    │ Hidden   │    │ Death    │
              │ Divergence│    │ State    │    │ Layer    │
              │ (88%)     │    │ Shift    │    │ Cascade  │
              └──────┬───┘    └─────┬────┘    └─────┬────┘
                     │              │               │
                     └──────┬───────┘               │
                            ▼                       ▼
                     ┌──────────┐            ┌──────────┐
                     │ Accuracy  │            │ Complete  │
                     │ Change    │            │ Collapse  │
                     │ (±20pp)   │            │ (-23pp+)  │
                     └──────────┘            └──────────┘
```

---

## Node Statistics

| Node | In-Degree | Out-Degree | Classification |
|------|-----------|------------|----------------|
| Capability (baseline %) | 3 | 2 | **Root cause** |
| TT Training | 2 | 2 | Mediator |
| TT Prediction | 2 | 2 | Mediator |
| Layer Selection | 2 | 1 | **Branching point** |
| Steering Intervention | 3 | 3 | **Critical mediator** |
| Token Divergence | 1 | 1 | Leaf (measurement) |
| Hidden State Shift | 1 | 1 | Leaf (irrelevant alone) |
| Death Layer Cascade | 1 | 1 | Leaf (outcome) |
| Accuracy Change | 1 | 0 | **Ultimate outcome** |

---

## Branching Points (Out-degree ≥ 2)

### BP-1: Steering Intervention (out-degree: 3)
**Description**: The intervention branches to three outcomes: token divergence (88% of tokens change), hidden state shift (desired effect), and death layer cascade (undesired effect).
**What determines which branch dominates?** The layer identity (L8 vs L9) and the sign of α.
**Current control**: Manual (choose layer, choose ±α).

### BP-2: Layer Selection (out-degree: 2)
**Description**: Layer selection simultaneously affects which layers get steering and which layers are excluded. With 28 layers, the combinatorial space is 2^28.
**Current control**: Manual sweep (28 experiments).

### BP-3: Capability (out-degree: 2)
**Description**: Capability affects both trajectory quality (which determines TT quality) and steerability (whether steering has an effect).
**Current control**: None (we select models, don't modify them).

---

## Counterfactuals

### CF-1: "What if we had used negative α on L9?"
- **Intervention**: In the L9 steering experiment, use α = −0.1 instead of +0.1
- **Predicted outcome**: L9 goes from −23pp to approximately +23pp (symmetric effect), making it the second-best layer after L8
- **Causal chain**: Steering Intervention → (α reversed) → Death Layer Cascade becomes Hidden State Shift → Accuracy Change = positive
- **Testability**: **HIGH** — Run `run_math15_sweep.py --layers 9 --alpha -0.1`

### CF-2: "What if we had tested L8 with 2× the α?"
- **Intervention**: In L8 experiment, use α = 0.2 instead of 0.1
- **Predicted outcome**: Accuracy may increase further (maybe +30pp) until α threshold where token divergence destroys coherence
- **Causal chain**: α↑ → Steering Intervention strength↑ → Token Divergence↑ → Accuracy↑ until threshold → Accuracy↓ after threshold
- **Testability**: **HIGH** — Run sweep with multiple α values for L8

### CF-3: "What if we had used contrastive TT first instead of standard TT?"
- **Intervention**: Skip standard TT training; train contrastive TT directly and evaluate
- **Predicted outcome**: Two possibilities:
  - (Optimistic) Contrastive TT immediately finds L8 as trim-tab with higher gain (+30pp instead of +20pp)
  - (Pessimistic) Contrastive TT fails because correct/incorrect manifolds are not separable for Qwen2.5-7B
- **Causal chain**: TT Training (contrastive mode) → TT Prediction (v_correct − v_incorrect) → Steering Intervention → Accuracy Change = ???
- **Testability**: **HIGH** — Evaluation already pending (contrastive TTs trained)

### CF-4: "What if the model had been trained on 10× more data?"
- **Intervention**: Scale trajectory dataset from 500 problems to 5000 problems
- **Predicted outcome**: TT R² improves marginally (already 0.85-0.94; diminishing returns). Steering effect improves marginally (better TT on same model, same layers).
- **Causal chain**: More data → TT Training (better R²) → TT Prediction (slightly better direction) → Accuracy Change (slightly better)
- **Testability**: MEDIUM — Requires 10× compute for data collection

### CF-5: "What if we steer L8 AND L9 simultaneously with opposite signs?"
- **Intervention**: Apply both: L8: +α, L9: −α simultaneously
- **Predicted outcome**: Two possibilities:
  - (Additive) Accuracy = baseline + 20pp (L8) + 23pp (L9 inverted) = **+43pp**
  - (Interference) Accuracy = baseline + small positive (effects cancel or interact nonlinearly)
- **Causal chain**: Layer Selection = {L8, L9} → Steering Intervention (simultaneous) → Effects interact → Accuracy = ???
- **Testability**: **HIGH** — Requires code change to support per-layer sign

### CF-6: "What if the TT had never been trained (random steering)?"
- **Intervention**: Replace TT prediction with random direction (unit vector sampled uniformly)
- **Predicted outcome**: Steering with random direction at L8 → approximately 0pp change (random walk ≈ no effect)
- **Causal chain**: No TT → Random direction → Steering Intervention → Accuracy = baseline ± noise
- **Testability**: **HIGH** — Run steering with random vectors as control experiment

---

## Delay Mapping

| Causal Edge | Estimated Delay | Evidence |
|-------------|----------------|----------|
| Model Scale → Capability | Fixed (model property) | Known at experiment start |
| Architecture → Steering Surface | Fixed (arch property) | Known at experiment start |
| Capability → Trajectory Quality | Seconds (generation time) | Same as generation |
| Trajectory Collection → TT Training | Minutes-Hours (I/O + epochs) | ~30 min for 500 problems |
| TT Training → TT Prediction | Instant (once trained) | Forward pass in ms |
| TT Prediction → Steering Intervention | Milliseconds (same forward pass) | Negligible |
| Steering Intervention → Token Divergence | Instant (next token) | Next token generation |
| Steering Intervention → Accuracy Change | Minutes (full evaluation) | 100 problems × generation time |

**Critical Delay**: The delay between "steering intervention" and "accuracy measurement" is a full evaluation cycle (minutes to hours). This prevents real-time adaptation within a single generation. All steering decisions must be made open-loop (without outcome feedback).

---

## Intervention Feasibility

| Intervention Point | Feasibility | Cost | Expected Information Gain |
|-------------------|-------------|------|--------------------------|
| α sign per layer | **IMMEDIATE** | ~1 hour | **CRITICAL** — Resolves D11 |
| α magnitude per layer | **IMMEDIATE** | ~2 hours | HIGH — Optimal α per layer |
| Multi-layer combinations | **MEDIUM** | ~4 hours | HIGH — Layer interaction test |
| Contrastive TT evaluation | **PENDING** | Already set up | CRITICAL — Next logical step |
| Self-improving loop | **COMPLEX** | Days-weeks | HIGH — Long-term potential |
| Random steering baseline | **IMMEDIATE** | ~1 hour | HIGH — Null hypothesis test |
