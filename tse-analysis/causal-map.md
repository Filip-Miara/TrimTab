# Phase 7: Causal Mapping & Counterfactual Analysis

---

## Causal DAG

### Node Definitions

| Node | Label | Description | Measurable? |
|------|-------|-------------|-------------|
| N1 | Data_Quality | Diversity + quantity of training trajectories | Yes (coverage metrics) |
| N2 | Norm_Strategy | Global vs per-layer normalization | Yes (binary choice) |
| N3 | Input_Hidden_States | Preprocessed hidden state tensor [B, L, 3584] | Yes |
| N4 | TT_Forward | Transformer forward pass | Yes |
| N5 | Predicted_Velocity | Output of TT [B, L, 3584] | Yes |
| N6 | Velocity_Error | MSE/Cosine loss | Yes |
| N7 | Gradient_Update | Weight adjustment via AdamW | Yes (traceable) |
| N8 | Per_Layer_Gradient_Magnitude | Gradient norm per output dimension | Yes |
| N9 | Training_Stability | Loss variance, absence of NaN/CUDA crashes | Yes |
| N10 | Model_Capacity | Effective capacity per output dimension | Yes (params/dim) |
| N11 | Inference_Velocity | TT applied to new data | Yes |
| N12 | Steering_Accuracy | Directional alignment of velocity | Yes (cosine) |
| N13 | Reasoning_Accuracy | Downstream task performance | Yes (GSM8K score) |
| N14 | Distribution_Shift | MMD/Wasserstein between BnB and AWQ | Yes |
| N15 | Catastrophic_Forgetting | AWQ performance - BnB performance after fine-tune | Yes |

### Causal Edges

```
N1 (Data_Quality) ──positive──→ N3 (Input Hidden States)
N2 (Norm_Strategy) ──modulates──→ N3
N3 ──feeds──→ N4 (TT_Forward)
N4 ──produces──→ N5 (Predicted_Velocity)
N5 ──compared_to──→ N6 (Velocity_Error)
N6 ──drives──→ N7 (Gradient_Update)
N7 ──updates──→ N4 (feedback loop)
N8 (Per_Layer_Gradient) ──indicates──→ N6 (spatial distribution of error)
N6 ──negatively──→ N9 (Training_Stability) [if error spikes]
N2 ──determines──→ N8 [through per-layer signal preservation]
N10 (Model_Capacity) ──modulates──→ N6 [higher capacity → lower error]
N11 (Inference_Velocity) ──depends_on──→ N5 (same TT weights)
N11 ──determines──→ N12 (Steering_Accuracy)
N12 ──causes──→ N13 (Reasoning_Accuracy) [assumed, not validated]
N14 (Distribution_Shift) ──causes──→ N6 [shifted → higher error]
N14 ──causes──→ N15 (Catastrophic_Forgetting) [if fine-tuned]
N9 ──modulates──→ N6 [stable → lower error]
```

### Key Causal Paths

| Path | Type | Length | Criticality |
|------|------|--------|-------------|
| N2→N3→N4→N5→N6→N7→N4 | Causal feedback loop (reinforcing) | 7 | HIGH — normalization choice propagates through entire training loop |
| N14→N6→N7→N4→N6(N14→N5) | Causal interference | 6 | HIGH — distribution shift corrupts the entire chain |
| N2→N8→N6 | Modulation | 3 | MEDIUM — norm determines gradient balance |
| N11→N12→N13 | Forward inference | 3 | CRITICAL — but N12→N13 is UNVALIDATED (missing evidence) |

---

## Branching Points (out-degree ≥ 2)

| Node | Out-degree | Branches | Leverage |
|------|-----------|----------|----------|
| **N2 (Norm_Strategy)** | 4 | →N3 (input quality), →N8 (gradient balance), →N6 (error), →N9 (stability) | HIGHEST |
| **N1 (Data_Quality)** | 3 | →N3 (input), →N14 (distribution), →N6 (error via diversity) | HIGH |
| **N6 (Velocity_Error)** | 3 | →N7 (gradients), →N9 (stability), →N12 (steering via N5) | HIGH |
| **N10 (Model_Capacity)** | 2 | →N6 (error via capacity), →N12 (steering quality) | MEDIUM |

---

## Counterfactuals

### CF-1: "What if normalization were per-layer instead of global?"

| Attribute | Value |
|-----------|-------|
| **Intervention** | Replace global mean(dim=(0,1)) with per-layer (dim=(1,)) |
| **Scope** | All 90K training trajectories + 500 validation |
| **Predicted Outcome** | R² increases from 0.848 to 0.87-0.90. AWQ transfer improves from 0.45 to 0.55-0.65. |
| **Mechanism** | Per-layer signal preserved → input proj receives richer features → transformer learns layer-specific velocity dynamics |
| **Testability** | Train with per-layer norm on 10K subset. Compare R² after 1 epoch. |

### CF-2: "What if loss decomposed into direction + magnitude?"

| Attribute | Value |
|-----------|-------|
| **Intervention** | L_total = α·L_cosine(v_pred, v_true) + (1-α)·L_huber(||v_pred||, ||v_true||) |
| **Scope** | Training loop only; no data changes |
| **Predicted Outcome** | Cosine similarity increases from 0.770 to 0.82-0.85. R² stays similar or slightly lower. Downstream steering quality improves. |
| **Mechanism** | Directional loss aligns velocity vectors; magnitude loss allows correct scale without dominating gradient |
| **Testability** | Train with α=0.5 on current data. Measure Cos and R² separately. |

### CF-3: "What if training data included 3 quantization formats?"

| Attribute | Value |
|-----------|-------|
| **Intervention** | Generate trajectories from BnB + AWQ + GPTQ Qwen. Mix 1:1:1. |
| **Scope** | Data generation (1-2 days), no model changes |
| **Predicted Outcome** | AWQ R² > 0.70, BnB R² > 0.80. ChatGPT R² > 0.70. Quantization-robust representation emerges. |
| **Mechanism** | TT learns to ignore quantization artifacts and focus on shared velocity structure |
| **Testability** | Generate 10K AWQ trajectories. Fine-tune on mixed set. Test all three. |

### CF-4: "What if we computed the noise ceiling first?"

| Attribute | Value |
|-----------|-------|
| **Intervention** | Run Qwen twice on same input with different seeds (deterministic vs non-deterministic). Compute velocity variance across runs. |
| **Scope** | 1000 trajectories, 2 runs each |
| **Predicted Outcome** | If noise ceiling > 0.15 MSE → R²=0.85 IS the ceiling and architecture changes won't help. If noise ceiling < 0.10 MSE → normalization/loss changes can push past 0.85. |
| **Mechanism** | Noise ceiling = irreducible prediction error from stochasticity alone |
| **Testability** | Direct measurement: run 2 forward passes, compute velocity variance |

### CF-5: "What if we added a correction network for AWQ adaptation?"

| Attribute | Value |
|-----------|-------|
| **Intervention** | Train a small MLP (3-layer, 5M params) that maps AWQ hidden states → BnB-like hidden states. TT stays frozen on BnB. |
| **Scope** | Additional network before TT; no TT changes |
| **Predicted Outcome** | AWQ R² > 0.70 with zero BnB forgetting. |
| **Mechanism** | Correction network absorbs distribution shift; TT sees normalized features |
| **Testability** | Train correction network on paired BnB/AWQ trajectories. R² measured after correction. |

---

## Intervention Points (feasible external modulation)

| Point | Node | Feasibility | Cost | Expected Impact |
|-------|------|-------------|------|-----------------|
| IP-1 | N2 (Norm_Strategy) | HIGH | 1 hour | HIGH — changes entire cascade |
| IP-2 | N6 (Velocity_Error / Loss) | HIGH | 3 hours | HIGH — gradient signal quality |
| IP-3 | N1 (Data_Quality) | MEDIUM | 2 days | HIGH — emergent quantization robustness |
| IP-4 | N10 (Model_Capacity via PCA) | MEDIUM | 1 day | HIGH — best capacity utilization |
| IP-5 | N14 (Distribution_Shift via correction net) | MEDIUM | 2 days | MEDIUM — solves AWQ without TT changes |
| IP-6 | N9 (Training_Stability) | HIGH | 1 day | LOW — environmental fix |
