# Causal Map — RankAdaptation

## Causal DAG

```
Training Phase:                               Generation Phase:
                                        
TT_training_data ──→ TT(Trained) ──→ v = TT(h_seq) ──→ h' = h + α·v
     ↑                                       ↑              │
     │                                       │              │
Correct/Incorrect Labels                     │              ↓
     ↑                                       │         K' = W_k(h'), V' = W_v(h')
     │                                       │              │
     └── Contrastive training ───────────────┘              │
                                                            ↓
                                                KV_cache[t] ← modified
                                                            │
                                                     ┌──────┘
                                                     ↓
                                                Attention at step t+1
                                                     │
                                                     ↓
                                                Next token: T(t+1)
                                                     │
                                        ┌────────────┼────────────┐
                                        ↓            ↓            ↓
                                   If correct    If incorrect   Continue
                                   manifold      manifold       generation
                                        │            │
                                        ↓            ↓
                                   Higher P(correct)  Lower P(correct)
                                        │            │
                                        ↓            ↓
                                   GSM8K accuracy   GSM8K accuracy
                                   ↑                ↓
                                   ←── Reinforcing ──→ Self-limiting
                                       loop            loop
```

### Edges

| Edge | Type | Delay | Confidence |
|------|------|-------|------------|
| TT_training_data → TT | Training | Hours (full training) | 10/10 |
| TT → v | Inference | ~2ms (TT forward) | 10/10 |
| h_seq + v → h' | Additive | ~0.001ms (vector add) | 10/10 |
| h' → K', V' | Projection | ~0.01ms (linear layer) | 10/10 |
| K', V' → KV_cache[t] | Assignment | ~0.001ms (tensor copy) | 10/10 |
| KV_cache → Attention | Retrieval | ~0.01ms (attention compute) | 10/10 |
| Attention → Next token | Decoding | ~5ms (full layer stack) | 10/10 |
| Next token → Manifold | State transition | ~0ms (deterministic) | 9/10 |
| Correct manifold → Accuracy | Evaluation | Generation end | 10/10 |
| Contrastive training → v | Directional | Training hours | 8/10 |
| Capability threshold → Steering_effectiveness | Conditioning | 0 (static property) | 8/10 |

## Branching Points (out-degree ≥ 2)

| Node | Out-Degree | Description |
|------|-----------|-------------|
| **TT** | 4 | Determines v vector, which affects all downstream: h', K', V', KV_cache, attention |
| **h'** | 3 | Modified hidden state produces new K, V, and residual stream for next layer |
| **KV_cache[t]** | 3 | Affects all future attention operations that attend to position t |
| **Next token T(t+1)** | 2+ (per step) | Each token is a branching point — determines next h_seq and continues/terminates generation |

## Counterfactuals

### CF-1: "What if α is negative at L9 (death layer)?"
- **Current**: α = 0.1 at L9 → 0% accuracy (-23pp vs baseline)
- **Counterfactual**: α = -0.1 at L9
- **Predicted Outcome**: If L9 is a trim-tab with inverted polarity, accuracy might increase to match L8 (+20pp). If L9 is genuinely harmful, accuracy remains at 0%.
- **Testability**: IMMEDIATE — change sign of α in `run_per_layer_sweep.py` line 40. Runs in ~20 min for 30 problems.

### CF-2: "What if the first-step skip is removed?"
- **Current**: `if not first_step:` prevents steering at t=0
- **Counterfactual**: Remove skip, steer at first generation step
- **Predicted Outcome**: Either (a) larger accuracy improvement (+25-30pp) because first token has highest leverage, or (b) worse accuracy because prompt hidden states have different distribution and TT predictions are less reliable (R²=0.62 for prompt vs 0.85 for generation).
- **Testability**: IMMEDIATE — remove the `first_step = True` guard and compute velocity from first forward pass's hidden states.

### CF-3: "What if we steer at t=0 using prompt-trained TT?"
- **Current**: Generation-trained TT used at all steps
- **Counterfactual**: Use prompt-trained TT specifically for first step (where hidden states are more prompt-like)
- **Predicted Outcome**: Might combine best of both — prompt TT has reasonable R² on prompt-like states, generation-trained TT has better R² on subsequent steps.
- **Testability**: SHORT-TERM — load both TTs, select which to use based on step index.

### CF-4: "What if the TT is fine-tuned on steered-generation data?"
- **Current**: TT trained once on unsteered generation data, used frozen
- **Counterfactual**: TT fine-tuned on 1-2 batches of steered generation data
- **Predicted Outcome**: TT learns to predict velocities in the distribution it will actually encounter during steering, reducing distribution mismatch and improving steering efficacy.
- **Testability**: SHORT-TERM — collect steered trajectories, fine-tune TT with low learning rate.

### CF-5: "What if we intervene on the residual stream instead of KV cache?"
- **Current**: h' = h + α·v, then K' = W_k(h'), V' = W_v(h')
- **Counterfactual**: h' = h + α·v, then residual_stream_output = residual_stream + h' (direct modification of the main residual stream, bypassing K/V projections)
- **Predicted Outcome**: More direct effect on downstream computation (residual stream carries information through all layers). Less architectural constraint (works on any architecture).
- **Testability**: MEDIUM-TERM — requires modifying the model's forward pass directly, not just the KV cache.

### CF-6: "What if we interpolate between v_c and v_i instead of subtracting?"
- **Current**: v_contrastive = v_c - v_i (subtraction)
- **Counterfactual**: v_interp = λ·v_c + (1-λ)·v_i (interpolation with λ ∈ [0,1])
- **Predicted Outcome**: At λ=1, steer toward correct manifold (like v_c alone). At λ=0, steer away from correct (v_i). At λ=0.5, neutral (v_std). The interpolation allows smooth control over steering strength.
- **Testability**: IMMEDIATE — modify `run_contrastive_eval.py` line 111 to compute `v_steer = λ * v_c + (1-λ) * v_i` instead of `v_c - v_i`.

## Intervention Points

| Point | Feasibility | Expected Leverage | Risk |
|-------|-------------|-------------------|------|
| α sign at L9 | IMMEDIATE | HIGH (could convert death→trim) | LOW (no new code) |
| First-step gate removal | IMMEDIATE | HIGH (max leverage point) | MED (unreliable prompt TT) |
| λ interpolation (contrastive) | IMMEDIATE | MED (smoother control) | LOW |
| Steered-data TT fine-tune | SHORT | HIGH (closes loop) | LOW |
| Residual stream steering | MED | HIGH (architecture-agnostic) | HIGH (modifies model internals) |
| Per-head steering | MED | VERY HIGH (precision) | MED (complex implementation) |
| Self-supervised contrastive | LONG | VERY HIGH (label-free) | HIGH (unproven approach) |
