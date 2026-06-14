# Phase 1: Atomic Decomposition & Pyramid Construction

---

## 1.1 Concept Extraction

Subjects: The RankAdaptation project as a whole, decomposed into atomic concepts.

### Level 0 — Atoms (indecomposable without destroying meaning)

| ID | Atom | Definition | Domain | Evidence |
|----|------|-----------|--------|----------|
| A01 | hidden_state | Vector in R^d (d=2048 for 7B, 1536 for 1.5B, 960 for 360M) representing transformer layer output | LM internals | Uncontroversial — standard LM definition |
| A02 | velocity | v[l] = h[l+1] − h[l]; the finite difference between consecutive hidden states | LM internals | Code: trajectory_transformer.py trains on this target |
| A03 | trajectory | Ordered sequence [h[0], h[1], ..., h[L]] of hidden states across all layers at a single token position | LM internals | Collected in *trajs.py scripts |
| A04 | KV_cache | Key/value pairs at each layer used by attention to attend to past tokens | LM internals | HuggingFace standard; kv_cache_steering.py modifies it |
| A05 | steering_vector | α · v_pred; the modification applied to hidden state | Intervention | kv_cache_steering.py:69 |
| A06 | alpha | Scalar steering strength parameter | Hyperparameter | Swept [0.01, 0.03, 0.05, 0.1, 0.3, 0.5] in run_autonomous_sweep.py |
| A07 | trim_tab_layer | Layer where positive steering improves accuracy | Empirical discovery | L8: +20pp on 7B |
| A08 | death_layer | Layer where positive steering degrades accuracy | Empirical discovery | L9: −23pp on 7B |
| A09 | baseline_accuracy | Model's raw GSM8K accuracy without steering | Metric | 73% for Qwen2.5-7B-Instruct |
| A10 | R_squared | Variance explained by TT's velocity predictions | Metric | 0.85-0.94 across models |
| A11 | generation_step | One autoregressive token prediction | Process | Standard LM generation loop |
| A12 | token_logits | Output distribution over vocabulary at each step | LM output | Standard |
| A13 | attention_computation | Transformer's causal self-attention using K, Q, V | Mechanism | Standard MHA or GDN+FA hybrid |
| A14 | chat_template | Instruction-tuned model's expected input format | Preprocessing | apply_chat_template() in 7B evaluation |
| A15 | projection_adaptation | Linear layer aligning source/destination model hidden dims for cross-model transfer | Mechanism | run_cross_model_transfer.py |
| A16 | contrastive_difference | v_correct − v_incorrect; difference between two TT predictions | Intervention | run_contrastive_eval.py:111 |
| A17 | capability_threshold | Minimum baseline accuracy above which steering can improve performance | Empirical boundary | ~40% GSM8K |
| A18 | hidden_manifold | The data manifold of activation vectors at a given layer | Theoretical concept | Assumed by h' = h + α·v |
| A19 | correct_answer | Ground-truth answer for a GSM8K problem | Ground truth | From dataset |
| A20 | quantization | 4-bit model compression (bitsandbytes) for fitting large models in GPU memory | Infrastructure | run_autonomous_sweep.py:121 |

### Level 1 — Composites

| ID | Composite | Atoms | Description |
|----|-----------|-------|-------------|
| C01 | TrajectoryTransformer (TT) | A01, A02, A03 | Neural net predicting velocities from hidden state trajectories |
| C02 | KV-steering mechanism | A04, A05, A06, A13 | Process: compute v, modify KV entries with α·v |
| C03 | Per-layer sweep | A07, A08, A09, C02 | Systematic test of steering each layer individually |
| C04 | Contrastive TT pair | A16, C01 | Two TTs (correct, incorrect) differenced for normative signal |
| C05 | Cross-model transfer | A15, C01, C02 | Project-dim TT from model A → apply to model B |
| C06 | Baseline evaluation | A09, A14, C01 | Eval without steering using chat template |
| C07 | Cross-dataset generalization | A07, A08, A09 | Replicate per-layer pattern on SVAMP |
| C08 | Async data pipeline | A09 (derived), A20 | Background GPU cache, prefetching, checkpointing |
| C09 | Capability threshold hypothesis | A17, C03 | Empirical generalization about steering limits |
| C10 | All-layers steering | A05, A06, C02, A07, A08 | Apply same α to all layers simultaneously |

### Level 2 — Higher Composites

| ID | Composite | Level-1 Parts | Description |
|----|-----------|---------------|-------------|
| P01 | Steering Framework | C01, C02, C06, C08 | Full pipeline: collect trajectories → train TT → steer KV cache → evaluate |
| P02 | Empirical Results | C03, C04, C05, C07, C09 | The body of experimental evidence |
| P03 | Contrastive Pipeline | C04, C03, C02 | Contrastive training → per-layer sweep → evaluation |

### Level 3 — Peak Concept

| ID | Concept | Description |
|----|---------|-------------|
| PC | Velocity-based Latent Steering | The overall thesis: hidden state velocities learn structure that can guide LM generation toward correct reasoning |

## 1.2 Junction Typology

### Compositional Junctions

| JID | Type | From | To | Description |
|-----|------|------|----|-------------|
| J01 | compositional | A01, A02, A03 | C01 | Velocity is computed from hidden states; trajectory is a sequence of hidden states |
| J02 | compositional | A04, A05, A06, A13 | C02 | KV steering combines velocity prediction + alpha scaling + KV projection |
| J03 | compositional | A16, C01 | C04 | Contrastive pair is two TTs differenced |
| J04 | compositional | A09, A14 | C06 | Baseline eval = raw model + chat template |
| J05 | compositional | C01, C02, C06, C08 | P01 | Full framework composes all sub-pipelines |

### Causal Junctions

| JID | Type | From | To | Confidence | Description |
|-----|------|------|----|-----------|-------------|
| J06 | causal | A02 | A18 | 7/10 | Velocity dynamics determine the hidden manifold geometry |
| J07 | causal | C01 | C03 | 9/10 | TT enables per-layer sweep (need velocity predictions) |
| J08 | causal | C03 | A07, A08 | 9/10 | Per-layer sweep discovers trim tabs + death layers |
| J09 | causal | C03 | A17 | 8/10 | Sweep across models establishes capability threshold |
| J10 | causal | A06 | C02 | 10/10 | Alpha determines steering strength |
| J11 | causal | C04 | C02 | 6/10 | Contrastive difference provides normative steering direction |
| J12 | causal | A14 | A09 | 10/10 | Chat template was primary driver of 4%→73% baseline jump |
| J13 | causal | A17 | C09 | 7/10 | Capability threshold is an empirical consequence of sweep results |

### Temporal Junctions

| JID | Type | From | To | Description |
|-----|------|------|----|-------------|
| J14 | temporal | A03 | A11 | Trajectory is collected AFTER generation step |
| J15 | temporal | C01 | C02 | TT trained BEFORE steering applied |
| J16 | temporal | C06 | C03 | Baseline established BEFORE per-layer sweep |
| J17 | temporal | C03 | C07 | Per-layer results on GSM8K BEFORE cross-dataset test |

### Hierarchical Junctions

| JID | Type | From | To | Description |
|-----|------|------|----|-------------|
| J18 | hierarchical | A07, A08 | C03 | Trim tabs and death layers are output categories of per-layer sweep |
| J19 | hierarchical | A17 | C09 | Capability threshold is a property derived from C03 |

### Constraint Junctions

| JID | Type | From | To | Description |
|-----|------|------|----|-------------|
| J20 | constraint | A15 | C05 | Projection adaptation dimension mismatch constrains cross-model transfer |
| J21 | constraint | A20 | C06 | 4-bit quantization constrains model size that fits in GPU memory |
| J22 | constraint | A06 | C02 | Alpha cannot be arbitrarily large (prev/pending: degrades to noise) |
| J23 | constraint | A17 | C02 | Cannot steer models below capability threshold |

### Antagonistic Junctions

| JID | Type | From | To | Description |
|-----|------|------|----|-------------|
| J24 | antagonistic | A07 | A08 | Trim tabs and death layers oppose each other — steering L8 (+20pp) while avoiding L9 (−23pp) |
| J25 | antagonistic | C10 | A07 | All-layers steering kills the trim-tab benefit |
| J26 | antagonistic | A04 (hybrid) | C02 | Hybrid attention architecture resists standard KV-cache steering |

### Synergistic Junctions

| JID | Type | From | To | Description |
|-----|------|------|----|-------------|
| J27 | synergistic | C04 | C02 | Contrastive TT + per-layer selectivity may be additive |
| J28 | synergistic | C08 | P01 | Async pipeline + GPU caching enables efficient iteration |
| J29 | synergistic | A14 | A09 | Chat template + enough max tokens enables baseline to be meaningful |

## 1.3 Hallucinatory Pre-Seeds

For each atom, the ideal form if constraints were lifted:

| Atom | Pre-Seed |
|------|----------|
| A01 hidden_state | Continuous differentiable manifold with known geodesic distances |
| A02 velocity | Exact (not predicted) ground-truth velocity from oracle |
| A03 trajectory | Full 2D probability path over the hidden manifold (not just a single sequence) |
| A04 KV_cache | Infinite context window, zero memory cost, differentiable key/value operations |
| A05 steering_vector | Optimally learned by RL per (layer, token, context) |
| A06 alpha | Dynamic per-token α learned by meta-controller |
| A07 trim_tab_layer | Explainable via mechanistic interpretability: we know WHY L8 is special |
| A08 death_layer | Explainable via mechanistic interpretability: we know WHY L9 destroys accuracy |
| A09 baseline_accuracy | Perfect 100% — then steering would be unnecessary |
| A10 R_squared | Perfect 1.0 — velocity is fully deterministic |
| A11 generation_step | Zero-latency, infinite batch, no memory |
| A12 token_logits | Calibrated, robust, interpretable |
| A13 attention_computation | Full quadratic attention over 10M tokens |
| A14 chat_template | Universal format that all models accept |
| A15 projection_adaptation | Zero-shot without training; preserves velocity direction exactly |
| A16 contrastive_difference | Analytically derived optimal direction rather than learned |
| A17 capability_threshold | Zero — steering works for any baseline |
| A18 hidden_manifold | Known closed-form; can compute geodesic steering paths |
| A19 correct_answer | Known for all problems; no extraction errors |
| A20 quantization | Zero loss; exact weights preserved at any bit-width |
