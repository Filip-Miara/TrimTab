# Phase 1: Atomic Decomposition & Pyramid Construction

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14
**Evidence Grounding**: Enabled

---

## Atoms (Level 1 — Indecomposable Concepts)

| ID | Atom | Definition | Evidence |
|----|------|-----------|----------|
| A1 | **Hidden State** | The residual stream activation at a given layer and token position | Core LM mechanism |
| A2 | **Velocity** | The difference between consecutive hidden states: h_{t+1} − h_t | Defined in TT training |
| A3 | **KV Cache** | Cached Key/Value projection tensors for batched attention computation | Standard transformer mechanism |
| A4 | **TrajectoryTransformer** | Neural network predicting velocity from a sequence of past hidden states | R²=0.85-0.94 on 3 models |
| A5 | **Steering Coefficient α** | Scalar multiplier applied to predicted velocity before KV-cache modification | Hyperparameter (0.0-1.0) |
| A6 | **Trim-Tab Layer** | Layer where positive α improves accuracy | L8: +20pp on Qwen2.5-7B |
| A7 | **Death Layer** | Layer where positive α degrades accuracy | L9: -23pp on Qwen2.5-7B |
| A8 | **GSM8K Accuracy** | Percentage of correct answers on GSM8K math word problems | Proxy for reasoning quality |
| A9 | **Contrastive Signal** | v_{correct} − v_{incorrect}: the difference between velocity predictions from correct and incorrect trajectories | Trained, evaluation pending |
| A10 | **Capability Threshold** | Minimum baseline accuracy (~40%) below which steering cannot improve results | Supported by 5-model comparison |
| A11 | **Hidden State Manifold** | The high-dimensional space in which hidden states reside | Assumed to have structure (R²>0 supports) |
| A12 | **Generation Trajectory** | Sequence of hidden states produced during autoregressive generation | ~200 tokens for GSM8K |
| A13 | **Token Divergence** | Percentage of tokens that differ between steered and unsteered generations | 88% measured |
| A14 | **Prompt Template** | Chat template applied to input before model forward pass | `apply_chat_template()` |
| A15 | **Cross-Model Projection** | Linear layer projecting TT predictions from source to target model dimension | SmolLM2 (960) → Qwen2.5 (3584) |
| A16 | **DeltaNet/GDN** | Gated Delta Network — recurrent-style attention alternative in hybrid architectures | Used in Qwen3.5 |
| A17 | **Standard MHA** | Standard Multi-Head Attention with K/V caching | Preferred architecture |
| A18 | **Logit Correction** | Modifying output logits based on predicted velocities | Failed (0% generation) |
| A19 | **PPL Modulation** | Only applying steering when the model's confidence (PPL) is below threshold | Failed (<0.1% gate rate) |
| A20 | **Hidden State Dimensionality** | Size of the residual stream per layer | SmolLM2: 960, Qwen2.5-7B: 3584 |

## Composites by Level

### Level 2 — Simple Composites

| ID | Composite | Constituents | Description |
|----|-----------|--------------|-------------|
| C2-1 | **Velocity Predictor** | A2, A4, A12 | TT trained on generation trajectories outputs predicted velocity |
| C2-2 | **Steering Operator** | A3, A5, C2-1 | Applying predicted velocity × α to KV cache entries at a layer |
| C2-3 | **Per-Layer Steering** | C2-2, A6, A7 | Applying steering to a single layer; discovering trim-tab/death classification |
| C2-4 | **Contrastive TT Pair** | A9, C2-1 | Two TTs (correct, incorrect) whose outputs are subtracted |
| C2-5 | **All-Layers Steering** | C2-2 × N_layers | Applying same α to all layers simultaneously |
| C2-6 | **Cross-Model Transfer** | A15, C2-1 | TT trained on model A, projected and evaluated on model B |
| C2-7 | **Trajectory Collection** | A12, A14, A17 | Pipeline to record hidden states during generation with prompt formatting |
| C2-8 | **Training Pipeline** | C2-7, A11, A13 | Async data loading, GPU caching, checkpoint resume |
| C2-9 | **Architecture Interface** | A16, A17, A3 | Abstraction over attention types for KV-cache access |
| C2-10 | **PPL Gate** | A19, C2-2 | Only steer when model's perplexity indicates uncertainty |

### Level 3 — Complex Composites

| ID | Composite | Constituents | Description |
|----|-----------|--------------|-------------|
| C3-1 | **Standard Steering System** | C2-2, C2-3, C2-5 | The basic framework: train TT, steer per-layer, evaluate accuracy |
| C3-2 | **Contrastive Steering System** | C2-4, C2-3, C3-1 | Extends standard system with contrastive signal |
| C3-3 | **Cross-Validated Steering** | C3-1, C2-6, A15 | Validates steering across models, datasets |
| C3-4 | **Capability Assessment** | A10, A8 | Determines whether a model can benefit from steering |

### Level 4 — Peak Composite

| ID | Composite | Constituents | Description |
|----|-----------|--------------|-------------|
| P | **RankAdaptation System** | C3-1, C3-2, C3-3, C3-4 | Complete velocity-based latent steering framework |

---

## Junctions (Typed Relationships)

| ID | Type | Source(s) | Target(s) | Description |
|----|------|-----------|-----------|-------------|
| J1 | **Causal** | A2 (Velocity) | C2-1 (Velocity Predictor) | Velocity is the target variable the TT learns to predict |
| J2 | **Compositional** | C2-1, A5 | C2-2 (Steering Operator) | Predicted velocity + α form the steering operator |
| J3 | **Dependency** | C2-2 | A3 (KV Cache) | Steering operator modifies KV cache (depends on cache structure) |
| J4 | **Constraint** | A10 (Capability Threshold) | C2-3 (Per-Layer Steering) | Only works when baseline > ~40% |
| J5 | **Causal** | C2-3 | A8 (GSM8K Accuracy) | Steering at trim-tab/death layers causes accuracy changes |
| J6 | **Hierarchical** | C2-3 | C3-1 (Standard Steering) | Per-layer steering is a component of the full system |
| J7 | **Dependency** | A12 (Generation Trajectory) | C2-7 (Trajectory Collection) | Collection pipeline depends on generating trajectories |
| J8 | **Antagonistic** | A6 (Trim-Tab) | A7 (Death Layer) | Some layers help, others harm — they are in tension |
| J9 | **Dependency** | A17 (Standard MHA) | C2-2 (Steering Operator) | Standard MHA is required for KV-cache modification to work |
| J10 | **Causal** | A9 (Contrastive Signal) | C3-2 (Contrastive Steering) | v_correct − v_incorrect is the steering direction for contrastive system |
| J11 | **Synergistic** | C3-1, C2-6 | C3-3 (Cross-Validated Steering) | Standard + cross-model transfer strengthens validation |
| J12 | **Constraint** | A16 (DeltaNet/GDN) | C2-2 (Steering Operator) | Hybrid architectures limit KV-cache steering to ~25% of layers |
| J13 | **Temporal** | A12 (Generation Trajectory) | A2 (Velocity) | Velocity is computed over the temporal sequence of hidden states |
| J14 | **Causal** | A14 (Prompt Template) | A8 (GSM8K Accuracy) | Template caused 4%→73% baseline jump |
| J15 | **Dependency** | A15 (Cross-Model Projection) | C2-6 (Cross-Model Transfer) | Projection layer is required for transfer to work |
| J16 | **Compositional** | C2-4, C2-3 | C3-2 (Contrastive Steering) | Contrastive pair + per-layer selection form contrastive system |
| J17 | **Antagonistic** | C2-5 (All-Layers) | A8 (GSM8K Accuracy) | All-layers steering is net negative |
| J18 | **Dependency** | C2-7 (Trajectory Collection) | C2-8 (Training Pipeline) | Training depends on collected trajectories |
| J19 | **Temporal** | C2-8 → C2-1 → C2-3 → A8 | — | Pipeline: train → steer → evaluate |
| J20 | **Constraint** | A18 (Logit Correction) | A8 (GSM8K Accuracy) | Logit correction failed to improve accuracy |

---

## Hallucinatory Pre-Seeds (Unconstrained Ideals)

| Atom | Pre-Seed | Lifted Constraint |
|------|----------|-------------------|
| A1 | Zero-loss latent representation where all task-relevant features are linearly separable | No compute/storage limits on hidden state dimension |
| A2 | Velocity directly encodes the "next reasoning step" — not the next token, but the next semantic unit | Removing tokenization constraint |
| A3 | KV cache that can be arbitrarily modified at any granularity (element-level, not just layer-level) | No hardware cache size limits |
| A4 | Universal TT that predicts velocities for any model, any task, any layer | Removing model-specific training |
| A5 | Learned α per (layer, head, token, task) — optimal steering magnitude everywhere | No search cost constraint |
| A6 | Every layer is a trim-tab for some task | Removing per-layer behavior variance |
| A7 | Zero death layers — all steering is non-negative | Removing antagonistic behavior |
| A8 | Steering achieves 100% on GSM8K | Removing the gap to perfect accuracy |
| A9 | Contrastive signal is provably normative — corrected trajectory matches ground-truth reasoning | Removing ambiguity between fluency and correctness |
| A10 | No capability threshold — any model can be steered toward its ceiling | Removing the 40% barrier |
| A11 | Flat manifold where all steering directions are in-distribution | Removing curvature/violation concern |
| A12 | Generation trajectory is available instantly for any input | Removing generation latency |
| A13 | Token divergence is perfectly controllable — steer exactly the tokens that need changing | Removing side-effect tokens |
| A14 | Universal prompt template that works optimally for any model/task | Removing template engineering |
| A15 | Zero-loss cross-model projection — velocities transfer perfectly across any model pair | Removing dimensionality barriers |
| A16 | GDN layers have interpretable, steerable recurrent states | Removing black-box recurrent dynamics |
| A17 | Every attention variant is steerable via a unified interface | Removing architecture-specific limitations |
| A18 | Logit correction perfectly adjusts output distribution without side effects | Removing generation collapse |
| A19 | PPL gate is 100% accurate — steers exactly when needed, never when not | Removing confidence-calibration gap |
| A20 | Hidden state dimension adapts dynamically to task complexity | Removing fixed-architecture constraint |
