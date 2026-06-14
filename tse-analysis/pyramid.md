# Concept Hierarchy — RankAdaptation

## Atomic Concepts (Level 1 — Indecomposable)

| ID | Atom | Evidence | Confidence |
|----|------|----------|------------|
| A1 | **KV Cache**: Key-value pairs stored per token, per layer during autoregressive generation | `kv_cache_steering.py`, `run_7b_steering.py` | 10/10 |
| A2 | **Hidden State (h[l,t])**: Activation vector at layer l, token position t | `run_collect_gen_trajs_7b.py:96-98` | 10/10 |
| A3 | **Velocity (v[l,t])**: Difference h[l+1,t] - h[l,t] — direction of state change through layers | `run_collect_gen_trajs_7b.py:98` | 10/10 |
| A4 | **TrajectoryTransformer (TT)**: Transformer that predicts velocity from hidden state sequence | `src/adapters/trajectory_transformer.py` | 10/10 |
| A5 | **Steering Vector (α·v)**: Scaled velocity added to hidden state at a specific layer | `kv_cache_steering.py:69` | 10/10 |
| A6 | **Trim-Tab Layer**: Layer where steering improves accuracy (e.g., L8, L2) | PROJECT_DEBRIEF Finding 2 | 9/10 |
| A7 | **Death Layer**: Layer where steering destroys accuracy (e.g., L9, L15+) | PROJECT_DEBRIEF Finding 2 | 9/10 |
| A8 | **Per-Layer α**: Alpha coefficient, potentially unique per layer | `run_autonomous_sweep.py:232-233` | 10/10 |
| A9 | **Token Generation Step**: Single autoregressive step producing one token | standard transformer behavior | 10/10 |
| A10 | **GSM8K Accuracy**: Binary metric (correct/incorrect) for math word problem | all eval scripts | 10/10 |
| A11 | **Contrastive Velocity**: v_c - v_i = TT_correct(h) - TT_incorrect(h) | `run_contrastive_eval.py:111` | 9/10 |
| A12 | **Reading Head**: Linear probe from Perceiver latents → perplexity (r=0.86) | `run_reading_heads.py` | 8/10 |
| A13 | **Per-layer Influence Centrality**: Degree to which a layer's output affects downstream token selection | conceptual | 6/10 |
| A14 | **Attention Head**: Within-layer attention mechanism computing weighted token mixtures | model architecture | 10/10 |
| A15 | **Capability Threshold**: Minimum model baseline accuracy (~40% GSM8K) needed for steerability | PROJECT_DEBRIEF Finding 3 | 8/10 |
| A16 | **Distribution Shift**: Difference between prompt-time and generation-time hidden state distributions | CROSS_SESSION_BRIEF | 9/10 |
| A17 | **Projection Adaptation**: Transferring TT between models by adapting input/output projection layers | `run_cross_model_transfer.py:36-86` | 8/10 |
| A18 | **Quantization**: 4-bit model loading via bitsandbytes to fit 7B in 8GB VRAM | `run_7b_steering.py:124` | 10/10 |
| A19 | **GQA (Grouped Query Attention)**: K/V heads fewer than Q heads (4 KV, 28 Q for 7B) | standard architecture | 10/10 |
| A20 | **R² Metric**: Coefficient of determination for velocity prediction quality | `run_train_gen_tt_7b.py:24-31` | 10/10 |

## Composites (Level 2 — Direct Compounds)

| ID | Composition | Definition | Emergent Property |
|----|------------|------------|-------------------|
| C2-1 | A1 + A9 | **KV-cache step**: one autoregressive step's cache | Position-indexed memory |
| C2-2 | A2 + A3 | **Steered hidden state**: h' = h + α·v | Modified representation |
| C2-3 | A4 + A2 | **Velocity prediction**: TT(h_seq) → v | Cross-layer pattern capture |
| C2-4 | A6 + A7 | **Layer polarity**: trim-tab or death-layer | Layer-specific effect direction |
| C2-5 | A11 + A6 | **Contrastive trim-tab**: v_c - v_i applied to layer | Normative correction |
| C2-6 | A15 + A5 | **Threshold-gated steering**: only steer if model has capability | Safety filter |

## Composites (Level 3 — System Components)

| ID | Composition | Definition |
|----|------------|------------|
| C3-1 | C2-1 × N_layers × N_tokens | **Full KV cache**: entire cache across all layers and positions |
| C3-2 | C2-3 + A1 + A9 | **KV-cache steering mechanism**: use TT to predict velocity, modify cache |
| C3-3 | C3-2 × C2-4 | **Per-layer steering**: steer only one layer (trim-tab or death-layer identification) |
| C3-4 | A2 + A3 + A5 | **Velocity-injected generation**: generation with steering applied each step |
| C3-5 | A12 + C3-4 | **Confidence-gated steering**: steer only when reading head says "uncertain" |

## Composites (Level 4 — Pipelines)

| ID | Composition | Definition |
|----|------------|------------|
| C4-1 | C3-2 + C3-3 | **Per-layer sweep pipeline**: iterate over all layers, steering each independently |
| C4-2 | A10 + C3-4 | **Steering evaluation pipeline**: generate with steering, measure GSM8K accuracy |
| C4-3 | A11 + C4-1 | **Contrastive sweep pipeline**: sweep layers using contrastive velocity |
| C4-4 | A17 + C4-2 | **Cross-model transfer pipeline**: apply TT from model A to model B |
| C4-5 | C4-2 × C3-4 | **Autonomous sweep**: 4-stage pipeline (per-layer → α sweep → combos → α vector) |

## Peak Concept (Level 5)

| ID | Definition |
|----|------------|
| P | **Velocity-based latent steering**: Modify language model hidden states during generation using predicted velocities (R²=0.85-0.94), applied selectively per layer (trim-tab principle) to amplify correct reasoning without creating capability from nothing |

## Junction Types

| ID | Type | Source → Target | Direction |
|----|------|----------------|-----------|
| J1 | compositional | A2 + A3 + A5 → C2-2 | h + αv = h' |
| J2 | causal | C2-3 → C2-2 | TT prediction causes steered state |
| J3 | causal | C2-2 → A1 | Steered state → modified KV cache entry |
| J4 | causal | A1 → A9 | Modified KV cache → different next token |
| J5 | dependency | A15 → A5 | Model must have capability for steering to work |
| J6 | hierarchical | C2-4 → C3-3 | Layer polarity determines whether steering helps |
| J7 | antagonistic | A6 ↔ A7 | Trim-tab and death layers are opposites |
| J8 | temporal | A9 → A9 | Generation steps are sequential (token t → t+1) |
| J9 | compositional | A11 → C2-5 | Contrastive velocity constructs normative steering |
| J10 | constraint | A18 → C3-2 | 4-bit quantization constrains model precision |
| J11 | dependency | A16 → C2-3 | Distribution shift affects TT accuracy at generation time |
| J12 | synergistic | A12 + C3-4 | Reading head could gate steering for precision |
| J13 | compositional | A17 → C4-4 | Projection adaptation enables cross-model transfer |

## Hallucinatory Pre-Seeds (for Phase 4)

| Atom | Ideal Form (Constraints Lifted) |
|------|--------------------------------|
| A1 | Infinite context KV cache with perfect recall and no memory cost |
| A2 | Fully interpretable hidden states where each dimension maps to a concept |
| A3 | Velocity that points exactly toward the correct answer for any input |
| A4 | TT that predicts both velocity AND direction of correctness (normative from scratch) |
| A5 | Per-layer adaptive α learned via meta-RL in real-time |
| A6 | Every layer is a trim-tab for some input type |
| A7 | No death layers exist — model internal representation is perfectly aligned |
| A10 | Continuous accuracy metric (expected correctness, not binary) |
| A11 | v_c - v_i is everywhere defined and monotonically improves accuracy |
| A15 | Every model, regardless of size, is steerable — capability threshold is algorithmic not architectural |
