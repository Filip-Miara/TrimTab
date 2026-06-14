# Phase 1: Atomic Decomposition & Pyramid Construction

**Subject**: RankAdaptation — Velocity-based latent steering
**Date**: 2026-06-14

---

## Decomposition Method

Recursive subdivision of the project into atoms — concepts that cannot be further decomposed without destroying meaning. Each atom is grounded in evidence from PROJECT_DEBRIEF.md. Junctions are typed before nodes are analyzed (junction-first processing).

---

## Level 1: Atomic Concepts (Atoms)

### A1: Hidden State Velocity
- **Definition**: The change in hidden state activation across consecutive tokens during autoregressive generation: v_t = h_{t+1} − h_t
- **Evidence**: Core innovation; trajectory collection pipeline records these; TT predicts them
- **Measurable**: R²=0.85-0.94 across models
- **Hallucinatory Pre-Seed**: An oracle velocity field that points directly from current state toward the correct answer's hidden state, regardless of model capability

### A2: TrajectoryTransformer (TT)
- **Definition**: A neural network trained to predict next-token velocity from current hidden state
- **Evidence**: best_gen_tt_7b.pt (192MB); R²=0.855 Qwen2.5-7B
- **Architecture**: Predictor network, possibly transformer-based
- **Hallucinatory Pre-Seed**: A universal velocity predictor that generalizes across all models, tasks, and languages without retraining

### A3: KV-Cache Steering
- **Definition**: Modifying key/value entries in the KV cache during generation by adding α · v_pred to K/V
- **Evidence**: 88% token divergence achieved; L8 +20pp accuracy improvement
- **Mechanism**: v_pred added to K/V at specific layer(s), scaled by α
- **Hallucinatory Pre-Seed**: Non-invasive steering that modifies attention distributions directly without touching K/V values, eliminating the off-manifold problem

### A4: Per-Layer Selectivity
- **Definition**: Steering individual layers rather than all layers simultaneously
- **Evidence**: L8 +20pp vs all-layers -45pp; L9 death layer at -23pp
- **Scope**: Single-layer intervention with layer-specific α
- **Hallucinatory Pre-Seed**: Automatic discovery of optimal per-layer steering policy through meta-learning

### A5: Trim-Tab Layer
- **Definition**: A layer where steering improves accuracy beyond baseline
- **Evidence**: L8 (+20pp), L2 (+17pp), L3 (+13pp), L5 (+13pp), L10 (+17pp) on Qwen2.5-7B
- **Pattern**: Consistent across GSM8K and SVAMP; preserved under cross-model transfer
- **Hallucinatory Pre-Seed**: Layers whose computation is naturally aligned with the steering direction, such that steering reinforces rather than disrupts

### A6: Death Layer
- **Definition**: A layer where steering degrades accuracy below baseline
- **Evidence**: L9 (-23pp), L7 (-14pp), L15+ (complete collapse) on Qwen2.5-7B
- **Characteristic**: Steering these layers destroys model output quality
- **Hallucinatory Pre-Seed**: A diagnostic technique that identifies death layers as those performing critical ordering/structuring computation, and routes steering around them

### A7: Capability Threshold
- **Definition**: The minimum baseline accuracy required for steering to be effective
- **Evidence**: SmolLM2 (4%) → all harmful; Qwen2.5-7B (73%) → +20pp; Qwen2.5-Math-1.5B (38%) → all harmful
- **Boundary**: ~40% GSM8K (empirically estimated)
- **Hallucinatory Pre-Seed**: No threshold exists; steering works at all capability levels with the right intervention method

### A8: Contrastive TT
- **Definition**: Two TTs trained separately on correct and incorrect trajectories; steering uses v_correct − v_incorrect
- **Evidence**: Two TTs trained (R²≈0.83 each); evaluation pending
- **Rationale**: Convert descriptive TT (reproduces errors) to normative TT (points toward correct)
- **Hallucinatory Pre-Seed**: A single model trained with contrastive loss that directly maximizes the angle between v_pred and the incorrect trajectory direction

### A9: GSM8K Benchmark
- **Definition**: Grade-School Math dataset (8K problems); primary evaluation metric
- **Evidence**: Baseline 73% on Qwen2.5-7B-Instruct; L8 improves to 65% (note: lower than 73% — this is a subset of 100 problems with different sampling)
- **Hallucinatory Pre-Seed**: A multi-domain benchmark suite that reveals the full capability profile of steering across tasks

### A10: Steering Strength (α)
- **Definition**: Scalar multiplier applied to predicted velocity before adding to K/V
- **Evidence**: Default α=0.1 used in all experiments; no systematic α sweep reported
- **Hallucinatory Pre-Seed**: An α that varies per token position, per attention head, and per layer, learned through RL

### A11: Token Divergence
- **Definition**: Measure of how much the steered generation diverges from the unsteered generation
- **Evidence**: 88% token divergence achieved; steering does change model output
- **Hallucinatory Pre-Seed**: Controlled divergence — steer only the reasoning trace, not the answer format or boilerplate

### A12: Cross-Model Transfer
- **Definition**: Using TT trained on model A to steer model B via projection adaptation
- **Evidence**: SmolLM2 TT (R²=0.94, 960-dim) → Qwen2.5-7B (3584-dim); L8 preserved as best trim tab
- **Hallucinatory Pre-Seed**: Zero-shot transfer without projection (model-agnostic velocity representation)

### A13: GQA (Grouped Query Attention)
- **Definition**: Multi-query attention where key/value heads are shared across query groups
- **Evidence**: SmolLM2 (5KV heads), Qwen2.5 (2-4KV heads) handled correctly in steering
- **Hallucinatory Pre-Seed**: Attention-aware steering that operates on query-specific rather than head-specific values

### A14: Trajectory Collection Pipeline
- **Definition**: Infrastructure to record hidden states during autoregressive generation
- **Evidence**: 83 files × 7B trajectories on HDD (35GB); works across models
- **Hallucinatory Pre-Seed**: Real-time trajectory collection without storage bottleneck, streaming directly into training

### A15: Async Data Loading
- **Definition**: Background thread prefetching trajectory data to hide disk I/O
- **Evidence**: Implemented and working; GPU utilization improved from ~50%
- **Hallucinatory Pre-Seed**: Zero-copy data loading direct from GPU to GPU, eliminating CPU bottleneck entirely

### A16: GPU Cache
- **Definition**: Keeping trajectory data in VRAM across training epochs
- **Evidence**: Eliminates per-batch GPU transfers; faster training
- **Hallucinatory Pre-Seed**: Streaming GPU memory management that dynamically swaps trajectory data based on access patterns

### A17: Checkpoint Resume
- **Definition**: Saving and restoring optimizer state + epoch for crash recovery
- **Evidence**: Resumable training with `--resume` flag
- **Hallucinatory Pre-Seed**: Continuous checkpointing with zero-overhead, eliminating any loss of progress

### A18: Hybrid Attention (GDN+FA)
- **Definition**: GatedDeltaNet with FullAttention — recurrent + attention hybrid
- **Evidence**: Qwen3.5-2B uses this; only 25% layers have standard K/V caches; steering fails
- **Hallucinatory Pre-Seed**: A universal steering mechanism that works on any attention architecture, including recurrent and hybrid variants

### A19: Logit Correction
- **Definition**: Modifying logit outputs directly instead of K/V cache
- **Evidence**: Failed (0% gen on prompt-trained; =baseline on gen-trained)
- **Hallucinatory Pre-Seed**: Logit correction that works because it operates on the contrastive logit difference

### A20: PPL-Modulated Correction
- **Definition**: Only apply steering when model perplexity indicates uncertainty
- **Evidence**: <0.1% gate rate; model confidently wrong — perplexity doesn't correlate with correctness
- **Hallucinatory Pre-Seed**: An uncertainty estimation method that actually correlates with correctness for steering-gating

---

## Level 2: Composite Concepts

| Composite | Constituent Atoms | Junctions |
|-----------|------------------|-----------|
| C1: Velocity Prediction System | A1 (Velocity) + A2 (TT) + A14 (Trajectory Collection) | A1←A2 (TT predicts velocity); A14→A2 (collection feeds TT) |
| C2: Steering Mechanism | A3 (KV-Cache Steering) + A10 (α) + A4 (Per-Layer Selectivity) | A10 modulates A3; A4 determines where A3 applies |
| C3: Trim-Tab Discovery System | C1 + C2 + A9 (GSM8K) | C1 predicts, C2 steers, A9 measures |
| C4: Layer Analysis Suite | A4 (Selectivity) + A5 (Trim-Tab) + A6 (Death Layer) + A9 (GSM8K) | A4 enables discovery of A5 and A6; A9 quantifies both |
| C5: Contrastive Steering System | A8 (Contrastive TT) + C2 (Steering Mechanism) + C4 (Layer Analysis) | A8 replaces A2 in C1; C4 evaluates both |
| C6: Transfer System | A12 (Cross-Model Transfer) + A13 (GQA Handling) + A1 (Velocity) | A12 adapts; A13 ensures architecture compatibility |
| C7: Training Infrastructure | A14 (Collection) + A15 (Async Loading) + A16 (GPU Cache) + A17 (Checkpoint) | All four pipeline stages; sequential dependency |
| C8: Failed Mechanisms | A18 (Hybrid) + A19 (Logit) + A20 (PPL) | Each failed independently; share "wrong intervention surface" pattern |
| C9: Knowledge Base | All findings, data files, checkpoints | Cross-referenced across experiments |

---

## Level 3: Composites of Composites

| Composite | Constituent Composites | Emergent Property |
|-----------|----------------------|-------------------|
| L3-1: Full Steering Pipeline | C3 (Trim-Tab Discovery) + C7 (Training Infrastructure) | End-to-end system that collects, trains, steers, and evaluates |
| L3-2: Research Program | L3-1 + C5 (Contrastive) + C6 (Transfer) | The complete experimental apparatus |
| L3-3: Theoretical Framework | C4 (Layer Analysis) + C8 (Failed Mechanisms) + Knowledge Base | Understanding of WHY steering works/fails |

---

## Level 4: Peak Concept

| Peak | Constituent |
|------|-------------|
| P: RankAdaptation System | L3-2 (Research Program) + L3-3 (Theoretical Framework) |

---

## Junction Typology

| Junction ID | Type | From | To | Description |
|-------------|------|------|-----|-------------|
| J1 | Causal | A1 (Velocity) | A2 (TT) | Velocity existence enables TT training |
| J2 | Compositional | A2 (TT) | A3 (Steering) | TT output is input to steering |
| J3 | Modulatory | A10 (α) | A3 (Steering) | α scales steering magnitude |
| J4 | Hierarchical | A4 (Selectivity) | A5 (Trim-Tab) | Selectivity is the method; trim-tabs are discovered instances |
| J5 | Antagonistic | A5 (Trim-Tab) | A6 (Death Layer) | One improves, the other degrades; they coexist in same model |
| J6 | Conditional | A7 (Threshold) | A5 (Trim-Tab) | Capability threshold must be met for trim-tabs to exist |
| J7 | Dependency | A14 (Collection) | A2 (TT) | Collection must run before TT can be trained |
| J8 | Temporal | A15 (Async) | A7 (Threshold) | Async loading is a performance optimization, not fundamental |
| J9 | Constraint | A13 (GQA) | A3 (Steering) | GQA architecture constrains how steering is applied |
| J10 | Causal | A8 (Contrastive) | A5 (Trim-Tab) expected | Contrastive TT should produce trim-tabs (untested) |
| J11 | Antagonistic | A18 (Hybrid) | A3 (Steering) | Hybrid attention prevents standard KV-cache steering |
| J12 | Compositional | C8 (Failed) | L3-3 (Theory) | Failed mechanisms inform theoretical understanding |

---

## Hallucinatory Pre-Seeds Index

Each atom's hallucinatory pre-seed represents its "ideal form if constraints were lifted." These will seed Phase 4's divergent pulse.

| Atom | Pre-Seed Summary |
|------|------------------|
| A1 | Oracle velocity field pointing to correct answer |
| A2 | Universal velocity predictor across models |
| A3 | Non-invasive attention distribution steering |
| A4 | Meta-learned per-layer steering policy |
| A5 | Computation naturally aligned with steering |
| A6 | Diagnostic that routes around death layers |
| A7 | No capability threshold with right method |
| A8 | Unified contrastive loss instead of two TTs |
| A9 | Multi-domain benchmark suite |
| A10 | Per-token, per-head, per-layer learned α |
| A11 | Controlled divergence (trace only) |
| A12 | Zero-shot cross-model transfer |
| A13 | Attention-aware head-specific steering |
| A14 | Streaming real-time collection without storage |
| A15 | Zero-copy GPU-to-GPU data loading |
| A16 | Streaming dynamic GPU memory management |
| A17 | Continuous zero-overhead checkpointing |
| A18 | Universal architecture-agnostic steering |
| A19 | Contrastive logit correction |
| A20 | Accurate uncertainty estimation for gating |
