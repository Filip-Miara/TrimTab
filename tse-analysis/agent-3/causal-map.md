# Phase 7: Causal Mapping & Counterfactual Analysis

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Causal DAG

```
                          ┌───────────────────────┐
                          │ A14: Prompt Template   │
                          └──────────┬────────────┘
                                     │ causes
                                     ▼
┌──────────────────┐         ┌───────────────────┐
│ A17: MHA Architecture│─────→│ A12: Generation    │
│ (enables steering)│         │ Trajectory        │
└──────────────────┘         └────────┬──────────┘
                                      │ provides data for
                                      ▼
┌──────────────────┐         ┌───────────────────┐
│ A16: GDN/Hybrid  │         │ C2-7: Trajectory  │
│ (blocks steering)│         │ Collection        │
└──────────────────┘         └────────┬──────────┘
                                      │ feeds
                                      ▼
┌──────────────────┐         ┌───────────────────┐
│ A10: Capability  │         │ C2-8: Training    │
│ Threshold        │         │ Pipeline          │
│ (constrains)     │         └────────┬──────────┘
└────────┬─────────┘                   │ trains
         │ blocks if <40%              ▼
         │                    ┌───────────────────┐
         ▼                    │ C2-1: Velocity    │
┌──────────────────┐         │ Predictor (TT)    │
│ A8: GSM8K        │         └────────┬──────────┘
│ Accuracy         │◄────┐            │ predicts A2
│ (target metric)  │     │            ▼
└──────────────────┘     │   ┌───────────────────┐
         ▲               │   │ A2: Velocity      │
         │ caused by     │   └────────┬──────────┘
         │               │            │ multiplied by
         │               │            ▼
         │               │   ┌───────────────────┐
         │               │   │ A5: α (coefficient)│
         │               │   └────────┬──────────┘
         │               │            │ forms
         │               │            ▼
         │               │   ┌───────────────────┐       ┌───────────────────┐
         │               │   │ C2-2: Steering    │──────→│ A3: KV Cache     │
         │               │   │ Operator          │modifies│ (at specific layer)│
         │               │   └────────┬──────────┘       └────────┬──────────┘
         │               │            │                           │ changes
         │               │            ▼                           ▼
         │               │   ┌────────────────────────┐  ┌───────────────────┐
         │               │   │ C2-3: Per-Layer       │  │ A13: Token        │
         │               │   │ Steering              │  │ Divergence (88%)  │
         │               │   └────────┬───────────────┘  └───────────────────┘
         │               │            │
         │               │            ▼
         │               │   ┌───────────────────┐
         │               │   │ A6/A7: Layer      │
         │               │   │ Classification    │
         │               │   │ (trim-tab/death)  │
         │               │   └────────┬──────────┘
         │               │            │ modulates
         │               └────────────┘
         │                    (via J5)
         │
         │   ┌───────────────────┐
         │   │ A9: Contrastive   │
         │   │ Signal            │
         │   │ (v_cor − v_inc)   │
         │   └────────┬──────────┘
         │            │ normative direction
         │            ▼
         │   ┌───────────────────┐
         │   │ C3-2: Contrastive │
         │   │ Steering System   │
         │   └────────┬──────────┘
         │            │ may improve A8
         └────────────┘
```

### Node Statistics

| Node | In-Degree | Out-Degree | Description |
|------|-----------|------------|-------------|
| A14 (Prompt Template) | 0 | 1 | Exogenous |
| A17 (MHA Architecture) | 0 | 1 | Exogenous |
| A16 (GDN/Hybrid) | 0 | 1 | Exogenous |
| A10 (Capability Threshold) | 0 | 1 | Exogenous |
| A12 (Generation Trajectory) | 1 | 1 | |
| C2-7 (Trajectory Collection) | 1 | 1 | |
| C2-8 (Training Pipeline) | 1 | 1 | |
| C2-1 (Velocity Predictor) | 1 | 1 | |
| A2 (Velocity) | 1 | 1 | |
| A5 (α) | 0 | 1 | Exogenous (hyperparameter) |
| C2-2 (Steering Operator) | 2 | 1 | |
| A3 (KV Cache) | 1 | 1 | |
| C2-3 (Per-Layer Steering) | 1 | 2 | **Branching point** |
| A6/A7 (Layer Classification) | 1 | 1 | |
| A13 (Token Divergence) | 1 | 0 | |
| A8 (GSM8K Accuracy) | 2 | 0 | Sink |
| A9 (Contrastive Signal) | 0 | 1 | Exogenous (training decision) |
| C3-2 (Contrastive Steering) | 1 | 1 | |
| A15 (Cross-Model Projection) | 0 | 1 | Exogenous |
| C2-6 (Cross-Model Transfer) | 1 | 1 | |

### Edge Delays

| Edge | Type | Estimated Delay | Notes |
|------|------|----------------|-------|
| A12→C2-7 | Causal (data flow) | 1-2 hours | Generation time for 500 problems |
| C2-7→C2-8 | Causal (pipeline) | Minutes | Data preprocessing |
| C2-8→C2-1 | Causal (training) | 15-30 min | TT training time |
| C2-1→C2-2 | Causal (model load) | <1 sec | In-memory |
| C2-2→A3 | Causal (modification) | <1 ms | Per generation step |
| A3→C2-3 | Causal (steering) | ~200 tokens | Full generation delay |
| C2-3→A8 | Causal (accuracy effect) | ~5 min per sweep | Evaluation time per layer |
| C2-3→A13 | Causal (side effect) | ~200 tokens | Same timeframe as accuracy |
| C2-3→A6/A7 | Causal (discovery) | ~2-4 hours | Full per-layer sweep |

---

## Branching Points (Top-5 by Out-Degree)

| Rank | Node | Out-Degree | Description | Leverage |
|------|------|------------|-------------|----------|
| 1 | **C2-3 (Per-Layer Steering)** | 2 | Determines BOTH accuracy AND token divergence. Small changes in which layer is steered produce dramatically different outcomes. | 9/10 |
| 2 | **C2-2 (Steering Operator)** | 1 (+1 implicit) | Produces both the intended accuracy effect AND unintended token divergence. The branching is in the dual effect of the same operation. | 8/10 |
| 3 | **A10 (Capability Threshold)** | 1 (implicit branching) | If above threshold → steering possible. If below → all attempts fail. Binary branching with no middle ground. | 7/10 |
| 4 | **A17 (MHA Architecture)** | 1 (implicit branching) | Standard MHA → steering possible. Hybrid GDN → steering fails. Architecture choice is a branching decision made before steering begins. | 6/10 |
| 5 | **A5 (α coefficient)** | 1 (continuous branching) | Different α produces continuous variation in outcome. At critical α, behavior may change qualitatively (phase transition). | 5/10 |

---

## Counterfactual Analysis

### CF-1: "What if we never discovered the L8 trim-tab?"

**Scenario**: The per-layer sweep was never performed (or only tested L0-L5). All-layers steering was accepted as the only mode.

**Predicted Outcome**: The project would have concluded that "velocity-based steering doesn't improve accuracy" — the net effect of all-layers steering is negative. The L8+L9 cancelation would mask the trim-tab effect. The entire approach would be abandoned.

**Testability**: Counterfactual is provably true — all-layers steering produced 0% or negative improvement in every model tested.

**Implication**: The per-layer sweep is the single most important experimental decision in the project. Without it, all results would be negative.

### CF-2: "What if we applied negative α to L9 instead of positive α?"

**Scenario**: The signed per-layer sweep (RECOMB-FP5) is performed: α ∈ {-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3} on L9.

**Predicted Outcome**: If L9's death effect is direction-dependent, negative α could produce accuracy IMPROVEMENT (L9 becomes a trim-tab for negative α). This would transform our understanding of death layers — they're not inherently harmful, they're layers where the TT predicts a "wrong" direction.

**Testability**: EASY — use existing sweep infrastructure with negative α values. Time: 2 hours.

**Impact if True**: Doubles or triples the number of steerable layers. All-layers steering becomes viable (positive α on L2, L8; negative α on L7, L9; zero α on neutral layers). Net effect could exceed +20pp.

### CF-3: "What if we trained the TT on steered trajectories?"

**Scenario**: Instead of training TT on unsteered generation trajectories, use the current best TT to generate steered trajectories, then train a new TT on those steered trajectories.

**Predicted Outcome**: The new TT would be more accurate for the steered-model dynamics (resolving the D9 distribution-shift disparity). Two possible regimes:
- **Convergent**: New TT is similar to old TT → one iteration suffices
- **Divergent**: New TT is very different → could oscillate or spiral (each iteration changes the trajectory distribution, requiring further iterations)

**Testability**: MEDIUM difficulty — requires implementing an iterative training loop. Time: 4-8 hours for 3 iterations.

**Impact if Convergent**: Steering becomes self-consistent — the TT predicts the dynamics of the model it's actually steering, not the unsteered model. More accurate steering → larger improvements.

### CF-4: "What if we intervened at the attention head level instead of layer level?"

**Scenario**: Instead of modifying ALL K/V heads at a layer equally (current approach), modify individual attention heads within a layer.

**Predicted Outcome**: Within any layer, some heads are trim-tabs and some are death-heads. Head-level steering would (1) increase the signal-to-noise ratio within trim-tab layers, (2) possibly recover signal from predominantly-death layers that contain a few beneficial heads, (3) reduce token divergence (fewer heads modified = more targeted effect).

**Testability**: HARD — requires significant code changes to split KV cache modifications per head. Time: 1-2 days.

**Impact if True**: Could achieve +30pp+ on GSM8K by isolating the specific attention heads that perform reasoning computation, ignoring heads that do other functions (syntax, position encoding, etc.).

### CF-5: "What if we used a different proxy metric than GSM8K?"

**Scenario**: Replace GSM8K with (a) a syntactic task (e.g., punctuation restoration) or (b) a different reasoning task (e.g., ARC or BBH).

**Predicted Outcome**: 
- **Syntactic task**: Steering at L8 would have NO effect (or negative effect) — trim-tab improvement is specific to reasoning tasks, not general output modification
- **Different reasoning task (ARC)**: L8 would show a different effect size or different optimal layer — trim-tab pattern is moderately task-specific

**Testability**: EASY — run existing steered model on ARC/BBH benchmarks without re-training. Time: 2-3 hours.

**Impact if True**: Validates or invalidates the "trim-tab pattern generalizes to all reasoning" hypothesis. If the pattern changes substantially, per-task trim-tab discovery is needed.

---

## Intervention Points (Feasible External Modulations)

| Node | Intervention | Feasibility | Expected Effect | Cost |
|------|-------------|-------------|-----------------|------|
| A5 (α) | Change from 0.1 to range [-0.3, 0.3] | ✅ Immediate | Discover optimal α per layer | 2 hours eval |
| C2-3 (Layer selection) | Signed per-layer sweep | ✅ Immediate | Identify direction-dependent trim-tabs | 4 hours |
| C2-1 (TT retraining) | Train on steered trajectories | ⚠️ Medium | Resolve distribution shift | 1 day |
| A15 (Projection) | Improve cross-model projection | ⚠️ Medium | Better transfer | 1 day |
| C2-2 (Steering operator) | Add manifold projection | ⚠️ Medium | Prevent OOD states | 1 day |
| A3 (KV cache) | Head-level modification | ❌ Hard | Finer granularity | 2+ days |
| C2-8 (Training pipeline) | Add uncertainty estimation | ❌ Hard | Better calibration | 2+ days |
| A8 (Evaluation) | Add non-math tasks | ✅ Immediate | Generalization evidence | 3 hours |
| A9 (Contrastive) | Evaluate existing TTs | ✅ Immediate | Validate normative direction | 2 hours |
