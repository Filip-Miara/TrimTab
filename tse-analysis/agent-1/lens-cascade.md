# Phase 2: Multi-Lens Analysis Cascade

**Subject**: RankAdaptation — Velocity-based latent steering
**Date**: 2026-06-14

---

## Lens 1: ANALOGICAL

**Input**: Pyramid atoms + junctions from Phase 1

### Structural Findings

| Analogue Domain | Analogous Structure | Mapping to RankAdaptation |
|-----------------|-------------------|---------------------------|
| Aviation (trim tabs) | Small control surface that adjusts the control surface that adjusts the control surface | Trim-tab layers (L8) are small corrections to specific layers that amplify through residual stream |
| Neuroscience (neuroplasticity) | Brain regions compensate after damage through reorganization | Death layers (L9) may represent regions whose computation is critical; steering them creates compensation demands |
| Classical control theory | Feedforward control with disturbance rejection | TT predicts velocity (feedforward); death layers are unmodeled disturbances |
| Chemistry (catalysis) | Catalyst lowers activation energy without being consumed | TT steering at trim-tab layers may reduce the "activation energy" for correct reasoning paths |
| Optics (interference) | Constructive/destructive interference patterns | Trim-tab (+20pp) and death (-23pp) resemble constructive/destructive interference at specific layers |
| Music (resonance) | Systems have resonant frequencies; driving at those frequencies amplifies output | L8 may be the "resonant frequency" of the reasoning system; L9 is anti-resonant |
| Evolutionary biology | Fitness landscapes with local optima | Capability threshold (40%) ≈ fitness valley that prevents steering from climbing the hill |
| Economics (leverage) | Small capital controlling large assets through leverage | Per-layer steering (one layer modified) controls the entire 32-layer model's output |

### Relational Findings

| Domain | Relational Analogue | Application |
|--------|-------------------|-------------|
| Surgical robotics | "Keyhole" surgery — minimal incision at precise location | Per-layer selectivity is keyhole surgery on the model's computation |
| Immunology | Helper T-cells activate/modulate immune response at specific checkpoints | Trim-tab layers are "helper" layers that modulate the reasoning computation |
| Finance (options) | Gamma hedging — small position adjustments to manage risk | Steering with α may be gamma hedging the model's reasoning trajectory |
| Psychophysics | Just-noticeable difference (JND) — minimal stimulus for detectable change | The effective α represents the JND for each layer |

### Potential Findings

| Transfer | Application |
|----------|-------------|
| Architecture trim tabs | Apply per-layer selectivity to non-MHA architectures (MoE, hybrid) |
| Evolutionary search for trim tabs | Use evolutionary algorithms to find optimal (layer, α) combinations |
| Clinical trial methodology | Design steering experiments like dose-finding clinical trials (α sweep per layer) |
| Adaptive optics (astronomy) | Deformable mirrors correct for atmospheric distortion in real-time; analogous to real-time steering correction during generation |

### Blind-Spot Alert for Lens 2
The analogical lens reveals that the trim-tab/death-layer pattern strongly resembles interference phenomena, suggesting there may be a phase relationship between layers that steering disrupts. This relational dynamic was invisible in the structural analysis.

---

## Lens 2: DIALECTICAL

**Input**: Lens 1 output + blind-spot alert (interference/phase relationships)

### Thesis
Velocity-based steering can improve reasoning accuracy by modifying hidden state trajectories at specific layers. The method is sound because (a) velocities are learnable with high accuracy (R²=0.94), (b) per-layer selectivity isolates beneficial from harmful effects, and (c) the trim-tab pattern generalizes across datasets and model families.

### Antithesis
Velocity-based steering is fundamentally limited because (a) high R² describes, not prescribes — faithful error reproduction has no corrective value, (b) steering requires capability that the model already possesses, making it amplification rather than creation (limited ceiling), and (c) the death-layer phenomenon shows that steering often disrupts rather than improves — you're injecting noise into a delicate computation.

### Synthesis
Velocity-based steering is a *modulation technique* — it works when the underlying computation is already near-correct and the steering direction aligns with the natural computation of the target layer. The trim-tab/death-layer dichotomy is not arbitrary; it reflects the *alignment between the steering vector and the layer's natural computation direction*. L8 works because its computation is parallel to the corrective direction; L9 fails because its computation is orthogonal or antagonistic. The correct framing is: **steering amplifies, but it can also selectively amplify only the correct components** when applied at the right layer. This reframes the "amplification vs creation" debate into a "alignment vs interference" one.

### Blind-Spot Alert for Lens 3
The dialectical synthesis reveals a hidden third position: perhaps the death layers are not inherently harmful, but become harmful because the steering vector at α=0.1 overpowers their computation. An α sweep might find a regime where L9 is neutral or even beneficial.

---

## Lens 3: BLENDING

**Input**: Lens 2 output + blind-spot alert (α-dependent death layers)

### Structural Blend Candidates

| Blend | Atoms | Resulting Composite | Novelty (1-5) |
|-------|-------|--------------------|----------------|
| B1 | A2 (TT) + A8 (Contrastive) | Single model trained with Siamese contrastive loss predicting direction toward correctness | 4 |
| B2 | A10 (α) + A9 (GSM8K) | α learned per problem by reinforcement learning on GSM8K outcomes | 5 |
| B3 | A5 (Trim-Tab) + A6 (Death Layer) | Adaptive layer routing: use trim-tabs when available, bypass death layers | 3 |
| B4 | A12 (Cross-Model) + A1 (Velocity) | Universal velocity representation learned across all models via multi-task training | 5 |
| B5 | A20 (PPL) + A8 (Contrastive) | Gate contrastive steering by uncertainty, but use contrastive signal as the uncertainty estimate itself | 4 |
| B6 | A3 (KV-Cache) + A19 (Logit) | Dual-surface steering: modify both K/V and logits with coordinated signals | 3 |
| B7 | A4 (Per-Layer) + A10 (α) | Learn a vector α per layer via Bayesian optimization | 4 |

### Junction Blends

B8: Blend J1 (Causal: Velocity→TT) with J10 (Causal: Contrastive→Trim-Tab) → Train a single model that learns both velocity prediction AND contrastive direction simultaneously via multi-task learning.

### Blind-Spot Alert for Lens 4
The blending lens reveals a combinatorial explosion of steering architectures. The key question is not "which single mechanism works best" but "how do steering mechanisms interact when combined." The project has only tested mechanisms in isolation.

---

## Lens 4: SYSTEMS

**Input**: Lens 3 output + blind-spot alert (interaction effects)

### Feedback Loops

| Loop | Type | Structure | Polarity |
|------|------|-----------|----------|
| FL1 | Reinforcing | L8 steering → better answer → more correct trajectories in training data → better TT → better L8 steering | + |
| FL2 | Balancing | L9 steering → worse answer → increased token divergence → incoherent output → model collapse | − |
| FL3 | Reinforcing | High confidence on wrong answer → no steering applied (PPL gate) → model never corrected → continued wrong answers | + (harmful) |
| FL4 | Balancing | Better TT → more steering → greater divergence → higher risk of off-manifold degradation | − |
| FL5 | Delayed | Capability threshold creates a dead zone: models below threshold can't generate correct data → no correct trajectories → no contrastive signal → can't escape threshold | − (stable) |

### Causal Structure

```
[Trajectory Collection] → [TT Training] → [Steering Direction]
                                               ↓
[Baseline Generation] → [Hidden State Velocity] ←→ [α Scaling]
                                               ↓
                                        [KV Cache Modification]
                                               ↓
                                        [Modified Generation]
                                               ↓
                                     [Accuracy Measurement] → [TT Update] (FL1)
                                               ↓
                                     [Token Divergence] → [Degradation Check] (FL2/FL4)
```

### Leverage Points

1. **Quality of trajectory data** (most leverage): If trajectories capture diverse reasoning strategies, TT learns richer dynamics. If data is homogeneous, TT overfits to common patterns.
2. **Layer selection**: Single tunable parameter (which layer) with binary outcome (±20pp swing). Highest leverage-per-decision in the system.
3. **α parameter**: Under-explored; could be a continuous dial from beneficial to harmful.

### Blind-Spot Alert for Lens 5
The systems lens identifies a critical hidden dynamic: the capability threshold (Finding 3) creates a *self-reinforcing poverty trap* — models below 40% can't generate correct trajectories, which means no high-quality training data for contrastive TT, which means no escape. This is a structural property, not just a property of the model.

---

## Lens 5: ABDUCTIVE

**Input**: Lens 4 output + blind-spot alert (poverty trap)

### Structural Abduction

**Observation**: L8 improves accuracy (+20pp) while L9 destroys it (-23pp).

**Best explanation**: The model's residual stream computes a progressive reasoning chain. L8 produces intermediate representations that benefit from alignment toward "correct" direction — it's a layer whose output direction is broadly informative about correctness. L9 produces representations that are *structurally organizing* — they position information for later layers. Pushing L9 disrupts the organization, causing cascading failure.

**Alternative**: The steering vector at L8 happens to align with the natural direction of the residual stream (low angle between v_pred and v_natural), while at L9 it's anti-aligned. The effect is purely geometric, not functional.

### Relational Abduction

**Observation**: Qwen2.5-Math-1.5B has 38% baseline but no trim tabs.

**Best explanation**: Math-1.5B is a *base model*, not instruct-tuned. Instruct tuning creates a "reasoning manifold" that standardizes how the model processes reasoning tasks. Without this standardized manifold, the velocity field is chaotic and layer-specific patterns don't emerge. The trim-tab effect depends on the model having learned a consistent reasoning strategy during instruction tuning.

**Alternative**: The 1.5B hidden state has intrinsically lower dimensionality than 7B, so the velocity manifold is more constrained and any steering pushes states off-manifold.

### Potential Abduction

**Observation**: Cross-model transfer (SmolLM2→7B) preserves L8 as best trim-tab.

**Best explanation**: Velocity dynamics are determined by the *attention architecture* rather than the model size. Since both use standard MHA (with different head counts), they share a common velocity structure. Under this explanation, the TT learned by SmolLM2 captures MHA-native velocity dynamics.

**Unconsidered explanation**: The projection layer (960→3584) acts as a strong regularizer that happens to preserve the dominant eigenvector of the velocity field, which is the trim-tab direction. A random projection might preserve any sufficiently dominant pattern.

### Blind-Spot Alert for Lens 6
The abductive lens suggests that the trim-tab pattern may be an artifact of instruct-tuning's standardized reasoning manifold — not a universal property of LLMs. This has not been tested.

---

## Lens 6: TRAJECTORY

**Input**: Lens 5 output + blind-spot alert (instruct-tuning dependency)

### Structural Evolution

| Timeline | Development | Key Insight |
|----------|------------|-------------|
| Session 1 (Jun 10) | Define problem, build collection pipeline | Velocity prediction concept born |
| Session 2 | Train TT, initial steering attempts | All-layers steering fails |
| Session 3 | Discover per-layer selectivity, L8 trim tab | Breakthrough: layer matters more than α |
| Session 4 | Cross-model transfer, SVAMP replication | Pattern generalizes |
| Session 5 | Contrastive TT training, infrastructure optimization | Next direction: normative prediction |
| Projected (S6+) | Contrastive evaluation, multi-layer combinations | Production-ready steering? |

### Trajectory Extrapolation

| Variable | Direction | Projected State (S10, ~2 weeks) |
|----------|-----------|--------------------------------|
| Steering Accuracy | ↑ | Contrastive TT should outperform standard TT if hypothesis holds |
| Understanding of mechanism | ↑ | Mechanistic analysis of L8/L9 functional roles |
| Number of viable mechanisms | ↑ | Multi-layer, combined-surface, learned α |
| Infrastructure maturity | ↑ | Streaming collection, zero-copy loading |
| Theoretical framework | ↑ | From "steering works" to "steering works because..." |

**Divergent Scenario**: Contrastive TT fails or produces no trim tabs. This would force a fundamental rethinking of the approach, possibly shifting to (a) residual stream steering, (b) MLP intervention, or (c) attention redistribution.

**Convergent Scenario**: Contrastive TT produces stronger trim tabs (+30pp?), multi-layer combination shows synergy, and the method works on ARC/BBH/MMLU. The project converges on a deployable steering technique.

### Blind-Spot Alert for Lens 7
The trajectory lens shows the project is at a "pre-paradigmatic" stage — it has strong empirical results but no mechanistic theory for WHY steering works. This theory gap is the biggest risk for future progress.

---

## Lens 7: METACOGNITIVE

**Input**: Lens 6 output + blind-spot alert (theory gap)

### Structural Blind Spots

| Blind Spot | Description | Why Missed |
|------------|-------------|------------|
| BS1 | What is the functional role of each layer? | No mechanistic interpretability analysis done; layers are identified by layer number only |
| BS2 | What happens to non-steered layers when one layer is modified? | Only final accuracy measured, not intermediate state changes |
| BS3 | How does the steering signal propagate through the residual stream? | Assumed propagation but not measured |
| BS4 | What is the distribution of steering directions across tokens? | Single α per layer; per-token variation averaged out |

### Relational Blind Spots

| Blind Spot | Description | Why Missed |
|------------|-------------|------------|
| BS5 | How do trim-tab and death-layer effects interact? | Only single-layer experiments; no 2x2 design (L8+L9 together) |
| BS6 | Is the capability threshold continuous or discrete? | Only 5 data points (models); insufficient resolution |
| BS7 | Does the contrastive signal decorrelate with accuracy as generation progresses? | Only measured final R², not per-token direction quality |

### Potential Blind Spots

| Blind Spot | Description | Why Missed |
|------------|-------------|------------|
| BS8 | Could steering work on sub-threshold models with a different α? | Default α=0.1 used everywhere; α not tuned per model |
| BS9 | Is there a "universal death layer" pattern across all models? | Only tested L9 on one model extensively |
| BS10 | Does steering create latent inconsistencies that persist after generation? | No post-generation analysis of model state |

### Blind-Spot Alert for Lens 8
The metacognitive lens identifies 10 significant blind spots. The most critical is BS1 (functional role of layers) — without understanding what each layer computes, trim-tab/death-layer discovery remains correlational rather than causal.

---

## Lens 8: INSPIRATION

**Input**: Lens 7 output + blind-spot alert (layer function ignorance)

### Foreign-Domain Structural Inspirations

| Domain | Structure | Inspiration for RankAdaptation |
|--------|-----------|-------------------------------|
| C. elegans connectome | Complete neural wiring diagram; each neuron has known function | Build a "reasoning connectome" for the LLM — map every layer's function |
| VLSI design | Critical path analysis identifies timing bottlenecks | Identify the "critical path" through layers for reasoning; steer only critical layers |
| Quantum error correction | Redundant encoding protects against bit flips | Steering may be analogous to error correction: adding a corrective signal to survive through layers |
| Immune system memory | B-cells remember past pathogens for faster response | Cache steering vectors per task type for rapid adaptation |
| Agile software development | Iterative sprint cycles with retrospective | Apply the trim-tab discovery as a "sprint": find one layer, optimize, evaluate, repeat |

### Foreign-Domain Dynamic Inspirations

| Domain | Dynamic | Application |
|--------|---------|-------------|
| Climbing (rock) | Find holds (trim-tabs), avoid loose rocks (death layers), use optimal sequence | Reasoning as climbing: L8 is a jug hold, L9 is loose rock, optimal sequence is multi-layer routing |
| Chess (opening theory) | Known opening sequences with accepted trade-offs | Develop "steering theory" — catalog of known (layer, α) combinations and their effects |
| Music (counterpoint) | Multiple independent melodic lines that harmonize | Multi-layer steering as counterpoint: each layer contributes an independent voice that harmonizes |

### Blind-Spot Alert for Lens 9
The inspiration lens suggests that steering could be made *adaptive* — the optimal (layer, α) changes depending on the token position and problem difficulty. The project has only tested static steering.

---

## Lens 9: ADVERSARIAL

**Input**: Lens 8 output + blind-spot alert (static vs adaptive steering)

### Structural Attack: Cheapest Path to Failure

**Attack**: Apply steering at "known bad" layers (L9, L15+) with large α. This is already documented. The real attack is **multi-layer ensemble steering** — applying small perturbations at ALL layers simultaneously, each within "safe" bounds, but cumulatively destroying output.

**Cost**: Negligible. Requires only changing the layer selection in existing scripts.

**Defense**: Monitor total perturbation norm across all layers. Implement a safety budget.

### Relational Attack: Break the Critical Relationship

**Critical relationship**: The steering-signal-to-natural-computation alignment (J4: Selectivity→L8). If this relationship is disrupted (e.g., by using a noisy TT, or by applying steering at incorrect timing), the entire system fails.

**Attack**: Slightly perturb the TT output with a learned adversarial perturbation designed to minimize accuracy. This is essentially the "anti-steering" attack.

**Defense**: Adversarial training of TT against a red-team attacker.

### Potential Misapplication

**Misuse**: Malicious steering to produce harmful content. If steering can amplify existing capability (Finding 3), it can amplify harmful capabilities in models that already know how to produce harmful content.

**Severity**: HIGH. A +20pp steering effect on harmful generation would be dangerous.

**Mitigation**: Maintain exclusive focus on reasoning tasks; monitor steering applications; do not release TT weights for unchecked models.

### Blind-Spot Alert for Lens 10
The adversarial lens reveals that the steering technique is symmetric: the same mechanism that amplifies correct reasoning can amplify harmful reasoning. No safety guardrails are documented.

---

## Lens 10: PARADOXICAL

**Input**: Lens 9 output + blind-spot alert (safety symmetry)

### Structural Paradox

**Paradox**: The same steering mechanism that produces a +20pp improvement at L8 produces a -23pp degradation at L9. The mechanism is identical; only the layer differs. The behavior inverts without any change to the method.

**Gödel Sentence**: "This layer's steering effect inverts the system's behavior." Applied to L9: steering L9 turns the model from "sometimes correct" to "always wrong." The steering operation at this layer is a fixed point — applying it reliably produces the inverse of its intended effect.

### Relational Paradox

**Paradox**: The project depends on baseline capability (Finding 3) to produce steering improvements, but the steering improvements themselves alter the baseline, invalidating the very condition they depend on. You need a model that can already answer correctly to steer it toward more correct answers — but if it could already answer correctly, why steer?

**Resolution**: Steering helps on the *margin* — the model answers some fraction correctly (73%); steering pushes the borderline cases over the threshold. The paradox is resolved by recognizing that steering doesn't move ALL answers, only the near-correct ones.

### Potential Paradox

**Paradox**: If contrastive TT works (v_correct − v_incorrect), then in principle it encodes the *difference* between correct and incorrect reasoning trajectories. But if this difference exists as a learnable signal, then the model already "knows" the difference implicitly. Steering is making explicit what was already implicit — which means the model could theoretically correct itself without external steering.

**Inversion**: The limit case of self-steering is... just letting the model think longer (chain-of-thought, self-consistency). This suggests that steering is a *compressed* form of the computation that the model would do given more tokens. At the limit, steering merges with self-correction techniques.

### Convergent Check: Lens Agreement Matrix

| Finding | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | Agreement | Confidence |
|---------|----|----|----|----|----|----|----|----|----|-----|-----------|------------|
| Trim-tab pattern real | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | 10/10 | HIGH |
| Steering amplifies, doesn't create | Y | Y | — | Y | Y | Y | Y | Y | Y | Y | 9/10 | HIGH |
| Layer selectivity essential | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | 10/10 | HIGH |
| Death layers are destructive | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | 10/10 | HIGH |
| Instruct tuning shapes velocity manifold | — | — | — | — | Y | Y | — | — | — | — | 2/10 | LOW |
| α is under-explored parameter | Y | Y | Y | Y | — | Y | Y | Y | Y | Y | 9/10 | HIGH |
| No mechanistic theory yet | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | 10/10 | HIGH |
| Interaction effects unknown | — | Y | Y | Y | — | — | Y | Y | Y | Y | 7/10 | MED |
| Steering is symmetric (good/harm) | — | — | — | — | — | — | — | — | Y | Y | 2/10 | LOW |
| Multi-model universal velocity | — | — | Y | — | Y | — | — | Y | — | — | 3/10 | LOW |

### Persistent Blind Spots

| Blind Spot | Why Unaddressed | Severity |
|------------|-----------------|----------|
| BS1: Functional role of each layer | Requires mechanistic interpretability beyond project scope | HIGH |
| BS3: Steering signal propagation | Requires intermediate state capture during steering | HIGH |
| BS5: Layer interaction effects | Requires combinatorial experiments (2^k layers) | MED |
| BS8: α per model | Simple sweep not yet run | LOW |
| BS10: Post-steering state | Requires additional analysis infrastructure | MED |
