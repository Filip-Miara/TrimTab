# Phase 2: Multi-Lens Analysis Cascade

---

## Lens 1: ANALOGICAL

**Structural findings**: The TT→steering pipeline closely mirrors **Model Predictive Control (MPC)** in control theory. MPC uses a learned dynamics model to predict future states and applies corrective control inputs. Here, TT = learned dynamics model, steering vector = control input, KV cache modification = plant actuation. The trim-tab/death-layer pattern maps to **actuator placement** in structural control — some actuators (e.g., on wing flaps) are efficient, others (on the fuselage) cause destructive vibration.

**Relational findings**: The relationship between baseline accuracy and steerability (capability threshold) maps to **linearizability** in control theory — nonlinear systems can only be steered effectively near their operating point. The correct/incorrect trajectory separation maps to **class-conditional density estimation** in statistical learning.

**Potential findings**: **Escape dynamics** from physics (particle in a potential well) — maybe incorrect trajectories are in a deeper local minimum and steering provides the energy to escape. **Epidemiology**: trim-tab layers = "superspreader" nodes in a network that amplify any signal passing through.

**Blind spot alert**: The analogy to control theory suggests checking for **stability margins** — can steering drive the model into unstable (nonsensical) regions? Not tested in the project.

---

## Lens 2: DIALECTICAL

**Thesis**: Hidden state velocities are learnable and contain structure that can steer reasoning toward correct answers.

**Antithesis**: The observed steering effect is an artifact — +20pp on L8 from a **single** alpha value (0.1) on a small test set (100 problems) with high variance. The TT may be learning auto-regressive smoothness (all adjacent hidden states are similar), not causal dynamics. The capability threshold observation may be a ceiling effect: models with >70% baseline have more headroom before saturation.

**Synthesis**: The pattern is robust across three independent axes (per-layer sweeps, cross-dataset, cross-model transfer), but the **mechanism** remains unknown. A synthesized view: velocity-based steering is a real phenomenon, layer-specific, and the L8 effect is genuine — but it operates on a **different physical mechanism than hypothesized**. The hypothesis "velocity predicts future hidden states → steering improves reasoning" may be wrong at the mechanistic level while being correct at the behavioral level. The true mechanism may involve **attention distribution shifts** rather than "pushing toward correct manifolds."

**Blind spot alert**: The dialectic reveals a missing **null hypothesis test**: does a randomly initialized TT (same architecture, no training) produce similar layer patterns? This would distinguish signal from architecture-specific noise.

---

## Lens 3: BLENDING

**Structural blend candidates**:
- **TT + LoRA adapter**: Instead of steering KV cache, directly fine-tune a low-rank adapter at trim-tab layers using velocity predictions as training signal.
- **Alpha + RL policy**: Learn α per (layer, token) via RL rather than fixed sweep.
- **Contrastive TT + ensemble**: Bootstrap multiple contrastive pairs (N correct vs N incorrect) and average their steering directions.

**Junction blend candidates**:
- **Per-layer + per-head steering**: Blend J02 (KV-steering) with attention head-level instead of layer-level granularity.
- **GSM8K + SVAMP blending**: Train a TT that predicts the "maximally general" steering direction across both datasets.

**Potential blends**:
- **Adversarial diffusion bridge**: Train a diffusion model over hidden states conditioned on velocity, then steer by diffusing from incorrect → correct trajectory.
- **Weight-flow steering**: Blend the TT's velocity predictions with the weight-flow expert (src/adapters/flow_weight_expert.py) to modulate both activations and weights simultaneously.

**Blind spot alert**: Blending assumes components are composable. The blend of TT with RL (learning α) may fail if the velocity field is not smooth enough for RL exploration to work.

---

## Lens 4: SYSTEMS

**Feedback loops detected**:
- **Reinforcing loop R1**: Better steering → higher accuracy → more data for contrastive TT → better steering.
- **Reinforcing loop R2**: High R² → confidence in TT → more steering experiments → more discoveries.
- **Balancing loop B1**: Steering pushes model toward correct → model's own auto-regressive dynamics pull it back → net effect is limited.
- **Balancing loop B2**: Death layers actively oppose trim tabs → all-layers steering is a sum of conflicting forces.

**Delays**:
- TT training to steering evaluation: ~30 min (one epoch)
- Per-layer sweep: ~4 hours (28 layers × 100 problems)
- Contrastive evaluation: still pending (contrastive results file not found)

**Leverage points**:
- **Trim-tab layer identification** is the highest-leverage control knob — selecting the right layer is worth +20pp.
- **Alpha tuning** is lower leverage — within a reasonable range (0.01-0.5), it's less impactful than layer selection.
- **Architecture choice** (MHA vs hybrid) is a design-stage leverage point.

**Side effects**:
- Steering at L8 may degrade performance on non-math tasks (not tested).
- Strong steering (α > 0.5) causes token divergence (88% at α=0.1) — generation quality degrades.

**Blind spot alert**: The system is missing a **feedback loop from steering results back to TT training**. Currently, TT is trained once and frozen. An online learning loop (TT updated based on which steering directions produce correct answers) is unexplored.

---

## Lens 5: ABDUCTIVE

**Observation**: L8 steering produces +20pp on Qwen2.5-7B.

**Best explanation candidates**:
1. **Hypothesis H_mech**: L8 is the layer where the model transitions from "understanding the problem" to "generating the answer." Steering here amplifies the transition.
2. **Hypothesis H_comp**: L8 is a "competence layer" where the model's internal computation of the answer is most developed — steering adds the right nudge.
3. **Hypothesis H_artifact**: L8 happens to be the layer where the TT's velocity predictions have lowest error (best R² for that layer), so the steering signal is cleanest.
4. **Hypothesis H_noise**: With 100 test problems at 73% baseline, a +20pp shift is within the 95% confidence interval (±9.6pp for n=100) and statistical fluctuation explains it.

**Observation**: Math-1.5B shows no trim tabs despite 38% baseline.

**Best explanation candidates**:
1. **H_nomanifold**: For this model, correct and incorrect trajectories are not separable in hidden state space — the "correct" manifold may be degenerate.
2. **H_base**: This is a base model (not instruct-tuned). Instruction tuning may be necessary for steering to work — it creates the latent capability structure.
3. **H_small**: At 1.5B, the model lacks representational capacity to support separable manifolds.
4. **H_alpha**: The alpha values tested (0.01-0.5) are wrong for this model — perhaps it needs larger or negative alpha.

**Abductive inference**: H_mech (L8 is a transition layer) is the most satisfying explanation because it (a) explains the directional effect, (b) is consistent with known transformer layer roles (early layers = encoding, middle = reasoning, late = output), (c) predicts that different models would have different trim-tab layers based on architecture. H_artifact is the strongest competing explanation and must be ruled out.

**Blind spot alert**: No mechanistic interpretation of L8 was performed (this is the open question in the debrief). The abduction is unfalsified in the current project.

---

## Lens 6: TRAJECTORY

**Historical trajectory** (5 sessions, 18 tags):
- **Session 1**: Proved velocities are learnable (R²=0.94 on SmolLM2). Explored logit correction, reading heads, PPL modulation — all failed.
- **Session 2**: KV-cache steering mechanism invented. Initial results on Qwen3.5-2B were confusing (hybrid architecture).
- **Session 3**: Discovered per-layer effect on SmolLM2 (trim tabs). Switched to Qwen2.5-7B for real results.
- **Session 4**: Confirmed L8 as trim-tab (+20pp) and L9 as death layer (−23pp) on 7B. Cross-model transfer succeeded. Chat template fix boosted baseline 4%→73%.
- **Session 5**: Contrastive TTs trained. Cross-dataset generalization (SVAMP). Infrastructure mature.

**Temporal pattern**: Failed approaches (logit correction, PPL modulation, recurrent steering) consumed ~60% of effort. The KV-cache + per-layer insight emerged as the single successful path. This is a classic **exploration funnel** — many hypotheses tested, one dominant pattern survived.

**Extrapolation**:
- Next experiments: Contrastive evaluation (pending), multi-layer combos, asymmetric α, non-math datasets.
- Near-term (1-2 weeks): L8 mechanism interpretability, α optimization via RL, multi-head contrastive ensembles.
- Long-term (1-3 months): Real-time adaptive steering, automated trim-tab discovery, steering as a general LM capability amplifier.

**Blind spot alert**: The project trajectory shows a **survivorship bias** — failed approaches are documented but not deeply analyzed. What could we learn from the logit correction failure that applies to KV-cache steering? Both share the same velocity predictions, only the actuation surface differs.

---

## Lens 7: METACOGNITIVE

**Structural blind spots**:
- **Missing baseline**: No random TT baseline (untrained TT with same init). The pattern could be an artifact of the architecture rather than learned velocity structure.
- **No ablation of α direction**: Only positive α tested (pushing in velocity direction). What about negative α (pushing against velocity)? If velocity always points toward "most likely next state," negative α would push away from model's natural trajectory — could be more informative.
- **No per-layer R² reporting**: The project reports overall R² (0.855) but not per-layer R². If L8 happens to have the highest or lowest R², this would explain the trim-tab effect trivially.

**Relational blind spots**:
- **No interaction analysis**: The relationship between layers is not modeled. Is L8-L9 interaction synergistic or antagonistic?
- **No model of "why steering works at all"**: The causal chain from "velocity prediction" to "better token selection" is not mechanistically traced.

**Potential blind spots**:
- **Unacknowledged silent failures**: Many results files (contrastive_results.json, math15_full_results.json) exist on disk but their evaluation was not completed or analyzed in the debrief. The contrastive pipeline specifically has trained TTs but no evaluation results published.
- **Alternative steering surfaces**: Only KV-cache was successful. But MLP activations, layer normalization parameters, and embedding modifications were not tested as steering surfaces.

**Blind spot alert**: The **most persistent blind spot** across all lenses is the lack of mechanistic interpretability. The project demonstrates *what* works but not *why*. This limits generalization — we don't know if L8 would be the trim tab in a different model with a different number of layers.

---

## Lens 8: INSPIRATION

**Foreign-domain structure**: **Neural ODEs** (Chen et al., 2018) — hidden state evolution across layers is a discretized ODE. The TT is learning the vector field. Steering is an external forcing term. This frame predicts that the vector field has fixed points (attractors) and steering pushes the trajectory toward a different attractor basin.

**Foreign-domain dynamic**: **Eigenvector centrality** in social networks — some nodes (layers) have disproportionate influence on the network's dynamics. Trim-tab layers are "bridge nodes" between communities (attention heads), and death layers are "peripheral nodes" where perturbations cause noise amplification.

**Foreign-domain solution**: **Adaptive cruise control** in automotive — uses MPC with a learned dynamics model of the car. The "velocity + corrective input" framework maps exactly to TT + steering. The automotive solution adds **system identification** (online estimation of model dynamics) which the project lacks.

**Mapping**:
- L8 in transformer ≈ throttle in car
- L9 in transformer ≈ brake in car
- All-layers steering ≈ pressing all pedals simultaneously → crash
- Per-layer α ≈ individual wheel torque vectoring

**Blind spot alert**: The automotive analogy suggests that **system identification** should be run online during generation — the TT should adapt to the current generation context rather than being a fixed predictor. This is unexplored.

---

## Lens 9: ADVERSARIAL

**Cheapest structural attack**: The steering pipeline is vulnerable at the **TT model checkpoints**. If an attacker gains access to best_gen_tt_7b.pt, they can (a) analyze it to find what steering directions work and craft adversarial prompts that exploit this, or (b) poison the training data so TT learns to steer toward harmful outputs.

**Critical junction**: The KV-cache modification (J02) is the single point of failure. If the KV cache is modified incorrectly (e.g., at wrong layer, wrong alpha), the model produces nonsensical output. **If attention distribution is disrupted at L9, the generation can collapse completely** (observed: 0% accuracy, effectively random output).

**Misapplication of potential**: The most dangerous scenario is successful steering toward *wrong* answers. If contrastive TT is trained on data where "incorrect" includes subtly wrong answers (plausible but wrong), it would steer the model into confident wrong answers. The model's high confidence + plausible output is more dangerous than raw model errors.

**Malicious input**: An adversarial input could be designed to make the TT predict a large velocity toward the hidden state manifold of an unsafe or toxic continuation, and the steering mechanism would amplify it.

**Blind spot alert**: The project has **no safety constraints** on α. A sufficiently large α at any layer could destroy the model's output. No guardrail against adversarial steering.

---

## Lens 10: PARADOXICAL

**Structural paradox**: The TT achieves R²=0.94 at predicting velocities, but the model's actual hidden states are deterministic given the input. If the TT is perfectly accurate (R²=1.0), then the steering vector is exactly the direction the model would naturally go. **Perfect prediction implies zero steering effect** — you're pushing the model where it would already go. The paradox: TT must be *imperfect* to be useful, but the steering improvement comes from the *error* between TT prediction and actual dynamics.

**Relational paradox**: The L8 trim-tab effect (+20pp) coexists with L9 death-layer effect (−23pp) in what is effectively a single computational unit (one transformer). How can adjacent layers have opposite effects? This is a **Gödel-sentence phenomenon** — the system's behavior cannot be fully described by layer-level analysis because the computation is distributed across layers.

**Potential paradox**: If contrastive steering works (v_correct − v_incorrect), then the optimal steering direction is the vector that maximally separates correct and incorrect trajectories. But this vector points in a direction the model *cannot naturally go* — otherwise the model would already produce correct answers. **The steering direction that would most help is the direction the model cannot follow.**

**Resolution hypothesis**: The paradoxes resolve if steering operates by **altering attention patterns**, not by "pushing hidden states." The velocity vector modifies the KV cache, which changes what future tokens attend to. The TT may predict which components of the hidden state are most "attention-sensitive" at each layer. L8 is special because the model's attention at that layer is most "malleable."

**Blind spot alert**: The paradoxical nature of the findings suggests a **fundamental incompleteness** in the layer-level analysis framework. The true mechanism may involve cross-layer interactions that the current analysis (which treats layers independently) cannot capture.

---

## Convergent Check

### High-Confidence Findings (≥5 lenses agree)

| Finding | Agreeing Lenses |
|---------|----------------|
| F1: The per-layer trim-tab/death-layer pattern is empirically robust | 1 (analogical: actuator placement), 2 (dialectical: synthesis), 4 (systems: leverage point), 5 (abductive: H_mech), 7 (metacognitive: this is the core result) |
| F2: The steering mechanism is not mechanistically understood | 1 (control theory analogy lacks system ID), 5 (abductive: competing explanations), 6 (trajectory: survivorship bias), 7 (metacognitive: persistent blind spot), 10 (paradoxical: Gödel sentence) |
| F3: Architecture matters (MHA preferred, hybrid resistant) | 1 (actuator placement), 4 (systems: design-stage leverage), 6 (trajectory: historical), 8 (inspiration: hybrid incompatible), 9 (adversarial: attack surface differs) |
| F4: Contrastive steering is the next critical experiment | 2 (dialectical: synthesis direction), 4 (systems: new feedback loop), 6 (trajectory: extrapolated next step), 7 (metacognitive: acknowledged gap), 9 (adversarial: dual-use potential) |

### Contested Findings

| Finding | Disagreeing Lenses | Reason |
|---------|-------------------|--------|
| F5: High R² indicates learnable velocity structure | Lens 10 (TT must be imperfect to be useful) vs Lenses 1,4,6 (R²=0.94 is genuine signal) | The paradox: high R² could mean TT is just predicting auto-regressive smoothness, not causal dynamics. |
| F6: Capability threshold (~40%) is a hard barrier | Lens 3 (blending: could be overcome with alternative steering surfaces) vs Lens 5 (abductive: strongest explanation for small model failures) | The threshold may be specific to KV-cache steering, not steering in general. |
| F7: Per-layer selectivity is always optimal | Lens 7 (metacognitive: interaction effects unmodeled) vs Lens 1 (analogical: actuator efficiency varies) | Multi-layer steering with per-layer α vectors might outperform single-layer; not tested. |

### Persistent Blind Spots

| # | Blind Spot | Why Unaddressed | Action Required |
|---|-----------|----------------|-----------------|
| BS1 | No mechanistic interpretation of L8 vs L9 | No interpretability tools applied to layer internals | Phase 8 analysis: probe attention patterns at L8 |
| BS2 | No random TT baseline | Assumed a learned TT is necessary | Train random-init TT and compare layer patterns |
| BS3 | No per-layer R² or per-layer error analysis | Only overall R² reported | Compute per-layer TT error and correlate with trim-tab score |
| BS4 | No safety constraints on α | Not considered necessary | Implement α clipping + sanity checks |
| BS5 | No interaction term analysis | Only single-layer and independent multi-layer sweeps | Fit regression model: accuracy = β₀ + ΣβᵢLᵢ + ΣβᵢⱼLᵢLⱼ |
