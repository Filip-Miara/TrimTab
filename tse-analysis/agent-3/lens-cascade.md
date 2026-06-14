# Phase 2: Multi-Lens Analysis Cascade

**Subject**: Velocity-based latent steering for language model reasoning
**Lens Count**: 10
**Date**: 2026-06-14

---

## Lens 1: ANALOGICAL

**Structural**: What analogous structures exist in other domains?

- **Control Theory**: The steering mechanism (vel_pred × α → KV modify) is structurally identical to a proportional controller (P-controller). The hidden state is the plant state, the TT is the forward model, α is the proportional gain, and the KV cache is the actuator. No integral or derivative term exists (no PID).
- **Neuroscience**: The trim-tab/death layer pattern mirrors cortical microstimulation — certain brain regions (e.g., MT for motion perception) respond strongly to microstimulation while neighboring regions produce noise or seizure (death layer analog). L8 is the "MT" of this model.
- **Cellular Biology**: The distinction between trim-tab and death layers resembles the difference between regulatory transcription factors (which amplify existing expression) and general transcription factors (which cause catastrophic over-expression).
- **Aerospace Engineering**: The term "trim tab" is borrowed from aviation — a small surface that adjusts a larger control surface. The analog holds: L8 is a small adjustment that changes the model's output trajectory.
- **Economics**: The capability threshold (40%) resembles a "development trap" — below a certain GDP threshold, development aid (steering) cannot lift an economy out of poverty. Above the threshold, targeted interventions amplify growth.

**Relational**: What analogous interactions exist?

- The TT→Steering→Accuracy causal chain mirrors: sensor→controller→plant in control systems
- The contrastive signal (v_correct − v_incorrect) is analogous to error-driven learning in the brain (prediction error in dopamine neurons)
- Cross-model transfer resembles transfer learning in biology: antibodies trained on one pathogen often generalize to related ones
- All-layers steering being net negative is analogous to "agonist-antagonist" muscle pairs — simultaneously activating all muscles produces less force than selective activation

**Potential**: What analogous solutions could be transplanted?

- **PID Control**: Add integral and derivative terms to the steering law: α·v + β·∫v·dt + γ·dv/dt could capture accumulated error and curvature
- **Adaptive Gain Scheduling**: α should vary by layer and token position — analogous to how autopilot systems adjust gain at different flight phases
- **Model Predictive Control (MPC)**: Instead of single-step velocity prediction, roll out a multi-step trajectory and optimize the steering sequence as a planning problem
- **Hebbian Steering**: "Neurons that fire together wire together" — steer layers in proportion to their co-activation with correct output tokens
- **Resonance Steering**: Apply oscillatory α (sinusoidal modulation over steps) to exploit resonance frequencies in the hidden state dynamics

**Blind Spot Alert**: Analogies may be superficial. Control theory assumes linearity and known plant dynamics — neither holds for neural network hidden states.

---

## Lens 2: DIALECTICAL

**Thesis**: Velocity-based KV-cache steering can improve reasoning accuracy in language models by pushing hidden states toward the correct trajectory.

**Antithesis**: Steering cannot create capability — it only amplifies existing latent reasoning. The trim-tab/death layer pattern is a fundamental limitation: the model's internal architecture dictates that some layers will always be adversarial to steering, and the mechanism fails entirely below a capability threshold. The high R² of the TT reflects prediction of token-fluency, not reasoning-quality.

**Synthesis**: Steering is a "cognitive amplifier" — it works when and where the model already knows the right direction. The trim-tab layers are the model's existing reasoning circuits; death layers are bottlenecks or information-theoretic chokepoints where the residual stream is maximally compressed. The task is to (1) identify trim-tab layers for each model/task, (2) understand WHY specific layers are trim-tabs (mechanistic interpretability), and (3) build steering mechanisms that target the trim-tab circuit without affecting death layers.

**Key Tension**: The central tension is between *descriptive accuracy* (TT has high R²) and *normative utility* (steering improves accuracy). The bridge from descriptive to normative is the contrastive signal — but it may be that the descriptive/normative gap is inherent and cannot be fully bridged with any linear steering method.

---

## Lens 3: BLENDING

**Which atoms can be blended?**

| Blend | Constituents | Resulting Composite | Quality |
|-------|-------------|-------------------|---------|
| B1 | A4 (TT) × A8 (Accuracy) | **Accuracy-Gated Steering**: Only steer when TT predicts a trajectory that would lead to higher accuracy (requires a learned accuracy predictor) | Novelty: 4/5, Risk: 3/5 |
| B2 | A5 (α) × A20 (Dimensionality) | **Per-Dimension α**: Different steering magnitudes for different dimensions of the hidden state | Novelty: 4/5, Risk: 3/5 |
| B3 | A6 (Trim-Tab) × A7 (Death) | **Inverse Death Steering**: Apply negative α to death layers (push AWAY from their predicted velocity) | Novelty: 5/5, Risk: 4/5 |
| B4 | A9 (Contrastive) × A13 (Token Divergence) | **Contrastive Token Masking**: Only steer tokens where contrastive signal is largest | Novelty: 4/5, Risk: 2/5 |
| B5 | A19 (PPL) × A2 (Velocity) | **PPL-Weighted Velocity**: Scale predicted velocity by the model's confidence (perplexity) at each step | Novelty: 3/5, Risk: 2/5 |
| B6 | A10 (Capability) × A16 (DeltaNet) | **Recurrent-State Steering**: Apply TT to the recurrent state of GatedDeltaNet instead of KV cache | Novelty: 5/5, Risk: 4/5 |
| B7 | A14 (Template) × A8 (Accuracy) | **Template Optimization via Steering**: Search for prompt templates that maximize L8 steering effect | Novelty: 3/5, Risk: 2/5 |

**Blind Spot Alert**: Blending assumes compatibility of underlying representations — hidden states from different layers may not blend linearly.

---

## Lens 4: SYSTEMS

**Feedback Loops**:

- **Reinforcing Loop R1** (Virtuous): Correct steering → higher accuracy → more correct trajectories collected → better TT training → better steering
- **Reinforcing Loop R2** (Vicious): Incorrect steering → lower accuracy → more incorrect trajectories → TT learns incorrect patterns → worse steering
- **Balancing Loop B1**: Steering → token divergence → out-of-distribution hidden states → model becomes uncertain → PPL increases → steering is gated off
- **Reinforcing Loop R3** (Selection): Trim-tab layer identified → more steering at that layer → layer gets stronger signal → becomes even more of a trim-tab (contrastive learning for layers)

**Delays**:

- Delay between steering application and accuracy measurement: ~200 tokens of generation (substantial — makes real-time adaptation hard)
- Delay between trajectory collection and TT training: hours (batch training)
- Delay between TT update and steering improvement: one full eval cycle

**Side Effects**:

- Steering at trim-tab layers may impair other capabilities (catastrophic forgetting of non-math skills) — NOT tested
- Token divergence (88%) means the model produces different outputs for ALL tokens, not just reasoning-critical ones — may introduce new errors
- Contrastive TT may learn to predict the difference in fluency (correct answers are longer/more confident) rather than reasoning quality
- Cross-model transfer may fail silently if the projection layer is poorly conditioned

**Leverage Points**:

- **L8 trim-tab** is the highest-leverage intervention point (leverage score: 9/10)
- **Contrastive TT training** converts descriptive→normative (leverage: 8/10, but unproven)
- **Prompt template quality** (leverage: 7/10, already exploited)
- **α per token position** (leverage: 6/10, unexplored)

**Blind Spot Alert**: System dynamics assume the LM is a stationary system — model weights are fixed. But hidden state dynamics may shift during long generations as the model enters different "reasoning phases."

---

## Lens 5: ABDUCTIVE

**What structure best explains the observed failures?**

**Failure: Small model steering all harmful**
- Best explanation: Capability threshold hypothesis (A10) — models below ~40% baseline lack the latent reasoning manifold
- Alternative 1: Small models have lower-dimensional hidden states, so ANY steering causes more relative perturbation → OOD
- Alternative 2: Small models' KV caches capture less task-relevant structure — steering perturbs noise, not signal
- Alternative 3: The optimal α for small models is much smaller (0.01 vs 0.1) — we didn't search the hyperparameter space

**Failure: All-layers steering is net negative**
- Best explanation: Death layers dominate — the net effect of compounding noise from 2-3 death layers outweighs trim-tab benefits
- Alternative 1: All-layers steering creates a "global distraction" effect — modifying all layers simultaneously confuses the residual stream
- Alternative 2: The attention pattern across layers is such that death layers "read from" trim-tab layers and corrupt their output

**Failure: Logit correction failed**
- Best explanation: Distribution shift — TT trained on generation hidden states applied to logits (different representation space)
- Alternative: Logits are more sensitive to perturbation — even small corrections push the softmax into saturated regions

**Failure: PPL modulation gated <0.1% of tokens**
- Best explanation: The model is confidently wrong — confidence (PPL) doesn't correlate with correctness
- Alternative: The PPL threshold was set too aggressively — a larger threshold would gate more tokens but with weaker steering signal

**Blind Spot Alert**: Abductive reasoning may converge on explanations consistent with the current theoretical framework, missing alternative explanations that require paradigm shifts.

---

## Lens 6: TRAJECTORY

**How has the structure evolved over time?**

- **Session 1** (v0.21): Initial hypothesis — hidden states during generation might be predictable. Prototype TT on SmolLM2.
- **Session 2** (v0.22-0.24): Confirmed TT works (R²=0.94). First steering attempts — logit correction, then KV-cache. All-layers steering fails.
- **Session 3** (v0.25-0.30): Per-layer sweep on Qwen2.5-7B. Discovered L8 (+20pp) and L9 (-23pp). Cross-dataset validation (SVAMP). Cross-model transfer.
- **Session 4** (v0.31-0.35): Contrastive TT training. Failed attempts on Qwen3.5, Math-1.5B. Infrastructure optimization (async loading, GPU cache).
- **Session 5** (v0.36-0.38): Contrastive evaluation pending. Multi-head contrastive exploration. Documentation.

**Extrapolation**:
- Short-term (next session): Contrastive TT evaluation on Qwen2.5-7B — if positive, major validation of the framework
- Medium-term: Asymmetric α sweeps, multi-layer combination, head-level steering
- Long-term: Real-time adaptive α, cross-task (non-math) validation, publication

**Trajectory Arc**: The project has moved from exploratory (does steering work?) to confirmatory (when/why does it work?) to engineering (how to make it practical and reliable).

**Blind Spot Alert**: The trajectory extrapolation assumes linear progress — it may be that contrastive TT evaluation shows NO improvement, which would require rethinking the entire normative direction.

---

## Lens 7: METACOGNITIVE

**Structural blind spots**:
1. **Missing causal graph of LM internals**: The analysis treats layers as black boxes — we know L8 helps and L9 hurts, but don't know what computations these layers perform. Without mechanistic interpretability, we cannot predict trim-tab layers for new models.
2. **No characterization of the hidden state manifold**: Is the velocity prediction operating on a linear subspace? Is manifold curvature significant? We measure R² but not the geometric properties of the state space.
3. **No control experiments**: The +20pp improvement on L8 has no "sham steering" baseline (apply random vectors of same magnitude) — we don't know if the improvement is specific to TT predictions or any perturbation of that magnitude.

**Relational blind spots**:
1. **No interaction analysis between layers**: Steering L8 may change how L9 behaves — but we only measure independent layer sweeps. Layer interactions may be significant.
2. **No sequence-position analysis**: Does L8 steering matter more at the beginning, middle, or end of generation? Token-position effects are unknown.
3. **No null model for contrastive signal**: We haven't tested whether v_correct − v_incorrect is better than random vectors with the same statistical properties (mean, variance, autocorrelation).

**Potential blind spots**:
1. **What if steering works in the opposite direction**: Negative α on death layers might be MORE beneficial than positive α on trim-tabs — but we haven't systematically tested negative α.
2. **What if the contrastive signal is additive**: Standard TT + β·contrastive might outperform either alone.
3. **What if task-specific steering is required**: The trim-tab pattern for math might differ from the pattern for reasoning tasks — we've only tested math.

**Blind Spot Alert**: Metacognition is inherently limited by self-knowledge — the most important blind spots are those we cannot identify.

---

## Lens 8: INSPIRATION (Foreign Domain)

**Foreign Domain: Evolutionary Biology**

**Structural Map**: The trim-tab/death layer pattern maps to genotype→phenotype mapping. Some genes (trim-tabs) are "hub genes" that regulate many downstream processes with consistent effect. Other genes (death layers) are "lethal knockouts" where any perturbation causes catastrophic failure.

**Dynamic**: Natural selection (evaluation framework) automatically identifies trim-tab genes — those that improve fitness when upregulated. This suggests: **Can we automate the trim-tab discovery process using evolutionary strategies?** Instead of sweeping layers manually, use a genetic algorithm to find the optimal (layer, α) combination.

**Transplanted Solution**: **Evolutionary Layer Optimization** — represent a steering policy as a chromosome: array of (α_0, α_1, ..., α_N) for N layers. Use a population of 50 such policies, evaluate each on 20 problems (cheap), select top 10%, mutate/crossover, repeat for 20 generations. This automatically discovers (1) which layers to steer, (2) with what α, (3) possibly multi-layer combinations.

**Foreign Domain: Classical Music Orchestration**

**Structural Map**: Trim-tab layers are the first violins (carry the melody, small adjustment creates large effect). Death layers are the percussion section during a quiet passage (any addition destroys the mood). The steering coefficient α is the volume knob.

**Dynamic**: An orchestrator doesn't adjust all instruments equally — they have a deep understanding of each instrument's role and the score's structure. The contrastive signal is like having a recording of the "correct" performance — subtract the incorrect performance to find what the first violins should change.

**Transplanted Solution**: **Orchestration Analog** — composite steering where different layers get different α at different generation phases (exposition, development, recapitulation = problem reading, reasoning, answer generation). Layer roles change during generation.

**Blind Spot Alert**: Foreign domain analogies may be inspiring but should be validated, not assumed correct.

---

## Lens 9: ADVERSARIAL

**Cheapest structural attack**:
- **Attack 1**: Steer L9 with a very large α (even moderately positive α at L9 costs -23pp). An adversary who can modify the steering direction at L9 can completely destroy model performance.
- **Attack 2**: Poison the trajectory data with subtly incorrect trajectories — the TT learns to predict wrong velocities, making even L8 steering harmful.
- **Attack 3**: Use the 88% token divergence to insert adversarial content — since steering changes nearly all tokens, a malicious α at multiple layers could generate specific incorrect tokens.

**What relationship, if broken, collapses the system?**
- **J3** (Steering Operator → KV Cache): If the KV cache modification interface breaks (e.g., using a model variant without standard K/V cache access), the entire steering mechanism fails.
- **J4** (Capability Threshold → Per-Layer Steering): If the capability threshold is real and universal, then steering cannot improve models that need it most — it's a binary gate that limits the approach to already-capable models.
- **J10** (Contrastive Signal → Contrastive Steering): If the contrastive signal doesn't capture the correct→incorrect direction meaningfully, the contrastive system is no better than standard TT (which is descriptive, not normative).

**What misapplication of potential causes harm?**
- **Harm 1**: Applying steering to a model without first identifying trim-tab layers can destroy performance (all-layers steering is net negative — the default approach is harmful).
- **Harm 2**: Over-reliance on GSM8K as proxy — improvements on math may not generalize and may even reduce non-math performance (catastrophic forgetting of other capabilities).
- **Harm 3**: Using TT-predicted velocities as a "ground truth" signal for model editing — if R² is high but predictions are wrong on critical tokens, model editing based on these predictions could introduce systematic errors.

**Blind Spot Alert**: Adversarial analysis assumes a malicious actor — the more likely scenario is accidental misuse (e.g., thinking all-layers steering is safe).

---

## Lens 10: PARADOXICAL

**Structural self-reference creating paradox**:
- **The Steering Paradox**: Steering modifies the model's hidden states based on TT predictions. But the TT was trained on the UNMODIFIED model's trajectories. If steering changes the trajectory, the TT's predictions become less accurate for the steered model. The system is solving a moving-target problem — the dynamics it's steering by are not the dynamics it's steering.
- **The Descriptive-Normative Paradox**: To train a normative TT (contrastive), we need correct and incorrect trajectories. But correct trajectories depend on the model being steered correctly — which is what we're trying to achieve. This creates a chicken-and-egg: we need good steering to get good training data, but we need good training data to get good steering.

**Relational Gödel sentence**:
- "This steering mechanism can only improve models that don't need improvement." — If the capability threshold holds, the models that benefit most from steering (weak reasoners) cannot be steered, while models that already reason well (73%) get marginal improvement. The system is limited to making good models slightly better.

**What limit, when pushed, inverts the system?**
- **α beyond critical threshold**: Pushing α too high at ANY layer (even L8) likely degrades performance. The trim-tab effect is non-monotonic — there's an optimal α that depends on the layer and the model. Push harder → worse results.
- **Capability threshold inversion**: If we could steer SMALL models successfully, the finding would invert the capability threshold assumption. This would transform the approach from "amplification for capable models" to "capability creation for weak models."
- **Contrastive inversion**: If contrastive TT works WORSE than standard TT, our theoretical understanding would be inverted — the descriptive signal (faithful reproduction) is more useful than the normative signal (correction direction). This would mean velocity dynamics are intrinsically tied to fluency/surface features, not reasoning quality.

**Convergent Check**:

**High-Confidence Findings (≥5 lenses agree)**:
1. **L8 is a genuine trim-tab layer** (L1-L7, L9, L10: 8 lenses agree) — HIGH confidence
2. **Death layers exist and dominate all-layers steering** (L1, L2, L4, L5, L9: 5 lenses agree) — HIGH confidence
3. **Capability threshold limits steering applicability** (L1, L2, L4, L5: 4 lenses agree) — MODERATE confidence
4. **Contrastive signal is the most promising direction** (L2, L4, L5, L9, L10: 5 lenses agree) — HIGH confidence but evaluation-pending

**Contested Findings (≥3 lenses disagree)**:
1. **Mechanism of steering improvement**: L2 (amplification of latent reasoning) vs L1 (surface/fluency artifact) vs L9 (adversarial randomness) — 3 interpretations disagree
2. **Capability threshold explanation**: L5 (data quality/α search artifact) vs L2 (fundamental limitation) — unresolved

**Persistent Blind Spots**:
1. **No mechanistic understanding of WHY L8 is a trim-tab**: This persisted through all 10 lenses. We consistently identify the pattern but cannot explain the mechanism.
2. **No characterization of hidden state manifold geometry**: All lenses treat velocities as Euclidean vectors in a flat space; manifold curvature is ignored.
3. **No null-model validation**: None of the lenses proposed or executed a proper control experiment (e.g., random vectors vs TT predictions).
4. **No interaction effect analysis**: The layer independence assumption persists — we don't know if L8+L2 simultaneously is better or worse than either alone.
5. **No non-math task validation**: All lenses assume math generalization implies general reasoning improvement, but this is untested.
