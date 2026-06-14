# Phase 2: Multi-Lens Analysis Cascade

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Lens 1: ANALOGICAL

### Structural Findings
- **Biological trim tabs**: The L8/L9 pattern is structurally isomorphic to trim tabs on aircraft — small control surfaces that adjust larger control surfaces. L8 is a "tab" that adjusts the model's reasoning trajectory; L9 is a "tab in the wrong direction" that destabilizes.
- **Gene regulation networks**: Per-layer effects mirror transcription factor specificity — some TFs activate (L8) while closely related ones repress (L9), just as adjacent layers have opposite steering effects.
- **Control theory**: The KV-cache steering is a feedforward control signal with α as the gain. All-layers steering with uniform α creates destructive interference (negative control interaction).

### Relational Findings
- **Epidemiological analogy**: The relationship between baseline accuracy and steerability mirrors herd immunity thresholds — below a critical point, the "reasoning signal" can't propagate.
- **Amplifier cascade**: L8's +20pp suggests it's a "pre-amplifier" stage — small changes there cascade through subsequent layers multiplicatively. L9 is a "clipping stage" that distorts the signal.

### Potential Findings
- **Crystallization analogy**: The TT discovering velocity dynamics is analogous to supersaturated solution crystallization — once the first correct prediction crystalizes (high R² at some layer), the structure propagates.
- **Transplantable solutions**: Cochlear implant signal processing uses similar "predictive coding" — forward prediction of neural response is used to generate compensatory signals. Same architecture: predict → compensate → improve.

### Blind Spot Alert
The analogical lens may over-fit biological analogies. Neural network dynamics are not biological neurons. Trim tabs in aircraft are mechanical and independent; transformer layers are densely coupled through the residual stream.

---

## Lens 2: DIALECTICAL

### Thesis
Velocity-based steering works because hidden state trajectories have learnable structure, and per-layer KV-cache modification can nudge the model toward better reasoning: **"Steering is amplification of latent capability."**

### Antithesis
Velocity-based steering cannot work fundamentally because (1) the TT learns to reproduce errors not correct them (descriptive not normative), (2) KV-cache modification at one layer propagates unpredictably through the residual stream, and (3) the +20pp result is a statistical artifact of small sample size (100 problems) or prompt-formatting interaction: **"Steering is a confounded illusion."**

### Synthesis
Velocity-based steering works on some layers for some models under some conditions, but the mechanism is not "steering toward correct reasoning" — it's _selective amplification of specific computational pathways_. L8 steering works because that layer's computation is most aligned with the steering direction. L9 fails because its computation is orthogonal or antagonistic. The TT is neither purely descriptive nor fully normative — it's _selective_: it preferentially amplifies certain trajectory features, and the sign (\(\pm\)α) of those features determines whether the effect helps or hurts.

### Blind Spot Alert
The dialectic assumes a single resolution. There may be multiple simultaneous truths: steering works for different reasons on different layers, and the mechanism may be fundamentally different at L8 vs L9 vs L15+.

---

## Lens 3: BLENDING

### Blendable Atoms
- **A_V (Velocity) × A_C (Contrastive)**: Blend velocity prediction with contrastive direction → "contrastive velocity" that predicts where correct trajectories go differently from incorrect ones.
- **A_α (Alpha) × A_L (Layer Selectivity)**: Blend α selection with layer selection → learned α(layer) mapping via a small network.
- **A_TT (TrajectoryTransformer) × A_M (Model Capability)**: Blend TT with capability probe → a TT that predicts _whether_ steering will help at the same time as predicting _where_ to steer.
- **A_H (Hidden State) × A_D (Token Divergence)**: Blend state representation with divergence measurement → a "steering cost" that penalizes divergence while rewarding accuracy.

### Novel Composites from Blends
- **C_BLEND1**: Contrastive TrajectoryTransformer that predicts v_correct − v_incorrect directly (instead of training two TTs and subtracting)
- **C_BLEND2**: Adaptive α(layer, token) network trained jointly with TT via meta-learning
- **C_BLEND3**: Steering gating network that predicts "steer or not" per layer per token
- **C_BLEND4**: Multi-task TT that simultaneously predicts velocity AND steering effect (a "steering advantage function")

### Blind Spot Alert
Blending assumes compatibility — contrastive and standard TT may have incompatible loss landscapes, preventing joint training. The blends may not be trainable due to optimization conflicts.

---

## Lens 4: SYSTEMS

### Feedback Loops

| Loop | Type | Description | Gain |
|------|------|-------------|------|
| FL1 | **Reinforcing** | Better accuracy → more trajectories collected → better TT → better steering → better accuracy | Medium |
| FL2 | **Balancing** | Steering changes → token distribution shifts → trajectories differ from training → TT predictions become less accurate → steering degrades | **Critical** |
| FL3 | **Reinforcing** (negative) | Stronger α → more token divergence → output distribution shifts toward randomness → accuracy drops | High |
| FL4 | **Balancing** | Model capability threshold filters out low-ability models → reduces experiment space → fewer false positives | Low |

### Delays
- **D1**: TT training delay = 30 epochs (~hours). During this delay, the model's behavior is unsteered.
- **D2**: Trajectory collection delay = time to run N problems (~minutes per 100). TT cannot be updated in real time.
- **D3**: Effect measurement delay = full evaluation cycle (~hours for 500 problems). Real-time adaptation is impossible.
- **D4**: The critical delay: between applying a steering intervention and seeing its effect on accuracy, the generation must complete. No mid-generation adaptation.

### Leverage Points (System Nudges)
1. **Reduce D2** by streaming trajectory collection and TT training simultaneously (online learning)
2. **Break FL2** by periodically fine-tuning TT on steered trajectories (distribution matching)
3. **Amplify FL1** by automating the experiment loop with a meta-controller that selects layers to explore

### Side Effects
- Steering at L8 may change L9's behavior even without direct modification at L9 (residual stream propagation)
- Per-layer steering may create "representational drift" in downstream layers, causing unpredictable effects on later tokens
- The TT itself may learn to predict the steering intervention, creating a "steering pollution" effect where future TTs predict the steered trajectory, not the natural one

### Blind Spot Alert
The systems lens assumes the system is stationary — but the model's behavior during steering is non-stationary (each token changes the state distribution). Feedback loops may interact nonlinearly.

---

## Lens 5: ABDUCTIVE

### What best explains the observed failures?

**Failure 1: All small models fail to steer (below ~40% baseline)**
- Best explanation: The correct hidden state manifold either doesn't exist (model lacks reasoning circuitry) or is too small to be found by linear perturbations
- Alternative: The TT finds velocity structure but it's the _error_ manifold — pushing toward the error manifold makes things worse
- Alternative 2: Small models' hidden states are higher-dimensional relative to their capacity (less organized structure) so the TT finds spurious patterns

**Failure 2: Math-1.5B fails despite 38% baseline (near threshold)**
- Best explanation: The model is a base model, not instruct-tuned; instruction tuning may be necessary for steering to work (it creates the "attractor" that steering can push toward)
- Alternative: 38% is still below the true threshold, which is ~50%+
- Alternative 2: The hidden state manifold of Math-1.5B is fundamentally different (dense/small model specialization)

**Failure 3: No trim tabs on contrastive Math-1.5B**
- Best explanation: Correct/incorrect trajectories are not separable for this model — error patterns and correct patterns occupy overlapping regions
- Alternative: The contrastive training is flawed (not enough data, wrong architecture)

**Failure 4: PPL-modulated correction gates at <0.1%**
- Best explanation: The model is confidently wrong — confidence (low PPL) is orthogonal to correctness for reasoning tasks
- Alternative: The correction head prediction is too noisy to trigger the gate

### Blind Spot Alert
Abductive reasoning finds the _best_ explanation but may miss the _true_ explanation, especially when multiple explanations are equally consistent with the data. The abduction may favor simpler explanations over correct ones.

---

## Lens 6: TRAJECTORY

### How the project evolved

```
v0.21 (Jun 10): Initial hypothesis — can we predict hidden state velocities?
v0.24: Prompt-trained TT (R²=0.62) → need generation data
v0.27: Gen-trained TT (R²=0.94) → velocities ARE learnable
v0.30: All-layers steering → 0% (net negative)
v0.32: Per-layer discovery → L8: +20pp! 
v0.34: Cross-model transfer → pattern generalizes
v0.36: SVAMP replicates → pattern generalizes across datasets
v0.38: Contrastive TTs trained → evaluation pending
```

### Trajectory extrapolation

6 steps forward based on current trajectory:
1. **Immediate**: Contrastive TT evaluation on Qwen2.5-7B (hours-days)
2. **Short**: Asymmetric α sweeps, multi-layer combinations (days)
3. **Medium**: Mechanistic interpretability of L8 vs L9 (weeks)
4. **Medium**: Online α learning via meta-RL (weeks)
5. **Long**: Multi-head contrastive ensemble (months)
6. **Long**: Theoretical framework for steering limits (months)

### Where the system is heading
- **If contrastive evaluation succeeds**: Paradigm shift toward multi-head contrastive ensembles with online α adaptation. Project becomes engineering-heavy (building infrastructure for real-time steering).
- **If contrastive evaluation fails**: Return to mechanistic interpretability of L8/L9. Project pivots from "make it work" to "understand why it works."

### Blind Spot Alert
Trajectory extrapolation assumes linear progress — the contrastive TT evaluation is a binary branch point that will fundamentally redirect the project. The actual trajectory may involve unexpected discoveries (new death layers, new beneficial patterns) that reshape the direction.

---

## Lens 7: METACOGNITIVE

### Structural Blind Spots
1. **The TT architecture itself is a blind spot** — we know R²=0.85-0.94 but don't know _what the TT learned_. It may be a frequency predictor, not a dynamics predictor.
2. **The steering mechanism (K/V addition) is never validated** — we don't know if the added vector actually changes attention patterns as intended.
3. **The residual stream composition is unexamined** — steering at L8 may affect L8's output but also change the residual stream for all subsequent layers.

### Relational Blind Spots
1. **Layer interaction effects are unmeasured** — we assume independence but have never tested multi-layer combinations systematically.
2. **The relationship between α and accuracy is assumed monotonic** — there may be a "sweet spot" α that varies by layer.
3. **The correlation between token divergence and accuracy is unknown** — some token changes may be critical (reasoning steps) while others are superficial.

### Potential Blind Spots
1. **Alternative steering surfaces are unexplored** — MLP activations, output embeddings, attention logits
2. **Alternative training objectives are untested** — contrastive loss, adversarial loss, reinforcement learning
3. **The possibility that steering is fundamentally limited** — theoretical upper bound from information theory, not yet quantified

### Persistent Blind Spots (unaddressed after other lenses)
1. **What does the TT actually represent?** — No lens provided mechanistic insight into TT internals
2. **Are layer effects truly independent?** — Multiple lenses flagged this but none resolved it
3. **What is the true capability threshold?** — The 40% threshold is descriptive, not explanatory

---

## Lens 8: INSPIRATION

### Foreign-Domain Structural Maps

| Domain | Structure | Mapping to Velocity Steering |
|--------|-----------|------------------------------|
| **Neuroscience (motor cortex)** | The cerebellum learns forward models of limb dynamics and sends corrective signals via Purkinje cells | TT is the forward model; steering is the corrective Purkinje signal; death layers are neural noise introduced by mistargeted stimulation |
| **Aerospace (fly-by-wire)** | Control augmentation systems prevent pilot inputs from exceeding aircraft structural limits — the pilot's input is _shaped_ not overridden | Steering should shape the model's trajectory within "structural limits" (coherence bounds) — α is a gain limiter |
| **Quantum error correction** | Surface codes detect and correct errors using syndrome measurements; correction is applied to specific qubits | Per-layer steering as syndrome measurement (TT detects error) and targeted correction (apply to specific layers) |
| **Economics (nudge theory)** | Small choice architecture changes produce large behavioral shifts; effectiveness depends on existing preference structure | Per-layer nudges only work if the model's "preference" (latent capability) already inclines toward correctness |
| **Compiler optimization** | Profile-guided optimization (PGO) collects runtime profiles, then recompiles with optimized paths | TT is the profiling pass; steering is the optimized "compiled" path; generation is the optimized execution |

### Foreign-Domain Dynamics

| Domain | Dynamic | Application |
|--------|---------|-------------|
| **Cellular differentiation** | Morphogen gradients determine cell fate based on concentration thresholds | α and layer selection form a "steering gradient" that determines token-level generation fate |
| **Musical harmony** | Overtones create consonance/dissonance based on frequency ratios | Layer steering vectors may interfere constructively (harmony/L8) or destructively (dissonance/L9) |
| **Military strategy (OODA loop)** | Observe-Orient-Decide-Act cycles at different tempos | Steering is the "Orient" modification — changing the model's orientation before it decides |

### Blind Spot Alert
Inspiration from other domains may suggest inappropriate mappings. The cerebellum analogy is particularly risky — neural networks and biological brains have fundamentally different learning mechanics (backprop vs spike-timing plasticity).

---

## Lens 9: ADVERSARIAL

### Cheapest Structural Attack
**Attack**: Train a TT that deliberately learns harmful velocity patterns, then apply its predictions as steering. Since the TT's R² is high (0.85-0.94), it would faithfully reproduce the harmful trajectory if trained on adversarial examples.
- **Cost**: Training a single TT (~30 min)
- **Impact**: Guaranteed -23pp+ (death layer effect on all layers)
- **Detection**: Standard evaluation would not distinguish adversarial from benign TT

### Relationship Whose Failure Would Collapse the System
**Most critical relationship**: J13 (Steering→Accuracy). If the +20pp effect is a statistical artifact or confounded with other variables (e.g., prompt formatting changes), the entire steering paradigm lacks empirical support.

**Second most critical**: J2 (Velocity→TT). If the TT's R² is from predicting trivial structure (e.g., token position encoding), the TT is a confounded predictor.

### Misapplication of Potential

| Misapplication | Harm | Mechanism |
|---------------|------|-----------|
| Steering a medical diagnosis model at "L9-equivalent" layer | Patient misdiagnosis | Amplifying error direction |
| Automated α search without safety bounds | Complete output collapse | α too high → chaotic outputs |
| Deploying steering without per-layer validation | Silent accuracy degradation | Death layers dominate at inference |
| Using steering on untested architectures | Unexpected behavior | Steering surface doesn't exist as assumed |

### Blind Spot Alert
The adversarial lens focuses on intentional attacks but the most likely "attack" is unintentional — the experimenter's own confirmation bias. The project finds L8=+20pp and stops looking; but L8 may be the best of a bad set if the steering paradigm has a hidden flaw.

---

## Lens 10: PARADOXICAL

### Structural Self-Reference
**Paradox**: The TT predicts where the hidden state is going *in the natural generation process*. Steering changes where the hidden state goes. So the TT's prediction becomes *false once steering is applied* because the trajectory has been modified by the steering that the TT's prediction enabled.

This is a **self-undermining prediction**: the act of using the prediction invalidates it.

**Resolution**: The system must either (1) re-predict after each steering step (online updating), or (2) accept that the steering pushes toward a *target* trajectory that was learned from unsteered data, and the divergence from that target is the price of improvement.

### Relational Gödel Sentence
**Gödel-like statement**: "This steering direction cannot be evaluated by this TT." — The TT predicts the *direction* of steering but cannot predict whether steering in that direction *improves* anything. The evaluation requires a separate mechanism (accuracy measurement) that the TT has no access to.

**Implication**: The TT is inherently incomplete — it can predict velocity but cannot predict the *value* of modifying that velocity. This is a fundamental limitation, not an engineering artifact.

### Inversion at the Limit
**Hypothesis**: As α → ∞ (infinite steering), the output distribution becomes uniform random (all steering directions cancel out, or model saturates). But at very low α, steering has no effect. Therefore the α-accuracy function must have a maximum at some intermediate α, after which increased steering *decreases* accuracy.

**Inversion point**: The L8 +20pp result is already past the optimal α for other layers — L9 shows -23pp at the same α, suggesting L9's optimal α is negative (steer *against* L9's velocity prediction) or zero (don't steer L9 at all).

**Paradoxical resolution**: The "death layer" is not a layer that cannot be steered — it's a layer that must be steered in the *opposite* direction. If the TT correctly predicts L9's velocity (R² is high across layers), then applying −α (steer opposite to prediction) should also produce improvement.

### Blind Spot Alert
The paradoxical lens reveals that the fundamental question — "does the TT predict the right direction?" — has a self-referential answer: "it depends on whether you're already steering." This suggests the need for an **online adaptive TT that is aware of its own steering effects**.

---

## Convergence Analysis

### High-Confidence Findings (≥5 lenses agree)

| Finding | Supporting Lenses | Confidence |
|---------|------------------|------------|
| 1. Velocity dynamics have learnable structure (R²=0.85-0.94) | 1, 2, 4, 5, 6, 7, 8 | **HIGH** (7/10) |
| 2. Per-layer effects are heterogeneous (trim-tabs vs death layers) | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 | **HIGH** (10/10) |
| 3. The TT is descriptive, not normative | 2, 5, 7, 9, 10 | **HIGH** (5/10) |
| 4. Layer independence is an unsupported assumption | 1, 4, 7, 8, 9, 10 | **HIGH** (6/10) |
| 5. Steering requires existing model capability | 2, 4, 5, 6, 7, 8 | **HIGH** (6/10) |

### Contested Findings (≥3 lenses disagree)

| Finding | Pro-Lenses | Con-Lenses | Status |
|---------|-----------|------------|--------|
| 1. The +20pp effect is "real" (not noise) | 2, 6, 8 | 5, 7, 9, 10 | **DISPUTED** — Sample size concerns vs consistent replication |
| 2. Contrastive TT will produce better steering | 3, 7, 8 | 2, 5, 10 | **DISPUTED** — Theoretical appeal vs known failure on Math-1.5B |
| 3. Cross-model transfer proves model-agnostic dynamics | 1, 6, 8 | 7, 9, 10 | **DISPUTED** — One transfer pair vs assumption concerns |

### Persistent Blind Spots (unaddressed after all 10 lenses)

| Blind Spot | Why Missed | Potential Resolution |
|------------|-----------|---------------------|
| 1. What does the TT actually represent? | No lens examines TT internals | Phase 8 mechanistic check |
| 2. Are layer effects independent? | Requires multi-layer experiments not yet done | Phase 7 causal mapping |
| 3. What is the correct steering direction sign? | No lens tests the `+` vs `−` α assumption | Phase 4 divergent generation |
| 4. What is the theoretical upper bound on steering? | All lenses are empirical | Phase 10 hyperstitional bridge |
| 5. Does steering at L8 change L9's computation? | No multi-layer coupling experiments | Phase 7 causal mapping |
