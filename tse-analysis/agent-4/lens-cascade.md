# Phase 2: Multi-Lens Analysis Cascade

## Lens 1: ANALOGICAL

**Structural**: The velocity steering architecture mirrors a proportional-integral (PI) controller in control theory — the current hidden state is the "position," velocity is the "derivative," and α is the "gain." In this analogy, the TT acts as a state observer estimating system dynamics. Death layers correspond to control inputs with wrong sign (positive feedback in a negative-feedback system). Trim-tab layers correspond to well-calibrated actuators.

**Relational**: The contrastive steering signal (v_c - v_i) is structurally isomorphic to the gradient of a value function in reinforcement learning — it points from worse states toward better states. This is precisely the relationship between the policy gradient ∇_θ log π(a|s)·A(s,a) and the advantage function A. The "capability threshold" maps to the exploration-exploitation boundary: a random policy (low baseline) has no meaningful advantage signal.

**Potential**: Analogous solution from robotics: "operational space control" (Khatib, 1987) decouples complex control into task-space and null-space components. Applying this: trim-tab layers are the task-space controller (changes output), death layers are null-space perturbations that need to be projected out. Solution: project velocity onto the subspace orthogonal to death layer directions.

**Blind Spot Alert**: The control theory analogy overlooks that hidden states are NOT physical positions — they're embedded in a high-dimensional space where Euclidean operations (addition, scaling) may not have the semantics assumed.

---

## Lens 2: DIALECTICAL

**Thesis**: "Velocity-based steering improves reasoning by nudging hidden states toward better trajectories."
- Supported by: R²=0.94, L8:+20pp, cross-model transfer, SVAMP generalization

**Antithesis**: "Velocity-based steering is fundamentally limited — it can only amplify existing capability, cannot create it, and produces death layers that destroy performance."
- Supported by: 5 models below threshold all failed, L9:-23pp, all-layers:-45pp, Math-1.5B no trim tabs

**Synthesis**: Steering is a *latent capability amplifier* requiring both (a) the model to possess target capability in its hidden state manifold AND (b) per-layer selection to avoid anti-signal. The synthesis suggests a **diagnostic-first** protocol: before steering, verify that the target direction exists in the model's latent space. Models failing the diagnostic should be fine-tuned (not steered). This transforms "steering requires capability" from a limitation into an operational principle.

**Blind Spot Alert**: The dialectic assumes the thesis and antithesis exhaust the space — it misses the possibility that steering with different mechanisms (e.g., weight-space rather than activations) could bypass the capability threshold entirely.

---

## Lens 3: BLENDING

**Blendable atoms**:
- A05 (α) × A17 (per-layer α) → **Adaptive per-layer α**: Blend each layer's α with a learned function of the current hidden state
- A03 (velocity pred) × A14 (contrastive) → **Contrastive-weighted velocity**: v_standard + β·(v_c - v_i) where β is learned per token
- A19 (architecture) × A04 (KV surface) → **Architecture-aware steering**: Different steering mechanics for GDN vs MHA layers within the same model
- A20 (manifold) × A09 (death layer) → **Manifold-gated steering**: Only steer when hidden state is in a "safe" region of manifold

**Novel composite**: **Mixture-of-Steerings (MoS)**: Train N steering experts, each specializing in different regions of the hidden state manifold, with a router that selects which steering vector to apply per token.

**Blind Spot Alert**: Blending assumes combinability — not all atoms blend productively. Blending A04 (KV mod) with A22 (reasoning quality) might produce a composite that's neither good steering nor good evaluation.

---

## Lens 4: SYSTEMS

**Feedback loops**:
1. **Reinforcing (positive)**: Better steering → higher accuracy → more confident steering → even higher accuracy (if on trim-tab layers)
2. **Balancing (negative)**: Steering pushes hidden state → model compensates (internal homeostasis) → steering effect attenuated
3. **Reinforcing (negative)**: Death layer steering → accuracy drops → model enters wrong manifold → subsequent tokens get worse → cascading failure
4. **Delayed feedback**: Steering applied at token t affects tokens t+1...t+N through KV cache → effect is spread across future positions

**Leverage points**:
1. **Rate of steering accumulation**: Steering accumulates per token; early errors compound. Most important point: the FIRST steered token determines trajectory quality.
2. **Death layer damping**: Adding a "death layer gate" (if layer ∈ death set, skip steering) would eliminate the main failure mode.
3. **α as damping coefficient**: Perceptual evidence suggests α=0.1 gives +20pp; α=1.0 likely oversteers. This is a classic PID D-term tuning problem.

**Blind Spot Alert**: System dynamics assume hidden state evolution is smooth and linear — but attention is a discrete operation (token→token) that can produce discontinuous jumps in hidden state space.

---

## Lens 5: ABDUCTIVE

**What explains the trim-tab/death layer pattern?**
- **Best explanation**: Layers in a transformer have functional specialization. Middle layers (L8-10 in 28-layer model) perform "reasoning composition" — binding concepts from earlier layers into coherent thought. Early layers (L0-3) handle token embedding and syntax; late layers (L15+) produce token predictions. Steering middle layers amplifies reasoning; steering prediction layers destroys logit calibration.
- **Evidence**: L8 is the strongest trim-tab; L9 (one layer deeper) is a death layer. This suggests a narrow "sweet spot" where reasoning is composed but not yet projected to vocabulary. L2's +17pp (early) is surprising — suggests some reasoning composition happens earlier than assumed.
- **How L9 might kill accuracy**: If L9 performs "logit normalization" or "vocabulary sharpening," modifying its KV cache scrambles the model's output distribution permanently.

**What explains Math-1.5B's lack of trim tabs?**
- **Best explanation**: Math-1.5B is a base model (not instruct-tuned). Instruct-tuning creates the "correct answer manifold" that steering exploits. Without instruct-tuning, correct and incorrect hidden states are not separated enough for contrastive steering to work.

**Blind Spot Alert**: Abduction is vulnerable to confirmation bias — the "functional specialization" story fits the data but hasn't been directly verified (no activation patching experiments).

---

## Lens 6: TRAJECTORY

**Evolution of the project** (Session 1→5):
1. Session 1: Logit correction on Qwen3.5-2B → 0% improvement ❌
2. Session 2: Per-layer KV-cache steering on SmolLM2 → moderate insight (high R², but model can't do math)
3. Session 3: Qwen2.5-7B per-layer sweep → L8:+20pp discovery 💡
4. Session 4: Cross-model transfer + SVAMP generalization → pattern confirmed ✅
5. Session 5: Contrastive TT training → pending evaluation ⏳

**Extrapolation trajectory**:
- Next 2 sessions: Contrastive evaluation, asymmetric α sweeps, multi-layer combos
- Next 10 sessions: Per-head steering, RL-based α optimization, multi-head contrastive ensemble
- Next 20 sessions: Steering as a general capability enhancement method (not just math)

**Blind Spot Alert**: Project trajectory is accelerating in complexity but not in fundamental understanding of WHY steering works. Without mechanistic interpretability, future gains will be empirical and fragile.

---

## Lens 7: METACOGNITIVE

**Structural blind spots**:
1. No analysis of what the TT actually learned (feature attribution on TT predictions)
2. No ablation study on TT capacity (is 6-layer, d_model=768 necessary? sufficient?)
3. No characterization of the velocity field's nullspace (directions that TT never predicts)

**Relational blind spots**:
1. Correlation between per-layer α and hidden state norm not examined
2. No analysis of attention head distribution change after steering
3. Relationship between R² at a specific layer and its steering effectiveness not computed

**Potential blind spots**:
1. Could steering be applied to keys and values separately (not just both)?
2. Could steering be applied at different strengths per attention head?
3. Could steering be applied to specific token positions (not just the most recent)?

**Blind Spot Alert**: The analysis itself lacks a "second-order" perspective — it has not considered whether the steering paradigm itself (modify KV cache) is the right abstraction or whether a fundamentally different approach (e.g., guided decoding with a separate verifier) would dominate.

---

## Lens 8: INSPIRATION

| Domain | Mapped Structure | Transplantable Solution |
|--------|-----------------|------------------------|
| **Neuroscience** | Cortical columns as specialized processors | Trim-tab layers correspond to cortical microcolumns performing domain-general computation; death layers are output pathways. Solution: **layer-specific microstimulation protocols**. |
| **Aerospace** | Trim tabs on aircraft control surfaces | The project already uses this metaphor — but aircraft trim tabs are ALWAYS positive adjustments. Solution: model death layers as **control surface reversal** (supersonic aileron reversal) — at certain speeds, control inputs invert. Detect the condition and flip α sign. |
| **Economics** | Pigouvian taxation (corrective tax on negative externality) | Death layers impose a negative externality on all other layers. Solution: **internalize the externality** by adding a "death layer tax" that adjusts α inversely to layer harmfulness. |
| **Quantum control** | Dynamical decoupling (pulse sequences that cancel noise) | Apply steering as a **pulse sequence**: alternate between positive and negative α at different frequencies to cancel death layer effects while preserving trim-tab effects. |
| **Music** | Equalization (EQ): boost/cut per frequency band | Steering is EQ for hidden states. Trim-tab = boost, death = cut. Solution: **compressed sensing** to identify which "frequency bands" (hidden state directions) are trim-tab vs death, then apply adaptive EQ. |

**Blind Spot Alert**: Cross-domain inspiration is analogical, not causal. The aerospace trim tab metaphor is seductive but may oversimplify the nonlinear dynamics of transformer hidden states.

---

## Lens 9: ADVERSARIAL

**Cheapest structural attack** on the steering system:
1. Train a model specifically to RESIST steering — add a small adversarial loss during fine-tuning that penalizes hidden state change under α perturbation. This would be a "steering-resistant" model.
2. The steering system has no defense: it assumes the model is passive. A model that actively counter-steers would defeat it.

**Relationship whose breakage collapses the system**:
- J03 (velocity → KV modification): If the projection from hidden state to K/V is non-injective in steering-relevant directions, steering becomes impossible. This is EXACTLY what happens with GDN layers.

**Misapplication of potential that causes harm**:
- Alpha too high (α > 1.0) for trim-tab layers: overshoots the correct manifold, converting a trim-tab into a death layer.
- Contrastive signal applied to a model where correct/incorrect trajectories are NOT separable: v_c - v_i points to a meaningless direction, reducing accuracy.

**Blind Spot Alert**: The adversarial lens assumes a malicious actor, but the most dangerous failure mode is benign: a user who doesn't understand the capability threshold applies steering to a weak model and concludes the technique is worthless.

---

## Lens 10: PARADOXICAL

**Structural paradox**: The TT predicts where hidden states ARE going (descriptive), but steering requires knowing where they SHOULD go (normative). The contrastive approach attempts to convert descriptive→normative by subtraction, but **v_c - v_i may be neither** — it could be a spurious direction that happens to differ between the two trajectory sets.

**Relational paradox**: Steering requires the model to have the target capability (A11), but the steering signal comes FROM the model's own behavior (v_c and v_i are both from the same model's generations). **The system is steering itself using its own dynamics** — this is a self-referential closed loop. If the model can't reason, it can't generate correct trajectories, so it can't create a steering signal that would teach it to reason.

**Potential paradox**: If steering improves accuracy from 73%→93% (upper bound speculation), the remaining failures will be the hardest problems where the model's hidden state dynamics are most erratic and the TT prediction is least reliable. **Steering works best on easy cases and worst on hard ones** — the opposite of what's needed.

**Limit inversion**: Push α to a very large value (α → ∞). At this limit, the KV cache is completely dominated by the steering signal, and the model's own hidden states are irrelevant. The system becomes a **steering-only token generator** — essentially a new model that uses the original model's K/V projections as a fixed transformation. This is either useless (pure noise) or transformative (a new way to control generation).

**Blind Spot Alert**: The paradoxical lens identifies self-reference in the system but may overstate its practical importance — many successful engineering systems (power steering, cruise control) have similar self-referential dynamics.

---

## Convergent Check

### High-Confidence Findings (≥5 lenses agree)

| Finding | Lenses | Confidence |
|---------|--------|------------|
| Per-layer selectivity is non-negotiable | 2, 4, 5, 6, 9 | HIGH |
| The capability threshold is a real constraint | 1, 2, 5, 6, 10 | HIGH |
| Architecture (MHA vs hybrid) determines steering surface | 3, 5, 6, 8, 9 | HIGH |
| Contrastive steering's value is unconfirmed (testable claim) | 1, 5, 7, 9, 10 | HIGH |
| Death layers likely correspond to output/logit-projection layers | 1, 4, 5, 8, 10 | HIGH |

### Contested Findings (≥3 lenses disagree)

| Finding | Supporting Lenses | Opposing Lenses | Status |
|---------|-------------------|-----------------|--------|
| α=0.1 is a near-optimal fixed value | 4 (PID tuning intuition) | 3 (blending suggests dynamic α), 9 (α too high hurts) | REOPEN for investigation |
| Cross-model transfer is robust | 1 (analogy: dynamics generalize), 6 (empirical) | 5 (n=1, might be coincidence), 7 (no mechanism analysis) | REOPEN with more model pairs |
| Trim-tab layers are functionally specialized for reasoning | 5 (abductive), 8 (cortical inspiration) | 2 (dialectical: pattern may be artifact), 7 (no mechanistic verification) | REOPEN for mechanistic study |

### Persistent Blind Spots

| Blind Spot | Why Missed | Severity |
|------------|------------|----------|
| What does the TT's latent space look like? | No feature attribution or probe analysis | HIGH |
| What is the per-token optimal α? | Only layer-level sweeps done | MEDIUM |
| Can steering work on non-math tasks? | Only GSM8K and SVAMP tested | MEDIUM |
| What happens at very small (α→0) or very large (α→∞) extremes? | Only α=0.1 tested per layer | HIGH |
| Is the death layer pattern consistent across random seeds? | No repeated sweeps with different seeds | MEDIUM |
