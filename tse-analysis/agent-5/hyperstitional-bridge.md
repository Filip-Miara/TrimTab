# Phase 10: Hyperstitional Bridge

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Structural Hypotheses

### H-1: Death Layer Sign Inversion
**Type**: Structural
**Statement**: "Steering a death layer (L9, L7, L15+) with −α (opposite to TT prediction) will produce a positive accuracy change comparable in magnitude to the current negative effect (+23pp for L9)."

**Falsification Criteria**:
- **Refutes**: L9 with −α produces ≤+10pp or still shows ≤0pp
- **Confirms**: L9 with −α produces ≥+15pp
- **Minimum Experiment**: `run_math15_sweep.py --layers 9 --alpha -0.1 --n-test 200`

**Risk**: If false, the death layer mechanism is not sign-related — it's architectural (specific computation being disrupted)
**Value**: If true, it doubles the number of beneficial layers and suggests symmetry in steering dynamics
**Channel**: `[experiment]` — Immediate next experiment

### H-2: Multi-Layer Additivity
**Type**: Structural
**Statement**: "Steering multiple beneficial layers simultaneously with per-layer optimal signs and α values produces additive accuracy improvements (L8: +α, L9: −α → net ~ +43pp)."

**Falsification Criteria**:
- **Refutes**: Multi-layer steering ≤ best single layer (non-additive interaction)
- **Confirms**: Multi-layer steering ≥ sum of individual effects
- **Minimum Experiment**: Steer L8(+α) + L9(−α) simultaneously

**Risk**: High — if layers interact destructively, the additivity assumption is invalid
**Value**: If true, total steering gain could reach +40-60pp, potentially surpassing the model's human-level performance
**Channel**: `[experiment]`

### H-3: Residual Stream Amplification
**Type**: Structural
**Statement**: "The steering effect at layer L propagates through the residual stream to subsequent layers, and the trim-tab/death-layer classification corresponds to whether this propagation is constructive (aligned with correct reasoning) or destructive (misaligned)."

**Falsification Criteria**:
- **Refutes**: Measuring hidden states at layer L+1 after steering at layer L shows no systematic difference from unsteered
- **Confirms**: Hidden state direction change at L+1 is predictable from steering direction at L
- **Minimum Experiment**: Record hidden states at all layers during L8 and L9 steering; compare trajectories

**Risk**: Low (measurement only)
**Value**: If true, provides mechanistic understanding of why per-layer effects differ
**Channel**: `[theory]` → `[experiment]`

---

## Relational Hypotheses

### H-4: TT Captures Spurious Correlations, Not Causal Dynamics
**Type**: Relational
**Statement**: "The TrajectoryTransformer achieves R²=0.85-0.94 by learning token-position frequency patterns and smoothness heuristics, not by modeling causal hidden state dynamics. Evidence: shuffling token positions will reduce R² by <20% (positions are not the primary signal), while ablating token embeddings will reduce R² significantly (token identity is the signal)."

**Falsification Criteria**:
- **Refutes**: Position shuffle reduces R² by >50% (TT was learning positional dynamics), OR token ablation reduces R² by <20% (TT wasn't using token identity)
- **Confirms**: Position shuffle reduces R² by <20% AND token ablation reduces R² by >50%
- **Minimum Experiment**: Run E1 (position shuffle) and E2 (ablate token embedding) from Phase 8

**Risk**: Fundamental — if confirmed, the entire steering paradigm is built on a confounded predictor
**Value**: If refuted (TT learns causal dynamics), the approach is validated at a deep level
**Channel**: `[experiment]` — Critical to resolve

### H-5: Correct/Incorrect Manifold Separability
**Type**: Relational
**Statement**: "The hidden state trajectories of correct and incorrect answers on Qwen2.5-7B occupy linearly separable manifolds. Steering from the incorrect manifold toward the correct manifold via contrastive TT will produce accuracy improvements > any standard TT improvement."

**Falsification Criteria**:
- **Refutes**: A linear probe cannot distinguish correct from incorrect hidden states (accuracy < 60%)
- **Confirms**: Linear probe accuracy > 80% AND contrastive TT steering outperforms standard TT by ≥5pp
- **Minimum Experiment**: Train linear probe on (hidden_state, correct/incorrect) pairs; evaluate contrastive TT

**Risk**: Medium — if manifolds are not separable, contrastive TT cannot work
**Value**: If confirmed, provides theoretical grounding for contrastive approach
**Channel**: `[experiment]` — Pending evaluation

### H-6: Steering Surface Architecture Dependence
**Type**: Relational
**Statement**: "The effectiveness of K/V-cache steering is mediated by the attention mechanism — models with standard MHA have steerable K/V caches, while models with hybrid attention (GatedDeltaNet + FA) do not because only 25% of layers have standard K/V, and the recurrent state of GDN cannot be steered via K/V modification."

**Falsification Criteria**:
- **Refutes**: A hybrid attention model shows positive steering results on any layer with standard K/V modification
- **Confirms**: No hybrid model layer improves with K/V steering, AND the effect correlates with percentage of standard K/V layers
- **Minimum Experiment**: Identify and steer only the FA layers of Qwen3.5-2B; compare with MHA-only control

**Risk**: Low (already observed across models)
**Value**: Guides architecture selection for future steering experiments
**Channel**: `[codebase]` — Architectural recommendation

### H-7: PPL-Correctness Independence
**Type**: Relational
**Statement**: "Model confidence (perplexity/loss on the generated token) is orthogonal to correctness for reasoning tasks. A model can be confidently wrong, and the PPL-based gating mechanism (<0.1% gate rate) fails because the model assigns high confidence to incorrect reasoning steps."

**Falsification Criteria**:
- **Refutes**: There exists a PPL threshold that separates correct from incorrect answers with >80% accuracy
- **Confirms**: No PPL threshold separates correct from incorrect better than random (55% accuracy)
- **Minimum Experiment**: Compute PPL for all tokens in correct and incorrect trajectories; find optimal threshold

**Risk**: Low (already measured: r=0.85 but gate rate <0.1%)
**Value**: If confirmed, explains why PPL-based steering fails and justifies alternatives
**Channel**: `[theory]`

---

## Potential Hypotheses

### H-8: Self-Improving Steering Loop Emergence
**Type**: Potential
**Statement**: "A closed-loop system that steers → evaluates → collects new trajectories → retrains the TT → updates steering policy will discover steering configurations with accuracy gains >2× the best manual configuration, because the self-loop enables the TT to learn from steered trajectories (solving the distribution-shift problem)."

**Falsification Criteria**:
- **Refutes**: The self-loop shows no improvement after 10 generations, OR converges to a suboptimal solution
- **Confirms**: The self-loop shows monotonic improvement over 10+ generations, reaching ≥+40pp
- **Minimum Experiment**: Build the automated pipeline (Phase D1)

**Risk**: High (engineering effort, potential for no convergence)
**Value**: Transformative — would make the approach fully autonomous
**Channel**: `[codebase]` — Long-term infrastructure

### H-9: Resonant Steering Frequency
**Type**: Potential
**Statement**: "Each transformer layer has a characteristic 'frequency response' to steering — defined as the function accuracy_gain = f(steering_frequency) where steering_frequency is how often the steering vector is recomputed. Layers like L8 have a resonance peak at 'every token' (frequency=1), while L9 would respond best at a different frequency."

**Falsification Criteria**:
- **Refutes**: Varying steering frequency produces the same accuracy gain across all layers
- **Confirms**: Different layers have different optimal steering frequencies
- **Minimum Experiment**: Steer at frequencies {every_token, every_2_tokens, every_4_tokens, every_8_tokens} and compare

**Risk**: Medium (may show no frequency effect)
**Value**: If confirmed, adds a new controllable dimension to steering
**Channel**: `[experiment]`

### H-10: Capability Threshold Is a Phase Transition
**Type**: Potential
**Statement**: "The model capability threshold (~40% GSM8K) is not a gradual barrier but a phase transition — below 40%, the hidden state manifold lacks the 'correct reasoning attractor' entirely; above 40%, the attractor exists and can be amplified. Models at 38% (Math-1.5B) vs 73% (Qwen2.5-7B) differ not just in degree but in kind of hidden state organization."

**Falsification Criteria**:
- **Refutes**: A model at 38% shows a small but nonzero steering effect (e.g., +5pp), suggesting continuous improvement
- **Confirms**: All models below 40% show zero positive steering effect, while 73% shows +20pp
- **Minimum Experiment**: Find a model with 40-50% baseline and test; OR artificially degrade Qwen2.5-7B to 40% and test

**Risk**: Low-Medium
**Value**: If confirmed, reframes the theoretical understanding of hidden state organization
**Channel**: `[theory]`

---

## Research Calls

| Hypothesis | Target Audience | One-Sentence Pitch |
|------------|----------------|-------------------|
| H-4 (TT spurious) | Mechanistic interpretability researchers | "Does your velocity predictor actually learn dynamics, or does it exploit token-position smoothness? We found a test." |
| H-5 (Manifold separation) | Representation learning community | "Correct and incorrect reasoning traces in transformers may occupy distinct subspaces — we can test this with linear probes." |
| H-1 (Death layer sign) | Steering/activation engineering community | "Before declaring a layer a 'death layer,' did you try steering in the opposite direction? We think it's the right direction inverted." |
| H-8 (Self-improving loop) | RL for LM alignment | "Can a language model learn to steer itself better than we can steer it? Our architecture for this is here." |
| H-9 (Resonant frequency) | Computational neuroscience ↔ AI | "Transformer layers may have resonant frequencies for steering, analogous to neural oscillations. Test your favorite model." |
| H-10 (Phase transition) | Scaling laws / emergent abilities | "The 'capability threshold' for steering may be a phase transition in hidden state topology, not a gradual improvement." |

---

## Hypothesis Summary Table

| ID | Statement | Type | Cost to Test | Value if True | Channel |
|----|-----------|------|-------------|---------------|---------|
| H-1 | Death layers work with −α | Structural | 1 GPU-hr | **Transformative** | `[experiment]` |
| H-2 | Multi-layer additivity | Structural | 8 GPU-hrs | High | `[experiment]` |
| H-3 | Residual stream amplification | Structural | 4 GPU-hrs | Medium | `[theory]` |
| H-4 | TT learns spurious correlations | Relational | 2 GPU-hrs | **Fundamental** | `[experiment]` |
| H-5 | Manifold separability | Relational | 2 GPU-hrs | High | `[experiment]` |
| H-6 | Surface architecture dependence | Relational | Already known | Confirmed | `[codebase]` |
| H-7 | PPL-correctness independence | Relational | 1 GPU-hr | Confirmed | `[theory]` |
| H-8 | Self-improving loop | Potential | Weeks | **Transformative** | `[codebase]` |
| H-9 | Resonant steering frequency | Potential | 2 GPU-hrs | Medium | `[experiment]` |
| H-10 | Phase transition threshold | Potential | 4 GPU-hrs | High | `[theory]` |
