# Phase 2: Multi-Lens Analysis Cascade

---

## Lens 1: ANALOGICAL

| Domain | Analogue | What It Reveals |
|--------|----------|-----------------|
| **Control Theory** | PID controller with feedforward | TT = feedforward; α = P-term. Missing integral (accumulated error) and derivative (rate of change of error) |
| **Neurobiology** | LTD/LTP | Steering strength α should depend on recent activity (metaplasticity) |
| **Aerodynamics** | Trim tab on control surface | Different layers are different trim tabs. L8 is the elevator trim, L9 is the rudder trim (broken) |
| **Financial Markets** | Alpha decay | Steering signal may decay as generation progresses — should refresh every N steps |
| **Evolutionary Biology** | Phenotypic plasticity | Model may adapt hidden states to resist steering (homeostatic response) |
| **Quantum Mechanics** | Measurement problem | Steering changes the state, invalidating the TT's prediction basis (observer effect) |

**Blind spot alert**: All analogies assume the steering target is STATIONARY. If the "correct" hidden state trajectory changes during generation, our static TT is insufficient.

---

## Lens 2: DIALECTICAL

- **Thesis**: Velocity-based latent steering improves reasoning by pushing hidden states toward correct trajectories
- **Antithesis**: The model's hidden state manifold is a successive refinement process. Adding velocity adds NOISE, not signal (all steering ≤ baseline in initial experiments)
- **Synthesis**: Per-layer steering (trim tabs) converts noise into signal. The synthesis is LAYER SELECTIVITY — steer only trim-tab layers, avoid death layers. Contrastive training converts descriptive TT into normative TT

**High confidence**: ≥5 lenses agree that per-layer selectivity is essential.

---

## Lens 3: SYSTEMS

**Feedback loops identified:**
1. **Reinforcing (positive)**: L8 steering → better tokens → better trajectories → better TT predictions → stronger L8 steering
2. **Balancing (homeostatic)**: Model's residual stream resists perturbation → diminishing returns beyond α threshold
3. **Reinforcing (negative)**: L9 steering → wrong tokens → worse trajectory → cascade failure (0% accuracy)

**Leverage point**: The L8 reinforcing loop can be exploited; the L9 loop must be avoided. The system has a narrow "sweet spot" of 2-3 trim-tab layers.

---

## Lens 4: ABDUCTIVE

Given 88% token divergence + 0% accuracy improvement + R²=0.85-0.94:

**Best explanation**: The TT predicts the AVERAGE trajectory, not the CORRECT trajectory. High R² means confident reproduction of error patterns. Contrastive TT fixes this by specifically isolating the correction direction.

**Alternative explanation** (not ruled out): Steering changes FORMATTING tokens (function words, punctuation) while reasoning-critical tokens are unchanged. The 88% divergence is on surface-level tokens only.

---

## Lens 5: TRAJECTORY

**Field trajectory**: Latent steering (Subbiah 2024, Turner 2024) showing influence but not reliable improvement. The field is at a local maximum of "we can steer but don't know where."

**Our trajectory**: Raw signal (R²=0.62) → Generation-trained (R²=0.94) → Contrastive (normative) → Asymmetric α → Multi-head ensemble. Each step should improve the signal-to-noise ratio.

**Prediction**: Within 3 months, successful approaches will combine: (1) per-layer steering, (2) contrastive/normative direction, (3) adaptive α, (4) online evaluation.

---

## Lens 6: ADVERSARIAL

**Cheapest attack**: Train a TT on steered trajectories → the TT learns to absorb the steering signal → steering becomes invisible.

**Three failures to explain:**
1. All-layers steering: 23% → 0% (net -23pp, bad layers dominate)
2. SmolLM2 steering: 20% → 0% (model too small for math, no capability to steer)
3. Qwen3.5-2B FA-only: 10% → 0% (hybrid arch blocks 75% of layers)

---

## Persistent Blind Spots

1. **What does the TT actually learn?** — Is it capturing causal velocity dynamics or just memorizing common trajectories? (Mechanistic interpretability not done)
2. **Are "correct" and "incorrect" manifolds actually separable?** — Contrastive approach assumes this; we haven't verified.
3. **Is GSM8K the right metric?** — Single-domain, single-format, English-only. Steering might work on other tasks but not this one.

---

## High-Confidence Findings (≥5 lenses agree)

1. Per-layer selectivity is essential (L8 good, L9 bad)
2. Generation-trained TT dramatically better than prompt-trained
3. Contrastive direction should outperform single TT
4. Steering requires model capability
