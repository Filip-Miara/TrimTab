# Conceptual Diffusion: Mechanistic Understanding Gap × Candidate Mechanisms

**Mode**: Deep conceptual diffusion between the central failure (NO analysis explains WHY steering works) and 8 candidate mechanisms proposed across TSE agents 1-5 and the complex-α analysis.

**Date**: 2026-06-14

---

## 1. THE CENTRAL VOID

The single most robust finding of the entire RankAdaptation meta-analysis is **negative**: no agent, no lens, no recombination event produces a validated mechanism. The empirical phenomenon is well-characterized (L8: +20pp, L9: -23pp, per-layer selectivity, cross-model transfer, R² up to 0.94) but causal understanding is zero.

The void has a specific structure:
- **We know that** adding α·v_l to the KV-cache at layer l changes accuracy by Δ_acc(l, α)
- **We do not know why** this particular perturbation at this particular surface produces this particular effect
- **We cannot predict** Δ_acc(l, α) for novel l, novel α, or novel tasks from first principles
- **We cannot rule out** that α·v is simply noise injection with per-layer random efficacy

This document enumerates ALL candidate mechanisms, formalizes each, designs minimal disambiguation, estimates priors, and produces a single protocol that distinguishes all 8.

---

## 2. CANDIDATE MECHANISM CATALOG

### C1: K/V AMPLIFICATION

**Source**: Agent 2 EM-2, Agent 5 mechanistic check §1, final meta-synthesis Phase 2.10 Paradox 2

**Core claim**: The steering effect operates through the attention mechanism specifically. Adding α·v to the K/V cache entry at position t modifies the attention logits at all future positions t+1, t+2, ..., because queries at future positions attend to the modified key. The attention softmax acts as an amplifier — small perturbations in key vectors produce large shifts in attention distribution when they cross the softmax threshold.

**Formal statement**:
Let A_ij = softmax( (W_Q·h_i)^T · (W_K·h_j) / sqrt(d_k) ) be the attention weight from position i to position j. Under steering at layer l, position t:
    K'_t = K_t + α · v_t
    ΔA_ij ∝ exp( (W_Q·h_i)^T · (W_K·(α·v_t)) / sqrt(d_k) ) - 1
For positions i > t, this shifts attention mass by:
    ||ΔA_i:|| ≈ α · ||v_t|| · ||W_K|| · ||W_Q·h_i|| / sqrt(d_k)
The softmax competition means that even if ΔA_ij is small, the relative shift in attention probability can be large if competing keys have similar scores. This is the **amplification** — a 0.1 change in logit produces a ~10% change in attention probability in the softmax regime.

**Predicted signature**:
1. Attention entropy at positions > t_steer changes measurably under steering
2. The effect is proportional to attention logit sensitivity, not hidden state trajectory smoothness
3. K-only steering produces different pattern than V-only steering (K modifies attention distribution, V modifies value aggregation)
4. The effect is amplified in layers with sharp attention distributions (low entropy)

**Falsification experiment**:
- **B4 (K/V split)**: Steer K only, V only, and both. If C1 is correct, K-only ≈ full steering; V-only ≈ 0.
- **Attention entropy correlation**: Compute correlation between pre-steering attention entropy at layer l and Δ_acc(l). If C1 is correct, ρ < 0 (lower entropy → higher amplification → larger effect).

---

### C2: OFF-MANIFOLD PERTURBATION

**Source**: Agents 1,3,4 proposed; final meta-synthesis counter-assumption ¬I3; implicit in 88% token divergence

**Core claim**: The hidden state trajectory during autoregressive generation lies on a low-dimensional manifold embedded in ℝ³⁵⁸⁴. Adding α·v pushes states OFF this manifold into regions where the model has not been trained to compute reliably. When this off-manifold perturbation happens to move states toward regions associated with correct reasoning, accuracy improves; when it moves toward chaotic regions, it degrades.

**Formal statement**:
Let M ⊂ ℝ³⁵⁸⁴ be the natural data manifold of hidden states during generation. Let T_h M be the tangent space at h. A natural trajectory segment satisfies h_{t+1} - h_t ∈ T_h M (the next state stays on the manifold). The velocity v_t = h_{t+1} - h_t is a tangent vector.

When we add α·v to the KV cache, we modify the key/value that future queries attend to. The effective perturbation to the hidden state computation is:
    h'_t = h_t + α · v_t + ε_model
where ε_model is the model's compensatory response to the modified attention. If α·v lies in T_h M (on the manifold), the model can process it naturally. If α·v has a component orthogonal to M, the state is pushed off-manifold.

The critical parameter is sin(θ) = ||proj_⟂(v)|| / ||v||, the angle between v and the manifold tangent. When sin(θ) is large, the perturbation is primarily off-manifold → unreliable computation.

**Predicted signature**:
1. The intrinsic dimensionality of the trajectory manifold (measurable via PCA) predicts steering efficacy — low-dim manifolds are harder to steer (v is more likely orthogonal)
2. Off-manifold distance ||h'_t - proj_M(h'_t)|| correlates with token divergence
3. α > 0.5 causes quality collapse because perturbation exceeds manifold thickness

**Falsification experiment**:
- **R1/IPCA**: Compute PCA of all hidden states across all generations. Measure intrinsic dimension d_95 (components to explain 95% variance). Compute tangent angle sin(θ) = ||v - proj_PCA(v)|| / ||v||. Correlate sin(θ) with Δ_acc per layer.
- **Prediction if C2 is true**: sin(θ) at L8 < sin(θ) at L9 (L8's velocity is more on-manifold)
- **Prediction if C2 is false**: sin(θ) is uniform across layers and uncorrelated with Δ_acc

---

### C3: DIRECTION MISALIGNMENT

**Source**: Agent 5 core finding (death layer sign inversion); all 5 agents implicitly; final meta-synthesis M4

**Core claim**: The TT's velocity predictions are directionally accurate (v points in the direction the hidden state moves during correct generation), but some layers have the WRONG sign — the direction of natural velocity for those layers is opposite to the direction of correctness improvement. L8 has positive alignment (adding v moves toward correct computation), L9 has negative alignment (adding v moves toward incorrect computation). No nonlinear amplification is needed — the effect is purely directional.

**Formal statement**:
Let v_correct(layer l, position t) be the velocity of correct trajectories and v_incorrect(l) be the velocity of incorrect trajectories at the same layer. The TT learns v̂ ≈ v_incorrect (because incorrect trajectories are the majority at that layer, or because the TT averages across all states). The optimal steering direction is:
    v_optimal(l) = sign(l) · v̂(l)
where sign(l) ∈ {+1, -1} is the per-layer polarity.

Under C3:
- sign(L8) = +1: TT output is aligned with improvement
- sign(L9) = -1: TT output is anti-aligned with improvement
- Steering at L8 with +α·v improves; steering at L9 with -α·v also improves

**Predicted signature**:
1. Negative α at L9 converts -23pp to approximately +20pp (symmetric around L8)
2. The cosine similarity between v_correct and v_incorrect varies significantly by layer, and this variation predicts sign(l)
3. Simple sign inversion is sufficient — no need for different α magnitudes per layer

**Falsification experiment**:
- **Signed α sweep (B1)**: Test L9 with α ∈ {-2, -1, -0.5, -0.1}. If C3 is correct, L9(-0.1) ≈ L8(+0.1) ≈ +20pp.
- **Cosine similarity calculation**: Compute cos_sim(l) = cos(v_correct(l), v_incorrect(l)) from existing trajectory data. Correlate with Δ_acc(l). If C3 is correct, cos_sim(L8) > 0 and cos_sim(L9) < 0.

---

### C4: FREQUENCY MODULATION

**Source**: Agent 2 EM-2 (music harmonics, PCA decomposition); Agent 3 frequency-domain PCA blend; complex-α lens 1 (signal processing)

**Core claim**: The velocity field v(t) across token positions can be decomposed into frequency components. Different layers specialize in different frequency bands of the hidden state trajectory. L8 operates in the low-frequency regime (smooth reasoning structure — the "carrier wave" of correct computation). L9 operates in the high-frequency regime (token-level noise, surface form fluctuations). Steering at L8 amplifies the reasoning signal; steering at L9 amplifies noise.

**Formal statement**:
Let V(ω) = FFT(v(t)) be the Fourier transform of the velocity across token positions. Define:
    P(ω) = |V(ω)|²  (power spectrum of velocity)
    P_l(ω) = power spectrum of velocity restricted to layer l

For each layer l, define the frequency centroid:
    ω_centroid(l) = ∫ ω · P_l(ω) dω / ∫ P_l(ω) dω

C4 predicts:
- ω_centroid(L8) < ω_centroid(L9) (L8 operates at lower frequencies)
- Steering efficacy Δ_acc(l) is proportional to the overlap between P_l(ω) and the "correct reasoning frequency band"
- Death layers have high ω_centroid (they modulate noise, not signal)

**Predicted signature**:
1. PCA decomposition of velocity: top components form smooth curves (low frequency) at L8, jagged curves (high frequency) at L9
2. Steering using only the top-k PCA components of v (low-pass filtered) preserves L8 efficacy but may INCREASE L9 efficacy (by removing noise)
3. The frequency centroid ω_centroid(l) predicts Δ_acc(l) with ρ > 0.5

**Falsification experiment**:
- **PCA-filtered steering**: Compute PCA of v across all layers. Steer using only top-N components (low-pass). Compare Δ_acc for L8 vs L9. If C4 is correct: L8(PCA-filtered) ≈ L8(full), L9(PCA-filtered) > L9(full).
- **Frequency centroid correlation**: Compute ω_centroid(l) from existing trajectory data. Correlate with Δ_acc(l). If C4 is correct, ρ > 0.4.

---

### C5: SMOOTHNESS EXPLOITATION

**Source**: Agent 5 E3 (naive baseline test); final meta-synthesis M8, Agent 4 H-5

**Core claim**: The TT does NOT learn causal velocity dynamics. It exploits the mathematical fact that hidden states change smoothly — h_{t+1} ≈ h_t + ε with small ε. The optimal prediction for a smooth trajectory is v̂ = 0 (predict the next state is the same as the current state). The TT's high R² comes from predicting near-zero velocity, not from understanding the model's computation.

**Formal statement**:
Let the naive baseline prediction be v̂_baseline = 0 (predict h_{t+1} = h_t). The TT prediction is v̂_TT. Define:
    R²_baseline = 1 - MSE(v=0) / Var(v)
    R²_TT = 1 - MSE(v̂_TT) / Var(v)

Under C5:
- R²_baseline is close to the maximum possible R² (smoothness accounts for most explainable variance)
- R²_TT is only marginally better than R²_baseline
- The residual R²_TT - R²_baseline comes from token-level regularities (punctuation, step markers), not causal reasoning

**Predicted signature**:
1. R²_baseline ≈ R²_TT within 0.05 (smoothness accounts for >90% of TT's performance)
2. Position shuffle (E1) does NOT reduce R² (because smoothness is position-independent)
3. Steering with random vectors of equal norm to TT predictions produces similar per-layer patterns

**Falsification experiment**:
- **E3 (naive baseline)**: Compute R² for v̂ = 0. If R²_baseline > 0.9 × R²_TT, C5 is supported.
- **E1 (position shuffle)**: Shuffle token positions, recompute R². If R² drops significantly (< 0.5 × original), C5 is weakened (position matters, not just smoothness).
- **Random vector steering**: If random vectors produce the same per-layer pattern as TT steering, C5 is strongly supported (any perturbation works equally well).

---

### C6: COMPLEX α PHASE

**Source**: Complex-α analysis (02-lens-cascade.md); Agent 5 (death inversion as special case); final meta-synthesis blending

**Core claim**: Steering has two degrees of freedom per layer — the acceleration component (second derivative of hidden state w.r.t. layer) is as important as the velocity component. The ratio of velocity to acceleration determines a "phase" θ where tan(θ) = ||a|| / ||v||. L8 has θ ≈ 0 (velocity-dominated), L9 has θ ≈ π/2 (acceleration-dominated). Steering with only α·v at L9 is like pushing perpendicular to the direction of motion — maximally inefficient.

**Formal statement**:
Let h[l] be the hidden state at layer l. Define:
    v[l] = h[l+1] - h[l]   (velocity, already used)
    a[l] = v[l+1] - v[l] = h[l+2] - 2h[l+1] + h[l]   (acceleration)

Complex steering modifies the KV cache by:
    δh[l] = r · (cos(θ) · v[l] + sin(θ) · a[l])
where r = |α| is the magnitude and θ is the phase.

Under C6:
- At L8, θ_opt ≈ 0: δh = α·v (current method works because velocity is the right direction)
- At L9, θ_opt ≈ π/2: δh = α·a (steering with velocity is wrong because v and a are orthogonal; optimal steering is along acceleration)
- At L9(+α·v), we are steering in an irrelevant direction → -23pp (not harmful per se, but ineffective → the effective steering magnitude is α·cos(θ) ≈ 0)

**This subsumes C3 as a special case**: if θ_opt at L9 is π instead of π/2 (i.e., steering along negative velocity), then C6 degenerates to C3 (simple sign inversion). C6 is more general and makes a different prediction.

**Predicted signature**:
1. L9 responds to pure acceleration steering (α=0, r > 0) with positive Δ_acc
2. L9 responds to pure negative-velocity steering (θ=π, r > 0) with ZERO effect (not positive, not negative) — because pushing backward is as useless as pushing forward when the system needs lateral force
3. The acceleration field a[l] has comparable predictive power to the velocity field v[l] when predicting optimal steering direction

**Falsification experiment**:
- **Acceleration-only steering (phase sweep)**: Steer L9 with δh = α·a (no velocity component). If C6 is correct, L9(α·a) > 0. If C3 is correct (simple sign inversion), L9(α·a) ≈ 0.
- **Phase sweep at L9**: Test θ ∈ {0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4} with fixed r = 0.1. C6 predicts peak at θ ≈ π/2. C3 predicts peak at θ = π. The two are distinguishable.
- **Compute a[l] R²**: Train TT on acceleration. If R²_acceleration ≥ 0.5 × R²_velocity, the acceleration signal is strong enough for C6 to be viable.

---

### C7: ATTENTION LOGIT PERTURBATION

**Source**: Implicit in K/V amplification; Agent 5 mechanistic check; project methodology (KV-cache surface selection)

**Core claim**: The steering effect is mediated by attention logit changes, not hidden state changes. The key insight: the KV-cache is the ATTENTION surface, not the hidden state surface. When we modify K'_t = K_t + α·v_t, we are changing the attention logits at future positions, which changes which tokens are attended to. The hidden state itself is only indirectly affected (through attention output).

This is subtly different from C1 (K/V amplification). C7 claims the attention logit change IS the mechanism (not a mediator), and the hidden state change is a downstream consequence. The difference matters for predicting generalization.

**Formal statement**:
Let the attention logit at position i (query) to position j (key) be:
    L_ij = (W_Q·h_i)^T · (W_K·h_j) / sqrt(d_k)

Under steering at position t, the logit from future position i > t to position t changes by:
    ΔL_it = (W_Q·h_i)^T · W_K · (α·v_t) / sqrt(d_k)

This changes the attention distribution at position i: all future tokens redistribute attention mass. The attention distribution change causes a change in the context vector c_i = Σ_j A_ij · (W_V·h_j), which then propagates through the MLP and residual stream.

Under C7, the mechanism is:
    α·v_t → ΔK_t → ΔL_it (for all i > t) → ΔA_i: → Δc_i → Δh_i → Δc_{i+1} → ...

**Predicted signature**:
1. Attention pattern divergence (||A_original - A_steered||) at positions > t_steer correlates with Δ_acc
2. The effect is strongest at layers with high attention entropy (more attentional flexibility)
3. Steering at the last token (t = T, no future positions) has ZERO effect — because there are no future queries to be affected by the modified key
4. The effect is proportional to the number of remaining tokens after position t

**Falsification experiment**:
- **Last-token steering**: Steer at the LAST token of the prompt. If C7 is correct, Δ_acc ≈ 0 (no future queries). If other mechanisms are at work, Δ_acc may still be non-zero.
- **Attention divergence measurement**: Before/after steering, compute attention matrices. If C7 is correct, ||ΔA|| (Frobenius norm of attention change) correlates with Δ_acc with ρ > 0.5.
- **K-only vs V-only split**: C7 predicts K-only ≈ full effect (because attention logits depend on K); V-only ≈ 0 (value weights don't affect attention distribution). If V-only has significant effect, C7 is weakened.

---

### C8: RESIDUAL STREAM AMPLIFICATION

**Source**: Implicit in transformer architecture; Agent 5 mechanistic check §2; residual stream analysis

**Core claim**: The residual stream is the sum of all previous layers' outputs. Adding a perturbation at layer l_steer permanently changes the residual stream for ALL downstream layers. This is not amplification per se but CONVECTION — the perturbation is carried forward and affects every subsequent computation.

What makes it nonlinear: while the perturbation is additive (h_L = h_0 + Σ_i output_i + α·v_l), the model's computation uses the full residual stream as input. A small additive change at layer l causes different outputs at layer l+1, which causes a different residual at l+2, etc. The effect compounds multiplicatively with remaining depth.

**Formal statement**:
Let the residual stream at layer L > l be:
    h_L = h_0 + Σ_{i=0}^{L} f_i(h_i)   (where f_i is the i-th layer's contribution)

Under steering at layer l, the residual becomes:
    h'_L = h_L + Δ_L
where Δ_L propagates as:
    Δ_{l+1} = α·v_l   (initial perturbation)
    Δ_{L+1} = Δ_L + (f_L(h_L + Δ_L) - f_L(h_L))   (recursive propagation)

The growth of Δ depends on the Lipschitz constant of the layers. If the transformer computation is contractive (||f_L(x) - f_L(y)|| < ||x - y||), the perturbation decays. If it's expansive, the perturbation grows.

**Predicted signature**:
1. Token divergence grows with remaining depth after steering layer
2. Late-layer steering (L20+) has less effect than early-layer steering (L1-L10) because there are fewer layers to compound through
3. This does NOT explain per-layer selectivity by itself — it explains why effects at L8 (middle of model) are more visible than at L0 (too early, computation not yet structured) or L27 (too late, no layers left to compound)

**Falsification experiment**:
- **Residual stream tracing**: Measure ||Δ_L|| at each downstream layer L > l_steer. If C8 is correct, ||Δ_L|| grows (or at least doesn't shrink) with L. If it decays exponentially, C8 is weakened.
- **Deep vs shallow steering**: Compare steering at L4 (early) vs L12 (mid) vs L24 (late). C8 predicts shallower steering → more compounding → larger effect. If L24 > L12 > L4 (steering later is better), C8 is falsified.
- **Lipschitz measurement**: Compute ||f_L(x) - f_L(y)|| / ||x - y|| for random perturbation directions. If average Lipschitz < 1 across layers, perturbations decay and C8 is wrong.

---

## 3. RELATIONSHIP BETWEEN CANDIDATES

### Inclusion Hierarchy

```
C8 (residual stream) — universal precondition for ANY mechanism
  ├── C1 (K/V amplification) — specific channel: attention mechanism
  │       └── C7 (attention logit perturbation) — reframing of C1, same predictions mostly
  ├── C2 (off-manifold) — competing: geodesic interpretation
  ├── C3 (direction misalignment) — sign-level: subsumed by C6 as special case (θ=π)
  ├── C4 (frequency modulation) — signal-level: operates in frequency domain
  ├── C5 (smoothness exploitation) — most reductive: TT doesn't learn anything causal
  └── C6 (complex α phase) — most general: subsumes C3, adds acceleration dimension
```

C8 is **always true** — the residual stream does carry perturbations forward. The question is whether this is the dominant mechanism or a passive conduit for other mechanisms.

C6 is the most general candidate (subsuming C3). C5 is the most reductive (undermining the entire paradigm). C1 and C7 are nearly identical (C7 emphasizes logit perturbation, C1 emphasizes softmax amplification).

### Orthogonal Dimensions

The 8 candidates vary along independent axes:

| Axis | Candidates | Contradictory Pairs |
|------|-----------|-------------------|
| **Surface of action** | Attention (C1, C7) ↔ Hidden state (C2, C3, C4, C5, C6, C8) | C1/C7 vs C2-C6 |
| **Direction specificity** | Direction-specific (C3, C4, C6) ↔ Direction-agnostic (C2, C5) | C3 vs C5 |
| **Nonlinearity requirement** | Requires nonlinear amplification (C1, C2, C8) ↔ Linear effect (C3, C4, C5, C6) | C1/C2/C8 vs C3/C4/C5/C6 |
| **TT learns something meaningful?** | Yes (C1, C3, C4, C6) ↔ No (C5) | C5 vs all others |
| **Frequency domain relevant?** | Yes (C4, C6) ↔ No (C1, C2, C3, C5, C7, C8) | C4/C6 vs rest |
| **Attention patterns change causally?** | Yes (C1, C7) ↔ No (C2-C6, C8) | C1/C7 vs rest |

---

## 4. PRIOR PROBABILITY ESTIMATES

Based on evidence to date (TSE agents 1-5, complex-α analysis, all experiments run as of 2026-06-14):

| Candidate | Prior | Rationale |
|-----------|-------|-----------|
| C3 (direction misalignment) | 0.22 | Simplest explanation consistent with all observations. L9 sign inversion is the single most testable hypothesis. Requires no new physics. |
| C1 (K/V amplification) | 0.18 | Attention-based explanation is elegant, consistent with 88% token divergence (attention shifts produce large token changes). But requires nonlinearity that hasn't been confirmed. |
| C5 (smoothness exploitation) | 0.16 | If true, explains R² paradox completely. Supported by high R² on both correct AND incorrect trajectories (TT doesn't distinguish quality). BUT: cross-model transfer (SmolLM2→7B) argues against pure smoothness exploitation (smoothness is architecture-dependent). |
| C2 (off-manifold) | 0.14 | Manifold hypothesis is well-supported in transformer literature. Consistent with α > 0.5 collapse. BUT: we have zero manifold measurements in this project. |
| C6 (complex α phase) | 0.11 | Elegant theoretical framework, subsumes C3. BUT: acceleration R² is unknown; may overfit by adding parameters. |
| C8 (residual stream) | 0.07 | Always true in background but rarely dominant. Supported by general transformer theory. BUT: doesn't explain per-layer selectivity (all layers have residual streams). |
| C4 (frequency modulation) | 0.07 | Beautiful prediction about frequency structure. BUT: no frequency analysis has been done yet; current evidence is entirely analogical. |
| C7 (attention logit perturbation) | 0.05 | Nearly identical to C1 but narrower. Predicts last-token steering ≈ 0, which may already be falsified (last-token K/V modification may still affect attention if any future tokens remain — but during generation, each token is generated autoregressively, so the "future" is the remaining generation). |

**Sum**: 1.00

**Caveat**: These priors are correlated. If C1 is true, C7 is approximately 0.9× as true (C7 is a subset of C1). If C6 is true, C3 is approximately 0.5× as true (C6 subsumes C3 but adds additional structure).

---

## 5. PAIRWISE DISAMBIGUATION MATRIX

For each pair (i, j), the minimal experiment that distinguishes them:

| Pair | Distinguishing Experiment | Expected Outcome (i > j) | Expected Outcome (j > i) | Cost |
|------|-------------------------|--------------------------|--------------------------|------|
| C1 vs C2 | K/V split @ L8: K-only vs V-only | K-only ≈ full effect, V-only ≈ 0 | K-only ≈ V-only ≈ full effect × 0.5 | 1 GPU-hr (B4) |
| C1 vs C3 | Attention entropy correlation with Δ_acc | ρ < -0.4 (low entropy → high amplification) | ρ ≈ 0 (sign matters, not amplification) | 1 GPU-hr (B4 + entropy) |
| C1 vs C4 | PCA-filtered steering: steer with top-5 PCA components of v vs full v | PCA-filtered ≈ full (amplification happens anyway) | PCA-filtered preserves L8, IMPROVES L9 (removes noise) | 2 GPU-hrs |
| C1 vs C5 | Random vector steering | random < TT (TT-specific direction matters) | random ≈ TT (any perturbation works) | 0.9 GPU-hrs (in A1) |
| C1 vs C6 | Phase sweep at L9: test θ = π/2 (acceleration-only) | L9(θ=π/2) ≈ 0 (C1 has no acceleration channel) | L9(θ=π/2) > 0 (C6 predicts acceleration steering) | 1 GPU-hr (phase sweep) |
| C1 vs C7 | Last-token steering | last-token-steering ≈ 0 (C7: no future queries) | last-token-steering > 0 (C1: amplification doesn't require future queries specifically) | 0.5 GPU-hrs |
| C1 vs C8 | Lipschitz measurement: compute per-layer contractivity | High Lipschitz → amplification (C1 needs attention amplification) | Low Lipschitz → perturbation decays (C8 says residual carries it regardless) | 1 GPU-hr |
| C2 vs C3 | Intrinsic dimension d_95 vs sign(l): compute both | d_95 predicts Δ_acc (C2: manifold geometry) | sign(l) predicts Δ_acc (C3: directional alignment) | 0 GPU-hrs (zero-cost) |
| C2 vs C4 | sin(θ) vs ω_centroid: which predicts Δ_acc better? | sin(θ) (C2: off-manifold angle) | ω_centroid (C4: frequency centroid) | 0 GPU-hrs (zero-cost, existing data) |
| C2 vs C5 | Off-manifold distance vs random-baseline R² | Off-manifold distance correlates with Δ_acc | Random baseline R² correlates with Δ_acc (C5: smoothness) | 0.5 GPU-hrs |
| C2 vs C6 | Acceleration-only steering at death layers: does a[l] produce positive effect? | No (C2: any off-manifold perturbation harms) | Yes (C6: acceleration is the right direction for some layers) | 1 GPU-hr |
| C3 vs C4 | Cosine similarity cos(v_correct, v_incorrect) vs ω_centroid: correlation with Δ_acc | cos_sim(l) predicts Δ_acc (C3: direction alignment) | ω_centroid(l) predicts Δ_acc (C4: frequency) | 0 GPU-hrs (zero-cost) |
| C3 vs C5 | Signed α sweep at L9: does -α produce +Δ_acc? | Yes (C3: sign inversion) | No (C5: no direction matters; random ≈ TT) | 2 GPU-hrs (B1 partial) |
| C3 vs C6 | Phase sweep at L9: is peak at θ=π (C3) or θ=π/2 (C6)? | θ=π (C3: simple sign inversion) | θ=π/2 (C6: acceleration phase) | 1 GPU-hr (phase sweep) |
| C4 vs C5 | PCA-filtered steering on sub-threshold model (Math-1.5B) | Frequency structure exists even on weak models (C4) | No structure; any vector works equally (C5) | 2 GPU-hrs |
| C4 vs C6 | Train TT on acceleration a[l]; compute R²_acceleration | R²_acceleration >> 0 (C4: frequency of velocity and acceleration are coupled) | R²_acceleration ≈ 0 (C6: acceleration is independent, harder to predict) | 0.5 GPU-hrs |
| C5 vs C6 | Random vector steering at L9 with -α: does -random work? | Yes (C5: any perturbation, sign doesn't matter) | No (C6: specific phase relationship needed for L9) | 0.5 GPU-hrs |
| C5 vs C7 | Last-token steering with random vector | random ≈ TT even at last token (C5: any perturbation works anywhere) | random ≈ 0 at last token (C7: attention mechanism needs future queries) | 0.5 GPU-hrs |
| C6 vs C7 | Phase sweep with K-only vs full steering | K-only ≈ full for all phases (C7: attention mediates everything) | Phase-dependent: K-only fails for acceleration phases (C6: hidden state phase is real, not just attention) | 1 GPU-hr |
| C6 vs C8 | Acceleration-only steering at early layer (L4) vs late layer (L24): does depth amplify acceleration effect? | Depth amplifies (C8: more compounding) | Depth doesn't amplify (C6: phase is intrinsic to layer, not depth-dependent) | 1 GPU-hr |

---

## 6. DISAMBIGUATION PROTOCOL: DISTINGUISH ALL 8

### Strategy

The protocol exploits the orthogonal dimensions identified in §3 to prune the candidate set in stages:

1. **Prune C5 first** (cheapest, highest impact — if TT is smoothness predictor, paradigm is trivial)
2. **Prune C1 vs C7 vs C2** (attention vs hidden-state surface)
3. **Prune C3 vs C6** (simple sign vs phase space)
4. **Prune C4 vs C8** (frequency vs residual propagation)

### Experiment Set (Total: 4.5 GPU-hours)

#### E1: Random Baseline → Prune C5 (0.9 GPU-hrs, in A1)

**Already in Phase A1 protocol.**

- Run random vectors (same norm as TT predictions) across all 28 layers
- Compare Δ_acc for random vs TT vs baseline
- **If random ≈ TT (>80% same pattern)**: C5 is the dominant mechanism. C1, C3, C6, C7 all compete for residual variance but smoothness exploitation accounts for the bulk. **Stop** — publish as "steering is indistinguishable from random perturbation" (valuable negative result).
- **If random < TT (<30% same pattern)**: C5 is ruled out. Proceed.

#### E2: K/V Split + Last-Token + Attention Divergence → Prune C1, C2, C7 (2 GPU-hrs)

Run three sub-experiments on L8 (the layer with strongest signal):

**E2a: K-only vs V-only steering (1 GPU-hr)**
- Steer at L8 with: (a) K only, (b) V only, (c) both (standard)
- Measure Δ_acc and attention divergence ||ΔA|| for each
- **If K-only ≈ full and V-only ≈ 0**: C7 is supported (attention logit perturbation is the channel)
- **If K-only ≈ V-only ≈ full × 0.5**: C2 is supported (hidden state changes propagate through both channels equally)
- **If K-only ≈ V-only ≈ full**: C1 is supported (both channels amplify through softmax)

**E2b: Last-token steering (0.5 GPU-hrs)**
- Steer at the FINAL token of the prompt (position T)
- Measure Δ_acc
- **If Δ_acc ≈ 0**: C7 supported (no future queries)
- **If Δ_acc ≈ standard steering**: C2 or C1 supported (hidden state changes propagate regardless)

**E2c: Attention entropy × Δ_acc correlation (0.5 GPU-hrs, data from E2a)**
- Compute pre-steering attention entropy at each layer
- Correlate with Δ_acc
- **If ρ < -0.4**: C1/C7 supported (low entropy = sharp attention = more amplification)
- **If ρ ≈ 0**: C2 supported (entropy doesn't matter for off-manifold perturbation)

#### E3: Signed α Sweep + Phase Sweep → Prune C3, C6 (1.5 GPU-hrs)

**E3a: Signed α sweep at L9 (0.5 GPU-hrs)**
- Test α ∈ {-2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2} at L9
- **If L9(-0.1) ≈ L8(+0.1) ≈ +20pp**: C3 supported (simple sign inversion)
- **If L9(-α) ≈ L9(+α) ≈ -23pp**: C3 is weakened; sign alone doesn't fix L9
- **If L9(-α) ≈ 0 (neither helps nor harms)**: C3 weakened; L9 has deeper pathology

**E3b: Phase sweep at L9 (1 GPU-hr)**
- Test δh = r·(cos(θ)·v + sin(θ)·a) at L9 with fixed r = 0.1, θ ∈ {0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4}
- Requires computing acceleration a = v[l+1] - v[l] from TT predictions
- **If peak at θ = π**: C3 supported (simple sign inversion: L9 needs -α)
- **If peak at θ = π/2**: C6 supported (L9 needs acceleration steering)
- **If peak at θ = π/4**: C6 supported with mixed phase (needs both velocity and acceleration)
- **If all θ produce Δ_acc < 0**: Neither C3 nor C6 is sufficient; L9 may be a true death layer

#### E4: PCA-Filtered + Frequency Centroid → Prune C4, C8 (0.5 GPU-hrs, mostly zero-cost)

**E4a: Frequency centroid correlation (0 GPU-hrs, zero-cost)**
- From existing trajectory data: compute ω_centroid(l) for each layer
- Correlate with Δ_acc(l) from A1
- **If |ρ| > 0.4**: C4 supported (frequency structure predicts steering efficacy)
- **If |ρ| < 0.2**: C4 weakened

**E4b: PCA-filtered steering at L8 and L9 (0.5 GPU-hrs)**
- Compute PCA of v across all layers
- Steer using only top-5 PCA components (low-pass filtered v)
- Compare Δ_acc for L8(full) vs L8(filtered) and L9(full) vs L9(filtered)
- **If L8(filtered) ≈ L8(full) AND L9(filtered) > L9(full)**: C4 confirmed (frequency filtering improves death layers)
- **If L8(filtered) < L8(full)**: C8 may be dominant (low-pass filtered v has less residual-stream impact because components cancel)
- **If L8(filtered) ≈ L8(full) AND L9(filtered) ≈ L9(full)**: C4 weakened; frequency structure not relevant

### Decision Tree

```
START (E1: Random baseline)
  │
  ├── random ≈ TT → C5 CONFIRMED → STOP: paradigm invalidated
  │                              → Publish negative result
  │
  └── random << TT → C5 RULED OUT → E2 (K/V split + last-token)
        │
        ├── E2a: K-only ≈ full, V-only ≈ 0 → C7 SUPPORTED
        │   └── E2b: last-token ≈ 0 → C7 CONFIRMED
        │   └── E2b: last-token > 0 → C7 partial; C1 also present (attention amplification)
        │
        ├── E2a: K-only ≈ V-only ≈ full × 0.5 → C1 SUPPORTED (both channels amplify)
        │   └── E2c: ρ < -0.4 → C1 CONFIRMED (entropy correlates)
        │   └── E2c: ρ ≈ 0 → C1 active but C2 also present (mixed mechanism)
        │
        ├── E2a: K-only ≈ V-only ≈ full → C2 SUPPORTED (both channels equally effective)
        │   └── E2b: last-token ≈ standard → C2 CONFIRMED
        │   └── E2b: last-token ≈ 0 → C2 partial (requires future context)
        │
        └── → E3 (Phase sweep at L9)
              │
              ├── E3a: L9(-0.1) ≈ +20pp → C3 CONFIRMED (sign inversion)
              │   └── E3b: peak at θ = π → C3 CONFIRMED conclusively
              │   └── E3b: peak at θ = π/2 → C6 present alongside C3 (mixed)
              │
              ├── E3a: L9(-α) ≈ 0 → neither C3 nor C6 dominates
              │   └── E3b: peak at θ = π/2 → C6 CONFIRMED (acceleration phase needed)
              │   └── E3b: no peak → L9 is true death layer (deeper mechanism)
              │
              └── E3a: L9(-α) ≈ -23pp → sign alone insufficient
                  └── E3b: peak at π/2 → C6 CONFIRMED (but L9 still harmful even with acceleration? unusual)
                  └── E3b: no peak → C3 + C6 both weakened; L9 has separate mechanism
              
              └── → E4 (Frequency + Residual analysis)
                    │
                    ├── E4a: |ρ| > 0.4 → C4 SUPPORTED
                    │   └── E4b: L9(filtered) > L9(full) → C4 CONFIRMED
                    │   └── E4b: L9(filtered) ≈ L9(full) → C4 partial (correlation not causal)
                    │
                    └── E4a: |ρ| < 0.2 → C4 WEAKENED
                        └── → C8 (residual stream) is the remaining default
                            └── Requires Lipschitz measurement: if contractive < 1 → C8 WEAKENED
                            └── If expansive > 1 → C8 ACTIVE (but is it dominant?)
```

### Maximum Complexity Case

If all candidates remain active (no single mechanism accounts for >50% of variance), the final synthesis is:

**Steering is driven by multiple simultaneous mechanisms**:
- C8 (residual stream) is the universal passive conduit
- C1/C7 (attention amplification) mediates the dominant effect at layers with low attention entropy
- C6 (phase space) explains sign-dependent effects — optimal steering is complex (r, θ)
- C4 (frequency modulation) explains why some layers respond better to low-pass filtered steering
- C2 (off-manifold) explains the α > 0.5 collapse boundary
- C3 (direction misalignment) is subsumed by C6 as the θ = π special case

In this scenario, the practical recommendation is: **steer using C6 formulation (complex α) with C7 surface (K-only modification) filtered through C4 lens (frequency-domain PCA)**, monitored via C8 (residual stream divergence) and bounded by C2 (manifold thickness).

This is the complete mechanistic theory. It predicts that optimal steering is:
- **Surface**: K-only (not V, not K+V)
- **Direction**: Complex phase θ per layer (not just α magnitude)
- **Filtering**: Low-pass filtered velocity (remove high-frequency noise)
- **Magnitude**: Constrained by manifold thickness (α < α_critical ≈ 0.5)
- **Position**: Applied at layers with low attention entropy (sharp distributions amplify signal)

---

## 7. MINIMAL DISAMBIGUATION PROTOCOL (MVP)

If only one experiment can be run with limited compute, this sequence of 4 experiments (in dependency order) distinguishes ALL 8 candidates:

### Protocol

| Step | Experiment | Compute | Distinguishes | Cumulative |
|------|-----------|---------|---------------|------------|
| 1 | Random baseline (A1) | 0.9 GPU-hrs | C5 vs everyone | 0.9 GPU-hrs |
| 2 | K/V split + last-token (B4 extended) | 1.0 GPU-hrs | C1 vs C2 vs C7 | 1.9 GPU-hrs |
| 3 | Phase sweep at L9: θ ∈ {0, π/2, π} | 0.5 GPU-hrs | C3 vs C6 | 2.4 GPU-hrs |
| 4 | PCA-filtered steering + ω_centroid (R1+E4) | 0.5 GPU-hrs | C4 vs C8 | 2.9 GPU-hrs |

**Total**: 2.9 GPU-hours — distinguishes all 8 candidates.

### After Each Step, Stop If:

- **After Step 1**: random ≈ TT → C5 wins. Paradigm invalidated. Total compute: 0.9 GPU-hrs. **Action**: Publish negative results.
- **After Step 2**: K-only ≈ 0, V-only ≈ 0 → C1, C2, C7 all weakened. Attention/hidden-state surface is not the main channel. Go to Step 3-4 to find the real mechanism.
- **After Step 3**: L9(θ=π) ≈ +20pp → C3 wins. Simple sign inversion explains everything. **Action**: Publish signed steering protocol.
- **After Step 4**: ω_centroid(l) predicts Δ_acc with ρ > 0.6 → C4 wins. Frequency structure explains selectivity. **Action**: Publish frequency-domain theory.

### If All 4 Steps Are Necessary:

The full 8-mechanism synthesis (§6 maximal complexity) is supported. **Action**: Publish comprehensive mechanistic theory of KV-cache steering with implications for attention modification, phase-aware perturbation, and manifold-constrained optimization.

---

## 8. META-ANALYSIS: WHAT THE DIFFUSION REVEALS

### The Mechanism Gap is Structural, Not Empirical

The original finding (NO analysis explains WHY steering works) is not a failure of individual analyses — it's a structural consequence of the experimental design. No single experiment run to date distinguishes between the 8 candidates because ALL experiments used the same protocol (α·v at KV-cache, layer sweep at α=0.1, GSM8K evaluation). The mechanism is underdetermined by 8:1.

### C5 is the Critical Gate

The single most important distinction is C5 (smoothness exploitation) vs all others. If C5 is true, steering is trivial — any perturbation works, and the TT's role is incidental. If C5 is false, steering is real and the remaining 7 candidates describe its mechanism. The random baseline experiment (0.9 GPU-hrs) is therefore the highest-value experiment in the entire project.

### C6 is the Most General Resolution

If steering is real (C5 false), C6 (complex α phase) is the most general candidate that subsumes C3, C7, and partially C1. The acceleration dimension provides the degrees of freedom needed to explain both trim-tab layers (θ ≈ 0) and death layers (θ ≈ π/2). The frequency dimension (C4) may offer a second complementary resolution.

### The Residual Stream (C8) is the Silent Partner

C8 is always present but rarely sufficient. It explains propagation but not selectivity. Its role is to amplify whatever mechanism is active at the steering layer. This is why L8 (middle of model, 8 layers of compounding remaining) shows stronger effects than L2 (early, less structure) or L24 (late, few layers remaining).

### Attention is the Likely Primary Surface

C1/C7 (attention-based mechanisms) have the strongest theoretical support: the KV-cache IS the attention surface, and attention is the most sensitive operation in the transformer. The K/V split experiment will likely confirm that attention modifications dominate hidden-state modifications. If it does, the correct framing is "KV-cache steering is an attention-modulation technique, not a hidden-state perturbation technique" — a fundamental reframing with implications for generalization.
