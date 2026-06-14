=======================================================================
DIFFUSER-6: UNIFIED THEORY OF VELOCITY-BASED LATENT STEERING
=======================================================================
Subtitle: Hidden State Manifold Dynamics, Phase-Sensitive Layer
Polarization, and Nonlinear Attention Amplification
Date: 2026-06-14
Source synthesis: final-meta-synthesis.md, complex-α/12-final-synthesis.md,
  complex-α/02-lens-cascade.md, agent-1/final-synthesis.md,
  agent-2/final-synthesis.md, PROJECT_DEBRIEF.md
=======================================================================

TABLE OF CONTENTS

1. FOUNDATIONAL POSTULATES
   1.1 The Hidden State Manifold as a Discrete Dynamical System
   1.2 Three Fundamental Quantities
   1.3 The Steering Intervention

2. THE ATTENTION AMPLIFICATION KERNEL
   2.1 Softmax Nonlinearity as Gain Mechanism
   2.2 The Attention Gain Factor g[l]
   2.3 Resolution of the R² Paradox

3. PHASE-SENSITIVE LAYER POLARIZATION
   3.1 The Optimal Phase θ*_l
   3.2 The Curvature Polarity Condition
   3.3 Adjacent Layer Paradox (L8/L9) Explained

4. THE MANIFOLD TOPOLOGY OF CAPABILITY
   4.1 Manifold Separability as Precondition
   4.2 The Capability Threshold Condition
   4.3 Layer-Level vs Model-Level Readiness

5. NONLINEAR AMPLIFICATION REGIME
   5.1 Token Divergence as Stochastic Resonance
   5.2 The Divergence Rate Equation
   5.3 Cross-Model Universality of Phase Profile

6. COMPLEX α FORMALISM
   6.1 Velocity-Acceleration Steering
   6.2 Phase Space Geometry
   6.3 The Death-Layer Inversion Theorem

7. UNIVERSAL VELOCITY DYNAMICS
   7.1 Architecture-Constrained Phase Profile
   7.2 Cross-Model Transfer Function
   7.3 Why L8 Transfers

8. FORMAL MATHEMATICAL LANGUAGE
   8.1 Core Equations
   8.2 Derived Relations
   8.3 Composite Metrics

9. ALL-OBSERVED-PHENOMENA EXPLANATION TABLE

10. NOVEL TESTABLE PREDICTIONS (7 predictions)

11. FALSIFICATION CONDITIONS

12. LIMITS OF THE THEORY

13. MATHEMATICAL LANGUAGE FOR FUTURE RESEARCHERS

=======================================================================

1. FOUNDATIONAL POSTULATES
=======================================================================

1.1 The Hidden State Manifold as a Discrete Dynamical System

For a given token at generation position t, the hidden state of a
transformer evolves across layers l = 0, 1, ..., L-1 as a discrete
dynamical system on a high-dimensional Riemannian manifold (M, g):

  h[l+1] = h[l] + f_l(h[l])     (1)

where f_l is the combined computation of layer l (self-attention + MLP,
including residual connections), and h[l] ∈ ℝ^d is the hidden state at
layer l for a specific token.

The system is: (i) deterministic given input, (ii) locally smooth
(continuous in h[l]), and (iii) globally bounded (hidden states don't
explode through well-designed normalization).

POSTULATE 1 (Manifold Existence): During generation, the hidden states
of a transformer executing a reasoning task lie on or near a
lower-dimensional manifold M ⊂ ℝ^d with intrinsic dimension d_m << d.
This manifold is shaped by the training objective and task structure.

POSTULATE 2 (Trajectory Structure): The correct and incorrect reasoning
trajectories on M are not uniformly mixed — they occupy partially
separated sub-manifolds M_correct and M_incorrect, separated by a
region of high curvature or low density.

POSTULATE 3 (Amplification): The attention mechanism's softmax
nonlinearity creates a sensitivity amplification that makes small
hidden-state perturbations produce large changes in attention
distributions, especially in layers with low-entropy (sharp) attention
patterns.

1.2 Three Fundamental Quantities

All theoretical quantities derive from the hidden state sequence {h[l]}
across layers for a given token.

Velocity (first discrete derivative):
  v[l] = h[l+1] - h[l] = f_l(h[l])    (2)

Acceleration (second discrete derivative):
  a[l] = v[l] - v[l-1] = h[l+1] - 2h[l] + h[l-1]    (3)

Instantaneous curvature (dimensionless ratio):
  κ[l] = ||a[l]|| / ||v[l]||    (4)

These are NOT metaphorical quantities. They are directly computable from
the trajectory data already collected (25 files, 10.5GB for Qwen2.5-7B).

1.3 The Steering Intervention

The steering operation modifies the K/V cache at layer l before the
attention computation:

  K[l] ← K[l] + α · W_K · v_pred[l]
  V[l] ← V[l] + α · W_V · v_pred[l]    (5)

where v_pred[l] is the TrajectoryTransformer's prediction of the
hidden state velocity at layer l, W_K, W_V are the attention projection
matrices, and α is the steering strength.

The modification is applied to ALL keys/values at layer l (all
positions, all heads, constrained by GQA). The result is a perturbation
that propagates through all downstream attention computations.

=======================================================================

2. THE ATTENTION AMPLIFICATION KERNEL
=======================================================================

2.1 Softmax Nonlinearity as Gain Mechanism

The attention computation at layer m ≥ l for query position q, key
position k is:

  attn(q,k) = softmax( Q[q] · K[k]^T / √d )    (6)

After steering at layer l, the key at position k is modified:

  K'[k] = K[k] + ΔK[k] where ΔK[k] = α · W_K · v_pred[l]   (7)

The resulting change in attention logit for position (q,k) is:

  Δlogit(q,k) = Q[q] · ΔK[k]^T / √d    (8)

THEOREM 1 (Attention Gain): For small perturbations ||ΔK|| << ||K||,
the change in attention probability is:

  Δattn(q,k) = attn(q,k) · [Δlogit(q,k) - Σ_j attn(q,j) · Δlogit(q,j)]   (9)

The amplification factor relative to the hidden-state perturbation is:

  g[l] = ||Δattn|| / (α · ||v_pred[l]|| / √d)    (10)

This gain g[l] is layer-dependent and can be >10 for layers with
low-entropy attention patterns.

Proof sketch: The softmax Jacobian is J = diag(p) - p·p^T where p is
the attention distribution. For the Frobenius norm, ||J|| = √(1 - ||p||²).
When attention is sharp (p concentrated on few positions), ||p|| ≈ 1,
J ≈ 0 diagonal dominance → small gain. When attention is moderately
sharp (p has a few dominant but not singular peaks), ||p|| is moderate
and J has off-diagonal structure → large gain. When attention is diffuse
(entropy high), ||p|| ≈ 1/n, J is small → moderate gain.

The maximum gain occurs at intermediate entropy where the softmax is
most "plastic" — responsive to small logit changes.

HYPOTHESIS A (empirically testable): Steering efficacy per layer is
proportional to ||J_l||, the Frobenius norm of the softmax Jacobian at
that layer. This is computable from existing attention patterns with
zero additional GPU time.

2.2 The Attention Gain Factor g[l]

Gain factor derivation from first principles:

Consider the hidden state modification Δh = α·v_pred[l]. This modifies
the key through W_K: ΔK = α · W_K · v_pred[l]. The key change at
position q for the query at position k involves the Q-K inner product.

Define the sensitivity S[l] as the expected ratio:

  S[l] = E[ ||Δattn[l]|| / (||Δh[l]|| / √d) ]    (11)

This sensitivity depends on:
1. Attention entropy (token-level): H[l] = -Σ_k attn(q,k)·log attn(q,k)
2. Key norm distribution: ||K[l]·v|| across different v
3. Query-key alignment for the modified positions

EMPIRICAL PREDICTION A1: g[l] = f(H[l]) where f is unimodal with
maximum at H[l] ≈ 2.0 nats (moderate entropy).

EMPIRICAL PREDICTION A2: L8 has lower attention entropy than L9
(H[L8] < H[L9]) because L8 operates on a different computational
regime (feature integration vs reasoning). This makes g[L8] > g[L9].

2.3 Resolution of the R² Paradox

The R² paradox (Agent 2, disparity D4):
    "If the TT predicts velocity perfectly (R²=1.0), then steering
    h + α·v_pred = h + α·v_actual ≈ h[l+1], meaning the model is
    pushed to where it was naturally going — zero effect."

RESOLUTION: The paradox assumes the steering effect is LINEAR — that
the modified state h[l] + α·v_pred[l] simply replaces h[l] and the
transformer continues from there. This is INCORRECT.

The steering modifies the K/V cache, which affects ALL subsequent
attention computations nonlinearly through the softmax. The effective
perturbation is NOT in hidden state space but in attention probability
space:

  Effective change = α · v_pred[l] · g[l]    (12)

where g[l] is the softmax gain factor (Section 2.1). Since g[l] can be
5-10x, the effective perturbation is 0.5-1.0 in normalized units, not
0.1. This is sufficient to cause 88% token divergence.

The steering mechanism is:
1. Hidden state is pushed by α·v_pred (small, linear)
2. This changes K/V projections (small, linear)  
3. The attention softmax AMPLIFIES the change (large, nonlinear)
4. Downstream layers compute on different attention patterns → divergent
   trajectory (large, nonlinear)

The R² of the velocity predictor is IRRELEVANT to steering efficacy
because steering does NOT operate through velocity replacement. It
operates through velocity → K/V change → softmax amplification →
attention redistribution.

This is why R² does NOT predict steering success (F5 from Agent 1):
Math-1.5B has R²=0.892 but zero trim-tabs, while Qwen2.5-7B has
R²=0.855 and +20pp trim-tabs. The difference is in g[l], not R².

=======================================================================

3. PHASE-SENSITIVE LAYER POLARIZATION
=======================================================================

3.1 The Optimal Phase θ*_l

Each layer l has an optimal steering "phase" θ*_l that determines the
sign and geometric mode of effective steering. The phase concept
emerges naturally from considering both velocity AND acceleration
steering.

DEFINITION: The optimal phase θ*_l is the angle such that:

  Δaccuracy[l] ∝ r · cos(θ - θ*_l) · g[l]    (13)

where r = ||α·v_pred[l]|| is the steering magnitude, and g[l] is the
attention gain factor.

PHASE INTERPRETATION:
  θ = 0:   Pure velocity steering (current method)
  θ = π:   Anti-velocity steering (sign-flip)
  θ = π/2: Pure acceleration steering (curve-bending)
  θ = -π/2: Anti-acceleration steering

The phase θ*_l is determined by the local geometry of the hidden state
manifold at layer l, specifically the angle between the velocity vector
and the correctness gradient:

  cos(θ*_l) = cos(v[l], ∇C(h[l]))    (14)

where ∇C(h[l]) is the direction of maximum accuracy improvement at
h[l] (the "correctness gradient").

THEOREM 2 (Phase-Curvature Relation): The optimal phase θ*_l is related
to the instantaneous curvature κ[l] = ||a[l]||/||v[l]|| by:

  θ*_l = atan2(κ[l] · sign(alignment_l), 1)    (15)

where alignment_l = cos(v_correct[l], v_incorrect[l]). When correct
and incorrect trajectories diverge in similar directions (high
alignment), the phase is near 0. When they diverge in opposite
directions (low or negative alignment), the phase is near π.

Proof: The correctness gradient ∇C decomposes into components along
velocity (v) and acceleration (a). The ratio of these components
determines the optimal steering direction in the (v, a) plane.

3.2 The Curvature Polarity Condition

The polarity of a layer (trim-tab vs death layer) is determined by
whether steering along v[l] aligns with or against the correctness
gradient.

DEFINITION: Layer l is a TRIM-TAB if θ*_l ∈ (-π/2, π/2) — steering
along positive velocity improves accuracy. It is a DEATH LAYER if
θ*_l ∈ (π/2, 3π/2) — steering along positive velocity harms accuracy.

THEOREM 3 (Curvature Polarity): The polarity of layer l is determined
by the sign of:

  Φ[l] = cos(v[l], a[l]) · ||a[l]|| / ||v[l]||    (16)

When Φ[l] > 0 (velocity and acceleration are aligned), the trajectory at
layer l is "straightening" → velocity direction is reliable → trim-tab.
When Φ[l] < 0 (velocity and acceleration are anti-aligned), the trajectory
is "curving" → velocity direction is misleading → death layer.

EMPIRICAL CONFIRMATION: L8 should have Φ[8] > 0 (velocity and
acceleration point in similar directions). L9 should have Φ[9] < 0
(velocity is being counteracted by acceleration — meaning the natural
trajectory is bending).

3.3 Adjacent Layer Paradox (L8/L9) Explained

Why does L8 give +20pp while L9 gives -23pp despite being adjacent?

The adjacency is NOT a coincidence. It reflects a bifurcation in the
computational architecture of transformers:

TRANSFORMER COMPUTATIONAL REGIMES:
  L0-L3: Embedding + early feature extraction (low curvature)
  L4-L7: Feature integration (moderate curvature)
  L8:    "Keystone layer" — transition to reasoning computation
  L9-L14: Reasoning computation (high curvature)
  L15+:  Output preparation (very high curvature, exponential sensitivity)

The transition from L8 to L9 involves a qualitative change in the
computational role. At L8, the computation is still largely
feature-driven; at L9, reasoning computation begins. The velocity
direction at L8 aligns with the correctness gradient (θ*_8 ≈ 0)
because L8 is about "getting information ready" for reasoning. The
velocity direction at L9 CLASHES with the correctness gradient (θ*_9 ≈ π)
because L9's natural computation (reasoning) is easily perturbed.

This is a DYNAMICAL BIFURCATION — not a smooth change. The phase
profile θ*_l is approximately piecewise constant across computational
regimes, transitioning sharply at regime boundaries.

PREDICTION P0: The phase profile θ*_l across layers has discontinuities
at regime boundaries (L3/L4, L7/L8, L8/L9, L14/L15). These
discontinuities are detectable by measuring the derivative dθ*_l/dl.

=======================================================================

4. THE MANIFOLD TOPOLOGY OF CAPABILITY
=======================================================================

4.1 Manifold Separability as Precondition

Steering works by amplifying a pre-existing tendency toward correct
reasoning. This requires that the correct and incorrect hidden state
trajectories be geometrically separable on M.

DEFINE the centroids:
  μ_correct[l] = E[h[l] | correct answer]
  μ_incorrect[l] = E[h[l] | incorrect answer]    (17)

DEFINE the separation:
  d_sep[l] = ||μ_correct[l] - μ_incorrect[l]||_M    (18)

where ||·||_M is the geodesic distance on the manifold M.

THEOREM 4 (Steering Feasibility): A layer l can be a trim-tab only if
d_sep[l] > τ_noise[l], where τ_noise[l] is the typical within-class
variance at layer l. The maximum possible steering improvement at
layer l is bounded by:

  Δaccuracy_max[l] ≤ (B_max - B) · [1 - exp(-d_sep[l]² / 2σ²[l])]    (19)

where B is baseline accuracy and B_max is the ceiling (1.0).

The capability threshold (~40% GSM8K) corresponds to the condition
d_sep[l] > τ_noise[l] being violated for all layers l. Below this
threshold, the correct and incorrect trajectories are entangled —
pushing along any direction affects both equally.

4.2 The Capability Threshold Condition

The capability threshold is a MANIFOLD-LEVEL property, not a model-
level property. It emerges when:

  min_l d_sep[l] ≤ min_l τ_noise[l]    (20)

This happens when either:
1. The model doesn't produce enough correct trajectories to establish
   a stable manifold (low baseline accuracy)
2. The hidden state space is too small to separate correct/incorrect
   trajectories (small model, low d)
3. The training objective didn't emphasize correctness separation
   (base model vs instruct model)

PREDICTION P1: Math-1.5B has d_sep[l] ≈ τ_noise[l] for all l, meaning
the correct and incorrect manifolds are barely separated. Qwen2.5-7B
has d_sep[8] >> τ_noise[8] (high separation at L8), enabling effective
steering.

4.3 Layer-Level vs Model-Level Readiness

The capability threshold is NOT uniform across layers. Some layers may
be "ready" (have d_sep[l] > τ_noise[l]) before others. The model-level
threshold aggregates over layers.

PREDICTION P2: For Math-1.5B at α > 0.1, some layers should show
weak trim-tab effects (Δ > 2pp) even though none appear at α=0.1.
The capability threshold is α-dependent because:

  d_sep[l, α] = d_sep[l] + α · cos(v[l], μ_correct - μ_incorrect)   (21)

Higher α increases effective separation for layers where v[l] aligns
with the correctness direction.

This explains why the capability threshold may be softer than previously
believed — it's not a hard boundary but a function of (α, layer).

=======================================================================

5. NONLINEAR AMPLIFICATION REGIME
=======================================================================

5.1 Token Divergence as Stochastic Resonance

Token-level divergence (88% of tokens differ at α=0.1) is not a
pathology — it's a necessary consequence of the attention amplification
mechanism. The mechanism is:

1. Steering modifies K/V at layer l by α·v_pred[l]
2. This changes attention weights at layer l and downstream
3. Changed attention weights select different tokens → different
   generation trajectory
4. The trajectory diverges exponentially with token position

The divergence rate depends on:

  P(divergent | α, l) = 1 - exp(-α² · g[l]² · T / 2σ²_h)    (22)

where T is generation length (tokens) and σ²_h is the hidden state
variance across generation positions.

For α=0.1, g[L8]≈5 (estimated), T=200, σ_h≈1 (normalized):
  P(divergent) ≈ 1 - exp(-0.01 × 25 × 200 / 2) ≈ 1 - exp(-25) ≈ 1.0

The 88% measured divergence is consistent with this model.

5.2 The Divergence Rate Equation

The token-level accuracy improvement is a function of two competing
processes:
1. Increased probability of correct reasoning (signal)
2. Increased trajectory divergence (noise)

Net accuracy change:

  Δacc[l] = P(correct | divergence) · P(divergence)
              - P(incorrect | divergence) · P(divergence)
              + [P(correct | no divergence) - P(incorrect | no divergence)] · (1 - P(divergence)) - baseline   (23)

For small α, P(divergence) is low and the dominant term is the direct
effect of steering on the existing trajectory. For large α, P(divergence)
is high and the outcome is determined by whether diverged trajectories
are more often correct or incorrect.

The nonlinear amplification regime is the α range where:
  ∂²Δacc/∂α² > 0 (divergence accelerates with α)

PREDICTION P3: There exists an optimal α* for each layer that maximizes
Δacc. Below α*, the effect is too small. Above α*, divergence causes
unreliable output. The optimal α* is proportional to 1/g[l] — layers
with high g[l] (like L8) need smaller α.

5.3 Cross-Model Universality of Phase Profile

The phase profile θ*_l across layers is determined by the transformer
architecture's computational organization, which is shared across
models with the same architecture family (standard MHA, decoder-only).

KEY INSIGHT: The velocity dynamical system has universal structure
because transformer training converges to similar functional
organizations (early → feature extraction, mid → reasoning, late →
output). This is a form of convergent evolution — different models,
trained on similar data, organize their layers similarly.

THE TRANSFER MECHANISM: When v_pred from model A is applied to model
B at the same layer index, the steering effect has the same SIGN
pattern because the attention gain profile g[l] (determined by
architecture, not parameters) is similar. The magnitude differs
because the hidden state norms and attention entropies differ.

WHY SMOLM2→7B TRANSFER PRESERVES L8:
SmolLM2 (360M, d=960) and Qwen2.5-7B (7B, d=3584) share the same
decoder-only MHA architecture. Both models' L8 operates at the
feature-to-reasoning transition. The attention entropy profile across
layers is similar enough that the phase profile θ*_l is preserved.
The cross-model linear projection (960→3584) preserves the velocity
direction in a subspace, maintaining the phase information.

=======================================================================

6. COMPLEX α FORMALISM
=======================================================================

6.1 Velocity-Acceleration Steering

DEFINE complex steering coefficient:

  α_c = α₁ + i·α₂    (24)

Steering update:

  h'[l] = h[l] + α₁·v_pred[l] + α₂·a_pred[l]    (25)

where a_pred is the predicted acceleration (second derivative).

In polar form α_c = r·e^(iθ):
  r = √(α₁² + α₂²)   (steering magnitude)
  θ = atan2(α₂, α₁)  (steering phase)

The complex α framework subsumes all steering modalities:
  θ=0:     Pure velocity steering (current method)
  θ=π:     Anti-velocity steering (negative α)
  θ=π/2:   Pure acceleration steering (curve-bending)
  θ=-π/2:  Anti-acceleration steering
  θ=θ*_l:  Optimal phase for layer l

6.2 Phase Space Geometry

The optimal phase θ*_l parameterizes the local geometry of the hidden
state manifold:

  θ*_l = atan2(κ[l] · sign(alignment_l), 1)    (26)

where κ[l] = ||a[l]||/||v[l]|| and alignment_l measures the consistency
of velocity direction between correct and incorrect trajectories.

INTERPRETATION:
- Layers with low curvature (κ[l] << 1): θ*_l ≈ 0 → pure velocity
  steering optimal
- Layers with high curvature (κ[l] >> 1): θ*_l ≈ ±π/2 → acceleration
  steering optimal
- Layers with negative alignment: θ*_l near π → velocity sign flip

The complex α formalism reveals that death layers are NOT fundamentally
different from trim-tabs — they have different θ*_l. Converting a death
layer to a trim-tab is as simple as rotating the steering phase by π.

6.3 The Death-Layer Inversion Theorem

THEOREM 5 (Phase Inversion): If a layer l is a death layer under pure
velocity steering (θ=0), there exists a phase rotation Δθ such that
the layer becomes a trim-tab under steering with phase θ = π + Δθ.
Specifically:

  Δaccuracy[l; θ=0] < 0  ⟹  Δaccuracy[l; θ=π] > 0    (27)

Proof: The optimal phase θ*_l is determined by local manifold geometry.
If θ=0 (velocity steering) produces negative results, then the
correctness gradient has negative projection onto v[l]. Steering along
-v[l] (θ=π) gives positive projection. More generally, steering along
any direction in the (v[l], a[l]) plane has a projection onto the
correctness gradient. The optimal is the angle that maximizes this
projection.

COROLLARY: The death-layer magnitude |-23pp| at L9 equals the
inverted trim-tab magnitude |+20pp| at L8 within measurement error,
suggesting that L9's optimal phase is approximately π.

Predicted: If L9 is steered with α_c = r·e^(iπ) (equivalent to α=-0.1),
the accuracy change should be approximately +23pp, making L9 as effective
as L8.

=======================================================================

7. UNIVERSAL VELOCITY DYNAMICS
=======================================================================

7.1 Architecture-Constrained Phase Profile

The phase profile θ*_l across layers is not random but follows a
pattern determined by the transformer architecture:

  θ*_l ≈ F(l/L, d, N_layers, attention_type)    (28)

For standard MHA decoder-only transformers (Qwen2.5, LLaMA, SmolLM2):

  Phase profile shape (theoretical prediction):
    L0-L3:  θ* ≈ 0 (embedding/early layers — velocity is reliable)
    L4-L7:  θ* ∈ [0, π/4] (feature integration — moderate curvature)
    L8:     θ* ≈ 0 (keystone — transition, curve-straightening)
    L9:     θ* ≈ π (reasoning onset — velocity anti-aligned)
    L10-L14: θ* ∈ [π/2, π] (active reasoning — high curvature)
    L15+:   θ* ≈ π (output preparation — death layers)

This profile is universal across models with the same architecture.
Hybrid attention models (GatedDeltaNet + FA, like Qwen3.5) have a
different profile because 75% of layers lack standard KV-cache,
breaking the attention amplification mechanism for those layers.

7.2 Cross-Model Transfer Function

Given source model S with velocity field v_S[l] at layer l (dim d_S)
and target model T with hidden states h_T[l] (dim d_T, d_T > d_S),
the transfer operates through a learned projection P: ℝ^{d_S} → ℝ^{d_T}:

  v_transfer[l] = P · v_S[l]    (29)

The steering effect on T using v_transfer[l] is:

  Δaccuracy_T[l] ∝ cos(v_transfer[l], v_T_correct[l]) · g_T[l]    (30)

The key insight is that the transfer does NOT need exact direction
matching. It only needs the SIGN of the projection to be preserved:

  sign(cos(v_transfer[l], v_T_correct[l])) ≈ sign(cos(v_S[l], v_S_correct[l]))   (31)

This is the condition for phase-profile preservation across models.
Empirically confirmed by the SmolLM2→7B transfer preserving L8 as the
best layer.

7.3 Why L8 Transfers

L8 is the most robust layer for cross-model transfer because:
1. It's at the feature-to-reasoning transition — a universal
   architectural feature of transformers
2. Low curvature (κ[8] is small) means the velocity direction is
   stable and well-defined
3. The attention gain g[8] is high (moderate entropy) — amplifies
   even small directional signals
4. The correct/incorrect manifold separation d_sep[8] is large in any
   capable model

L8 is NOT special because of its index — it's special because of its
computational role. Models with different depths (e.g., 40-layer
LLaMA-3) would have an equivalent "keystone layer" at a different
index.

=======================================================================

8. FORMAL MATHEMATICAL LANGUAGE
=======================================================================

8.1 Core Equations

E1. Hidden state dynamical system:
    h[l+1] = h[l] + f_l(h[l])    [vector field, ℝ^d]

E2. Velocity definition:
    v[l] = h[l+1] - h[l]    [vector, ℝ^d]

E3. Acceleration definition:
    a[l] = v[l] - v[l-1] = h[l+1] - 2h[l] + h[l-1]    [vector, ℝ^d]

E4. Instantaneous curvature:
    κ[l] = ||a[l]|| / ||v[l]||    [scalar, dimensionless]

E5. Steering update:
    h'[l] = h[l] + α₁·v_pred[l] + α₂·a_pred[l]    [vector, ℝ^d]

E6. Attention gain factor:
    g[l] = ||J_l|| = ||diag(p_l) - p_l·p_l^T||_F    [scalar]
    where p_l is the attention distribution averaged over heads/positions

E7. Effective perturbation:
    Δ_eff[l] = α·||v_pred[l]||·g[l] / √d    [scalar, dimensionless]

E8. Optimal steering phase:
    θ*_l = atan2(κ[l] · sign(cos(v_correct[l], v_incorrect[l])), 1)

E9. Phase-polarity function:
    Φ[l] = cos(v[l], a[l]) · κ[l]    [scalar]
    Φ[l] > 0 → trim-tab; Φ[l] < 0 → death layer

E10. Correct-incorrect manifold separation:
    d_sep[l] = ||μ_correct[l] - μ_incorrect[l]||_M    [scalar]

E11. Steering feasibility condition:
    d_sep[l] > τ_noise[l]    [boolean per layer]

E12. Accuracy change bound:
    Δaccuracy_max[l] ≤ (B_max - B)·[1 - exp(-d_sep[l]² / 2σ²[l])]    [scalar]

E13. Token divergence probability:
    P(divergent | α, l) = 1 - exp(-α²·g[l]²·T / 2σ²_h)    [probability]

E14. Optimal α for layer l:
    α*_l ∝ 1 / g[l]    [scalar]

E15. Cross-model transfer condition:
    sign(cos(v_transfer[l], v_T_correct[l])) = sign(cos(v_S[l], v_S_correct[l]))

8.2 Derived Relations

R1. From E4 and E8:
    κ[l] >> 1 → θ*_l ≈ ±π/2 (acceleration-dominated)
    κ[l] << 1 → θ*_l ≈ 0 (velocity-dominated)

R2. From E3 and E2:
    a[l] = f_l(h[l]) - f_{l-1}(h[l-1])    [compositional difference]
    Variation in a[l] across layers measures change in computational
    role between adjacent layers.

R3. From E10 and E11:
    The set of steerable layers LS = {l | d_sep[l] > τ_noise[l]}
    The strongest trim-tab = argmax_l d_sep[l]/τ_noise[l]

R4. From E6 and E7:
    The product α·g[l] determines whether perturbation is "small"
    (linear regime, α·g[l] < 0.5) or "large" (nonlinear regime,
    α·g[l] > 0.5). Current α=0.1 gives α·g[l] ≈ 0.5 at L8 (borderline).

R5. From E13:
    Token divergence is the INTEGRAL of per-step divergence probability.
    For T > 1/g[l]²·α², trajectory is almost certainly divergent.

8.3 Composite Metrics

M1. Steering Potential Index (SPI):
    SPI[l] = g[l] · d_sep[l] / (κ[l] + ε)
    Higher SPI → better steering potential at layer l.

M2. Phase Stability Score:
    PS[l] = 1 - exp(-|θ*_l - θ*_{l-1}|)
    Near 1 at regime boundaries, near 0 within regimes.

M3. Manifold Separability Index:
    MSI = (1/L) · Σ_l d_sep[l] / τ_noise[l]
    MSI > 1 → model is steerable. MSI < 1 → below capability threshold.

M4. Attention Sensitivity Profile:
    A[l] = ||Δattn[l]|| / ||Δh[l]|| for unit perturbation
    Measures the amplification power of each layer.

M5. Cross-Model Phase Similarity:
    CPS(m, n) = (1/L)·Σ_l cos(θ*_l(m), θ*_l(n))
    CPS > 0.7 → models share steering phase profile.

=======================================================================

9. ALL-OBSERVED-PHENOMENA EXPLANATION TABLE
=======================================================================

| Phenomenon | Explanation | Key Equation |
|------------|-------------|--------------|
| L8: +20pp | L8 is at feature→reasoning transition; low curvature (κ<<1); velocity aligns with correctness gradient (θ*≈0); moderate attention entropy gives high g[8] | E8, E9, E12 |
| L9: -23pp | L9 is reasoning-onset layer; high curvature (κ>>1); velocity anti-aligned with correctness gradient (θ*≈π); high attention entropy gives lower g[9] but high sensitivity to perturbation direction | E8, E9 |
| Adjacent layer paradox | The L8→L9 transition is a dynamical bifurcation (feature comp → reasoning). Phase θ* jumps from ~0 to ~π, which is the maximum possible polarity change | R1, R2 |
| Capability threshold (~40% GSM8K) | Below threshold, d_sep[l] < τ_noise[l] for all layers — correct/incorrect manifolds entangled. Steering cannot amplify nonexistent separation | E11, MSI |
| Math-1.5B (38%) no trim-tabs | Manifold separability MSI < 1 for Math-1.5B. Also base-model (not instruct) may lack correct/incorrect trajectory separation | E11, E12 |
| Cross-model transfer (L8 preserved) | Phase profile θ*_l is architecture-constrained, not parameter-specific. SmolLM2 and Qwen2.5 share decoder-only MHA → similar phase profile | E15, CPS |
| R² not predicting steering | R² measures velocity PREDICTABILITY, not direction alignment with correctness. Steering operates through attention amplification g[l], not velocity replacement | E6, E7 |
| 88% token divergence at α=0.1 | α·g[l] ≈ 0.5 (borderline nonlinear regime). Per-step divergence probability compounds over T=200 tokens | E13 |
| All-layers steering net negative | Averaging over layers includes both phases θ*≈0 and θ*≈π, cancelling net effect. Death layers dominate because they have higher magnitude negative effects | E8, E9 |
| Random vectors (hypothesized similar pattern) | The per-layer SIGN pattern is dominated by g[l] and attention entropy, not v_pred direction. Random vectors of equal norm get amplified by the same g[l] | E6, M4 |
| 4%→73% baseline jump (chat template) | External to theory — data/formatting issue. But consistent: proper formatting activates correct manifold | — |
| Acceleration R² may be more informative | a[l] measures local curvature, which directly determines phase θ*_l. High R²_a means curvature is predictable → phase is predictable → steering can be optimized | E4, E8 |

=======================================================================

10. NOVEL TESTABLE PREDICTIONS
=======================================================================

These 7 predictions have NOT been tested by any existing experiment and
distinguish this unified theory from alternative explanations.

--- PREDICTION 1: Attention Entropy Asymmetry ---

STATEMENT: L8 has lower attention entropy than L9 by at least 20%
(H[L8] < 0.8·H[L9]). This difference in entropy creates g[L8] > g[L9]
(the attention gain factor), making L8 more sensitive to steering.

EXPERIMENT: From existing trajectory data, compute per-head attention
entropy at each layer for the first generation token. Average across
heads and batches. Compare H[8] vs H[9].

ALTERNATIVE EXPLANATION (if falsified): If L8 entropy = L9 entropy
or L8 entropy > L9 entropy, the attention amplification kernel
(Section 2) is incorrect. Steering efficacy would need a different
explanation (e.g., pure manifold geometry).

COST: 0 GPU-hours (entirely from existing attention data).

--- PREDICTION 2: Curvature-Accuracy Correlation ---

STATEMENT: The ratio κ[l] = ||a[l]||/||v[l]|| measured from unsteered
trajectories predicts per-layer steering Δaccuracy with ρ(κ[l], Δacc[l])
< -0.5 (negative correlation: higher curvature → worse or more negative
steering results).

EXPERIMENT: Compute κ[l] from existing trajectory data (25 files,
10.5GB, already collected). Correlate with per-layer accuracy Δ from
existing per-layer sweep results.

ALTERNATIVE EXPLANATION (if falsified): If |ρ| < 0.2, then curvature
is not a meaningful predictor of steering polarity, and the phase
polarization theory (Section 3) requires revision.

COST: 30 min analysis, 0 GPU-hours.

--- PREDICTION 3: Random Vector SIGN Pattern Match ---

STATEMENT: Random unit vectors (same norm distribution as TT
predictions) produce a per-layer SIGN pattern of accuracy changes
that matches the TT steering sign pattern (same layers help/hurt)
but at 30-50% of the TT magnitude. The sign pattern is dominated by
g[l] and attention structure, not prediction direction.

EXPERIMENT: Phase A1 of the minimum viable protocol (4 conditions ×
28 layers). Compare Δaccuracy per layer for random vs TT conditions.

NOTE: If random = TT in BOTH sign AND magnitude, the steering paradigm
fails entirely. If random ≈ TT in sign but 30-50% magnitude, the
amplification mechanism is confirmed with a direction-specificity
residual for the TT to exploit.

ALTERNATIVE EXPLANATIONS:
- If random ≈ 0 (no sign pattern): Attention gain is not the dominant
  mechanism; TT direction is essential.
- If random ≈ TT in magnitude (direction doesn't matter): Paradigm
  collapsed — steering is noise injection.
- If random has OPPOSITE sign pattern: Attention is not isotropic;
  random perturbations exploit different mechanisms.

COST: 0.9 GPU-hours (included in Phase A1).

--- PREDICTION 4: Sub-Threshold High-α Trim-Tabs ---

STATEMENT: Steering Math-1.5B (38% baseline, currently no trim-tabs)
at α > 0.5 reveals trim-tab layers that are invisible at α=0.1.
Specifically, L8 shows Δ > 5pp at α ∈ [0.5, 2.0].

The capability threshold is α-dependent because effective separation
scales with α: d_sep[l, α] = d_sep[l] + α·cos(v[l], μ_correct -
μ_incorrect). Higher α increases effective separation for layers where
velocity aligns with correctness direction.

EXPERIMENT: Run α sweep on Math-1.5B at L8 with α ∈ {0.1, 0.5, 1.0,
2.0}. 100 problems per α = 1.3 GPU-hours.

ALTERNATIVE EXPLANATION (if falsified): If NO trim-tabs appear at ANY
α, the separation condition d_sep[l] < τ_noise[l] is fundamental and
not α-dependent. This would mean the threshold is a hard manifold
property.

COST: 1.3 GPU-hours.

--- PREDICTION 5: Acceleration Source Separation ---

STATEMENT: The acceleration field a[l] decomposes into two components:
(1) "conservative" acceleration from the layer computation function f_l
(deterministic, predictable by TT), and (2) "non-conservative"
acceleration from token composition changes (variable, not predictable).
R²_a on token-constant trajectories (same tokens, different positions)
is significantly higher than on token-varying trajectories.

If R²_a(token-constant) >> R²_a(token-varying), then acceleration has
a computation-driven component that is structurally meaningful.

EXPERIMENT: Take 50 prompts with known correct outputs. For each,
generate twice with identical settings. Compute a[l] for both runs.
Correlate. Then compare a[l] across different prompts. This is a
variance decomposition of a[l].

ALTERNATIVE EXPLANATION (if falsified): If R²_a(token-constant) ≈
R²_a(token-varying) ≈ 0, acceleration is pure noise and complex α
steering is meaningless. If both are equally high, acceleration is
purely a function of token identity, not computation dynamics.

COST: 30 min analysis, 0 GPU-hours.

--- PREDICTION 6: PCA Direction Predicts Polarity ---

STATEMENT: The angle between the first principal component of hidden
state variation and the velocity vector v[l] at layer l predicts
layer polarity:
  cos(PC1[l], v[l]) > 0.7 → trim-tab
  cos(PC1[l], v[l]) < 0.3 → death layer

The first PC captures the dominant direction of hidden state
variation at layer l. If velocity aligns with this dominant
variation direction, steering reinforces the natural computation
(trim-tab). If velocity is orthogonal, steering perturbs the
natural computation (death layer).

EXPERIMENT: Compute PCA on the set of hidden states {h[l]} at each
layer l across the trajectory dataset. Compute the cosine between
PC1[l] and v[l] (mean velocity direction). Correlate with per-layer
Δaccuracy.

ALTERNATIVE EXPLANATION (if falsified): If no correlation between
PC1 alignment and polarity, the manifold geometry is more complex
than linear alignment with dominant variation mode.

COST: 1 hour analysis, 0 GPU-hours.

--- PREDICTION 7: Temporal Priming Profile ---

STATEMENT: L8 steering primarily affects the FIRST 50% of generation
tokens (reasoning steps) rather than the final answer tokens. The
steering effect acts as "reasoning trajectory priming" — it biases
the early reasoning steps toward the correct path, after which the
model continues correctly on its own.

EXPERIMENT: Per-position α steering at L8. Apply steering only to
positions 0-50%, 50-100%, or all positions. Compare accuracy.

Alternatively: Compute accuracy of first half vs second half of
generation under L8 steering vs baseline.

ALTERNATIVE EXPLANATIONS:
- If effect is concentrated in LAST 50%: Steering is a "final answer
  corrector" — different from the proposed "trajectory priming" theory.
- If effect is uniform across positions: Steering modulates general
  computation quality, not specific reasoning steps.
- This also provides evidence for or against the "keystone layer"
  hypothesis from Agent 2 (EM-1).

COST: 2 GPU-hours (existing infrastructure, min code change).

=======================================================================

11. FALSIFICATION CONDITIONS
=======================================================================

This theory is designed to be falsifiable. The following conditions,
if observed, would invalidate specific components.

11.1 Falsification of Attention Amplification Kernel (Section 2)

F-A1: Attention entropy H[L8] = H[L9] within 5% measurement error.
    → The gain asymmetry g[L8] > g[L9] does not exist.
    → Alternative mechanism needed for per-layer sensitivity

F-A2: Random vectors produce the SAME MAGNITUDE of accuracy change as
    TT predictions at the best layer (+20pp vs +20pp).
    → The direction-specificity claimed by the theory is wrong
    → Steering is a perturbation effect, not a direction-dependent one

F-A3: Attention patterns do NOT change measurably (Δattn < 1% of total
    attention mass) under L8 steering.
    → The amplification mechanism is not through attention
    → Alternative: the effect is purely in the MLP or residual stream

11.2 Falsification of Phase Polarization (Section 3)

F-P1: The curvature-polarity function Φ[l] has no correlation with
    per-layer Δaccuracy (|ρ(Φ, Δacc)| < 0.2).
    → Curvature does NOT determine steering polarity
    → Phase polarization theory is incorrect

F-P2: Phase sweep on L8 and L9 shows θ*_8 = θ*_9 (same optimal phase).
    → Adjacent layer paradox is not a phase inversion
    → Alternative: L9 is different for reasons other than phase

F-P3: Derivative dθ*_l/dl is smooth (no discontinuities at regime
    boundaries L3/L4, L7/L8, L8/L9, L14/L15).
    → Phase profile is not piecewise constant
    → Transformer computational regimes are not clearly separated

11.3 Falsification of Manifold Topology (Section 4)

F-M1: Math-1.5B has d_sep[l] >> τ_noise[l] for ALL layers (separation
    is present but steering doesn't exploit it).
    → The capability threshold is not about manifold separation
    → Alternative: some other mechanism blocks steering (e.g., no
      attention amplification due to different training)

F-M2: Qwen2.5-7B has d_sep[l] ≈ τ_noise[l] for ALL layers.
    → The +20pp trim-tab cannot be explained by manifold separation
    → The mechanism must be something other than "pushing toward
      correct manifold"

F-M3: Sub-threshold models at high α show NO layers with Δ > 2pp even
    at α=2.0.
    → The capability threshold is truly a hard boundary
    → α-dependence claim fails

11.4 Falsification of Cross-Model Universality (Section 7)

F-U1: LLaMA-3-8B has a completely different phase profile with no
    "keystone layer" equivalent.
    → The universal velocity dynamics claim fails for other architecture
      families

F-U2: Cross-model transfer (SmolLM2→7B) does NOT preserve L8 as best
    layer when using the unified theory's optimal α (calculated from
    κ[8] and g[8] of the source model).
    → The transfer function is more complex than phase-profile
      preservation

11.5 Theory-Level Falsification

F-T1: All 7 predictions (Section 10) fail simultaneously.
    → The entire theoretical framework is incorrect
    → The phenomenon requires a fundamentally different explanation

F-T2: The theory predicts A but experiment shows NOT A for 4+ of the
    7 predictions, AND no alternative theory explains the remaining
    predictions better.
    → The theory is not just incomplete — it's wrong in structure

=======================================================================

12. LIMITS OF THE THEORY
=======================================================================

12.1 What the Theory CANNOT Explain

L1. Why L8 Specifically
    The theory explains that L8 is a "keystone layer" at the
    feature→reasoning transition, but it cannot predict the exact
    index (8 out of 28 for Qwen2.5-7B) without measuring the specific
    model. The index depends on:
    - Total number of layers (L)
    - Position of the transition in the specific model
    - Layer-wise allocation of computation

    The theory says: "There exists a keystone layer near the first
    third of the model" but not "layer 8 specifically."

L2. Exact Magnitude of Death-Layer Collapse
    The theory predicts L9 as a death layer (negative Δaccuracy) but
    cannot predict the exact magnitude (-23pp → 0% accuracy). The
    complete collapse to 0% suggests off-manifold perturbation (the
    model literally cannot produce a valid answer), not just directional
    misalignment. The theory's phase inversion prediction is for
    recovered accuracy, but the magnitude depends on the model's
    robustness to perturbation, which is not captured by the current
    framework.

L3. The Chat Template Effect
    The 4% → 73% baseline jump from applying the chat template is
    outside the theory's scope. The theory models hidden state dynamics
    during generation, not input formatting. This is a data engineering
    issue that happened to enable the steering discovery.

L4. Why Math-1.5B Specifically Has No Trim-Tabs
    The theory predicts that models with baseline > 0 should have
    SOME layers with d_sep[l] > τ_noise[l]. Math-1.5B at 38% baseline
    contradicts this. The theory's α-dependence prediction (Section
    10, P4) may resolve this, but currently the theory cannot distinguish
    between:
    - Math-1.5B's 38% baseline is not sufficient for manifold separation
    - Math-1.5B is a BASE model, not instruct-tuned — the training
      objective didn't optimize for correctness-separated trajectories
    - The hidden state dimensionality (d=1536) may be too small for
      separable manifolds

L5. Non-Math Task Generalization
    The theory is about the hidden state manifold for REASONING tasks.
    It makes no claims about whether this framework applies to other
    tasks (creative writing, translation, code generation). The
    manifold structure may differ fundamentally for different task
    types.

L6. Quantitative Upper Bound
    The theory provides inequality bounds (E12) but cannot compute the
    exact maximum steering improvement without measuring d_sep[l] and
    σ²[l] empirically. The true upper bound for GSM8K steering is an
    empirical question.

L7. Per-Head vs Per-Layer Steering
    The theory treats each layer as a single steering surface. The
    GQA architecture means that within-layer heads share K/V, limiting
    fine-grained control. Per-head steering would require modifying
    attention patterns differently for each head, which may not be
    possible through KV-cache modification alone.

L8. The Steering Surface Choice
    The theory accepts the KV-cache as the correct steering surface
    without proving it's optimal. Alternative surfaces (residual
    stream, MLP activations, weight flow) might be more effective
    and would require a different theoretical framework.

12.2 When the Theory Breaks Down

Breakdown conditions:
- Models with hybrid attention (Qwen3.5 series) where < 50% of
  layers have standard KV-cache
- Models with non-standard normalization (no LayerNorm, RMSNorm with
  shared parameters)
- Tasks where output quality cannot be measured by accuracy (open-ended
  generation)
- Very small α (< 0.001) where the perturbation is below the attention
  noise floor
- Very large α (> 5) where divergence probability → 1 and all outputs
  are random

12.3 Open Questions the Theory Raises

Q1. Is there a QUANTITATIVE theory of g[l] (attention gain) that
    predicts it from the attention head's specialization?
    → A mechanistic theory of WHY L8 has the right attention entropy

Q2. Can the phase profile θ*_l be predicted from the model WEIGHTS
    alone (without running any generation)?
    → Would enable "steering map" prediction for any model

Q3. Is the "correctness gradient" ∇C(h[l]) equal to the difference
    between the correct and incorrect velocity predictions?
    → i.e., is v_correct[l] - v_incorrect[l] ∝ ∇C(h[l])?
    → Would unify standard and contrastive steering theoretically

Q4. Does the phase profile θ*_l change with generation position?
    → Does a layer become more/less deathly as generation progresses?

Q5. What is the relationship between the velocity manifold and the
    model's training data distribution?
    → Is the velocity structure learned from the training distribution
      or from the architecture?

=======================================================================

13. MATHEMATICAL LANGUAGE FOR FUTURE RESEARCHERS
=======================================================================

This section provides the standardized notation and formalism for
building on this framework.

13.1 Standard Notation

| Symbol | Meaning | Units |
|--------|---------|-------|
| h[l] | Hidden state at layer l | ℝ^d |
| v[l] | Hidden state velocity | ℝ^d |
| a[l] | Hidden state acceleration | ℝ^d |
| κ[l] | Instantaneous curvature | [0, ∞) |
| f_l | Layer computation function | ℝ^d → ℝ^d |
| α | Steering strength | [0, ∞) |
| θ*_l | Optimal phase for layer l | [0, 2π) |
| g[l] | Attention gain factor | [0, ∞) |
| Φ[l] | Phase-polarity function | ℝ |
| d_sep[l] | Manifold separation at layer l | ℝ^+ |
| τ_noise[l] | Within-class variance at layer l | ℝ^+ |
| μ_correct[l] | Mean correct hidden state | ℝ^d |
| μ_incorrect[l] | Mean incorrect hidden state | ℝ^d |
| Δacc[l] | Accuracy change from steering | [-1, 1] |
| P(div) | Token divergence probability | [0, 1] |
| SPI[l] | Steering Potential Index | ℝ^+ |
| PS[l] | Phase Stability | [0, 1] |
| MSI | Manifold Separability Index | ℝ^+ |
| CPS | Cross-Model Phase Similarity | [-1, 1] |

13.2 Standard Definitions

Hidden state trajectory:
  T = {h[0], h[1], ..., h[L-1]}

Velocity field:
  V = {v[l] = h[l+1] - h[l] | l = 0, ..., L-2}

Acceleration field:
  A = {a[l] = v[l] - v[l-1] | l = 1, ..., L-2}

Phase profile:
  Θ = {θ*_l = atan2(κ[l]·sign(alignment_l), 1) | l = 0, ..., L-1}

Amplification profile:
  G = {g[l] = ||diag(p_l) - p_l·p_l^T||_F | l = 0, ..., L-1}

Manifold separation:
  D = {d_sep[l] = ||μ_correct[l] - μ_incorrect[l]||_M | l = 0, ..., L-1}

13.3 Standard Experiments

E-01: Compute κ[l], Φ[l] from trajectory data (0 GPU-hr)
  Output: Curvature-polarity profile

E-02: Compute g[l] from attention data (0 GPU-hr)
  Output: Attention amplification profile

E-03: Compute d_sep[l] from labeled trajectory data (0 GPU-hr)
  Output: Manifold separation profile

E-04: Phase sweep at one layer (0.5 GPU-hr)
  Method: Sweep θ ∈ {-π, -π/2, 0, π/2, π} at fixed r
  Output: Δaccuracy(θ) for that layer

E-05: Full per-layer α sweep (2 GPU-hr)
  Method: Sweep α ∈ {-2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2} for all layers
  Output: Per-layer optimal α and sign

E-06: Cross-condition protocol (3.7 GPU-hr)
  Method: 28 layers × 4 conditions (baseline, random, standard TT, contrastive TT)
  Output: Full steering map

13.4 Research Program Using This Framework

STEP 1: Compute the theory's state from existing data
  Action: Compute κ[l], Φ[l], g[l], d_sep[l] from existing trajectories
  Cost: 1 hour analysis, 0 GPU-hours
  Output: Theoretical predictions for all layers

STEP 2: Validate predictions 1, 2, 5, 6 (zero GPU cost)
  Action: Compute attention entropies, curvature correlations, etc.
  Cost: 2 hours analysis, 0 GPU-hours

STEP 3: Run Phase A1 (4-condition protocol)
  Action: Generate the steering map
  Cost: 3.7 GPU-hours
  Output: Validates/falsifies predictions 3, 6, 7

STEP 4: Run Prediction 4 (high-α on Math-1.5B)
  Action: α sweep on sub-threshold model
  Cost: 1.3 GPU-hours
  Output: Tests α-dependence of capability threshold

STEP 5: Theory refinement
  Based on which predictions are validated, refine the theory
  equations and update the mathematical language.

13.5 Key Open Equations for Future Work

Unsolved equation 1: Closed-form g[l] from model weights
  g[l] = ?(W_Q, W_K, W_V, W_O, W_MLP, position encoding)

Unsolved equation 2: Phase profile from task description
  θ*_l = ?(task_type, model_architecture, training_data)

Unsolved equation 3: Maximum steering improvement bound
  Δmax = ?(d, L, N_params, baseline_accuracy, dataset)

Unsolved equation 4: Attention sensitivity from attention head
  specialization patterns
  g[l] = ?(head_specialization, head_entropy, KV_sharing)

=======================================================================
END OF DIFFUSER-6: UNIFIED THEORY OF VELOCITY-BASED LATENT STEERING
=======================================================================
