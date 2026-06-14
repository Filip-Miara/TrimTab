# Phase 4b: Emergent Discovery

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Unconventional Recombinations

### Cross-Level Recombination

**RECOMB-1**: Combine A_V (Velocity — Level 1 atom) with PEAK (Velocity-Based Steering — Level 5 peak)
- **Protocol**: What does A_V (1-step delta) reveal about the Peak concept that intermediate composites hide?
- **Finding**: The 1-step velocity definition hides the fact that steering effects are NOT local in time. A_V measures instantaneous change, but PEAK's effect is a *cumulative trajectory modification*. The intermediate composites (C2_VEL, C3_STEER) both assume locality-in-time, but the peak effect is non-local. The 1-step atom contradicts the peak's cumulative nature.
- **Novelty Score**: 4/5

**RECOMB-2**: Combine A_H (Hidden State — Level 1 atom) with PEAK (Level 5 peak)
- **Protocol**: What does A_H reveal about PEAK that the intermediate composites hide?
- **Finding**: Hidden states are high-dimensional (3584 for 7B). The intermediate composites treat H as a single vector, but steering may only affect a low-dimensional subspace. The peak concept (velocity steering) may be operating on a 50-dim subspace of a 3584-dim space. Intermediate composites miss this subspace structure.
- **Novelty Score**: 5/5

**RECOMB-3**: Combine A_α (Alpha — Level 1 atom) with C4_KNOWLEDGE (Theoretical Framework)
- **Protocol**: What does the scalar α reveal about the theoretical framework?
- **Finding**: Using a single scalar α implies the framework assumes isotropic steering — all dimensions of the velocity vector are equally important. The theoretical framework should instead be anisotropic (dimension-specific α). Intermediate composite C2_STEER never questions isotropy.
- **Novelty Score**: 4/5

---

### Domain-Transposed Recombinations

**RECOMB-4**: Transpose the pyramid into **Quantum Physics**
- **Mapping**: Hidden state → quantum state vector |ψ⟩; Velocity → time derivative ∂|ψ⟩/∂t; TT → Hamiltonian Ĥ (predicts evolution); Steering → external potential V (modifies evolution); Layer → energy level; Death layer → forbidden transition
- **Insight**: In quantum mechanics, the Hamiltonian completely determines evolution. If our TT is analogous to a Hamiltonian, then steering is analogous to adding a potential V that changes the eigenstates. The "death layer" is a resonance where V causes destructive interference. The "trim-tab" is constructive interference.
- **Novelty Score**: 5/5

**RECOMB-5**: Transpose the pyramid into **Architecture (Building Design)**
- **Mapping**: Hidden state → structural load; Velocity → load distribution change; TT → finite element analysis; Steering → adding a support column; Layer → floor level; Death layer → resonance frequency of floor matches sway → building collapses
- **Insight**: The "death layer" is a resonant frequency problem — steering at that layer excites a natural mode of the transformer that amplifies error. The "trim-tab" is damping — it dissipates error energy.
- **Novelty Score**: 4/5

**RECOMB-6**: Transpose the pyramid into **Cooking**
- **Mapping**: Hidden state → ingredient mixture; Velocity → how flavor changes as you cook; TT → recipe (predicts next flavor); Steering → adding seasoning; Layer → cooking step; Death layer → adding salt at the wrong time ruins the dish; α → amount of seasoning
- **Insight**: Per-layer steering is like seasoning each step of a recipe separately. The same amount of salt (α) is wrong for different steps — each step needs different seasoning. The order matters (can't salt after baking). This maps directly to: the same α is wrong for different layers, and layer order in computation matters.
- **Novelty Score**: 3/5

---

### Forbidden Pairs

**RECOMB-7**: A_V (1-step velocity) + ¬A_V (velocity should be multi-step) [from Phase 0 counter-assumption]
- **Constituents**: Standard velocity + counter-assumption that velocity should be Δt = {1, 2, 4, 8} tokens
- **Predicted Behavior**: The hybrid uses a multi-scale velocity bank — the TT takes max over scales as the prediction, with scaling factor per scale. Steering at scale 1 produces local effects; steering at scale 8 produces global trajectory bends.
- **Resolution**: The 1-step velocity is not wrong; it's incomplete. A multi-scale velocity bank is compatible with 1-step — it's an extension, not a contradiction.

**RECOMB-8**: B1 (sign direction is correct) + ¬B1 (sign direction is wrong, steer opposite) [from Phase 0 counter-assumption]
- **Constituents**: Standard steering direction + counter-assumption that we should steer opposite to TT prediction
- **Predicted Behavior**: The system simultaneously tries both directions per layer and compares outcome. This is the oscillating α strategy (M12): on odd tokens steer +α, on even tokens steer −α, predict which direction works.
- **Resolution**: The contradiction may be resolvable per-layer — maybe L8 needs +α while L9 needs −α. The forbidden pair is actually the key discovery: **each layer has its own optimal sign**.

**RECOMB-9**: B3 (layer independence) + ¬B3 (layer effects are interdependent) [from Phase 0 counter-assumption]
- **Constituents**: Per-layer sweep methodology + counter-assumption that layers interact
- **Predicted Behavior**: A multi-layer experiment where layers L8 and L9 are simultaneously steered with opposite signs — L8: +α (trim-tab), L9: −α (inverted death layer). This tests whether the effects are additive, multiplicative, or nonlinear.
- **Resolution**: If layer effects are independent, L8(+α) + L9(−α) = expected net ~0. If interdependent, result is unpredictable. This experiment itself resolves the forbidden pair.

---

### Self-Application

**RECOMB-10**: Feed PEAK (velocity-based steering) back through the pyramid as an atomic concept
- **Protocol**: Apply the full analysis to the claim "velocity-based steering works for improving reasoning"
- **Result**: 
  - A_H (Hidden State) = "what the system is steering"
  - A_V (Velocity) = "how the steering effect changes over time" (meta-velocity)
  - A_TT (TrajectoryTransformer) = "what predicts the steering's trajectory"
  - A_α (Alpha) = "how strongly to steer the steering system"
  - A_L (Layer) = "which claims about steering to steer"
  - A_M (Capability) = "does the steering project have the baseline capability to be improved by meta-steering?"
  
- **Key Insight**: The project itself exhibits the pattern it studies — the trajectory of the project (v0.21→v0.38) has learnable structure (discoveries accumulate), steering at different "project layers" (infrastructure, experiments, theory) has differential effects (infrastructure improvements had high leverage), and the project's "capability" (understanding of steering) must reach a threshold before meta-improvements work. This is a **strange loop**.

---

## Emergent Capability Analysis

### EM-1: Self-Improving Steering Loop (from RECOMB-3: cross-level)
**Source**: RECOMB-3 (A_α × C4_KNOWLEDGE)
**Description**: A system that steers itself by treating its own steering process as a trajectory to be learned
**Qualification**:
- Q1 (Qualitatively distinct?): **Y** — Self-steering is not a sum of steering × 2; it's a qualitatively different process (meta-cognition)
- Q2 (Not predictable from constituents?): **Y** — Given complete knowledge of velocity steering, one could not predict that the system would start steering its own steering process
- Q3 (Synergy in kind?): **Y** — Self-steering produces a new capability (autonomous improvement) that neither steering alone nor the theoretical framework alone possesses
- **Classification**: **CONFIRMED EMERGENT**
- **Trigger Conditions**: Requires (1) automated pipeline that can run without human intervention, (2) evaluation feedback within the loop, (3) α that is allowed to go to zero (system can choose not to steer)
- **Latent Path**: Build automated pipeline → add meta-controller that adjusts steering parameters → measure meta-improvement over multiple generations

### EM-2: Anisotropic Steering (from RECOMB-2: cross-level)
**Source**: RECOMB-2 (A_H × PEAK)
**Description**: Steering that operates in a learned low-dimensional subspace of the hidden state, with per-dimension α
**Qualification**:
- Q1 (Qualitatively distinct?): **Y** — Scalar-α steering operates on the full velocity vector; anisotropic steering operates on latent factors — these are different spaces
- Q2 (Not predictable from constituents?): **N** — Given knowledge that hidden states have low-dimensional structure, anisotropic steering follows predictably
- Q3 (Synergy in kind?): **N** — The improvement is quantitative (more precise) not qualitative (new capability)
- **Classification**: **QUANTITATIVE ENHANCEMENT**

### EM-3: Multi-Scale Velocity Steering (from RECOMB-7: forbidden pair)
**Source**: RECOMB-7 (A_V + ¬A_V)
**Description**: Steering using velocity predictions at multiple time scales simultaneously
**Qualification**:
- Q1 (Qualitatively distinct?): **Y** — Multi-scale steering captures hierarchical dynamics (token-level + phrase-level + sentence-level), which single-scale cannot
- Q2 (Not predictable from constituents?): **N** — Given 1-step velocity and the concept of multi-scale, the extension is predictable
- Q3 (Synergy in kind?): **N** — More comprehensive, not fundamentally new
- **Classification**: **QUANTITATIVE ENHANCEMENT**

### EM-4: Death Layer Inversion (from RECOMB-8: forbidden pair)
**Source**: RECOMB-8 (B1 + ¬B1)
**Description**: Detecting which layers need +α and which need −α, and applying the appropriate sign per layer
**Qualification**:
- Q1 (Qualitatively distinct?): **Y** — Per-layer sign detection is a new capability (it requires distinguishing between steering directions)
- Q2 (Not predictable from constituents?): **Y** — The fact that some layers systematically need −α while others need +α is not predictable from the TT's prediction alone (which predicts magnitude, not sign utility)
- Q3 (Synergy in kind?): **Y** — Per-layer sign detection enables a "correct by construction" steering system where no layer is harmful — a qualitative improvement over current "accept death layers" approach
- **Classification**: **CONFIRMED EMERGENT**
- **Trigger Conditions**: Requires (1) testing both ±α per layer, (2) detecting that L9 is −α beneficial, (3) automated sign detection mechanism
- **Latent Path**: Run ±α experiment → confirm L9(−α) works → build automatic sign classifier → deploy steerable-all-layers system

### EM-5: Resonant Steering (from RECOMB-5: domain transpose → Architecture)
**Source**: RECOMB-5 (Architecture transpose)
**Description**: The insight that steering effectiveness is determined by resonance between steering frequency and the transformer's natural computational frequencies
**Qualification**:
- Q1 (Qualitatively distinct?): **Y** — The concept of "steering resonance" is not present in any constituent concept
- Q2 (Not predictable from constituents?): **Y** — The architecture domain provided the resonant frequency concept, which is not derivable from velocity steering mechanics alone
- Q3 (Synergy in kind?): **Y** — If true, it reframes the entire problem: find the resonant frequency of each layer and match steering to it
- **Classification**: **CONFIRMED EMERGENT** (theoretical)
- **Trigger Conditions**: Requires validation — measure the "frequency response" of each layer by applying oscillatory steering and measuring output amplitude
- **Latent Path**: Build frequency-response curve for L8 vs L9 → characterize "steering transfer function" of each layer → design steering to match resonant frequencies

---

## Synergy Map

### Pairwise Synergy Scores

| Pair | Score | Description | Classification |
|------|-------|-------------|----------------|
| (Layer Selection, Alpha) | 9/10 | Joint optimization of {layer, α} has superlinear payoff — wrong α on right layer = wasted, right α on wrong layer = harmful | **Qualitative** (enables full-system optimization) |
| (TT Quality, Contrastive Direction) | 8/10 | A contrastive TT that predicts v_correct − v_incorrect directly synergizes TT and contrastive into one model | **Quantitative** (more efficient, not new) |
| (Capability Threshold, Self-Improving Loop) | 7/10 | Self-improving loop may bootstrap capability, breaking the threshold barrier | **Qualitative** (capability creation) |
| (Death Layer Inversion, Resonant Steering) | 7/10 | Death layers may be layers with opposite resonance — both insights point to the same mechanism | **Qualitative** (unifies two phenomena) |
| (Multi-Scale Velocity, Anisotropic Steering) | 6/10 | Multi-scale predictions require anisotropic treatment per scale | **Quantitative** |
| (All steering mechanisms, PEAK) | 5/10 | No steering mechanism alone produces the full effect | Baseline |

### Higher-Order Synergy

**Triple: (Layer Selection, Alpha, Self-Improving Loop)**
- Interaction(Layer_Selection, Alpha, Self_Improving) = 8/10
- Sum of pairwise: (9 + 5 + 7) = 21/30
- Higher-Order Score = 8 − 21/30 ≈ 0.7 on normalized scale → **self-organization detected**
- **Explanation**: The self-improving loop learns optimal {layer, α} pairs automatically, which neither (Layer, α) synergy alone nor (Layer, Self-Improving) alone would produce. The triple creates an autonomous optimization system.

**Quadruple: (Layer Selection, Alpha, TT Quality, Self-Improving Loop)**
- Self-Organization Score: **YES** — This quadruple forms a complete autonomous steering system. The components organize into a feedback loop where each improves the others:
  1. Self-Improving Loop selects next {layer, α} to try
  2. TT Quality improves as more (layer, α) combinations are explored (data diversity)
  3. Layer Selection improves as TT reveals better steering directions
  4. Alpha selection improves as TT provides better uncertainty estimates
- This is a genuine emergence: the quadruple exhibits a capability (autonomous optimization) that no subset possesses.

### Summary Statistics

| Metric | Count |
|--------|-------|
| CONFIRMED EMERGENT | 3 (Self-Improving Loop, Death Layer Inversion, Resonant Steering) |
| QUANTITATIVE ENHANCEMENTS | 2 (Anisotropic Steering, Multi-Scale Velocity) |
| COMPOSITIONAL | 0 |
| REDUCTIVE | 0 |
| Highest Pairwise Synergy | (Layer Selection, Alpha) = 9/10 |
| Highest Higher-Order Synergy | Quadruple (Layer, Alpha, TT, Loop) |
| Self-Organization Detected | **YES** — The quadruple forms an autonomous optimization system |
