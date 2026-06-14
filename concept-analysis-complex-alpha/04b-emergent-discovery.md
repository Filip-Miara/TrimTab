# Emergent Discovery — Complex α (Cross-Pollinated)

## Unconventional Recombinations

### RECOMB-C1: Complex α × Per-Head Steering (cross-level)
- **Protocol**: Within L8, each head h gets complex (r_h, θ_h). Heads known for syntax (usually early heads) get θ ≈ π/2 (acceleration = curve-bending), while reasoning heads get θ ≈ 0 (velocity = pushing).
- **Novelty**: 5/5 — Combines two emergent capabilities from main analysis
- **Prediction**: The 3-5 trim-tab heads within L8 likely cluster at specific phases. Head-level phase clustering is a *new observable*.

### RECOMB-C2: Complex α × GDN Recurrent State (domain-transposed)
- **Domain**: Qwen3.5's GatedDeltaNet recurrent state S accumulates k⊗v outer products.
- **Transposition**: Replace S += β(k⊗v) with S += β·e^(i·φ)·(k⊗v). The phase φ acts as a *rotation gate* on the recurrent memory.
- **Emergent insight**: The recurrent state naturally supports phase because outer products have eigenstructure — the phase φ is the rotation of the principal eigenvector of S. Complex α on GDN layers = rotating the dominant memory direction.

### RECOMB-C3: Acceleration Field × Capability Threshold (forbidden pair)
- **From main synthesis**: Capability threshold (~40% GSM8K) blocks all velocity steering.
- **Counter-assumption**: Acceleration structure may exist even when velocity structure is absent.
- **Protocol**: Compute R²_v and R²_a for small models (SmolLM2, Math-1.5B). If R²_a >> R²_v, complex α (pure acceleration) can steer below-threshold models.

### RECOMB-C4: Complex α applied to the steering research itself (self-application)
- Current research "velocity": moving from velocity-only to per-layer to contrastive.
- Current research "acceleration": the rate of change of the research direction.
- Complex α at the research level: θ = π/2 → pure acceleration research (stop pushing the current direction, "bend" the research into a new direction — like investigating self-supervised contrastive instead of supervised contrastive).

## Emergent Capability Analysis

### EM-C1: Phase-Disambiguated Steering
- **Source**: RECOMB-C2 (complex α × per-head)
- **Description**: Steering with phase separates the "correction direction" from the "correction type" — θ selects the type (velocity push vs acceleration bend), r selects the strength.
- **Q1 — Qualitatively distinct?**: Y — Real α only controls strength; complex α controls BOTH strength and geometric mode.
- **Q2 — Not predictable from constituents?**: Y — The interaction between velocity and acceleration at a layer is synergistic (cross-Lens 2, 4, 8). An omniscient observer knowing v and a separately could not predict h' = r(cosθ·v + sinθ·a) without knowing θ.
- **Q3 — Synergy in kind?**: Y — Enables *differential geometric intervention* — pushing along the trajectory vs bending the trajectory — a qualitatively different capability.
- **Classification**: **CONFIRMED EMERGENT**

### EM-C2: Phase-Locked Multi-Layer Resonance
- **Source**: RECOMB-C1 (cross-level) + main synthesis synergy map (quadruple combo)
- **Description**: Multiple layers steered with ALIGNED phases produce constructive interference — the total effect > sum of individual layer effects.
- **Q1 — Qualitatively distinct?**: Y — Layer resonance is a collective phenomenon. No single layer exhibits it.
- **Q2 — Not predictable?**: Y — Requires knowing the phase alignment metric, which is a second-order property of the joint steering configuration.
- **Q3 — Synergy in kind?**: Y — Changed from "independent per-layer gains" to "coordinated phased array" — a new paradigm.
- **Classification**: **CONFIRMED EMERGENT**

## Synergy Mapping (cross-pollinated)

| Pair | Score (0-10) | Rationale |
|------|-------------|-----------|
| {Complex α, Contrastive Direction} | 9.5 | Phase rotation of the contrastive vector gives BOTH normative direction AND geometric mode |
| {Complex α, Per-head steering} | 9.0 | Each head gets (r, θ) — exponentially richer control surface |
| {Complex α, Confidence gate} | 8.5 | Adaptive phase based on uncertainty — bend when uncertain, push when confident |
| {Complex α, GDN recurrent state} | 8.0 | Natural mapping: GDN state eigenphase = steering phase |
| {Complex α, Adaptive α(t)} | 8.5 | θ(t) schedule could oscillate between velocity and acceleration |

**Highest Higher-Order**: {Complex α × Contrastive × Per-head × Confidence Gate} — Score: 9.7/10
- Cross-pollinated from main synthesis quadruple (score 9.5). Complex α ADDS a geometric dimension to every component.
