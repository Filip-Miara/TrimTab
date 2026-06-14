# Phase 0: VOID — Assumption Surfacing & Bracketing

**Subject**: Velocity-based latent steering for language model reasoning (RankAdaptation)
**Date**: 2026-06-14
**Mode**: Full

---

## Structural Scan — Explicit Assumptions

| # | Assumption | Source | Evidence Grounding |
|---|-----------|--------|-------------------|
| A1 | Hidden states during generation have learnable temporal structure | R²=0.85-0.94 across models | High (confirmed empirically) |
| A2 | KV-cache entries can be modified at specific layers during generation | Implementation works with 88% token divergence | High (confirmed empirically) |
| A3 | Per-layer steering produces layer-specific effects (trim-tabs vs death layers) | L8: +20pp, L9: -23pp on Qwen2.5-7B | High (confirmed on multiple datasets) |
| A4 | Velocity = hidden state delta between consecutive tokens (h_{t+1} − h_t) | Trajectory collection pipeline | Medium (definitional choice) |
| A5 | Steering vector = predicted velocity × scalar α | TT output * α applied to K/V | Medium (design choice) |
| A6 | Training on generation trajectories is superior to prompt trajectories | R² 0.94 vs 0.62 | High (confirmed) |
| A7 | TrajectoryTransformer (small MLP) can model hidden state dynamics | R²=0.85-0.94 | High (confirmed) |
| A8 | GSM8K accuracy is the correct metric for reasoning improvement | Standard benchmark | Medium (proxy assumption) |
| A9 | Steering requires the model to already have the target capability | All models <40% baseline failed | High (confirmed across 5 models) |
| A10 | Layer patterns generalize across datasets | SVAMP replicates GSM8K pattern | Medium (2 datasets only) |
| A11 | Cross-model transfer preserves trim-tab pattern | SmolLM2→7B preserves L8 | Medium (1 transfer pair) |
| A12 | Standard MHA architectures are preferred over hybrid attention | Qwen3.5 failures | Medium (1 hybrid model tested) |

---

## Relational Scan — Implicit Assumptions

| # | Assumption | Inference Chain | Evidence Grounding |
|---|-----------|----------------|-------------------|
| B1 | Predicted velocity direction (sign) is the correct steering direction | TT predicts next state; applying prediction pushes toward that state | Low (untested — assumes descriptive→normative mapping) |
| B2 | A single scalar α is sufficient per layer | All experiments use per-layer α, not per-token or per-head | Medium (design constraint) |
| B3 | Layer effects are independent (intervention at one layer doesn't cascade) | Per-layer sweeps treat layers as independent | Low (untested — layers are sequential) |
| B4 | The TT learns causal dynamics, not just statistical regularities | R² measures prediction accuracy, not causal structure | Low (correlation ≠ causation) |
| B5 | Correct and incorrect trajectories occupy distinct manifolds | Contrastive TT approach assumes separability | Medium (R²=0.83 for each class, but separability unconfirmed) |
| B6 | Steering magnitude α is constant across all tokens in a generation | Single α per experiment | Low (no per-token α tested) |
| B7 | The 7B model's 73% baseline is a "capable enough" starting point | Threshold inference from model comparison | Medium (plausible but unconfirmed) |
| B8 | KV-cache steering at layer L affects only that layer's computation | Mechanistic assumption about transformer architecture | Low (residual stream propagates) |
| B9 | Death layers (L9, L15+) are "bad" universally, not task-specific | Only tested on math reasoning | Low (single domain) |
| B10 | The trained TT weights generalize to unseen problem distributions | Tested on held-out problems from same distribution | Medium (within-distribution only) |

---

## Potential Scan — Counter-Assumptions

| Assumption | Counter-Assumption | Severity if True |
|------------|-------------------|-----------------|
| A1 | Hidden state dynamics are chaotic, not learnable — R² overfits to trivial structure | **FUNDAMENTAL** — undermines entire approach |
| A3 | Trim-tab effects are statistical noise (small sample: 100 problems) | **MAJOR** — reduces confidence in findings |
| A4 | Velocity should be h_{t} − h_{t-k} (multi-step delta), not 1-step | **MEDIUM** — changes TT architecture |
| B1 | The correct steering direction is opposite to predicted velocity (TT learns error patterns) | **FUNDAMENTAL** — explains why steering sometimes hurts |
| B2 | Per-token α is essential — different tokens need different steering strength | **MAJOR** — explains flat results in all-layers steering |
| B3 | Layer effects are highly interdependent — L8 steering changes L9's computation | **MAJOR** — undermines per-layer independence assumption |
| B4 | TT captures spurious correlations (position, token identity) not velocity dynamics | **FUNDAMENTAL** — TT is a confounded predictor |
| B5 | Correct/incorrect manifolds overlap completely — contrastive signal is noise | **MEDIUM** — contrastive TT cannot work |
| B8 | KV-cache modification at layer L propagates to all subsequent layers via residual stream | **MAJOR** — L8's effect is actually upstream/downstream contamination |
| B9 | Death layers are task-beneficial on non-math tasks (different directionality) | **MEDIUM** — death layer classification is task-specific |
| C1 | The 73% baseline is due to prompt formatting, not reasoning capability | **MAJOR** — steering effect is just prompt-matching amplification |
| C2 | The +20pp improvement is within measurement noise (±5pp for 100 problems) | **MAJOR** — effect size overestimated |

---

## Bracket Statement

These assumptions are set aside for the main analysis. They will be re-examined in Phase 6 (Disparity Detection) to check if any assumption violation is the root cause of observed failures. In particular:

1. **B1** (sign direction correctness) is the most critical untested assumption — if the TT predicts error trajectories, applying its prediction amplifies errors.
2. **B3** (layer independence) and **B8** (KV-cache locality) together form a nested assumption about transformer mechanics that may not hold.
3. **A4** (1-step velocity definition) is a design choice that precludes multi-step dynamics.
4. **B4** (causal vs correlational) is the fundamental epistemic risk — the entire steering paradigm depends on the TT learning true dynamics, not confounded correlations.

---

## Assumption Risk Matrix

| ID | Assumption | Criticality | Evidence Level | Risk |
|----|-----------|-------------|---------------|------|
| B1 | Predicted sign is correct direction | **Critical** | None tested | **HIGH** |
| B3 | Layer independence | **Critical** | None tested | **HIGH** |
| B4 | TT learns causal dynamics | **Critical** | None tested | **HIGH** |
| A4 | 1-step velocity definition | **Structural** | Design choice | **MEDIUM** |
| B2 | Single α per layer suffices | **Structural** | Constrained by experiments | **MEDIUM** |
| A3 | Trim-tab pattern is real (not noise) | **Core Finding** | 100 problems only | **MEDIUM** |
| A9 | Steering requires capability | **Core Finding** | 5 models, consistent | **LOW** |
| A2 | KV-cache modification works | **Infrastructure** | Confirmed (88% divergence) | **LOW** |
