# Master Regulators — RankAdaptation

## Ranking Methodology
Score = Influence Centrality (0-10) × Junction Leverage (0-10)
Max possible = 100

---

## #1: Layer 8 (Trim-Tab) — Score: 84

| Attribute | Value |
|-----------|-------|
| **Influence Centrality** | 9/10 — Layer 8 is an intermediate layer where attention/MLP computation transitions from pattern detection to reasoning execution. Steering L8 affects ALL downstream layers (9-27). |
| **Junction Leverage** | 9.3/10 — J2 (TT → h') and J3 (h' → KV cache) combine at L8 to produce the largest observed accuracy effect (+20pp). The leverage is validated on two datasets (GSM8K, SVAMP) and cross-model transfer. |
| **Modulation Strategies** | Existing: α·v steering via KV-cache modification at L8. Proposed: (1) Per-head steering within L8, (2) Multi-step α schedule for L8 (start strong, decay), (3) RL-optimized α for L8 as a learned parameter. |
| **Expected Impact** | HIGH — L8 is the confirmed best trim tab with +20pp. Further optimization (per-head, α schedule) may yield +25-30pp. |
| **Risk** | Over-steering L8 could convert it to a death layer. The α must be kept in the [0.01, 0.3] range. |

## #2: Contrastive Direction (v_c - v_i) — Score: 78

| Attribute | Value |
|-----------|-------|
| **Influence Centrality** | 8/10 — Acts as a *meta-signal* that changes the goal from "predict where the model is heading" to "predict where it should be heading." Affects all downstream steering decisions. |
| **Junction Leverage** | 9.8/10 — J9 (contrastive composition) is the theoretical bridge from descriptive to normative. If confirmed, it changes the fundamental approach to steering. |
| **Modulation Strategies** | Existing: contrastive evaluation script (`run_contrastive_eval.py`). Proposed: (1) Asymmetric contrastive weighting γ·v_c - (1-γ)·v_i, (2) Bootstrapped contrastive pairs (multi-head), (3) Online contrastive update during generation. |
| **Expected Impact** | HIGH — If contrastive direction produces trim tabs on 7B (evaluation pending), this becomes the primary steering mechanism. |
| **Risk** | Cancellation: v_c - v_i may lose shared structure (both correct and incorrect trajectories share basic language modeling dynamics). |

## #3: Per-Layer α Vector — Score: 65

| Attribute | Value |
|-----------|-------|
| **Influence Centrality** | 7/10 — Replaces a single global α with a vector of 28 layer-specific coefficients. Each layer's steering is independently modulated. |
| **Junction Leverage** | 9/10 — J6 (layer polarity) combined with J1 (α·v composition) creates a per-layer gain control surface. Allows trim tabs to be amplified while death layers are attenuated. |
| **Modulation Strategies** | Existing: 4-stage autonomous sweep (`run_autonomous_sweep.py`). Proposed: (1) Gradient-based α learning via validation accuracy, (2) Bayesian optimization over α vectors, (3) Dynamic α(t) per token position. |
| **Expected Impact** | MED-HIGH — Multi-layer combinations with per-layer α may outperform single-layer steering. Stage 3-4 of autonomous sweep tests this. |
| **Risk** | Combinatorial explosion: 28 layers × continuous α → infinite search space. Needs efficient optimization. |

## #4: Token Position t=0 (First Generation Step) — Score: 60

| Attribute | Value |
|-----------|-------|
| **Influence Centrality** | 8/10 — The first token after the prompt determines the reasoning path. All subsequent tokens are conditioned on this first generated token. Steering at t=0 has maximum causal leverage. |
| **Junction Leverage** | 7.5/10 — J8 (temporal sequence) means perturbations at early steps compound through the chain. However, current pipeline skips steering at step 0 (first_step=True guard in all eval scripts). |
| **Modulation Strategies** | Proposed: (1) Remove first_step skip — steer at t=0 for maximum leverage, (2) Exploration schedule: high α at t=0 decaying linearly, (3) Predict velocity from prompt hidden states (not generation hidden states) to steer the very first token. |
| **Expected Impact** | MED — First-token steering may produce the largest per-step divergence but accuracy impact is uncertain. |
| **Risk** | Steering the first token could derail the entire generation if α is too high. |

## #5: Layer Polarity Signature (Trim-tab vs Death-layer Identification) — Score: 55

| Attribute | Value |
|-----------|-------|
| **Influence Centrality** | 6/10 — Knowing which layers are trim-tabs and which are death-layers determines the SET of layers available for safe steering. |
| **Junction Leverage** | 9/10 — J6 (hierarchical: polarity → steering strategy) gates all downstream decisions. Without polarity knowledge, steering cannot be selective. |
| **Modulation Strategies** | Existing: per-layer sweep (`run_per_layer_sweep.py`). Proposed: (1) Polarity prediction from model architecture alone (without running sweep), (2) Dynamic polarity — trim-tab layers may change per input type, (3) Polarity as a learned function of problem difficulty. |
| **Expected Impact** | MED — Knowing layer polarity is necessary but not sufficient for improvement. The α and contrastive direction still need optimization. |
| **Risk** | Polarity may not be stable across tasks. L8 might be trim-tab for math but death-layer for code. |
