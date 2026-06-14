# Phase 7: Causal Mapping & Counterfactual Analysis

---

## 7.1 Causal DAG

### Nodes

| Node | Description | In-Degree | Out-Degree | Type |
|------|-------------|-----------|------------|------|
| N1 | Baseline model (Qwen2.5-7B) | 0 | 3 | Input |
| N2 | Chat template applied | 1 (N1) | 1 (N3) | Preprocessing |
| N3 | Baseline accuracy (73%) | 1 (N2) | 3 (N4, N5, N6) | Measurement |
| N4 | Trajectory collection pipeline | 1 (N3) | 1 (N7) | Infrastructure |
| N5 | Per-layer sweep infrastructure | 1 (N3) | 2 (N8, N9) | Infrastructure |
| N6 | Contrastive TT training decision | 1 (N3) | 1 (N10) | Decision |
| N7 | Training trajectories (storage) | 1 (N4) | 2 (N11, N12) | Data |
| N8 | Trim-tab layer identification (L8) | 1 (N5) | 3 (N13, N14, N15) | Finding |
| N9 | Death layer identification (L9, L15+) | 1 (N5) | 2 (N16, N17) | Finding |
| N10 | Trained TT (correct) | 1 (N6) | 1 (N18) | Artifact |
| N11 | Trained TT (standard, all data) | 2 (N7, N5) | 2 (N13, N19) | Artifact |
| N12 | Trained TT (incorrect) | 1 (N7) | 1 (N18) | Artifact |
| N13 | Standard TT steering (L8, α=0.1) | 2 (N8, N11) | 1 (N20) | Experiment |
| N14 | Cross-model transfer (SmolLM2→7B) | 2 (N8, N11) | 1 (N21) | Experiment |
| N15 | Cross-dataset generalization (SVAMP) | 2 (N8, N11) | 1 (N22) | Experiment |
| N16 | Per-layer sweep (Math-1.5B) | 1 (N9) | 1 (N23) | Experiment |
| N17 | All-layers steering | 1 (N9) | 1 (N24) | Experiment |
| N18 | Contrastive steering evaluation | 2 (N10, N12) | 1 (N25) | Experiment |
| N19 | Standard TT alpha sweep | 1 (N11) | 1 (N26) | Experiment |
| N20 | GSM8K L8 result (+20pp) | 1 (N13) | 0 | Output |
| N21 | Cross-model transfer result (pattern preserved) | 1 (N14) | 0 | Output |
| N22 | SVAMP result (L8:+4pp, L9:-14pp) | 1 (N15) | 0 | Output |
| N23 | Math-1.5B result (no trim tabs) | 1 (N16) | 0 | Output |
| N24 | All-layers result (negative) | 1 (N17) | 0 | Output |
| N25 | Contrastive sweep results | 1 (N18) | 0 | Output (pending) |
| N26 | Alpha sensitivity data | 1 (N19) | 0 | Output |

### Edges

| Edge | From | To | Causal Type | Estimated Delay | Strength |
|------|------|----|-------------|-----------------|----------|
| E1 | N1 | N2 | computational | ~10s (model load) | Required |
| E2 | N2 | N3 | computational | ~2 min (100 problems) | Deterministic |
| E3 | N3 | N4 | logical | ~30 min (setup) | 1.0 |
| E4 | N3 | N5 | logical | ~30 min (setup) | 1.0 |
| E5 | N3 | N6 | logical | ~30 min (setup) | 1.0 |
| E6 | N4 | N7 | computational | ~2 hrs (500 problems) | 1.0 |
| E7 | N5 | N8 | computational + analytical | ~4 hrs (28 layers × 100 problems) | 1.0 |
| E8 | N5 | N9 | computational + analytical | ~4 hrs (same as N5) | 1.0 |
| E9 | N8 | N13 | logical | ~30 min (eval setup) | 1.0 |
| E10 | N8 | N14 | logical | ~30 min (transfer setup) | 0.8 |
| E11 | N8 | N15 | logical | ~30 min (SVAMP setup) | 1.0 |
| E12 | N9 | N16 | logical | ~4 hrs (Math-1.5B per-layer) | 1.0 |
| E13 | N9 | N17 | logical | ~30 min | 1.0 |
| E14 | N11 | N13 | computational | ~5 min (TT inference per eval) | Required |
| E15 | N11 | N14 | computational | ~5 min | Required |
| E16 | N11 | N15 | computational | ~5 min | Required |
| E17 | N10 | N18 | computational | ~2 hrs (200 problems) | Required |
| E18 | N12 | N18 | computational | ~2 hrs (same eval) | Required |

### Structural Observations

- **Total nodes**: 26, **Total edges**: 18
- **Source nodes** (in-degree 0): N1 (baseline model)
- **Sink nodes** (out-degree 0): N20, N21, N22, N23, N24, N25, N26
- **Dominant node** (highest betweenness centrality): N3 (baseline accuracy) — everything depends on it
- **Critical path**: N1 → N2 → N3 → N5 → N8 → N13 → N20 (the +20pp result)

---

## 7.2 Branching Points

### BP1: N5 (Per-layer sweep infrastructure → identification of trim tabs)

**Out-degree**: 2 (N8, N9)
**Type**: Analytical branching point
**Description**: The same per-layer sweep produces both the trim-tab finding AND the death-layer finding AND the capability-threshold finding. The interpretation of these results branches depending on whether the experimenter focuses on positive or negative outcomes.

**Branch A** (optimistic): Focus on L8 → +20pp. Leads to: cross-model transfer, cross-dataset generalization, publishing.
**Branch B** (pessimistic): Focus on L9 → −23pp. Leads to: investigation of all-layers steering failures, questioning the approach.
**Branch C** (neutral): Focus on capability threshold → steering-only-amplifies. Leads to: constrained-scope steering applications, focus on capable models only.

**Project took**: Branch A + C (both optimistic and constrained).

### BP2: N11 (Trained standard TT → application targets)

**Out-degree**: 3 (N13, N14, N19)
**Type**: Resource allocation branching
**Description**: The same TT checkpoint can be used for L8 confirmation, cross-model transfer, and alpha sweeps.

### BP3: N6 (Contrastive TT training decision → evaluation)

**Out-degree**: 1 (N10) but invisible branching after N10
**Hidden branch**: Whether contrastive evaluation produces positive, neutral, or negative results. This is the most important unresolved branch.

---

## 7.3 Counterfactuals

### CF-1: "What if the chat template was NOT the issue?"

**Intervention**: Replace N2 (chat template application) with raw-text prompting.
**Predicted outcome**: Baseline accuracy would remain at ~4% (as observed before fix). L8 steering would show +5pp from 4% → 9% (amplification of existing capability). But 9% is still useless. The entire project line would have appeared to fail. The "steering requires capability" finding would have been discovered sooner, but the +20pp result would have been missed.
**Testability**: Already observed — this was the state before the chat template fix.

### CF-2: "What if the per-layer sweep had started from L9 instead of L0?"

**Intervention**: Test layers in reverse order (L27 → L0 instead of L0 → L27).
**Predicted outcome**: The first tested layers would be death layers (L15+, L9). The experimenter would see −23pp immediately, potentially concluding steering is harmful and stopping. **The order of testing could have caused abandonment of the project.**
**Testability**: Not testable (would require time travel), but highlights the fragility of the discovery process. A random layer order could have reached a completely different conclusion.

### CF-3: "What if only SmolLM2-360M data were used?"

**Intervention**: No 7B experiments conducted; only SmolLM2 and smaller models.
**Predicted outcome**: Conclusion would be "velocity-based steering is impossible" because all models tested have <38% baseline. The capability threshold finding would dominate. The +20pp trim-tab pattern would never be discovered.
**Testability**: Counterfactual only, but directly informs that **model choice is the most important experimental variable**.

### CF-4: "What if contrastive TT is evaluated and shows NO improvement?"

**Intervention**: Run run_contrastive_eval.py and find that v_c − v_i produces worse accuracy than standard TT at all layers.
**Predicted outcome**: The "contrastive→normative" narrative collapses. The project must pivot to understanding WHY the difference vector doesn't point toward correct answers. Possible explanations: (a) correct/incorrect manifolds not separable, (b) TT_correct and TT_incorrect have different prediction biases that make their difference noisy, (c) the steering mechanism (K/V projection) distorts the contrastive signal.
**Testability**: **IMPENDING** — the evaluation is currently pending. This is the most urgent experiment.

### CF-5: "What if L8 is steered with negative α?"

**Intervention**: α = −0.1 at L8 (push against velocity direction).
**Predicted outcome**: If the velocity field always points toward "most likely continuation," then negative α should push away from the default continuation. For correct problems, this could push the model away from the correct answer → accuracy decrease. For incorrect problems (where the default continuation is wrong), negative α could push toward correct → accuracy increase. Net effect depends on the ratio of correct/incorrect problems the model would have gotten right/wrong anyway.
**Testability**: Simple — modify the alpha sign in steer_layer() and evaluate.

### CF-6: "What if the TT is replaced by a linear baseline (constant velocity per layer)?"

**Intervention**: Replace C01 (TT) with a learned constant vector v_const[l] per layer (no input dependence).
**Predicted outcome**: If the per-layer trim-tab pattern persists with constant steering (same v_const for all inputs), then the TT's input-dependence is irrelevant — the pattern is purely layer-specific, not input-specific. This would mean the TT's high R² is learned structure but not USEFUL learned structure.
**Testability**: Train 28 constant vectors (one per layer) by averaging v_actual over training data. Evaluate steering with these constants. Compare to TT-based steering.

---

## 7.4 Intervention Points

| Node | Feasibility of Intervention | Expected Leverage | Method |
|------|---------------------------|-------------------|--------|
| N1 (base model) | HIGH | HIGH (CF-3 shows model choice dominates) | Swap to different model family |
| N3 (baseline accuracy) | MEDIUM | HIGH (everything downstream depends on it) | Prompt engineering, temperature, few-shot |
| N6 (contrastive decision) | HIGH | VERY HIGH (unresolved counterfactual CF-4) | Train different split strategies |
| N8 (L8 identification) | HIGH | MEDIUM (already identified; optimization is next) | Refine to sub-layer or head level |
| N13 (steering parameters) | HIGH | MEDIUM (α, layer combo optimization) | RL-based α optimization |

### Delay Analysis

The critical delay in the causal graph is between N5 (per-layer sweep) and N8 (L8 identification). This is ~4 hours. More importantly, the delay between N18 (contrastive eval start) and N25 (results) is currently unbounded — this is the primary blocker.

The overall project follows a **sequential pipeline** with few parallel branches. The causality is largely linear (N1→N2→N3→N4/N5/N6), meaning any bottleneck at an early node blocks the entire downstream.
