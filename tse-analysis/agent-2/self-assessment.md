# Phase 11: Recursive Self-Assessment

---

## 11.1 Structural Self-Analysis

### Completeness of Atomic Decomposition

**Completed atoms**: 20 atoms (A01-A20). 
**Missing atoms identified**:
- **A21 (attention_head)**: The current decomposition treats "attention computation" as atomic (A13), but steering operates at the layer level. Per-head steering is a plausible variant not captured as an atom.
- **A22 (projection_matrix)**: W_k and W_v projections are treated as infrastructure, but they may be the causal mechanism (H-2 suggests K/V amplification). They should be atoms.
- **A23 (training_loss)**: The loss function (MSE on velocity) is not decomposed. The choice of MSE vs cosine vs L1 has implications for what the TT learns.
- **A24 (token_position)**: The decomposition lacks the time/position dimension. Steering effects may vary by generation step, an atom-level property.

**Correctness of Junctions**:
- J04 (compositional: chat_template + baseline_eval) should be **causal** (E2 in causal map), not just compositional. The chat template CAUSED the 4%→73% jump.
- J24 (antagonistic: trim_tab vs death_layer) is correctly typed but the antagonism may be more nuanced — L8 and L9 could be "complementary" under different conditions (e.g., L8 with positive α, L9 with negative α work together).
- Missing junction: **no hierarchical junction between A06 (alpha) and A18 (hidden_manifold)** — the alpha determines the perturbation size relative to the manifold curvature, which is a hierarchical relationship.

**Pyramid Level Balance**:
- The pyramid has 20 Level-0 atoms, 10 Level-1 composites, 3 Level-2 composites, and 1 Level-3 peak. This is well-balanced.
- However, Level-1 has many "pipeline" composites (C01-C08) but few "theoretical" composites. A "Steering Theory" composite (combining A17, A18, B1-B9 assumptions) would improve the pyramid.

### Missing Composites

A composite notably absent: **Steering Theory** — an explicit framework that explains WHY steering works. The project has empirical results but no theoretical composite that integrates findings 1-6 into a coherent explanation.

---

## 11.2 Relational Self-Assessment — Lens Cascade Applied to the Analysis

### What Did This Analysis Miss?

Applying the 10 lenses to the analysis output:

| Lens | What the Analysis Missed |
|------|-------------------------|
| ANALOGICAL | The analysis didn't explore **gradient descent** as an analogy. TT predicts the "gradient" of hidden state evolution; steering is a "gradient descent step" toward correct answers. The learning rate = α. This is a potentially exact mapping. |
| DIALECTICAL | The analysis didn't properly capture the tension between "steering requires capability" (Finding 3) and "cross-model transfer works" (Finding 5). If a model without capability (SmolLM2) produces TTs that work on capable models, then the TT captures model-agnostic velocity structure, not model-specific capability. This undermines the "capability threshold" framing — the TT itself may have the capability, not the model. |
| BLENDING | The analysis didn't blend the StreamFusion adapter research (the other major codebase component) with the steering research. The project has ~80 adapter variants (DoRA, BoRA, EBVoRAN, etc.) and a mature StreamFusion architecture. Steering could be viewed as a SPECIAL CASE of activation-level adaptation. |
| SYSTEMS | **The analysis missed the feedback loop between ablation experiments and TT training.** No online learning loop was proposed in the causal map. |
| ABDUCTIVE | The primary abductive inference (H_mech: L8 is a transition layer) is plausible but the analysis should have considered a **fifth hypothesis**: L8 works because the 7B's TT has lowest error at L8 (per-layer R² varies, and the overall R²=0.855 masks significant per-layer variation). This is testable immediately. |
| TRAJECTORY | The analysis didn't explore the **long-term trajectory** of the project relative to the broader field of LM steering. Is this a niche technique or a general paradigm? The analysis can't answer this without external signal. |
| METACOGNITIVE | **Persistent self-blind spot**: The analysis treats the TT as a black box (predictor analysis in Phase 8 was theoretical, not empirical). The analysis should have run actual PCA on TT representations. |
| INSPIRATION | Missed **protein folding** analogy: hidden state trajectory = protein folding pathway, steering = chaperone protein that guides folding. The trim-tab is the "hydrophobic core" whose correct folding determines the entire protein structure. |
| ADVERSARIAL | Missed the **stealth attack** surface: since steering requires 88% token divergence, an attacker could detect steering by monitoring token divergence and construct countermeasures. |
| PARADOXICAL | The analysis identified D4 (R² vs steering effect paradox) but didn't push it to the limit: **if the TT perfectly predicts velocities, and velocities ARE the model's dynamics, then any "steering" at α>0 is attacking the model with its OWN dynamics — it's self-sabotage that somehow helps.** This is the deepest unresolved paradox. |

### Blind Spots Discovered

| # | Blind Spot | Why Missed | How to Catch Next Time |
|---|-----------|-----------|----------------------|
| BS-TSE-1 | Per-layer R² of TT not computed | Project debrief only reports overall R²; analysis didn't require per-layer breakdown | Always request per-layer metrics when TT architecture is layer-aware |
| BS-TSE-2 | No cross-reference with adapter research (StreamFusion) | Adapter research is in separate codebase area; TSE treated steering as isolated project | Include cross_subjects parameter pointing to other project components |
| BS-TSE-3 | No external literature review | TSE is self-contained; no PubMed/arXiv queries for related work | Invoke deep-cross-research skill for Phase 2 analogical lens |
| BS-TSE-4 | Theoretical analysis without empirical verification | Phase 8 conclusions (e.g., PCA of TT, K/V amplification) are theoretical | Add "grounding requirement" to Phase 8: every finding must cite either code or data |

---

## 11.3 Potential Self-Assessment — What Could the Analysis Have Become?

### Alternative Framings Not Explored

1. **Steering as fine-tuning**: Frame the entire project as "parameter-efficient fine-tuning without weight updates" — the TT and steering mechanism are a form of hypernetwork that generates activation shifts. This framing connects to the extensive PEFT literature (LoRA, AdaLoRA, etc.).

2. **Steering as adversarial attack**: Frame steering as a white-box adversarial attack on the generation process, where TT predicts the most efficient attack direction and α controls attack strength. The fact that "+20pp" is the result of "manipulating internal activations" is ethically and methodologically related to adversarial machine learning.

3. **Steering as cognitive science experiment**: Frame the per-layer effect as a "lesion study" of the transformer's reasoning process. L8 lesion improves reasoning (paradoxical), L9 lesion impairs it. This is a methodology from cognitive neuroscience applied to transformers.

### What Was NOT Found and Why

- **No optimal multi-layer combination discovered**: The project tested multi-layer combos only naively (same α for all layers). The TSE analysis identified per-layer α vectors as the next step, but didn't compute the theoretical optimum.
- **No mechanistic explanation of L8 vs L9**: Despite 8 phases of analysis, we still don't know WHY L8 and L9 differ. The H-1 (frequency-specific) and H-2 (K/V amplification) hypotheses are the best candidates but untested.
- **No cross-task validation**: The analysis assumed GSM8K generalizes to other reasoning tasks, but the SVAMP result (L8: +4pp vs +20pp on GSM8K) shows significant magnitude variation. The task-dependence of the trim-tab effect is unexplored.

**Why these were not found**: The project debrief is empirically rich but theoretically sparse. It answers "what" extensively but "why" minimally. The TSE analysis compensated by generating hypotheses (Hyperstitional Bridge), but without running experiments, these remain speculative.

---

## 11.4 Proposed Updates to TSE

Based on this run, the following improvements to the Triadic Synthesis Engine are proposed:

| Update | Rationale | Expected Improvement |
|--------|-----------|---------------------|
| **U1**: Add "empirical grounding check" before Phase 8 | Phase 8 made theoretical claims without empirical support (PCA, K/V amplification) | Grounds analysis in data rather than speculation |
| **U2**: Require cross_subjects parameter when project has multiple components | The steering analysis missed the adapter research entirely | Prevents blind spots from project decomposition |
| **U3**: Add "counterfactual order analysis" to Phase 7 | CF-2 (reverse layer order could have killed project) is a critical finding type | Reveals experimental design fragility |
| **U4**: Pre-built lens templates with domain-specific heuristics | Inspirational lens (8) was weak because the analyst lacks domain breadth | Better quality forced collisions from diverse domains |
| **U5**: Add "null model check" to Phase 1 atom definitions | Every atom should have a "what if this didn't exist?" variant | Makes the VOID analysis more actionable |

---

## 11.5 Confidence Assessment

### Overall Confidence: 7/10

| Phase | Confidence | Rationale |
|-------|-----------|-----------|
| Phase 0 (VOID) | 8/10 | Assumptions surfaced from explicit project documentation; some implicit assumptions may remain hidden |
| Phase 1 (Pyramid) | 7/10 | Complete decomposition of steering pipeline; missing adapter cross-reference (BS-TSE-2) |
| Phase 2 (Lenses) | 7/10 | 10 lenses applied thoroughly; external knowledge gaps weaken analogical/inspirational lenses |
| Phase 3 (Master Regulators) | 8/10 | Top-5 regulators are clearly justified; ranking is subjective but defensible |
| Phase 4 (Divergent) | 6/10 | Many speculative variants; quality scores are rough estimates |
| Phase 4b (Emergent) | 5/10 | Unconventional recombinations are genuinely novel; emergence classification is theoretical |
| Phase 5 (Convergent) | 7/10 | Filter thresholds are reasonable; top-5 ranking is justified |
| Phase 6 (Disparity) | 8/10 | Resolved/unresolved classification is clear; D4 paradox is correctly identified as critical |
| Phase 7 (Causal) | 8/10 | Causal DAG is well-grounded in experimental pipeline; counterfactuals are insightful |
| Phase 8 (Mechanistic) | 5/10 | Theoretical analysis WITHOUT empirical validation (U1 violation); PCA and K/V claims are speculative |
| Phase 9 (Temporal) | 8/10 | Resource budget is realistic; decision tree handles failure cases |
| Phase 10 (Hypersitional) | 6/10 | Hypotheses are specific and falsifiable; H-3 requires significant additional resources |
| Phase 12 (Final) | 7/10 | Integration is comprehensive; recommendations are actionable |

### What Would Increase Confidence

1. **Run the null model experiments** (Phase B1) — this alone would increase Phase 8 confidence from 5/10 to 8/10.
2. **Complete the contrastive evaluation** (Phase A1) — resolves the single largest unknown.
3. **Run PCA on TT predictions** (Phase B2) — grounds the frequency hypothesis in data.
4. **External literature review** — check if similar per-layer steering effects have been observed in other labs.

### Negative Space (What Was NOT Found)

| Not Found | Why | Should It Be Investigated? |
|-----------|-----|---------------------------|
| Interaction between steering and model quantization | Analysis focused on steering accuracy, not steering + quantization interaction | YES — steering a 4-bit model may introduce quantization error amplification |
| Steering as debiasing tool | Could steering reduce gender/racial bias in LM outputs? | YES — ethical implications |
| Comparison to finetuning baselines | How does steering compare to LoRA finetuning on GSM8K? | YES — establishes relative value of steering vs tuning |
| Upper bound of steering improvement | What's the max achievable accuracy with optimal steering? | YES — theoretical limit question from debrief's open questions |
