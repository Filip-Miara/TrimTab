# DIFFUSER-5: ADVERSARIAL STRESS TEST (Hostile Review)

**Subject**: RankAdaptation — Velocity-based latent steering for LM reasoning
**Role**: Hostile peer reviewer — burden of proof is on the paradigm
**Standard**: Extraordinary claims require extraordinary evidence
**Date**: 2026-06-14

---

## PREAMBLE: The Asymmetric Burden

The steering paradigm claims: *Modifying KV-cache entries with predicted velocity vectors at specific layers causally improves reasoning accuracy by +20pp.*

This is an extraordinary claim for three reasons:
1. **Mechanistic implausibility**: A 3584-dimensional hidden state is perturbed by ~10% of a typical velocity norm at a single layer. That this perturbation survives 27 subsequent layers of nonlinear computation to improve a token-level classification task is *a priori* surprising.
2. **Effect magnitude**: +20pp on a saturated metric (73% baseline → 93% is impossible; 45% → 65% is a 44% relative reduction in error). The effect is large enough to be suspicious.
3. **Lack of mechanism**: No causal chain explains how KV-cache modification → attention pattern change → improved token selection. Without mechanism, the default assumption must be artifact.

I will hold every claim to the standard: **falsifiable, replicated, mechanism-specified, confound-controlled, and multiple-comparison-corrected**.

---

## 1. THE 10 STRONGEST ARGUMENTS AGAINST THE STEERING PARADIGM

Ranked by how hard they are to refute (1 = hardest, 10 = easiest to dismiss).

### Argument #1 (Refutation Difficulty: 10/10): The Smoothness Confound

**Claim**: The TT predicts velocity v̂ = h_{t+1} − h_t with R² = 0.855.

**Critique**: Any sufficiently smooth trajectory has high next-step predictability. The naive baseline — predict h_{t+1} = h_t (zero velocity) — would produce R² > 0.9 on a locally Lipschitz trajectory. The TT's R² is *guaranteed* to be high because hidden states change slowly across adjacent tokens. The perceived "learnability" of velocities is an artifact of trajectory smoothness, not evidence that the TT captures causal reasoning dynamics.

**Why this is devastating**: If R² is inflated by smoothness, the entire paradigm collapses. The TT is not "learning where reasoning is going" — it's learning that h_{t+1} ≈ h_t. Steering with this signal is steering with approximate identity, which means the observed L8 +20pp is either (a) random perturbation that happens to help on some layers, or (b) an artifact of some other confound.

**Confirming experiment**: Compute R²_naive = 1 − MSE(v̂=0) / Var(h_{t+1} − h_t). If R²_naive > 0.85, the TT's apparent performance is smoothness-exploitation. This experiment costs 15 minutes and zero GPU (existing data).

**Current defense weakness**: Agent 5's E3 (naive baseline) is proposed but NOT RUN. The project has never computed this.

---

### Argument #2 (Refutation Difficulty: 9.5/10): The Random Baseline Equivalence

**Claim**: TT-predicted velocity direction is specifically steering toward "correct reasoning."

**Critique**: No experiment distinguishes TT-predicted direction from random direction of equal norm. The +20pp at L8 could be produced by ANY perturbation of the KV cache at L8 with magnitude |α·v̂|. Consider: KV-cache modification at a specific layer (L8) with magnitude ~10% of typical hidden state norm changes the attention pattern at that layer. This change propagates through ~19 remaining layers. The effect on final accuracy depends on whether the perturbation *happens to align with* or *happen to oppose* the natural computation at L8. With 28 layers sampled, the probability of at least one layer showing a "positive" effect of +20pp from random perturbation is calculable.

**Why this is devastating**: The central causal claim — "velocity direction causes improvement" — has zero controls. Without random baseline, every steering result is consistent with the null hypothesis: KV-cache perturbation of any kind at any layer changes accuracy in a layer-dependent but direction-independent way.

**Confirming experiment**: 4 conditions (baseline, random vector of norm |α·v̂|, standard TT, contrastive TT) × 28 layers. If the distribution of random-vector accuracy changes matches TT accuracy changes, the paradigm is noise. Cost: 3.7 GPU-hours.

**Current defense weakness**: Explicitly identified by agents 3, 4, and the meta-synthesis as the critical missing control. Not run.

---

### Argument #3 (Refutation Difficulty: 9/10): The Position-Frequency Artifact

**Claim**: The TT learns causal velocity dynamics of the hidden state that correspond to reasoning processes.

**Critique**: Transformer hidden states are dominated by positional encoding patterns. At position t, the hidden state h_t encodes: token identity, positional embedding, accumulated context, and layer computation residual. The *change* in hidden state h_{t+1} − h_t is dominated by: the difference between positional encodings at t and t+1, plus the residual stream update from the current layer. For typical transformers, positional encoding differences follow regular patterns (sinusoidal curves with position-dependent frequency). The TT may be learning these positional frequency patterns, not semantic velocity.

**Why this is devastating**: If the TT is a frequency predictor, then "steering" is injecting a position-dependent perturbation that happens to correlate with correctness at L8 but would fail at any layer where the frequency pattern is misaligned. The trim-tab/death-layer pattern would be *position-dependent, not reasoning-dependent*.

**Confirming experiment**: Shuffle token positions in test trajectories. If R²_shuffled ≥ 0.9 × R²_original, the TT is learning positional frequency, not dynamics. Cost: 30 min.

**Current defense weakness**: Agent 5's E1 proposed but not run.

---

### Argument #4 (Refutation Difficulty: 8.5/10): The 88% Token Divergence = Model Damage

**Claim**: Steering improves accuracy by guiding the model toward correct reasoning.

**Critique**: At L8 with α=0.1, 88% of generated tokens differ from the unsteered trajectory. This is not steering — this is *trajectory disruption*. The model is saying almost completely different things. When 88% of tokens change, "accuracy improvement" on a fixed answer set measures something closer to: *does a randomly perturbed model happen to produce more correct answers by chance?*

Consider: If a model's output changes 88% of its tokens, the probability of getting any specific answer right is approximately the baseline accuracy (73%) × probability that the perturbation doesn't destroy the answer + baseline error rate (27%) × probability that the perturbation accidentally corrects the answer. With 88% token divergence, almost every answer is completely different — accuracy on the new answers can be higher or lower depending on the layer and perturbation sign, with no mechanistic connection to "reasoning improvement."

**Why this is devastating**: +20pp at L8 coexisting with 88% token divergence is consistent with *accuracy lottery* — each layer's perturbation independently samples a new accuracy from a distribution whose mean is 45% (baseline) and whose variance is layer-dependent. L8's distribution has mean 65%. But this doesn't mean L8 steering *improved reasoning* — it means L8 perturbation happened to sample a better accuracy. The mechanism is not steering toward correctness; it's perturbation that moves the model into a different region of output space that happens to have higher accuracy for L8.

**Confirming experiment**: Measure accuracy variance under random perturbations of equal norm at L8. If accuracy at L8 under random vectors has std > 10pp, the "accuracy improvement" from TT steering is within the noise of random perturbation at that layer. Cost: 1 GPU-hour.

**Current defense weakness**: No accuracy-variance-under-perturbation measurement exists.

---

### Argument #5 (Refutation Difficulty: 8/10): The L8/L9 Adjacency Paradox Is a Boundary Artifact

**Claim**: L8 and L9 have genuinely different functional roles — L8 is a "trim-tab" for reasoning, L9 is a "death layer."

**Critique**: Adjacent transformer layers do not serve fundamentally different functional roles. The L8/L9 boundary (+20pp vs -23pp) is far more consistent with a *boundary artifact* in the steering implementation than a genuine functional difference. Consider: KV-cache modification at L8 affects the attention computation at L8, which then passes through L9-L27. KV-cache modification at L9 affects L9's attention, passing through L10-L27 (one fewer layer of processing). If L9 is more sensitive to perturbation because it has fewer layers to "recover" from the steering modification, the death-layer pattern is an artifact of steering position, not layer function.

**Supporting evidence**: L15+ layers ALL show similarly destructive effects (-23pp+). These are the last layers before the output head. The pattern is: layers closer to the output show more negative effects. This is consistent with: *any perturbation applied closer to the output has less time to be integrated/compensated for by the remaining computation*. The L8 effect is positive not because L8 is special, but because it's the first layer that simultaneously (a) participates in reasoning computation AND (b) has enough downstream layers to "absorb" the perturbation.

**Confirming experiment**: Test L8 at variable positions by inserting the same perturbation at different layer depths. If the effect monotonically decreases with distance from output regardless of layer identity, the trim-tab pattern is a position artifact. Cost: 2 GPU-hours.

**Current defense weakness**: No distance-from-output control in any experiment. The meta-synthesis notes the L8/L9 paradox but attributes it to functional difference, not procedural artifact.

---

### Argument #6 (Refutation Difficulty: 7.5/10): Per-Layer R² Does Not Correlate With Steering Success

**Claim**: Learnable velocity dynamics (high R²) are a prerequisite for effective steering.

**Critique**: The Math-1.5B model has R² = 0.892 (higher than Qwen2.5-7B's R² = 0.855) but shows ZERO trim-tabs under steering — all layers are neutral or harmful. This single counterexample falsifies the claim that R² predicts steering success. More importantly: if R² is decorrelated from steering efficacy, then the entire theoretical foundation for steering (velocity = direction of reasoning improvement) is unsupported.

But the damage goes deeper: the SmolLM2 model has R² = 0.94 (highest of all) but baseline accuracy is 4% — steering is impossible. The two models with the highest R² both fail to produce trim-tabs (SmolLM2: no capability, Math-1.5B: no trim-tabs). The model with the lowest useful R² (7B: 0.855) is the only one that works. This is evidence AGAINST the velocity-based theory.

**Why this is devastating**: The central theoretical proposition — "velocity learnability enables steering" — is contradicted by the project's own data. R² does not predict steering success. Without this link, the paradigm has no theoretical core — only unexplained empirical observations.

**Confirming experiment**: Compute the Spearman correlation between per-layer R² and per-layer Δ accuracy for the 7B model. If ρ < 0.3 (or negative), the theoretical foundation is unsupported. Cost: 0 GPU-hours (existing data).

**Current defense weakness**: Meta-synthesis identifies this as a critical unresolved finding (F5: R² paradox resolved by K/V amplification), but the "resolution" is untested speculation.

---

### Argument #7 (Refutation Difficulty: 7/10): The SVAMP Generalization Is Illusory

**Claim**: The trim-tab pattern generalizes from GSM8K to SVAMP (L8: +4pp, L9: -14pp).

**Critique**: SVAMP is also a math word-problem dataset — same format, same reasoning structure, same token distribution as GSM8K. The +4pp on SVAMP (vs +20pp on GSM8K) is a 5× reduction in effect size. This is consistent with: *any dataset-specific perturbation that transfers to similar datasets with attenuated effect*. The reduction from +20pp to +4pp is suspicious — it suggests the steering effect is specific to GSM8K's problem distribution, not general to mathematical reasoning.

More importantly: SVAMP and GSM8K share the same underlying model capability (arithmetic word problems). Steering on SVAMP "generalizing" from GSM8K means: the perturbation that helps on GSM8K also helps (less) on very similar problems. This is not evidence for general reasoning improvement — it's evidence for perturbation specificity within a narrow domain.

**Why this is devastating**: If the steering effect doesn't transfer to non-math tasks (ARC, BBH, MMLU), the claim "steering improves reasoning" is false. The effect is "steering improves GSM8K-like accuracy on models that can already do GSM8K." This is a much weaker claim.

**Confirming experiment**: Evaluate L8 steering on ARC (science questions), BBH (BIG-Bench Hard, diverse reasoning), and MMLU (multidisciplinary knowledge). Any null result on non-math tasks confines the paradigm severely. Cost: 2 GPU-hours (Phase B5).

**Current defense weakness**: Cross-task evaluation is Phase B5 — not yet prioritized for Phase A.

---

### Argument #8 (Refutation Difficulty: 6.5/10): The α=0.1 Choice Is an Optimization Artifact

**Claim**: α=0.1 is a reasonable default for steering magnitude.

**Critique**: α=0.1 was not chosen because it's optimal — it was chosen *because it doesn't destroy output*. At α=0.1, 88% of tokens still change. At α=0.5 or α=1.0, token divergence approaches 100% and accuracy collapses. The "optimal" α=0.1 is the highest value that doesn't completely break the model, not a principled scaling factor.

This is analogous to: "I found that hitting my TV with a hammer at force level 3 improves the picture by 20%. Higher forces break the screen." The α=0.1 discovery is an *edge of stability* phenomenon, not a steering parameter. The +20pp at α=0.1 may be a proximity-to-collapse effect: the model is just barely maintaining coherent generation, and L8 perturbation happens to push it toward slightly better answers on the edge of chaos.

**Why this is devastating**: If α=0.1 is the maximum tolerable perturbation before collapse, then the "steering effect" is operationally indistinguishable from *critical slowing down* — a dynamical systems phenomenon where systems near a phase transition show amplified sensitivity to perturbations. L8 may be the layer where the model is most sensitive (largest Lyapunov exponent), and the +20pp is sensitivity, not steering.

**Confirming experiment**: Sweep α from 0.01 to 0.5 at L8. If accuracy peaks at exactly the maximum α before collapse (α ≈ 0.1-0.15) and drops on both sides, the effect is critical slowing down, not optimal steering. If there's a genuine internal optimum (improves at α=0.05, degrades at α=0.2), the effect is more likely real. Cost: 1 GPU-hour.

**Current defense weakness**: No α sweep beyond α=0.1. The meta-synthesis flags this but hasn't resolved it.

---

### Argument #9 (Refutation Difficulty: 6/10): The Chat Template Confound Cascade

**Claim**: The 73% baseline is accurate; steering improves from this baseline.

**Critique**: The baseline was 4% before the chat template fix. The fix involved `apply_chat_template()`, which changes the prompt structure substantially. The baseline jump (4% → 73%) is itself suspicious — a simple formatting change produced a 69pp improvement. This suggests the model is *extremely* sensitive to input formatting. If the model is formatting-sensitive, then KV-cache modification — which changes the internal representation at specific layers — may work by *format-mimicking* rather than *reasoning-amplifying*.

Specifically: L8 KV-cache modification may produce hidden states that the model "interprets" as being closer to the chat-template-formatted distribution. The +20pp improvement would then be the model recovering accuracy that it would have had with a better prompt, not improved reasoning from steering.

**Why this is damaging**: The 4% → 73% baseline jump (69pp) dwarfs the +20pp steering effect. A formatting change produces 3.5× more improvement than steering. This suggests: (a) the model is capable of much higher accuracy with the right internal state, (b) steering might be approximating the "right internal state" at L8, (c) L9 might push away from it. The finding is real but the INTERPRETATION is wrong: it's not "steering improves reasoning" but "prompt sensitivity artifacts concentrated at L8."

**Confirming experiment**: Evaluate L8 steering with the RAW text format (no chat template). If L8 improves from 4% to, say, 15% — similar relative improvement — the effect is formatting-independent. If L8 steering does nothing on raw format (0% → 0%), the effect is chat-template-interaction artifact. Cost: 1 GPU-hour.

**Current defense weakness**: No experiment controls for prompt-format interaction with steering.

---

### Argument #10 (Refutation Difficulty: 5/10): The Cross-Model Transfer Is Trivial Projection

**Claim**: SmolLM2's TT transferring to Qwen2.5-7B proves velocity dynamics are model-agnostic.

**Critique**: The transfer works by projecting 960-dim SmolLM2 hidden states → 3584-dim Qwen2.5-7B hidden states using a learned linear projection. This is a 960×3584 matrix with ~3.4M parameters. A random projection of equal size would also map any input to any output up to the projection's rank. The "transfer" may be the linear projection doing all the work — learning to map SmolLM2's velocity patterns to whatever works for Qwen2.5-7B.

The fact that L8 remains the best layer after transfer is the only nontrivial result, but even this is expected: L8 is the best layer on the 7B's OWN TT. A projection trained to map SmolLM2 velocities to 7B velocities would naturally align the patterns at whatever layer the 7B finds most responsive. The transfer experiment tests projection quality, not universality of dynamics.

**Why this is damaging**: The cross-model transfer result is frequently cited as evidence for universal velocity dynamics, but the experimental design conflates projection learning with dynamic universality. A proper test would use a fixed projection (e.g., PCA-based or zero-padding) and compare TT transfer quality with vs without learned projection.

**Confirming experiment**: Repeat cross-model transfer using zero-padding (append zeros to 960-dim to make 3584-dim) instead of learned projection. If L8 remains best, dynamics are universal. If not, the transfer was projection-driven. Cost: 1 GPU-hour (existing infrastructure).

**Current defense weakness**: No fixed-projection control. All transfer results use learned projection with substantial capacity.

---

## 2. CONFIRMING EXPERIMENTS (Summary Table)

| # | Argument | Critical Experiment | Result That Confirms Critique | Cost | Status |
|---|----------|-------------------|-------------------------------|------|--------|
| 1 | Smoothness confound | R² naive baseline (predict v̂=0) | R²_naive > 0.85 | 0 GPU-hr | **NOT RUN** |
| 2 | Random baseline equivalence | 4-condition × 28-layer sweep | TT accuracy distribution ≈ random distribution | 3.7 GPU-hr | **NOT RUN** |
| 3 | Position-frequency artifact | Shuffle token positions, recompute R² | R²_shuffled ≥ 0.9 × R²_original | 0.5 GPU-hr | **NOT RUN** |
| 4 | 88% divergence = damage | Accuracy variance under random L8 perturbation | std(L8 accuracy under random) > 10pp | 1 GPU-hr | **NOT RUN** |
| 5 | L8/L9 boundary artifact | Insert same perturbation at variable depths | Effect decreases monotonically with depth regardless of layer | 2 GPU-hr | **NOT RUN** |
| 6 | R² does not predict success | Per-layer ρ(R², Δ accuracy) | ρ < 0.3 | 0 GPU-hr | **NOT RUN** |
| 7 | SVAMP generalization illusory | Evaluate on ARC, BBH, MMLU | No significant improvement on non-math tasks | 2 GPU-hr | **NOT RUN** |
| 8 | α=0.1 is critical slowing down | α sweep (0.01-0.5) at L8 | Accuracy peak at collapse boundary | 1 GPU-hr | **NOT RUN** |
| 9 | Chat template confound | L8 steering on raw text format | L8 improvement << +20pp on raw format | 1 GPU-hr | **NOT RUN** |
| 10 | Transfer is projection-driven | Zero-padding transfer (no learned projection) | L8 not best layer without projection | 1 GPU-hr | **NOT RUN** |

**Total cost to validate ALL 10 critiques**: ~12.2 GPU-hours
**Total cost to validate the SINGLE most damaging critique**: 0 GPU-hours (Argument #1 — already exists in data)

---

## 3. THE CHEAPEST FALSIFICATION EXPERIMENT

### Winner: Smoothness Baseline (Argument #1) — **Cost: $0, Time: 15 minutes**

**Procedure**: From existing trajectory data:
1. Extract all consecutive hidden state pairs (h_t, h_{t+1}) from all 25 trajectory files
2. Compute v_actual = h_{t+1} − h_t
3. Compute MSE_tt = mean(||v_actual − v̂_tt||²) where v̂_tt is the TT's prediction
4. Compute MSE_naive = mean(||v_actual||²) (the error from predicting v̂=0, i.e., h_{t+1}=h_t)
5. Compute R²_naive = 1 − MSE_naive / Var(v_actual) and R²_tt = 1 − MSE_tt / Var(v_actual)

**If R²_naive > 0.85**: The TT is smoothness-exploiting. The entire project's interpretation of "learnable velocities" is invalid. Steering is perturbation, not guidance.

**If R²_naive < 0.3 AND R²_tt > 0.8**: The TT genuinely learns something beyond smoothness. The paradigm survives this specific critique.

**Expected outcome (my prediction)**: Given that hidden states change slowly (adjacent tokens share most of their content), R²_naive is likely 0.75-0.90. If >0.85, this is the single cheapest falsification of the entire paradigm.

### Second-Cheapest: Per-Layer ρ(R², Δ Accuracy) — **Cost: $0, Time: 30 minutes**

**Procedure**: From existing per-layer steering results:
1. For each of the 28 layers, compute R² (from TT evaluation on that layer's data) — this requires layer-specific TT predictions, which may not exist. Alternative: use per-layer R² of the global TT on each layer's hidden states.
2. For each of the 28 layers, compute Δ accuracy = accuracy_steered − accuracy_baseline
3. Compute Spearman ρ between R² per layer and Δ accuracy per layer

**If ρ < 0.2**: The theoretical foundation (velocity learnability → steering efficacy) is unsupported. The paradigm has no predictive theory.

**If ρ > 0.4**: R² is a useful proxy for steering success. The theoretical foundation is supported.

---

## 4. HIDDEN CONFOUNDS: What Could Cause +20pp at L8 That Is NOT Steering?

### C1: Baseline Regression to the Mean (Severity: HIGH)

The baseline of 45% was measured once on 100 problems. The standard error is ~4.4pp. The "true" baseline could be 50-55% with a lucky low sample. The steering result at L8 (65%) could be an unlucky high sample. Combined, the +20pp could be 2× standard error fluctuation. This is the simplest confound and the hardest to rule out without formal significance testing.

**Control**: Compute baseline over 500 problems (not 100). Phase B3.

### C2: Seed x Layer x Evaluation Interaction (Severity: HIGH)

The evaluation uses a fixed random seed for generation. Different layers + same seed = the random sampling is identical up to the steering perturbation point. This creates a *correlated noise structure* — L8's accuracy advantage might come from the interaction of steering direction with a specific random seed's sampling pattern.

**Control**: Repeat top-3 layers across 10 different random seeds. If the +20pp at L8 is robust to seed, the effect is real. If seed variance at L8 is >10pp, the effect is fragile.

### C3: Ordering Effects in Per-Layer Sweep (Severity: MEDIUM-HIGH)

Per-layer sweeps evaluate layers in order L0 → L27. If there's any time-dependent drift (GPU temperature, memory pressure, CUDA kernel caching), the earlier layers (L0-L8) are evaluated under systematically different conditions than later layers (L15-L27). The +20pp at L8 could be a position-in-sweep artifact.

**Control**: Randomize layer evaluation order. If L8 still emerges as best, the effect is genuine.

### C4: Evaluation Metric Ceiling Effects (Severity: MEDIUM)

GSM8K has a fixed answer set (sometimes as few as 2-3 answer choices per problem). Models at 73% baseline may be near the "interpretable prediction" ceiling for 7B models. The +20pp at L8 may be a combination of: (a) perturbation breaks the model out of a local accuracy minimum, (b) the maximum achievable accuracy with this model is ~80-85%. If the natural ceiling is 85%, then +20pp from a 45% floor is less impressive than it appears.

**Control**: Compute the model's maximum achievable accuracy via supervised fine-tuning on GSM8K training set. If SFT accuracy is, say, 82%, then steering to 65% is only 50% of the gap to ceiling (not impressive for an "amplification" mechanism).

### C5: Token Position Sampling Bias (Severity: MEDIUM)

GSM8K problems have variable answer lengths. If L8-steered answers are systematically shorter (more token divergence → earlier branching → different stopping), the accuracy comparison is biased: shorter answers have fewer chances to make errors. If L8 steering reduces average answer length, +20pp could follow from reduced error exposure, not improved reasoning.

**Control**: Measure average answer length (in tokens) under L8 steering vs baseline. If L8 answers are >20% shorter, control for length by truncating baseline answers to match.

### C6: GPU Nondeterminism Amplified by Steering (Severity: MEDIUM)

KV-cache modification at L8 changes the attention computation at L8, which changes the floating-point accumulation in the softmax. GPU operations are nondeterministic at the bit level (especially with Tensor Cores). The 88% token divergence means that L8 steering creates a completely different computational path. The measured +20pp could be: steering at L8 sends the computation into a different nondeterministic trajectory that happens to yield higher accuracy for that specific seed.

**Control**: Run L8 steering 10 times with different GPU nondeterminism settings (Tensor Cores on/off, deterministic algorithms). Measure variance of L8 accuracy across runs. If deterministic accuracy matches nondeterministic accuracy, this confound is ruled out.

### C7: Prompt Template X Layer Interaction (Severity: MEDIUM-HIGH)

The chat template adds specific formatting tokens: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`. These tokens occupy specific positions. If L8 is the layer where these formatting tokens' positional information is consumed, then KV-cache modification at L8 disproportionately affects the model's awareness of "what role am I playing?" — potentially improving assistant-mode behavior (more careful answering, fewer hallucinations) rather than improving mathematical reasoning.

**Control**: Test L8 steering on non-conversational tasks (e.g., factual QA without chat template) to isolate reasoning effects from role-awareness effects.

### C8: Data Leakage Between TT Training and Steering Evaluation (Severity: LOW-MEDIUM)

The TT is trained on generation trajectories. The steering evaluation uses GSM8K test problems that may overlap with the training generation problems. If the model generated trajectories on problems similar to or overlapping with the evaluation problems, the TT has implicit access to evaluation-distribution information.

**Control**: Verify problem-level separation between TT training data and steering evaluation data. If >10% overlap, contamination is possible.

### C9: The "Lottery Ticket" Layer Selection (Severity: MEDIUM)

With 28 layers tested, the probability that at least one layer shows a "significant" effect (|z| > 2) is approximately 1 − (1 − 2×0.025)^{28} = 1 − 0.95^{28} = 76% under the null of independent layers. The project found L8 at z=4.4 (p ≈ 10^{-5}), which does survive correction. But the claim is not just "L8 works" — the claim is "the trim-tab/death-layer pattern is real." With 28 layers, finding BOTH a positive trim-tab (L8) and a negative death layer (L9) is: P(L8 positive | H0) × P(L9 negative | H0) ≈ 0.025 × 0.025 = 0.000625 for this specific pair. But with 27 adjacent layer pairs, the probability of ANY such opposite-sign pair is ~27 × 0.00625 = 0.17 (17%) — not significant.

**The trim-tab/death-layer opposition pattern itself may be a multiple-comparisons artifact.**

---

## 5. THE REPLICATION CRISIS SCENARIO

If an independent researcher attempted to replicate this project, here is what would fail, in order of probability:

### Failure #1 (95% probability): Baseline Does Not Reach 73%

The independent researcher uses Qwen2.5-7B-Instruct (same model) and GSM8K (same dataset) with the chat template. Their baseline is 45-55% (not 73%). The 73% baseline reported by this project may depend on:
- A specific version of the chat template (transformers version-dependent)
- A specific random seed for generation
- A specific subset of GSM8K test problems
- A specific max_new_tokens setting (400 is generous)

With a 45-55% baseline, the +20pp at L8 becomes +10-15pp (effect shrinks). The statistical significance drops. The "capability threshold" explanation (steering requires baseline > 40%) is invoked ad hoc.

### Failure #2 (80% probability): Random Baseline Matches TT Steering

The independent researcher runs the 4-condition protocol (proposed in meta-synthesis as Phase A1). They find: random vectors of equal norm produce statistically indistinguishable accuracy changes from TT predictions across all 28 layers. The trim-tab pattern is a perturbation artifact, not a velocity-direction effect.

### Failure #3 (70% probability): The TT Is a Smoothness Predictor

The independent researcher computes R²_naive (predict v̂=0). They find R²_naive > 0.85. The TT's apparent performance is smoothness-exploitation. The project's central claim ("velocities are learnable") is demoted to "hidden states change slowly."

### Failure #4 (60% probability): SmoLLM2 Transfer Does Not Replicate

The independent researcher uses zero-padding instead of learned projection for cross-model transfer. L8 is NOT the best layer. The "cross-model transfer" was projection-driven, not dynamic-universality-driven.

### Failure #5 (50% probability): Non-Math Tasks Show No Improvement

The independent researcher evaluates on ARC and MMLU. L8 steering produces 0pp or negative effects. The steering paradigm is confined to math word problems — a severe limitation that changes the publication story from "improving reasoning" to "improving GSM8K performance via KV-cache perturbation."

### Failure #6 (40% probability): L8 Improvement Is Not Robust to Seed

The independent researcher evaluates L8 steering across 10 random seeds. The +20pp effect varies from +5pp to +35pp. The mean is +15pp with std 10pp. The effect is real but highly variable — too unreliable for practical use and too noisy for mechanistic interpretation.

### Failure #7 (30% probability): The Effect Does Not Exist at All Under Proper Blinding

The independent researcher runs a blind evaluation: evaluator doesn't know which condition (steered vs baseline) produced each answer. Under blinding, the +20pp effect disappears. The original evaluation had an unconscious confirmation bias in scoring.

---

## 6. THE P-HACKING AUDIT

### Effective Hypothesis Test Count

Let me count every hypothesis test that could be considered "evidence for the steering paradigm":

| Category | Tests | Justification |
|----------|-------|---------------|
| Per-layer accuracy (28 layers vs baseline) | 28 | Each layer's accuracy compared to baseline |
| Per-layer accuracy (positive vs negative) | 28 | Each layer tested for trim-tab or death classification |
| Per-layer × per-α (future: 8 α values) | 28 × 8 = 224 | Full signed α sweep |
| Steering mechanisms (6 mechanisms tested) | 6 × 28 = 168 | Each mechanism × layer combination |
| Models tested (8 models) | 8 × 28 = 224 | Each model × layer |
| Metrics (accuracy, token divergence) | 2 × 28 = 56 | Two metrics per layer |
| Datasets (GSM8K, SVAMP) | 2 × 28 = 56 | Two datasets per layer |
| TT variants (standard, correct-only, incorrect-only, contrastive) | 4 × 28 = 112 | Each TT variant × layer |
| Cross-model transfer tests | 1 model pair × 6 layers = 6 | SmolLM2→7B |

**Conservative total**: 28 (single α, single metric, single dataset) = 28 tests
**Reasonable total**: All conducted tests with familywise grouping = ~200-400 tests
**Liberal total**: Every combination that could be reported = ~900 tests

### Adjusted Significance Thresholds

| Correction | Tests | Threshold | z-score required | L8 at z=4.4 survives? |
|------------|-------|-----------|------------------|----------------------|
| None | 1 | p < 0.05 | 1.96σ | ✅ Yes (p ≈ 10^{-5}) |
| Bonferroni (single-α, 28 layers) | 28 | p < 0.00179 | 2.91σ | ✅ Yes |
| Bonferroni (all reasonable) | 300 | p < 0.000167 | 3.46σ | ✅ Yes |
| Bonferroni (all liberal) | 900 | p < 0.0000556 | 3.87σ | ✅ Yes (barely) |
| Holm-Bonferroni step-down (300) | 300 | p < 0.000167 | 3.46σ | ✅ Yes |
| Benjamini-Hochberg FDR (q=0.05, 300) | 300 | q < 0.05 | 2.71σ | ✅ Yes |
| **Effective test count (this project)** | **28-200** | **p < 0.00025-0.00179** | **3.0-3.5σ** | **✅ Yes** |

### Results That SURVIVE Correction

| Claim | z-score | Raw p | Survives Bonferroni (28)? | Survives Bonferroni (200)? | Survives Holm-Bonferroni (200)? |
|-------|---------|-------|--------------------------|---------------------------|--------------------------------|
| L8 +20pp on GSM8K | 4.4 | 5.4×10^{-6} | ✅ | ✅ | ✅ |
| L8 +20pp on SVAMP | ~1.1 | 0.27 | ❌ | ❌ | ❌ |
| L2 +17pp | 3.9 | 4.8×10^{-5} | ✅ | ✅ | ✅ |
| L9 -23pp | 5.2 | 1.0×10^{-7} | ✅ | ✅ | ✅ |
| R² = 0.855 (velocity learnability) | N/A (descriptive) | — | N/A | N/A | N/A |
| Transfer preserves L8 pattern | N/A (qualitative) | — | N/A | N/A | N/A |
| SmolLM2 R² = 0.94 | N/A (descriptive) | — | N/A | N/A | N/A |
| 88% token divergence | N/A (descriptive) | — | N/A | N/A | N/A |
| Contrastive TT improves over standard | Not yet evaluated | — | N/A | N/A | N/A |

### Results That DO NOT Survive

| Claim | Why It Fails | Adjusted p |
|-------|-------------|------------|
| L3 +13pp (z ≈ 3.0) | Marginal after correction for 200 tests | p_adj = 0.09 (NS) |
| L5 +13pp (z ≈ 3.0) | Same as above | p_adj = 0.09 (NS) |
| L4 +7pp (z ≈ 1.6) | Not significant even uncorrected | p = 0.11 (NS) |
| L10 +17pp (z ≈ 3.9) | Survives 28 but marginal for 200 | p_adj = 0.036 (Barely ✅ at 0.05) |
| L7 -14pp (z ≈ 3.2) | Marginal after correction for 200 tests | p_adj = 0.052 (NS at 0.05) |
| SVAMP L8 +4pp (z ≈ 1.1) | Not significant at any level | p = 0.27 (NS) |
| SVAMP L9 -14pp (z ≈ 3.2) | Marginal for SVAMP family (28 tests) | p_adj = 0.07 (NS) |

**Key insight**: The MAIN claims (L8 +20pp, L9 -23pp, L2 +17pp) survive even aggressive correction. The BROADER pattern (multiple trim-tabs, the trim-tab/death opposition, SVAMP generalization) does NOT survive correction. The project should focus its publication claims on the robust results and be transparent about which results are suggestive.

### The p-Hacking Vulnerability Score

| Practice | Risk Level | Evidence |
|----------|-----------|----------|
| Stopping when p < 0.05 | LOW | L8 at p=10^{-5} found during systematic sweep |
| Multiple metrics | MODERATE | Accuracy is primary, but token divergence is selectively reported |
| Multiple models | HIGH | Only 1/8 models shows trim-tabs; post-hoc explanation (capability threshold) |
| Multiple layers | MODERATE | All 28 layers reported, not cherry-picked |
| Post-hoc subgroup | MODERATE | "Capability threshold" identified after seeing data |
| Reporting selective outcomes | LOW | All layers' results reported |
| Covariate adjustment | NONE | No covariates used |
| Failing to report null results | LOW | Null results reported (Math-1.5B, SmolLM2, Qwen3.5) |
| Adding data after seeing results | UNKNOWN | Could not assess from project records |

**Overall p-hacking vulnerability**: **MODERATE** (3/7 indicators flagged)

---

## 7. PUBLICATION-RISK ANALYSIS

### Safe to Publish (Low Risk — robust claims)

| Claim | Evidence | Risk | Recommended Confidence | Suggested Wording |
|-------|----------|------|----------------------|-------------------|
| **Hidden state velocities during generation are learnable (R²=0.8-0.9)** | Replicated across 3 model families, 2 data types | **LOW** | 9/10 | "Generation trajectories show strong learnable structure, with velocity prediction R² exceeding 0.8 across models." |
| **Per-layer KV-cache modification produces layer-dependent accuracy changes** | Replicated on 7B, pattern consistent across datasets | **LOW** | 9/10 | "Modifying the KV cache at specific layers during generation causes accuracy changes that depend strongly on which layer is modified, with effects ranging from +20pp to -23pp." |
| **The effective sign and magnitude depend on layer identity** | L8 trim-tab, L9 death layer replicable | **LOW-MODERATE** | 8/10 | "The sign of the effect depends on layer identity — adjacent layers can show opposite effects (L8 positive, L9 negative) — indicating sharp functional boundaries in the residual stream." |
| **Steering requires baseline capability > ~40%** | Consistent across 6 models; no exception | **MODERATE** | 7/10 | "Effective steering appears to require that the model already exhibits above-chance performance on the target task, suggesting steering amplifies rather than creates capability." |

### Risky to Publish (Needs Additional Controls)

| Claim | Risk | Missing Control | Recommended Action Before Publication |
|-------|------|----------------|--------------------------------------|
| **Velocity direction causes improvement (not just any perturbation)** | **HIGH** | Random baseline not run | Run Phase A1 4-condition protocol. If TT > random +5pp on ≥2 layers, claim is safe. |
| **The TT learns causal reasoning dynamics (not smoothness)** | **HIGH** | Naive baseline (v̂=0) not computed | Compute R²_naive. If R²_tt − R²_naive < 0.05, TT is smoothness-exploiting. |
| **The trim-tab pattern generalizes across tasks** | **HIGH** | Only math tasks tested | Evaluate on ARC, BBH, MMLU. If ALL are null, claim is domain-limited. |
| **Contrastive TT is a viable normative steering mechanism** | **EXTREME** | Not evaluated at all | Must be evaluated before any publication mention. Current evidence is: "Two TTs exist." |
| **Cross-model transfer proves universal dynamics** | **MODERATE** | No fixed-projection control | Repeat with zero-padding. If L8 is not best, claim is projection-driven. |
| **L8 is a "reasoning trim-tab" (mechanistic claim)** | **HIGH** | No mechanistic validation | Need at minimum: (a) attention pattern change measurement, (b) K/V split analysis, (c) position shuffle test. Without mechanism, claim only: "L8 perturbation improves accuracy." |
| **Effect size is exactly +20pp** | **MODERATE** | Single run of 100 problems | Report as "observed +20pp on a 100-problem sample (95% CI: ±8.8pp)". Need 500-problem evaluation for precise estimate. |

### Publication Strategy by Claim Strength

**Tier 1 (Publishable Now)** — Velocity learnability, per-layer effects, capability threshold
- Target: Workshop paper or short conference paper
- Risk: Low — claims are descriptive, not causal
- Value: Moderate — novel finding but mechanism unknown

**Tier 2 (Publishable After Phase A1)** — Causal steering claim
- Target: Main conference paper
- Risk: Moderate — depends on random baseline result
- Value: High — if TT > random, first demonstration of directional KV-cache steering

**Tier 3 (Publishable After Phase A + B)** — Comprehensive steering framework
- Target: High-impact conference or journal
- Risk: Moderate-High — requires multiple replications
- Value: Very high — full characterization of steering space

**Tier 4 (Never Publish Without Mechanism)** — "Steering improves reasoning"
- Claim is too strong without any mechanistic understanding
- Publish only as: "Per-layer accuracy effects from KV-cache perturbation" — purely descriptive
- Remove "reasoning improvement" language until causal mechanism is established

---

## APPENDIX A: What Would It Take to Convince Me?

As a hostile reviewer, here is the minimum evidence I would need to accept the paradigm:

1. **Random baseline control**: TT predictions must outperform random vectors of equal norm by >10pp at L8 (not just "not equal" — practically significant superiority). This must be replicated at L2.

2. **Mechanistic validation**: Show that L8 steering changes attention patterns in a way that is consistent with improved reasoning (e.g., increased attention to relevant tokens, decreased attention to irrelevant tokens). This requires attention logit capture during steering.

3. **Naive baseline rejection**: R²_naive < 0.5 (predicting v̂=0 is substantially worse than TT). If R²_naive > 0.7, the TT's performance is mostly smoothness, and I will not accept the "learnable velocity" claim.

4. **Cross-task validation**: At least one non-math task must show positive effects. If the effect is domain-specific, the claim is "steering improves math word problem accuracy" — publishable but not "improving reasoning."

5. **Multiple-comparison-corrected significance**: Report all 28 layers' results with corrected p-values. If only L8 and L9 survive (which they do), the broader pattern claims (4+ trim-tabs) must be presented as suggestive, not confirmatory.

6. **Independent replication**: Another lab must reproduce the L8 +20pp result at least once. This is a reliability requirement for any empirical claim.

Absent any ONE of these, the paradigm is "suggestive but unconfirmed." Absent TWO or more, it is "likely artifactual."

---

## APPENDIX B: The Most Generous Interpretation

Even as a hostile reviewer, I acknowledge:

1. The per-layer effect is REAL — modifying the KV cache at specific layers does change accuracy in a layer-dependent way. This is a genuine empirical finding worthy of publication.

2. The L8 +20pp effect survives multiple comparisons correction even under aggressive assumptions. This is not p-hacking.

3. The capability threshold observation is consistent and has face validity — you can't amplify nonexistent capability.

4. The infrastructure and methodology are sound for a first investigation.

The debate is not about whether the effect exists — it's about whether the INTERPRETATION (velocity-based reasoning improvement) is correct. My position: **the effect likely exists, but the interpretation is likely wrong, and the paradigm is likely confounded by one or more of the artifacts described above.** The three most dangerous confounds, in order:

1. **Smoothness confound** (cheapest to check, most likely fatal if true)
2. **Random baseline equivalence** (requires 3.7 GPU-hours, second most likely fatal)
3. **Position-frequency artifact** (requires 0.5 GPU-hours, third most likely fatal)

The project should prioritize ALL THREE of these before any further development. If all three survive, the paradigm is genuinely interesting. If any one falls, the paradigm needs fundamental revision.

---

*Generated by Diffuser-5 — Adversarial Conceptual Diffusion — 14 June 2026*
*Role: Hostile peer reviewer — Standard: Extraordinary claims require extraordinary evidence*
