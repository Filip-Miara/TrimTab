# VOID Record — Assumption Surfacing & Bracketing

## Explicit Assumptions

1. **A1**: Hidden state velocities (v[l] = h[l+1] - h[l]) during generation are learnable and can be predicted by a transformer (R² = 0.85-0.94 across models).
   - *Source*: PROJECT_DEBRIEF.md Finding 1, empirical R² values.

2. **A2**: KV-cache modification at specific layers can steer generation toward more accurate answers.
   - *Source*: PROJECT_DEBRIEF.md Finding 2, L8→+20pp on Qwen2.5-7B.

3. **A3**: The per-layer trim-tab/death-layer pattern is robust and generalizes across datasets and model families.
   - *Source*: PROJECT_DEBRIEF.md Finding 2, SVAMP + cross-model transfer data.

4. **A4**: Steering requires the model to already have the target capability (cannot amplify nonexistent reasoning).
   - *Source*: PROJECT_DEBRIEF.md Finding 3, small models all failed.

5. **A5**: All-layer steering compounds noise from death layers and is net negative.
   - *Source*: PROJECT_DEBRIEF.md Finding 4.

6. **A6**: Standard MHA architectures (Qwen2.5, LLaMA, SmolLM2) are preferred over hybrid attention (GDN+FA) for KV-cache steering.
   - *Source*: PROJECT_DEBRIEF.md Finding 6.

7. **A7**: The TrajectoryTransformer learns a descriptive model (faithfully reproduces existing errors) rather than a normative model (predicts the correct velocity for wrong trajectories).
   - *Source*: PROJECT_DEBRIEF.md Key Lesson 5.

8. **A8**: Contrastive TT (v_correct - v_incorrect) converts descriptive to normative prediction.
   - *Source*: PROJECT_DEBRIEF.md Key Lesson 5, Open Question 1.

9. **A9**: The reading head (r=0.86 for perplexity prediction) can serve as a confidence-gating mechanism.
   - *Source*: CROSS_SESSION_BRIEF.md, reading head correlation experiments.

10. **A10**: chat_template formatting is critical for instruct-tuned models (baseline jumped from 4% to 73% on Qwen2.5-7B after applying it).
    - *Source*: PROJECT_DEBRIEF.md Infrastructure issue 6.

## Implicit Assumptions

1. **IA1 (relational)**: The relationship "velocity at layer l = h[l+1] - h[l]" captures the information necessary to affect token selection. We assume that perturbing hidden states along velocity directions changes attention patterns in subsequent layers *in the right way*.

2. **IA2 (structural)**: The 28-layer decomposition of Qwen2.5-7B is the right granularity. Trim-tab effects might be finer (sub-layer, head-level) or coarser (block-level groups).

3. **IA3 (relational)**: The velocity direction that improves accuracy at one generation step also helps at other steps — i.e., the correct manifold is consistent across tokens within a reasoning chain.

4. **IA4 (structural)**: Generation trajectories from GSM8K training on 500 problems are representative of the general reasoning manifold. The TT doesn't overfit to GSM8K-specific patterns.

5. **IA5 (relational)**: The contrastive signal v_correct - v_incorrect is the *difference* between two manifolds, and interpolating toward the correct manifold via α·(v_c - v_i) is monotonic (more α = more correct behavior).

6. **IA6 (potential)**: The steering mechanism doesn't interact destructively with the model's internal computations (e.g., attention patterns, MLP updates). We assume K/V perturbations are "absorbed" cleanly.

7. **IA7 (structural)**: The 4-bit quantization used for 7B model loading preserves the steering-relevant latent structure.

8. **IA8 (relational)**: The reading head's uncertainty signal (r=0.86) can be used at generation time to gate steering, despite being trained on frozen latents.

## Counter-Assumptions

1. **¬A1**: Hidden state velocities during generation are *not* learnable in a way useful for steering — the high R² reflects systematic noise patterns, not meaningful structure.

2. **¬A2**: KV-cache steering's effect on accuracy is spurious or dataset-specific. The +20pp on L8 could be a statistical fluctuation (only 100 problems, 45% baseline).

3. **¬A3**: The per-layer pattern does NOT generalize — SVAMP results (+4pp vs GSM8K's +20pp) show weakening, and Math-1.5B showed no trim tabs at all.

4. **¬A4**: The capability threshold is an artifact of small sample sizes. With more problems or per-token analysis, small models might show steering signal that's lost in noise.

5. **¬A5**: All-layer steering could work with an *adaptive* per-layer α vector instead of uniform α.

6. **¬A6**: Standard MHA is preferred *only* because the steering mechanism was designed for it. GatedDeltaNet might be steerable via its recurrent state mechanism with the right approach.

7. **¬A7**: The TT actually learns more than descriptive dynamics — the R² = 0.85-0.94 means it captures 85-94% of variance, leaving room for normative correction only at the margin.

8. **¬A8**: Contrastive TT may cancel out shared structure (both correct and incorrect trajectories share many dynamics), reducing effective signal.

9. **¬A9**: Reading head gating at generation time fails because the distribution shift (frozen Perceiver latents vs generation-time latents) corrupts the uncertainty signal.

10. **¬IA2**: Per-layer granularity is WRONG — trim-tab effects may be at the head level within layers, and averaging across heads within a layer dilutes the signal.

## Bracket Statement

The above assumptions are temporarily set aside for the analysis. They will be re-examined in Phase 6 (Disparity Detection) to check if assumption violations explain observed failures (Math-1.5B trim-tab absence, distribution shift effects, contrastive TT evaluation pending).
