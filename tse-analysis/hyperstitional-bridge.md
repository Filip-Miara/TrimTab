# Hyperstitional Bridge — RankAdaptation

---

## H-1: Death-Layer Sign Inversion
- **Type**: Structural
- **Statement**: "Death layers (L7, L9, L15+) are inverted trim-tabs — steering them with negative α (away from predicted velocity) improves accuracy as much as positive α improves at L8."
- **Falsification**: Steering L9 with α = -0.1 on Qwen2.5-7B yields accuracy ≤ baseline (45%).
- **Confirmation**: L9(-0.1) accuracy ≥ 65% (L8's baseline-matched effect).
- **Minimum Experiment**: Run `run_per_layer_sweep.py` with `--alpha -0.1` for L9 only, 100 problems.
- **Risk**: If wrong, we lose confidence in steering-based polarity analysis. But it's a minimal-cost test.
- **Value**: If correct, converts 4+ death layers into trim-tabs overnight, potentially doubling the improvement (from +20pp to +40pp with L8+flipped L9).

## H-2: Velocity Prediction Trivial Baseline
- **Type**: Structural (null hypothesis)
- **Statement**: "The TT's R²=0.85-0.94 is primarily explained by predicting the norm-growth pattern of hidden states through layers (||h[l+1]|| > ||h[l]||), not by learning meaningful directional dynamics."
- **Falsification**: A baseline that predicts v[l] = ((||h[l+1]||/||h[l]||) - 1) · h[l] achieves R² < 0.5, significantly below the TT's R².
- **Confirmation**: Norm-based baseline achieves R² ≥ 0.7, explaining most of the TT's predictive power.
- **Risk**: If confirmed, the entire steering approach needs rethinking — steering along a norm-growth direction is equivalent to amplifying the hidden state magnitude, which might not be specific to reasoning.
- **Value**: If falsified, it proves the TT learns real directional dynamics, strengthening the theoretical foundation.

## H-3: Instruct-Tuning Separates Manifolds
- **Type**: Relational
- **Statement**: "Instruct-tuned models have separable correct/incorrect hidden state manifolds (enabling contrastive steering), while base models (even with similar parameter counts and baseline accuracy) have entangled manifolds where contrastive steering fails."
- **Falsification**: A base model (e.g., Qwen2.5-1.5B base, not instruct) shows trim-tab effects with contrastive steering comparable to an instruct-tuned model of similar size.
- **Confirmation**: Qwen2.5-7B-Instruct (instruct-tuned) shows trim tabs; Qwen2.5-Math-1.5B (base model) shows none, despite both having baseline accuracy > 38%.
- **Minimum Experiment**: Run the same per-layer contrastive sweep on a similarly-sized instruct-tuned model (e.g., Qwen2.5-1.5B-Instruct) vs Math-1.5B base model.
- **Risk**: Low — reveals architectural requirement for steering.
- **Value**: If confirmed, it guides model selection (always use instruct-tuned) and explains why Math-1.5B failed.

## H-4: λ Interpolation Outperforms Subtraction
- **Type**: Potential
- **Statement**: "Steering with v = λ·v_c + (1-λ)·v_i (interpolation) for λ ∈ [0,1] achieves higher accuracy at λ > 0.5 than v_c - v_i (subtraction), because subtraction cancels shared language-modeling dynamics essential for coherence."
- **Falsification**: Accuracy at λ=1 (v_c only) ≤ accuracy with v_c - v_i.
- **Confirmation**: Accuracy peaks at λ=0.7-0.9, outperforming both v_c alone and v_c - v_i.
- **Minimum Experiment**: Sweep λ ∈ {0, 0.25, 0.5, 0.75, 1.0} at L8 on 100 problems.
- **Risk**: Low — minor code change.
- **Value**: If confirmed, it provides a theoretically cleaner contrastive formulation.

## H-5: First-Step Steering Sweet Spot
- **Type**: Temporal
- **Statement**: "Steering the FIRST token of generation (t=0) at L8 with the best layer-specific α produces a disproportionate accuracy gain (+25-30pp) compared to steering from t=1 onward, because the first token determines the entire reasoning trajectory."
- **Falsification**: First-token steering accuracy ≤ current t≥1 steering accuracy.
- **Confirmation**: First-token steering yields the highest per-step Δ or enables more aggressive early steering that compounds into later tokens.
- **Minimum Experiment**: Remove first_step skip in `run_7b_steering.py`, compare 50 problems with vs without first-step steering.
- **Risk**: Medium — first-step steering uses hidden states from the prompt forward pass (not generation), which have different distribution.
- **Value**: High — if it works, it's a free improvement with zero architectural changes.

## H-6: Layer Polarity Generalization
- **Type**: Relational (cross-domain)
- **Statement**: "The trim-tab/death-layer pattern is consistent across diverse reasoning tasks — L8 improves GSM8K, SVAMP, and ARC, while L9 degrades them all — because layer polarity reflects the model's general computational architecture, not task-specific memorization."
- **Falsification**: On a non-math task (e.g., BBH or MMLU), L8 steering degrades accuracy or L9 steering improves it.
- **Confirmation**: Polarity signs match across GSM8K, SVAMP, ARC — L8 always trim-tab, L9 always death-layer.
- **Minimum Experiment**: Run per-layer sweeps (L0, L8, L9, L15) on ARC (easy, no math) and BBH (complex reasoning) with 100 problems each.
- **Risk**: Medium — per-layer sweeps take ~2 hours per task.
- **Value**: If confirmed, it establishes velocity-based steering as a general reasoning amplifier, opening the door to broader applications.

## H-7: Residual Stream Steering > KV-Cache Steering
- **Type**: Structural (mechanism comparison)
- **Statement**: "Directly modifying the residual stream (h' = h + α·v at the residual level, producing output that feeds into the next layer's attention and MLP) is more effective than KV-cache steering because it affects BOTH attention and MLP computations, not just attention."
- **Falsification**: Residual stream steering produces ≤ KV-cache steering accuracy at the same layer and α.
- **Confirmation**: Residual steering matches or exceeds KV steering for the same configuration.
- **Minimum Experiment**: Implement residual stream steering by adding α·v to the hidden state before it enters the next layer's attention+MLP, bypassing K/V projection. Test on 50 problems.
- **Risk**: High — requires model surgery (intercepting the residual stream inside the transformer forward method).
- **Value**: If confirmed, it unlocks a simpler, more general steering mechanism that works on any transformer architecture.
