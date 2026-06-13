# Phase 0: VOID — Assumption Surfacing & Bracketing

**Subject**: RankAdaptation — velocity-based latent steering via TrajectoryTransformers  
**Date**: 2026-06-13  
**Mode**: full  

---

## Explicit Assumptions

| # | Assumption | Source | Confidence |
|---|-----------|--------|-----------|
| A1 | Hidden state velocities v[l] = h[l+1] - h[l] are the right prediction target for steering | Project design | 6/10 |
| A2 | Adding velocity to the hidden state (h' = h + α·v) changes future token selection usefully | KV-cache steering literature | 7/10 |
| A3 | A TT trained on generation trajectories (R²=0.87-0.94) produces steering signals better than random | SmolLM2 proof (88% divergence) | 8/10 |
| A4 | The KV cache modification approach works equivalently across MHA and GQA architectures | Transformers implementation | 8/10 |
| A5 | GSM8K accuracy is the right metric for evaluating steering quality | Common practice | 5/10 |
| A6 | Contrastive TT (v_correct − v_incorrect) produces normative (not just descriptive) steering | Theoretical inference | 7/10 |
| A7 | Layer-specific trim-tab effects generalize across model families and sizes | Qwen2.5-7B + SmolLM2 evidence | 7/10 |
| A8 | The steering direction is the same across all tokens in a generation | Current implementation | 3/10 |
| A9 | 4-bit quantization does not degrade steering quality | Untested | 4/10 |
| A10 | The TT's transformer body (d_model=768) is model-agnostic | Transfer experiment evidence | 6/10 |

## Implicit Assumptions

| # | Assumption | Inference Chain |
|---|-----------|----------------|
| IA1 | The TT learns velocity dynamics, not surface statistics | High R² could mean it just memorized common trajectories |
| IA2 | "Correct" and "incorrect" trajectories form separable manifolds in hidden state space | Contrastive approach requires this |
| IA3 | Steering at one layer doesn't undo steering at another | Layer independence assumption |
| IA4 | The optimal α for steering is the same across all problems | Single α per experiment |
| IA5 | The model's hidden state space is Euclidean (vector addition is meaningful) | Required for h' = h + α·v |
| IA6 | Few-shot prompting elicits the model's latent reasoning capability | Not testing raw model capability |
| IA7 | 50K trajectories are sufficient to train the TT | Arbitrary threshold |
| IA8 | The TT's position embedding (layer index) captures meaningful ordering | Learned embedding trained on 28 positions |

## Counter-Assumptions

| Assumption | ¬Assumption | Implication if True |
|-----------|-------------|-------------------|
| A1 | Velocity is the wrong target; should predict contrastive direction | Our entire TT approach is misaligned |
| A3 | Even R²=0.99 TT can't improve accuracy | Steering is fundamentally noise, not signal |
| A5 | GSM8K accuracy is the WRONG metric | We're optimizing for the wrong thing |
| A6 | Contrastive TT just learns the difference between two noisy distributions | No useful signal |
| A7 | Trim-tab layers are model-specific, not universal | Transfer experiments won't generalize |
| IA1 | TT memorizes trajectory patterns without understanding | Will fail on distribution shift |
| IA2 | Correct/incorrect manifolds are entangled | Contrastive signal is zero |
| IA4 | Optimal α varies per token by 10× | Single α is a poor approximation |
| IA6 | Few-shot examples don't elicit reasoning, just mimicry | We're measuring in-context learning, not steering |

## Bracket Statement

*These assumptions are set aside for the analysis. They will be re-examined in Phase 6 (Disparity Detection) to check if any assumption violation is the root cause of observed failures. The most critical are A1, IA1, and IA2 — if any of these are false, the entire approach collapses.*
