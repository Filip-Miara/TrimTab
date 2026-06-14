# Phase 0: VOID — Assumption Surfacing & Bracketing

## Explicit Assumptions

| # | Assumption | Source | Grounding |
|---|------------|--------|-----------|
| A1 | Hidden state velocities during generation are learnable | Finding 1, R²=0.85-0.94 across models | Empirical: 6 measurements across 4 models |
| A2 | Per-layer selectivity is mandatory for effective steering | Finding 2, per-layer sweep on Qwen2.5-7B | Empirical: L8:+20pp vs all-layers:-45pp |
| A3 | Steering requires the model to already have the target capability | Finding 3, capability threshold ~40% GSM8K | Empirical: 5 models below threshold → 0% improvement |
| A4 | KV-cache modification is the correct steering surface | Method declaration in kv_cache_steering.py | Architectural reasoning about attention geometry |
| A5 | Generation-trained TTs are substantially better than prompt-trained | Finding 4, R²=0.94 vs 0.62 | Empirical: 2 measurements |
| A6 | Cross-model transfer preserves trim-tab patterns | Finding 5, SmolLM2→7B | Empirical: 1 measurement (n=1) |
| A7 | Standard MHA architectures are preferred over hybrid attention | Finding 6, Qwen3.5 all failures | Empirical: 1 model family comparison |
| A8 | The contrastive signal (v_correct - v_incorrect) is normative | Method in run_train_contrastive_tt.py, line 4 | Theoretical assertion, NOT yet empirically confirmed |
| A9 | Linear projection (960→3584) suffices for cross-model transfer | Implementation in run_cross_model_transfer.py | Architectural assumption, not tested against alternatives |
| A10 | GSM8K accuracy is the correct evaluation metric | All experiments use GSM8K | Implicit from problem choice |
| A11 | α (steering strength) is constant per layer | Per-layer sweep implementation | Always uses fixed α=0.1 |
| A12 | 100 problems is sufficient for per-layer evaluation | run_per_layer_sweep.py line 86-87 | Sample size assumption |

## Implicit Assumptions

| # | Assumption | Inference Chain | Who/What Assumes |
|---|------------|-----------------|-------------------|
| B1 | Velocity prediction error and steering quality are monotonically related | "High R² → good prediction → good steering" chain, but steering quality depends on direction, not just magnitude | Project narrative |
| B2 | Layer independence during steering | Sweeping one layer at a time assumes layers interact linearly; superposition effects are ignored | Experimental design |
| B3 | The "trim-tab" vs "death layer" distinction is intrinsic to layers, not an artifact of α=0.1 | Only one α tested per layer; different α might reveal different patterns | Methodology |
| B4 | Hidden state dynamics are Markovian (only current state matters for velocity prediction) | TT takes h[l] → predicts v[l] using only current position | TrajectoryTransformer architecture |
| B5 | Correct and incorrect trajectories occupy separable manifolds in hidden state space | Contrastive approach assumes v_correct and v_incorrect point in meaningfully different directions | Contrastive TT design |
| B6 | The "capability threshold" is continuous (models below 40% can never benefit) | Finding 3 interpretation | Conclusion section |
| B7 | Death layers are universally harmful (no token position where they help) | L9 consistently harmful across GSM8K and SVAMP, but only 2 datasets tested | Generalization claim |
| B8 | 23 hidden states (for Qwen2.5-7B with 28 layers: h[0]..h[27], v[0]..v[22]) captures the full steering-relevant dynamics | Sampling the last hidden state per layer skips intra-layer dynamics | Implementation |
| B9 | The TT architecture (6-layer transformer, d_model=768) is sufficient for learning the velocity field | Architecture chosen without systematic search over capacity | Implementation assumption |
| B10 | Trajectory data from autoregressive generation is sufficient; no need to mask erroneous intermediate tokens | Only final answer correctness is labeled, not intermediate reasoning step quality | Data labeling |

## Counter-Assumptions

| # | Against | Counter-Assumption |
|---|---------|-------------------|
| ¬A1 | Velocities are learnable | What if the high R² is an artifact of smoothness in hidden state trajectories (trivial prediction: h[l+1] ≈ h[l] + noise)? |
| ¬A2 | Per-layer selectivity is mandatory | What if per-layer patterns are an artifact of α=0.1 and a different per-layer α schedule removes the death layer problem? |
| ¬A3 | Steering requires capability | What if steering can bootstrap capability through multi-step iterative refinement, even from low baselines? |
| ¬A4 | KV-cache is the correct surface | What if steering internal residual stream activations (not just KV cache) produces stronger effects? |
| ¬A5 | Gen-trained TTs are better | What if prompt-trained TTs fail not because of distribution shift but because prompt trajectories contain *corrective* signals that gen trajectories lack? |
| ¬A6 | Cross-model transfer works | What if the single successful transfer (SmolLM2→7B) is coincidence — the 7B's own TT already identifies L8, so any projection preserving layer identity works? |
| ¬A7 | MHA is preferred | What if hybrid attention just needs different steering mechanisms (e.g., steering GDN's recurrent state instead of KV cache)? |
| ¬A8 | Contrastive signal is normative | What if v_correct - v_incorrect amplifies spurious differences between the two trajectory sets rather than meaningful steering directions? |
| ¬A9 | Linear projection suffices | What if linear projection destroys inter-layer structure that a nonlinear adapter would preserve? |
| ¬A10 | GSM8K is sufficient | What if GSM8K's single-number answer format hides the true effect (e.g., steering improves reasoning quality but not final answer extraction)? |
| ¬A11 | Fixed α per layer | What if α needs to be dynamically computed per token (some tokens need strong steering, others need none)? |
| ¬A12 | 100 problems is enough | What if the results are subject to high variance (95% CI on 100 problems: ±10pp for 73% baseline)? |
| ¬B1 | R² → steering quality | What if steering requires directional precision (angular error) and R² is insensitive to angular alignment? |
| ¬B2 | Layer independence | What if trim-tab layers only work because death layers are NOT steering — and concurrent multi-layer steering with per-layer α cancels death layer effects? |
| ¬B3 | α=0.1 is optimal | What if each layer has a different optimal α, and the trim-tab/death classification shifts with α? |
| ¬B4 | Markovian dynamics | What if velocity depends on longer-range context (h[l-2], h[l-1], h[l]) and the TT's position-wise prediction misses this? |
| ¬B5 | Separable manifolds | What if correct and incorrect trajectories lie on the same manifold but in different regions, and v_correct - v_incorrect points to an intermediate (still incorrect) region? |

## Bracket Statement

The 12 explicit assumptions (A1-A12), 10 implicit assumptions (B1-B10), and their associated counter-assumptions (¬A1-¬A12, ¬B1-¬B5) are hereby bracketed for the duration of this analysis. They will be kept in awareness but not allowed to constrain the field of inquiry during Phases 1-5. In Phase 6 (Disparity Detection), we will systematically check whether any assumption violation explains observed failures. Assumptions marked with empirical support at n≥3 are given higher prior confidence; those with n=1 (A6, A8, B5) or purely theoretical (A8, B1, B5) are flagged as critical re-examination targets.
