# RankAdaptation — Project Roadmap

## Status (June 2026)

Three independent steering mechanisms proven:
- **Phase 1/3**: Latent→Logit Correction — +13.8pp next-token accuracy
- **Phase 2**: Layer-0 Velocity Injection — 100% divergence at α=2.0

## Path 1: End-to-End Integration (IN PROGRESS)

Combine all three phases into a single generation pipeline.

**Status**: 
- ✅ Per-token accuracy: +13.8pp (Phase 1/3, prompt-trained heads)
- ⚠️ Distribution shift: heads trained on prompts fail at generation time (0% accuracy)
- 🔄 Fix: training heads on generation-time data (background process running)

**Current understanding**: The correction heads learn to predict logit offsets from
prompt-time hidden states. But during generation, hidden states include the model's
own generated tokens. The distribution shift causes the heads to produce nonsense
outputs. Solution: collect training data during autoregressive generation.

The `run_e2e_v2.py` script (background, Jun 12 16:00) collects generation-time
data from 200 GSM8K problems, retrains the 3-head ensemble, and evaluates both
per-token accuracy and full-generation accuracy.

Key question: do the per-token accuracy gains translate to full-answer accuracy?
Test: run on GSM8K, compare corrected vs baseline accuracy.

## Path 2: Reasoning-Step Trajectories

Current Perceiver/Transformer trained on **layer-to-layer** trajectories (h[l] → h[l+1] within one forward pass). Training on **token-to-token** trajectories (h[t] → h[t+1] across generated tokens) would make predictions directly relevant to generation.

| Approach | What it predicts | Training data | Potential R² |
|----------|-----------------|---------------|-------------|
| Layer-to-layer (current) | h[l+1] - h[l] | Single forward pass | 0.62 |
| Token-to-token (new) | h[t+1] - h[t] | Generated sequences | ? |

Hypothesis: token-to-token trajectories have higher structure because they capture the model's reasoning evolution, not just layer computation.

## Path 3: KV-Cache Optimization

Layer-0 injection currently requires **two forward passes** (one to compute velocity, one to inject). Folding the velocity into the KV cache reduces this to **one pass**:

```
After generating token t:
  1. Forward pass → hidden states + KV cache entry for token t
  2. Perceiver → velocity
  3. Modify KV cache entry for token t: k' = k_proj(h + αv), v' = v_proj(h + αv)
  4. Token t+1 attends to modified cache → steering without second forward pass
```

This requires accessing k_proj/v_proj per layer. Qwen3.5-2B uses mixed attention (Qwen3_5GatedDeltaNet for 20/24 layers + Qwen3_5Attention for 4/24 layers) — need to handle both.

## Path 4: Scale to Larger Models

All experiments used Qwen3.5-2B (4GB model, 8GB GPU). Scaling to 4B/7B tests:

- Does R² scale with model size? (Hypothesis: larger models have more structured trajectories)
- Does the correction head transfer? (Trained on 2B latents → fine-tune for larger model)
- Does accuracy improve with base model capability?

Scale targets:
- Qwen3.5-4B (currently in the hub, ~8GB) — would need P40 or A10 GPU
- Llama-3.2-3B — small enough for 8GB with quantization
- Qwen3.5-7B — needs 4-bit quantization (GPTQ/AWQ) for 8GB

## Path 5: Full DiMAE Architecture

Replace the monolithic correction MLP with the full DiMAE (Diffusive Mixture of Attentive Experts):

- Per-latent expert heads (32 experts, each 128→d_model)
- Entropy-weighted fusion (low entropy = specialized = higher weight)
- Diffusive refinement over expert outputs (3-5 denoising steps)
- Dynamic K pruning/growth based on attention entropy

Components already built (from phases 1-3):
- Latent extraction (return_latents=True)
- Correction head
- Multi-head ensemble

Missing:
- Per-latent entropy computation (need cross-attention weights from Perceiver)
- Dynamic K integration
- Diffusive denoising over expert outputs
