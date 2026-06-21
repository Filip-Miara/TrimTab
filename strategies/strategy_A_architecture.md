# KV Consensus Mesh: Multi-Agent Inference Architecture

## Qwen2.5-3B-AWQ Reference Dimensions
| Param | Value | Notes |
|---|---|---|
| Layers | 36 | Index 0-35 |
| Q heads | 16 | d_head=128, d_model=2048 |
| KV heads | 2 | GQA, shared across 8 Q heads each |
| head_dim | 128 | K/V per head |

## 1. Communication Topology: Hierarchical Star-over-Ring
Two-tier topology: Tier 1 (Intra-GPU) shared-memory batch, Tier 2 (Inter-GPU) NCCL AllGather Ring.

Memory layout (single GPU, 64 instances):
- Model weights (AWQ 4-bit): ~2.0 GB
- KV caches (64 × 36 MB): ~2.3 GB
- Activations + scratch: ~1.0 GB
- Consensus buffer (L10 h2&3): ~1.0 MB
- Free: ~2.7 GB

## 2. Cache Serialization: CacheCard Protocol
Wire format for KV consensus: KVConsensusPacket with instance_id, gpu_rank, step, confidence, entropy, K_heads23, V_heads23 (~1,032 B per packet).

## 3. Merging Strategy: Robust Trimmed Confidence-Weighted Blend
Algorithm per consensus step:
1. FILTER: Remove low-confidence and relative outlier instances
2. COMPUTE WEIGHTS: w_i = conf_i × (1 - entropy_i / ln(n_kv_heads))
3. ROBUST MERGE: MAD-protected weighted mean with 5×MAD outlier clamp
4. BLEND: β_i = β_base × (1 − conf_i), β_base = 0.75

## 4. Instance Orchestration: Three-Phase Sync Barrier
- Phase 1: Prompt Processing (no consensus) — ~10 ms
- Phase 2: Consensus Generation (first K=30 tokens) — ~360 ms
- Phase 3: Free Generation (tokens 31..T) — ~4,820 ms
Total: ~5.2 s for 512 tokens

## 5. Critical Design: Dual-Path Token Selection
Token t selected from pre-consensus logits; KV consensus modifies what instance attends to when generating token t+1. Avoids chicken-egg problem.

## 6. Failure Handling: Triangular Redundancy
Divergence detection via tracking per-instance divergence score d_i over sliding window of 3 tokens. Exclude divergent from consensus merge, inject consensus KV at FULL replacement (β=1.0) for recovery.

## Implementation
```python
class KVConsensusMesh:
    """Multi-agent inference engine with KV cache consensus."""
    def consensus_step(self, past_kv, step):
        # Extract, merge, blend KV at L10 heads 2&3
    def generate(self, prompt):
        # Prompt → Consensus → Free generation with ensemble voting
```
