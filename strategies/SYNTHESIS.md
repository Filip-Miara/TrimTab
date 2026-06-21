# Multi-Agent KV Inference — Strategy Synthesis

Three strategists have produced independent plans for the multi-agent KV-sharing system.
This document merges them into a coherent whole.

## Shared Goal
32-64 parallel Qwen2.5-3B-AWQ instances communicating bidirectionally via KV cache sharing,
collectively achieving 7B+ reasoning accuracy on GSM8K.

## Key Design Decisions (from strategist convergence)

| Decision | Strategy A | Strategy B | Strategy C | Consensus |
|----------|-----------|-----------|-----------|-----------|
| Topology | Star-over-Ring | Ring (primary) | Hybrid ring+broadcast | Ring with star fallback |
| Execution | Batched on 1 GPU | Sequential | Sequential + CPU offload | Sequential (VRAM-bound) |
| Merge | Confidence-weighted blend | Weighted average | Democratic voting | Confidence-weighted blend |
| Timing | Sync barrier per token | Periodic (K=5-10) | Per-token barrier | Periodic (every 5-10 tokens) |
| β strategy | β_base × (1-conf_i) | Sweep in Phase 3 | β=0.75 (proven) | Adaptive per-instance β |
| Compaction | MAD thresholding | Attention Matching | Velocity-magnitude | Phase 2a: velocity; Phase 2b: AM |

## Phased Implementation Plan

### Phase 0: Infrastructure (3 files, ~800 LoC)
- `src/infra/kv_serializer.py` — serialize/deserialize DynamicCache to flat tensors
- `src/infra/model_farm.py` — manage N ModelInstance workers on 1 GPU
- `src/infra/kv_compactor.py` — Latent Briefing-inspired KV compaction

### Phase 1: 2-Instance Validation (1 script, ~200 LoC)
- Instance A generates with L10 steering → extracts KV → sends to Instance B
- Instance B generates using A's steered KV (no steering itself)
- Validates H-1: Steering signal survives KV transfer

### Phase 2: 8-Instance Star Topology
- KV compaction reduces transfer cost
- Steer-Once-Broadcast-Many pattern
- Target: 8 workers each within 3pp of orchestrator

### Phase 3: Bidirectional + Voting
- Democratic KV voting protocol
- Instances share confidence, vote on best trajectory
- Diversity bonus prevents herding

### Phase 4: 32-64 Instance Hierarchy
- Hierarchical voting (workers → pods → global)
- CPU-side KV storage for scaling beyond GPU VRAM
- Target: ≤7B-5pp on GSM8K

## Critical Prerequisites
1. Per-layer sweep on 3B (confirm L10 heads 2&3 optimal for 3B)
2. Baseline ensemble curve (M independent 3B instances, M=1..128)
3. KV serialization validation (round-trip exact match)
