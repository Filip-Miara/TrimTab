# Integration Roadmap: Multi-Agent KV-Shared LLM Inference System

## Phase 0: Infrastructure Foundation
Files: src/infra/kv_serializer.py (~250 LoC), src/infra/kv_ipc.py (~200 LoC), src/infra/model_farm.py (~300 LoC), src/infra/config.py (~80 LoC)
Tests: test_infra_kv_serializer.py, test_infra_model_farm.py
Total: ~830 LoC across 4 files + 2 test files

## Phase 1: MVP — Two Instances, One-Way KV Share
Worker receives steered KV from orchestrator, achieves steering benefit without running steering computation.
File: run_multiagent_phase1.py (~200 LoC)
Directly tests TSE H-1 (Steering Signal Persistence)

## Phase 2: Scale to 8 Instances with Latent Briefing Compaction
File: src/infra/kv_compactor.py (~350 LoC)
Implements Steer-Once-Broadcast-Many pattern. KV compaction ≥40% with ≤3pp accuracy loss.
Compaction methods: velocity_magnitude, attention_matching, entropy

## Phase 3: Bidirectional Communication with KV Voting
File: src/infra/kv_protocol.py (~200 LoC), src/infra/kv_voting.py (~250 LoC)
Democratic KV voting: instances share confidence scores, vote on best trajectory, blend KV.
8-instance voting outperforms best single instance by ≥3pp.

## Phase 4: Full 32-64 Instance Hierarchical System
Hierarchical architecture: Level 0 (workers), Level 1 (pod coordinators), Level 2 (global coordinator)
File: src/infra/hierarchical_voting.py (~300 LoC), src/infra/scaling_utils.py (~150 LoC)
Target: 32-instance ensemble matches 7B baseline accuracy (≥65% on GSM8K)

## Dependency & Gates
P0 → P1 → P2 → P3 → P4
Gate 1: serialize/deserialize exact match
Gate 2: worker accuracy ≥ orchestrator - 5pp
Gate 3: compaction accuracy ≥ no-compaction - 5pp
Gate 4: voting accuracy ≥ best-single + 3pp

## Total: ~3,500 LoC across ~17 files, ~82 GPU-hours
