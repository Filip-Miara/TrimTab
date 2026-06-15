# Session Compaction Recovery — 2026-06-14/15

## Recovery Status: ✅ RESTORED

Full project state captured in git (master branch, 40+ tags).
Background process: TT training on Qwen3.5-0.8B trajectories running (epoch 7/20, R²=0.862).

## Project State

**Project:** RankAdaptation / TrimTab
**Repository:** https://github.com/Filip-Miara/TrimTab
**Branch:** master
**Location:** /home/filip/Projects/Personal/AI/RankAdaptation
**Python:** /home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python
**GPU:** NVIDIA GeForce RTX 4060 Laptop GPU, 8.2GB total (cooler improved, 45W limit)

### Tag Timeline (Recent)

| Tag | Description |
|-----|-------------|
| v0.34.0 | PPL-modulated correction (negative result) |
| v0.35.0 | All steering mechanisms tested (conclusive negative) |
| v0.36.0 | Per-layer trim-tab discovery, cross-model transfer |
| v0.37.0 | Contrastive TT, SVAMP generalization confirmed |
| v0.38.0 | Autonomous 4-stage sweep pipeline |

### Files Created This Session

| File | Purpose |
|------|---------|
| `run_alpha_sweep.py` | Alpha sweep on L8 (100 problems, 9 alphas) |
| `run_epoch0_analysis.py` | Zero-cost analyses from existing trajectories |
| `run_epoch1_sweep.py` | 4-condition × 28-layer protocol (baseline, random, std, contrastive) |
| `run_contrastive_eval.py` | Contrastive TT evaluation |
| `run_cross_model_transfer.py` | Cross-model TT transfer via projection adaptation |
| `run_autonomous_sweep.py` | All 4 stages autonomous pipeline |
| `run_svamp_generalization.py` | SVAMP cross-dataset test |
| `run_per_layer_sweep.py` | Per-layer trim-tab/death-layer sweep |
| `run_7b_dora_steering.py` | DoRA-inspired direction/magnitude separation |
| `run_7b_steering.py` | Qwen2.5-7B steering evaluation |
| `run_collect_math15_trajs.py` | Math-1.5B trajectory collection |
| `run_train_contrastive_tt.py` | Contrastive TT training (modes: correct, incorrect, all) |
| `run_math15_sweep.py` | Math-1.5B sweep scripts |
| `run_collect_gen_trajs_7b.py` | 7B trajectory collection |
| `run_7b_baseline.py` | Qwen2.5-7B baseline |
| `colab_l4_alpha_sweep.ipynb` | Colab L4 notebook for alpha sweep |
| `SESSION_DEBRIEF.md` | Session debrief document |
| `PROJECT_DEBRIEF.md` | Full project debrief |
| `tse-analysis/` | Triadic Synthesis Engine analysis (12 phases) |

### Data on Storage

| Path | Content | Size |
|------|---------|------|
| `/home/.../data/qwen35_08b_trajs/` | Qwen3.5-0.8B trajectories (10 files) | 6.9GB |
| `/run/media/.../project_data/qwen25_7b_gen_trajs/` | 7B trajectories (83 files) | 35GB |
| `/run/media/.../project_data/math15_gen_trajs/` | Math-1.5B trajectories (37 files) | 13GB |

### Checkpoints

| File | Content | R² |
|------|---------|-----|
| `best_gen_tt_7b.pt` | Qwen2.5-7B standard TT (all data) | 0.855 |
| `best_tt_correct.pt` | Qwen2.5-7B TT_correct (subset) | 0.832 |
| `best_tt_incorrect.pt` | Qwen2.5-7B TT_incorrect (subset) | ~0.83 |
| `best_tt_all.pt` | Qwen2.5-7B TT_all (subset) | 0.853 |
| (moved to HDD) Math-1.5B TTs | Math-1.5B contrastive TTs | 0.873/0.909 |
| `best_tt_08b.pt` (training) | Qwen3.5-0.8B standard TT (in progress) | ~0.862 |

## Restored Objectives

| # | Priority | Objective | Status |
|---|----------|-----------|--------|
| 1 | HIGH | Collect Qwen3.5-0.8B trajectories | ✅ Done (10 files, 6.9GB) |
| 2 | HIGH | Train TT on 0.8B data | 🔄 Running (epoch 7/20, R²=0.862) |
| 3 | HIGH | Per-layer sweep: 24 layers × 5 α, FA + GDN | 📋 Pending (needs TT checkpoint) |
| 4 | HIGH | Refine best configs | 📋 Pending |
| 5 | MEDIUM | Multi-layer combinations | 📋 Pending |
| 6 | MEDIUM | TSE analysis of all results | 📋 Pending |

## Sub-Agent Processes

| ID | Task | Status | Latest Output |
|----|------|--------|---------------|
| `proc_2026-06-15T0357_250fce` | TT 0.8B training (epochs 1-20) | **Running** | ep=7 r2=0.862 (150s/epoch) |

## Key Findings

| Finding | Evidence | Confidence |
|---------|----------|-----------|
| **Trim-tab layers exist** | L8: +20pp, L2: +17pp (Qwen2.5-7B) | 9/10 |
| **Death layers exist** | L9: -23pp (Qwen2.5-7B) | 9/10 |
| **Cross-dataset generalization** | SVAMP: L8 +4pp, L9 -14pp | 8/10 |
| **Cross-model transfer works** | SmolLM2→7B preserves L8 pattern | 7/10 |
| **Gen trajectories learnable** | R²=0.94 (SmolLM2), 0.855 (7B) | 9/10 |
| **HF cache corruption** | Duplicate snapshot directories | 6/10 |
| **Math-1.5B has no trim tabs** | All layers ≤ baseline at α=0.1 | 8/10 |
| **Contrastive TTs trained** | R²=0.83 each (7B subset) | 8/10 |
| **Qwen3.5-0.8B: bf16 > 4-bit** | 64 vs 48 it/s | 9/10 |

## Key Numbers

| Metric | Value |
|--------|-------|
| Per-layer sweep (7B): best layer | L8: +20pp |
| Per-layer sweep (7B): worst layer | L9: -23pp |
| Per-layer sweep (7B): best combination | L2: +17pp, L5: +13pp |
| Contrastive TT R² (correct) | 0.832 |
| Contrastive TT R² (incorrect) | ~0.83 |
| Standard TT R² (full 7B data) | 0.855 |
| Standard TT R² (Math-1.5B data) | 0.892 |
| Qwen3.5-0.8B baseline | 23% (7/30) |
| Qwen3.5-0.8B TT R² (training) | 0.862 (epoch 7, improving) |
| bf16 forward speed (0.8B) | 64 it/s |
| 4-bit forward speed (0.8B) | 48 it/s |
| Step-by-step generation (0.8B) | 39 tok/s |
| GPU VRAM | 1.5GB / 8GB (bf16 0.8B model) |
| System RAM | 14GB total, swap: 8.5G + 12G |

## Blocker Status

| Blocker | Why | Severity | Unblock |
|---------|-----|----------|---------|
| TT training slow | 150s/epoch from CPU→GPU batch transfers | WARNING | Acceptable; completes in ~50 min |
| Math-1.5B no trim tabs | All layers neutral/harmful | WARNING | Try lower α or different architecture |
| GPU thermal throttling | 45W limit on laptop RTX 4060 | WARNING | Better cooling installed, stable at 51°C idle |

## Next Immediate Action

Wait for TT training to complete (`best_tt_08b.pt` will be saved). Then run per-layer sweep with the TT:

```bash
# After training completes:
# Run per-layer sweep on Qwen3.5-0.8B (24 layers, both FA and GDN)
cd /home/filip/Projects/Personal/AI/RankAdaptation
/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u run_alpha_sweep.py --n-test 100 --layer 8 --alphas -0.5 -0.2 -0.1 0.0 0.05 0.1 0.2 0.5 1.0

# Check training progress:
read_background_process_output proc_2026-06-15T0357_250fce tail 5
```

---
*Recovery completed at: 2026-06-15 04:00 UTC*
*System health: degraded (swap active, but TT training progressing)*
