# Session Compaction Recovery — 2026-06-16

## Recovery Status: ✅ RESTORED

Full project state captured in git (master branch, v0.40.2).
GPU driver crashed near end of session; system needs restart.

## Project State

**Project:** RankAdaptation / TrimTab
**Repository:** https://github.com/Filip-Miara/TrimTab.git
**Branch:** master (tag: v0.40.2)
**Location:** /home/filip/Projects/Personal/AI/RankAdaptation
**Python (main):** /home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python
**Python (AWQ):** /mnt/windows/awq_env/bin/python (activate with `source awq_activate.sh`)
**GPU:** NVIDIA GeForce RTX 4060 Laptop GPU, 8GB (currently hung — needs reboot)

## Key Checkpoints

| File | Content | R² |
|------|---------|:---:|
| `best_gen_tt_7b.pt` | Qwen2.5-7B standard TT (old pipeline) | 0.855 |
| `best_gen_tt_7b_new.pt` | 7B TT improved pipeline (epoch 2 of 30) | **0.768** |
| `best_gen_tt_3b.pt` | Qwen2.5-3B TT (d_model=1280, d_ff=4, per-layer norm) | ~0.66 |
| `best_tt_08b.pt` | Qwen3.5-0.8B TT | 0.8775 |

## Restored Objectives

| # | Priority | Objective | Status | Next Step |
|---|----------|-----------|--------|-----------|
| 1 | HIGH | 7B AWQ model downloaded | ✅ Done | Located at `/mnt/windows/Qwen2.5-7B-AWQ/qwen7b_awq/` |
| 2 | HIGH | 7B TT training (improved pipeline) | 🔄 R²=0.768 at epoch 2, crashed at epoch 3 | Restart with `--bs 64` |
| 3 | HIGH | 7B mmap built | ✅ Done | `/mnt/windows/trajs_7b_rmmap/` (90K trajectories) |
| 4 | MEDIUM | AWQ 7B batch throughput benchmark | ✅ Done | BS=32 optimal (448 tok/s, 5.9GB VRAM) |
| 5 | MEDIUM | 7B→3B cross-model transfer | ✅ Done | Transferred TT validates trim-tabs on 3B |
| 6 | LOW | GDN output-hook experiment | ⏸ Pending | Script ready at `run_gdn_output_steer.py` |
| 7 | LOW | Negative α L9 death sign flip | ⏸ Pending | Script ready at `run_death_sign_flip.py` |

## Key Files Created/Modified This Session

| File | Size | Purpose |
|------|:----:|---------|
| `colab_awq_7b_trajs.ipynb` | — | Colab notebook: 7B AWQ + trajectory collection to Drive |
| `colab_awq_quant.ipynb` | — | Colab notebook: 3B AWQ quantization |
| `run_train_gen_tt_7b.py` | — | 7B TT training script (mmap, per-layer norm, AMP, prefetch) |
| `run_train_gen_tt_3b.py` | — | 3B TT training script (same improved pipeline) |
| `run_sweep_3b_batched.py` | — | Batched per-layer sweep for 3B (BS=50, WIP) |
| `run_3b_full_sweep.py` | — | Full 36-layer sweep for 3B |
| `run_3b_probe.py` | — | Quick 3B trim-tab probe (transferred TT) |
| `best_gen_tt_7b_new.pt` | 192MB | 7B TT checkpoint, improved pipeline (epoch 2) |
| `best_gen_tt_3b.pt` | 650MB | 3B TT checkpoint (d_model=1280, d_ff=4) |
| `awq_activate.sh` | — | AWQ env activation script |

## Data Files

| Path | Contents | Size |
|------|----------|:----:|
| `/mnt/windows/trajs_7b_rmmap/` | 7B trajectory mmap (28 layers, 3584D) | ~35GB |
| `/mnt/windows/trajs_rmmap/` | 3B trajectory mmap (36 layers, 2048D, 88K trajs) | ~25GB |
| `/mnt/windows/trajs_3b_bs32/` | 3B raw trajectory batches (BS=32) | ~12GB |
| `/mnt/windows/trajs_3b_bs64/` | 3B raw trajectory batches (BS=64) | ~5GB |
| `/mnt/windows/Qwen2.5-7B-AWQ/qwen7b_awq/` | AWQ 4-bit quantized 7B model | ~3.5GB |

## AWQ 7B Batch Throughput (BS=32 optimal)

| BS | Tok/s | Per-seq | VRAM |
|:--:|:-----:|:-------:|:----:|
| 8 | 126 | 15.8 | 5.68GB |
| **32** | **448** | **14.0** | **5.90GB** |
| 64 | 408 | 6.4 | 6.19GB |
| 128 | 406 | 3.2 | 6.78GB |

## Key Findings

1. **7B improved TT pipeline works**: Per-layer norm + AMP + chunked prefetch achieves R²=0.768 at epoch 2 (old pipeline was ~0.70 at same point). Should reach 0.85+ by epoch 15-20.
2. **3B TT is limited**: Max R²≈0.66 for 3B vs 0.855 for 7B. Smaller model + more layers = noisier velocity dynamics.
3. **d_ff=4 optimal**: d_ff=2 too weak (R²=0.54), d_ff=6 unstable (overfitting).
4. **7B AWQ model downloaded**: `/mnt/windows/Qwen2.5-7B-AWQ/qwen7b_awq/` — benchmarked at BS=32 (448 tok/s, 5.9GB VRAM).
5. **AWQ Marlin kernel compiled**: Working via `awq_activate.sh` (MarlinLinear layers). 7B model loads in 21s.
6. **Negative α L9**: Not yet tested (script ready).
7. **GDN steering**: Output-hook approach not yet tested (script ready).

## Blocker Status

| Blocker | Severity | Unblock |
|---------|:--------:|---------|
| GPU driver hung (CUDA unknown error) | BLOCKER | `sudo reboot` |
| 7B TT training crashed at epoch 3 (BS=128 OOM) | WARNING | Restart with `--bs 64` |
| AWQ tokenizer version bug | WARNING | Use `use_fast=False` or load from original model |

## Next Immediate Action

After reboot:

```bash
# 1. Check GPU
nvidia-smi

# 2. Mount Windows partition if not auto-mounted
# (the AWQ env + mmap are there)
ls /mnt/windows/awq_env/bin/python

# 3. Continue 7B TT training (resume from epoch 3, R²=0.768)
cd /home/filip/Projects/Personal/AI/RankAdaptation
source awq_activate.sh  # if AWQ model is needed

# Resume 7B TT training with BS=64
/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u \
  run_train_gen_tt_7b.py --epochs 30 --d-model 768 --d-ff-ratio 4 --bs 64

# 4. After TT training: validate with per-layer sweep
# Using AWQ 7B model (BS=32)
```

## Commit History (Recent)

```
v0.40.2  TT per-layer norm, d_ff=4, CPU-buffer prefetch
v0.40.1  double-buffer VRAM chunk prefetch
v0.40.0  3B AWQ-Marlin + TT training + chunked prefetch
v0.38.0  Autonomous 4-stage sweep pipeline
```

---
*Recovery completed at: 2026-06-16 ~15:20 UTC*
*GPU status: degraded (needs reboot)*
