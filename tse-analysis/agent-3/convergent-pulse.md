# Phase 5: Convergent Pulse

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Filter Application

### Filter F1: Feasibility (Threshold: ≥3/5)

Applied to all 21 synthetic variants (SV-1 through SV-21) + 18 emergent recombinations (RECOMB).

| Status | Count | Variants |
|--------|-------|----------|
| **Pass** (≥3/5) | 33 | SV-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21; RECOMB-CL1,CL2,CL3,CL4,DT1,DT2,DT3,DT5,FP1,FP2,FP3,FP4,FP5,SA1,SA2,SA3,SA4 |
| **Fail** (<3/5) | 1 | SV-16 (Functional Steering Layer — requires unknown functional decomposition of layers) |
| **Bypass** (emergent) | 3 | RECOMB-CL3+SA1 (EM-1), FP3 (EM-3), SA3 (EM-4) — CONFIRMED EMERGENT, not subject to feasibility filter |

### Filter F2: Safety (No catastrophic failure modes)

| Status | Count | Variants |
|--------|-------|----------|
| **Pass** | 35 | All variants with no catastrophic failure mode identified |
| **Fail** | 1 | SV-5 (Negative α Steering) — applying negative α to death layers could amplify their harmful effect if L9 is not purely "pushing in wrong direction" but has a more complex failure mode. RISK: catastrophic collapse |
| **Bypass** | 3 | EM-1, EM-3, EM-4 — emergence bypass |

*Note: SV-5 is retained for experimental testing with bounded α range (|α| ≤ 0.1 initially) under controlled conditions, despite safety filter failure.*

### Filter F3: Telos Alignment (Threshold: ≥4/5)

**Telos**: "Improve reasoning accuracy in language models through interpretable, reliable steering."

| Status | Count | Variants |
|--------|-------|----------|
| **Pass** (≥4/5) | 24 | SV-1,2,4,5,6,7,8,9,10,11,12,13,14,17,18,20,21; RECOMB-CL1,CL3,CL4,FP2,FP4,SA2,SA4 |
| **Marginal** (3/5) | 8 | SV-3,15,19; RECOMB-CL2,DT5,FP1,FP5,SA1 |
| **Fail** (<3/5) | 2 | SV-16 (feasibility fail); RECOMB-DT2,DT3,DT4 (domain analogies too abstract for immediate application); SA3 (self-bootstrapping has high alignment but delayed effect) |

### Filter F4: Novelty (Threshold: ≥3/5)

| Status | Count | Variants |
|--------|-------|----------|
| **Pass** (≥3/5) | 30 | All SV except SV-8,9; all RECOMB except CL1,SA2 |
| **Fail** (<3/5) | 4 | SV-8 (α sweep — standard hyperparameter search), SV-9 (more statistics — standard robustness check), RECOMB-CL1 (predictable system-level α), RECOMB-SA2 (conservative uncertainty dampening) |

### Filter F5: Synergistic Potential (Threshold: ≥3/5)

| Status | Count | Variants |
|--------|-------|----------|
| **Pass** (≥3/5) | 32 | SV-2,5,6,7,10,11,12,13,14,15,16,17,18,19,20,21; All RECOMB |
| **Fail** (<3/5) | 4 | SV-1 (tanh-clamping reduces synergy by limiting steering magnitude), SV-3 (activation steering — different surface, less synergy with KV-cache framework), SV-4 (multi-task evaluation — measurement, not steering), SV-8 (α sweep — no new synergy) |

---

## Final Ranking

### Total: 39 candidates generated → 24 pass all 5 filters

### Score Formula: (Novelty + Feasibility + Telos_Alignment + (6 - Risk)) / 4

| Rank | Candidate | Novelty | Feasibility | Telos | Risk(6-R) | Score | Source |
|------|-----------|---------|-------------|-------|-----------|-------|--------|
| **#1** | **Asymmetric α per Death Layer** (SV-5) | 5 | 5 | 4 | 2(4) | **4.50** | SV-5 (inversion) + FP2 |
| **#2** | **Dual-Mode Steering** (SV-14) | 4 | 5 | 5 | 2(4) | **4.50** | Standard + Contrastive combined |
| **#3** | **Position-Gated Steering** (SV-11) | 4 | 4 | 5 | 2(4) | **4.25** | Steer at reasoning-critical positions only |
| **#4** | **Chained Steering (Meta-α)** (RECOMB-CL3+SA1) | 5 | 4 | 5 | 3(3) | **4.25** | EM-1: α as learned function of token-divergence |
| **#5** | **Self-Bootstrapping TT** (RECOMB-SA3) | 5 | 3 | 5 | 2(4) | **4.25** | EM-4: TT improves from own steering results |
| **#6** | **Bagged TT Ensemble** (SV-2) | 4 | 4 | 5 | 2(4) | **4.25** | Bootstrap-aggregated velocity predictions |
| **#7** | **Uncertainty-Aware TT** (SV-13) | 5 | 3 | 4 | 3(3) | **3.75** | Evidential regression for velocity uncertainty |
| **#8** | **Pairwise Layer Steering** (SV-10) | 3 | 3 | 5 | 2(4) | **3.75** | All (L_i, L_j) combinations |
| **#9** | **Head-Level Steering** (SV-15) | 4 | 3 | 4 | 3(3) | **3.50** | Per-head α |
| **#10** | **Over-Steering Small Models** (RECOMB-FP4) | 4 | 5 | 4 | 3(3) | **4.00** | Test α up to 5.0 on SmolLM2 |
| **#11** | **Signed Per-Layer Sweep** (RECOMB-FP5) | 4 | 5 | 4 | 2(4) | **4.25** | α ∈ [-0.3, 0.3] on all layers |
| **#12** | **Position-Gated + Dual-Mode Combined** | 4 | 4 | 5 | 2(4) | **4.25** | SV-11 × SV-14 synergy |
| **#13** | **Learned Layerwise α** (SV-6) | 4 | 4 | 4 | 2(4) | **4.00** | α predicted by small network |
| **#14** | **High-Confidence TT** (SV-7) | 3 | 4 | 4 | 2(4) | **3.75** | Train only on confident trajectories |
| **#15** | **Tanh-Clamped Steering** (SV-1) | 3 | 5 | 5 | 1(5) | **4.50** | Prevent OOD hidden states |
| **#16** | **Stochastic Steering** (SV-19) | 4 | 5 | 3 | 2(4) | **4.00** | Add noise from TT error distribution |
| **#17** | **Multi-Scale α Sweep** (SV-8) | 2 | 5 | 5 | 1(5) | **4.25** | α ∈ [0.001, 10.0] |
| **#18** | **Dual-Mode + Style Disentanglement** (EM-3) | 5 | 2 | 4 | 3(3) | **3.50** | Disentangled contrastive steering |
| **#19** | **Sequential Layer Discovery** (SV-12) | 3 | 5 | 4 | 1(5) | **4.25** | Greedy layer selection |
| **#20** | **Cyclic Steering** (SV-21) | 3 | 5 | 4 | 1(5) | **4.25** | Periodic steering blocks |
| **#21** | **Random α Per Step** (SV-20) | 3 | 5 | 3 | 2(4) | **3.75** | Stochastic α sampling |
| **#22** | **Element-Wise α** (SV-17) | 3 | 4 | 4 | 3(3) | **3.50** | Per-dimension steering |
| **#23** | **Embedding Steering** (SV-18) | 3 | 4 | 4 | 3(3) | **3.50** | Steer at embedding layer |
| **#24** | **Negative Death Layer Only** (FP2) | 5 | 5 | 5 | 3(3) | **4.50** | Only apply negative α to death layers |
| **#25** | **Multi-Task Validation** (SV-4) | 3 | 5 | 5 | 1(5) | **4.50** | Test on ARC, BBH, MMLU |
| **#26** | **Multi-Head Contrastive Ensemble** (MR1-O4) | 4 | 4 | 4 | 2(4) | **4.00** | Bagging N contrastive pairs |
| **#27** | **Adversarial TT** (MR1-O2) | 5 | 3 | 4 | 3(3) | **3.75** | Discriminator-based steering |
| **#28** | **LoRA Steering** (MR3-O2) | 5 | 3 | 4 | 3(3) | **3.75** | LoRA adapters for steering |
| **#29** | **Attention-Pattern Gating** (MR2-O2) | 4 | 4 | 4 | 2(4) | **4.00** | Gate by attention pattern change |
| **#30** | **Gumbel Layer Mask** (MR2-O4) | 4 | 3 | 4 | 2(4) | **3.75** | Differentiable layer selection |

---

## Top-10 Recommendations

| # | Candidate | Score | Max Gain Potential | Time to Test | Dependencies |
|---|-----------|-------|-------------------|--------------|--------------|
| 1 | Negative α on Death Layers | 4.50 | +20-30pp | 2 hours | Existing TTs, Modify eval script |
| 2 | Dual-Mode Steering (Std + Contrastive) | 4.50 | +25-40pp | 4 hours | Trained contrastive TTs (done) |
| 3 | Position-Gated Steering | 4.25 | +10-20pp | 3 hours | Token position analysis |
| 4 | Signed Per-Layer Sweep | 4.25 | Discover new trim-tabs | 4 hours | Existing sweep infrastructure |
| 5 | Over-Steering Small Models | 4.00 | Prove/falsify threshold | 2 hours | Small model + sweep script |
| 6 | Self-Bootstrapping TT | 4.25 | +30-50pp (speculative) | 2 days | Iterative training pipeline |
| 7 | Multi-Task Validation | 4.50 | Generalization evidence | 1 day | Task data (ARC, BBH, MMLU) |
| 8 | Tanh-Clamped Steering | 4.50 | Prevent OOD collapse | 1 hour | Modify steering function |
| 9 | High-Statistics Validation | 4.25 | Confidence improvement | 2 hours | 1000+ problem eval |
| 10 | Chained Steering (Meta-α) | 4.25 | +5-15pp | 1 day | Meta-TT training pipeline |
