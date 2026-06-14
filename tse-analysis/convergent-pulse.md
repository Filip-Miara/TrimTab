# Convergent Pulse — RankAdaptation

## Filter Application

### Total Candidates: 20 (11 from Phase 4 + 4b entries + key mutations)

### Filter Results

| Candidate | F1 (Feas ≥3) | F2 (Safe) | F3 (Telos ≥4) | F4 (Novelty ≥3) | F5 (Synergy ≥3) | Pass? |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Per-head steering (EM-1) | 3 | ✅ | 5 | 5 | 5 | ✅ |
| Adaptive α(t) (EM-2) | 3 | ✅ | 5 | 4 | 5 | ✅ |
| GDN recurrent state steering (RECOMB-5) | 4 | ⚠️ | 4 | 5 | 4 | ✅ |
| Steered-data reading head (RECOMB-6) | 4 | ✅ | 4 | 3 | 4 | ✅ |
| Death-layer sign flip (M2) | 5 | ⚠️ | 5 | 4 | 5 | ✅ |
| Hybrid std+contrastive (M4) | 5 | ✅ | 5 | 4 | 4 | ✅ |
| Self-supervised contrastive (EM-4) | 2 | ✅ | 5 | 5 | 5 | ❌ (Feas) |
| Cross-task polarity (EM-3) | 4 | ✅ | 3 | 3 | 3 | ❌ (Telos) |
| Skip-layer velocity (M1) | 5 | ✅ | 3 | 3 | 3 | ❌ (Telos) |
| Per-token α (M3) | 3 | ✅ | 4 | 4 | 5 | ✅ |
| L8 + Contrastive combo | 4 | ✅ | 5 | 3 | 5 | ✅ |
| Confidence-gated steering | 4 | ✅ | 4 | 4 | 5 | ✅ |

### Safety Warnings

- **GDN recurrent state steering (RECOMB-5)**: Modifying recurrent states could cause unbounded state drift. Requires clamping mechanism.
- **Death-layer sign flip (M2)**: Flipping α sign on L9 could make accuracy worse or destroy generation entirely. Must test with small α first.

## Ranking

Score = (Novelty + Feasibility + Telos_Alignment + (6 - Risk) + Emergent_Potential) / 5

### Top-12 Ranked

| # | Candidate | Novelty | Feas | Telos | Risk(1=low) | Emerg | **Score** | Phase |
|---|-----------|---------|------|-------|-------------|-------|-----------|-------|
| **1** | Death-layer sign flip (L9, α < 0) | 4 | 5 | 5 | 2 | 5 | **4.6** | Immediate |
| **2** | Hybrid standard + contrastive steering | 4 | 5 | 5 | 2 | 4 | **4.4** | Immediate |
| **3** | L8 + Contrastive combo | 4 | 5 | 5 | 2 | 4 | **4.4** | Immediate |
| **4** | Confidence-gated steering (PPL gate) | 4 | 4 | 4 | 2 | 5 | **4.2** | Short-term |
| **5** | Per-head steering (EM-1) | 5 | 3 | 5 | 3 | 5 | **4.2** | Medium-term |
| **6** | Adaptive α(t) via RL (EM-2) | 4 | 3 | 5 | 3 | 5 | **4.0** | Medium-term |
| **7** | GDN recurrent state steering | 5 | 4 | 4 | 3 | 4 | **4.0** | Short-term |
| **8** | Steered-data reading head | 3 | 4 | 4 | 2 | 4 | **3.8** | Short-term |
| **9** | Per-token α (temporal decay) | 4 | 3 | 4 | 3 | 5 | **3.8** | Medium-term |
| **10** | Cross-task polarity transfer (EM-3) | 3 | 4 | 3 | 2 | 3 | **3.4** | Long-term |
| **11** | Skip-layer velocity (l+2 - l) | 3 | 5 | 3 | 1 | 2 | **3.2** | Exploratory |
| **12** | Self-supervised contrastive (EM-4) | 5 | 2 | 5 | 4 | 5 | **3.2** | Long-term |

### Key Insight
The top-3 candidates are all IMMEDIATE experiments — they require no new infrastructure, only reconfiguration of existing code:
1. **Death-layer sign flip**: Change `alpha` to `-alpha` for L9 (`run_per_layer_sweep.py` line 40)
2. **Hybrid steering**: Add β·(v_c - v_i) to standard v in `run_contrastive_eval.py` line 111
3. **L8 + Contrastive**: Already partially implemented — just needs combined evaluation

The first step (death-layer sign flip) is the single highest-ROI experiment: it costs 0 new code, tests a fundamental hypothesis, and could instantly convert the worst layer into a new trim-tab.
