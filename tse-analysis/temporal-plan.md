# Phase 9: Resource-Budgeted Temporal Phasing

---

## Available Resources

| Resource | Estimate | Notes |
|----------|----------|-------|
| GPU compute | ~40-80 GPU-hours / week | Assuming 1 A100/H100 |
| Storage | ~35 GB current + 50 GB available | 17GB per tensor, 2 tensors |
| Developer time | ~20 hours / week | Single researcher |
| Data generation | 7B model forward pass | ~2 seconds per trajectory on 1 GPU |
| Budget | Low | Research context; no large funding assumed |

---

## Phase A: Diagnostic (Immediate — ≤2 hours)

**Cost**: 2 GPU-hours, 1 hour developer time

### Experiments (all run in parallel)

| # | Experiment | Cost | Go/No-Go Criterion |
|---|-----------|------|-------------------|
| A1 | **Noise ceiling measurement**: Run Qwen twice on same 100 inputs; compute velocity MSE between runs | 1 GPU-hour | If noise MSE / total MSE > 0.50 → noise ceiling confirmed (stop, change approach). If < 0.30 → headroom exists (proceed). |
| A2 | **Per-layer normalization on 10K subset**: Re-normalize, train TT for 1 epoch | 1 GPU-hour | If R² improves ≥0.01 (from baseline epoch 1) → normalization is a lever. If no improvement → focus elsewhere. |
| A3 | **Synthetic data validation**: 3-layer toy model, known velocity function | 0.5 GPU-hour | If R² > 0.95 → architecture can learn velocities. If < 0.80 → fundamental architecture problem. |

### Success Criterion
All three experiments complete. At least 2 of 3 show positive signal.

### Failure Criterion
- A1 shows noise ceiling reached → **Terminate**: velocity prediction improvement is limited; pivot to usage
- A3 shows architecture inadequacy → **Terminate**: rethink TT architecture entirely

### Go/No-Go Decision
- All 3 positive → Full commitment to Phase B
- 2/3 positive → Proceed with Phase B but limit scope
- 1/3 or fewer positive → Return to research; do NOT proceed to Phase B

---

## Phase B: Short-Term Targeted (≤1 day)

**Cost**: 8 GPU-hours, 2 days developer time

### Prerequisites: Phase A success

### Experiments

| # | Experiment | Cost | Expected R² Gain |
|---|-----------|------|-----------------|
| B1 | **Per-layer normalization full**: Re-normalize all 90K trajectories; train full TT | 6 GPU-hours | +0.02-0.05 |
| B2 | **Decomposed loss**: Implement cosine + Huber loss. Train with α sweep {0.1, 0.3, 0.5, 0.7, 0.9} | 4 GPU-hours | +0.03-0.06 |
| B3 | **Layer-index embedding**: Add learnable layer embedding; train | 2 GPU-hours | +0.01-0.02 |
| B4 | **PCA diagnostics**: Compute PCA on velocity targets; measure intrinsic dimensionality | 1 GPU-hour | Information only |

### Success Criterion
Combined R² improvement ≥ 0.05 (to R² ≥ 0.90). OR directional cos improvement ≥ 0.05.

### Failure Criterion
Total R² improvement < 0.02. This would indicate the R² ceiling is real and close.

### Contingency
If B1 shows no improvement but Phase A's A2 showed improvement → check for implementation error (data leakage, incorrect normalization).

---

## Phase C: Medium-Term Architectural (≤1 week)

**Cost**: 40 GPU-hours, 5 days developer time

### Prerequisites: Phase B success (≥0.03 R² improvement)

### Experiments (in dependency order)

| Step | Experiment | Prerequisites | Cost | Expected Gain |
|------|-----------|---------------|------|---------------|
| C1 | **Multi-format data generation**: BnB + AWQ + GPTQ trajectories (30K each) | None | 60 GPU-hours data gen | Foundation for C2-C5 |
| C2 | **Multi-format mixed training**: Train TT on combined 90K trajectories | C1 | 12 GPU-hours | AWQ R² → 0.70+, BnB R² preserved |
| C3 | **PCA-compressed TT**: PCA→256, train TT in compressed space | B4 (PCA done) | 6 GPU-hours | +0.02-0.08 R², 14× output reduction |
| C4 | **Domain-contrastive loss**: Add gradient reversal layer for quantization invariance | C1, B2 | 8 GPU-hours | AWQ R² → 0.75+ |
| C5 | **Correction network**: MLP mapping AWQ→BnB hidden states | C1 | 4 GPU-hours | AWQ→BnB bridge, TT stays frozen |

### Success Criterion
- AWQ transfer R² > 0.70 (from current 0.45)
- BnB R² > 0.88 (improvement from current 0.848)
- CPT Qwen variant R² > 0.65 (generalization)

### Failure Criterion
- AWQ R² remains < 0.60 post all interventions
- Multi-format training degrades BnB R² below 0.80

### Decision Tree
```
Phase B success → Phase C start
    ├── Multi-format training (C2) works → Continue
    │   ├── PCA compression (C3) works → Full deployment
    │   │   └── AWQ transfer (C4, C5) works → Phase D
    │   └── PCA doesn't help → Skip to C4/C5
    └── Multi-format degrades performance → Roll back to single-format
        └── Focus on normalization + loss only (B1+B2)
```

---

## Phase D: Long-Term Fundamental (≤1 month)

**Cost**: 200 GPU-hours, 15 days developer time

### Prerequisites: Phase C success

### Research Directions (parallel, choose 1-2)

| Direction | Approach | Cost | Risk | Potential |
|-----------|----------|------|------|-----------|
| D1 | **End-to-end RL training**: Differentiate through KV-cache steering; train TT with PPO on GSM8K accuracy | 50 GPU-hours | HIGH (RL stability) | VERY HIGH (bypasses proxy metric) |
| D2 | **Online TT training**: Qwen forward pass during TT training; no stored trajectories | 100 GPU-hours | HIGH (engineering) | VERY HIGH (unlimited data, eliminates I/O) |
| D3 | **Uncertainty-aware TT**: Predict velocity distribution; abstain from steering when uncertain | 30 GPU-hours | MEDIUM | HIGH (safer steering) |
| D4 | **Mamba/SSM TT**: Replace transformer with Mamba for linear-time inference | 40 GPU-hours | MEDIUM | HIGH (efficiency + state tracking) |
| D5 | **Layer-group experts**: 3 specialized sub-TT for early/mid/late layers with soft routing | 40 GPU-hours | MEDIUM | HIGH (specialization) |

### Recommended Prioritization

```
Rank by expected value per GPU-hour:
1. D1 (End-to-end RL) — 10× impact if it works, even if risky
2. D3 (Uncertainty-aware) — Safest, most practical improvement
3. D5 (Layer-group experts) — Most likely to work reliably
4. D4 (Mamba/SSM) — Research-heavy
5. D2 (Online training) — Engineering-heavy, defer if other directions succeed
```

### Phase D Success Criterion
- End-to-end validation: TT→Steering→GSM8K accuracy improvement > 5% (absolute)
- Or: Uncertainty-aware TT enables safe deployment in production

### Phase D Failure Criterion
- No direction yields >2% GSM8K improvement → Reconsider velocity prediction as approach

---

## Decision Tree Diagram

```
START
  │
  ▼
Phase A: Diagnostic (2 hours)
  ├── [Noise ceiling reached] → RECONSIDER approach (stop)
  ├── [Architecture inadequate] → REDESIGN TT (return to research)
  └── [All positive] → ▼
                        │
                   Phase B: Short-term (1 day)
                    ├── [R² gain < 0.02] → ACCEPT ceiling (stop improvements)
                    │                        Pursue deployment with current R²
                    └── [R² gain ≥ 0.03] → ▼
                                            │
                                       Phase C: Medium-term (1 week)
                                        ├── [AWQ R² < 0.60] → ROLLBACK to single-format
                                        │                      Focus on B1+B2 alone
                                        └── [AWQ R² ≥ 0.70] → ▼
                                                              │
                                                          Phase D: Long-term (1 month)
                                                           ├── D1: End-to-end RL
                                                           ├── D3: Uncertainty-aware
                                                           └── D5: Layer-group experts
```

## Budget Summary

| Phase | GPU-hours | Developer Days | Cumulative Cost |
|-------|-----------|----------------|-----------------|
| A | 2 | 0.25 | 2 GPU-hours |
| B | 13 | 2 | 15 GPU-hours |
| C | 90 | 5 | 105 GPU-hours |
| D | 200 | 15 | 305 GPU-hours |

**Total budget**: ~305 GPU-hours + 22 developer days = ~2.5 months for complete program.

**Minimum viable improvement (Phase A+B only)**: 15 GPU-hours, ~2 days → Expected R² 0.87-0.90.
