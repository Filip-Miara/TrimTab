# Phase 5: Convergent Pulse

## Filtering, Ranking, and Top-5 Selection

---

## Filter Results

Total candidates from Phase 4 (mutants + forced collisions + pre-seed refinements): **37**

### F1 — Feasibility (≥3/5)

| Outcome | Count | Reason for Failure |
|---------|-------|-------------------|
| Passed | 24 | Implementable within current infrastructure or minor additions |
| Failed | 13 | Requires new data (5), new hardware (3), access to Qwen internals (3), unproven methods (2) |

**Notable failures**: V7.2 (QK/OV circuit space — requires mechanistic interpretability infra), V9.2 (GNN on layers — unproven), V12.2 (closed-loop training — requires differentiable steering)

### F2 — Safety (No catastrophic failure modes)

| Outcome | Count | Failures |
|---------|-------|----------|
| Passed | 22 | All safe |
| Failed | 2 | V2.1 (unnormalized data → training instability risk), V10.1 (no normalization → gradient explosion) |

### F3 — Telos Alignment (≥4/5 — moves toward desired state)

| Outcome | Count | Reason for Failure |
|---------|-------|-------------------|
| Passed | 18 | Directly improves R², AWQ transfer, or reasoning accuracy |
| Failed | 4 | Tangential (V10.2 noise addition, V3.3 depth-adaptive — complex with uncertain payoff) |

### F4 — Novelty (≥3/5 — genuinely different from current approach)

| Outcome | Count | Reason for Failure |
|---------|-------|-------------------|
| Passed | 16 | Novel contributions |
| Failed | 2 | V3.1 (just scaling transformer — already obvious) |

### F5 — Synergistic Potential (≥3/5 — combines well with others)

| Outcome | Count | Reason for Failure |
|---------|-------|-------------------|
| Passed | 15 | All synergic with at least 2 other candidates |
| Failed | 1 | V4.1 (pure causal — already shown worse, no synergy) |

### Final Survivors: 15

---

## Ranking

Score = (Novelty + Feasibility + Telos Alignment + (6 - Risk)) / 4

| Rank | ID | Variant | Novelty | Feasibility | Telos | Risk(rev) | Score | Based On |
|------|----|---------|---------|-------------|-------|-----------|-------|----------|
| **1** | V1.1 | **Per-layer normalization** | 3 | 5 | 5 | 5 (risk=1) | **4.5** | M1: SUBSTITUTE |
| **2** | V6.1 | **Direction+magnitude decomposed loss** | 4 | 4 | 5 | 4 (risk=2) | **4.25** | M6: SPLIT |
| **3** | EM-1 | **Multi-format mixed training (quantization-robust)** | 5 | 3 | 5 | 3 (risk=3) | **4.0** | Emergent Discovery |
| **4** | V7.1 | **PCA-compressed velocity prediction (top-256)** | 4 | 4 | 4 | 4 (risk=2) | **4.0** | M7: ABSTRACT |
| **5** | V8.1 | **Layer-index embedding** | 3 | 5 | 4 | 5 (risk=1) | **4.25** | M8: CONCRETIZE |
| 6 | V4.2 | Hybrid bidirectional encoder + causal decoder | 4 | 3 | 4 | 4 (risk=2) | 3.75 | M4: REORDER |
| 7 | V4.3 | Mamba/SSM replacement | 4 | 4 | 3 | 3 (risk=3) | 3.5 | M4: REORDER |
| 8 | EM-3 | Uncertainty-aware prediction (mean+variance) | 4 | 3 | 4 | 3 (risk=3) | 3.5 | V11.1 |
| 9 | V5.2 | U-Net style TT with skip connections | 3 | 4 | 3 | 4 (risk=2) | 3.5 | M5: MERGE |
| 10 | PC-2 | Next-hidden-state prediction (velocity derived) | 5 | 2 | 4 | 2 (risk=4) | 3.25 | Paradoxical |
| 11 | V12.1 | Interleaved MSE+cosine training | 3 | 5 | 3 | 4 (risk=2) | 3.75 | M12: OSCILLATE |
| 12 | V6.2 | Layer-group block experts | 3 | 4 | 3 | 3 (risk=3) | 3.25 | M6: SPLIT |
| 13 | V8.2 | Attention-pattern features as input | 4 | 3 | 3 | 3 (risk=3) | 3.25 | M8: CONCRETIZE |
| 14 | MR-3 | Domain-adversarial loss (gradient reversal) | 5 | 2 | 5 | 2 (risk=4) | 3.0 | Orthogonal Mechanism |
| 15 | PC-3 | Binary steer/no-steer per-token classification | 5 | 3 | 3 | 2 (risk=4) | 3.0 | Paradoxical |

---

## Top-5 Ranked Candidates (With Rationale)

### #1: Per-layer normalization (V1.1) — Score: 4.5/5

**What**: Replace global mean(dim=(0,1)) normalization with per-layer normalization (28 separate mean/std calculations).

**Rationale**: Lowest cost, highest potential impact. Directly addresses the #1 root cause identified by 5 different lenses. Estimated R² gain: +0.02-0.05. AWQ indirect benefit: more accurate per-layer statistics reduce distribution mismatch.

**Implementation**: Change 5 lines of preprocessing code. Run in 1 hour.

### #2: Direction + Magnitude decomposed loss (V6.1) — Score: 4.25/5

**What**: Replace single MSE with two losses: cosine similarity for direction + Huber loss for magnitude. Combine via learned gating parameter α.

**Rationale**: Addresses the fundamental issue that MSE conflates direction and magnitude errors. Cosine loss is more aligned with steering quality (direction matters more than magnitude). Synergistic with V1.1.

**Implementation**: Modify loss function in training loop. 3 hours of code. Add ~10% compute overhead.

### #3: Multi-format mixed training (EM-1) — Score: 4.0/5

**What**: Generate trajectories from BnB + AWQ + GPTQ variants of Qwen. Mix 1:1:1 during training.

**Rationale**: CONFIRMED EMERGENT capability — creates quantization-robust velocity representations. Solves the AWQ transfer problem at the root. Higher-order synergy with #1 and #2 (triple interaction score: 12.5/10).

**Implementation**: Requires 1-2 days of data generation. No model changes. Training time increases 3× (more data).

### #4: PCA-compressed velocity prediction (V7.1) — Score: 4.0/5

**What**: Run PCA on velocity targets. Keep top-K components (estimated: K=256). Predict in PCA space. Transform back.

**Rationale**: If velocity lies on low-dimensional manifold (likely), this drastically reduces output dimensionality (3584→256 = 14× reduction). Makes 48M capacity more effective per output dimension. Also acts as denoising (PCA discards noise dimensions).

**Implementation**: 1 day of analysis + code. Compute PCA on training velocities first.

### #5: Layer-index embedding (V8.1) — Score: 4.25/5

**What**: Add learnable layer-index embedding (28-dim or 64-dim) concatenated to the 3584-dim input at each position.

**Rationale**: Helps TT distinguish between early/mid/late layers explicitly. Very low cost (adds 28×64 = 1,792 parameters). Synergistic with per-layer normalization.

**Implementation**: 45 minutes of code. Negligible compute overhead.

---

## Top Candidates (Ranked)

| Rank | Candidate | Score | Expected R² Gain | AWQ Transfer | Effort |
|------|-----------|-------|-------------------|--------------|--------|
| 1 | Per-layer normalization | 4.5 | +0.02-0.05 | Indirect | 1 hour |
| 2 | Direction+magnitude loss | 4.25 | +0.03-0.06 | Medium | 3 hours |
| 3 | Multi-format training | 4.0 | +0.01-0.03 | HIGH (0.4→0.75+) | 2 days |
| 4 | PCA velocity compression | 4.0 | +0.02-0.08 | Low | 1 day |
| 5 | Layer-index embedding | 4.25 | +0.01-0.02 | Indirect | 45 min |
