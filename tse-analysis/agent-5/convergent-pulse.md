# Phase 5: Convergent Pulse

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Filter Application

Total candidates from Phase 4: 10 synthetic variants + 10 speculative analogues + 5 orthogonal mechanisms per master regulator + 3 paradoxical combinations + 5 junction variants
**Total Candidates Generated**: 43
**Passed All Filters**: 8

---

## Filter Details

### F1 — Feasibility (≥3/5)

| Candidate | Score | Notes |
|-----------|-------|-------|
| V1: Negative α on death layers | 5 | Trivially feasible — change sign in existing code |
| V2: Activation steering (MLP) | 3 | Requires new hooks into MLP forward pass |
| V3: Multi-scale TT | 5 | Extend TT input to include (h_t, h_{t-1}, ..., h_{t-k}) |
| V4: Oscillating α | 4 | Simple code change to alternate sign every 2 tokens |
| V5: Online TT training | 3 | Requires real-time training loop, potentially heavy |
| V6: Self-improving steering loop | 3 | Requires full automation pipeline — doable but complex |
| V8: Null hypothesis test | 5 | Trivially feasible — run more random seeds |
| PC-1: Zero steering (null test) | 5 | Obviously feasible |
| PC-2: Opposite direction on death layers | 5 | Equivalent to V1 |
| PC-3: Train TT on steered trajectories | 3 | Requires generating steered data first |

### F2 — Safety (No catastrophic failure modes)

| Candidate | Safety Assessment | Pass? |
|-----------|------------------|-------|
| V1 | No new failure mode — same mechanism, opposite sign | ✅ |
| V2 | MLP modification could cause gradient explosion | ⚠️ (pass with bounds) |
| V3 | More compute, no new safety concern | ✅ |
| V4 | Oscillation explores both directions naturally | ✅ |
| V5 | Online training could diverge if not stabilized | ⚠️ (pass with gradient clipping) |
| V6 | Self-improvement loop could amplify harmful steering | ⚠️ (requires guardrails) |
| V8 | No risk — just measurement | ✅ |
| PC-1 | No risk | ✅ |
| PC-2 | Same as V1 | ✅ |
| PC-3 | Mild risk — steered data is off-distribution for training | ⚠️ (pass with validation) |

### F3 — Telos Alignment (≥4/5, toward improved reasoning)

| Candidate | Score | Rationale |
|-----------|-------|-----------|
| V1: Negative α on death layers | 5 | Directly addresses the biggest problem (death layer dominance) |
| V3: Multi-scale TT | 4 | Captures hierarchical structure of reasoning |
| V4: Oscillating α | 4 | Efficiently explores both steering directions |
| V6: Self-improving loop | 5 | The highest-potential approach — autonomous continuous improvement |
| V8: Null hypothesis test | 3 | Important but doesn't directly improve steering |
| PC-3: Train on steered data | 4 | Addresses distribution shift problem |

### F4 — Novelty (≥3/5)

| Candidate | Score | Comparison to Existing |
|-----------|-------|----------------------|
| V1: Negative α on death layers | 5 | No existing experiment tests this simple idea |
| V3: Multi-scale TT | 3 | Obvious extension of 1-step TT |
| V4: Oscillating α | 5 | No prior work on oscillatory steering |
| V6: Self-improving loop | 5 | Novel application of RL to steering optimization |
| V8: Null hypothesis test | 3 | Standard scientific practice |
| PC-3: Train on steered data | 4 | Simple but unexplored |

### F5 — Synergistic Potential (≥3/5)

| Candidate | Score | Compatible With |
|-----------|-------|-----------------|
| V1: Negative α on death layers | 5 | All other variants — orthogonal improvement |
| V3: Multi-scale TT | 4 | V5, V6, PC-3 |
| V4: Oscillating α | 5 | V1, V6, V8 |
| V6: Self-improving loop | 5 | Everything (it's a meta-technique) |
| V8: Null hypothesis test | 4 | Informative for all other experiments |

---

## Ranked Survivors

Score = `(Novelty + Feasibility + Telos_Alignment + (6 - Risk)) / 4`

| Rank | Candidate | Novelty | Feasibility | Telos | Risk (6-R) | Score | Rationale |
|------|-----------|---------|-------------|-------|------------|-------|-----------|
| **#1** | **V1: Negative α on death layers** | 5 | 5 | 5 | 3 | **5.0** | Highest impact per effort. A single code change. Currently the most likely explanation for death-layer effect. |
| **#2** | **V4: Oscillating α** | 5 | 4 | 4 | 4 | **4.25** | Efficiently discovers per-layer sign optimality. Compatible with V1. |
| **#3** | **V6: Self-improving steering loop** | 5 | 3 | 5 | 2 | **3.75** | Highest potential but requires automation infrastructure. Phase-C candidate. |
| **#4** | **V3: Multi-scale TT** | 3 | 5 | 4 | 4 | **4.0** | Solid improvement to TT architecture. Low risk. |
| **#5** | **V8: Null hypothesis test** | 3 | 5 | 3 | 5 | **4.0** | Foundational — must be done before claiming results are real. |
| 6 | PC-2: Opposite direction on death layers | 5 | 5 | 5 | 3 | 5.0 | Same as V1 (duplicate) |
| 7 | PC-3: Train TT on steered trajectories | 4 | 3 | 4 | 2 | 3.25 | Good but requires V6 infrastructure |
| 8 | V5: Online TT training | 4 | 3 | 4 | 3 | 3.5 | Good but complex |

---

## Top-5 Detailed

### #1: Negative α on Death Layers (V1)
**Description**: For layers classified as "death layers" (L9, L7, L15+), steer with −α instead of +α. The TT's R² is symmetric and high on all layers; if applying the TT prediction hurts, applying the opposite should help.
**Implementation**: Modify `run_math15_sweep.py` / `run_contrastive_eval.py` to accept negative α and run per-layer sweep with both signs.
**Expected Effect**: L9→+23pp, L15+→+23pp+. Potentially all layers become trim tabs.
**Cost**: 1 hour compute (28 layers × ±α × 100 problems)
**Risk**: 3/5 — May not work if TT predictions on L9 are truly random in direction. But this is the single highest-value experiment.

### #2: Oscillating α (V4)
**Description**: On each token, alternate between +α and −α. After generation, compare accuracy for tokens steered +α vs −α. This efficiently estimates the optimal sign per layer without separate experiments.
**Implementation**: Add a token-position-based αₜ = α × (-1)^t to the steering code.
**Expected Effect**: Estimates per-layer sign preference in one pass instead of two.
**Cost**: 1 hour compute (28 layers × 100 problems with alternating α)
**Risk**: 2/5 — Tokens are not independent; alternating may cause interference.

### #3: Self-Improving Steering Loop (V6)
**Description**: Automate the full loop: (1) steer with current policy → (2) evaluate accuracy → (3) collect new trajectories → (4) train new TT → (5) update policy → repeat. The policy learns {layer, α, sign} via reinforcement learning over multiple generations.
**Implementation**: Build on `run_autonomous_sweep.py`. Add RL agent that selects {layer, α} per episode.
**Expected Effect**: Converges to optimal steering policy after N generations.
**Cost**: Days-weeks to implement and run. Requires automation infrastructure.
**Risk**: 4/5 — May not converge; RL hyperparameter sensitivity.

### #4: Multi-Scale TrajectoryTransformer (V3)
**Description**: Modify TT to take (h_t, h_{t-1}, ..., h_{t-k}) as input and predict (v_t, v_{t+1}, ..., v_{t+k}) as output. A 2-layer transformer with learned position encoding.
**Implementation**: Extend TT input layer and add multi-step prediction head.
**Expected Effect**: Better velocity prediction by conditioning on trajectory context.
**Cost**: 2 hours to implement; 30 min to train.
**Risk**: 2/5 — May not outperform 1-step TT; longer context could introduce noise.

### #5: Null Hypothesis Significance Test (V8)
**Description**: Run 1000 independent random seeds of the steering experiment with true baseline (no steering) to establish the null distribution of accuracy differences. Test whether the +20pp result exceeds the 99.9th percentile.
**Implementation**: Run baseline evaluation repeatedly with different random seeds.
**Expected Effect**: Either confirms the result is real (p < 0.001) or reveals it's noise.
**Cost**: 2 hours compute (1000 evaluations × 100 problems)
**Risk**: 1/5 — No risk; merely informative.
