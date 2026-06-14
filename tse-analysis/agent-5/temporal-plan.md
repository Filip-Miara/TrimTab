# Phase 9: Resource-Budgeted Temporal Phasing

**Subject**: Velocity-based latent steering for language model reasoning
**Date**: 2026-06-14

---

## Available Resources

| Resource | Quantity | Notes |
|----------|----------|-------|
| GPU | 1× NVIDIA (likely 3060/3070/3080 based on 7B model fit) | ~10GB VRAM |
| Internal SSD | ~71GB total (~25GB free after model + trajectories) | Fast I/O |
| External HDD | ~500GB+ | Slow I/O |
| CPU RAM | ~32GB (estimated from OOM errors with large trajectories) | Sufficient with batch management |
| Model | Qwen2.5-7B-Instruct (primary), SmolLM2-360M, Qwen2.5-Math-1.5B | Already downloaded |
| API Budget | $0 (local compute) | No API costs |
| Time | Variable (research project) | Assume sessions of 2-8 hours |
| Trajectory Data | 10.5GB SSD (subset), 35GB HDD (full) | Already collected |

---

## Phase A — Immediate / Diagnostic (≤2 hours each)

### A1: Null Hypothesis Significance Test
**Cost**: 2 GPU-hours (1000 baseline evaluations × 100 problems)
**Experiment**:
```bash
# Run 1000 baselines with different seeds
for i in $(seq 1 1000); do
  python run_math15_sweep.py --stage 2 --seed $i --n-test 100 --baseline-only
done
# Compute 99.9th percentile of accuracy differences
```
**Success Criterion**: +20pp exceeds 99.9th percentile → effect is real
**Failure Criterion**: +20pp within 95th percentile → effect may be noise
**Go/No-Go**: If effect is real, proceed to A2. If noise, re-examine methodology.
**Contingency**: If inconclusive (p between 0.05 and 0.001), increase N from 100 to 500.

### A2: Negative α on Death Layers
**Cost**: 1 GPU-hour (28 layers × ±α × 100 problems)
**Experiment**:
```bash
# Sweep both signs for death layers
python run_math15_sweep.py --stage 1 --n-test 100 --alpha -0.1 --layers 7 9 15 16 17 18 19 20 21 22 23 24 25 26 27
# Compare with +α results from existing sweep
```
**Success Criterion**: At least one death layer (e.g., L9) shows significant positive accuracy change with −α
**Failure Criterion**: No death layer responds to −α (all still negative)
**Go/No-Go**: If any death layer becomes a trim tab with −α, proceed to A3 (combined signs).
**Contingency**: If L9(−α) fails but L15+ succeed, the pattern is still valuable.

### A3: Per-Layer ±α Sweep (Complete)
**Cost**: 3 GPU-hours (28 layers × ±α × 200 problems for statistical power)
**Experiment**:
```bash
# Full sweep of both signs for all layers
python run_math15_sweep.py --stage 1 --n-test 200 --alpha 0.1
python run_math15_sweep.py --stage 1 --n-test 200 --alpha -0.1
# Compute optimal sign per layer
```
**Success Criterion**: Map of optimal sign per layer revealed
**Failure Criterion**: Sign optimality is random (no systematic pattern)
**Go/No-Go**: If clear sign pattern emerges, proceed to Phase B.

### A4: Random Steering Baseline
**Cost**: 1 GPU-hour
**Experiment**: Steer L8 with random directions (same magnitude as TT prediction)
**Success Criterion**: Random steering produces 0pp ± noise
**Failure Criterion**: Random steering accidentally produces +10pp+ (would undermine TT's role)
**Go/No-Go**: Always informative; no binary decision.

---

## Phase B — Short-Term / Targeted (≤1 day each)

### B1: Asymmetric α Optimization
**Cost**: 4 GPU-hours
**Prerequisites**: Phase A3 (sign map) and A4 (random control) completed
**Experiment**:
```bash
# Bayesian optimization over α per layer (positive and negative ranges)
python run_math15_sweep.py --stage 1 --n-test 200 --alpha-range -0.5 0.5 --optimize-bayesian --layers 0 1 2 3 4 5 6 7 8 9
# Repeated for each layer
```
**Success Criterion**: Optimal α varies significantly by layer (not all same)
**Failure Criterion**: All layers have same optimal α (undermines per-layer approach)
**Go/No-Go**: If per-layer α is confirmed, proceed to B2. If uniform, simplify to global α.

### B2: Contrastive TT Evaluation
**Cost**: 1 GPU-hour (already trained, just need evaluation)
**Prerequisites**: Contrastive TTs already trained (best_tt_correct.pt, best_tt_incorrect.pt)
**Experiment**:
```bash
python run_contrastive_eval.py --n-test 200 --alpha 0.1 --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
# Compare: standard TT, contrastive TT, standard+contrastive combined
```
**Success Criterion**: Contrastive TT produces larger accuracy gains than standard TT on at least one layer
**Failure Criterion**: Contrastive TT ≤ standard TT everywhere
**Go/No-Go**: If contrastive is better, proceed to B3 (optimize). If not, question the approach.

### B3: Multi-Layer Combination
**Cost**: 8 GPU-hours
**Prerequisites**: B1 (optimal α per layer) and B2 (contrastive direction)
**Experiment**:
```bash
# Grid search over top-5 layer combinations with optimized signs
python run_multi_layer_sweep.py --layers 8 2 9 7 10 --signs + + - - + --alphas 0.15 0.10 0.08 0.05 0.08
# Test all pairs, then top triples, then all-5 together
```
**Success Criterion**: Multi-layer combination outperforms best single layer (L8: +20pp)
**Failure Criterion**: Multi-layer never exceeds single-layer best (suggests negative interference)
**Go/No-Go**: If multi-layer is additive/synergistic, proceed to Phase C.

### B4: Oscillating α Proof of Concept
**Cost**: 2 GPU-hours
**Prerequisites**: A3 (sign map) completed
**Experiment**:
```bash
python run_oscillating_alpha.py --layers 0 through 27 --alpha 0.1 --frequency 2
# Compare accuracy for +α tokens vs -α tokens within the same generation
```
**Success Criterion**: Within-generation sign preference matches per-layer sign map from A3
**Failure Criterion**: No within-generation sign preference detectable
**Go/No-Go**: If oscillation works, it enables efficient single-pass sign estimation.

---

## Phase C — Medium-Term / Architectural (days-weeks)

### C1: Multi-Scale TrajectoryTransformer
**Cost**: Implementation: 2 days; Training: 2 hours
**Prerequisites**: B1 (optimal α understanding)
**Experiment**:
1. Extend TT to accept (h_t, h_{t-1}, ..., h_{t-4}) as input
2. Add multi-head prediction: (v̂_t, v̂_{t+1}, v̂_{t+2})
3. Train on existing trajectory data
4. Evaluate steering with multi-scale predictions
**Success Criterion**: Multi-scale TT improves steering effect over single-scale by ≥5pp
**Failure Criterion**: Multi-scale TT does not improve steering (or degrades it)
**Go/No-Go**: If multi-scale helps, integrate into pipeline.

### C2: Mechanistic Interpretability Study of L8 vs L9
**Cost**: 1-2 weeks of analysis
**Prerequisites**: Phase A and B complete
**Experiment**:
1. Measure attention patterns at L8 and L9 with and without steering
2. Compute activation patching: intervene at L8, measure effect on L9 output
3. Train sparse autoencoder on L8/L9 hidden states to find interpretable features
4. Identify what computation each layer performs (via probing and causal tracing)
**Success Criterion**: Identify specific computational reason why L8 is a trim tab and L9 is a death layer
**Failure Criterion**: No interpretable difference found (layers are black boxes)
**Go/No-Go**: Always informative, but may not produce actionable insights.

### C3: Learnable Alpha Network
**Cost**: Implementation: 3 days; Training: 4 hours
**Prerequisites**: B1 (α optimization data)
**Experiment**:
1. Train small MLP to predict α(layer, position, hidden_state_norm, task)
2. Training data: sweep results mapping (layer, α) → accuracy
3. Deploy α-network in generation loop
**Success Criterion**: α-network predicts optimal α that outperforms best fixed α
**Failure Criterion**: α-network performance ≤ best fixed α
**Go/No-Go**: If successful, integrate into steering pipeline.

---

## Phase D — Long-Term / Fundamental (months)

### D1: Self-Improving Steering Loop
**Cost**: Implementation: 2-4 weeks; Training: ongoing
**Prerequisites**: C1, C2, C3 all providing value
**Experiment**:
1. Build automated pipeline: steer → evaluate → collect trajectories → train TT → update policy → repeat
2. Policy network (PPO or similar) selects {layer_set, α_per_layer, sign_per_layer}
3. Run for 100+ generations, tracking accuracy improvement trajectory
**Success Criterion**: Monotonic accuracy improvement over generations
**Failure Criterion**: Accuracy plateaus or oscillates
**Go/No-Go**: This is the ultimate goal; if successful, the project achieves autonomous steering optimization.

### D2: Theoretical Upper Bound Analysis
**Cost**: 1-2 months of analysis (no compute)
**Prerequisites**: D1 data or extensive Phase A-C results
**Experiment**:
1. Analyze steering improvement as a function of: model size, baseline accuracy, number of steerable layers, TT quality
2. Fit scaling laws: accuracy_gain = β₀ + β₁·log(baseline) + β₂·n_layers + β₃·R² + noise
3. Estimate maximum achievable improvement via steering
**Success Criterion**: Predictable scaling law that extrapolates to larger models
**Failure Criterion**: No scaling law emerges (results are model-specific)
**Go/No-Go**: Publish as theoretical contribution regardless.

### D3: Multi-Head Contrastive Ensemble
**Cost**: Implementation: 1 week; Training: 1 week
**Prerequisites**: D1 (automated loop), B2 (contrastive evaluation)
**Experiment**:
1. Train N bootstrapped contrastive TT pairs (different trajectory subsets)
2. Aggregate steering directions via majority vote or weighted average
3. Evaluate ensemble steering vs single-pair steering
**Success Criterion**: Ensemble outperforms best single pair by ≥5pp
**Failure Criterion**: Ensemble ≤ best single pair
**Go/No-Go**: If successful, publish as method paper.

---

## Decision Tree

```
START
  │
  ▼
Phase A1: Null hypothesis test
  │
  ├── Effect is real (p < 0.001) ──────► Phase A2: Negative α on death layers
  │                                         │
  │                                         ├── Some death layers become trim tabs ──► A3: Full ±α sweep
  │                                         │                                              │
  │                                         │                                              ▼
  │                                         │                                       Phase B1: Per-layer α opt
  │                                         │                                              │
  │                                         │                                              ▼
  │                                         │                                       Phase B2: Contrastive eval
  │                                         │                                              │
  │                                         │                            ┌─────────────────┼──────────────────┐
  │                                         │                            ▼                  ▼                  ▼
  │                                         │                    Phase B3: Multi-layer  Phase B4: Osc α   Phase C: Arch
  │                                         │                            │                  │                  │
  │                                         │                            └────────┬─────────┘                  │
  │                                         │                                     ▼                            │
  │                                         │                              Phase C1: Multi-scale TT         │
  │                                         │                                     │                            │
  │                                         │                                     ▼                            │
  │                                         │                              Phase C2: Mech Interpret          │
  │                                         │                                     │                            │
  │                                         │                                     ▼                            │
  │                                         │                              Phase C3: α-Network              │
  │                                         │                                     │                            │
  │                                         │                                     ▼                            │
  │                                         │                              Phase D1: Self-Improving Loop   │
  │                                         │                                     │                            │
  │                                         │                                     ▼                            │
  │                                         │                         Phase D2-D3: Theory & Ensemble        │
  │                                         │                                              │
  ├── Effect is noise (p > 0.05) ──────► STOP — Re-evaluate methodology
  │
  └── Inconclusive ──► Increase N to 500, re-run A1
    
  If A3 fails (no sign pattern):
    │
    └──► Phase B1: α optimization (sign may be irrelevant)
          │
          ├── Per-layer α confirmed ──► Continue to B2...
          │
          └── Uniform α ──► Simplify to global α, focus on contrastive direction
```

---

## Total Resource Budget

| Phase | GPU Hours | Wall Time | Experiments |
|-------|-----------|-----------|-------------|
| A1 (Null test) | 2 | 2 hours | 1000 baselines |
| A2 (Neg α) | 1 | 1 hour | 28 layers × 2 signs |
| A3 (Full ±α) | 3 | 3 hours | 56 experiments |
| A4 (Random) | 1 | 1 hour | 28 layers |
| **Phase A total** | **7** | **7 hours** | |
| B1 (α opt) | 4 | 4 hours | Bayesian opt |
| B2 (Contrastive) | 1 | 1 hour | Already trained |
| B3 (Multi-layer) | 8 | 8 hours | Grid search |
| B4 (Osc α) | 2 | 2 hours | 28 layers |
| **Phase B total** | **15** | **~2 days** | |
| C1 (Multi-scale TT) | 2 + impl | ~3 days | New model |
| C2 (Mech Interp) | ~10 | 1-2 weeks | Analysis |
| C3 (α-Network) | 4 + impl | ~5 days | Training |
| **Phase C total** | ~16 | ~3 weeks | |
| D1 (Self-improving) | ~50 | ~1 month | RL loop |
| D2 (Theory) | ~5 | ~2 months | Analysis |
| D3 (Ensemble) | ~20 | ~2 weeks | Training |
| **Phase D total** | ~75 | ~3 months | |

**Total project budget (to D3 completion)**: ~113 GPU-hours, ~4 months wall time
