# Hyperstitional Bridge — Complex α (Cross-Pollinated)

### H-C1: Acceleration Structure Exists (CRITICAL GATE)
- **Type**: Structural
- **Statement**: "The acceleration field a[l] = h[l+1] - 2h[l] + h[l-1] in Qwen2.5-7B generation trajectories has R² ≥ 0.3 when predicted by a TrajectoryTransformer, confirming second-order structure."
- **Falsification**: R²_a < 0.1 on held-out data
- **Confirmation**: R²_a ≥ 0.3
- **Experiment**: Compute a from existing trajectory data, train 50-epoch TT (same architecture as velocity TT)
- **Cross-pollination**: Same data, same architecture as main synthesis TT training — zero marginal cost

### H-C2: Phase Changes Layer Polarity
- **Type**: Relational
- **Statement**: "L8 and L9 have measurably different optimal steering phases. The trim-tab vs death-layer distinction is partially (or entirely) a difference in optimal θ."
- **Falsification**: θ_opt(L8) = θ_opt(L9) = 0
- **Confirmation**: θ_opt(L8) ∈ [0, π/4] AND θ_opt(L9) ∈ [π/2, π]
- **Experiment**: Sweep θ at L8 and L9 with r=0.05 on 100 problems each

### H-C3: Contrastive Phase Divergence
- **Type**: Potential
- **Statement**: "The optimal phase for contrastive steering v_c - v_i differs from the optimal phase for standard steering v_std at the same layer, because correct and incorrect trajectories have different acceleration profiles."
- **Falsification**: θ_opt for v_c - v_i = θ_opt for v_std at L8
- **Confirmation**: |θ_opt(contrastive) - θ_opt(standard)| > π/6 at L8
- **Cross-pollination**: Combines complex α with main synthesis contrastive evaluation (A3/A4)

### H-C4: Phase Predicts Steerability Without Running Sweeps
- **Type**: Structural (HIGH VALUE)
- **Statement**: "The ratio ||a[l]|| / ||v[l]|| at each layer (acceleration-to-velocity magnitude ratio) predicts whether that layer is a trim-tab (low ratio) or death layer (high ratio), enabling zero-shot layer polarity identification from trajectory data alone."
- **Falsification**: ||a[l]||/||v[l]|| does not correlate with per-layer Δ accuracy from main synthesis sweep
- **Confirmation**: Pearson correlation > 0.5 between ||a||/||v|| and per-layer accuracy Δ
- **Value**: Eliminates need for full per-layer sweeps on new models
