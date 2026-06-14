# Causal Map — Complex α

## Key Causal Chain

```
Existing trajectory data 
  → compute a[l] = h[l+1] - 2h[l] + h[l-1]     [10 min, no generation needed]
  → compute std(a) and R²_a (TT trained on a)    [30 min training]
  → if R²_a >> 0:                                [CRITICAL GATE]
    → h' = h + r·(cosθ·v + sinθ·a)              [complex steering implemented]
    → steer at L8 with θ sweep                   [1 hour eval]
    → if θ_opt ≠ 0 for L8:                      [confirms complex α > real α]
      → expand to L9 θ sweep
        → if L9 θ_opt ≠ L8 θ_opt:              [confirms phase distinguishes trim-tab from death]
          → per-layer (r, θ) vector              [full complex α framework]
```

## Counterfactuals (cross-pollinated)

| CF | Question | Prediction | Test |
|----|----------|------------|------|
| CF-C1 | What if a[l] ≈ 0 (acceleration is noise)? | Complex α collapses to real α → no improvement | Acceleration R² measurement (10 min) |
| CF-C2 | What if L8 θ_opt = 0 but L9 θ_opt = π? | Sign flip is sufficient — phase just inverts | L8 vs L9 phase sweep (1 hour) |
| CF-C3 | What if L9 θ_opt ≠ {0, π}? | Death layers need phase rotation, not just sign flip | L9 θ sweep (1 hour) — NEW FINDING if true |
| CF-C4 | What if phase is consistent across token positions? | per-token phase is unnecessary | Compare θ_opt at early vs late tokens |
| CF-C5 | What if contrastive TTs have different θ_opt? | Contrastive steering needs different phase than standard | Phase-sweep with v_c - v_i |
