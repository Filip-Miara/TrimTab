# Phase 1: Atomic Decomposition & Pyramid Construction

**Subject**: RankAdaptation — velocity-based latent steering  

---

## Atoms (12 identified, evidence-grounded)

| ID | Atom | Evidence | Confidence |
|----|------|----------|-----------|
| A1 | **Velocity prediction via TT** | R²=0.94 (SmolLM2), 0.873/0.909 (Math-1.5B contrastive) | 9/10 |
| A2 | **KV-cache modification** | 88% token divergence at α=0.1 (SmolLM2) | 9/10 |
| A3 | **Per-layer trim-tab effect** | L8:+20pp, L9:-23pp (Qwen2.5-7B); SVAMP replicates pattern | 8/10 |
| A4 | **Contrastive direction signal** | v_correct − v_incorrect gives normative steering | 7/10 |
| A5 | **Generation trajectory data** | 92K tokens (7B, 40% correct); 70K tokens (Math-1.5B) | 8/10 |
| A6 | **Distribution shift (prompt→gen)** | Prompt TT R²=0.62 vs gen TT R²=0.94 | 9/10 |
| A7 | **Confidence-accuracy misalignment** | Reading head r=0.85, but τ=1.5 gates <0.1% of tokens | 8/10 |
| A8 | **Attention architecture constraint** | Hybrid (GDN+FA) vs pure MHA — different steering surfaces | 9/10 |
| A9 | **Cross-model transfer** | SmolLM2 TT → 7B preserves L8 as best layer | 7/10 |
| A10 | **Steering requires capability** | <2B models fail GSM8K (0-4%) regardless of steering | 9/10 |
| A11 | **Asymmetric steering potential** | α_c · TT_correct − α_i · TT_incorrect — independent attraction/repulsion | 5/10 |
| A12 | **Multi-head ensemble potential** | Bagging N contrastive pairs reduces variance | 4/10 |

## Pyramid

```
                         ┌────────────────────────────────────┐
                         │      P: LAYERED STEERING           │
                         │  (velocity-based KV-cache modify)  │
                         └──────┬─────────────────────┬───────┘
                    ┌────────────┴──────────┐   ┌────┴──────────────┐
               ┌────▼─────────┐     ┌───────▼┐  │ ┌─▼──────────┐  ┌─▼──────┐
               │ C3 (L3):     │     │ C4(L3): │  │ │ C1(L2):    │  │ C2(L2):│
               │ Steering     │     │ Arch-   │  │ │ Steering   │  │ Eval   │
               │ + Refinement │     │ Aware   │  │ │ Pipeline   │  │ Frame- │
               │ (C1, C2, C4) │     │ Eval    │  │ │ (A1,A2,A8) │  │ work   │
               └┬──────┬──────┘     │ (C2,C3) │  │ └──┬────┬────┘  │ (A3,A5)│
         ┌──────┘      │      ┌─────┘         │  │    │    │       └───┬─────┘
    ┌────▼──┐    ┌─────▼───┐  │ ┌─▼──────┐    │  │    │    │           │
    │C1(L2):│    │ C4(L2): │  │ │ C5(L2):│    │  │    │    │           │
    │Steer  │    │ Refine  │  │ │ Multi- │    │  │    │    │           │
    │Pipe   │    │ (A4,A11,│  │ │ Head   │    │  │    │    │           │
    │A1,A2  │    │ A12)    │  │ │ Ens.   │    │  │    │    │           │
    └──┬────┘    └───┬─────┘  │ │ A12    │    │  │    │    │           │
       │             │       └─┴────────┘    │  │    │    │           │
       A1            A4          A12          A2 A8   A3   A5          A10
       (TT)          (Contr)    (Ensemble)   (KV)(Arch)(Trim)(Data)   (Capability)
                     A11                     A6      A7      A9
                     (Asym α)               (Shift) (Conf) (Transfer)
```

## Junctions

| ID | Type | From | To | Description |
|----|------|------|-----|-------------|
| J1 | compositional | A1 | A2 | TT output → KV modify. R² propagates |
| J2 | evaluative | A1 | A3 | TT quality → trim-tab detection quality |
| J3 | constraint | A8 | A2 | Arch type → modification surface. Hybrid blocks 75% |
| J4 | causal | A6 | A1 | Distribution shift degrades TT's gen-time predictions |
| J5 | regulatory | A7 | A4 | Confidence signal modulates contrastive strength |
| J6 | synergistic | A4 | A11 | Contrastive + asymmetric = richer steering space |
| J7 | compositional | A12 | A4 | Ensemble of contrastive pairs reduces variance |
| J8 | dependency | A10 | A3 | Capability threshold required for measurable steering |
| J9 | transfer | A9 | A3 | Cross-model transfer preserves trim-tab pattern |

## Hallucinatory Pre-Seeds

| Atom | Ideal Form |
|------|-----------|
| A1 | TT that predicts steering direction with R²=0.99 and <1ms latency |
| A4 | Contrastive signal that perfectly separates correct/incorrect manifolds |
| A11 | Learned α vector optimized per (layer, token, problem) in real-time |
| A12 | Ensemble of 100 TTs with bootstrapped data, fully parallel on GPU |
