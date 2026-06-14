# Phase 5: Convergent Pulse

---

## Filter Application

Total candidates from Phase 4: 52 (12 mutation variants + 10 analogues + 15 orthogonal mechanisms + 3 paradoxical combinations + 8 unconventional recombinations + 4 emergent capabilities)

### Filter F1: Feasibility (≥3/5)

Candidates that fail feasibility:

| Candidate | Feasibility | Reason |
|-----------|-------------|--------|
| Graph-based steering (M7 abstract) | 1/5 | Requires full computation graph access and influence propagation — not implemented in any existing framework |
| Per-head steering (M3 scale) | 2/5 | Requires TT to predict per-head velocities (head_dim × n_heads × layers = 32× more outputs) |
| CUDA Graph steering loop (M8 concretize) | 2/5 | Overly complex for current pipeline; requires static graph |
| Diffusion steering (M9 transpose) | 2/5 | Requires repurposing diffusion model for hidden state denoising — speculative |
| Genetic algorithm α (MR4 orthogonal) | 2/5 | Too slow for per-generation alpha optimization |

All others pass F1.

### Filter F2: Safety (No catastrophic failure modes)

Candidates that fail safety:

| Candidate | Risk | Reason |
|-----------|------|--------|
| No TT — random steering (M10 negate) | 4/5 risk of destabilizing generation | Random high-dimensional vectors could push model off-manifold catastrophically |
| Dropout on steering layers (M11 randomize) | 4/5 risk | Random layer selection could accidentally hit death layer |
| Large α negative steering | 4/5 risk | If L8 is high-gain, negative α could amplify errors |

Note: These are not blocked permanently — they could be run with bounded α and monitored — but they fail the "no catastrophic failure" threshold.

### Filter F3: Telos Alignment (≥4/5)

_Telos: Improve reasoning accuracy via velocity-based latent steering._

All high-scoring candidates align with the telos directly (they are about improving steering). The following have lower alignment:

| Candidate | Telos Alignment | Reason |
|-----------|----------------|--------|
| torch.compile for TT (M8 concretize) | 3/5 | Infrastructure improvement, doesn't directly improve reasoning |
| Sparse α (M8 concretize) | 3/5 | Efficiency, not accuracy |
| Steer embedding models for RAG (M9 transpose) | 2/5 | Different task domain |
| Self-application meta-analysis (RECOMB-8) | 2/5 | Meta-level, doesn't improve model reasoning |

### Filter F4: Novelty (≥3/5)

Candidates that fail novelty:

| Candidate | Novelty | Reason |
|-----------|---------|--------|
| Steering + CoT prompting (M5 merge) | 2/5 | Trivial combination |
| torch.compile (M8 concretize) | 2/5 | Standard optimization |
| Sparse α (M8 concretize) | 2/5 | Already implicitly done in per-layer sweep |
| Per-layer α vector (M3 scale) | 3/5 | Already planned in Stage 4 of autonomous sweep |
| Negative α (M2 invert) | 3/5 | At threshold — passes but barely |

### Filter F5: Synergistic Potential (≥3/5)

| Candidate | Synergy Score | Reason |
|-----------|--------------|--------|
| Multi-stage steering (M6 split) | 4/5 | Could combine L2 (encoding) + L8 (reasoning) |
| Dual-surface steering (RECOMB-6) | 5/5 | Opens entirely new dimension of steering |
| Frequency decomposition (RECOMB-5) | 5/5 | Applies to all steering experiments |
| Orchestrated weight+activation steering | 5/5 | EM-3: qualitatively different |
| Keystone layer hypothesis (RECOMB-4) | 4/5 | Reframes all existing results |
| RMS uncertainty-aware α (MR4) | 3/5 | At threshold — moderate synergy |

---

## Survivors & Ranking

Total candidates passing all 5 filters: **31**

### Top-10 Ranked

Score = (Novelty + Feasibility + Telos_Alignment + (6 - Risk)) / 4

| Rank | Candidate | Novelty | Feasibility | Telos | Risk | Score | Category |
|------|-----------|---------|-------------|-------|------|-------|----------|
| **1** | **Dual-surface steering** (EM-3) | 5 | 3 | 5 | 3 | **3.75** | Emergent |
| **2** | **Frequency-decomposed steering + PCA analysis** (EM-2) | 5 | 4 | 5 | 2 | **4.25** | Emergent |
| **3** | **Keystone layer ablation test** (RECOMB-4) | 4 | 5 | 5 | 1 | **4.50** | Emergent |
| **4** | **Contrastive TT vs standard TT on L8** (PC-1 test) | 4 | 5 | 5 | 1 | **4.50** | Experiment |
| **5** | **Per-layer α vector on all layers** (M3) | 3 | 5 | 5 | 2 | **4.00** | Variant |
| **6** | **Multi-stage steering (L2→L8)** (M6) | 5 | 3 | 5 | 3 | **3.75** | Variant |
| **7** | **Asymmetric α sweep L8** (PC-2 test) | 4 | 5 | 5 | 2 | **4.25** | Experiment |
| **8** | **RL-policy for per-token α** (MR4) | 5 | 3 | 5 | 3 | **3.75** | Mechanism |
| **9** | **TT ablation (zeroing) at L8** | 4 | 5 | 4 | 1 | **4.25** | Experiment |
| **10** | **Prototypical steering** (MR2 orthogonal) | 4 | 4 | 5 | 2 | **4.00** | Mechanism |

### Top-5 Rationale

**#3 — Keystone layer ablation test**: Highest-priority experiment. If zeroing L8 steering collapses generation quality, this confirms L8 as a true keystone layer — the most important empirical finding in the entire project. Cost: trivial (modify α=0 for L8 specifically). Risk: almost none.

**#4 — Contrastive vs standard TT on L8**: Resolves the fundamental question of whether contrastive steering is additive or antagonistic with the trim-tab effect. Currently the single most important unknown. Cost: already set up (run_contrastive_eval.py). 

**#2 — Frequency-decomposed steering**: Explains WHY L8 works. If L8 modulates a specific frequency component of hidden states, this is a mechanistic interpretation. Could unlock a general theory of trim-tab layers.

**#5 — Per-layer α vector**: Full optimization of the discovered mechanism. The most straightforward way to maximize the +20pp result.

**#1 — Dual-surface steering**: Highest ceiling but highest cost. If successful, represents a paradigm shift in how steering works (activation + weight modulation simultaneously). Suitable for Phase C.
