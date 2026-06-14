# Phase 5: Convergent Pulse

**Subject**: RankAdaptation — Velocity-based latent steering
**Date**: 2026-06-14

---

## Filter Application

Total Candidates Generated from Phase 4 + Phase 4b: 37 (12 M1-M12 mutations + 6 M-subvariants + 5 orthogonal mechanism sets + 3 paradoxical combinations + 3 cross-level recombinations + 3 domain transpositions + 3 forbidden pairs + 2 emergent capabilities)
+

### F1: Feasibility (≥3/5)

| Candidate | Score | Rationale | Pass? |
|-----------|-------|-----------|-------|
| M2-1: Anti-steering (flip α sign) | 5/5 | Trivial code change, immediate diagnostic value | ✅ |
| M4-1: Early-only steering | 4/5 | Requires token-position tracking, feasible | ✅ |
| M5-3: Multi-layer combination | 2/5 | Needs sweeping combinatorial infrastructure | ❌ |
| M6-1: Unit-vector steering | 4/5 | Simple norm calculation, feasible | ✅ |
| M5-1: Dual-surface steering | 2/5 | Requires residual stream access + KV access | ❌ |
| M8-1: Head-specific steering | 1/5 | Requires head-level intervention infrastructure | ❌ |
| M3: α sweep per layer | 5/5 | Trivial parameter change to existing scripts | ✅ |
| M10-1: Negative contrastive | 5/5 | Already have contrastive TTs; just flip sign | ✅ |
| M11-1: Random direction baseline | 4/5 | Simple random vector generation | ✅ |
| EM-1: Self-correcting loop | 2/5 | Requires real-time health monitor + adaptive α | ❌ |
| EM-2: Universal manifold verification | 1/5 | Needs 5+ diverse model families on hardware | ❌ |
| RECOMB-FP1: L2+L8 multi-layer | 3/5 | Extends per-layer sweep to pairs; moderate infra cost | ✅ |
| RECOMB-FP2: High α on sub-threshold | 5/5 | Just change α parameter on existing scripts | ✅ |
| RECOMB-FP3: TT → logits | 3/5 | Needs new steering mechanism implementation | ✅ |
| M12-1: Oscillating α | 3/5 | Simple sinusoidal modulation | ✅ |

### F2: Safety (No catastrophic failures)

| Candidate | Safety Assessment | Pass? |
|-----------|------------------|-------|
| M2-1: Anti-steering | May degrade output but not catastrophic | ✅ |
| M4-1: Early-only | Lower total modification = safer | ✅ |
| M6-1: Unit-vector | Same magnitude as baseline = same risk | ✅ |
| M3: α sweep | Some α values may destroy output; diagnostic only | ✅ |
| M10-1: Negative contrastive | Adds intentional degradation; diagnostic use only | ⚠️ (diagnostic bounds) |
| M11-1: Random direction | Diagnostic only; not meant for production | ⚠️ (diagnostic bounds) |
| RECOMB-FP1: L2+L8 | Error compounding risk; start with low α | ⚠️ (needs guard) |
| RECOMB-FP2: High α on sub-threshold | Could destroy already-weak model | ⚠️ (monitor) |
| RECOMB-FP3: TT → logits | Novel surface, unknown failure modes | ✅ (logits less invasive) |
| M12-1: Oscillating α | Generally safe; acts as natural experiment | ✅ |

### F3: Telos Alignment (≥4/5)

| Candidate | Score | Alignment with Project Goal |
|-----------|-------|---------------------------|
| M2-1: Anti-steering | 5/5 | Directly tests death-layer mechanism hypothesis |
| M4-1: Early-only | 4/5 | Tests token-position-steering connection |
| M6-1: Unit-vector | 3/5 | Tests direction vs magnitude importance |
| M3: α sweep | 5/5 | Directly addresses the largest unknown parameter |
| M10-1: Negative contrastive | 4/5 | Tests contrastive direction assumptions |
| M11-1: Random direction | 5/5 | Critical ablation — tests if steering specificity matters |
| RECOMB-FP1: L2+L8 | 5/5 | Addresses multi-layer interaction (BS5) |
| RECOMB-FP2: High α on sub-threshold | 5/5 | Tests the capability threshold hypothesis |
| RECOMB-FP3: TT → logits | 3/5 | Lower priority — logit correction failed before |
| M12-1: Oscillating α | 2/5 | Novel but tangential to core questions |

### F4: Novelty (≥3/5)

| Candidate | Score | Distinction from Existing Work |
|-----------|-------|-------------------------------|
| M2-1: Anti-steering | 5/5 | Not done in any LLM steering work |
| M4-1: Early-only | 3/5 | Related to timing experiments in other contexts |
| M6-1: Unit-vector | 4/5 | Direction/magnitude decoupling is novel |
| M3: α sweep | 3/5 | Obvious experiment, not yet done |
| M10-1: Negative contrastive | 5/5 | Novel inversion of contrastive signal |
| M11-1: Random direction | 4/5 | Critical baseline, not yet done |
| RECOMB-FP1: L2+L8 | 4/5 | Multi-layer steering not yet explored |
| RECOMB-FP2: High α on sub-threshold | 5/5 | Capability threshold stress test is novel |
| RECOMB-FP3: TT → logits | 2/5 | Similar to failed logit correction |
| M12-1: Oscillating α | 4/5 | Novel time-varying approach |

### F5: Synergistic Potential (≥3/5)

| Candidate | Score | Combination Potential |
|-----------|-------|----------------------|
| M2-1: Anti-steering | 5/5 | Combines with α sweep to test layer×α×direction |
| M4-1: Early-only | 4/5 | Combines with M3 and M6 |
| M6-1: Unit-vector | 3/5 | Limited combination potential |
| M3: α sweep | 5/5 | Foundational — every other experiment benefits from optimal α |
| M10-1: Negative contrastive | 4/5 | Combines with M2-1 for complete direction analysis |
| M11-1: Random direction | 3/5 | Baseline only, not combinable |
| RECOMB-FP1: L2+L8 | 5/5 | Gateway to full multi-layer exploration |
| RECOMB-FP2: High α on sub-threshold | 4/5 | Informs MR#4 (capability threshold) strategy |
| RECOMB-FP3: TT → logits | 2/5 | Niche mechanism |
| M12-1: Oscillating α | 3/5 | Specialized, limited combination |

---

## Ranking

Score = (Novelty + Feasibility + Telos_Alignment + (6 - Risk)) / 4

Risk: 1 (very safe) → 5 (very risky) — inverted in formula as (6-Risk)

| Rank | Candidate | Nov (1-5) | Feas (1-5) | Tel (1-5) | Risk (1-5) | Score | Rationale |
|------|-----------|-----------|------------|-----------|------------|-------|-----------|
| **#1** | M3: Per-layer α sweep | 3 | 5 | 5 | 2 | **4.75** | Highest-value experiment: addresses MR#2, trivial to implement, could transform all subsequent results |
| **#2** | M2-1: Anti-steering at death layers | 5 | 5 | 5 | 2 | **4.75** | Critical diagnostic: flipping α at L9 tests whether death is directional or absolute |
| **#3** | M11-1: Random direction baseline | 4 | 4 | 5 | 1 | **4.50** | Essential ablation: tests whether steering signal specificity matters at all |
| **#4** | RECOMB-FP2: High α on sub-threshold | 5 | 5 | 5 | 3 | **4.50** | Tests the fundamental "capability threshold" claim |
| **#5** | RECOMB-FP1: L2+L8 multi-layer | 4 | 3 | 5 | 3 | **4.25** | Gateway to combinatorial steering; addresses BS5 |
| **#6** | M4-1: Early-only steering | 3 | 4 | 4 | 1 | **4.00** | Tests token-position dependency |
| **#7** | M10-1: Negative contrastive | 5 | 5 | 4 | 3 | **4.00** | Tests contrastive direction assumptions |
| **#8** | M6-1: Unit-vector steering | 4 | 4 | 3 | 2 | **3.75** | Decouples direction from magnitude |
| **#9** | RECOMB-FP3: TT → logits | 2 | 3 | 3 | 2 | **3.25** | Lower priority |
| **#10** | M12-1: Oscillating α | 4 | 3 | 2 | 2 | **3.25** | Novel but tangential |

---

## Special Bypass: Phase 4b Emergent Candidates

Per the TSE spec, candidates classified as CONFIRMED EMERGENT bypass F1 (Feasibility):
- EM-1 (Self-correcting loop): Bypasses F1, passes F2-F5 → included as Rank #0 (Theoretical Priority)
- EM-2 (Universal velocity manifold): Bypasses F1, passes F2-F5 → included as ongoing research direction
- EM-4 (Reasoning topography): Bypasses F1, passes F2-F5 → included as theoretical framework

---

## Summary

| Metric | Count |
|--------|-------|
| Total Candidates Generated | 37 |
| Passed All 5 Filters | 10 experiments + 3 emergent |
| Rank #1-#3 (Execute Immediately) | α sweep, anti-steering, random baseline |
| Rank #4-#7 (Execute Short-term) | High-α sub-threshold, multi-layer, early-only, negative contrastive |
| Bypass (Theoretical Priority) | Self-correcting loop, universal manifold, reasoning topography |
