=======================================================================
TRIADIC SYNTHESIS ENGINE — RAPID+EMERGENT MODE
STRATEGY A SOLO DEEP-DIVE
=======================================================================
Subject: Strategy A — KV Consensus Mesh Architecture (Alone)
Mode: rapid (phases 0-5, 4b, 11)
Date: 2026-06-21
Engine: TSE v1.0.0 — Concept-Wise × Meta-Synthesis × Autopoietic

Focus: Technical flaws, failure modes, scalability bottlenecks, edge cases
Target questions:
  (Q1) Is Dual-Path Token Selection sound?
  (Q2) Does Robust Trimmed Merge work for KV tensors (not just scalars)?
  (Q3) What happens at boundaries (prompt→consensus, consensus→free)?

Source: /home/filip/Projects/Personal/AI/RankAdaptation/strategies/strategy_A_architecture.md

=======================================================================
EXECUTIVE SUMMARY
=======================================================================

Strategy A (KV Consensus Mesh) is a bold architecture with a critical
formula-level bug and three deep structural flaws that render the
stated design partially unimplementable as written:

CRITICAL BUG [BLOCKER]: The weight formula `w_i = conf_i ×
(1 - entropy_i / ln(n_kv_heads))` with n_kv_heads=2 produces NEGATIVE
weights for any entropy > ln(2) ≈ 0.693 nats. Since real attention
entropies at L10 for non-trivial tokens are reliably above this
threshold, the formula assigns negative weights to most or all
instances at most tokens. The weighted mean then becomes an
ANTI-consensus — pushing the merged KV away from every instance's
values. This is not a corner case; it is the steady-state behavior.

STRUCTURAL FLAW #1 [HIGH]: The β-blend formula `β_i = β_base ×
(1 − conf_i)` is INVERTED relative to the stated goal. High-confidence
instances (conf_i → 1.0) receive β_i → 0.0, meaning their KV is
almost fully replaced by consensus. Low-confidence instances
(conf_i → 0.0) receive β_i → 0.75, keeping most of their own KV.
This creates a self-reinforcing bifurcation: confident instances
converge to identical KVs while unconfident instances diverge further.
After K=30 steps, "consensus" represents only the confident instances.

STRUCTURAL FLAW #2 [HIGH]: The MAD-protected robust merge is
ill-specified for tensor-valued data. MAD is a scalar robust
statistic; applying it to KV tensors requires choosing an aggregation
norm (per-element? L2-per-head? Frobenius-per-instance?). Each choice
has radically different outlier behavior. The document specifies none.
Furthermore, the median of 64 tensors is not unique — element-wise
median differs from marginal median differs from geometric median.

STRUCTURAL FLAW #3 [MED]: The boundary between Phase 2 (consensus)
and Phase 3 (free generation) leaves each instance with an internally
INCOHERENT KV cache: L10 heads 2&3 are blends of consensus + self,
L10 heads 0,1 are purely from self, and layers 11-35 are entirely from
self. This hybrid state creates representational inconsistency that
layers 11-35 have no mechanism to resolve. Free generation starts from
an incoherent baseline and divergence accelerates from there.

The highest-leverage interventions are: (1) fix the entropy denominator
to ln(16) or replace with a bounded confidence modifier, (2) invert
the β formula or replace with conf_i directly, (3) specify the MAD
norm and median definition for tensor-valued consensus, and
(4) extend consensus to cover all KV heads at L10 (not just 2&3) to
prevent within-layer incoherence.

=======================================================================
PART I: ATOMIC DECOMPOSITION
=======================================================================

Due to space constraints, the full 30-atom decomposition is summarized
by functional groups below. The analysis focuses on the 12 atoms most
relevant to the target questions.

A. Model & Compute Layer
   A1  Qwen2.5-3B-AWQ — 36 layers, GQA (16Q/2KV), d_head=128
   A2  64 instances on single GPU — AWQ 4-bit, ~2.0 GB weights
   A3  Per-instance KV cache — 36 MB each, ~2.3 GB total
   A4  Activations + scratch — ~1.0 GB
   A5  Consensus buffer (L10 h2&3) — ~1.0 MB
   A6  Free memory — ~2.7 GB (of 24 GB consumer GPU)

B. Communication Topology
   B1  Tier 1: Intra-GPU shared memory (64 instances, zero-copy)
   B2  Tier 2: Inter-GPU NCCL AllGather Ring
   B3  CacheCard Protocol — KVConsensusPacket wire format

C. Consensus Engine (CRITICAL FOCUS)
   C1  Dual-Path Token Selection          [Q1 target]
   C2  Confidence-weighted blend (3-step) [Q2 target]
   C3  MAD-protected robust merge         [Q2 target]
   C4  Beta blending factor               [Q2 target]
   C5  Weight formula: w_i = conf_i × (1 - entropy/ln(2))

D. Instance Orchestration
   D1  Phase 1: Prompt processing (~10 ms, no consensus)
   D2  Phase 2: Consensus gen (~360 ms, K=30 tokens, per-token sync)
   D3  Phase 3: Free gen (~4,820 ms, no consensus)
   D4  Total: ~5.2 s for 512 tokens

E. Failure Recovery
   E1  Divergence detection (sliding window of 3 tokens)
   E2  Triangular redundancy (exclude → full replacement β=1.0)

=======================================================================
PART II: VOID — ASSUMPTION SURFACING
=======================================================================

2.1 Explicit Assumptions (from the document)

| # | Assumption | Stated In | Criticality |
|---|-----------|-----------|-------------|
| E1 | L10 heads 2&3 carry sufficient consensus signal | §2, §3 | HIGH — if wrong, entire channel is wasted |
| E2 | GPUs are homogeneous (no straggler, no hardware variation) | §1 (implied by 64 instances) | MED — affects MAD threshold |
| E3 | N=64 instances fit on single GPU | §1 (memory budget) | LOW — verified by calculation |
| E4 | Per-token sync barrier during K=30 is optimal | §4 | HIGH — 30× per-token sync vs fewer |
| E5 | AWQ 4-bit preserves sufficient quality for consensus signal | §1 (uses AWQ) | MED — quantization noise could mask consensus |
| E6 | Three-phase division (prompt/consensus/free) is correct | §4 | MED — alternative splits untested |
| E7 | Token selection from pre-consensus logits avoids chicken-egg | §5 | CRITICAL — [Q1] |
| E8 | MAD + trimmed mean is the right robust merge | §3 | CRITICAL — [Q2] |
| E9 | Confidence × (1-entropy/ln(2)) is the right weight | §3 | CRITICAL — [Q2, formula bug] |
| E10 | β_base=0.75 is the right blend strength | §3 | MED — untested value |
| E11 | 3-token window is sufficient for divergence detection | §6 | MED — could be too short (false positives) |

2.2 Implicit Assumptions (inferred, not stated)

| # | Assumption | Why It Matters |
|---|-----------|----------------|
| I1 | Confidence is computable per-instance per-token | §3 (formula uses conf_i) — BUT NOT DEFINED |
| I2 | Entropy is computable over a well-defined distribution | §3 (formula uses entropy_i) — BUT NOT DEFINED |
| I3 | The merge operates on KV tensors compatibly | §3 (ROBUST MERGE) — MAD for tensors unspecified |
| I4 | The consensus KV can replace instance KV without incoherence | §3 (β blend) — layers 11-35 unmodified |
| I5 | All instances use the same token after ensemble voting | §5 (Dual-Path) — instances that lost vote still attend |
| I6 | L10 is the correct layer for consensus | §2 — no per-layer analysis provided |
| I7 | Ensemble voting mechanism for token selection is defined | §5 — not specified HOW tokens are selected |
| I8 | Free phase token selection continues via ensemble | §4 — not specified what happens after consensus |

2.3 Counter-Assumptions (What If Not)

| Assumption | What If ¬ | Impact |
|-----------|-----------|--------|
| E1 (L10 h2&3 sufficient) | Consensus signal at L10 h2&3 is noise or redundant | Entire architecture is consensus on irrelevant features |
| E7 (Dual-Path avoids chicken-egg) | Post-consensus KV is incompatible with pre-consensus token | Token-KV mismatch degrades every generation step |
| E8 (MAD works for tensors) | Per-element MAD destroys covariance structure | Merged KV has broken attention patterns |
| I1 (confidence definable) | LLM confidence (max softmax) does not correlate with correctness | Weighted merge = random weighting |
| I4 (no incoherence from blend) | L10 hybrid + unblended L11-35 produces incoherent attention | Every instance has internal representational inconsistency |
| I5 (all use same token) | Losing instances still generate with their own logit distribution | Vote winner generates with attention from different distribution |

Bracket Statement: These assumptions are set aside for analysis.
Assumptions I1, I2, and I3 are FALSIFIABLE by inspection of the
document — the architecture design depends on undefined quantities.
They are flagged as BLOCKER-level gaps, not assumptions.

=======================================================================
PART III: DUAL-PATH TOKEN SELECTION DEEP ANALYSIS [Q1]
=======================================================================

3.1 Algorithm Reconstruction

The document states: "Token t selected from pre-consensus logits; KV
consensus modifies what instance attends to when generating token t+1."

Reconstructed execution for a single consensus step at position t:

  Step 1: Each instance i has KV_cache_i[0..t-1] (accumulated through
          previous steps, including prior consensus blends)
  Step 2: Each instance i computes logits for position t:
          logits_i = model.forward(KV_cache_i, position=t)
  Step 3: Token t is selected via ensemble voting on {logits_i}:
          token_t = vote({logits_i})  [MECHANISM NOT SPECIFIED]
  Step 4: KV consensus merges all KV_cache_i:
          KV_consensus = robust_merge({KV_cache_i})
  Step 5: Each instance blends:
          KV_cache_i[t] = β_i × KV_cache_i[t] + (1-β_i) × KV_consensus[t]
  Step 6: Each instance generates token t+1:
          logits_i^{t+1} = model.forward(KV_cache_i[0..t], token_t)

3.2 Critical Analysis

3.2.1 Soundness Issue #1: Logit-Computation Incoherence

At Step 2, KV_cache_i contains entries built from PREVIOUS blended
states. After K=5 consensus steps, KV_cache_i is:
  - KV_cache_i[0..4] = blended versions (instance i with consensus mixing)
  - These entries are different for each i because β_i differs

This means when computing logits for position t, each instance
processes the same sequence tokens 0..t-1 but through DIFFERENT KV
representations. The logit distribution for position t is therefore
not "pre-consensus" in any meaningful sense — it is already shaped by
prior consensus blends.

VERDICT: The claim "pre-consensus logits" is misleading. After step 0,
every logit distribution is already influenced by prior consensus
blends. The separation between "token from pre-consensus" and "KV from
post-consensus" is a distinction without a difference after the first
consensus step.

3.2.2 Soundness Issue #2: Voting Mechanism Gap

"Ensemble voting" is referenced but NEVER SPECIFIED:
  - Majority vote? Only works if 33+ of 64 instances agree on the same
    token — unlikely for free-form generation where token space is 32K
  - Plurality? Rare for large vocabularies — vote is often a single
    token with 1-2 instances, rest scattered
  - Weighted by confidence? Requires confidence to be computed pre-vote
  - Logit averaging? Average the logit distributions, then sample

If voting reduces to: each instance samples independently, majority
wins for each token — this fails for open-ended generation where
32K-vocab plurality is near-zero.

If voting is: average logits → this IS ensemble logit averaging, not
voting. Token selection from averaged logits is a different mechanism.

The most plausible mechanism is logit averaging (compute mean or
weighted mean of the 64 logit distributions, then sample from the
averaged distribution). But this means:
  - The "vote" is continuous (averaging), not discrete
  - The token selection is already a form of consensus
  - The distinction between "token from pre-consensus" and "KV from
    post-consensus" is further blurred because logit averaging IS
    a consensus mechanism operating before KV consensus

3.2.3 Soundness Issue #3: Conservative Token Selection Degrades Quality

Because token t is selected BEFORE KV consensus, the selected token
t is based on the AVERAGE logit distribution across all 64 instances.
Token t+1 is generated using the consensus-merged KV cache.

Consider two cases:
  (a) Correct token: most instances predict token A → vote selects A
  (b) Ambiguous token: 40 instances predict A, 24 predict B → A wins

In case (b), the 24 instances that predicted B now must generate
token t+1 using a KV cache that was blended toward consensus — a
consensus that was computed from states that predicted B. But token A
was selected. The 24 instances are now generating token t+1 with:
  - A token (A) they didn't predict
  - A KV cache partially based on their own (B-predicting) states
  - Attention at L10 heads 2&3 partially replaced with consensus (which
    was computed from both A-predicting and B-predicting states)

This is a representational mismatch: the instance's entire forward
pass for position t+1 is conditioned on token A, but its KV cache
at positions 0..t was computed from a different trajectory (one that
led to predicting B).

3.2.4 Soundness Issue #4: Autoregressive Error Propagation

Over K=30 steps, the representational mismatch compounds:
  - Step 1: 60% of instances predicted A → A selected, 40% have
    trajectory mismatch
  - Step 2: The 40% with mismatch produce different logits → their
    influence on the consensus blend shifts the merged KV
  - Step 3+: The instances increasingly fall into two groups:
    "aligned" (predicted majority token every time) and "unaligned"
    (mismatch accumulated)

After 30 steps, the unaligned instances have KV caches that encode a
fundamentally different token sequence than what was actually
generated. The consensus blend then mixes real-trace KV with
alternate-timeline KV. This is NOT consensus — it is a mixture of
incompatible computational histories.

3.3 Summary: Q1 Verdict

┌─────────────────────────────────────────────────────────────────────┐
│ Q1: Is Dual-Path Token Selection sound?                            │
│                                                                    │
│ VERDICT: NOT SOUND as described. Three distinct failure modes:     │
│   1. After step 0, "pre-consensus" logits are already influenced   │
│      by prior consensus blends — the claimed separation is false.  │
│   2. The voting mechanism is undefined. Logit averaging (most      │
│      plausible) IS already a consensus operation, making the       │
│      "token before KV" distinction meaningless.                   │
│   3. Autoregressive mismatch: instances that lose the vote on      │
│      token t still generate token t+1 from KV built on the        │
│      trajectory that predicted a DIFFERENT token. This compounds   │
│      over K=30 steps.                                              │
│                                                                    │
│ Severity: BLOCKER — the core design has a temporal causality       │
│ inconsistency that cannot be fixed by parameter tuning.            │
│                                                                    │
│ Possible Fix: Vote on KV before token selection, not after. Use    │
│ the merged KV to compute logits AND select the token. This merges  │
│ the "chicken-egg" they try to avoid (but which is actually less    │
│ harmful than the mismatch they create).                            │
└─────────────────────────────────────────────────────────────────────┘

=======================================================================
PART IV: ROBUST TRIMMED MERGE FOR KV TENSORS [Q2]
=======================================================================

4.1 The Algorithm (Reconstructed from §3)

Step 1 — FILTER: Remove low-confidence and relative outlier instances
  - "Low-confidence": unspecified threshold
  - "Relative outlier": unspecified distance metric

Step 2 — COMPUTE WEIGHTS: w_i = conf_i × (1 - entropy_i / ln(2))

Step 3 — ROBUST MERGE: MAD-protected weighted mean, 5×MAD outlier clamp
  - Compute weighted median of 64 KV tensors
  - Compute MAD (median absolute deviation) of 64 KV tensors
  - Clamp any instance's KV at 5×MAD from median
  - Compute weighted mean of clamped values

Step 4 — BLEND: β_i = 0.75 × (1 - conf_i)
  - KV_final_i = β_i × KV_i + (1 - β_i) × KV_merged

4.2 Critical Bug: Weight Formula Produces Negative Weights

The entropy denominator: ln(n_kv_heads) = ln(2) ≈ 0.693

The per-instance weight: w_i = conf_i × (1 - entropy_i / 0.693)

For typical attention distributions at L10:
  - Very sharp attention: H ≈ 0.3-0.6 → 1 - 0.6/0.693 = 1 - 0.87 = 0.13
  - Typical attention: H ≈ 1.0-2.0 → 1 - 2.0/0.693 = 1 - 2.89 = -1.89
  - Diffuse attention: H ≈ 2.5-3.5 → 1 - 3.5/0.693 = 1 - 5.05 = -4.05

Since most attention distributions have entropy well above 0.693 for
non-trivial tokens, the (1 - entropy/ln(2)) factor is NEGATIVE for
most instances. The weights become negative.

In a weighted mean with negative weights:
  - The "consensus" KV is pushed AWAY from instances with negative
    weights — i.e., away from most instances
  - If ALL instances have negative weights, the result is an
    anti-consensus: KV values opposite to what any instance computed
  - If some instances have negative and some positive, the weighted
    mean could produce arbitrarily extreme values (the weights divide
    but don't sum to 1 — normalization by sum of weights is implicit
    but with mixed signs, the denominator can approach zero, producing
    arbitrarily large KV values)

Even if entropy is interpreted differently (e.g., over 16 Q heads
instead of 2 KV heads, where max = ln(16) ≈ 2.77):
  - w_i = conf_i × (1 - entropy_i / 2.77)
  - Still negative for entropy > 2.77 nats (diffuse attention over
    all 16 heads), which is common for early layers (L10 is early)

IT DOES NOT MATTER which entropy interpretation is intended — the
formula is broken for both plausible definitions.

MATHEMATICAL PROOF OF INSTABILITY:

Let S = Σ w_i (sum of weights).
Let KV_consensus = (1/S) × Σ w_i × KV_i (weighted mean).

If some w_i are negative:
  - S can be close to zero (opposing weights cancel)
  - As S → 0: KV_consensus → arbitrarily large (division by near-zero)
  - The norm ||KV_consensus|| grows without bound

This is not a theoretical edge case — for a 64-instance swarm where
entropy follows a roughly normal distribution around H≈1.5,
approximately 95% of weights are negative (since H > 0.693). The few
positive weights (from very sharp attention) are small in magnitude
(conf_i × small remainder). The sum S is dominated by negative
contributions. KV_consensus is qualitatively meaningless.

4.3 MAD Is Undefined for Tensors

The MAD (Median Absolute Deviation) is defined for SCALARS:
  MAD = median(|x_i - median(x)|)

For K and V tensors of shape [2, t, 128], there are multiple
inequivalent generalizations:

┌─────────────────────────────────────────────────────────────────────┐
│ Norm Choice     │ Definition                   │ Behavior            │
│─────────────────│──────────────────────────────│─────────────────────│
│ Per-element MAD │ For each of 2×t×128 elements│ Destroys covariance  │
│                 │ independently: compute       │ between positions &  │
│                 │ median_i, MAD_i, clamp_i     │ heads — attention    │
│                 │                              │ pattern collapses    │
│─────────────────│──────────────────────────────│─────────────────────│
│ Frobenius MAD   │ ||KV_i||_F = sqrt(Σ x²)     │ Loses per-position  │
│                 │ MAD of 64 scalar Frobenius   │ info — an instance   │
│                 │ norms → one clamp per        │ with high norm but  │
│                 │ instance (scale the whole    │ correct values gets │
│                 │ tensor)                      │ uniformly scaled     │
│─────────────────│──────────────────────────────│─────────────────────│
│ Per-position    │ For each position p:         │ Most plausible —     │
│ MAD             │ compute median_i of KV_i[p], │ preserves per-       │
│                 │ MAD_i of KV_i[p], clamp      │ token structure but  │
│                 │ KV_i[p] in place             │ independent per-     │
│                 │                              │ token clamping still │
│                 │                              │ destroys cross-token │
│                 │                              │ attention patterns   │
│─────────────────│──────────────────────────────│─────────────────────│
│ Per-head MAD    │ For each of 2 heads:         │ Preserves head       │
│                 │ vector MAD over 64×128-d     │ structure but 128-d  │
│                 │ points → one clamp per head  │ vector MAD is itself │
│                 │                              │ undefined (needs     │
│                 │                              │ distance metric)     │
└─────────────────────────────────────────────────────────────────────┘

The document specifies NONE of these. Each produces qualitatively
different behavior. Furthermore:

  - The 5×MAD clamp threshold assumes the data follow a symmetric,
    unimodal distribution (like Gaussian). KV tensors have complex
    structure — outliers may be meaningful (representing correct
    but unusual reasoning) rather than erroneous.
  - MAD is asymptotically 37% efficient for normal data — with N=64,
    the MAD estimate has high variance. Using 5×MAD cuts off
    approximately everything beyond the most extreme 2% of instances.
    With N=64, that's about 1.3 instances — effectively no clamping.

4.4 Median of Tensors Is Not Unique

The "median" of 64 KV tensors requires a definition:
  - Element-wise median: median of 64 scalars per element → produces
    a tensor that may not correspond to ANY instance's KV (elements
    taken from different instances)
  - Geometric median: min_x Σ ||x - KV_i|| → unique but expensive
    to compute (iterative optimization, O(N·d·iter))
  - Marginal median: per-element median → same issue as element-wise
  - Medoid: actual instance closest to geometric median → loses
    signal from all other instances

The weighted median (for the weighted mean) is similarly
ill-defined for tensors. The standard weighted median requires
sorting scalars — not applicable to KV tensors.

4.5 β-Blend Formula Is Inverted

β_i = β_base × (1 - conf_i) = 0.75 × (1 - conf_i)

┌───────────────┬─────────┬─────────────────────────────────────┐
│ conf_i        │ β_i     │ Effect on KV_final_i                │
├───────────────┼─────────┼─────────────────────────────────────┤
│ 0.9 (high)    │ 0.075   │ 7.5% self + 92.5% consensus        │
│ 0.5 (medium)  │ 0.375   │ 37.5% self + 62.5% consensus        │
│ 0.1 (low)     │ 0.675   │ 67.5% self + 32.5% consensus        │
│ 0.0 (none)    │ 0.75    │ 75% self + 25% consensus            │
└───────────────┴─────────┴─────────────────────────────────────┘

INTUITIVE EXPECTATION: High-confidence instances should KEEP more
of their own KV (high self-weight). Low-confidence instances should
be pulled toward consensus (low self-weight).

ACTUAL BEHAVIOR: INVERTED — confident instances have their KV almost
entirely replaced by consensus.

This creates a self-reinforcing bifurcation over K=30 steps:

  Step 1:
    Instance A (conf=0.9): 92.5% consensus, 7.5% self
    Instance B (conf=0.2): 32.5% consensus, 67.5% self

  Step 2:
    A is now very close to consensus, generates plausible next token
      → stays confident → stays near consensus
    B is still largely itself, generates potentially different token
      → may lose confidence → stays divergent

  Step 30:
    All initially-confident instances have nearly identical KV
    All initially-unconfident instances have diverged further
    "Consensus" represents only the confident trajectory
    Other instances are excluded from the effective system

This is not consensus — it is MAJORITY CAPTURE. The system converges
to a single trajectory determined by whichever instances happened to
be confident at step 1. Any diversity is systematically eliminated.

4.6 Summary: Q2 Verdict

┌─────────────────────────────────────────────────────────────────────┐
│ Q2: Does Robust Trimmed Merge work for KV tensors (not scalars)?   │
│                                                                     │
│ VERDICT: NO — fails on three independent axes:                     │
│                                                                     │
│ AXIS 1 (MATHEMATICAL): The weight formula w_i = conf_i ×            │
│   (1 - entropy/ln(2)) produces negative weights for the vast        │
│   majority of instances at most tokens. The weighted mean with      │
│   mixed-sign weights is numerically unstable (division by near-zero │
│   sum), and the result is an anti-consensus rather than a true      │
│   consensus. This is a BLOCKER-level formula bug.                  │
│                                                                     │
│ AXIS 2 (STATISTICAL): MAD-robust merge is ill-specified for         │
│   tensor-valued data. Four unequalivalent norm choices exist,       │
│   each with different outlier behavior. Median of tensors is        │
│   not unique (element-wise ≠ geometric ≠ marginal). 5×MAD clamp    │
│   at N=64 is effectively no clamping (affects ~1 instance).        │
│                                                                     │
│ AXIS 3 (DYNAMIC): The β-blend formula is INVERTED — confident      │
│   instances are pulled toward consensus, unconfident instances      │
│   stay divergent. Over K=30 steps, this creates majority capture:   │
│   the "consensus" represents only the initially-confident          │
│   trajectory, and all diversity is eliminated. This is not          │
│   multi-agent consensus; it's single-agent determinism.            │
│                                                                     │
│ Severity: BLOCKER — three fundamental flaws that each independently │
│   invalidate the merge as described.                               │
│                                                                     │
│ Recommended Fixes:                                                  │
│   1. Replace entropy term with bounded modifier:                    │
│      w_i = conf_i × max(0, 1 - H_i / H_max) with H_max=ln(16)     │
│      — GUARANTEES non-negative weights.                            │
│   2. Specify MAD norm explicitly as per-position L2:                │
│      for each position p: compute ||KV_i[p]||_2 over 2 heads × 128 │
│      dim, then scalar MAD on 64 norms. Clamp KV_i[p] as tensor if  │
│      norm exceeds 5×MAD.                                            │
│   3. Invert β formula: β_i = β_base × conf_i (NOT 1 - conf_i)     │
│      — high confidence keeps more of self, low confidence gets     │
│      pulled to consensus. This matches intuition and prevents      │
│      majority capture.                                             │
└─────────────────────────────────────────────────────────────────────┘

=======================================================================
PART V: BOUNDARY ANALYSIS [Q3]
=======================================================================

5.1 Boundary 1: Prompt Processing → Consensus Generation (Phase 1 → 2)

State at the boundary:
  64 instances × KV cache of shape [36, 2, prompt_len, 128]
  prompt_len: length of input prompt (e.g., 150-400 tokens for GSM8K)
  KV caches built WITHOUT consensus — independent prompt processing

5.1.1 Expected State: Near-Identical KV (Best Case)

If prompt processing is fully deterministic (greedy decoding, no
dropout, deterministic CUDA kernels), all 64 instances should have
nearly identical KV caches after prompt processing. The first
consensus step is a near-trivial merge of nearly-identical tensors.

ISSUE: Non-deterministic CUDA operations (flash attention, atomic
adds in AWQ matmul) cause bit-level differences in the KV cache
across 64 parallel forward passes. These accumulate across
prompt_len tokens. The differences may be small in L2 norm but
grow with prompt length. For a 400-token prompt, the divergence
at L10 from numerical non-determinism alone could be significant.

5.1.2 Realistic State: Small but Systematic Divergence

Even without sampling non-determinism:
  - Flash attention uses stochastic rounding in some implementations
  - AWQ 4-bit matrix multiplication uses atomic adds
  - GPU warps execute in non-deterministic order

Each instance processes the SAME prompt but with different numerical
rounding. The KV differences at L10 after prompt_len tokens are:
  - Tiny per position (≈ 1e-6 relative error)
  - Accumulated across prompt_len (≈ prompt_len × 1e-6)
  - At prompt_len=400: ≈ 4e-4 relative error in KV tensor norms

The first consensus step must distinguish between:
  (a) "True divergence" — instances that genuinely disagree on
      how to represent the prompt (should be rare for same prompt)
  (b) "Numerical divergence" — bit-level differences from
      non-deterministic hardware (present in all instances)

Since both types exist, the MAD + outlier detection will inevitably
flag some instances as "divergent" when they are merely victims of
numerical noise. This causes false positives in divergence detection.

5.1.3 Prompt Engineering Issue: Multi-Turn / Chat Format

If the prompt includes few-shot examples or chat formatting, the
prompt processing phase handles identical tokens for all instances.
But if the system uses different prompt templates or few-shot
selections per instance (for diversity), the KV caches encode
different prompts. Consensus on the FIRST token after different
prompts is meaningless.

Assumption E1 (L10 heads 2&3 carry consensus signal) is violated
if the prompts differ — the KV representations at L10 encode
different contexts that should NOT be merged.

5.2 Boundary 2: Consensus Generation → Free Generation (Phase 2 → 3)

State at the boundary (after K=30 consensus tokens):
  64 instances, each with KV cache of shape [36, 2, (prompt_len+30), 128]
  At L10 heads 2&3: KV is a blend of self + consensus (different per
    instance based on β_i trajectory)
  At L10 heads 0,1: KV is purely from self (never blended)
  At all layers 0-9, 11-35: KV is purely from self (never blended)

5.2.1 Critical Incoherence: Within-Layer Hybrid State

At L10, the 2 KV heads (2&3) have been consensus-blended, but the
other 0 KV heads... wait. The architecture has only 2 KV heads total
(GQA with 2 KV heads for Qwen2.5-3B). So L10 heads 2&3 = ALL KV heads
at layer 10.

CORRECTION: With GQA ratio 16:2, there are exactly 2 KV heads total.
Heads 2 and 3 (0-indexed? Or heads 2 and 3 of 16 Q heads?)...

Let me re-read: "KV heads: 2 (GQA, shared across 8 Q heads each)"
"layer 10 heads 2&3"

With 2 KV heads labeled 0 and 1, "heads 2&3" is confusing. Possibly
heads 2 and 3 of the 16 Q heads — but KV heads are shared, so Q
heads 2 and 3 map to KV head 0 (assuming round-robin: Q heads 0-7
→ KV head 0, Q heads 8-15 → KV head 1).

This means "L10 heads 2&3" might reference Q heads 2 and 3, which
both map to KV head 0. If so, the consensus operates on only 1 of
2 KV heads. The other KV head (1) is never consensus-blended.

This is even worse: KV head 0 is blended, KV head 1 is not. At L10,
half the attention keys/values are consensus and half are not. The
attention computation at L10 mixes both heads:
  Attention = softmax(Q @ K^T / √d) where K = [K_head0 || K_head1]
  K_head0 is consensus-blended, K_head1 is not

This produces attention patterns that are part-consensus, part-self —
an incoherent hybrid.

Wait, let me re-examine. "Q heads: 16 | KV heads: 2 | GQA, shared
across 8 Q heads each." With d_head=128, d_model=2048.

For GQA with 16 Q heads and 2 KV heads:
  - Q heads 0-7 attend to KV head 0
  - Q heads 8-15 attend to KV head 1

"L10 Heads 2&3" — this could refer to indices among the 16 Q heads.
Q head 2 and Q head 3 both map to KV head 0 (since 0-7 → KV head 0).
So "heads 2&3" means: we extract and merge the K and V components
for Q heads 2 and 3, which both use KV head 0.

But KV head 0 serves Q heads 0-7. By only sharing/consensusing on
heads 2&3, the other heads 0,1,4,5,6,7 are left untouched.

This makes the "consensus buffer (L10 h2&3) ~1.0 MB" calculation:
  - 1 KV head × 2 (K+V) × 512 tokens × 128 dim × 4 bytes (FP32?) or
    2 bytes (FP16?)
  = 1 × 2 × 512 × 128 × 4 = 524,288 bytes ≈ 0.5 MB for FP32
  = 1 × 2 × 512 × 128 × 2 = 262,144 bytes ≈ 0.25 MB for FP16

The document says ~1.0 MB. If they serialize with metadata (instance
IDs, confidence, entropy, step) that adds ~1,032 bytes per packet ×
64 = 66 KB total metadata. So the KV data is ~1.0 - 0.066 = ~0.934 MB.

For 1 KV head: 2 × seq_len × 128 × bytes_per_element.
At seq_len=512: 2 × 512 × 128 × 4 = 0.5 MB (FP32)
So at seq_len=30: 2 × 30 × 128 × 4 = 30,720 bytes = 30 KB (per instance)
With 64 instances: 64 × 30 KB = 1.92 MB... hmm that doesn't match.

Actually, re-reading: the ~1.0 MB is the consensus buffer size for
L10 h2&3 at 512 tokens, not the per-step packet. The per-step packet
is ~1,032 B.

For the buffer: 1 KV head × 2 (K+V) × 2 bytes (FP16) × 512 × 128
= 1 × 2 × 2 × 512 × 128 = 262,144 = ~0.26 MB. Plus 64 instance slots
= 64 × 0.26 = ~16.6 MB? That's too high.

OR: The consensus buffer stores the consensus result (1 KV blend),
not per-instance KV:
  1 KV head × 2 (K+V) × 512 tokens × 128 dim × 2 bytes (FP16)
  = 262,144 bytes ≈ 0.26 MB

Plus per-step metadata buffers. The ~1.0 MB probably includes
working buffers, scratch space, and the merged result.

OK, the key insight stands: "heads 2&3" refers to Q heads 2 and 3,
both mapping to KV head 0. Consensus operates on only 1 of 2 KV
heads. HEAD 1 IS NEVER BLENDED.

This means:
  - At L10, KV head 0 is consensus-blended, KV head 1 is not
  - Q heads 0-7 use a hybrid: partly blended (KV head 0) and
    partly original (KV head 1)
  - This hybrid goes into the attention computation
  - Layers 11-35 receive this hybrid input but have no knowledge
    that any blending occurred

5.2.2 Cascading Incoherence Through Layers 11-35

The output of L10 attention is:
  O = Σ_{h=0}^{7} softmax(Q_h @ K^T / √d) × V_shard_h

Where K = [K_head0 || K_head1] and V similarly. K_head0 has been
consensus-blended; K_head1 has not.

The attention output O → L10 FFN → L11 input. L11 processes this as if
it were normal hidden states. But it contains an inconsistent mixture
of self and consensus information.

For layers 11-35, this inconsistency is invisible — they process
whatever L10 produces. Over 26 layers of forward pass, the
inconsistency can:
  - Grow (if later layers amplify the blended-original differences)
  - Shrink (if later layers are robust to the perturbation)
  - Shift (if different layers specialize in different feature types)

Without analysis or ablations, it's unknowable which happens. But
the RISK is that the representational inconsistency grows through the
stack, producing degraded generation quality that increases with the
proportion of blended tokens in the sequence.

5.2.3 Immediate Post-Boundary Divergence

At the consensus→free boundary:
  1. KV consensus stops — no more merge, no more blend
  2. Each instance's L10 KV head 0 is at its last blended state
  3. L10 KV head 1 and all other layers are purely self
  4. Instances generate independently from this point

The first free token (token 31) is generated from a set of 64
internally-inconsistent KV caches. Each cache has the same prompt and
first 30 tokens, but different KV representations. The instances'
logits for token 31 will D I V E R G E immediately.

At token 31, the KV caches are already different (different blend
trajectories over 30 steps). The free phase accelerates divergence
because:
  - No more consensus blending to pull divergent instances back
  - Each instance attends to different tokens (because logits differ)
  - KV caches encode different token sequences
  - No mechanism (not even ensemble voting, which is undefined for
    free phase) to re-align them

After 10 free tokens (tokens 31-40), the 64 instances are essentially
independent models generating different completions. The architecture's
consensus mechanism has no effect past the 30-token boundary.

5.2.4 Is 30 Tokens Enough for Consensus Benefits?

If the consensus signal from 30 tokens is sufficient to influence the
free-phase generation quality, then the boundary might still produce
benefit. But:
  - The last 10 consensus tokens (steps 21-30) are generated from
    increasingly divergent KV caches
  - The first 10 consensus tokens (steps 1-10) are overwritten in KV
    depth by later tokens
  - Only tokens ~15-30 at positions near the boundary strongly
    influence free-phase logits

This suggests that only ~15 tokens of effective consensus influence
free generation. The other ~15 tokens of consensus effort are
"wasted" on early positions that are overwritten by later context.

5.2.5 Prompt→Consensus Timing Asymmetry

Phase 1 (prompt): ~10 ms — NO consensus
Phase 2 (consensus K=30): ~360 ms — 30× per-token sync
Phase 3 (free): ~4,820 ms — NO consensus

The consensus phase (360 ms) is 7.5% of total generation time (5,200 ms)
but 100% of all communication overhead. For this 7.5% time investment,
the architecture achieves:
  - 30 tokens of shared representation
  - At the cost of severe internal incoherence (hybrid KV per instance)
  - After which the system reverts to independent generation

The benefit-cost ratio is questionable: 7.5% overhead for control of
the first 30 tokens only, with the side effect of corrupting the
internal consistency of 64 instances.

5.3 Summary: Q3 Verdict

┌─────────────────────────────────────────────────────────────────────┐
│ Q3: What happens at boundaries?                                     │
│                                                                     │
│ BOUNDARY 1 (Prompt→Consensus):                                      │
│   - First consensus step merges near-identical KV (same prompt)     │
│   - Numerical non-determinism creates small divergences that MAD    │
│     falsely flags as outliers                                       │
│   - RISK: false outlier detection poisons first consensus quality   │
│   - SEVERITY: MEDIUM (can be mitigated by skipping first merge)     │
│                                                                     │
│ BOUNDARY 2 (Consensus→Free):                                        │
│   - 64 instances have internally incoherent KV caches: KV head 0    │
│     at L10 has been consensus-blended; KV head 1 at L10 and all     │
│     other layers are unblended                                      │
│   - Layers 11-35 process this hybrid state with no knowledge of     │
│     the blend — representational inconsistency propagates and may   │
│     amplify                                                         │
│   - Free generation starts from 64 different mixed states →         │
│     immediate divergence, no re-alignment mechanism                 │
│   - Only ~15 of 30 consensus tokens meaningfully influence free     │
│     generation (early tokens overwritten by context)               │
│   - SEVERITY: HIGH — free phase is 93% of generation but starts    │
│     from an incoherent baseline with no recovery mechanism          │
│                                                                     │
│ RECOMMENDATION: Extend consensus to cover BOTH KV heads at L10      │
│   (prevent within-layer incoherence) and either (a) continue        │
│   lightweight consensus into free phase, or (b) eliminate the       │
│   boundary entirely by having consensus for all generated tokens    │
│   with adaptive frequency (sparser as generation stabilizes).       │
└─────────────────────────────────────────────────────────────────────┘

=======================================================================
PART VI: SCALABILITY BOTTLENECKS & SYSTEM-LEVEL FLAWS
=======================================================================

6.1 Memory Bandwidth Bottleneck: 64-Way Prompt Processing

Phase 1 processes the same prompt 64 times independently on ONE GPU.
For a 400-token prompt on Qwen2.5-3B:
  - Each forward pass: ~6.8 GFLOps (prefill)
  - 64 instances: ~435 GFLOps total prefill
  - With memory bandwidth ~900 GB/s (RTX 4090): ~8 ns per token per
    instance
  - 400 tokens × 64 instances × 8 ns = ~205 ms, not 10 ms

Wait, the document says Prompt Processing is ~10 ms. How?

With 64 instances batched into a single forward pass:
  - Batch size 64, each with prompt_len tokens
  - Flash attention handles batch × seq_len key-value pairs
  - The forward pass processes all 64 prompts simultaneously
  - Prefill time for batch 64 × 400 tokens: dominated by attention
    computation, not memory

Estimated prefill for Qwen2.5-3B on RTX 4090:
  - Single instance, 400 tokens: ~15-25 ms
  - 64 instances, 400 tokens: batch attention scales as O(B × S²)
    for the attention part but O(B × S) for the MLP part
  - Total prefill for 64 × 400: roughly 64× the single-instance time
    for the compute-bound portions, less for memory-bound portions
  - Estimate: 64 × 25 ms ÷ 4 (attention is more efficient batched) =
    ~400 ms, not 10 ms

The document's "~10 ms" for prompt processing seems optimistic by
~40× for a 400-token prompt. If the prompt is shorter (e.g., 50 tokens
for a simple question): 64 × 50-token prefill ~50-80 ms — still not
10 ms.

This discrepancy suggests the document underestimates prompt processing
time by a factor of 5-40x. If actual prompt processing is 200-400 ms,
the total generation time is not 5.2 s but 5.4-5.6 s — a minor
overall difference, but the PHASE RATIO changes: consensus phase is
a smaller fraction of total, reducing its potential impact.

6.2 Consensus Phase Compute Scaling

Per consensus step, the merge involves:
  - Extract K and V for L10 head 0 (Q heads 2&3) from each of 64
    instances — 64 × 2 × 128 × 2 bytes (FP16) = 32,768 bytes
  - Compute weights: 64 × (conf × entropy) = negligible
  - Compute element-wise or per-position median of 64 tensors
    [2, t, 128] → O(64 × 2 × t × 128) = O(16,384 × t) operations
  - Compute MAD: another O(16,384 × t) operations
  - Clamp at 5×MAD: O(16,384 × t) operations
  - Compute weighted mean: O(64 × 2 × t × 128) = O(16,384 × t) operations
  - Blend each instance: O(64 × 2 × t × 128) = O(16,384 × t) operations

Total per step: ~5 × 16,384 × t = ~82,000 × t operations

At t=30: ~2.5M operations — negligible on GPU (sub-millisecond)
At t=512 (if extended): ~42M operations — still sub-millisecond

The merge compute is NOT the bottleneck. The bottleneck is the
synchronization (waiting for all 64 instances to reach the sync
barrier), not the merge math.

6.3 Memory Fragmentation at 64 Instances

64 instances × per-instance KV cache (36 MB) = 2.3 GB
Model weights (AWQ 4-bit) = ~2.0 GB
Activations (batch 64) = ~1.0 GB
Consensus buffer = ~1 MB

Total: ~5.3 GB. On a 24 GB GPU, free: ~18.7 GB.

The 18.7 GB free is NOT usable for more instances because each
additional instance adds an extra 36 MB for KV cache + activation
memory proportional to batch size growth. For 128 instances:
  - KV: 128 × 36 MB = 4.6 GB
  - Model: 2.0 GB (same)
  - Activations: ~2.0 GB (batch 128)
  - Total: ~8.6 GB — still fits on 24 GB

So the architecture can actually scale to 128+ instances on a single
consumer GPU. The 64-instance limit is arbitrary. BUT:

  - Each instance computes its own forward pass (even with batching,
    the compute scales roughly linearly with batch size)
  - At 128 instances, the per-token latency increases proportionally
  - The 360 ms consensus phase at 64 instances becomes ~720 ms at
    128 instances
  - The free generation (4,820 ms at 64) becomes ~9,640 ms at 128

The architecture is compute-bound, not memory-bound. The 64-instance
choice trades latency for ensemble diversity. But with the inverted
β formula (Section 4.5), diversity is SYSTEMATICALLY DESTROYED,
making the 64 instances behave as ~1-2 effective trajectories. The
latency cost of 64 instances is paid without the diversity benefit.

6.4 NCCL AllGather Scaling (Multi-GPU)

For multi-GPU deployment (e.g., 8 GPUs × 8 instances = 64 total
or 8 GPUs × 64 instances = 512 total):

Tier 2 NCCL AllGather:
  - Per consensus step: local_Tier1_consensus (merged KV from each
    GPU's instances) shared across GPUs
  - Message size per GPU: 1 copy of merged KV = 2 (K+V) × 1 head ×
    t tokens × 128 dim × 2 bytes = 512 × t bytes
  - At t=30: 15,360 bytes per GPU → 8 GPUs → 120 KB total per step
  - NCCL AllGather on 8 GPUs: negligible latency (< 50 μs on NVLink)

At 512 instances (8 × 64): each GPU merges 64 instances locally
(Tier 1), then shares the merged result globally (Tier 2). The
per-GPU merge is O(64 × t × d), the global merge is O(8 × t × d).
Both are negligible.

Scaling conclusion: COMMUNICATION IS NOT THE BOTTLENECK.
Compute (64 or 128 forward passes per token) IS the bottleneck.
The architecture is compute-bound, not communication-bound.

=======================================================================
PART VII: ADDITIONAL FAILURE MODES & EDGE CASES
=======================================================================

7.1 Confidence Cliff: Collapse to Uniform Weights

If ALL 64 instances have high entropy (H_i → ln(2) = 0.693) at a
particular token, all weights w_i ≈ 0 (because (1 - H/ln(2)) → 0).
The weighted mean becomes a near-uniform average of 64 KV tensors.

This occurs naturally at:
  - Start-of-generation (token 1 after prompt) — attention is
    uncertain about what to focus on
  - Transition between reasoning steps — attention broadens as model
    shifts context
  - Ambiguous input — model is uncertain about interpretation

At these tokens, the consensus mechanism degenerates to "average
all KVs" with no confidence weighting. If any instance has a
catastrophically wrong KV, it is averaged in equally.

7.2 Entropy Measurement Source Ambiguity

The document uses "entropy_i" in the weight formula but NEVER
specifies what distribution entropy is computed over:

  (a) Attention entropy of KV head 0 at L10? Range [0, ln(8)] ≈ [0, 2.08]
      (8 Q heads per KV head). This measures attention sharpness.
  (b) Entropy of the KV value distribution? Meaningless — KV values
      are not probabilities.
  (c) Entropy of the logit distribution? Range [0, ln(32K)] ≈ [0, 10.4].
      This measures prediction uncertainty.
  (d) Entropy of the confidence scores across instances? Range [0, ln(64)].
      This measures swarm agreement.

Each choice has range different from ln(2) used in the formula,
making the denominator wrong for all plausible definitions.

7.3 Triangular Redundancy: Incorrect Recovery

The recovery mechanism: inject consensus KV at β=1.0 for divergent
instances.

Problem: After being excluded from consensus for 3+ tokens, a
divergent instance has:
  - KV head 0 at L10: its OWN (divergent) values
  - KV head 1 at L10: its own values (never blended)
  - All other layers: its own values

After β=1.0 replacement:
  - KV head 0 at L10: CONSENSUS values (abrupt switch)
  - KV head 1 at L10: still its own values
  - All other layers: still its own values

The instance now has a KV cache where:
  - The last 3+ tokens (divergent trajectory) have different keys/values
    for head 0 vs head 1 at L10
  - The consensus KV at L10 head 0 references tokens that the instance
    may have generated differently (if token selection during
    exclusion was different)

This creates a worse inconsistency than the original divergence.
The instance is not repaired — it's converted into a Frankenstein mix
of consensus head 0, self head 1, and self everywhere else.

7.4 Edge Case: Empty Divergence Window

If the divergence detection window (3 tokens) is initialized empty,
the first 3 consensus tokens have no divergence history. The system
can't detect divergence until step 4. If an instance diverges at
step 1, it corrupts the consensus for 3 steps before detection.

For K=30 total consensus steps, 3 steps of undetected divergence =
10% of consensus is corrupted.

7.5 Edge Case: All Instances Diverge

If all 64 instances have high divergence scores (e.g., on a very
ambiguous input), the MAD threshold excludes no one (all are similar
to each other in their divergence), and the triangular redundancy
mechanism has no effect (no single instance is an outlier). The
system merges 64 different KV representations with equal weights
(since all confidences are similar) — degenerating to a uniform
average.

=======================================================================
PART VIII: MASTER REGULATORS (Ranked by Influence × Leverage)
=======================================================================

#1: KV Head Coverage at Consensus Layer (§2 → §3)
  Influence: 0.95 | Leverage: 0.90
  Currently: Only L10, only 1 of 2 KV heads (Q heads 2&3 → KV head 0)
  Fix: Extend to BOTH KV heads at L10
  Impact: Eliminates within-layer incoherence (Flaw #3)
  Effort: 2× bandwidth, 2× compute — both negligible
  Risk: None

#2: Weight Formula Entropy Denominator (§3)
  Influence: 0.95 | Leverage: 0.85
  Currently: ln(2) ≈ 0.693 — causes negative weights
  Fix: Replace with max entropy over appropriate distribution
    (ln(8) if Q-head attention, ln(32K) if logit distribution)
  Impact: Fixes the most critical bug (BLOCKER)
  Effort: One constant change + validation
  Risk: Remaining if entropy source is undefined

#3: β-Blend Direction (§3)
  Influence: 0.90 | Leverage: 0.85
  Currently: β = 0.75 × (1 - conf) — INVERTED
  Fix: β = 0.75 × conf (inverted direction)
  Impact: Prevents majority capture, preserves diversity
  Effort: One formula change
  Risk: Low (intuitive direction)

#4: Voting Mechanism Definition (§5)
  Influence: 0.85 | Leverage: 0.80
  Currently: UNDEFINED — "Dual-Path Token Selection" without vote spec
  Fix: Logit averaging (weighted by confidence)
  Impact: Resolves ambiguity, enables correct analysis
  Effort: ~20 lines of code
  Risk: None (making explicit what is implicit)

#5: Consensus Phase Duration K (§4)
  Influence: 0.75 | Leverage: 0.70
  Currently: Fixed K=30
  Fix: K(N, divergence_rate) = min(30, 5 + 3×log2(N) + 10×divergence)
  Impact: Adaptive consensus length, reduces wasted consensus when
    diversity is low, extends when diversity is high
  Effort: ~30 lines + one tuning experiment

=======================================================================
PART IX: DIVERGENT PULSE — KEY VARIANTS
=======================================================================

V1 [FIX]: Non-Negative Weight Formula
  Mutation: Replace ln(2) with max-entropy bound
  w_i = conf_i × max(0, 1 - H_i / H_max)
  Where H_max = max possible entropy over the distribution
  Quality: Novelty=2, Feasibility=5, Coherence=5, Risk=1
  Score: 4.5 — TRIVIAL FIX, HIGHEST VALUE

V2 [FIX]: Correct β Direction
  Mutation: β_i = β_base × conf_i (NOT 1 - conf_i)
  High confidence → keeps more own KV
  Low confidence → pulled toward consensus
  Quality: Novelty=2, Feasibility=5, Coherence=5, Risk=1
  Score: 4.5 — TRIVIAL FIX

V3 [FIX]: Per-Position L2 MAD for Tensor Merge
  Mutation: For each position p in KV sequence, compute L2 norm
  of [KV_i[p]] across heads and dims → scalar MAD on 64 norms
  → Clamp KV tensors with outlier norms
  Quality: Novelty=3, Feasibility=4, Coherence=5, Risk=2
  Score: 4.0 — CLEAR SPECIFICATION

V4 [ADD]: Multi-Head Consensus (Both KV Heads at L10)
  Mutation: Consensus on both KV heads (0 and 1) at L10
  Eliminates within-layer incoherence
  Quality: Novelty=1, Feasibility=5, Coherence=5, Risk=2
  Score: 4.0 — OBVIOUS EXTENSION

V5 [ADD]: Adaptive Consensus Frequency
  Mutation: Replace fixed K=30 with divergence-triggered adaptive
  sync. Sync when mean pairwise divergence > threshold τ.
  Quality: Novelty=4, Feasibility=4, Coherence=4, Risk=3
  Score: 3.6 — SIGNIFICANT IMPROVEMENT

V6 [MODIFY]: Post-Consensus Logit Generation (Inverted Dual-Path)
  Mutation: Merge KV → compute logits from merged KV → generate
  token. Accepts the "chicken-egg" reality.
  Quality: Novelty=3, Feasibility=5, Coherence=4, Risk=2
  Score: 3.8 — CLEANER DESIGN

V7 [ADD]: Confidence-Weighted Logit Averaging
  Mutation: Vote by averaging logit distributions weighted by conf_i.
  This IS the voting mechanism (currently undefined).
  Quality: Novelty=2, Feasibility=5, Coherence=5, Risk=1
  Score: 4.4 — TRIVIAL, MAKES DESIGN COMPLETE

V8 [MODIFY]: Scalar-Only KV Consensus (Per-Position)
  Mutation: Instead of merging 128-d K/V vectors, merge the scalar
  norms ||K_i[p]|| and ||V_i[p]|| and use the consensus norm to
  rescale individual instances. Avoids tensor MAD entirely.
  Quality: Novelty=5, Feasibility=3, Coherence=3, Risk=4
  Score: 3.0 — INTERESTING BUT RISKY (loses directional info)

=======================================================================
PART X: EMERGENT DISCOVERY (Phase 4b)
=======================================================================

10.1 Unconventional Recombinations

RECOMB-1 (Cross-Level): β-Bifurcation as Implicit Specialization
  Constituents: C4 (β-blend) × E2 (Triangular redundancy)
  Unconventional insight: The β-blend's INVERTED behavior (confident
  instances converge, unconfident ones diverge) is NOT a bug — it
  creates IMPLICIT ROLE SPECIALIZATION. Confident instances become
  the "consensus core" (all identical KV). Unconfident instances
  become "exploration periphery" (diverse KV).
  Predicted: If β is intentionally designed this way (not a bug),
  the system has emergent specialist roles without explicit
  coordination. The confident instances drive accurate token
  selection, while the unconfident instances provide diversity.
  Novelty: 5/5 — reframes a bug as a feature
  EMERGENCE TEST:
    Q1 (Distinct?): Y — specialization is not present in either
      constituent alone
    Q2 (Unpredictable?): N — Given the β formula, an observer could
      predict convergence/divergence bifurcation
    Q3 (New kind?): Y — Role differentiation without coordination
      is a new capability
    Classification: QUANTITATIVE ENHANCEMENT — predictable from
      constituents but produces emergent role structure

RECOMB-2 (Domain-Transposed: Ensemble Kalman Filter → KV Consensus)
  Constituents: C4 (β-blend) × Kalman filter update equations
  Mapping: The KV consensus step is structurally identical to an
    Ensemble Kalman Filter (EnKF) analysis step:
      - Instances = ensemble members
      - KV = state vector
      - Consensus = observed measurement
      - β = Kalman gain
      - w_i = observation weight
  Insight: EnKF theory provides convergence guarantees and optimal
    gain calculation methods. The optimal β is NOT 0.75 — it depends
    on the ratio of ensemble spread to observation error. β should
    be ADAPTIVE: β_optimal = spread² / (spread² + observation_noise²)
  Novelty: 4/5
  Classification: CONFIRMED EMERGENT (Q1=Y, Q2=Y, Q3=Y) — the
    EnKF formalism provides new guarantees and optimality conditions
    not derivable from the original architecture

RECOMB-3 (Forbidden Pair: Consensus × Diversity as Complementary)
  Constituents: E1 (divergence = failure) × I2 (entropy as weight)
  Resolution: The same signal (entropy) is used both as a
    weight-reducing factor (high entropy → low weight) and as a
    divergence signal. This creates a contradiction: high entropy
    instances contribute less to consensus but are more likely to
    be flagged as divergent.
  Novel insight: Entropy should serve ONE role, not both. Split:
    use confidence for weighting, use entropy for divergence detection.
  Novelty: 3/5 — straightforward resolution of conflicting usage

RECOMB-4 (Self-Application): Apply Consensus to Consensus Design
  Constituents: The architecture itself as the subject
  Application: Treat the 7 stated design parameters (L10, K=30,
    β=0.75, MAD=5×, h2&3, 64 instances, 3-token window) as
    64 "design instances" — each is a possible configuration.
    The consensus mechanism itself (confidence-weighted blend)
    tells us which configuration to trust.
  Insight: The architecture has NO mechanism for validating its
    own design parameters. Applying consensus to the design space:
    - High-confidence designs (well-grounded in theory) get more
      weight in the "design consensus"
    - Low-confidence designs (arbitrary choices like β=0.75)
      should be questioned
  Result: The design parameter "L10" is well-grounded (theory-based),
    "β=0.75" is arbitrary (not grounded), "K=30" is weakly grounded
    (sequential execution constraint). Confidence-weighted design
    consensus would flag β_base and K as needing validation.
  Novelty: 4/5

10.2 Synergy Map

| Pair/Triple | Score | Type |
|-------------|-------|------|
| (V1 weight fix, V2 β fix, V3 tensor MAD spec) | 0.95 | Quantitative — all three needed for correct merge |
| (V4 multi-head consensus, V6 inverted dual-path) | 0.88 | Qualitative — fixes both incoherence sources |
| (β-bifurcation as specialization, EnKF formalism) | 0.85 | Qualitative — reframes bug as feature with optimal bounds |
| (V5 adaptive freq, V7 weighted logit avg) | 0.80 | Quantitative — adaptive sync + proper voting |
| (All 7 V-fixes applied together) | 0.75 | Compositional — 7 independent fixes produce correct system |

Self-Organization Detected: YES — the β-bifurcation (inverted formula)
unintentionally creates role specialization. If harnessed intentionally
(adaptive β based on desired diversification), it becomes a tunable
mechanism for exploration-vs-exploitation tradeoff.

=======================================================================
PART XI: CONVERGENT PULSE (Phase 5)
=======================================================================

| Rank | Fix | F1 (Feas) | F2 (Safe) | F3 (Telos) | F4 (Nov) | F5 (Syn) | Score |
|------|-----|:---------:|:---------:|:----------:|:--------:|:--------:|:-----:|
| #1   | V1: Non-negative weight formula | 5 | Safe | 5 | 2 | 5 | 4.75 |
| #2   | V7: Confidence-weighted logit avg | 5 | Safe | 5 | 2 | 5 | 4.50 |
| #3   | V2: Correct β direction | 5 | Safe | 5 | 2 | 4 | 4.25 |
| #4   | V4: Consensus on both KV heads | 5 | Safe | 5 | 1 | 5 | 4.25 |
| #5   | V3: Per-position L2 tensor MAD | 4 | Safe | 5 | 3 | 4 | 4.00 |
| #6   | V6: Inverted Dual-Path (post-consensus logits) | 5 | Safe | 4 | 3 | 4 | 3.75 |
| #7   | V5: Adaptive consensus frequency | 4 | Safe | 4 | 4 | 3 | 3.50 |
| #8   | V8: Scalar-only consensus (norms) | 3 | Medium | 3 | 5 | 2 | 2.75 |

Top-3 Recommendations:

#1: FIX WEIGHT FORMULA + β DIRECTION
  Make weights always non-negative: replace ln(2) with appropriate
  H_max (ln(8) for Q-head attention entropy). Invert β to
  β = β_base × conf_i. This single change fixes the BLOCKER bug
  and the majority-capture dynamic simultaneously.

#2: SPECIFY TENSOR MAD NORM
  Per-position L2 norm over heads × dims, then scalar MAD on 64
  norms. Document as part of the architecture specification.
  Without this, the merge is underspecified and unreproducible.

#3: EXTEND TO BOTH KV HEADS + POST-CONSENSUS LOGITS
  Consensus on all KV heads at L10 prevents within-layer incoherence.
  Generate logits from merged KV (not pre-merge) to eliminate the
  false temporal separation in Dual-Path.

=======================================================================
PART XII: RECURSIVE SELF-ASSESSMENT (Phase 11)
=======================================================================

12.1 Analysis Weaknesses

  W1: No access to actual Qwen2.5-3B attention entropy distributions.
    The claim that "most attention entropies exceed ln(2)" is based on
    published research on Llama/Mistral attention patterns. Qwen may
    differ. Low likelihood of difference (similar architecture), but
    acknowledged.

  W2: The document parsing assumes "heads 2&3 are Q heads" but the
    original could mean KV head indices. If "heads 2&3" means KV
    heads (0,1 → heads 2&3 doesn't exist with only 2 KV heads). The
    conclusion that only 1 of 2 KV heads is covered depends on this
    interpretation. If both KV heads are covered, the "within-layer
    incoherence" issue (5.2.1) is reduced but not eliminated (other
    layers are still unblended).

  W3: The EnKF analogy (RECOMB-2) is novel but untested. The
    optimal β calculation depends on ensemble spread and observation
    noise estimates that the architecture does not compute.

12.2 Blind Spots

  - Power consumption: 64-instance inference on a single GPU draws
    maximum power for extended periods. Thermal throttling may
    reduce throughput and change numerical behavior.
  - Task specificity: The analysis assumes math reasoning (GSM8K).
    KV consensus may behave differently for open-ended generation,
    code synthesis, or multi-turn dialogue.
  - Quantization effects: AWQ 4-bit quantization introduces
    non-linearities that may affect the consensus merge differently
    than FP16 inference.

12.3 Confidence Assessment

| Finding | Confidence | What Would Change It |
|---------|:----------:|---------------------|
| Weight formula bug (negative weights) | 10/10 | Mathematical proof shows it for H > ln(2) |
| MAD ill-specified for tensors | 10/10 | Document provides no norm, no median definition |
| β formula inverted | 9/10 | Unless "blend" operation is defined opposite to convention |
| Dual-Path temporal inconsistency | 8/10 | Would require empirical disproof showing pre/post consensus KV produce same quality |
| Only 1 KV head covered | 7/10 | Ambiguous indexing in document |
| Boundary incoherence (L11-35) | 7/10 | Would require activation analysis to confirm amplification |
| Consensus~free divergence at step 31 | 8/10 | Logical consequence of stopping consensus |
| Overall | 8/10 | Depends on interpretation of 3 ambiguous points (entropy source, head indexing, blend operation) |

=======================================================================
SUMMARY: WHAT TO FIX AND IN WHAT ORDER
=======================================================================

P0 — MUST FIX (BLOCKER):
  □ Fix weight formula: replace ln(2) with appropriate H_max
  □ Fix β direction: β = β_base × conf_i
  □ Specify MAD norm: per-position L2

P1 — SHOULD FIX (CRITICAL):
  □ Specify voting mechanism: confidence-weighted logit averaging
  □ Extend consensus to both KV heads at L10
  □ Clarify Dual-Path: generate logits from merged KV (post-consensus)

P2 — CONSIDER (IMPORTANT):
  □ Adaptive consensus frequency (divergence-triggered)
  □ Fix triangular redundancy recovery: gradual β annealing, not β=1.0
  □ Add confidence calibration measurement to validation

P3 — EXPLORE (RESEARCH):
  □ EnKF-inspired adaptive β_optimal from ensemble spread
  □ Deliberate β-bifurcation for implicit role specialization
  □ Scalar-only KV consensus (norm-based, not tensor-based)

=======================================================================
FILES
=======================================================================
Source:      /home/filip/Projects/Personal/AI/RankAdaptation/strategies/strategy_A_architecture.md
This diffuser: /home/filip/Projects/Personal/AI/RankAdaptation/diffusers/diffuser_A_deep.md
=======================================================================
