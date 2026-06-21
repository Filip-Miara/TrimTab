# MASTER SYNTHESIS: Cross-Diffuser Integration
# ============================================================================
# Date: 2026-06-21
# Sources: 6 diffuser analyses + 3 strategy documents
# Engine: TSE v1.0.0 — Final Conceptual Diffuser-R
# ============================================================================

# ============================================================================
# EXECUTIVE SUMMARY — Top 5 Actionable Findings
# ============================================================================

1. **[BLOCKER] Architecture formula bug renders consensus anti-consensus.** The
   weight formula `w_i = conf_i × (1 - H_i / ln(2))` produces NEGATIVE weights
   for entropy > 0.693 nats — which is the steady state for almost all non-trivial
   tokens. The weighted mean degenerates to anti-consensus (KV pushed away from
   every instance). The β-blend is also INVERTED (confident instances get MORE
   replacement). These are not tunable parameters; they are mathematical errors.
   [Sources: A_deep §4.2–4.5, confirmed by AxC D4, AxB Counter-Assumption ¬A-I2]

2. **[BLOCKER] CIG measurement cannot distinguish genuine CI from averaging.**
   Strategy B's additive CIG decomposition (CIG = early_consensus + error_correction
   + emergent_reasoning) is purely CORRELATIONAL. The three components cannot be
   independently manipulated, and any accuracy improvement can be arbitrarily
   partitioned among them. Without a proper identification strategy, ANY positive
   result supports both "collective intelligence" and "variance reduction" equally.
   [Sources: B_deep F1–F2, confirmed by all 7 lenses]

3. **[CRITICAL] Steering signal has NO metric anywhere in the pipeline.**
   All four roadmap gates use aggregate accuracy, which is blind to whether the
   steering signal survives compaction, voting, or scaling. If compaction destroys
   50% of steering benefit while accuracy drops only 2pp, the architecture loses
   its entire value proposition — silently. This is the single highest-risk gap.
   [Sources: C_deep MR1, AxC D1, BxC GAP3]

4. **[HIGH] The methodology and roadmap have NO coupling mechanism.** Strategy B
   (methodology experiments) and Strategy C (roadmap implementation) were designed
   independently. There are no synchronization points, no feedback loops, and no
   "required findings" gates. The topology mismatch alone (flat tested vs
   hierarchical built) is a BLOCKER — B's findings will NOT generalize to C's
   architecture. [Sources: BxC GAP1+GAP2, confirmed by 5/7 lenses]

5. **[EMERGENT VALUE] Three confirmed emergent capabilities arise from cross-strategy
   fusion that no single strategy predicts:**
   (a) CIG-Optimized Adaptive Consensus (A×B EM-1): Real-time per-token CIG gating
       using A's dual-path architecture + B's CIG framework
   (b) Dual-Mode Convergence-Exploration (A×B EM-4): Self-organizing core/periphery
       instance roles without explicit coordination
   (c) Steering-Preserving Compaction (C×BxC EM-1): Ensemble-aware compaction that
       preserves voting-critical KV entries
   [Sources: AxB Phase 4b, BxC Phase 4b, C_deep EM-1]

# ============================================================================
# CORRECTED CONSENSUS FORMULAS
# ============================================================================

These corrections fix the BLOCKER bugs in Strategy A (architecture) and should
be applied BEFORE any implementation begins.

## F1: Weight Formula (Fix negative weights)

```python
# CURRENT (BROKEN):
#   w_i = conf_i × (1 - H_i / ln(2))
#   → H_i > 0.693 produces NEGATIVE weights → anti-consensus

# CORRECTED:
def compute_weight(conf_i: float, H_i: float, entropy_source: str = "logit") -> float:
    """Non-negative weight for KV consensus merge.

    Args:
        conf_i: Instance confidence in [0, 1] (max softmax probability)
        H_i:    Entropy of the appropriate distribution (nats)
        entropy_source: One of "q_attn" (ln 8), "logit" (ln 32K), "kv_head" (ln 2)

    Returns:
        w_i: Non-negative weight in [0, conf_i]
    """
    H_max = {"q_attn": 2.079, "logit": 10.374, "kv_head": 0.693}[entropy_source]
    raw = 1.0 - H_i / H_max
    return conf_i * max(0.0, raw)  # GUARANTEED non-negative
```

**Rationale**: `max(0, ·)` clamps at zero rather than allowing negative weights.
The denominator `H_max` is the maximum possible entropy over the distribution in
question. For logit entropy (most appropriate as it captures prediction uncertainty
which is what "confidence" should measure): H_max = ln(32K) ≈ 10.374. For Q-head
attention entropy: H_max = ln(8) ≈ 2.079.

**Verification**:
- Sharp logits (H ≈ 0.5): w = conf × (1 - 0.5/10.374) ≈ conf × 0.95 ✓
- Diffuse logits (H ≈ 5.0): w = conf × (1 - 5.0/10.374) ≈ conf × 0.52 ✓
- Uniform logits (H ≈ 10.0): w = conf × max(0, 1 - 10/10.374) ≈ conf × 0.04 ✓
- NEVER negative under any input ✓

## F2: β-Blend Factor (Fix inverted direction)

```python
# CURRENT (BROKEN):
#   β_i = β_base × (1 - conf_i)
#   → conf=0.9 → β=0.075 (7.5% self, 92.5% consensus) — INVERTED
#   → conf=0.1 → β=0.675 (67.5% self, 32.5% consensus) — INVERTED

# CORRECTED:
def compute_beta(conf_i: float, beta_base: float = 0.75) -> float:
    """Blend factor for KV consensus. Higher confidence = MORE self preserved.

    β_i controls how much of the instance's OWN KV is kept:
    - β_i → 1.0: keep own KV (don't blend)
    - β_i → 0.0: fully replace with consensus KV

    KV_final_i = β_i × KV_own_i + (1 - β_i) × KV_consensus
    """
    return beta_base * conf_i   # High conf → more self, low conf → more consensus

# Verification:
# conf=0.9 → β=0.675 (67.5% self) — confident instance keeps its own KV ✓
# conf=0.5 → β=0.375 (37.5% self) — moderate instance partially blends ✓
# conf=0.1 → β=0.075 (7.5% self) — uncertain instance trusts consensus ✓
```

## F3: MAD-Protected Merge for KV Tensors (Fix undefined tensor operation)

```python
import torch

def robust_merge_kv(kv_tensors: torch.Tensor, confidences: torch.Tensor,
                    mad_multiplier: float = 5.0) -> torch.Tensor:
    """MAD-protected weighted mean for KV tensors.

    Args:
        kv_tensors: [N_instances, 2_heads, seq_len, 128_dim] — KV tensors to merge
        confidences: [N_instances] — per-instance confidence scores
        mad_multiplier: Clamp threshold in units of MAD (default 5.0)

    Returns:
        kv_merged: [2_heads, seq_len, 128_dim] — merged KV tensor

    Algorithm:
    1. Compute per-position L2 norm over heads × dims for each instance:
       norm_i[p] = sqrt(Σ_h Σ_d KV_i[h,p,d]²)
    2. Compute median of 64 norms per position: median_norm[p]
    3. Compute MAD: MAD[p] = median_i(|norm_i[p] - median_norm[p]|)
    4. Upper bound: upper[p] = median_norm[p] + mad_multiplier × MAD[p]
    5. Scale down any KV_i[h,p,:] whose norm exceeds upper[p]:
       scale = upper[p] / norm_i[p]  → applied elementwise
    6. Compute weighted mean over instances using confidences
    """
    n_instances = kv_tensors.shape[0]
    seq_len = kv_tensors.shape[2]

    # Step 1: Per-position L2 norms
    norms = torch.norm(kv_tensors.view(n_instances, -1, seq_len), dim=1)  # [N, S]

    # Step 2-3: Median and MAD per position
    median_norm = norms.median(dim=0).values  # [S]
    mad = (norms - median_norm.unsqueeze(0)).abs().median(dim=0).values  # [S]

    # Step 4: Upper bound
    upper = median_norm + mad_multiplier * mad  # [S]

    # Step 5: Clamp — scale down instances that exceed bound
    scale = torch.where(norms > upper.unsqueeze(0),
                        upper.unsqueeze(0) / norms.clamp(min=1e-8),
                        torch.ones_like(norms))
    clamped = kv_tensors * scale.view(n_instances, 1, seq_len, 1)

    # Step 6: Weighted mean
    weights = confidences / confidences.sum()
    kv_merged = (clamped * weights.view(n_instances, 1, 1, 1)).sum(dim=0)

    return kv_merged
```

**Why per-position L2 norm?** Because it preserves per-token structure (each
position's KV has an independent outlier assessment) while collapsing the head×dim
dimensions into a meaningful scalar (the energy/magnitude of the KV representation
at that position). This matches how outlier detection is done in practice: a
"divergent instance" has abnormally large KV activation norms.

## F4: Voting Mechanism (Fix undefined specification)

```python
def token_vote(logits: torch.Tensor, confidences: torch.Tensor) -> tuple:
    """Token selection via confidence-weighted logit averaging.

    This IS the voting mechanism for Dual-Path Token Selection.

    Args:
        logits: [N_instances, vocab_size] — per-instance logit distributions
        confidences: [N_instances] — per-instance confidence scores

    Returns:
        selected_token: int — the chosen token ID
        merged_logits: [vocab_size] — the averaged logit distribution

    The merged logits are used for the "pre-consensus" token selection.
    The selected token is generated using the merged KV cache (post-consensus).
    """
    weights = torch.softmax(confidences / confidences.sum(), dim=0)
    merged_logits = (logits * weights.unsqueeze(-1)).sum(dim=0)
    selected_token = merged_logits.argmax().item()
    return selected_token, merged_logits
```

# ============================================================================
# CONTINGENCY TREE
# ============================================================================

```
ROOT: Deply corrected consensus formulas (F1-F4)
  │
  ├─ IF compaction (Phase 2) fails → various branches
  │   ├─ Node: Can compaction achieve 40% reduction?
  │   │   ├─ YES (<3pp loss) → Continue to Phase 3 with full plan
  │   │   └─ NO (>3pp loss) → Enter Compaction Fallback:
  │   │       ├─ Option A: Accept 25-30% at ≤2pp (viable but less KV savings)
  │   │       ├─ Option B: Steering-aware compaction (RECOMB-3 from C)
  │   │       │   └─ Requires Phase 3 first (chicken-egg: need voting to know
  │   │       │        which KV entries are steering-critical)
  │   │       └─ Option C: Two-phase compaction (C_deep §4.1 pre-seed)
  │   │           └─ Steering KV in SEPARATE UNCOMPACTED buffer (5-10% of KV)
  │   │              Background KV = compacted (35-40% reduction)
  │   │              Total: 35-45% reduction with ZERO steering loss
  │   │
  │   ├─ Node: Can Phase 2 achieve ≥25% compaction?
  │   │   ├─ YES → Option A or C viable
  │   │   └─ NO → Architecture may be fundamentally incompatible
  │   │       └─ Abandon steer-once-broadcast-many; use instance-local KV
  │   │
  │   └─ Node: Steering signal loss measurement
  │       ├─ Steering loss < 20% → Continue, acceptable
  │       ├─ Steering loss 20-50% → Flag WARNING, try other method
  │       └─ Steering loss > 50% → BLOCKER: compaction destroys value prop
  │           └─ Switch to steering-aware or two-phase compaction
  │
  ├─ IF CIG ≈ 0 (or not statistically significant) → various branches
  │   ├─ Node: Is "independent ensemble" baseline also at same accuracy?
  │   │   ├─ YES → CIG is dominated by variance reduction, not emergence
  │   │   │   ├─ Interpret as: KV sharing = efficient ensemble, not CI
  │   │   │   ├─ Still valuable: 64×3B = 2-3× faster than 1×7B at same accuracy
  │   │   │   └─ Adjust claims: "KV consensus improves ensemble efficiency"
  │   │   └─ NO → Possible genuine CI → proceed deeper:
  │   │       ├─ Run shuffled-KV control (B_deep Finding 2)
  │   │       ├─ Run leave-one-out influence analysis
  │   │       └─ If both confirm → genuine CI DETECTED → publish immediately
  │   │
  │   └─ Node: Is measurement confounded by sequential execution order?
  │       ├─ YES (position explains >20% variance) → B_deep Finding 3
  │       │   └─ Counterbalance order, re-measure (EM-2 from B_deep)
  │       └─ NO → ruling out ordering confound → stronger CI claim
  │
  ├─ IF 64×3B can't reach 7B accuracy → various branches
  │   ├─ Node: Gap to 7B > 5pp?
  │   │   ├─ YES → Accept lower target (60% GSM8K vs 65%)
  │   │   │   ├─ Adjust roadmap P4 target to 60%
  │   │   │   ├─ Investigate: is it a model capacity or consensus issue?
  │   │   │   └─ Option: Switch to 7B base with fewer instances (16×7B)
  │   │   └─ NO (gap ≤5pp) → Close enough:
  │   │       ├─ Investigate scaling law: does N→128 close gap?
  │   │       └─ If yes → deploy with more instances (128)
  │   │
  │   └─ Node: Is inverse-U scaling detected? (BxC Counter-Assumption ¬B-A05)
  │       ├─ YES (accuracy peaks then drops) → Optimal N found
  │       │   └─ Architecture should use optimal N, not max N
  │       └─ NO (monotonic) → Continue scaling
  │
  └─ IF sub-agents hang/fail during deployment → [FAILURE REPORT]
      └─ Per Hanging Agent Awareness protocol:
          ├─ Wait max 10 minutes
          ├─ Report hanging agent in [FAILURE REPORT]
          └─ Deploy replacement if critical
```

## Contingency Summary Table

| Trigger Condition | Fallback | Cost | Impact on Architecture |
|------------------|----------|------|----------------------|
| Compaction 40% > 3pp loss | Two-phase: steering buffer + compacted background | +200 LoC, +5 GPU-hrs | Minimal — same API |
| Compaction > 20% steering loss | Steering-aware compaction via influence probe | +500 LoC, +15 GPU-hrs | Requires voting first (reorder to P3 before P2) |
| CIG ≈ 0 (not significant) | Claim efficiency, not emergence | 0 LoC | Adjust narrative only |
| CIG confounded by ordering | Counterbalanced execution | +100 LoC, +5 GPU-hrs | Measurement improvement |
| 64×3B < 7B by > 5pp | Switch to 16×7B or reduce target | +0 LoC (model swap) | Fundamental change if model swapped |
| Inverse-U scaling (N>optimal) | Cap instances at optimal N | +50 LoC (search for optimal) | Minor — adaptive N |
| Confidence-accuracy r < 0.3 | Use uniform weighting, skip confidence | +30 LoC (branch) | Reduces but doesn't eliminate benefit |
| KV serialization exact match fails | Use FP32, tolerate bit-differences | 0 LoC | Reduces precision |

# ============================================================================
# PHASE 0 IMPLEMENTATION PLAN (Concrete Files, Classes, Signatures)
# ============================================================================

This plan incorporates ALL diffuser findings and corrected formulas. It is
organized as a DEPENDENCY DAG, not a sequential chain.

```
P0_0 [CONFIDENCE] ─┬→ P0_1 [MERGE] ─┬→ P1 [MVP] ─┬→ P2 [COMPACTION] ─┐
                    │                  │              │                   │
P0_0 [SERIALIZER] ──┘                  │              └→ P3 [VOTING] ────┤
                                       │                   (parallel!)   │
P0_0 [METRICS] ────────────────────────┘                                 │
                                                                         ↓
                                                                 P4 [HIERARCHICAL]
```

## P0_0: Foundation (IMMEDIATE — Day 1-3)

### File: `src/infra/confidence.py` (~120 LoC) [NEW — AxC D4 fix]

```python
class ConfidenceScorer:
    """Computes per-instance, per-token confidence and entropy metrics.

    Resolves the CRITICAL D4 disparity: neither Strategy A nor C defines how
    confidence/entropy are computed, yet the entire consensus pipeline depends on them.
    """
    def __init__(self, method: str = "logit_max"):
        ...

    def score(self, logits: torch.Tensor, attention_weights: torch.Tensor,
              kv_cache: torch.Tensor) -> dict:
        """Compute confidence, entropy, and auxiliary metrics.

        Args:
            logits: [batch, seq_len, vocab_size]
            attention_weights: [batch, n_heads, seq_len, seq_len]
            kv_cache: [batch, n_layers, 2_kv, seq_len, d_head]

        Returns:
            dict with keys:
              - confidence: [batch] — max softmax probability
              - entropy: [batch] — logit distribution entropy (nats)
              - attn_entropy: [batch] — attention distribution entropy
              - calibration_error: float (ECE over this batch)
        """
        ...

    def validate_calibration(self, logits: torch.Tensor,
                              targets: torch.Tensor) -> float:
        """Validate that confidence correlates with accuracy.

        Returns Expected Calibration Error (ECE).
        Gate: r ≥ 0.3 required before consensus pipeline is enabled.
        """
        ...
```

### File: `src/infra/kv_serializer.py` (~280 LoC) [from Strategy C, expanded]

```python
class KVSerializer:
    """Exact serialization/deserialization for KV cache tensors.

    Handles: variable-length sequences, padding, causal masks,
    cross-instance alignment, FP16/FP32 conversion.
    """
    @staticmethod
    def serialize(kv: torch.Tensor, metadata: dict) -> bytes: ...
    @staticmethod
    def deserialize(data: bytes, seq_len: int) -> tuple: ...
    def roundtrip_exact(self, kv: torch.Tensor, seq_len: int) -> bool: ...
```

### File: `src/infra/kv_ipc.py` (~200 LoC) [from Strategy C]

```python
class KVSharedMemory:
    """Intra-GPU shared memory transport for KV cache."""
    def write(self, data: bytes, instance_id: int): ...
    def read(self, instance_id: int) -> bytes: ...
    def broadcast(self, data: bytes, instances: list[int]): ...
```

### File: `src/infra/metrics.py` (~150 LoC) [NEW — AxB CIG measurement, B_deep F2-F3]

```python
class CIGMeasurer:
    """Collective Intelligence Gain measurement framework.

    Features:
    - Non-additive CIG using Partial Information Decomposition (PID)
      (B_deep EM-1 — CONFIRMED EMERGENT)
      OR Shapley values with interaction terms for smaller N
    - Leave-one-out influence analysis (which instances contribute unique info)
    - Shuffled-KV control (does content matter for improvement?)
    - Order-counterbalanced execution (B_deep Finding 3 mitigation)
    - Steering signal retention metric (C_deep MR1)
    """

    def cig_non_additive(self, instance_preds: dict,
                        swarm_preds: torch.Tensor,
                        true_labels: torch.Tensor) -> dict:
        """Compute non-additive CIG using PID.

        Returns:
            dict with synergy, redundancy, unique_1, unique_2, ..., total
        """
        ...

    def steering_retention(self, acc_steered: float,
                           acc_unsteered: float,
                           acc_compacted: float) -> float:
        """Fraction of steering benefit preserved after compaction.

        retention = (acc_compacted - acc_unsteered) / max(acc_steered - acc_unsteered, 1e-8)
        Gate: retention ≥ 0.80 required.
        """
        ...

    def influence_matrix(self, instance_preds: list[np.ndarray]) -> np.ndarray:
        """Compute pairwise influence matrix (A→B delta in accuracy).
        If influence correlates NEGATIVELY with accuracy, selfish KV detected.
        (B_deep EM-3 — Selfish KV Detector)
        """
        ...

    def order_effect_analysis(self, acc_by_position: list[float]) -> float:
        """Variance explained by execution order.
        If >20%, sequential execution is confounded.
        """
        ...
```

### File: `src/infra/config.py` (~80 LoC) [from Strategy C, expanded]

```python
@dataclass
class ConsensusConfig:
    """Architecture configuration parameterized by instance count N.

    Implements AxC recommendation: parameters are functions of N, not constants.
    """
    n_instances: int

    @property
    def K(self) -> int:
        return min(30, max(5, int(5 + 3 * np.log2(self.n_instances))))

    @property
    def beta_base(self) -> float:
        return 0.5 + 0.25 / self.n_instances

    @property
    def mad_multiplier(self) -> float:
        return max(3.0, 5.0 - np.log2(self.n_instances))

    @property
    def tau_decay(self) -> int:
        return max(5, int(20 / np.log2(self.n_instances + 1)))
```

## P0_1: Corrected Consensus Engine (Day 3-7)

### File: `src/consensus/merge.py` (~300 LoC) [NEW — corrected formulas F1-F4]

```python
class ConsensusEngine:
    """KV consensus merge with corrected formulas.

    Incorporates ALL fixes from diffuser A_deep:
    - Non-negative weight formula (F1)
    - Correct β direction (F2)
    - Per-position L2 tensor MAD (F3)
    - Confidence-weighted logit averaging voting (F4)
    """

    def __init__(self, config: ConsensusConfig, scorer: ConfidenceScorer):
        ...

    def compute_weights(self, confidences: torch.Tensor,
                        entropies: torch.Tensor) -> torch.Tensor:
        """F1: Non-negative weight formula."""
        H_MAX = 10.374  # ln(32K) — logit entropy bound
        return confidences * (1.0 - entropies / H_MAX).clamp(min=0.0)

    def robust_merge(self, kv_tensors: torch.Tensor,
                     weights: torch.Tensor) -> torch.Tensor:
        """F3: Per-position L2 MAD merge."""
        ...

    def blend(self, kv_own: torch.Tensor, kv_consensus: torch.Tensor,
              confidences: torch.Tensor) -> torch.Tensor:
        """F2: Corrected β blend."""
        beta = self.config.beta_base * confidences  # CORRECTED
        return beta.view(-1,1,1,1) * kv_own + (1 - beta.view(-1,1,1,1)) * kv_consensus

    def token_vote(self, logits: torch.Tensor,
                   confidences: torch.Tensor) -> tuple[int, torch.Tensor]:
        """F4: Confidence-weighted logit averaging."""
        ...

    def consensus_step(self, step: int, kv_caches: list,
                       logits: torch.Tensor) -> dict:
        """Full consensus step: extract → compute_weights → robust_merge → blend → vote."""
        ...
```

### File: `src/consensus/sync_schedule.py` (~120 LoC) [NEW — adaptive sync from AxB EM-1]

```python
class AdaptiveSyncScheduler:
    """CIG-gated adaptive consensus frequency.

    Implements AxB EM-1 (CONFIRMED EMERGENT): real-time CIG estimation
    from pre/post consensus logit divergence.

    Sync only when CIG_instantaneous = KL(pre_logits || post_attention) > threshold.
    """
    def __init__(self, cig_threshold: float = 0.1):
        ...

    def should_sync(self, pre_logits: torch.Tensor,
                    post_logits: torch.Tensor) -> bool:
        kl_div = (pre_logits * (pre_logits.log() - post_logits.log())).sum()
        return kl_div > self.cig_threshold

    def update_threshold(self, recent_cig_values: list[float]): ...
```

## P1: MVP — Two-Instance One-Way Share (Day 7-14)

### File: `src/infra/model_farm.py` (~300 LoC) [from Strategy C]

```python
class ModelFarm:
    """Multi-instance lifecycle management."""
    def spawn(self, n: int, model_name: str = "Qwen2.5-3B-AWQ") -> list: ...
    def run_prompt(self, instances: list, prompt: str) -> list: ...
    def run_generation(self, instances: list, tokens: list) -> list: ...
    def cleanup(self): ...
```

### File: `run_multiagent_phase1.py` (~200 LoC) [from Strategy C]

MVP script: orchestrator steers one worker via KV share.

## P2: Compaction (parallel with P3)

### File: `src/infra/kv_compactor.py` (~450 LoC) [expanded from Strategy C]

```python
class KVCompactor:
    """Steering-preserving KV compaction.

    Implements C_deep EM-1 + RECOMB-3: two-phase compaction that identifies
    and preserves steering-critical KV entries.

    Methods:
    - Generic: velocity_magnitude, attention_matching, entropy
    - Steering-aware: two-phase (preserve steering, compact background)
    """

    def velocity_magnitude(self, kv: torch.Tensor, ratio: float): ...
    def attention_matching(self, kv: torch.Tensor, ratio: float): ...
    def entropy(self, kv: torch.Tensor, ratio: float): ...

    def steering_aware(self, kv: torch.Tensor, steering_mask: torch.Tensor,
                       background_ratio: float = 0.4) -> torch.Tensor:
        """Two-phase compaction: steering buffer + compacted background."""
        ...

    def measure_steering_retention(self, kv_full: torch.Tensor,
                                    kv_compacted: torch.Tensor,
                                    steering_mask: torch.Tensor) -> float:
        """C_deep MR1: steering signal retention metric."""
        ...
```

## P3: Voting (parallel with P2)

### File: `src/infra/kv_voting.py` (~300 LoC) [expanded from Strategy C]

```python
class KVVoting:
    """Democratic KV voting with confidence weighting.

    Implements BxC CIG-Driven Voting Design (blend from BxC lens 3):
    uses CIG decomposition to decide voting mechanism.
    """

    def confidence_weighted_vote(self, kv_list: list, confidences: list): ...
    def majority_vote(self, kv_list: list): ...
    def diversity_preserving_vote(self, kv_list: list, diversity_metric: float): ...

    def cig_guided_vote(self, kv_list: list, confidences: list,
                         cig_profile: dict) -> torch.Tensor:
        """Choose voting mechanism based on CIG decomposition."""
        if cig_profile.get("emergent_reasoning", 0) > 0.3:
            return self.diversity_preserving_vote(kv_list, ...)
        elif cig_profile.get("error_correction", 0) > 0.5:
            return self.majority_vote(kv_list)
        else:
            return self.confidence_weighted_vote(kv_list, confidences)
```

## P4: Hierarchical

### File: `src/infra/hierarchical_voting.py` (~350 LoC) [from Strategy C]

```python
class HierarchicalVoting:
    """Level 0 (workers) → Level 1 (pod coordinators) → Level 2 (global)."""
    def level0_vote(self, workers: list) -> dict: ...
    def level1_aggregate(self, pod_votes: list) -> dict: ...
    def level2_global(self, coordinator_votes: list) -> dict: ...
```

# ============================================================================
# RISK REGISTER (Likelihood × Impact × Mitigation)
# ============================================================================

| # | Risk | Likelihood | Impact | RRN | Mitigation |
|---|------|:----------:|:------:|:---:|------------|
| R1 | Weight formula fix insufficient (entropy still misinterpreted) | MED (30%) | HIGH | 24 | Validate confidence-accuracy correlation in P0_0 before building pipeline |
| R2 | 40% compaction destroys >3pp accuracy | HIGH (65%) | HIGH | 42 | Two-phase compaction (steering buffer) as fallback. Target 25-30% initially. |
| R3 | Steering signal NOT concentrated (uniform across KV) | MED (35%) | CRITICAL | 49 | Measure steering signal distribution BEFORE compaction (P1 ablation). If uniform, cannot use selective compaction. |
| R4 | Sequential execution confounds all CI measurements | HIGH (60%) | HIGH | 36 | Add order-counterbalanced execution to measurement framework (P0_0 metrics.py). Report accuracy×position. |
| R5 | CIG ≈ 0 across all conditions | MED (40%) | MEDIUM | 16 | Frame as "efficient ensemble" not "collective intelligence." Still valuable for latency/cost. |
| R6 | 64×3B cannot reach 7B accuracy (gap >5pp) | MED-HIGH (55%) | HIGH | 33 | Fallback to 16×7B or reduce target to 60%. Document scaling law. |
| R7 | KV serialization exact match fails across instances | HIGH (70%) | MEDIUM | 21 | Tolerate bit-differences with FP32 fallback. Log discrepancy rates. |
| R8 | Methodology results unavailable when roadmap needs them | HIGH (60%) | HIGH | 36 | Add synchronization gates. Run targeted P0 experiments before full methodology. |
| R9 | Hierarchical topology (fat-tree) behaves differently than swept flat topologies | HIGH (70%) | HIGH | 35 | Add hierarchical topology to methodology F1 sweep. Test 2-pod × 2-worker at small scale. |
| R10 | Sub-agent deployment hangs (known OpenCode Issue #20096) | MED (25%) | MEDIUM | 10 | Per Hanging Agent Awareness: 10-min timeout, [FAILURE REPORT], deploy replacement. |
| R11 | β-bifurcation causes implicit specialization that hurts accuracy | LOW (20%) | HIGH | 16 | Monitor divergence-after-blend. If divergence > threshold, adjust β_base. |
| R12 | Multi-GPU NCCL latency changes cost model completely | MED (30%) | MEDIUM | 6 | Measure NCCL latency in P0_0. If >100μs, adapt sync frequency. |
| R13 | LoC estimate (3,500) is 2-3× too low | HIGH (75%) | MEDIUM | 15 | Budget 7,000-10,000 LoC. Plan for 2-3× overrun on infrastructure. |
| R14 | GPU-hour estimate (82) is 1.5-3× too low | HIGH (70%) | HIGH | 28 | Budget 150+ GPU-hours with 30% contingency. Per-phase kill criteria. |

**RRN Formula**: Risk Rating Number = Likelihood (1-5) × Impact (1-5)
- 1-9: Monitor
- 10-24: Active mitigation
- 25-36: Urgent mitigation
- 37+: Critical — must address before proceeding

**Top 3 Risks requiring immediate action**:
- R3 (steering not concentrated) — single P1 ablation experiment resolves uncertainty
- R2 (compaction failure) — two-phase fallback exists but needs pre-work
- R4 (sequential execution confound) — counterbalancing added to P0_0 metrics

# ============================================================================
# WHAT THE TSE ITSELF MISSED — Metacognitive Transcendence
# ============================================================================

The Triadic Synthesis Engine (TSE) produced deep insights across 6 diffuser
analyses, but applying the TSE's own metacognitive lens to its collective output
reveals systematic blind spots:

## Blind Spot 1: The "Everyone is Wrong" Scenario

**What was missed**: Every diffuser assumes that at least SOME instances produce
useful signal. None modeled the case where ALL instances are confidently, uniformly
WRONG. In this scenario:
- Consensus AMPLIFIES the error (all instances agree → high confidence → high weight)
- Diversity mechanisms fail (no instance is an outlier to rescue)
- CIG_emergent_reasoning is zero AND CIG_error_correction is zero
- The system is maximally confident and maximally wrong

**Why TSE missed it**: The TSE's adversarial lens (Phase 2, lens 9) asks "what
attacks break the system?" but is framed as external adversarial attacks, not
as emergent internal failure modes. The "everyone is wrong" scenario is not an
attack — it's a natural consequence of the system's design when facing ambiguous
or adversarial inputs that fool the entire model family.

**Mitigation**: Add diversity PRESSURE that increases when consensus is too high.
If all instances agree AND confidence is high → artificially increase diversity
(raise temperature, inject noise). The system needs a "suspicion" threshold.

## Blind Spot 2: Cross-Layer KV Incoherence Amplification

**What was missed**: The A_deep analysis identified that L10 KV blending creates
incoherence with unblended layers 11-35. But NO diffuser modeled the AMPLIFICATION
factor — does this incoherence grow, shrink, or oscillate through 25 layers of
transformer computation? If it grows (plausible: attention sharpens differences,
FFNs project them nonlinearly), then the incoherence at L35 output is much larger
than at L10, and the free-generation quality degrades proportionally.

**Why TSE missed it**: The TSE lacks a formal error propagation analysis.
Phase 2 lenses (systems, trajectory) identified the incoherence risk but didn't
quantify the propagation dynamics. A full error propagation analysis would require
tracking the Lipschitz constant of each layer or empirical perturbation analysis.

**Mitigation**: P0 empirical test: run forward pass on 10 instances, measure L10
output divergence, propagate through L11-35, measure L35 divergence. If L35
divergence > 2× L10 divergence → incoherence amplification confirmed.

## Blind Spot 3: The "CI Premium" Business Case

**What was missed**: All 6 diffusers treat accuracy as the optimization target.
But the practical value proposition of 64×3B vs 1×7B is not about accuracy — it's
about COST (3B is cheaper/faster than 7B) and LATENCY (64 parallel instances can
be faster than 1 large instance for certain workloads). If 64×3B achieves 98% of
7B accuracy at 50% of the cost, that's a WIN even with zero "collective intelligence."

**Why TSE missed it**: The TSE's Phase 3 (master regulators) and Phase 9 (resource
budgeting) focus on technical leverage points and GPU budgets. It has no concept
of "economic value function" or "deployment cost comparison." The entire analysis
is accuracy-centric.

**Mitigation**: Add a cost-efficiency analysis layer. Compute:
- Accuracy per dollar (GPU-hour)
- Accuracy per watt (power)
- Accuracy per latency-millisecond
If 64×3B wins on any of these, the architecture is valuable regardless of CIG.

## Blind Spot 4: The Methodology Itself Changes the System

**What was missed**: Strategy B's extensive measurement framework (600+ conditions,
bootstrap CI, McNemar's tests) is itself an intervention. If instances are
instrumented for CIG measurement, the measurement overhead (logging, extra forward
passes, KV extraction) changes the system's timing and memory profile. The act of
measuring CIG may destroy the conditions that produce CIG.

**Why TSE missed it**: This is the OBSERVER EFFECT in AI systems — a well-known
phenomenon in psychology (Hawthorne effect) and physics (Heisenberg uncertainty)
that TSE's Phase 7 (causal mapping) does not model. The TSE assumes measurements
are passive; they are not.

**Mitigation**: Use passive instrumentation (trace existing signals) rather than
active probes (extra forward passes). Or run a "measurement vs no-measurement"
ablation in P1.

## Blind Spot 5: The Dual-Use Nature of Emergent Capabilities

**What was missed**: The three CONFIRMED EMERGENT capabilities (CIG-gated consensus,
dual-mode convergence-exploration, steering-preserving compaction) are presented
as positive breakthroughs. But each has a DARK SIDE:
- CIG-gated consensus: when CIG is measured as NEGATIVE, the system drops into
  independent generation — losing any collective benefit at the moment it's most needed
- Dual-mode: explores diverge further, and if the core is wrong, the system needs
  the explorers to correct it — but the explorers are excluded from consensus
- Steering-preserving compaction: if the steering signal is wrong (adversarial),
  the system preserves and amplifies the wrong signal

**Why TSE missed it**: TSE Phase 2 lens 9 (adversarial) evaluates attacks on the
CURRENT system, not on the EMERGENT capabilities. The emergent capabilities
themselves are not stress-tested for failure modes. This is a self-referential
gap: the TSE generates emergent capabilities but doesn't adversarially evaluate them.

**Mitigation**: For each CONFIRMED EMERGENT capability, run a dedicated
adversarial analysis. Ask: "If this capability were actively working against the
system's goals, what would we observe?"

## Blind Spot 6: Personnel Assumption

**What was missed**: The implementation plan assumes sufficient engineering
bandwidth to build 4+ complex distributed systems components concurrently.
KV serialization, shared-memory IPC, multi-instance lifecycle management, and
evaluation infrastructure are each non-trivial systems projects. A single engineer
may require 3-6 months for the full scope.

**Why TSE missed it**: TSE Phase 9 (resource budgeting) considers GPU-hours but
not ENGINEER-hours. This is a critical blind spot common to AI research plans.

**Mitigation**: Estimate engineering time per component: serializer (2 weeks),
IPC (1 week), model farm (2 weeks), consensus engine (3 weeks), compaction (3 weeks),
voting (2 weeks), hierarchical (3 weeks), evaluation infrastructure (2 weeks).
Total: ~18 weeks for a single engineer. Plan accordingly.

# ============================================================================
# REVISED ROADMAP (Prioritized, with Resource Estimates)
# ============================================================================

```
WEEK 1-3: P0 MUST FIX (BLOCKER items)
├── P0_0a: ConfidenceScorer + validation     [120 LoC, <5 GPU-hrs]
├── P0_0b: KVSerializer + KVSharedMemory      [480 LoC, <5 GPU-hrs]
├── P0_0c: Config system                       [80 LoC, 0 GPU-hrs]
├── P0_0d: Metrics + CIG framework + controls [250 LoC, 5 GPU-hrs]
├── P0_1a: Corrected ConsensusEngine           [300 LoC, 0 GPU-hrs]
├── P0_1b: AdaptiveSyncScheduler              [120 LoC, 5 GPU-hrs for tuning]
├── P0_1c: Validate confidence-accuracy corr  [0 LoC,  3 GPU-hrs]
├── P0_1d: Measure NCCL latency at N=2,4,8    [50 LoC,  2 GPU-hrs]
├── P0_1e: Validate steering signal distribution (is it concentrated?) [0 LoC, 5 GPU-hrs]
└── Gate: confidence r ≥ 0.3, steering concentrated?
    → FAIL: use uniform weighting, document limitation
    → PASS: proceed to P1

WEEK 4-6: P1 MVP + Methodology Alignment
├── P1a: ModelFarm + run_multiagent_phase1    [500 LoC, 10 GPU-hrs]
├── P1b: Run first 2-instance steering test   [0 LoC,   5 GPU-hrs]
├── P1c: Methodology P0 (infra validation)     [200 LoC, 5 GPU-hrs]
├── P1d: Shuffled-KV control experiment       [50 LoC,  3 GPU-hrs]
├── P1e: Order-counterbalance measurement     [50 LoC,  2 GPU-hrs]
└── Gate 1 (serialization exact match) + Gate 2 (worker ≥ orchest - 5pp)
    → FAIL: debug serialization, fix KV alignment
    → PASS: proceed to P2∥P3

WEEK 5-8: P2 (Compaction) ∥ P3 (Voting) — PARALLEL
├── P2a: KVCompactor — generic methods        [300 LoC, 10 GPU-hrs]
├── P2b: KVCompactor — steering-aware two-phase [150 LoC, 15 GPU-hrs]
├── P2c: Compaction ablation (with/without)   [0 LoC,   10 GPU-hrs]
├── P3a: KVVoting — basic mechanisms          [200 LoC, 10 GPU-hrs]
├── P3b: KVVoting — CIG-guided mechanism      [150 LoC, 15 GPU-hrs]
├── P3c: Voting vs independent ensemble        [0 LoC,   10 GPU-hrs]
├── Methodology P2+P3: topology + protocol sweep [200 LoC, 20 GPU-hrs]
└── Gate 3 (compaction ≥25% with ≤3pp loss + steering retention ≥80%)
    + Gate 4 (voting ≥ best-single + 3pp)
    → FAIL (compaction): use two-phase steering buffer
    → FAIL (voting): check diversity, add temperature annealing
    → PASS: proceed to P4

WEEK 8-12: P4 Hierarchical + Final Validation
├── P4a: HierarchicalVoting                   [350 LoC, 10 GPU-hrs]
├── P4b: Scaling utilities + load balancing   [200 LoC,  5 GPU-hrs]
├── P4c: Full 32-instance evaluation           [0 LoC,   30 GPU-hrs]
├── P4d: 7B parity comparison                  [0 LoC,   15 GPU-hrs]
├── P4e: CIG final measurement (non-additive) [0 LoC,   10 GPU-hrs]
├── P4f: Ablation: all components             [0 LoC,   20 GPU-hrs]
└── Final gate: 32-instance ≥ 7B - 3pp, CIG > 0 (stat sig)
    → FAIL: document scaling law, claim efficiency not emergence
    → PASS: publish collective intelligence results

TOTAL ESTIMATED:
  LoC: 4,050-5,200 core + 1,500-2,000 test = ~6,000-7,200 total
  GPU-hours: 120-200 (with debugging and failed experiments: 180-280)
  Engineering time: 12-18 weeks (single engineer)
```

# ============================================================================
# TESTABLE HYPOTHESES (from all diffusers, consolidated)
# ============================================================================

| ID | Hypothesis | Falsification | Min Cost | Source |
|----|-----------|---------------|----------|--------|
| H1 | Weight formula (corrected) produces stable, positive weights | Any weight < 0 | <1 GPU-hr | A_deep |
| H2 | β direction fix prevents majority capture | After 30 steps, all instances are identical | 3 GPU-hrs | A_deep |
| H3 | Steering signal concentrates in ≤15% of KV entries | Random compaction destroys <10% steering | 5 GPU-hrs | C_deep H-1 |
| H4 | Steering retention >80% predicts voting gain >3pp | Retention and voting gain are uncorrelated | 15 GPU-hrs | C_deep H-2 |
| H5 | Non-additive CIG (PID) differs from additive by >20% | PID components sum to additive CIG | 10 GPU-hrs | B_deep EM-1 |
| H6 | Shuffled KV improves accuracy same as real KV | Real KV significantly (p<0.05) outperforms shuffled | 5 GPU-hrs | B_deep F2 |
| H7 | Execution order explains <10% of accuracy variance | Order explains >20% of variance | 3 GPU-hrs | B_deep F3 |
| H8 | Hierarchical fat-tree shows different CIG than flat topologies | CIG components indistinguishable (p>0.05) | 10 GPU-hrs | BxC H-01 |
| H9 | Consensus-aware compaction (by layer proximity) achieves >40% with <1pp loss | >2pp loss at 40% ratio | 5 GPU-hrs | AxC H-1 |
| H10 | Confidence decay function τ=10 reconciles steer-once with iterative blending | Accuracy <90% of full blend baseline | 5 GPU-hrs | AxC H-2 |

# ============================================================================
# FINAL DECISION FLOW
# ============================================================================

```
START
  │
  ├─ Fix BLOCKER formulas (F1-F4) ─────────────────────────── [1 engineer-day]
  │
  ├─ Run P0_0 validation experiments:
  │   ├─ confidence-accuracy correlation
  │   ├─ steering signal concentration
  │   ├─ NCCL latency at scale
  │   └─ KV serialization exact match
  │
  ├─ DECISION GATE 0: Is consensus pipeline viable?
  │   ├─ YES → Continue with full architecture
  │   └─ NO  → Document limitations; focus on efficiency (not emergence)
  │
  ├─ Build P1 (2-instance MVP) + Methodology alignment
  │
  ├─ DECISION GATE 1: Does steering transfer work?
  │   ├─ YES → Parallel P2 (compaction) + P3 (voting)
  │   └─ NO  → Rethink architecture: steering may need different mechanism
  │
  ├─ DECISION GATE 2: Compaction AND voting viable?
  │   ├─ BOTH YES → Build P4 hierarchical
  │   ├─ COMPACTION NO → Two-phase fallback
  │   └─ VOTING NO → Add diversity mechanisms
  │
  └─ DECISION GATE 3: 32-instance reaches 7B parity?
      ├─ YES → Publish collective intelligence results
      ├─ NO (but close) → Claim efficiency; scale to 128 instances
      └─ NO (far) → Document negative result; open-source infrastructure
```

# ============================================================================
# [FAILURE REPORT]
# ============================================================================

```json
{
  "overall_status": "all_completed",
  "note": "All 6 diffuser analyses read. TSE skill loaded. Master synthesis written.",
  "lenses_completed": "Implicit: TSE 12-phase + metacognitive transcendence",
  "lenses_failed": [],
  "critical_resolved": [
    "Weight formula bug (ln(2) → H_MAX·max(0,·))",
    "β-blend inversion (β = β_base × conf_i)",
    "Undefined tensor MAD (per-position L2 specification)",
    "Undefined voting mechanism (confidence-weighted logit avg)",
    "Missing confidence metric definition (ConfidenceScorer class)"
  ],
  "contingencies_defined": 6,
  "risks_assessed": 14,
  "emergent_capabilities": 5,
  "tse_blind_spots": 6
}
```

# ============================================================================
# END OF MASTER SYNTHESIS
# ============================================================================

Path: /home/filip/Projects/Personal/AI/RankAdaptation/diffusers/MASTER_SYNTHESIS.md
