# Pyramid Ascent: Bottom-Up Propagation + Cross-Idea Analysis

---

## Level 1 → Level 2 Propagation (Atoms → Composites)

### C2_Q1: Qwen3.5 Hybrid Attention

| Property | Value |
|----------|-------|
| **Inherited from A_Q1** | State saturation, gate collapse, delta correction lag |
| **Inherited from A_Q2** | O(n²) cost on 25% of layers, GQA limitations |
| **Inherited from J_Q1** | 75/25 ratio is architectural guess — if wrong, entire attention quality degrades |
| **Emergent Weaknesses** | The transition between linear and softmax layers creates a representation discontinuity — Gated DeltaNet outputs feed into Gated Attention inputs. If their representation spaces are misaligned, the 25% softmax layers cannot effectively correct errors. |
| **Propagated Synthetics** | SYN-4 (replace 25% Gated Attention with MLA): C2_Q1 becomes 75% Gated DeltaNet + 25% MLA — all sub-quadratic. Effect: C2_Q1's compute complexity drops from O(0.75n + 0.25n²) to O(n). Risk: retrieval quality may drop. |

### C2_D1: DeepSeek V4 Hybrid Attention Stack

| Property | Value |
|----------|-------|
| **Inherited** | Indexer failure, irreversible compression loss, position ambiguity, fixed top-k, three-tier complexity |
| **Emergent Weaknesses** | **Tier incoherence**: SWA (local, n=128), CSA (sparse mid-range), HCA (global, 128:1). If the three tiers produce conflicting attention distributions, the model must resolve them via learned weights — potentially unstable. **No single authoritative attention signal** — the model works with three potentially contradictory views. |
| **Propagated Synthetics** | SYN-2 (adaptive top-k): Improves C2_D1's quality by allowing the indexer to select more entries for hard queries. Effect: reduces the "missed entry" failure mode. Cost: variable compute budget. |

### C2_K1: Kimi K2.6 Latent Attention System

| Property | Value |
|----------|-------|
| **Inherited** | Fixed rank limitation, head interference, borrowed architecture dependency |
| **Emergent Weaknesses** | **Dual projection mismatch**: A_K1 (MLA) compresses KV to rank 512, but A_K2 (MHA) requires 64 heads × KV dimension. The mismatch between compressed storage and expanded usage may lose information in the up-projection. **Train-inference gap**: MLA is trained with full precision but used with INT4 — the compression may amplify quantization errors. |
| **Propagated Synthetics** | SYN-3 (dynamic rank MLA): ameliorates head interference by allocating ranks per head. Effect: high-precision heads get more rank, reducing competition. |

---

## Level 2 → Level 3 Propagation (Composites → Higher Composites)

### C3_Q1: Qwen3.5 Efficient Core (Attention + MoE)

| Property | Value |
|----------|-------|
| **Inherited from C2_Q1** | All hybrid attention weaknesses |
| **Inherited from C2_Q2** | Expert death, router bottleneck, auxiliary loss interference |
| **Inherited from J_Q6** | Attention→MoE sequential dependency: if attention produces poor representations, MoE cannot compensate |
| **Inherited from J_Q7** | Linear attention → enables 512-expert MoE: if linear attention degrades under long context, the MoE loses input quality |
| **Emergent Weaknesses** | **Input starvation**: MoE quality depends entirely on attention output quality. If Gated DeltaNet saturates at long context, the MoE receives degraded input and cannot recover. **Compounding sparsity**: 75% linear attention (compressed state) + 2% MoE activation = only 1.5% of theoretical capacity is effectively used per token. Is 1.5% sufficient for frontier performance? |
| **Propagated Synthetics** | SYN-6 (anticipatory routing): replaces aux-loss routing. Effect on C3_Q1: removes the auxiliary loss vs. quality tension. Risk: anticipatory routing at 512 experts is untested. |

### C3_D1: DeepSeek V4 Core (Attention + MoE)

| Property | Value |
|----------|-------|
| **Inherited from C2_D1** | Three-tier incoherence, indexer failure |
| **Inherited from C2_D2** | Aggressive sparsity risk, hash routing rigidity |
| **Inherited from J_D9** | CSA's KV reduction enables 384-expert MoE — if compression degrades, MoE budget is affected |
| **Emergent Weaknesses** | **Routing on compressed representations**: The MoE router receives inputs from compressed attention — if the compression discards features the router needs, routing quality drops. **Sparsity compounding**: CSA (top-k=1024 out of 250K compressed) + MoE (Top-6 out of 384 experts) = extremely selective information flow. Are two levels of sparsity better than one? |
| **Propagated Synthetics** | SYN-5 (dense tuning islands): adds dense layers early in the model to provide deterministic feature extraction. Effect: reduces routing variance from compressed inputs. |

### C3_K1: Kimi K2.6 Core (Attention + MoE)

| Property | Value |
|----------|-------|
| **Inherited from C2_K1** | Fixed rank, head interference, dual projection mismatch |
| **Inherited from C2_K2** | Expert imbalance (no auxiliary loss), shared expert over-compensation |
| **Inherited from J_K8** | MLA savings → 8 experts budget: fixed at design time |
| **Emergent Weaknesses** | **No balance mechanism**: Kimi K2.6 has the most "hands-off" approach to MoE load balancing (no auxiliary loss) combined with a borrowed attention mechanism (MLA from DeepSeek V3). These two design choices may interact poorly — MLA's shared latent space may create correlated expert utilization patterns that the noaux_tc router cannot disentangle. |

---

## Level 3 → Peak (Composite → Peak P)

### P: Contemporary LLM Architecture Design Space

| Property | Value |
|----------|-------|
| **Inherited from C3_Q1** | Qwen3.5 core: linear+softmax hybrid + 512 MoE + auxiliary loss |
| **Inherited from C3_D1** | DeepSeek V4 core: CSA+HCA + 384 MoE + mHC + FP4 |
| **Inherited from C3_K1** | Kimi K2.6 core: MLA + 384 MoE + noaux_tc + Agent Swarm |
| **Emergent Weaknesses (peak level)** | |

### Emergent Weaknesses at Peak

| # | Weakness | Source | Severity |
|---|----------|--------|----------|
| PW1 | **No consensus on attention replacement** — three architectures, three strategies. The field lacks a clear winner or unified theory. | Emergent from C3_Q1 vs. C3_D1 vs. C3_K1 | critical |
| PW2 | **Complexity compounding** — each architecture adds mechanisms. There is no trend toward simplification. | Emergent (all architectures) | critical |
| PW3 | **Benchmark convergence despite architectural divergence** — Qwen, DeepSeek, and Kimi are within <5% on most benchmarks despite radically different internals. Suggests diminishing returns from architectural innovation. | Emergent | major |
| PW4 | **MoE appears converged** — all three use ~30:1 total:active ratio, shared experts, fine-grained routing. The community agrees on MoE. | Common pattern across nodes | informative |
| PW5 | **All fail on one dimension** — Qwen trails coding, DeepSeek trails multimodal/knowledge, Kimi trails reasoning. No model dominates across all axes. | Cross-composite propagation | major |
| PW6 | **Training cost vs. inference cost tradeoff is unsolved** — all three architectures optimize for inference but require enormous training compute. | Peak-level economic observation | major |

### Propagated Synthetics at Peak

| SYN | Applied To | Effect on Peak |
|-----|-----------|----------------|
| SYN-4 | C3_Q1 (replace softmax with MLA) | Makes Qwen fully sub-quadratic — tests whether softmax is necessary |
| SYN-8 | P (merge all three attention mechanisms) | Maximum quality at maximum complexity |
| SYN-12 | C3_D1 (add vision to DeepSeek V4) | Rounds DeepSeek's weakness — potentially dominates all benchmarks |
| SYN-11 | C3_K1 + C3_Q1 (Kimi swarm + Qwen MCP) | Best agentic platform from open components |

---

## Cross-Idea Analysis

### Structural Homologies

| Homology | Qwen3.5 | DeepSeek V4 | Kimi K2.6 |
|----------|---------|-------------|-----------|
| **H1: Attention replacement** | Gated DeltaNet (linear recurrence) | CSA+HCA (compression+sparsity) | MLA (low-rank latent) |
| **H2: MoE ratio** | 512/10 (30:1) | 384/6 (33:1) | 384/8 (32:1) |
| **H3: Shared expert** | Yes (1024 dim) | Yes | Yes |
| **H4: Hash routing** | No | First 3 layers | First 3 layers |
| **H5: Multiple reasoning modes** | 2 modes (per-token) | 3 modes (per-conversation) | 2 modes (per-call) |
| **H6: Multimodal** | Early fusion (text+image+video) | None (text-only) | Two-stage fusion (text+image) |
| **H7: Training precision** | FP8 | FP4+FP8 | BF16 |
| **H8: License** | Apache 2.0 | MIT | Modified MIT |

### Cross-Idea Disparities

| Disparity | Most Affected Pair | Severity |
|-----------|-------------------|----------|
| Linear attention vs. compression+sparsity | Qwen3.5 ↔ DeepSeek V4 | fundamental |
| Softmax necessity | Qwen3.5 (25% softmax) ↔ DeepSeek V4 (0% softmax) | fundamental |
| Per-token vs. per-conversation thinking | Qwen3.5 ↔ DeepSeek V4 | structural |
| Multimodal vs. text-only | DeepSeek V4 ↔ both others | structural |
| FP4 vs. FP8 vs. BF16 training | All three | fundamental |
| Auxiliary loss necessity | Qwen3.5 (yes) ↔ Kimi K2.6 (no) | fundamental |

### Cross-Idea Synthetic Composites

| SYN | Constituents From | Description |
|-----|-------------------|-------------|
| SYN-1 | A_Q1 (Qwen) + Engram (DeepSeek research) | Gated DeltaNet + conditional memory |
| SYN-4 | A_Q1 (Qwen) + A_K1 (Kimi) | Gated DeltaNet with MLA replacing softmax |
| SYN-7 | A_K9 (Kimi) + A_D7 (DeepSeek) | Kimi K2.6 with FP4 training |
| SYN-11 | A_Q10 (Qwen) + A_K7 (Kimi) | MCP + Agent Swarm |
| SYN-12 | DeepSeek V4 + A_Q5 (Qwen multimodal) | Multimodal DeepSeek V4 |

### Cross-Idea Emergence

**Emergent Pattern E1**: The convergence on MoE (~30:1 ratio) across all three architectures suggests a **universal scaling law** for sparse activation in frontier models: the optimal total:active parameter ratio is approximately 30:1, regardless of attention mechanism. If this holds, future architectures can fix this ratio and focus innovation on attention.

**Emergent Pattern E2**: The diversity of attention strategies (linear recurrence, compression+sparsity, low-rank latent) alongside MoE convergence suggests that **attention is the remaining open research frontier** in LLM architecture. The MoE problem is "solved" (we know the ratio and the shared-expert pattern), but the attention problem is not.

**Emergent Pattern E3**: All three architectures have similar benchmark performance (<5% gap on most metrics) despite radically different designs. This suggests a **capability ceiling** imposed not by architecture but by training data, compute budget, or fundamental limits of the Transformer paradigm. If true, further architectural innovation will yield diminishing returns until a genuinely new paradigm emerges.
