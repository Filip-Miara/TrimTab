# Synthetic Variant Catalog

---

## SYN-1: Gated DeltaNet + Engram Hybrid (Cross-Architecture Merge)

| Field | Value |
|-------|-------|
| **Base Concept** | A_Q1 (Gated DeltaNet) + Engram paper (conditional memory via hash lookup) |
| **Operator** | M5 (MERGE) |
| **Description** | Merge Qwen3.5's Gated DeltaNet with DeepSeek's Engram conditional memory. The Gated DeltaNet provides the linear recurrence backbone; the Engram hash table (stored in CPU DRAM) provides an O(1) lookup path for static knowledge (syntax, entities, facts). The gating vector `g_t` learns to decide whether to use the recurrent state, the Engram lookup, or both. |
| **Pros** | + Gated DeltaNet's O(n) recurrence handles dynamic context; Engram's O(1) hash handles static facts without polluting the state + Engram's 100B-parameter table in DRAM for 2.8% throughput cost is essentially free storage + The gating mechanism directly controls the Engram-retrieval ratio — per-token adaptive retrieval |
| **Cons** | - Two completely different memory systems must be trained jointly — optimization instability risk - Engram paper tested at 27B; scaling to 397B unknown - PCIe prefetching for hash lookups adds system complexity |
| **Merits** | Best for factual knowledge + long-context reasoning tasks. Could dramatically improve knowledge benchmarks (MMLU, SimpleQA) while keeping context costs linear. |
| **Risks** | Both systems (gate collapse, hash collision) could compound if not carefully managed. |
| **Quality Scores** | Novelty: 5 | Feasibility: 2 | Coherence: 3 | Risk: 4 | Robustness: 2 |

---

## SYN-2: CSA with Adaptive Top-K (Mutate DeepSeek V4)

| Field | Value |
|-------|-------|
| **Base Concept** | A_D1 (CSA with fixed top-k = 1024) |
| **Operator** | M3 (SCALE — make top-k input-dependent) |
| **Description** | Replace the fixed top-k=1024 in CSA with an input-dependent top-k: `k = f(query_token)`, where `f` is a learned function that predicts the required number of compressed KV entries. |
| **Pros** | + Adaptive compute budget — easy queries use fewer entries, hard queries use more + Could recover the "gap band" between SWA and CSA by increasing k when the query needs broad context + Better compute-quality tradeoff |
| **Cons** | - Variable-length attention requires dynamic batch scheduling - The learned function `f` adds complexity - If k varies widely, hardware utilization suffers |
| **Merits** | Ideal for variable-difficulty workloads (most real-world LLM use) — the model spends compute where it matters. |
| **Risks** | The adaptive function could be gamed by adversarial queries to force maximum k always, negating the benefit. |
| **Quality Scores** | Novelty: 4 | Feasibility: 3 | Coherence: 3 | Risk: 3 | Robustness: 3 |

---

## SYN-3: Dynamic Rank MLA (Mutate Kimi K2.6)

| Field | Value |
|-------|-------|
| **Base Concept** | A_K1 (MLA with fixed rank 512) |
| **Operator** | M3 (SCALE — make rank variable) |
| **Description** | Replace fixed-rank MLA with rank-variable MLA: `rank = r(query_token, head)`, where different heads get different latent ranks (e.g., heads 1-8 get rank 1024 for high-precision retrieval, heads 9-64 get rank 256 for broad coverage). |
| **Pros** | + Per-head rank allocation matches functional specialization + High-rank heads preserve detail; low-rank heads save cache - Total KV cache equals fixed-rank MLA but strategically allocated |
| **Cons** | - Per-head variable rank complicates the attention kernel - Determining optimal per-head rank requires extensive tuning or learned routing - Training stable with variable rank is unproven |
| **Merits** | Best for tasks requiring both precise retrieval (high-rank) and broad context (many low-rank heads). |
| **Risks** | Head-rank allocation may interact poorly with the routing in MoE. |
| **Quality Scores** | Novelty: 4 | Feasibility: 2 | Coherence: 3 | Risk: 4 | Robustness: 2 |

---

## SYN-4: Gated Attention Becomes MLA (Substitute in Qwen3.5)

| Field | Value |
|-------|-------|
| **Base Concept** | A_Q2 (Gated Attention, 25% softmax layers in Qwen3.5) |
| **Operator** | M1 (SUBSTITUTE — replace Gated Attention with MLA) |
| **Description** | Replace the 25% softmax Gated Attention layers in Qwen3.5 with DeepSeek/Kimi's MLA. The 75% Gated DeltaNet layers stay. The 25% layers now use MLA instead of full softmax — meaning *all* layers are linear/sub-quadratic. |
| **Pros** | + Eliminates the last O(n²) bottleneck in Qwen3.5 + MLA's compressed KV cache at rank 512 is more efficient than full GQA + Compatible with Qwen3.5's existing linear attention infrastructure |
| **Cons** | - Training from scratch or continued pretraining needed - MLA's interaction with Gated DeltaNet is untested - May lose the retrieval precision that softmax provides |
| **Merits** | If successful, this would be the first architecture with zero quadratic attention cost at full frontier quality — a potential breakthrough. |
| **Risks** | Without any full softmax attention, the model may lose long-range retrieval capability entirely — the Gated DeltaNet+MLA combination may both be lossy. |
| **Quality Scores** | Novelty: 5 | Feasibility: 2 | Coherence: 2 | Risk: 5 | Robustness: 1 |

---

## SYN-5: DeepSeek V4 with Dense Tuning Islands (Substitute for Hash Routing)

| Field | Value |
|-------|-------|
| **Base Concept** | A_D10 (Hash routing for first 3 MoE layers in DeepSeek V4) |
| **Operator** | M1 (SUBSTITUTE — replace hash routing with learned routing + dense islands) |
| **Description** | Instead of hash routing (deterministic expert assignment), use learned routing but add 2-3 dense FFN layers interspersed in the first few layers. These dense layers serve as "universal feature extractors" that process common token patterns deterministically, reducing the load on the learned router. |
| **Pros** | + Dense layers always have the computation needed for early features + Learned routing can adapt to input, unlike hash routing + Additional dense layers provide the "safety net" that hash routing lacks |
| **Cons** | - Dense layers add active parameter cost (all parameters always active) - Increases total FLOPs by ~5-10% - Early-layer dense+MoE combination is untested |
| **Merits** | Better than either pure hash routing or pure learned routing for early-layer stability. |
| **Risks** | Dense layers may learn patterns that should be expert-specialized, defeating the MoE's purpose. |
| **Quality Scores** | Novelty: 3 | Feasibility: 4 | Coherence: 4 | Risk: 2 | Robustness: 4 |

---

## SYN-6: Qwen3.5 with Anticipatory Routing (Cross-Architecture Borrow)

| Field | Value |
|-------|-------|
| **Base Concept** | A_Q3 (Qwen3.5 MoE routing with auxiliary loss) |
| **Operator** | M1 (SUBSTITUTE — replace routing with anticipatory routing from DeepSeek V4) |
| **Description** | Replace Qwen3.5's top-k softmax + auxiliary loss routing with DeepSeek V4's anticipatory routing (computing routing decisions Δt steps ahead). The auxiliary loss is removed — stability comes from the temporal decoupling. |
| **Pros** | + Eliminates the auxiliary loss's interference with quality-based selection + Decouples routing and backbone training dynamics |
| **Cons** | - Anticipatory routing is "insufficiently understood" (DeepSeek's own words) - Optimal Δt for 512 experts unknown - May cause oscillations at scale |
| **Merits** | If anticipatory routing works at 512 experts, it would be strictly better than auxiliary-loss routing. |
| **Risks** | The mechanism that DeepSeek itself describes as poorly understood may fail at larger expert counts. |
| **Quality Scores** | Novelty: 4 | Feasibility: 2 | Coherence: 3 | Risk: 4 | Robustness: 2 |

---

## SYN-7: Kimi K2.6 with FP4 Experts (Cross-Architecture Borrow)

| Field | Value |
|-------|-------|
| **Base Concept** | A_K9 (Kimi K2.6 native INT4) |
| **Operator** | M1 (SUBSTITUTE — replace INT4 with DeepSeek-style FP4) |
| **Description** | Apply DeepSeek V4's FP4 quantization-aware training to Kimi K2.6's MoE experts. Instead of post-training INT4 quantization, train the experts at FP4 from scratch. |
| **Pros** | + FP4 training ensures the model learns within precision constraints + INT4 inference on top of FP4 training may be even more efficient + Proven technique (DeepSeek V4) |
| **Cons** | - Requires QAT infrastructure investment - Kimi's sigmoid routing may be sensitive to FP4 precision - Training instability risk (as seen in DeepSeek V4) |
| **Merits** | Could make Kimi K2.6's 1T-parameter model deployable on single-node hardware. |
| **Risks** | Kimi's noaux_tc routing (no auxiliary loss) combined with FP4 quantization may produce training dynamics that are unstable without the infrastructure DeepSeek built (anticipatory routing, mHC). |
| **Quality Scores** | Novelty: 3 | Feasibility: 3 | Coherence: 3 | Risk: 3 | Robustness: 3 |

---

## SYN-8: Triple Hybrid — Gated DeltaNet + MLA + CSA (Cross-Architecture Merge)

| Field | Value |
|-------|-------|
| **Base Concept** | All three attention mechanisms (A_Q1 + A_K1 + A_D1) |
| **Operator** | M5 (MERGE) — maximal cross-architecture fusion |
| **Description** | A three-tier attention stack: (1) Gated DeltaNet for O(n) local recurrence (75% of compute), (2) MLA for compressed mid-range attention (15%), (3) CSA+HCA for sparse long-range attention (10%). Each tier handles a different range and type of dependency. |
| **Pros** | + Theoretically optimal — each mechanism does what it does best + Maximum flexibility for different input types + Would likely dominate all benchmarks |
| **Cons** | - Extreme architectural complexity (all three + MoE + mHC + ...) - Training stability is a massive unknown - Inference would require custom hardware or extremely optimized kernels - Essentially a research project, not a deployable model |
| **Merits** | The "platonic ideal" of attention — a model that uses every known efficiency trick simultaneously. |
| **Risks** | Combinatorial explosion of interacting mechanisms — failures could compound in unpredictable ways. |
| **Quality Scores** | Novelty: 5 | Feasibility: 1 | Coherence: 1 | Risk: 5 | Robustness: 1 |

---

## SYN-9: NEGATE Shared Expert (Remove from All Three)

| Field | Value |
|-------|-------|
| **Base Concept** | A_Q4, shared expert component (all three architectures have this) |
| **Operator** | M10 (NEGATE — remove shared expert) |
| **Description** | Remove the shared expert from Qwen3.5/DeepSeek V4/Kimi K2.6 entirely. All tokens go through routed experts only. |
| **Pros** | + Removes always-active parameter overhead + Forces routed experts to handle all knowledge (potentially better specialization) + Simplifies architecture |
| **Cons** | - Common knowledge must be replicated across multiple experts (waste) - Training stability may decrease without the shared "backbone" expert - Benchmark performance likely drops 1-3% |
| **Merits** | For extremely capacity-constrained deployments where every parameter counts. |
| **Risks** | Models trained with shared experts cannot simply remove them at inference — retraining needed. |
| **Quality Scores** | Novelty: 2 | Feasibility: 4 | Coherence: 4 | Risk: 2 | Robustness: 4 |

---

## SYN-10: DeepSeek V4 with 3 Reasoning Modes at Per-Token Granularity

| Field | Value |
|-------|-------|
| **Base Concept** | A_D9 (Three reasoning modes — full-conversation level) |
| **Operator** | M2 (INVERT — make per-token instead of per-conversation) |
| **Description** | Instead of setting reasoning mode (Non-Think/High/Max) once per conversation, allow the model to switch modes per token — exactly as Qwen3.5 does with enable_thinking, but with three levels instead of two. |
| **Pros** | + Optimal compute allocation — trivial tokens skip reasoning, complex tokens spend more + More flexible than either Qwen3.5 (2 levels) or DeepSeek V4 (3 levels but conversation-level) |
| **Cons** | - Requires token-level mode selection mechanism (significant architectural change) - Mode-switching overhead per token may negate the savings - Training the router for per-token effort prediction is hard |
| **Merits** | Best of both worlds — per-token granularity with three effort levels. |
| **Risks** | The per-token switching mechanism may interact poorly with causal attention (a token thinking hard produces different context for subsequent tokens). |
| **Quality Scores** | Novelty: 4 | Feasibility: 2 | Coherence: 3 | Risk: 4 | Robustness: 2 |

---

## SYN-11: Qwen3.5 with Kimi's Agent Swarm (Cross-Architecture Borrow)

| Field | Value |
|-------|-------|
| **Base Concept** | A_Q10 (Qwen3.5 MCP-native agent) + A_K7 (Kimi's Agent Swarm) |
| **Operator** | M5 (MERGE — combine agent frameworks) |
| **Description** | Integrate Kimi's Agent Swarm orchestration (300 sub-agents, 4000 steps) with Qwen3.5's MCP-native protocol and tool-calling (BFCL-V4: 72.9). Qwen3.5 serves as the swarm's reasoning backbone; Kimi's task decomposition and orchestration runs on top. |
| **Pros** | + Qwen3.5's strong function calling + Kimi's swarm scaling = best agentic platform + Both are open-weight — can be deployed together + MCP provides standardized tool protocol |
| **Cons** | - Two different architectures running in coordination adds latency - Swarm orchestration overhead may dominate for simple tasks - Integration complexity |
| **Merits** | The most capable open-source agentic platform — could rival Claude in agentic workflows. |
| **Risks** | The combined system may be unreliable — twice as many failure modes. |
| **Quality Scores** | Novelty: 4 | Feasibility: 3 | Coherence: 3 | Risk: 3 | Robustness: 3 |

---

## SYN-12: Native Multimodal DeepSeek V4 (Add Vision)

| Field | Value |
|-------|-------|
| **Base Concept** | DeepSeek V4 (text-only) |
| **Operator** | M8 (CONCRETIZE — add Qwen3.5's early-fusion multimodal) |
| **Description** | Add Qwen3.5's early-fusion multimodal training to DeepSeek V4: integrate a vision encoder (MoonViT or Qwen3.5's ViT) and continue training on image-text-video data with the FP4+FP8 mixed precision pipeline. |
| **Pros** | + DeepSeek V4's 80.6% SWE-bench + vision = best of both worlds + FP4 training could make multimodal more efficient than current approaches |
| **Cons** | - Vision encoder adds 400M parameters FP4-trained - Multimodal training at 1.6T scale is enormous effort - DeepSeek has already committed to multimodal as future work |
| **Merits** | A multimodal model with frontier coding ability does not exist yet — this could be the first. |
| **Risks** | The FP4 MoE training pipeline may not transfer to multimodal data (different data distribution). |
| **Quality Scores** | Novelty: 3 | Feasibility: 2 | Coherence: 4 | Risk: 4 | Robustness: 3 |
