# Concept Pyramid: Three Architecture Analysis

**Generated**: 2026-06-13
**Mode**: analytic
**Max Depth**: 5
**Evidence Grounding**: enabled

---

## Meta-Pyramid: Peak Concept

**Peak (P)**: Contemporary LLM Architecture Design Space (Qwen3.5 + DeepSeek V4 + Kimi K2.6)

**Cross-Cutting Composites**:
- C_X1: Attention Innovation Strategies (spanning all three)
- C_X2: MoE Scaling Philosophies
- C_X3: Training Precision Approaches
- C_X4: Multimodal Integration Strategies
- C_X5: Inference Mode Flexibility

---

## Architecture A: Qwen 3.5

### Atomic Concepts (Level 1)

| ID | Atom | Grounding | Confidence | Key Assumptions | Known Limitations |
|----|------|-----------|------------|-----------------|-------------------|
| A_Q1 | **Gated Delta Network (linear recurrence)** | arXiv:2412.06464, ICLR 2025 | 0.85 | Linear recurrence can match softmax attention quality; delta rule improves on Mamba2 | Only ~1 year of validation; less community optimization than softmax |
| A_Q2 | **Gated Attention (softmax, 25% layers)** | Qwen3.5 config.json | 0.95 | 25% softmax layers suffice for retrieval tasks | Adds O(n²) cost for those layers |
| A_Q3 | **MoE with 512 experts, Top-10 routing** | Qwen3.5 config.json | 0.90 | Extreme expert specialization improves quality-cost tradeoff | 512 experts require careful sharding; load balancing complex |
| A_Q4 | **Shared expert (dense FFN)** | Qwen3.5 config.json | 0.90 | Common knowledge can be captured in a shared pathway | Adds parameter cost that's always active |
| A_Q5 | **Early-fusion native multimodal training** | Qwen3.5 blog post | 0.85 | Joint vision-language training from scratch avoids adapter loss | ViT runs even on text-only inputs |
| A_Q6 | **FP8 training pipeline** | Qwen3.5 model card | 0.85 | FP8 precision sufficient for training without quality loss | May not scale to all architectures |
| A_Q7 | **248K vocabulary for 201 languages** | Qwen3.5 config.json | 0.90 | Expanded vocabulary improves encoding efficiency | Larger embedding layer; more parameters |
| A_Q8 | **Multi-Token Prediction (MTP)** | Qwen3.5 config.json | 0.80 | Predicting 2-4 future tokens improves throughput via speculative decoding | Extra training complexity |
| A_Q9 | **Per-token thinking/non-thinking control** | Qwen3.5 model card | 0.95 | API-level enable_thinking toggle works | Removed soft tokens from Qwen3; less flexible |
| A_Q10 | **MCP-native agent protocol** | Qwen3.5 model card | 0.85 | Standardized tool use protocol improves agent interoperability | Requires MCP-compatible client | |

### Composites by Level

**Level 2**:
| ID | Composition | Junctions |
|----|------------|-----------|
| C2_Q1 | **Hybrid Attention** = {A_Q1, A_Q2} | J_Q1: A_Q1→A_Q2 (compositional: 75/25 split ratio) |
| C2_Q2 | **MoE System** = {A_Q3, A_Q4} | J_Q2: A_Q3→A_Q4 (synergistic: shared expert stabilizes MoE) |
| C2_Q3 | **Thinking System** = {A_Q9} | (single atom at this level) |
| C2_Q4 | **Multimodal System** = {A_Q5, A_Q6, A_Q7} | J_Q3: A_Q5→A_Q6 (dependency: FP8 enables multimodal at scale), J_Q4: A_Q5→A_Q7 (synergistic: larger vocab+multilingual for global market) |
| C2_Q5 | **Agent System** = {A_Q10, A_Q8} | J_Q5: A_Q8→A_Q10 (synergistic: MTP accelerates agent loop) |

**Level 3**:
| ID | Composition | Junctions |
|----|------------|-----------|
| C3_Q1 | **Efficient Core** = {C2_Q1, C2_Q2} | J_Q6: C2_Q1↔C2_Q2 (compositional: attention→MoE per layer), J_Q7: A_Q1→A_Q3 (hierarchical: linear attention reduces KV, enabling larger MoE) |
| C3_Q2 | **Full-Stack Model** = {C3_Q1, C2_Q3, C2_Q4, C2_Q5} | J_Q8: C2_Q4→C2_Q3 (specifically: multimodal thinking), J_Q9: C2_Q5→C2_Q1 (synergistic: agent calls attention) |

---

## Architecture B: DeepSeek V4

### Atomic Concepts (Level 1)

| ID | Atom | Grounding | Confidence | Key Assumptions | Known Limitations |
|----|------|-----------|------------|-----------------|-------------------|
| A_D1 | **Compressed Sparse Attention (CSA, 4:1 + top-k)** | DeepSeek V4 Technical Report | 0.90 | Token compression + sparse selection maintains retrieval quality | Top-k may miss important tokens; complex dual-path design |
| A_D2 | **Heavily Compressed Attention (HCA, 128:1)** | DeepSeek V4 Technical Report | 0.85 | Aggressive 128:1 compression sufficient for global context | May lose fine-grained long-range dependencies |
| A_D3 | **Sliding Window Attention (SWA, n=128)** | DeepSeek V4 Technical Report | 0.90 | Local context requires fine-grained attention | Adds complexity; window size is a hyperparameter |
| A_D4 | **Lightning Indexer (top-k sparse selection)** | DeepSeek V4 Technical Report | 0.85 | Lightweight indexer can reliably select relevant KV entries | Indexer introduces its own compute; may have recall failures |
| A_D5 | **MoE with 384 experts, Top-6 routing** | DeepSeek V4 Technical Report | 0.90 | 6 experts sufficient per token given mHC and attention innovations | Resource conflict: fewer experts per token than Qwen/Kimi |
| A_D6 | **Manifold-Constrained Hyper-Connections (mHC)** | arXiv:2512.24880 | 0.80 | Birkhoff-constrained residual paths improve gradient flow | 6.7% wall-time overhead; Sinkhorn iterations add complexity |
| A_D7 | **FP4 + FP8 mixed precision (experts at FP4)** | DeepSeek V4 Technical Report | 0.85 | FP4 training produces frontier-quality models | Training instability risk; requires QAT infrastructure |
| A_D8 | **Muon optimizer** | DeepSeek V4 Technical Report | 0.80 | Muon provides 2× compute efficiency over AdamW at scale | Less tested at this scale; requires hybrid with AdamW |
| A_D9 | **Three reasoning modes (Non-Think/High/Max)** | DeepSeek V4 Technical Report | 0.95 | Different context budgets for different task difficulties | System-prompt-level control, not per-token |
| A_D10 | **Hash routing (first 3 MoE layers)** | DeepSeek V4 Technical Report | 0.85 | Deterministic expert assignment avoids routing instability | Less flexible than learned routing for common tokens |
| A_D11 | **Anticipatory Routing** | DeepSeek V4 Technical Report | 0.75 | Decoupling backbone/routing updates prevents instability | "Insufficiently understood" (direct quote from report) |
| A_D12 | **Partial RoPE (last 64 dims only)** | DeepSeek V4 Technical Report | 0.85 | Partial position encoding sufficient for hybrid attention | May limit position-sensitive tasks |

### Composites by Level

**Level 2**:
| ID | Composition | Junctions |
|----|------------|-----------|
| C2_D1 | **Hybrid Attention Stack** = {A_D1, A_D2, A_D3, A_D4, A_D12} | J_D1: A_D1→A_D2→A_D3 (compositional: interleaved layers), J_D2: A_D4→A_D1 (dependency: indexer enables CSA), J_D3: A_D12→A_D1 (compositional: partial RoPE within CSA) |
| C2_D2 | **MoE System** = {A_D5, A_D10, A_D11} | J_D4: A_D10→A_D5 (hierarchical: hash routing for first 3 layers), J_D5: A_D11→A_D5 (dependency: anticipatory routing stabilizes training) |
| C2_D3 | **Training Infrastructure** = {A_D7, A_D8} | J_D6: A_D7↔A_D8 (synergistic: Muon handles FP4 stability), J_D7: A_D7→A_D11 (causal: FP4 requires anticipatory routing) |
| C2_D4 | **Residual System** = {A_D6} | (single atom) |
| C2_D5 | **Reasoning System** = {A_D9} | (single atom) |

**Level 3**:
| ID | Composition | Junctions |
|----|------------|-----------|
| C3_D1 | **Attention+MoE Core** = {C2_D1, C2_D2} | J_D8: C2_D1↔C2_D2 (compositional: attention→MoE per layer), J_D9: A_D1→A_D5 (synergistic: CSA's KV reduction enables larger MoE) |
| C3_D2 | **Training+Residual System** = {C2_D3, C2_D4} | J_D10: A_D6→A_D7 (synergistic: mHC stabilizes FP4 training gradients) |
| C3_D3 | **Full Model** = {C3_D1, C3_D2, C2_D5} | J_D11: C3_D2→C3_D1 (dependency: training creates inference), J_D12: C2_D5→C3_D1 (constraint: reasoning modes bound attention usage) |

---

## Architecture C: Kimi K2.6

### Atomic Concepts (Level 1)

| ID | Atom | Grounding | Confidence | Key Assumptions | Known Limitations |
|----|------|-----------|------------|-----------------|-------------------|
| A_K1 | **Multi-Head Latent Attention (MLA)** | DeepSeek V3 arXiv:2412.19437 (adapted by Moonshot) | 0.90 | Low-rank KV compression (rank 512) preserves attention quality | Dependency on DeepSeek V3 codebase; not independently invented |
| A_K2 | **Full MHA (64 heads, no GQA)** | Kimi K2.6 config.json | 0.90 | MLA makes full MHA feasible at 1T scale | All 64 heads generate KV (though compressed) |
| A_K3 | **MoE with 384 experts, Top-8 routing** | Kimi K2.6 config.json | 0.90 | 8 experts per token balances capacity and efficiency | More active than DeepSeek's 6, less than Qwen's 10 |
| A_K4 | **Shared expert + dense first layer** | Kimi K2.6 config.json | 0.90 | Common knowledge + initial processing in dense pathway | First-layer dense adds parameter cost |
| A_K5 | **YaRN RoPE scaling (64×)** | Kimi K2.6 config.json | 0.85 | Frequency-interpolated RoPE maintains quality at 64× extension | Original 4K pretraining → 256K is ambitious |
| A_K6 | **MoonViT vision encoder (400M params)** | Kimi K2.6 config.json | 0.85 | Divided Fixed position embedding handles variable resolutions | 400M additional parameters; custom ViT |
| A_K7 | **Agent Swarm orchestration (300 agents)** | K2.5 arXiv:2602.02276 | 0.80 | Model-native swarm decomposition improves complex task completion | Requires sophisticated orchestration; latency scaling |
| A_K8 | **Thinking + Instant dual modes** | Kimi K2.6 model card | 0.95 | Per-call mode selection for latency vs. quality tradeoff | Less granular than DeepSeek's 3 modes |
| A_K9 | **Native INT4 quantization** | Kimi K2.6 config.json | 0.85 | compressed-tensors format at group_size=32 | Attention layers excluded from quantization |
| A_K10 | **noaux_tc routing (no auxiliary loss)** | Kimi K2.6 config.json | 0.85 | Sigmoid-based routing without load balancing loss | May lead to expert imbalance over long training |
| A_K11 | **Preserve Thinking (cross-turn reasoning)** | Kimi K2.6 model card | 0.80 | Retaining reasoning across turns improves multi-step tasks | Increases context usage; disabled by default |

### Composites by Level

**Level 2**:
| ID | Composition | Junctions |
|----|------------|-----------|
| C2_K1 | **Latent Attention System** = {A_K1, A_K2, A_K5} | J_K1: A_K1→A_K2 (enables: MLA makes full MHA feasible), J_K2: A_K5→A_K1 (compositional: YaRN extends MLA's context range) |
| C2_K2 | **MoE System** = {A_K3, A_K4, A_K10} | J_K3: A_K10→A_K3 (dependency: noaux_tc routing governs expert selection), J_K4: A_K4→A_K3 (hierarchical: shared+dense first = different from pure MoE) |
| C2_K3 | **Multimodal System** = {A_K6} | (single atom) |
| C2_K4 | **Agent System** = {A_K7, A_K11} | J_K5: A_K11→A_K7 (synergistic: preserve thinking enhances swarm coherence) |
| C2_K5 | **Inference System** = {A_K8, A_K9} | J_K6: A_K8↔A_K9 (synergistic: INT4 enables efficient dual-mode inference) |

**Level 3**:
| ID | Composition | Junctions |
|----|------------|-----------|
| C3_K1 | **Attention+MoE Core** = {C2_K1, C2_K2} | J_K7: C2_K1↔C2_K2 (compositional: per-layer sequence), J_K8: A_K1→A_K3 (synergistic: MLA's KV savings leave compute budget for 8 experts) |
| C3_K2 | **Agents+Inference** = {C2_K4, C2_K5} | J_K9: A_K7→A_K8 (temporal: swarm → inference mode selection per task) |
| C3_K3 | **Full Model** = {C3_K1, C2_K3, C3_K2} | J_K10: A_K6→C3_K1 (compositional: vision tokens enter the core), J_K11: C3_K2→C3_K1 (constraint: agent loop drives inference) |

---

## Junctions (Complete Listing)

### Intra-Architecture Junctions

#### Qwen 3.5
| ID | Type | Nodes | Description | Strength |
|----|------|-------|-------------|----------|
| J_Q1 | compositional | A_Q1, A_Q2 | 75% Gated DeltaNet + 25% Gated Attention form hybrid attention | Structural (ratio fixed) |
| J_Q2 | synergistic | A_Q3, A_Q4 | Shared expert stabilizes MoE training by capturing common knowledge | Strong |
| J_Q3 | dependency | A_Q5, A_Q6 | FP8 training enables scaling multimodal to trillions of tokens | Moderate |
| J_Q4 | synergistic | A_Q5, A_Q7 | Larger vocabulary + early-fusion multimodal enables 201-language vision | Strategic |
| J_Q5 | synergistic | A_Q8, A_Q10 | MTP speculative decoding accelerates agent tool-calling loops | Moderate |
| J_Q6 | compositional | C2_Q1, C2_Q2 | Attention → MoE per layer forms the basic transformer block | Structural |
| J_Q7 | hierarchical | A_Q1, A_Q3 | Linear attention's O(n) makes 512-expert MoE practical | Causal |
| J_Q8 | synergistic | C2_Q4, C2_Q3 | Multimodal inputs benefit from thinking-before-answering | Moderate |
| J_Q9 | synergistic | C2_Q5, C2_Q1 | Agent tool use calls attention mechanisms repeatedly | Implementation |

#### DeepSeek V4
| ID | Type | Nodes | Description | Strength |
|----|------|-------|-------------|----------|
| J_D1 | compositional | A_D1, A_D2, A_D3 | CSA + HCA + SWA interleaved in specific layer pattern | Structural |
| J_D2 | dependency | A_D4, A_D1 | Lightning indexer is internal to CSA — CSA cannot function without it | Strong |
| J_D3 | compositional | A_D12, A_D1 | Partial RoPE applied within CSA's MQA heads | Moderate |
| J_D4 | hierarchical | A_D10, A_D5 | Hash routing for first 3 MoE layers is a subset of MoE system | Weak (only 3/61 layers) |
| J_D5 | dependency | A_D11, A_D5 | Anticipatory routing prevents training collapse in MoE | Strong (critical for stability) |
| J_D6 | synergistic | A_D7, A_D8 | Muon optimizer is well-suited to FP4's reduced precision landscape | Moderate |
| J_D7 | causal | A_D7, A_D11 | FP4 training's instability → need for anticipatory routing | Causal link |
| J_D8 | compositional | C2_D1, C2_D2 | Basic transformer block: attention → MoE | Structural |
| J_D9 | synergistic | A_D1, A_D5 | CSA's KV reduction (2% of baseline) frees memory for 384-expert MoE | Strong |
| J_D10 | synergistic | A_D6, A_D7 | mHC's guaranteed gradient flow helps stabilize FP4 training | Moderate |
| J_D11 | dependency | C3_D2, C3_D1 | Training creates the inference model | Temporal |
| J_D12 | constraint | C2_D5, C3_D1 | Reasoning modes bound how attention/MoE is used at inference | Design constraint |

#### Kimi K2.6
| ID | Type | Nodes | Description | Strength |
|----|------|-------|-------------|----------|
| J_K1 | enables | A_K1, A_K2 | MLA's rank-512 compression makes full MHA (64 heads) practical at 1T scale | Strong (key design insight) |
| J_K2 | compositional | A_K5, A_K1 | YaRN 64× scaling extends MLA's effective context range | Moderate |
| J_K3 | dependency | A_K10, A_K3 | noaux_tc sigmoid routing determines expert selection | Strong |
| J_K4 | hierarchical | A_K4, A_K3 | First-layer dense + shared expert = hybrid dense-MoE design | Weak (only 1/61 layers) |
| J_K5 | synergistic | A_K7, A_K11 | Preserve thinking enables coherent multi-turn swarm execution | Moderate |
| J_K6 | synergistic | A_K8, A_K9 | INT4 makes dual-mode inference practical at lower hardware cost | Strong |
| J_K7 | compositional | C2_K1, C2_K2 | Basic transformer block: attention → MoE | Structural |
| J_K8 | synergistic | A_K1, A_K3 | MLA's KV savings leave compute budget for 8 active experts per token | Moderate |
| J_K9 | temporal | A_K7, A_K8 | Swarm first plans tasks, then selects inference mode per subtask | Temporal |
| J_K10 | compositional | A_K6, C3_K1 | Vision tokens enter the attention+MoE core | Structural |
| J_K11 | constraint | C3_K2, C3_K1 | Agent loop determines inference workload on the core | Design constraint |

### Cross-Architecture Junctions (Structural Homologies)

| ID | Type | Node A | Node B | Description |
|----|------|--------|--------|-------------|
| J_X1 | homological | C2_Q1 (Hybrid Attention) | C2_D1 (Hybrid Attention Stack) | Both use hybrid attention but with completely different mechanisms: Qwen = linear+softmax split by layers, DeepSeek = compression+sparsity within each layer |
| J_X2 | homological | C2_Q2 (MoE System) | C2_D2 (MoE System) | C2_K2 (MoE System) | All three use shared experts + routed experts + hash routing (DeepSeek/Kimi) or aux loss (Qwen) |
| J_X3 | homological | A_Q3 (512 experts, Top-10) | A_D5 (384 experts, Top-6) | A_K3 (384 experts, Top-8) | Expert count philosophy: Qwen maximizes specialization, DeepSeek maximizes efficiency, Kimi balances |
| J_X4 | homological | A_Q1 (Gated DeltaNet) | A_D1 (CSA) | A_K1 (MLA) | All three reject standard softmax attention but diverge on replacement: linear recurrence vs. compression+sparsity vs. low-rank latent |
| J_X5 | homological | A_Q5 (Early fusion multimodal) | A_K6 (MoonViT) | — | Qwen and Kimi both do native multimodal training; DeepSeek does none |
| J_X6 | homological | A_Q9 (Thinking/Non-thinking) | A_D9 (Non/High/Max) | A_K8 (Thinking/Instant) | All three offer multiple inference modes; DeepSeek has 3, others have 2 |
| J_X7 | antagonistic | A_Q1 (Gated DeltaNet) | A_D1 (CSA) | These are competing paradigms for attention replacement — one is linear recurrence, the other is compression+sparsity |
| J_X8 | antagonistic | A_Q3 (512 experts, Top-10) | A_D5 (384 experts, Top-6) | Competing MoE philosophies: maximize specialization vs. maximize efficiency |
| J_X9 | synergistic | C3_Q1 (Qwen Efficient Core) | C3_K1 (Kimi Core) | A_Q1 + A_K1: If Gated DeltaNet replaced MLA in Kimi's architecture, 75% linear attention + MLA compression on remaining 25% could be extreme |
| J_X10 | synergistic | A_D7 (FP4 training) | A_Q6 (FP8 training) | Combined FP4+FP8 strategy could achieve even greater efficiency than either alone |

---

## Evidence Grounding Notes

**High confidence (≥0.90)**: Config.json values, official model cards, published technical reports
**Medium confidence (0.80-0.89)**: Third-party verified analyses, papers under review, official but incomplete documentation
**Lower confidence (<0.80)**: Speculative mechanisms, researcher claims not yet replicated, "insufficiently understood" components

Key ungrounded assumptions: The exact training token counts, the specific routing stability mechanisms for noaux_tc, the precise CSA compression ratios (DeepSeek's own doc is sparse), and Gated DeltaNet's scaling behavior at >1T-parameter regimes.

---

## Peak Proposition

**P**: The three architectures represent three distinct strategies for escaping the Transformer's quadratic attention + dense FFN cost scaling:
1. **Qwen 3.5**: _Replace_ attention with linear recurrence (75% of layers), _specialize_ via 512 experts
2. **DeepSeek V4**: _Compress and sparsify_ attention (CSA+HCA), _stabilize_ via mHC+Muon, _train at extreme low precision_
3. **Kimi K2.6**: _Borrow and refine_ MLA from DeepSeek V3, _extend_ via YaRN, _orchestrate_ via Agent Swarm
