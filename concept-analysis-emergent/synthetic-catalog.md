# Synthetic Variant Catalog [EMERGENT MODE]

Operators: M11 (RANDOMIZE), M12 (OSCILLATE), M5 (MERGE), M9 (TRANSPOSE)
Focus: unconventional recombination, forbidden pairs, self-application

---

## RECOMB-1: Gated DeltaNet as MoE Router (Cross-Level Recombination)

| Field | Value |
|-------|-------|
| **Constituents** | A_Q1 (Gated DeltaNet state) + A_Q3 (MoE router) |
| **Rationale** | The Gated DeltaNet's compressed state contains a summary of all previous tokens. Why have a separate router that only sees the current token? The state already knows what the model has been doing. |
| **Predicted Behavior** | The router receives the current token + the compressed state from the Gated DeltaNet. Expert selection becomes context-aware and smooth — no abrupt routing changes because the state evolves continuously. |
| **Novelty Score** | 5 — no existing model routes experts through the attention state |
| **Operator** | M5 (MERGE) — fuse the compressed state path into the router input |

### Emergent Capability Analysis

| Check | Result |
|-------|--------|
| Qualitatively distinct from constituents? | Y — state-aware routing is not just "better routing," it's a new capability: the model can plan expert utilization across token boundaries |
| Not predictable from constituent properties alone? | Y — neither Gated DeltaNet nor MoE routing alone implies expert selection conditioned on full context state |
| Produces synergy (result > sum of parts in kind)? | Y — the state provides temporal coherence that a per-token router fundamentally cannot |
| **Verdict** | CONFIRMED EMERGENT |

**Trigger Conditions**: Requires a model where Gated DeltaNet and MoE layers are paired (Qwen3.5 architecture). The compressed state must be accessible to the router at the same layer index. Training must be from scratch (cannot retrofit).

**Latent Path**: Qwen3.5-35B-A3B scale → verify state-aware routing improves expert utilization → scale to 397B.

---

## RECOMB-2: Lightning Indexer as Neural Search Engine (Domain-Transposed)

| Field | Value |
|-------|-------|
| **Constituents** | A_D4 (Lightning Indexer) + external document corpus |
| **Rationale** | The indexer is a learned relevance scoring function: query → scores over compressed candidates. Currently it indexes only KV cache entries. But the mechanism is general — it can index ANY set of compressed representations. Applications: (1) index the web, (2) index a codebase, (3) index a knowledge base. |
| **Predicted Behavior** | The indexer processes an external corpus offline, building compressed representations. At inference, the same indexer selects relevant documents. The LLM then attends to the retrieved documents via the attention mechanism — the retrieval is end-to-end differentiable. |
| **Novelty Score** | 5 — no existing RAG system uses the LLM's own attention indexer for retrieval |
| **Operator** | M9 (TRANSPOSE) — apply CSA's indexing mechanism to information retrieval domain |

### Emergent Capability Analysis

| Check | Result |
|-------|--------|
| Qualitatively distinct from constituents? | Y — neural search with a component designed for attention is different from standard RAG (BM25, dense retrieval, etc.) |
| Not predictable from constituent properties alone? | Y — the indexer was designed for KV selection, not document retrieval. Its use as a general retrieval mechanism is an emergent property of its architecture. |
| Produces synergy (result > sum of parts in kind)? | Y — the indexer's lightweight compression (4:1) and multi-head scoring were optimized for attention context but turn out to be excellent for document retrieval. |
| **Verdict** | CONFIRMED EMERGENT |

**Trigger Conditions**: Requires the CSA architecture with its compression pipeline. The corpus must be pre-compressed into the same KV format. The indexer's top-k can be increased for larger corpora.

**Latent Path**: Take DeepSeek V4's indexer + compression → freeze them → use as a RAG retrieval module for any LLM. The indexer is trained, not hand-designed.

---

## RECOMB-3: Self-Applying Anticipatory Routing (Self-Application)

| Field | Value |
|-------|-------|
| **Constituents** | A_D11 (Anticipatory Routing — computing routing Δt steps ahead applied to ITS OWN computation) |
| **Rationale** | Anticipatory routing decouples backbone updates from routing decisions. What if the same principle is applied to the architecture design process itself? The model pre-computes its own architectural design decisions before the computation that depends on them — a self-referential prediction loop. |
| **Predicted Behavior** | The model has an "anticipatory planner" that predicts what architectural features it will need N steps ahead and pre-configures them. This is like an operating system that pre-fetches pages before they're needed, but applied to the model's own architecture. |
| **Novelty Score** | 5 — self-applying a design principle to itself is the definition of meta-cognition |
| **Operator** | Self-application of A_D11 to the concept of A_D11 itself |

### Emergent Capability Analysis

| Check | Result |
|-------|--------|
| Qualitatively distinct from constituents? | Y — anticipatory routing at the meta-level (architecture design) is different from the token-level (expert selection) |
| Not predictable from constituent properties alone? | Y — nothing about anticipatory routing at the token level implies it works for multi-step planning of architectural decisions |
| Produces synergy? | Y — the model becomes self-modifying: it predicts what architectural configuration it needs and adjusts itself. This is a step toward recursive self-improvement. |
| **Verdict** | CONFIRMED EMERGENT — if implementable. Feasibility is extremely low. |

**Trigger Conditions**: Requires a model architecture that supports dynamic reconfiguration (change experts, attention parameters, etc. at runtime). No current model supports this.

**Latent Path**: Hardware-software co-designed architecture with reconfigurable compute units (FPGA-like for neural nets).

---

## RECOMB-4: Forbidden Pair — Gated DeltaNet's State as KV for CSA (Cross-Architecture, Forbidden)

| Field | Value |
|-------|-------|
| **Constituents** | A_Q1 (Gated DeltaNet state) + A_D1 (CSA indexed attention) |
| **Rationale** | These were identified as "competing paradigms" in analytic mode. **Forbidden pair**: one is O(1) state recurrence, the other is compression+sparsity over all tokens. What if the Gated DeltaNet's state serves as the "compressed summary" that CSA treats as a single heavily compressed entry? The state becomes HCA's 128:1 compression — a single vector representing the entire sequence up to this point. |
| **Predicted Behavior** | Gated DeltaNet produces a running state. At each step: (1) the state is treated as a single heavily compressed KV entry, (2) CSA indexes over the actual token history with compression 4:1 + top-k, (3) the model attends to BOTH the state (global summary) and the selected sparse entries (specific tokens). This is a two-scale attention system: global via state, local via sparse tokens. |
| **Novelty Score** | 5 — no existing model combines linear recurrence with sparse attention in this way |
| **Operator** | M5 (MERGE) — forbidden pair intentionally combined |

### Emergent Capability Analysis

| Check | Result |
|-------|--------|
| Qualitatively distinct from constituents? | Y — the two-scale (state + sparse tokens) attention regime is different from either Gated DeltaNet alone or CSA alone |
| Not predictable? | Y — the synergy arises from covering different timescales: state covers everything (lossy), sparse tokens cover specific events (precise) |
| Produces synergy? | Y — the state prevents the "compression gap" problem (information lost between 4:1 and 128:1 compression) by providing a unified summary. CSA prevents the "state saturation" problem by offering direct token access. |
| **Verdict** | CONFIRMED EMERGENT — this is the strongest candidate for a practical breakthrough |

**Trigger Conditions**: Requires a multi-branch attention layer where one branch is the Gated DeltaNet state and another is CSA. The model must learn when to trust the state vs. when to look at specific tokens.

**Latent Path**: Add a Gated DeltaNet recurrence to a DeepSeek V4 layer. Or add CSA sparse indexing to a Qwen3.5 layer. Either path creates the two-scale system.

---

## RECOMB-5: Controllable Forgetting Gate as Privacy Mechanism (Domain-Transposed)

| Field | Value |
|-------|-------|
| **Constituents** | A_Q1 gate collapse vulnerability + privacy regulation (GDPR "right to be forgotten") |
| **Rationale** | The gate collapse vulnerability (g_t → 1, state becomes read-only, model cannot update its memory) is considered a failure mode. But what if it's a feature? If the gate can be forced to 1 for specific information, the model **cannot update its memory with that information** — it literally cannot learn something. |
| **Predicted Behavior** | An API parameter `forget: ["user_id_123", "specific_fact"]` that forces the gating vector to specific values for certain key tokens, preventing encoding. The model processes the input but its compressed state does not record it. This is machine unlearning at the architectural level, not post-hoc. |
| **Novelty Score** | 4 — controllability in LLMs is active research, but architectural-level forgetting is novel |
| **Operator** | M9 (TRANSPOSE) — take a failure mode and transpose it to the privacy domain |

### Emergent Capability Analysis

| Check | Result |
|-------|--------|
| Qualitatively distinct? | Y — architectural unlearning during generation is fundamentally different from post-hoc model editing or retraining |
| Not predictable? | Y — the gate collapse was identified as a weakness; its application to privacy is not obvious from the mechanism alone |
| Produces synergy? | Y — combines a "bug" (gate collapse) with a regulatory requirement (forgetting) into a feature (controllable memory) |
| **Verdict** | CONFIRMED EMERGENT — with the caveat that implementation requires exposing the gating vector as a controllable parameter, which is currently not possible |

**Trigger Conditions**: Gated DeltaNet architecture with exposed gate API. Requires runtime control of the gating vector.

---

## RECOMB-6: Heterogeneous Agent Swarm — Qwen Vision + DeepSeek Coding + Kimi Orchestration (Cross-Architecture, Forbidden Pair)

| Field | Value |
|-------|-------|
| **Constituents** | A_Q5 (early fusion vision) + A_D5 (SWE-bench 80.6% coding) + A_K7 (300-agent swarm) |
| **Rationale** | Each architecture has a unique strength: Qwen's vision, DeepSeek's coding, Kimi's swarm. These were designed independently and are not compatible. But what if they were combined as specialized agents in a heterogeneous swarm? |
| **Predicted Behavior** | Kimi's swarm decomposes a task into subtasks. Vision subtasks → Qwen agents. Coding subtasks → DeepSeek agents. Orchestration and communication → Kimi's swarm protocol. The total system can perform tasks none of the individual models can: *"Build this UI from a sketch and make it work"* — Qwen sees the sketch, DeepSeek codes it, Kimi's swarm manages the process. |
| **Novelty Score** | 5 — heterogeneous model swarm with complementary specializations |
| **Operator** | M5 (MERGE) — forbidden pair across architectures |

### Emergent Capability Analysis

| Check | Result |
|-------|--------|
| Qualitatively distinct? | Y — no single model can do all three: see, code, and orchestrate at this level |
| Not predictable? | Y — the specific synergy (vision → coding → deployment) is a chain of capabilities that no individual model possesses end-to-end |
| Produces synergy? | Y — the whole is not just more, but *different*. The capability to "build a UI from a sketch" is not just more vision or more coding — it's a new type of capability: visual programming. |
| **Verdict** | CONFIRMED EMERGENT — most practically impactful recombination |

**Trigger Conditions**: Requires a swarm orchestration protocol (Kimi's Agent Swarm or similar) that can route tasks to heterogeneous model backends. OpenRouter-like API routing at the agent level.

**Latent Path**: Use Kimi K2.6's swarm as the orchestrator, routing vision queries to Qwen3.5-Plus API and coding queries to DeepSeek V4-Pro API through a unified inference API.

---

## RECOMB-7: Precision Oscillation During Training (OSCILLATE)

| Field | Value |
|-------|-------|
| **Constituents** | A_D7 (FP4 training) + M12 (OSCILLATE operator) |
| **Rationale** | Currently, training precision is static (FP4 for experts for the whole training run). What if precision oscillates: train at FP4 for N steps, then BF16 for M steps, then FP4 again? The oscillation forces the model to alternately compress and expand its representations. |
| **Predicted Behavior** | During FP4 phases, the model learns to be robust to quantization. During BF16 phases, it "recovers" any information lost to compression. The oscillation creates a rhythmic training dynamic akin to annealing: periods of constraint, periods of freedom. |
| **Novelty Score** | 4 — oscillating precision during training is unexplored territory |
| **Operator** | M12 (OSCILLATE) — alternate between two precision regimes |

### Emergent Capability Analysis

| Check | Result |
|-------|--------|
| Qualitatively distinct? | Y — oscillating precision creates a training dynamic that static precision cannot: periodic constraint/relaxation |
| Not predictable? | Y — the interaction between alternating precision regimes is genuinely unknown |
| Produces synergy? | Potentially — the BF16 "recovery phases" could prevent the permanent information loss that concerns FP4 critics |
| **Verdict** | QUANTITATIVE ENHANCEMENT (needs empirical validation) |

**Trigger Conditions**: Requires a training infrastructure that supports dynamic precision switching. QAT must remain stable across precision changes.

---

## RECOMB-8: Self-Referential Architecture (Feeding Peak P as Input to Itself)

| Field | Value |
|-------|-------|
| **Constituents** | The entire Peak P (contemporary LLM architecture design space) applied to ITSELF |
| **Rationale** | Architecture design is itself a cognitive task performed by humans. What if an LLM with the combined insights of Qwen3.5, DeepSeek V4, and Kimi K2.6 is asked to design the NEXT generation of architecture? The peak concept applied to its own evolution. |
| **Predicted Behavior** | A meta-LLM (trained on all three architectures' design documents) proposes a fourth architecture that synthesizes their innovations. This is not human-designed — it's AI-generated architecture. The output of this analysis (this very document) becomes training data. |
| **Novelty Score** | 5 — AI-designed architectures are the holy grail of automated ML |
| **Operator** | Self-application of P to P |

### Emergent Capability Analysis

| Check | Result |
|-------|--------|
| Qualitatively distinct? | Y — AI-generated architecture design is categorically different from human-designed |
| Not predictable? | Y — the architecture that emerges from this process would be unpredictable by definition |
| Produces synergy? | Y — combines three architectures into a fourth that may transcend all three |
| **Verdict** | CONFIRMED EMERGENT (in principle — requires a meta-LLM trained on architecture design) |

**Trigger Conditions**: Requires a capable LLM (any of the three) fine-tuned on architecture design corpora (papers, configs, ablation studies). The model proposes architectures, a simulation validates them, the results train the next iteration — an architectural GAN.

**Latent Path**: Fine-tune DeepSeek V4 (best coder) on all three model's technical reports + configs. Prompt it to design "DeepSeek V5" that incorporates Qwen3.5's multimodal and Kimi's swarm. Evaluate the proposed architecture through simulation.

---

## RECOMB-9: Calibrated Uncertainty via MLA Rank (RANDOMIZE)

| Field | Value |
|-------|-------|
| **Constituents** | A_K1 (MLA rank) + confidence calibration + M11 (RANDOMIZE) |
| **Rationale** | The rank-512 limit means some KV information is always lost. The reconstruction error between original KV and compressed KV is a measurable uncertainty signal. **High reconstruction error = the model doesn't have good representation for this input = the model should be uncertain about its output.** |
| **Predicted Behavior** | The compression error of MLA serves as an internal confidence score. When rank-512 cannot faithfully represent the current KV, the model generates with higher uncertainty (entropy, sampling temperature). This is **introspective uncertainty** — the model literally knows when it doesn't know because its internal compression fails. |
| **Novelty Score** | 5 — introspection via compression failure is a novel uncertainty mechanism |
| **Operator** | M11 (RANDOMIZE) — use a "failure" (compression loss) as a signal for a different purpose (uncertainty) |

### Emergent Capability Analysis

| Check | Result |
|-------|--------|
| Qualitatively distinct? | Y — this is true introspective uncertainty: the model measures its own representation quality, not just output probabilities |
| Not predictable? | Y — the connection between MLP reconstruction error and output confidence is non-obvious |
| Produces synergy? | Y — turns a limitation (fixed rank) into a feature (uncertainty estimation) |
| **Verdict** | CONFIRMED EMERGENT — immediately implementable |

**Trigger Conditions**: Requires access to MLA's reconstruction error (original KV vs compressed-then-reconstructed KV). This is available at runtime — no training change needed.

**Latent Path**: In any MLA-equipped model (Kimi K2.6, DeepSeek V3/V4): compute `||KV_original - KV_reconstructed||` for each layer. If the norm exceeds a threshold, increase sampling temperature for that token. This is a zero-cost introspection mechanism.

---

## RECOMB-10: Temporal Memory Indexing via CSA + Gated DeltaNet (Synthesis of All Three)

| Field | Value |
|-------|-------|
| **Constituents** | A_Q1 (Gated DeltaNet state) + A_D4 (Lightning Indexer) + A_K1 (MLA latent) |
| **Rationale** | The Gated DeltaNet state maintains a compressed current context. The Lightning Indexer retrieves sparse past entries. MLA provides efficient storage. Combine all three for a **temporal memory system** that stores long-term memories (MLA compressed), indexes them (Lightning Indexer), and updates them with new context (Gated DeltaNet). |
| **Predicted Behavior** | Past conversations are stored as MLA's latent vectors in a vector database. When a new conversation begins, the Lightning Indexer scores the stored latents for relevance to the current query. Relevant memories are retrieved. The Gated DeltaNet's state is initialized from the retrieved memories. The model "remembers" past interactions by literally loading them into its active state. |
| **Novelty Score** | 5 — this is a complete long-term memory system using all three architectures' components |
| **Operator** | M5 (MERGE) — tripartite fusion of all three attention innovations |

### Emergent Capability Analysis

| Check | Result |
|-------|--------|
| Qualitatively distinct? | Y — no existing LLM has a long-term memory architecture built from first principles of the attention mechanism itself |
| Not predictable? | Y — the three components were designed for different purposes (attention, sparsity, compression); their use as a memory system is emergent |
| Produces synergy? | Y — each component solves a problem the others have: Gated DeltaNet needs more capacity (provided by MLA storage), CSA needs temporal coherence (provided by Gated DeltaNet state), MLA needs retrieval (provided by Lightning Indexer) |
| **Verdict** | CONFIRMED EMERGENT — most architecturally complete emergent capability |

**Trigger Conditions**: Requires a model that has all three mechanisms, or a pipeline that combines three separate models' capabilities. The vector database must store MLA-compressed latents. The Lightning Indexer must be able to score stored latents.

**Latent Path**: Build a memory layer on top of any existing model: (1) use MLA-style compression (rank-512 projection) to compress conversation history into latents, (2) store in a vector DB, (3) train a small Lightning Indexer to score stored latents, (4) retrieve and inject into the model's context as soft prompts.
