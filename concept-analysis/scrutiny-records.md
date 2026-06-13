# Scrutiny Records: 5 Cognitive Lenses

**Generated**: 2026-06-13

---

## 1. A_Q1: Gated Delta Network (Qwen3.5)

| Lens | Findings |
|------|----------|
| **Adversarial** | **Cheapest attack**: Feed a sequence with repetitive token patterns that cause the gating vector `g_t` to converge to a fixed point, causing memory overwrite failure. **Extreme values**: At seq_len → ∞, the compressed state `d × d` saturates — does forgetting keep pace? **Stress failure**: If the delta correction δ_t goes to zero for many consecutive tokens, the state becomes static and cannot recover (dead neuron scenario). **Single point of failure**: The gating vector `g_t` — if it collapses to 0 or 1 for all dimensions, memory becomes either read-only or write-only. **Precondition violation**: If training data lacks long-range dependencies, the model never learns to use the state effectively. |
| **Systems** | **Reinforcing loop**: Stronger gating → more precise state → better predictions → stronger gating (potentially over-confident). **Balancing loop**: State saturation → delta increases → more overwrite → less saturation. **Second-order**: If Gated DeltaNet produces different hidden representations than softmax attention, the downstream MoE router receives inputs it wasn't optimized for (the 25% softmax layers provide a "reference distribution"). **Delays**: State update is O(1) per token, but the delta correction requires computing an error signal before writing — one-token delay for correction. **Emergent vs. designed**: The memory state at position t is a designed linear recurrence, but the specific content of that state (what gets remembered vs. forgotten) is emergent from training. |
| **Temporal** | **Change over time**: As seq_len grows, the state transitions from "building initial representation" (first ~100 tokens) to "maintaining key information" (stable state) to "state saturation" at extreme lengths. **Degradation**: The gating mechanism may leak information over very long sequences (theoretical concern — empirical results at 256K are good). **Phase transition**: When the 25% softmax layers are involved, the model switches from linear recurrence (O(n)) to full attention (O(n²)) — this is a phase transition in compute cost. **Hysteresis**: The state depends on all previous tokens, so earlier incorrect predictions can corrupt later state unfixably (no "forgiving" mechanism beyond the learned gate). **Steady-state**: At inference with constant token rate, the model reaches a steady-state where new tokens are processed at constant O(1) cost. |
| **Domain-Bridge** | **BIOLOGY**: Gated DeltaNet is like a working memory system — the gate acts like the basal ganglia's thalamic gate, deciding what enters working memory and what is discarded. The delta rule is like synaptic plasticity (Hebbian learning's error-driven variant). **PHYSICS**: The state vector is analogous to a physical system's phase space — the gate determines damping (how much past state survives), the delta is the forcing function. Conservation law: information is neither created nor destroyed, only transformed between state and delta. **ENGINEERING**: Redundancy = 0 (no backup mechanism if gate collapses). Safety factor = moderate (25% softmax layers provide a fallback). The delta correction resembles a PID controller's derivative term. **MATHEMATICS**: The recurrence is a linear dynamical system — its stability depends on the spectral norm of the update matrix. The gating vector is a diagonal contraction mapping. **SOCIOLOGY**: Information flow resembles a bureaucracy — the gate is the manager deciding what information passes through, the delta is the correction based on new evidence. |
| **Paradoxical** | **Self-reference**: If the Gated DeltaNet applies to its own architecture description (a text about itself), would it remember the context? Paradox: The mechanism designed to overcome attention's O(n²) is itself limited by its own state capacity. **Gödel sentence**: "This model cannot remember the token that would make its gate fail." — a self-referential statement about the model's own limitations. **Limit behavior**: At seq_len → ∞, the model must forget everything that came before or saturate — perfect memory is impossible. The paradox of Gated DeltaNet is that it solves O(n²) by accepting O(1) memory, but O(1) memory cannot perfectly represent O(n) information. **Self-undermining**: If the model becomes too good at gating, it filters out information it might need later — optimal gating for immediate prediction may be suboptimal for long-range recall. **Perfect implementation**: A perfect Gated DeltaNet would have infinite state capacity (impossible) or perfect forgetting with perfect recall on demand (contradictory requirements). |

**Weaknesses Consolidated**:
1. **State saturation at extreme lengths** — O(1) state cannot capture all information from O(n) input (fundamental)
2. **Gate collapse failure mode** — if gating vector converges to 0 or 1 for all dimensions, all downstream quality degrades (critical)
3. **No recovery from corrupted state** — earlier errors propagate forward irreversibly (major)
4. **Delta correction lag** — one-token delay between input and correction (minor)
5. **Untested at true frontier scale** — Gated DeltaNet is ~1 year old; scaling laws at >1T params unknown (major)

**Key Assumptions Exposed**:
- Assumes the compressed state `d × d` is sufficient for all recall needs
- Assumes the gating mechanism learns optimal forgetting patterns
- Assumes 25% softmax layers can compensate for any linear attention failures
- Assumes no adversarial sequences can force gate collapse

---

## 2. A_D1: Compressed Sparse Attention (DeepSeek V4)

| Lens | Findings |
|------|----------|
| **Adversarial** | **Cheapest attack**: Craft a sequence where the Lightning Indexer's top-k (1024) misses a critical token buried in the middle of a compressed block. **Extreme values**: At 1M context with k=1024, the indexer selects only 1024/250000 = 0.4% of compressed entries — what if the answer requires information from entries ranked 1025-2000? **Stress failure**: If the indexer's ReLU scoring saturates (all scores ≈ equal), top-k selection becomes effectively random. **Single point of failure**: The Lightning Indexer — if it fails to identify the correct KV entries, CSA produces wrong results regardless of the attention quality. **Precondition violation**: If the compression weighting `S_a, S_b` is poorly learned (rare token types that didn't appear in training), the compressed representation loses information. |
| **Systems** | **Reinforcing loop**: Better indexer → better sparse selection → better attention → better model outputs → more training data → better indexer. **Balancing loop**: Top-k aggressively selected → missed entries lose gradient signal → indexer stops considering them → selection gets even narrower. **Second-order**: HCA's 128:1 compression provides a "safety net" for information missed by CSA's top-k — but at 128:1, is the safety net reliable? **Delays**: The indexer operates on compressed keys, which depend on earlier compression weights — a two-step delay between input and attention. **Emergent vs. designed**: The indexer's top-k selection criteria are learned, but the compression rate (4:1) and top-k count (1024/512) are designed hyperparameters — what if the optimal top-k is input-dependent? |
| **Temporal** | **Change over time**: At short context (<1024 tokens after compression), CSA degenerates to full attention over all entries (top-k > available). At long context (>100K), CSA operates in full sparse regime. **Degradation**: The 4:1 compression loses fine-grained positional information — the model cannot distinguish which of the 4 tokens within a compressed block contributed what. **Phase transition**: Adding SWA (n=128) creates a phase transition at token 128 — before 128, SWA covers all tokens; after, SWA covers only the local window. **Hysteresis**: Once a token is compressed into a block, the original representation is lost — no decompression path. **Steady-state**: At inference with long context, the system reaches steady-state where each step selects from ~1024 compressed KV entries + ~128 SWA entries. |
| **Domain-Bridge** | **BIOLOGY**: CSA is like the visual system's saccadic attention — the indexer acts as the superior colliculus directing "gaze" to salient regions, while HCA provides peripheral vision (low-resolution global context). **PHYSICS**: Compression is like downsampling in signal processing — at 4:1, the Nyquist frequency drops, and high-frequency information (fine token position) is aliased. **ENGINEERING**: The indexer+compression system resembles a cache hierarchy — L1 (SWA), L2 (CSA), L3 (HCA) with decreasing resolution but increasing coverage. **MATHEMATICS**: The compression is a projection onto a lower-dimensional subspace. The Johnston-McGraw sparsity principle: domain knowledge is required to design good sparsity patterns — DeepSeek's indexer learns this domain knowledge through training. **SOCIOLOGY**: The indexer is like a news editor — it reads all stories (compressed tokens) and selects which ones deserve full attention. The HCA is the "brief summary" column. |
| **Paradoxical** | **Self-reference**: If CSA is applied to its own architecture description (a text about CSA), the indexer must decide whether the compressed version of "Compressed Sparse Attention" is worth selecting. Paradox: The mechanism designed to solve attention's O(n²) problem suffers from its own selection problem — a recursive sparse attention deficit. **Gödel sentence**: "This model cannot find the token in the compressed block that the indexer chose to ignore." — the fundamental limitation of sparsity. **Limit behavior**: As compression rate → ∞, all information is lost (HCA at 128:1). As compression rate → 1, CSA degenerates to standard attention. There is an optimal intermediate rate that DeepSeek empirically found. **Self-undermining**: The more effective the compression (higher ratio), the more information is irreversibly lost. The more effective the indexer, the more information outside its top-k is lost. Efficiency and recall are fundamentally antagonistic. **Perfect implementation**: Perfect CSA would require infinite compression (zero memory) with perfect recall (all information preserved) — a physical impossibility (Landauer's principle: erasing information generates heat). |

**Weaknesses Consolidated**:
1. **Indexer failure mode** — missed relevant KV entries → incorrect attention (critical)
2. **Irreversible compression loss** — 4:1 + 128:1 means information is permanently discarded (fundamental)
3. **Fine-grained position ambiguity** — cannot distinguish tokens within a compressed block (major)
4. **Top-k is fixed** — 1024/512 regardless of input length or complexity (major)
5. **Three-tier complexity** — CSA + HCA + SWA creates debugging and optimization difficulty (major)

**Key Assumptions Exposed**:
- Assumes top-k (1024/512) is sufficient for any query at any context length
- Assumes compression weights are well-learned for all token types
- Assumes the indexer's learned scoring function generalizes to unseen input distributions
- Assumes overlapping windows (2m tokens per compressed entry) compensates for compression loss
- Assumes the three tiers (SWA/CSA/HCA) partition attention needs without gaps or overlaps

---

## 3. A_K1: Multi-Head Latent Attention (Kimi K2.6)

| Lens | Findings |
|------|----------|
| **Adversarial** | **Cheapest attack**: Maximize the rank of the KV matrix to exceed 512, causing the low-rank projection to lose information. **Extreme values**: At 256K context, the accumulated latent state spans extremely diverse information — can rank 512 capture it all? **Stress failure**: If the KV distribution shifts from training distribution (e.g., adversarial prompts), the low-rank basis may not capture the new directions. **Single point of failure**: The rank-512 projection matrix — if it's ill-conditioned (high condition number), small changes in input produce large changes in output. **Precondition violation**: Assumes that keys and values live on a low-dimensional manifold — if the training data doesn't enforce this, the projection loses information. |
| **Systems** | **Reinforcing loop**: Low-rank compression → smaller KV cache → larger feasible context → more diverse KV content → harder for low-rank to capture → lossy compression → quality degradation. **Balancing loop**: Higher rank → better capture → larger cache → less context → training fights the low-rank constraint. **Second-order**: The KV dimension reduction forces attention heads to compete for representation capacity — a head that needs unique KV content may find its signal compressed into a shared dimension. **Delays**: No inference delay (MLA is purely architectural, not recurrent). **Emergent vs. designed**: The rank-512 latent space is designed, but the specific directions captured within it are emergent. |
| **Temporal** | **Change over time**: MLA is feed-forward per token — no temporal state. Each token's KV is independently compressed. **Degradation**: The KV compression quality depends on the token's position in the manifold — some tokens are well-represented by rank-512, others are not. No temporal degradation. **Steady-state**: From token 1 to the end, compression quality is theoretically uniform (but practically depends on whether later tokens' KV lie in the same manifold as earlier tokens). |
| **Domain-Bridge** | **BIOLOGY**: MLA is like the brain's sparse coding in the hippocampus — the low-rank latent space is analogous to a basis set of concepts, and each token's KV is a sparse combination. **PHYSICS**: The projection from full dimension d to rank r is a compression analogous to the Rankine-Hugoniot jump conditions — a projection onto a lower-dimensional manifold that preserves essential information. **ENGINEERING**: Rank-512 compression in a 7168-dimensional space = 7% compression ratio, analogous to JPEG's DCT compression at high quality. **MATHEMATICS**: The Eckart-Young theorem guarantees that the best rank-r approximation of a matrix is its truncated SVD. MLA learns to approximate this via a neural projection. The 512 latent dimensions = the "informative subspace." **SOCIOLOGY**: MLA is like a parliamentary committee system — 512 committee members (latent dimensions) summarize the views of 7168 representatives (full space). |
| **Paradoxical** | **Self-reference**: If MLA is applied to its own KV representation (compressing the compressed), the information loss compounds — a Russian doll of approximation. **Gödel sentence**: "This compression cannot encode the information that would require the 513th latent dimension to express." — the hard limit of low-rank approximation. **Limit behavior**: As rank → d (full space), MLA becomes standard MHA (no compression, no benefit). As rank → 0, all information is lost. Rank-512 is the empirically determined sweet spot. **Self-undermining**: MLA's success depends on the assumption that KVs have low intrinsic dimensionality — but if training creates high-dimensional KVs (which it might, since the model also learns the KVs), MLA is designing the very manifold that constrains it. **Perfect implementation**: Perfect MLA would require rank = intrinsic dimension of the full KV space — but this intrinsic dimension is unknown and may vary by layer, head, and input. |

**Weaknesses Consolidated**:
1. **Fixed rank is fragile** — rank 512 may be insufficient for some inputs (major)
2. **Dependency on DeepSeek V3 codebase** — not independently invented; potential licensing/compatibility issues (moderate)
3. **No temporal reuse** — each token compressed independently; no cross-token KV optimization (minor)
4. **Training-inference mismatch** — if training uses full-precision KV but inference uses compressed, distribution shift (moderate)
5. **Competition among heads** — 64 heads share the same 512-dim latent; head-specific KV signals may interfere (major)

**Key Assumptions Exposed**:
- Assumes KV representations have intrinsic dimension ≤ 512
- Assumes all 64 attention heads can share the same latent space without destructive interference
- Assumes the projection is well-conditioned for all input distributions
- Assumes the low-dimensional manifold is stable across layers (not layer-specific)

---

## 4. A_Q3: MoE with 512 experts, Top-10 (Qwen3.5)

| Lens | Findings |
|------|----------|
| **Adversarial** | **Cheapest attack**: Craft a token that activates the experts in their least-loaded combination, forcing the router to balance poorly. **Extreme values**: 512 experts with Top-10 means only 2% of experts active per token — what if the correct knowledge is in expert #511? **Stress failure**: If the router's softmax is ill-calibrated (over-confident), it may repeatedly activate the same 10 experts, leaving 502 experts undertrained. **Single point of failure**: The router — if it degrades, all downstream quality suffers. **Precondition violation**: Assumes training data covers the expertise landscape evenly — if some domains are underrepresented, their experts never specialize properly. |
| **Systems** | **Reinforcing loop (positive)**: Router selects experts → selected experts get more training → become better at their specialized domain → router selects them more (Matthew effect). **Reinforcing loop (negative)**: Unselected experts get less training → become worse → router selects them less → potential dead experts. **Balancing**: Auxiliary loss counteracts the positive loop. **Second-order**: Expert specialization creates emergent "expert territories" — if two experts learn overlapping domains, the router's choice between them is arbitrary, creating non-determinism. **Delays**: Router training lags behind expert training (router needs to learn which experts are good, but they're improving simultaneously). |
| **Temporal** | **Change over time**: Early in training, experts are undifferentiated. Mid-training, specialization emerges. Late training, experts stabilize. **Degradation**: If training data distribution shifts, previously specialized experts may become irrelevant (expert drift). **Phase transition**: At the point where auxiliary loss starts to dominate, expert selection shifts from quality-based to balance-based. |
| **Domain-Bridge** | **SOCIOLOGY**: 512 experts is like a 512-person team — with Top-10 selection, it's a committee of 10 chosen from 512 specialists. The auxiliary loss is like an HR policy ensuring everyone gets work. **BIOLOGY**: The immune system uses ~10^8 antibody types (experts) with ~10^3 expressed at any time (Top-10) — Qwen3.5's 512/10 is a drastically simpler version. **ECONOMICS**: The Gini coefficient of expert utilization measures balance — an unhealthy concentration means most capacity is wasted. |
| **Paradoxical** | **Self-reference**: The router is itself a neural network — should it use MoE too? Meta-routing. **Gödel sentence**: "The expert that would be most useful for this token cannot be selected because the auxiliary loss penalizes its selection." — balance vs. quality tension. **Self-undermining**: More experts → more specialization potential → more routing complexity → more auxiliary loss interference. |

**Weaknesses Consolidated**:
1. **Expert death** — unselected experts receive diminishing gradient signal (critical)
2. **Router as bottleneck** — entire model quality depends on router's decisions (major)
3. **Static expert count** — 512 may be too many for some layers, too few for others (moderate)
4. **Auxiliary loss interference** — forced balance may override quality-based selection (major)

---

## 5. A_D5: MoE with 384 experts, Top-6 (DeepSeek V4)

| Lens | Findings |
|------|----------|
| **Adversarial** | **Cheapest attack**: Exploit the Sqrt(Softplus) activation — push its input to the linear regime where softplus ≈ x for large x, making routing scores grow unbounded. **Extreme values**: Top-6 out of 384 = 1.6% activation, the most aggressive of the three. What if the task requires 7 distinct specializations? **Stress failure**: Hash routing for first 3 layers hardcodes expert assignments — adversarial tokens could exploit known hash collisions. |
| **Systems** | **Reinforcing**: Sqrt(Softplus) suppresses low scores and amplifies high scores compared to Softmax — creates stronger expert specialization. **Balancing**: Sequence-wise balance loss (lightweight) counteracts. **Second-order**: Hash routing creates deterministic expert groups for common tokens — this is a caching effect that may reduce effective expert diversity for frequent inputs. |
| **Temporal** | **Phase transition**: Anticipatory routing (computing routing Δt steps ahead) decouples routing from backbone updates — creates a temporal offset whose optimal Δt is unclear. |
| **Domain-Bridge** | **ENGINEERING**: Top-6 with 384 experts and hash routing is like a 6-out-of-384 fault-tolerant system where the hash determines which 6 are primary. |
| **Paradoxical** | **Self-undermining**: Hash routing improves stability but reduces flexibility — the deterministic assignment for common tokens means the model cannot dynamically reallocate expert resources. |

**Weaknesses Consolidated**:
1. **Most aggressive sparsity** — only 1.6% experts active; risk of insufficient capacity (critical)
2. **Hash routing rigidity** — deterministic = fragile to distribution shift (moderate)
3. **Anticipatory routing Δt unknown** — "insufficiently understood" (major — report's own characterization)

---

## 6. A_K3: MoE with 384 experts, Top-8 (Kimi K2.6)

| Lens | Findings |
|------|----------|
| **Adversarial** | **Cheapest attack**: Exploit noaux_tc routing (no auxiliary loss) — the model will naturally concentrate expert selection, creating expert imbalance. **Stress failure**: Sigmoid scoring with norm_topk_prob creates soft probabilities — but without balance loss, some experts may receive 10× more training than others. |
| **Systems** | **Reinforcing (unchecked)**: No auxiliary loss means the Matthew effect (rich-get-richer expert selection) is unopposed — some experts dominate, others starve. **Balancing**: None explicit — only implicit gradient pressure from the shared expert. **Second-order**: The shared expert compensates by always being active — if it learns to handle the imbalance, the routed experts' specialization suffers. |
| **Temporal** | **Expert desertification**: Over long training runs, without auxiliary loss, expert imbalance tends to increase monotonically. |
| **Domain-Bridge** | **ECONOMICS**: No auxiliary loss = free market for expert selection. Efficient in the short term, prone to monopoly in the long term. |
| **Paradoxical** | **No load balancing paradox**: The design trusts that sigmoid routing will self-balance through training dynamics — but this assumes the optimization landscape naturally encourages balanced utilization, which is not theoretically guaranteed. |

**Weaknesses Consolidated**:
1. **Expert imbalance (unopposed)** — no auxiliary loss means Matthew effect unchecked (critical)
2. **No explicit load balancing** — relies on implicit training dynamics (major)
3. **Shared expert may over-compensate** — learns to handle all imbalance, reducing routed expert quality (moderate)

---

## 7. C2_Q1: Qwen3.5 Hybrid Attention (Composite)

| Lens | Findings |
|------|----------|
| **Adversarial** | **Cheapest attack**: Exploit the 75/25 split — craft sequences where the 25% softmax layers are too sparse to correct Gated DeltaNet errors. **Stress**: At which sequence length do the 25% softmax layers become the bottleneck? |
| **Systems** | **Feedback**: Gated DeltaNet layers pass to softmax layers — any errors in the linear recurrence must be corrected by the sparse softmax layers. The 25% ratio creates a 3:1 correction burden. |
| **Temporal** | **Ratio lock-in**: The 75/25 split is fixed at training time — cannot be adjusted per-task at inference. |
| **Paradoxical** | The hybrid architecture admits that linear attention alone is insufficient (hence 25% softmax), but committing 25% of layers to O(n²) means the best-case complexity is O(0.75n + 0.25n²). |

**Weaknesses Consolidated**:
1. **Still O(n²) for 25% of compute** — doesn't fully escape quadratic cost (major)
2. **75/25 ratio is architectural "guess"** — not theoretically derived (moderate)
3. **Softmax layers become bottleneck at very long context** (major)

---

## 8. C2_D1: DeepSeek V4 Hybrid Attention Stack (Composite)

| Lens | Findings |
|------|----------|
| **Adversarial** | **Cheapest attack**: Exploit the gap between SWA (n=128), CSA (top-k=1024 compressed entries), and HCA (128:1) — a sequence with critical information in the "gap band" between 128 and the nearest compressed-then-selected entry. **Stress**: What happens when the Lightning Indexer selects different entries for different heads within the same layer? Inter-head incoherence. |
| **Systems** | **Three-tier coverage**: SWA for local, CSA for mid-range, HCA for global — but is there a "dead zone" between tiers? If CSA misses something and HCA's 128:1 compression washes it out, it's lost. |
| **Temporal** | The three tiers create a temporal priority: SWA (immediate), CSA (selected past/context), HCA (distant past/context). |
| **Paradoxical** | The most architecturally complex system (CSA+HCA+SWA+mHC+indexer) was designed to achieve simplicity (O(n) attention cost). Complexity for simplicity. |

**Weaknesses Consolidated**:
1. **Three-tier coverage gap** — possibility of missing information between SWA/CSA/HCA regimes (critical)
2. **Massive architectural surface area** — 6+ interacting components increase failure modes (major)
3. **Cross-tier interference** — SWA/CSA/HCA outputs are combined; does the model learn to weight them correctly? (major)

---

## 9. C2_K1: Kimi K2.6 Latent Attention System (Composite)

| Lens | Findings |
|------|----------|
| **Adversarial** | **Cheapest attack**: Exploit the fixed rank-512 — feed inputs whose KV structure has intrinsic dimension > 512. **Stress**: At 256K context, the attention heads' KV diversity may exceed the latent space capacity. |
| **Systems** | **Head competition**: 64 heads share 512 latent dimensions — if heads 1-32 use all 512 dimensions for their KV, heads 33-64 are forced into a subspace. |
| **Temporal** | MLA has no temporal dependency (feed-forward) — but the 64× YaRN scaling creates a temporal dependency in position encoding. |
| **Paradoxical** | MLA achieves MHA quality with KV cache of size rank rather than dimension — but MHA at 1T params needs the compression, and the compression constrains MHA quality. Circular dependency. |

**Weaknesses Consolidated**:
1. **Head interference in shared latent space** — 64 heads competing for 512 dims (major)
2. **Fixed rank limitation** — cannot adapt compression to input complexity (major)
3. **Borrowed architecture** — dependency on DeepSeek V3 code; no independent innovation path (moderate)

---

## 10. P: Peak — Contemporary LLM Architecture Design Space

| Lens | Findings |
|------|----------|
| **Adversarial** | **Cheapest attack on the design space**: A single breakthrough (e.g., truly linear attention that matches full softmax quality) would invalidate the architectural differentiation of all three models simultaneously. **The threat**: All three are interim solutions awaiting a "unified theory of efficient attention" — if and when it arrives, the complex hybrid architectures all become historical footnotes. |
| **Systems** | **Reinforcing loop**: Better architectures → better models → more applications → more funding → more research → better architectures. **Balancing loop**: Rising inference demand → pressure for efficiency → architectural innovation. **The meta-pattern**: All three architectures converge on MoE (~30:1 total:active ratio) but diverge on attention — suggesting the community agrees on MoE but has not converged on attention alternatives. **Second-order**: The complexity of DeepSeek V4's architecture may be a leading indicator — as models scale, architectural complexity may grow superlinearly with parameter count. |
| **Temporal** | **Historical trajectory**: 2023 (dense Transformers) → 2024 (GQA/MoE adoption) → 2025 (MLA, Gated DeltaNet) → 2026 (CSA+HCA, hybrid linear+softmax). Each year increases architectural complexity. **Prediction**: 2027 will see either (a) convergence on a standard efficient attention mechanism, or (b) further diversification into task-specific architectures. |
| **Domain-Bridge** | **BIOLOGY (evolution)**: The three architectures are like three species occupying the same ecological niche (efficient LLM inference) but with different adaptations (attention replacement, compression, latent-space tricks). They are undergoing competitive speciation — whichever adaptation proves most fit will dominate. **PHYSICS (thermodynamics)**: The Second Law of AI — information entropy in the attention matrix tends to increase, requiring energy (compute) to decrease. All three architectures are different "heat engines" for doing this work more efficiently. |
| **Paradoxical** | **Grand paradox**: The pursuit of efficiency (fewer FLOPs, smaller KV cache) has produced architectures that are significantly more complex than the original Transformer. The "simple" Transformer has been replaced by systems with 6+ interacting mechanisms (Gated DeltaNet + Gated Attention + MoE + shared expert + FP8 + MTP for Qwen3.5 alone). Are we making models more efficient at inference but less efficient at design and implementation? **Gödel sentence of architecture design**: "No architecture can be simultaneously optimal in compute efficiency, memory efficiency, implementation simplicity, and generalization quality." |

**Weaknesses Consolidated**:
1. **Architectural complexity explosion** — each generation adds mechanisms rather than removing them (critical)
2. **No convergence** — three fundamentally different attention strategies suggests the field hasn't found the answer (major)
3. **Implementation cost** — each architecture requires custom inference stacks, slowing adoption (major)
4. **Diminishing returns** — the gap between Qwen3.5, DeepSeek V4, and Kimi K2.6 on most benchmarks is small (<5%) despite radically different architectures (major)
5. **All are interim solutions** — none fully solves the attention efficiency problem; all are engineering compromises (fundamental)

---
