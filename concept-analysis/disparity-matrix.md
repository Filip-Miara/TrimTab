# Disparity Matrix: Detection & Reconciliation

---

## DISP-1: Linear Recurrence vs. Sparse Attention (A_Q1 vs. A_D1)

| Field | Value |
|-------|-------|
| **Node A** | A_Q1 (Gated DeltaNet — linear recurrence, O(n) state, gated memory) |
| **Node B** | A_D1 (CSA — compression + sparsity, top-k selection, indexer) |
| **A_is_synthetic** | false |
| **B_is_synthetic** | false |
| **Disparity Type** | **fundamental assumption_clash** |
| **Description** | These two mechanisms have diametrically opposed core assumptions about how to handle long context. Gated DeltaNet assumes a compressed state `d × d` updated via recurrence captures all needed information. CSA assumes the answer is always in a small subset of tokens that can be identified via an indexer. One is "remember everything in compressed form"; the other is "find the right few tokens." These imply completely different allocation strategies and failure modes. They cannot both be correct at all problem scales — there exists a crossover point where one dominates. |
| **Classification** | fundamental |
| **Reconciliation** | **Resolution**: Use Gated DeltaNet for local/medium-range dependencies (where recurrence excels) and CSA for long-range sparse retrieval. This is effectively what SYN-8 proposes — but at the cost of massive architectural complexity. **Mechanism**: separation (partition by dependency range). **Resolved at level**: C3 (requires composite-level reconciliation, not atomic). |
| **If irreconcilable**: N/A — reconciled via separation. |

---

## DISP-2: Fixed Top-K vs. Adaptive Computation (A_D1 vs. SYN-2)

| Field | Value |
|-------|-------|
| **Node A** | A_D1 (CSA fixed top-k = 1024) |
| **Node B** | SYN-2 (CSA with adaptive top-k) |
| **A_is_synthetic** | false |
| **B_is_synthetic** | true |
| **Disparity Type** | **operational_incompatibility** |
| **Description** | The fixed top-k in A_D1 enables predictable compute budgets (constant-time sparse attention), which is critical for batch serving and hardware utilization. SYN-2's variable top-k would produce unpredictable batch times and potential latency spikes for complex queries. |
| **Classification** | structural |
| **Reconciliation** | **Resolution**: Use fixed top-k for serving (the actual deployed model) but train with adaptive top-k masking so the model learns to work with variable k even though inference forces it to a fixed k. **Mechanism**: synthesis (train adaptive, deploy fixed). **Resolved at level**: 2 (C2). |
| **Recursive scrutiny of reconciliation**: The train-adaptive, deploy-fixed solution creates a training-inference mismatch — the model may learn to rely on being able to use more entries, then fail when k is capped at inference. Mitigation: guarantee k_deploy >= k_train_max. |

---

## DISP-3: 512 Experts Top-10 vs. 384 Experts Top-6 (A_Q3 vs. A_D5)

| Field | Value |
|-------|-------|
| **Node A** | A_Q3 (Qwen3.5: 512 experts, Top-10, 2% activation) |
| **Node B** | A_D5 (DeepSeek V4: 384 experts, Top-6, 1.6% activation) |
| **A_is_synthetic** | false |
| **B_is_synthetic** | false |
| **Disparity Type** | **logical_contradiction** |
| **Description** | Both models achieved top-tier quality with MoE, but with different activation ratios (2% vs 1.6%). They cannot both be optimal — either more experts per token is better (Qwen's approach) or more aggressive sparsity is better (DeepSeek's approach). The 0.4% difference may seem small, but at 1.6T vs 397B total params, it represents dramatically different design philosophies. |
| **Classification** | creative (the tension suggests an unknown optimum) |
| **Reconciliation** | **Resolution**: The optimum depends on the attention mechanism. Qwen3.5's Gated DeltaNet does less per-token compute (O(n)), so it can afford 10 active experts. DeepSeek V4's CSA+HCA does more per-token compute (lightning indexer + sparse attention + SWA), so it needs fewer active experts to stay within budget. The expert count is not independently variable — it's traded off against attention cost. **Mechanism**: contextualization (expert count optimized jointly with attention cost). **Resolved at level**: 3 (C3 — requires system-level view). |
| **Recursive scrutiny**: This reconciliation implies that MoE architecture decisions cannot be evaluated independently of the attention mechanism. A corollary: comparing only MoE configurations across models is meaningless without accounting for attention cost. |

---

## DISP-4: Auxiliary Loss vs. No Auxiliary Loss (A_Q3 vs. A_K10)

| Field | Value |
|-------|-------|
| **Node A** | A_Q3 (Qwen3.5: top-k softmax + auxiliary loss for load balancing) |
| **Node B** | A_K10 (Kimi K2.6: noaux_tc sigmoid routing, no balance loss) |
| **A_is_synthetic** | false |
| **B_is_synthetic** | false |
| **Disparity Type** | **assumption_clash** |
| **Description** | Qwen3.5 assumes explicit load balancing is necessary (auxiliary loss coefficient 0.001). Kimi K2.6 assumes no balance loss is necessary (relies on sigmoid gating dynamics). DeepSeek V4 takes a middle path (sequence-wise balance loss, very lightweight). At least two of these are wrong about what's needed for stable MoE training. |
| **Classification** | fundamental |
| **Reconciliation** | **Bound**: This tension cannot be resolved theoretically — it is an empirical question that depends on architecture scale, training data, and optimization hyperparameters. There is no known theorem proving either approach is universally correct. **Implications**: The optimal load balancing strategy is architecture-specific and must be discovered empirically. Any new MoE architecture must treat load balancing as a tunable parameter, not a fixed design choice. |

---

## DISP-5: Full MHA vs. GQA (A_K2 vs. Standard Transformer Baseline)

| Field | Value |
|-------|-------|
| **Node A** | A_K2 (Kimi K2.6: 64 KV heads, full MHA) |
| **Node B** | Standard Transformer baseline: GQA (2-8 KV heads) |
| **A_is_synthetic** | false |
| **B_is_synthetic** | false |
| **Disparity Type** | **logical_contradiction** |
| **Description** | The field largely converged on GQA as the standard for large models (Llama 3, Qwen3.5, GPT-4 all use GQA). Kimi K2.6 uses full MHA (64 heads = 64 KV heads) — against the consensus. MLA compression makes this feasible, but it's a bet that MLA + MHA > GQA. |
| **Classification** | creative |
| **Reconciliation** | **Resolution**: MLA + MHA provides expressivity of full multi-head attention at ~15% of standard KV cache cost. This is strictly better than GQA if the MLA compression is high-quality, because GQA reduces expressivity while MLA reduces cache. **Mechanism**: synthesis (MLA's compression removes GQA's raison d'être). **Recursive note**: This implies GQA is an artifact of KV cache constraints, not an optimal architectural choice — if efficient KV compression (MLA or Gated DeltaNet) is available, full MHA becomes optimal again. |

---

## DISP-6: Per-Token vs. Per-Conversation Thinking Mode (A_Q9 vs. A_D9)

| Field | Value |
|-------|-------|
| **Node A** | A_Q9 (Qwen3.5: per-token enable_thinking toggle) |
| **Node B** | A_D9 (DeepSeek V4: per-conversation 3-mode selection via system prompt) |
| **A_is_synthetic** | false |
| **B_is_synthetic** | false |
| **Disparity Type** | **abstraction_mismatch** |
| **Description** | Qwen3.5 allows switching thinking on/off per token within a conversation (fine-grained but only 2 modes). DeepSeek V4 requires setting the mode at conversation start (3 modes, coarse-grained). Neither allows per-token, 3-mode control. |
| **Classification** | structural |
| **Reconciliation** | **Resolution**: SYN-10 proposes combining both — per-token, 3-mode control. Feasibility is the challenge, but the concept is straightforward: add a lightweight "effort router" that predicts required budget per token. **Mechanism**: synthesis. **Resolved at level**: 2 (new composite combining A_Q9 + A_D9). |

---

## DISP-7: FP8 Training vs. FP4 Training vs. BF16 Training (A_Q6 vs. A_D7 vs. BF16 baseline)

| Field | Value |
|-------|-------|
| **Node A** | A_Q6 (Qwen3.5: FP8 training for all parameters) |
| **Node B** | A_D7 (DeepSeek V4: FP4 experts + FP8 rest) |
| **Node C** | Standard baseline: BF16/FP16 training |
| **A_is_synthetic** | false |
| **B_is_synthetic** | false |
| **C_is_synthetic** | false |
| **Disparity Type** | **assumption_clash** (3-way) |
| **Description** | Three architectures take three different positions on precision: (1) BF16 — maximal precision, highest compute cost; (2) FP8 — good precision/compute tradeoff; (3) FP4 experts — extreme push for efficiency. The disparity is whether FP4 training quality degradation is acceptable (DeepSeek says yes, others implicitly say no by choosing higher precision). |
| **Classification** | fundamental |
| **Reconciliation** | **Bound**: This is an empirical "what quality loss does FP4 cause" question, and the answer depends on the architecture, training data size, and tasks. DeepSeek V4's 80.6% SWE-bench suggests the loss is small (or compensated by other architectural innovations). But until all three models train at all three precisions on the same data, the comparison is apples-to-oranges. **Implications**: Precision choice cannot be independently evaluated — it trades off against architecture capacity, expert count, and training compute. |

---

## DISP-8: Native Multimodal vs. Text-Only (A_Q5 vs. DeepSeek V4 [no multimodal])

| Field | Value |
|-------|-------|
| **Node A** | A_Q5 (Qwen3.5: early-fusion multimodal from training start) |
| **Node B** | DeepSeek V4 (no multimodal support) |
| **A_is_synthetic** | false |
| **B_is_synthetic** | false |
| **Disparity Type** | **goal_conflict** |
| **Description** | DeepSeek V4 achieves the best open-weight SWE-bench score but is text-only. Qwen3.5 leads in multimodal benchmarks but trails in coding. The market demands both. The disparity is a product of resource allocation: including vision training consumes compute that could be spent on coding ability. |
| **Classification** | creative |
| **Reconciliation** | **Resolution**: SYN-12 proposes adding vision to DeepSeek V4. The cost is real, but the benefit is clear — a model with both frontier coding and multimodal ability. **Mechanism**: synthesis (add vision training to coding-optimized model). **Resolved at level**: 3 (full model level). |

---

## DISP-9: 75% Linear Attention vs. 100% Compressed+Sparse (Qwen3.5 vs. DeepSeek V4 strategy)

| Field | Value |
|-------|-------|
| **Node A** | C2_Q1 (Qwen3.5: 75% linear + 25% softmax) |
| **Node B** | C2_D1 (DeepSeek V4: 100% CSA+HCA+SWA, no softmax) |
| **A_is_synthetic** | false |
| **B_is_synthetic** | false |
| **Disparity Type** | **fundamental — paradigm_clash** |
| **Description** | Qwen3.5 keeps 25% softmax attention because "linear attention alone is insufficient for retrieval." DeepSeek V4 has zero softmax attention and relies entirely on compression+sparsity. One of these claims must be wrong (or they define "retrieval" differently). This is the deepest architectural disparity between the three models. |
| **Classification** | fundamental |
| **Reconciliation** | **Bound**: This disparity cannot be resolved without a controlled experiment comparing both architectures on identical data and compute budgets. It's a testable hypothesis: is softmax attention necessary for frontier-quality retrieval? DeepSeek V4's benchmark performance suggests "no" (or that its compression+sparsity substitutes for softmax). Qwen3.5's performance suggests "yes" (or that Gated DeltaNet specifically needs softmax supplementation). **Implications**: Until such an experiment is done, the necessity of softmax attention remains an open question. This is THE open research question for efficient attention architectures. |

---

## DISP-10: One-Shot-Fused Multimodal vs. Two-Stage Multimodal (A_Q5 vs. A_K6)

| Field | Value |
|-------|-------|
| **Node A** | A_Q5 (Qwen3.5: early-fusion, joint training from start) |
| **Node B** | A_K6 (Kimi K2.6: MoonViT encoder + LLM, two-stage training) |
| **A_is_synthetic** | false |
| **B_is_synthetic** | false |
| **Disparity Type** | **structural** |
| **Description** | Qwen3.5 trains vision+text jointly from the beginning. Kimi K2.6 follows a two-stage process (train text-only base, add vision encoder, continue training). These imply different assumptions about whether vision and text need the same underlying representations. |
| **Classification** | creative |
| **Reconciliation** | **Resolution**: The two-stage approach is more practical (vision encoder can be swapped/upgraded without retraining the LLM) and is how most labs do it. The one-shot approach is more principled (representations are shared from the start). Empirical results favor one-shot (Qwen3.5's MathVision 88.6 > Kimi's 87.4) but the margin is small. **Mechanism**: contextualization (the benefit of one-shot vs two-stage depends on compute budget and whether independent vision encoder upgrades are needed). |

---

## DISP-11: SYN-4 (Gated DeltaNet + MLA, all layers sub-quadratic) vs. DISP-9's Unresolved Question

| Field | Value |
|-------|-------|
| **Node A** | SYN-4 (replace Qwen3.5's 25% softmax with MLA — all layers linear/sub-quadratic) |
| **Node B** | DISP-9 (open question: is softmax necessary for frontier retrieval?) |
| **A_is_synthetic** | true |
| **B_is_synthetic** | false (it's a disparity, not a concept) |
| **Disparity Type** | **operational_incompatibility** |
| **Description** | SYN-4's feasibility depends on whether softmax attention is genuinely necessary. If DISP-9 resolves as "softmax is necessary," SYN-4 fails. If DISP-9 resolves as "softmax is unnecessary," SYN-4 becomes a high-value test. |
| **Classification** | structural |
| **Reconciliation** | **Resolution**: SYN-4 should be the experiment that resolves DISP-9. Build SYN-4, train it at moderate scale (7B-13B active), and measure long-range retrieval quality. The result directly answers whether softmax is necessary. **Mechanism**: experimental validation. Resolved through action, not reasoning. |
| **Recursive scrutiny**: This is a meta-level reconciliation — using a synthetic variant as an experiment to resolve a fundamental disparity. It's the scientific method applied to architecture design. |
