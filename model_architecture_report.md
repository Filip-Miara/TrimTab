# Comparative Architectural Analysis: Qwen 3.5, DeepSeek V4, Kimi K2.6

**Date**: 2026-06-13
**Scope**: Exhaustive structural, methodological & qualitative comparison against standard Transformer baseline (Llama 3–style decoder-only Softmax-attention + dense FFN).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Standard Transformer Baseline](#2-standard-transformer-baseline)
3. [Qwen 3.5](#3-qwen-35)
   - 3.1 Architecture Overview
   - 3.2 Gated Delta Network (Linear Attention)
   - 3.3 Gated Attention (Full-Attention Layers)
   - 3.4 MoE Configuration
   - 3.5 Native Multimodal (Early Fusion)
   - 3.6 FP8 Training Pipeline
   - 3.7 Vocabulary & Multilingual: 201 Languages
   - 3.8 Thinking / Non-Thinking Modes
   - 3.9 Agentic Capabilities & MCP
   - 3.10 Pros / Cons / Drawbacks
4. [DeepSeek V4](#4-deepseek-v4)
   - 4.1 Architecture Overview
   - 4.2 Hybrid Attention: CSA + HCA
   - 4.3 KV Cache Reduction Mechanics
   - 4.4 MoE Configuration
   - 4.5 Manifold-Constrained Hyper-Connections (mHC)
   - 4.6 FP4 + FP8 Mixed-Precision Training
   - 4.7 Muon Optimizer
   - 4.8 Reasoning Modes (Non-Think / Think High / Think Max)
   - 4.9 Engram Conditional Memory (Separate Research)
   - 4.10 V4-Pro vs V4-Flash
   - 4.11 Pros / Cons / Drawbacks
5. [Kimi K2.6](#5-kimi-k26)
   - 5.1 Architecture Overview
   - 5.2 Multi-Head Latent Attention (MLA)
   - 5.3 MoE Configuration
   - 5.4 YaRN RoPE Scaling (64×)
   - 5.5 MoonViT Vision Encoder
   - 5.6 Thinking / Instant Modes
   - 5.7 Agent Swarm Orchestration
   - 5.8 Native INT4 Quantization
   - 5.9 K2.6 vs K2.7 Code
   - 5.10 Pros / Cons / Drawbacks
6. [Cross-Cutting Comparison](#6-cross-cutting-comparison)
   - 6.1 Attention Mechanisms
   - 6.2 MoE Design Philosophy
   - 6.3 KV Cache & Long-Context Strategy
   - 6.4 Training Precision
   - 6.5 Multimodal Approach
   - 6.6 Inference Modes
   - 6.7 Openness & Licensing
7. [Summary Table](#7-summary-table)

---

## 1. Executive Summary

| Dimension | Qwen 3.5 | DeepSeek V4 | Kimi K2.6 |
|-----------|----------|-------------|-----------|
| **Org** | Alibaba (Qwen Team) | DeepSeek (High-Flyer) | Moonshot AI |
| **Release** | Feb 16, 2026 | Apr 24, 2026 | ~May 19, 2026 |
| **Flagship Total Params** | 397B (A17B active) | 1.6T (49B active) | 1T (32B active) |
| **Primary Innovation** | Gated DeltaNet (linear attention) 75% layers | CSA+HCA hybrid sparse compression attention | MLA (Multi-Head Latent Attention) from DeepSeek V3 |
| **Licensing** | Apache 2.0 | MIT | Modified MIT |
| **Unique differentiator** | Natively multimodal, 201 languages, cheapest API | Extreme long-context cost efficiency, FP4 training | Agent Swarm (300 sub-agents), competitive coding |

---

## 2. Standard Transformer Baseline

Reference architecture for comparison:

- **Attention**: Full softmax self-attention, GQA or MHA, O(n²) compute, O(n) KV cache per layer
- **FFN**: Dense SwiGLU or ReLU, all parameters active per token
- **Position encoding**: RoPE (full-dimensional)
- **Residual connections**: Simple addition, Pre-LN or Post-LN
- **Precision**: BF16/FP16 training, often FP8 inference
- **Context**: 128K typical, degrades with length
- **Multimodal**: Separate VL model or adapter (CLIP + projector)
- **Optimizer**: AdamW

---

## 3. Qwen 3.5

### 3.1 Architecture Overview

- **Architecture**: Hybrid of Gated DeltaNet (linear recurrence) + Gated Attention (softmax) + MoE
- **Flagship**: Qwen3.5-397B-A17B (397B total, 17B active, 60 layers)
- **Layer layout**: `[GatedDeltaNet→MoE] × 3 → [GatedAttention→MoE] × 1`, repeated 15× (= 60 layers)
  - **75% Gated DeltaNet** (linear attention, O(n))
  - **25% Gated Attention** (full softmax attention, O(n²))
- **Context window**: 262K native, extendable to 1M via YaRN
- **Vocabulary**: 248,320 tokens (expanded from ~150K in Qwen3)
- **License**: Apache 2.0 (fully open-weight)

### 3.2 Gated Delta Network (Linear Attention)

**Origin**: Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv:2412.06464, ICLR 2025)

**Core Mechanism** — a linear recurrence with two key components:

1. **Gating** (from Mamba2/S4): A learned gating vector `g_t` controls how much of the previous state `s_{t-1}` is retained vs. overwritten by new input `x_t`. Enables **rapid memory erasure** — the model can selectively forget old information.

2. **Delta Update Rule**: Each new token computes an error signal (delta) against the current memory state and only writes the residual, enabling **precise memory modifications** rather than blanket overwriting.

```
s_t = g_t ⊙ s_{t-1} + (1 - g_t) ⊙ δ_t    (gating + delta correction)
```

- **Complexity**: O(n) in sequence length (vs. O(n²) for standard attention)
- **Memory**: Compressed state `d × d` per layer (vs. full KV cache `n × d`)
- **Implementation details**:
  - Linear Attention Heads (Value): 64 (397B), 32 (35B)
  - Linear Attention Heads (Key): 16
  - Head dimension: 128
  - Conv kernel dimension: 4 (1D convolution before SSM)
  - Activation: SiLU (no softmax, no traditional QKV split)

**How it differs from standard attention**:

| Property | Standard Softmax Attention | Gated DeltaNet |
|----------|---------------------------|----------------|
| Complexity | O(n²) | O(n) |
| KV cache | Full (n × d per layer) | Compressed state (d × d) |
| Memory scaling | Linear in seq len | Constant in seq len |
| Retrieval | Strong at all positions | Strong, improved via delta rule |
| Long-context extrapolation | Limited (position encoding) | Naturally extrapolates |
| Maturity | Billions of GPU-hours validated | ~1 year since ICLR 2025 |

### 3.3 Gated Attention (Full-Attention Layers)

25% of layers use a standard softmax attention variant with:
- Q Heads: 32 (397B), KV Heads: 2 (GQA pattern)
- Head dimension: 256
- RoPE dimension: 64
- Multi-Head Latent Attention (MLA) style with `attn_output_gate: true`

**Why 25% softmax?**: Linear attention excels at efficiency but can struggle with precise retrieval from long-range tokens. The 25% softmax layers serve as "retrieval specialists" — maintaining the model's ability to perform exact-match lookup and fine-grained positional reasoning that linear attention handles less well.

### 3.4 MoE Configuration

| Model | Total | Active | Experts | Active/Token | Shared Expert | Sparsity Ratio |
|-------|-------|--------|---------|-------------|---------------|----------------|
| **397B-A17B** | 397B | 17B | **512** | **10** + 1 shared | Yes (1024 dim) | ~3% |
| **122B-A10B** | 122B | 10B | **256** | **8** + 1 shared | Yes (1024 dim) | ~8% |
| **35B-A3B** | 35B | 3B | **256** | **8** + 1 shared | Yes (512 dim) | ~9% |

- **Routing**: Top-k softmax with auxiliary loss (`router_aux_loss_coef: 0.001`)
- **Shared expert**: Every token also passes through a shared dense FFN (captures common knowledge)
- **No dense islands**: All 60 layers use MoE (no early dense layers)
- **Massive expert count** (512): Enables extreme specialization — each expert can focus on narrow patterns

### 3.5 Native Multimodal (Early Fusion)

**Key difference from standard approach**: Standard VLMs bolt a vision encoder (CLIP, SigLIP) onto a frozen LLM via a projector, then fine-tune on image-text data. Qwen3.5 trains vision and text jointly from the start.

- Vision encoder: **27-layer ViT**, patch size 16×16, temporal patch size 2 (for video)
- **No "translation loss"**: Vision tokens and text tokens share the same latent space natively
- **Near-100% multimodal training efficiency**: Model suffers virtually no throughput penalty during multimodal vs text-only training
- Special tokens: `<|vision_start|>`, `<|vision_end|>`, `<|image|>`, `<|video|>`
- **Benchmark dominance**: MathVision 88.6 (vs GPT-5.2's 83.0), OmniDocBench 90.8, OCRBench 93.1
- Up to **2-hour long-form video understanding**

### 3.6 FP8 Training Pipeline

Qwen3.5 released **official FP8 quantized variants** of all major models:
- Qwen3.5-397B-A17B-FP8
- Qwen3.5-122B-A10B-FP8
- Qwen3.5-35B-A3B-FP8
- Qwen3.5-27B-FP8

Also NVIDIA NVFP4 variants available.

FP8 training applied to **activations, MoE routing, and matrix operations** — not just inference.
- ~50% activation memory reduction
- >10% training speedup
- Scales stably to tens of trillions of training tokens

### 3.7 Vocabulary & Multilingual: 201 Languages

- **248,320 token vocabulary** (expanded from ~150K in Qwen3)
- **201 languages and dialects** (up from 119)
- Includes low-resource languages: Hawaiian, Fijian, Niger-Congo languages
- Encoding efficiency improved 10-60% across most languages

**New additions**: Significant expansion in Niger-Congo (Swahili, Yoruba, Igbo, Zulu), Austronesian (Javanese, Sundanese), and Turkic (Uzbek, Kazakh, Uyghur) families.

### 3.8 Thinking / Non-Thinking Modes

- **Thinking mode (default)**: Generates `<think>reasoning...</think>\n\nanswer` — default for all models
- **Non-thinking mode**: Direct answer (opt-in via `enable_thinking: False`)
- **Removed from Qwen3**: The `/think` and `/no_think` soft tokens from Qwen3 are **gone** — only API-level control now

### 3.9 Agentic Capabilities & MCP

- **MCP-native**: First-class Model Context Protocol support
- **Tool calling**: `BFCL-V4: 72.9` — outperforms GPT-5.2 (63.1) on function calling
- **Visual agent**: AndroidWorld 66.8 (no other model scored), OSWorld-Verified 62.2
- **Search agent**: BrowseComp 78.6, HLE w/ tool 48.3 (best among open models)
- **Qwen-Agent framework**: Python with MCP support
- **Qwen Code**: CLI agentic coding tool (forked from Gemini Code)

### 3.10 Pros / Cons / Drawbacks

| Pros | Cons/Drawbacks |
|------|----------------|
| **O(n) attention** for 75% of layers — no quadratic bottleneck | **Linear attention is immature** — only ~1 year since publication; less community optimization |
| **Cheapest frontier API** (~$0.11/M tokens, ~13× cheaper than Claude) | **MoE deployment complexity** — 512 experts requires careful sharding |
| **Best open multimodal** — dominants MathVision, OCR, document understanding | **Needs bleeding-edge SGLang/vLLM** — stable releases don't yet support Qwen3.5 |
| **201 languages** — best-in-class among open models | **No soft thinking switch** — removed from Qwen3; only API-level toggle |
| **Outstanding instruction following** — IFBench 76.5 (best of any model) | **~398GB at FP16** — the 397B model requires massive hardware even with quantization |
| **MCP-native agent support** — first-class tool use protocol | **Vision encoder overhead** — even for text-only tasks |
| **Apache 2.0** — fully permissive | **Trails GPT-5.2 on pure math** — AIME 91.3 vs 96.7 |
| **Multi-Token Prediction** — enables 2-3× throughput via speculative decoding | **Middle-size gap** — no dense models between 27B and 397B (122B MoE partly fills) |
| **Efficiency per active param** — 35B-A3B delivers 85-95% of 27B dense at 11% FLOPs | **Agentic slightly behind Claude** — Tau2-Bench 86.7 vs 91.6 |

**Unique advantages over standard Transformer**:
- **75% linear attention** breaks the O(n²) wall
- **Native multimodal** without separate vision pipeline
- **512-expert MoE** enables extreme specialization
- **201 languages** in a single model

**Unique drawbacks vs standard Transformer**:
- **Linear attention ecosystem immaturity** vs softmax attention's billions of GPU-hours
- **Dense models also use Gated DeltaNet** — even the "simple" variants aren't standard
- **Early fusion multimodal** means ViT compute runs even on text-only inputs

---

## 4. DeepSeek V4

### 4.1 Architecture Overview

- **Architecture**: Hybrid CSA+HCA compressed sparse attention + MoE
- **V4-Pro**: 1.6T total, 49B active, 61 layers, hidden dim 7168
- **V4-Flash**: 284B total, 13B active, 43 layers, hidden dim 4096
- **Context window**: 1M default, 384K max output
- **License**: MIT

**Major architectural innovations** (vs standard Transformer):
1. **CSA + HCA hybrid attention** — replaces full softmax attention entirely
2. **Manifold-Constrained Hyper-Connections (mHC)** — replaces simple residual add
3. **FP4 + FP8 mixed precision** — MoE experts trained in native FP4
4. **Muon optimizer** — replaces AdamW for most parameters

### 4.2 Hybrid Attention: CSA + HCA

DeepSeek V4 does **not use standard softmax attention at all**. Instead, it uses a two-tier compression-attention system:

#### Compressed Sparse Attention (CSA)

**Step 1 — Token-Level Compression**:
- Every `m = 4` tokens are merged into one compressed KV entry via learned weighted combination
- Overlapping windows (each entry uses 2m tokens for smoothing)
- Compression rate: **4:1**

**Step 2 — Lightning Indexer (Sparse Selection)**:
- A lightweight indexer compresses queries into indexer keys
- For each query, computes index scores via multi-head ReLU scoring
- Retains only **top-k = 1024** (Pro) / **512** (Flash) highest-scoring compressed KV entries
- This is "DeepSeek Sparse Attention" — the sparse selection mechanism within CSA

**Step 3 — Multi-Query Attention (MQA)**:
- All attention heads share the same KV (from selected compressed entries)
- 128 heads (Pro) / 64 heads (Flash), grouped into 16 (Pro) / 8 (Flash) output projection groups

#### Heavily Compressed Attention (HCA)

- Compression rate: **m' = 128** (128 tokens → 1 KV entry)
- No sparse selection — dense attention over all heavily compressed entries
- Acts as a "big picture" attention layer covering long-range dependencies

#### Interleaved Configuration

- **V4-Pro**: First 2 layers = pure HCA; subsequent layers = interleaved CSA + HCA
- **V4-Flash**: First 2 layers = Sliding Window Attention (window 128); subsequent = interleaved
- Both add a **Sliding Window Attention** (n_win=128) supplementary branch for local fine-grained dependencies
- **Attention sink**: Learnable sink logits allow per-head attention sum ≠ 1
- **Partial RoPE**: Applied to last 64 dims of each Q/K/V vector

### 4.3 KV Cache Reduction Mechanics

The combined effect of all attention innovations at **1M context**:

| Factor | Reduction |
|--------|-----------|
| CSA compression (4:1) | 4× fewer KV entries |
| Top-k sparsity (k=1024 vs 1M) | ~1000× entries attended to (CSA layers) |
| HCA compression (128:1) | 128× fewer KV entries (HCA layers) |
| Mixed BF16/FP8 storage | ~2× less memory per entry |
| Shared KV (MQA) | All heads share one KV |

**Net effect vs V3.2 at 1M**:
- V4-Pro FLOPs: **27%** of V3.2
- V4-Pro KV cache: **10%** of V3.2 (≈ **2%** of standard GQA8 BF16 baseline)
- V4-Flash KV cache: **7%** of V3.2

**This is the key engineering achievement**: Maintaining frontier-quality output while slashing the memory/compute cost of long context to a fraction of what any standard Transformer requires.

### 4.4 MoE Configuration

| Parameter | V4-Pro | V4-Flash |
|-----------|--------|----------|
| Routed experts per layer | **384** | **256** |
| Active routed experts per token | **6** | **6** |
| Shared experts | 1 | 1 |
| Expert intermediate dim | 3072 | 2048 |
| Hash routing layers | First 3 MoE layers | First 3 MoE layers |

**Routing details**:
- **Top-6**, not Top-16 (common misconception)
- Activation function: **Sqrt(Softplus(·))** — changed from Sigmoid in V3
- **Auxiliary-loss-free load balancing** (inherited from V3) with slight sequence-wise balance loss
- **Hash routing** for first 3 MoE layers: deterministic expert assignment based on token ID hash (avoids routing instability for common tokens)

### 4.5 Manifold-Constrained Hyper-Connections (mHC)

Replaces simple residual `x + Attention(x)` with a more complex transformation:

- **Expansion factor `n_hc = 4`**: Residual stream width expanded from `d` to `4d`
- **Birkhoff polytope constraint**: The residual mapping matrix `B_l` is constrained to doubly stochastic matrices (rows and columns sum to 1, all non-negative)
- **Spectral norm ≤ 1**: Guarantees non-expansive residual transformation → stable signal propagation in deep stacks
- **Sinkhorn-Knopp algorithm** (20 iterations) projects raw parameters onto the Birkhoff manifold
- **Dynamic parameterization**: Both input-dependent and static components for input/output/residual mappings
- **Sigmoid-bounded** input/output transformations prevent signal cancellation
- Overhead: only **6.7% of wall-time** via fused kernels + recomputation

**Why it matters**: In standard Transformers, residual connections are a simple add that can suffer from vanishing gradients in deep stacks (61 layers for V4-Pro). mHC provides mathematically guaranteed gradient flow through the Birkhoff constraint.

### 4.6 FP4 + FP8 Mixed-Precision Training

DeepSeek's most distinctive training innovation: **native FP4 training for MoE experts**, not just post-training quantization.

| Component | Precision | Notes |
|-----------|-----------|-------|
| MoE expert weights | **FP4 (MXFP4, E2M1)** | Trained at FP4 from the start |
| Index scores (CSA) | FP4 → BF16 | Cached and multiplied in FP4, then dequantized |
| Non-expert parameters | FP8 (E4M3) | Attention projections, embeddings, etc. |
| Master weights | FP32 | For gradient accumulation |
| Gradient storage | BF16 | Stochastic rounding halves communication |

**Key insight**: FP4 → FP8 dequantization is **lossless** because FP8 (E4M3, 2 more exponent bits) has larger dynamic range. The fine-grained scale factors of FP4 sub-blocks are absorbed by FP8's extended range.

**QAT (Quantization-Aware Training)**: FP32 master weights → quantize to FP4 → dequantize to FP8 for computation. Gradients propagate through Straight-Through Estimator back to FP32 master weights.

**Why train at FP4 vs post-quantize**: A model trained at FP4 learns to produce high-quality outputs within reduced precision constraints. Post-quantizing a BF16-trained model always loses some quality. DeepSeek's FP4-trained experts retain frontier performance despite the aggressive quantization.

### 4.7 Muon Optimizer

Replaces AdamW for most parameters:
- Used for **all modules except** embeddings, prediction head, RMSNorm weights, and mHC biases/gating (these use AdamW)
- **Nesterov momentum** with momentum = 0.95
- **Hybrid Newton-Schulz iterations**: 2-stage orthogonalization (8 steps fast coefficients, 2 steps stabilizing)
- No QK-Clip needed (RMSNorm on queries/KV entries prevents exploding attention logits)
- **ZeRO hybrid strategy**: Knapsack assignment for dense params + expert-independent optimization for MoE
- **BF16 Newton-Schulz** remains stable (no precision loss)

**Advantage**: Muon provides ~2× compute efficiency vs AdamW at scale (consistent with Moonshot AI's findings on Muon).

### 4.8 Reasoning Modes (Non-Think / Think High / Think Max)

Unlike Qwen3.5's per-token thinking control, DeepSeek V4 offers **three distinct inference modes** selectable via system prompt:

| Mode | Behavior | Context Budget | Best For |
|------|----------|----------------|----------|
| **Non-think** | Fast, intuitive response | 8K | Trivial Q&A, classification |
| **Think High** | Deliberate logical analysis | 128K | Complex reasoning, math |
| **Think Max** | Maximum reasoning effort | 384K | Hardest problems, competitive coding |

- Works via a **single model** with different generation parameters
- The "Think Max" system prompt: *"Reasoning Effort: Absolute maximum with no shortcuts permitted..."* — same language as in system prompts across the ecosystem
- Enables per-task compute budgeting without switching models

### 4.9 Engram Conditional Memory (Separate Research)

**Important clarification**: This is **NOT integrated into V4**. It is a separate research paper from the same DeepSeek team (arXiv:2601.07372, Jan 2026), cited as future direction.

**Core idea**: Add a third axis of sparsity alongside MoE — **conditional memory** via O(1) hash lookup.

- **N-gram lookup table**: Massive hash-based embedding table stored in host DRAM (not GPU VRAM)
- **O(1) deterministic addressing**: Token N-grams hashed to get deterministic indices — no routing network
- **Context-aware gating**: Learned scalar `α_t` dynamically modulates integration of static memory vs backbone
- **Layered insertion**: Optimal at layers 2 and 6
- **Prefetching**: Deterministic addresses enable async PCIe prefetching, overlapping memory transfer with computation

**Key results**:
- 100B-parameter embedding table in CPU DRAM incurs only **2.8% throughput penalty** on H800
- MMLU +3.4, BBH +5.0, HumanEval +3.0, MATH +2.4
- Multi-Query NIAH retrieval from 84.2 → 97.0
- When ablated: factual knowledge collapsed to 29-44% retained, but reading comprehension stayed at 81-93% (backbone handles context, Engram handles facts)

### 4.10 V4-Pro vs V4-Flash

| Aspect | V4-Pro | V4-Flash |
|--------|--------|----------|
| Total params | 1.6T | 284B |
| Active params | 49B | 13B |
| Layers | 61 | 43 |
| Hidden dim | 7168 | 4096 |
| Routed experts | 384 | 256 |
| CSA top-k | 1024 | 512 |
| Attention heads | 128 | 64 |
| Output groups | 16 | 8 |
| HF weight size | 865 GB | 160 GB |
| API input price | $1.74/M | $0.14/M |
| API output price | $3.48/M | $0.28/M |
| SWE-bench Verified | **80.6%** | 79.0% |
| LiveCodeBench | **93.5** | 91.6 |
| SimpleQA | **57.9%** | 34.1% |
| Codeforces | **3206** | 3052 |

**When to use Flash**: Cost-sensitive, high-throughput workloads where 2-5% quality drop is acceptable.
**When to use Pro**: Maximum quality tasks where benchmark deltas matter.

### 4.11 Pros / Cons / Drawbacks

| Pros | Cons/Drawbacks |
|------|----------------|
| **Extreme long-context efficiency** — 2% KV cache vs standard GQA8 at 1M | **Architecture complexity** — "many preliminarily validated components and tricks" (V4 report's own words) |
| **Cheapest output tokens** — Flash at $0.28/M output | **Trails GPT-5.4 by 3-6 months** — acknowledged gap |
| **Best open-weights SWE-bench** — 80.6% tied with Gemini 3.1 Pro | **Knowledge gap** — SimpleQA 57.9 vs Gemini 3.1 Pro 75.6 |
| **Codeforces #23 among humans** — first open model matching GPT-5.4 | **Long-context retrieval** — MRCR 83.5 vs Claude Opus 4.6 92.9 |
| **Three reasoning modes in one model** | **Training instability** — required Anticipatory Routing and SwiGLU clamping |
| **MIT license** | **No multimodal** — text-only (multimodal "in progress") |
| **FP4 training breakthrough** — proves native low-precision training works at scale | **Huge total params** — 1.6T requires multi-node even quantized |
| **Best cost/performance ratio in market** | **No Jinja chat template** — custom encoding logic required |
|  | **V4-Flash large knowledge gap** — SimpleQA 34.1% vs 57.9% for Pro |

**Unique advantages over standard Transformer**:
- **CSA+HCA completely replaces softmax attention** — zero quadratic compute at inference
- **mHC** provides mathematically guaranteed stable gradient flow in deep stacks
- **FP4 native training** — frontier performance at 4-bit precision
- **Muon optimizer** replaces AdamW

**Unique drawbacks vs standard Transformer**:
- **Extreme architectural complexity** — many components required for stability
- **No established inference stack** — custom kernels needed (MegaMoE, TileLang)
- **Text-only at launch** — multimodal still in development

---

## 5. Kimi K2.6

### 5.1 Architecture Overview

- **Architecture**: MoE + MLA (Multi-Head Latent Attention, inherited from DeepSeek V3)
- **Total params**: 1T (32B active per token)
- **Layers**: 61 (1 dense + 60 MoE), hidden dim 7168
- **Context window**: 256K (262,144 tokens)
- **Vocabulary**: 160K (163,840)
- **License**: Modified MIT (restrictions >100M MAU or $20M monthly revenue)

K2.6 is a **general-purpose native multimodal agentic model**, sitting between K2.5 (Jan 2026, arXiv:2602.02276) and K2.7 Code (Jun 2026). It is essentially a **continued-training / RL-tuned version of K2.5** with the exact same architecture.

### 5.2 Multi-Head Latent Attention (MLA)

**Origin**: DeepSeek V3 (arXiv:2412.19437). Moonshot AI licensed/adapted this architecture for K2.6.

**Core concept**: Compress KV cache into a low-rank latent space rather than storing full keys and values.

| Parameter | K2.6 Value |
|-----------|-----------|
| Attention heads | 64 |
| KV heads | 64 (**MHA**, not GQA!) |
| KV LoRA rank | 512 |
| Q LoRA rank | 1536 |
| QK Nope (no position encoding) head dim | 128 |
| QK RoPE head dim | 64 |
| V head dim | 128 |

**How MLA works**:
1. Keys and values are projected down to a **low-rank latent space** (rank 512) before being cached
2. At attention time, the latent is projected back up to the full head dimension
3. This dramatically reduces the KV cache size per token — critical for long-context inference
4. **Decouples nope (no PE) and rope (rotary PE) dimensions** — allows position-aware and position-unaware attention to coexist in the same head

**Why MHA instead of GQA**: K2.6 uses full Multi-Head Attention (64 KV heads = 64 Q heads) rather than grouped query attention. This is unusual for a 1T-parameter model. The MLA compression makes the KV cache small enough that MHA becomes feasible — the latent-space compression (rank 512 vs full dim 7168) provides the savings that GQA normally would.

### 5.3 MoE Configuration

| Parameter | K2.6 |
|-----------|-------|
| Routed experts per layer | **384** |
| Active routed experts per token | **8** |
| Shared experts | 1 |
| Expert hidden dim | 2048 |
| Hash routing layers | First 3 MoE layers |
| Activation function | SwiGLU (silu) |
| Intermediate size (dense FFN) | 18,432 |
| Normalization | RMSNorm (eps=1e-5) |
| First layer | Dense FFN (not MoE) |

**Routing**:
- **`noaux_tc` topk method** — no auxiliary loss for load balancing
- **Sigmoid scoring** with `norm_topk_prob=true`
- Routed scaling factor: 2.827
- **Fine-grained expert decomposition** (384 experts) with 8 activated per token — same philosophy as DeepSeekMoE

**Comparison to DeepSeek V4's Top-6**: K2.6 activates 8 experts per token vs V4's 6, giving more capacity per token but at higher compute cost.

### 5.4 YaRN RoPE Scaling (64×)

- **YaRN** (Yet another RoPE extensioN) with **64× scaling factor**
- Original pretraining length: **4,096 tokens**
- Scaled to: **262,144 tokens** (64×)
- `beta_fast=32, beta_slow=1` — YaRN interpolation parameters

**Why YaRN**: Standard RoPE doesn't extrapolate well beyond trained length. YaRN applies frequency-domain interpolation with a ramp function that smoothly transitions from high-frequency (unchanged, gives fine-grained position resolution) to low-frequency (interpolated, enables longer ranges) bands.

### 5.5 MoonViT Vision Encoder

| Parameter | Value |
|-----------|-------|
| Architecture | ViT (27 layers) |
| Parameters | ~400M |
| Patch size | 14 |
| Vision hidden size | 1,152 |
| Attention heads | 16 |
| Position embedding | **Divided Fixed** (H:64, W:64, T:4) |
| Attention | Flash Attention 2 |
| Projector | **PatchMerger** (sd2_tpool, kernel 2×2) |

- **Divided Fixed position embedding**: Encodes height, width, and time dimensions separately — better for high-resolution images and variable aspect ratios than standard 1D position embeddings
- **PatchMerger**: Merges adjacent patch features before feeding into LLM (reduces token count from vision → text)

### 5.6 Thinking / Instant Modes

- **Thinking mode (temp=1.0)**: Generates chain-of-thought reasoning tokens before answer — for complex reasoning
- **Instant mode (temp=0.6)**: Direct answer without CoT — for low-latency applications
- **Preserve Thinking** (optional, disabled by default in K2.6): Retains full reasoning content across multi-turn interactions (useful for coding agents)

**Contrast with Qwen3.5**: K2.6 provides **soft switching** (API call parameter) unlike Qwen3.5 which removed soft switching. **Contrast with K2.7 Code**: K2.7 Code **forces thinking on** and removes Instant mode.

### 5.7 Agent Swarm Orchestration

K2.6's most distinctive high-level capability:

- Scales **horizontally to 300 sub-agents** executing **4,000+ coordinated steps**
- Dynamically decomposes complex tasks into parallel, domain-specialized subtasks
- Reduces latency by up to **4.5×** over single-agent baselines

**How it differs from standard agent frameworks**: Rather than a single agent looping on tools, K2.6 can spawn specialist sub-agents for different subtasks (search, code, analysis) and orchestrate them in parallel. This is a **model-native capability**, not an external framework.

**Benchmark results with swarm**:
- BrowseComp: 83.2 (single) → **86.3** (swarm)
- DeepSearchQA: 92.5 F1 (best among all models)

### 5.8 Native INT4 Quantization

K2.6 ships with **native INT4 weights** using `compressed-tensors` format:
- Group size: 32
- Attention self-attention layers excluded from quantization (kept in higher precision)
- Shared experts, MLP projections, LM head also excluded
- Same quantization method as Kimi K2-Thinking

**Advantage**: The model arrives pre-quantized — no need for GPTQ/AWQ/GGUF conversion. The model was evaluated at this precision, not degraded from a higher-precision target.

### 5.9 K2.6 vs K2.7 Code

| Dimension | K2.6 (General) | K2.7 Code (Coding Specialized) |
|-----------|----------------|--------------------------------|
| Release | ~May 19, 2026 | Jun 12, 2026 |
| Architecture | Identical | Identical |
| Parameters | 1T / 32B active | 1T / 32B active |
| Focus | General multimodal agentic | **Coding-focused agentic** |
| Built on | K2.5 (continued training) | **K2.6** (fine-tuned from) |
| Thinking mode | Optional (thinking + instant) | **Forced on** (always thinking) |
| Preserve Thinking | Optional (disabled by default) | **Forced on** (can't disable) |
| Thinking tokens | Baseline | **~30% fewer** via compressed reasoning chains |

**Benchmark gains (K2.6 → K2.7 Code)**:
- Kimi Code Bench v2: 50.9 → 62.0 (+21.8%)
- MLS Bench Lite: 26.7 → 35.1 (+31.5%)
- MCP Mark Verified: 72.8 → 81.1 (+8.3)

K2.7 Code is **strictly a fine-tune specialization** — same backbone, changed training distribution (coding-heavy) and inference mode (always thinking, compressed reasoning).

### 5.10 Pros / Cons / Drawbacks

| Pros | Cons/Drawbacks |
|------|----------------|
| **Strongest search agent** — DeepSearchQA 92.5 F1, BrowseComp 83.2 | **Trails on pure reasoning** — HLE (no tools) 34.7 vs GPT-5.4 39.8 |
| **Competitive coding** — SWE-bench Pro 58.6 (leads), Terminal-Bench 66.7 | **Weak fine-grained vision** — BabyVision 39.8 vs Gemini 3.1 Pro 51.6 |
| **Agent Swarm** — unique 300-agent orchestration | **Massive total params** — 1T requires substantial infrastructure |
| **Open weights** (Modified MIT) | **No standalone K2.6 paper** — changes from K2.5 not fully documented |
| **MLA efficiency** — enables 256K context with 64 KV heads | **Custom model type** — requires `trust_remote_code=True` |
| **Native INT4 quantization** — arrives pre-quantized | **Modified MIT restrictions** — >100M MAU or $20M monthly revenue |
| **Dual Thinking/Instant modes** — flexibility per task | **Chinese provenance concerns** — export control implications for some users |
| **Multi-language coding strength** — MLS Bench Lite +31.5% in K2.7 | **Dependency on DeepSeek V3 codebase** — not truly independent architecture |
| **Competitive vision** — MMMU-Pro 79.4, MathVision 87.4 | **Training data not disclosed** |
| | **Agentic slightly behind Claude** — Claw Eval 62.3 vs 70.4 |

**Unique advantages over standard Transformer**:
- **MLA** compresses KV to low-rank latent, enabling MHA at 1T scale
- **Agent Swarm** is a model-native capability, not external framework
- **Native INT4** — no post-training quantization needed
- **YaRN 64× scaling** from 4K to 256K

**Unique drawbacks vs standard Transformer**:
- **Derived from DeepSeek V3** architecture — not independent innovation in attention
- **Requires custom code** (`trust_remote_code=True`)
- **Modified MIT** has usage caps (unlike Apache 2.0 or MIT)

---

## 6. Cross-Cutting Comparison

### 6.1 Attention Mechanisms

| Aspect | Standard Transformer | Qwen 3.5 | DeepSeek V4 | Kimi K2.6 |
|--------|--------------------|----------|-------------|-----------|
| **Primary mechanism** | Softmax self-attention | **Gated DeltaNet** (linear recurrence) 75% + Gated Attention 25% | **CSA + HCA** (compressed sparse) | **MLA** (Multi-Head Latent Attention) |
| **Complexity** | O(n²) | ~O(n) (with O(n²) on 25% layers) | ~O(k×n) where k=top-k (near-O(n) in practice) | O(n²) but with compressed KV cache |
| **KV cache growth** | O(n) per layer | O(1) for 75% layers | ~O(1) for HCA, ~O(k) for CSA | O(n × rank) where rank << dim |
| **Theoretical basis** | Dot-product attention | State-space model + delta rule | Compression + sparsity | Low-rank approximation |
| **Position encoding** | Full RoPE | Full RoPE | **Partial RoPE** (last 64 dims) | **YaRN RoPE** (64× scaled) |
| **Maturity** | Very high | Low (~1 year) | Low (~2 months) | Medium (~1.5 year, from V3) |

**Key takeaway**: All three models **reject standard softmax attention** in favor of different efficiency mechanisms:
- Qwen: Linear recurrence (radically different paradigm)
- DeepSeek: Compression + sparsity (engineering innovation on attention)
- Kimi: Latent-space compression (elegant mathematical trick)

### 6.2 MoE Design Philosophy

| Aspect | Qwen 3.5 | DeepSeek V4 | Kimi K2.6 |
|--------|----------|-------------|-----------|
| **Experts per layer** | 512 (flagship) | 384 (Pro) | 384 |
| **Active per token** | 10 + 1 shared | **6 + 1 shared** | 8 + 1 shared |
| **Total:Active ratio** | 30:1 | 33:1 | 32:1 |
| **Routing** | Top-k softmax + aux loss | Sqrt(Softplus) + no aux loss | Sigmoid + no aux loss |
| **Shared expert** | Yes | Yes | Yes |
| **Hash routing** | No | First 3 layers | First 3 layers |

**Key differences**:
- Qwen3.5 has the **most experts** (512) and **most active per token** (10) — maximizes specialization
- DeepSeek V4 has the **fewest active per token** (6) — maximizes efficiency
- Kimi K2.6 is in between (8 active)
- All three use shared experts (common knowledge capture)
- Both DeepSeek and Kimi use hash routing for first 3 layers (deterministic assignment avoids routing instability for common tokens)

### 6.3 KV Cache & Long-Context Strategy

| Aspect | Qwen 3.5 | DeepSeek V4 | Kimi K2.6 |
|--------|----------|-------------|-----------|
| **Native context** | 256K | **1M** | 256K |
| **Max extended** | ~1M (YaRN) | 1M (native) | 256K (native) |
| **KV cache strategy** | Linear attention skips KV cache entirely for 75% layers | Compression (4:1 + 128:1 + top-k sparsity) | Low-rank latent compression (rank 512) |
| **KV cache vs standard** | ~25% of standard (only 25% layers have full KV) | **~2% of standard GQA8** | ~15% of standard (projected) |
| **Retrieval at long context** | Good (25% softmax layers serve as retrieval specialists) | Good (HCA handles global, SWA handles local) | Good (MLA maintains decent recall) |

**DeepSeek V4 wins decisively on KV cache efficiency** — but at the cost of architectural complexity.

### 6.4 Training Precision

| Aspect | Standard | Qwen 3.5 | DeepSeek V4 | Kimi K2.6 |
|--------|----------|----------|-------------|-----------|
| **Training precision** | BF16/FP16 | **FP8** | **FP4 + FP8** (experts FP4, rest FP8) | BF16 |
| **Quantization approach** | Post-training | Native FP8 training | **Native FP4 QAT** | Native INT4 inference |
| **Inference precision** | FP16/INT4 | FP8/FP16 | FP4 + FP8 | INT4 (native) |
| **Ambition** | Standard | Reduce training cost | Push limits of low-precision training | Enable deployment at low cost |

**DeepSeek V4 is the most aggressive**, training MoE experts from scratch at FP4. Qwen3.5's FP8 is closer to industry standard. Kimi K2.6 is conservative (BF16 training, INT4 inference).

### 6.5 Multimodal Approach

| Aspect | Qwen 3.5 | DeepSeek V4 | Kimi K2.6 |
|--------|----------|-------------|-----------|
| **Vision integration** | **Early fusion** (trained from start) | **Not supported** (text-only) | **Early fusion** (trained jointly) |
| **Vision encoder** | 27-layer ViT (patch 16) | N/A | MoonViT (27-layer, patch 14) |
| **Training modality** | Text + image + video jointly | Text only | Text + image jointly |
| **Benchmark strength** | MathVision 88.6 (best open) | N/A | MathVision 87.4, MMMU-Pro 79.4 |
| **Video** | Up to 2 hours | N/A | Not disclosed |

**Qwen 3.5 leads on multimodal** — broadest capability, highest benchmarks, and native video support. Kimi K2.6 is competitive on vision benchmarks. DeepSeek V4 has no multimodal support.

### 6.6 Inference Modes

| Aspect | Qwen 3.5 | DeepSeek V4 | Kimi K2.6 |
|--------|----------|-------------|-----------|
| **Thinking mode** | Default (enable_thinking=True) | Think High / Think Max | Thinking (temp=1.0) |
| **Non-thinking mode** | Opt-in (enable_thinking=False) | Non-Think (8K budget) | Instant (temp=0.6) |
| **Soft switch** | ❌ Removed from Qwen3 | ✅ Via system prompt | ✅ Via API param |
| **Number of modes** | 2 | **3** (Non-Think / High / Max) | 2 |
| **Per-token thinking** | ✅ Determined per token | ❌ Set per conversation | ✅ Set per conversation |

**DeepSeek V4 offers the most granular control** with 3 distinct reasoning budgets. Qwen3.5 has the most flexible per-token control but removed the convenient soft tokens.

### 6.7 Openness & Licensing

| Aspect | Qwen 3.5 | DeepSeek V4 | Kimi K2.6 |
|--------|----------|-------------|-----------|
| **License** | **Apache 2.0** | MIT | Modified MIT |
| **Weights** | Open (HuggingFace) | Open (HuggingFace) | Open (HuggingFace) |
| **Derivative models on HF** | 170,000+ | ~5,000 expected | ~5,000 expected |
| **Technical report** | ✅ Yes | ✅ Yes (310 pages) | ✅ Via K2.5 paper |
| **Model card** | ✅ Detailed | ✅ Detailed | ✅ Detailed |
| **Usage restrictions** | None | None | >100M MAU or >$20M/mo revenue |
| **Custom code needed** | No (transformers compatible) | No (HF transformers) | **Yes** (`trust_remote_code`) |

**Qwen 3.5 wins on openness** — Apache 2.0 is maximally permissive, and it has the largest ecosystem of derivative models.

---

## 7. Summary Table

| Dimension | Qwen 3.5 | DeepSeek V4 | Kimi K2.6 |
|-----------|----------|-------------|-----------|
| **Organization** | Alibaba (Qwen) | DeepSeek | Moonshot AI |
| **Release date** | 2026-02-16 | 2026-04-24 | 2026-05-19 |
| **Flagship model** | 397B-A17B | V4-Pro (1.6T / 49B) | K2.6 (1T / 32B) |
| **Attention innovation** | Gated DeltaNet (linear) | CSA + HCA (compressed sparse) | MLA (low-rank latent) |
| **MoE experts** | 512 / 10 active | 384 / 6 active | 384 / 8 active |
| **Context** | 256K native → 1M YaRN | **1M native** | 256K native |
| **KV cache vs standard** | ~25% | **~2%** | ~15% |
| **Training precision** | FP8 | FP4 + FP8 | BF16 |
| **Multimodal** | Text + Image + Video | ❌ Text-only | Text + Image |
| **Languages** | **201** | ~100 | ~50 |
| **Inference modes** | Thinking / Non-thinking | **3 modes** (Non/High/Max) | Thinking / Instant |
| **Agent capability** | MCP-native, BFCL 72.9 | Competitive (SWE 80.6%) | **Swarm 300 agents** |
| **API cheapest variant** | ~$0.11/M (Alibaba Cloud) | $0.14/M input (Flash) | $0.95/M input |
| **License** | **Apache 2.0** | MIT | Modified MIT |
| **Best use case** | Multimodal, multilingual, cost-sensitive | Long-context, coding, cost-effective | Search agents, coding, swarm orchestration |
| **Biggest strength** | Native multimodality + 201 languages | Extreme long-context cost efficiency | Agent Swarm orchestration |
| **Biggest weakness** | Linear attention immaturity | Architectural complexity | Modified MIT restrictions |
| **Unique architectural element** | Gated DeltaNet 75% layers | CSA+HCA + mHC + FP4 training | MLA + MHA at 1T scale |
| **Ecosystem maturity** | High (170K+ derivatives) | Low (recent) | Medium |

---

*Report compiled from: HuggingFace model cards & configs, official technical reports (DeepSeek V4 310-page report, K2.5 arXiv:2602.02276, Gated DeltaNet arXiv:2412.06464, Engram arXiv:2601.07372), third-party analyses (Morph, O-mega, Lushbinary, Artificial Analysis), and benchmark trackers (llm-stats, SWE-bench, LiveCodeBench).*
