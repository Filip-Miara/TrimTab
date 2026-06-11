# Latent Reasoning System: Analysis, Design & Evaluation Strategy

## 0. VOID PRECONDITION — Assumptions We Must Set Aside

**Assumptions we are making (and must suspend):**

1. The Perceiver bottleneck architecture is optimal for latent reasoning (it was designed for adapter fusion, not thinking)
2. Flow matching over weight trajectories transfers perfectly to thought-trajectory diffusion
3. The MetaController's flag-space is the right action space for reasoning-mode selection
4. More data and more parameters always improve reasoning quality
5. Our evaluation metrics (perplexity, loss) capture reasoning quality
6. The 0.8B Qwen model has sufficient capacity for emergent reasoning
7. Online adapter training during reasoning is computationally feasible
8. The reasoning loop should converge (fixed point) vs. diverge (creative exploration)

## 1. What We've Been Doing Wrong / Sub-Optimally

### 1.1 Disconnected Component Optimization
Each component (StreamFusion, MetaController, WeightDiffusion) was developed in isolation and evaluated separately. The **splinter analysis** skill warns precisely about this: fragmented capability profiles with no holistic metric.

**Root cause**: We optimized each module's local loss without a unified reasoning objective.

**Fix**: Define a single reasoning quality metric (Section 4) that all components optimize jointly.

### 1.2 Data Scarcity Masks Architecture Problems
We blamed poor weight-flow generalization on "only 12 trajectories" — but the deeper issue is that our conditioning (mean hidden state + gradient) is **information-theoretically insufficient** to determine the correct weight update. The mutual information `I(ctx; ΔW)` is low.

**Fix**: Use the **closed-form SVD solution** (v0.10) as a training target, not just observed SGD trajectories. The SVD gives the *optimal* update; SGD approximates it noisily.

### 1.3 We Treated Reasoning as Monolithic
Our architecture assumes a single reasoning trajectory (one chain of thoughts). The **overlap consistency** mechanism (v0.4) that works for overlapping sub-matrices naturally extends to **multiple parallel reasoning chains** — and we never implemented it.

### 1.4 No Termination Criterion
The MetaController decides when to stop reasoning via implicit confidence. The **temporal-causality-engine** skill would require an explicit halting condition: "Run reasoning steps until `Δthought < ε` for 3 consecutive steps."

### 1.5 Evaluation Overfit on Perplexity
Every evaluation compared loss/perplexity. But reasoning quality is NOT loss — it's correctness, coherence, and efficiency. The **advanced-evaluation** skill prescribes pairwise comparison with position-swap, which we never used.

## 2. What We've Omitted / Not Paid Enough Attention To

### 2.1 The Closed-Form SVD Solution (v0.10)
We proved that the optimal rank-r LoRA update is `U_r Σ_r V_r^T` from `SVD(R·X⁺)`. This is NOT just a theoretical curiosity — it's the **optimal transport map** from current weights to optimal weights. The velocity field should converge to approximating this operator.

**Omission**: We never used this as a training signal. Flow matching on observed SGD trajectories learns the (noisy) path, not the (optimal) destination.

### 2.2 Attention Head Specialization
The PerceiverFusion's cross-attention weights between thought latents and inputs are **interpretable** — they show which input regions each latent focuses on. We never analyzed these patterns.

### 2.3 The Temperature / Exploration-Exploitation Trade-off
The MetaController's `temperature` parameter controls how exploratory its flag suggestions are. We set it to 1.0 and never tuned it. In reasoning, high temperature early (exploring many reasoning modes) and low temperature later (exploiting the best one) is the canonical schedule.

### 2.4 Causal Tracing
The **temporal-causality-engine** would let us answer: "Which reasoning step caused the correct answer?" and "Which adapter expert contributed most to this step?" — we have zero causal infrastructure.

### 2.5 The 5 Laws of Elegant Defense (Code Philosophy)
Our codebase violates all 5:
1. **Early Exit**: No guard clauses for edge cases (e.g., empty expert pools, zero-length trajectories)
2. **Parse, Don't Validate**: Raw dicts passed between components instead of typed dataclasses
3. **Atomic Predictability**: Side effects in forward passes (cache invalidation, step counters)
4. **Fail Fast**: Silent fallbacks (e.g., `ctx = torch.randn` when hook fails)
5. **Intentional Naming**: `ctx`, `cache`, `flow`, `model` are overloaded across the codebase

## 3. What Research We Should Have Performed

### 3.1 Converging Paradigm Detection
From **deep-cross-research**: Three independent fields are converging on the same architecture:
- **Reasoning**: Chain-of-Thought (Wei et al., 2022) → iterative refinement in token space
- **Diffusion**: Denoising diffusion (Ho et al., 2020) → iterative refinement in latent space
- **Flow Matching**: Conditional flow matching (Lipman et al., 2023) → continuous normalizing flows

**The convergence**: All three are learning iterative refinement trajectories. The difference is the space (tokens vs. latents) and the training objective (MLE vs. denoising vs. velocity matching). Our WeightDiffusion already bridges diffusion + flow matching.

### 3.2 Splinter Skills in Reasoning
The **splinter-analysis** skill's core finding: some reasoning steps are "skills" (anomalously low loss) and some are "deficiencies" (anomalously high loss). We should train adapters on reasoning steps where loss is HIGH (hard steps need more capacity) and prune adapters where loss is LOW (easy steps waste capacity).

### 3.3 The Master Regulator
From **autopoietic-inquiry-engine**: The master control node in our system is the **PerceiverFusion's latent bottleneck size** (K latents). Too few → collapse (all thoughts are the same). Too many → diffusion (thoughts don't interact). The optimal K scales with reasoning complexity. We should dynamically adjust K per query.

## 4. System Architecture: Cohesive Latent Reasoning Engine

```
┌─────────────────────────────────────────────────────────────┐
│                     MetaController (v0.6)                     │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Per-Step Decisions:                                     ││
│  │  • How many reasoning steps? (via trajectory confidence) ││
│  │  • Which attention pattern? (flags: bi, vec, norm, ..)  ││
│  │  • When to halt? (Δthought < ε for 3 consecutive)      ││
│  │  • Temperature schedule: high→low over steps             ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────┘
                           │ flags, temperature
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Latent Reasoning Engine                     │
│                                                               │
│  Input → Encoder → Thought₀ ∈ ℝ^{K×d}                        │
│                                                               │
│  FOR step = 1..N:                                             │
│    ┌─────────────────────────────────────────────────────┐   │
│    │ 1. Cross-attend: Thoughts → Input                   │   │
│    │    (PerceiverFusion cross_in)                       │   │
│    │                                                     │   │
│    │ 2. Self-attend: Thoughts ↔ Thoughts                 │   │
│    │    (PerceiverFusion self_attn)                      │   │
│    │                                                     │   │
│    │ 3. FFN transform: t_n = FFN(t_n)                   │   │
│    │                                                     │   │
│    │ 4. Residual: Thoughts += t_n                        │   │
│    │                                                     │   │
│    │ 5. WeightDiffusion denoise:                         │   │
│    │    ε = predict_noise(Thoughts, step/N)              │   │
│    │    Thoughts = denoise(Thoughts, ε)                  │   │
│    │                                                     │   │
│    │ 6. Adapter Online Training:                         │   │
│    │    L = MSE(Thought_step, Thought_target)            │   │
│    │    StreamFusion.add_expert()                        │   │
│    │    StreamFusion.train(L)                            │   │
│    │    StreamFusion.absorb()                            │   │
│    │    StreamFusion.prune(low_contribution)             │   │
│    │                                                     │   │
│    │ 7. TaylorContribution: measure each step's impact   │   │
│    │    Prune steps with negative contribution           │   │
│    └─────────────────────────────────────────────────────┘   │
│                                                               │
│  Decoder: Thoughts → Answer                                   │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 Flow Matching Objective over Thoughts

Same as WeightDiffusion but over thought trajectories:

```
Clean:    Thought₀ → Thought₁ → ... → Thought_N
Noisy:    Thought₀ + ε₀ → Thought₁ + ε₁ → ... → Thought_N + ε_N
Training: L = λ_diff · MSE(ε_pred, ε) + λ_flow · MSE(v_pred, ΔThought)
```

The velocity field `v_θ(Thought_t, t, context)` predicts the optimal next thought. Training data comes from reasoning traces on instruction-following datasets.

### 4.2 Adapter Training During Reasoning

Each reasoning step trains a NEW StreamFusion expert on the gradient:

```
L_step = MSE(Thought_step, target(Thought_step))
∇ = ∂L_step/∂adapter_params
expert = StreamFusion.add_expert()
expert.train(∇)
StreamFusion.absorb(expert)  # compress into bottleneck
```

This is **self-supervised**: the target is the denoised thought from the WeightDiffusion model. The adapter learns to produce better thoughts.

### 4.3 Causal Tracing Integration

```
For each reasoning step k:
  CausalGraph.add_node(
    agent="LRE",
    action="reasoning_step",
    params={"step": k, "flags": flags_k, "temperature": temp_k},
    output={"thought": Thought_k, "confidence": confidence_k},
    state_delta={"adapter_count": +1, "thought_entropy": ΔH}
  )

After final answer:
  CausalGraph.backward_chaining(answer) → critical reasoning steps
  TaylorContribution.prune_low_contributors(non-critical steps)
```

## 5. Evaluation Strategy

### 5.1 Multi-Dimensional Reasoning Metrics

| Dimension | Metric | Implementation |
|-----------|--------|----------------|
| **Correctness** | Answer accuracy | Compare to ground truth |
| **Efficiency** | Steps to convergence | `Δthought < ε` count |
| **Consistency** | Same answer across seeds | **Pairwise with position-swap** (advanced-evaluation) |
| **Coherence** | Thought chain plausibility | LLM-as-Judge direct scoring on 1-5 scale |
| **Adaptivity** | Loss reduction per adapter | `(L_before - L_after)/L_before` |
| **Causal density** | Information-theoretic contribution | `I(Thought_k; Answer)` per step |

### 5.2 Pairwise Comparison Protocol (from advanced-evaluation)

```
For two reasoning runs A and B on the same query:

Pass 1: Compare A→answer vs B→answer (A first)
Pass 2: Compare B→answer vs A→answer (B first)

If passes agree → winner with calibrated confidence
If passes disagree → TIE (model isn't consistent enough)

Criteria: correctness, step efficiency, thought coherence
```

### 5.3 Null Hypothesis Gate (from splinter-analysis)

Before claiming any reasoning improvement:

```
E1: 10-seed reproducibility
  If Jaccard(splinter_steps) < 0.3 → effects are noise
  Skip — the reasoning trace isn't reproducible

E2: Random baseline
  If random adapter produces equivalent reasoning
  → the architecture, not the learning, is responsible
```

### 5.4 Causal Debugging Loop (from temporal-causality-engine)

```
1. Define failure F (e.g., "answered incorrectly on math query")
2. Trace backward from F through reasoning steps
3. Identify branching point (where did correct→incorrect diverge?)
4. Find what input/decision caused divergence
5. Formulate fix at branching point, not downstream
```

## 6. Component Rewiring Map

| Component | Current Role | New Role | Change Required |
|-----------|-------------|----------|-----------------|
| **PerceiverFusion** | Fuse adapter experts | Reason over thought latents | Swap adapter delta → thought tensor |
| **WeightDiffusion** | Denoise adapter weights | Denoise thought trajectories | Swap weight vector → thought matrix |
| **MetaController** | Suggest adapter flags | Suggest reasoning mode + halting | Add halting head |
| **StreamFusion** | Online adapter training | Train on reasoning gradient | New training signal (MSE on thoughts) |
| **TaylorContribution** | Prune low-contrib adapters | Prune low-impact reasoning steps | Same math, different object |
| **OverlapConsistency** | MSE over overlapping sub-matrices | MSE over parallel reasoning chains | Chain parallelization |
| **DataEncoder** | Encode (X, Y) for weight flow | Encode query for initial thought | Add query→thought projection |
| **Closed-form SVD** | Optimal LoRA weight update | Optimal thought update direction | Same linear algebra |

## 7. Other Questions We Should Be Asking

1. **What is the latent dimensionality K of a "thought"?** The Perceiver has `n_latents=16` — is this enough? Too many? The reasoning complexity of the query should determine K dynamically.

2. **Should reasoning be monotonic?** Does each step necessarily improve the thought, or can it explore (get worse briefly) then recover? The diffusion denoising schedule (noisy→clean) is monotonic by nature, but the flow matching velocity could overshoot.

3. **Where does the reasoning target come from?** In supervised reasoning, we have chain-of-thought traces. In unsupervised reasoning, the target is internal consistency (self-consistency decoding). Our approach (denoising) is unsupervised — the target is the "clean" thought.

4. **Can we measure reasoning in embedding space?** The thought trajectory in latent space may have geometric structure. Do correct reasoning paths follow geodesics? Do incorrect ones diverge?

5. **What is the inductive bias of the Perceiver bottleneck?** Latents must compete for representational capacity (softmax over latents in cross-attention). This competition may naturally implement the "attention as reasoning" mechanism.

6. **Should we initialize thoughts from query embeddings or random noise?** Diffusion models start from noise; flow matching can start from any distribution. If we start from a query encoding, the reasoning is conditional. If from noise, it's generative.

7. **How does reasoning depth scale with model capacity?** Do we need more latents for harder problems? The MetaController could learn to allocate more reasoning steps for high-uncertainty queries.
