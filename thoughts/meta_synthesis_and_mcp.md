# Dynamic K, Unified Meta-Skill & MCP Plugin Design

## 1. Dynamic K: Approaches for Evaluation

K is the number of latent slots in the Perceiver bottleneck. It controls reasoning capacity. Too few → representational collapse (all thoughts identical). Too many → diffusion (thoughts don't interact). We need to evaluate approaches that dynamically set K per query.

### 1.1 Approaches to Evaluate

| # | Approach | Mechanism | Cost | Risk |
|---|----------|-----------|------|------|
| **A** | **Entropy-thresholded** | Run forward pass with K_max, measure attention entropy per latent. Drop latents with entropy > threshold (they attend uniformly → useless). Set K = number of low-entropy latents. | 1 forward pass | Threshold is a hyperparameter |
| **B** | **MetaController-predicted** | Add a K-prediction head to MetaController. Input = query encoding. Output = K ∈ [K_min, K_max]. Train with reward = reasoning accuracy - α·K (penalize large K). | Negligible | Needs training data |
| **C** | **Gradient-based** | Compute |∂L/∂Thought_i| for each latent after backward pass. Prune latents with near-zero gradient. K decreases over reasoning steps. | 1 backward pass | Needs non-zero loss |
| **D** | **Spectral (SVD-based)** | Compute SVD of thought covariance matrix C = Thoughts^T · Thoughts. Set K = number of singular values above threshold τ·σ_max. | O(K²d) | SVD cost per step |
| **E** | **Consistency-ensemble** | Run reasoning with K ∈ {4, 8, 16, 32}. Measure answer consistency across runs. Pick smallest K that agrees with largest K on final answer. | 4 forward passes | 4× inference cost |
| **F** | **Adaptive growth** | Start with K=1. After each reasoning step, if `Δthought < ε`, add a new latent (K+=1). Latent is initialized as average of existing latents + noise. | K forward passes | Can grow unbounded |

### 1.2 Evaluation Protocol

For each approach, measure on a diverse reasoning benchmark:

| Metric | How |
|--------|-----|
| **Accuracy vs K** | Does higher K always help? Is there a plateau? |
| **Efficiency** | (Accuracy / K) — accuracy per unit capacity |
| **Stability** | Variance of K across similar queries |
| **Cost** | Extra FLOPs per query |
| **Interpretability** | Do low-entropy latents correspond to distinct reasoning steps? |

### 1.3 Hypothesis

**Approach A (entropy-thresholded) will win** because:
1. Attention entropy is a natural measure of latent specialization — it's already computed by the Perceiver
2. The threshold can be set once and hold across domains (unlike gradient-based which needs a loss)
3. It's a single forward pass, no extra training

**Approach F (adaptive growth)** is the dark horse — it mirrors how humans reason (start with a simple thought, elaborate when stuck). But it may grow unbounded on hard problems.

---

## 2. The Meta-Synthesis Skill

### 2.1 Synthetic Meta-Synthesis Skill

The following skill fuses all 7 applied skills (autopoietic-inquiry-engine, advanced-evaluation, splinter-analysis, improve-codebase-architecture, code-philosophy, deep-cross-research, temporal-causality-engine) plus insights from (context-fundamentals, multi-agent-patterns, self-modifying-agent, verification, tool-design, safety-guard, and the constitutional-layer) into a single coherent meta-skill.

```
Skill: meta-synthesis-engine
Version: 1.0.0
Derived from: fusion of 7 skills + 7 auxiliary skills across 14 sessions
Purpose: Maximum-depth analysis through recursive self-critique,
         cross-domain pattern detection, and architectural constraint propagation
```

**Core Architecture:**

```
Phase 0: VOID — Suspend all assumptions. Document them explicitly. Set aside.
Phase 1: DECOMPOSE — Break problem into orthogonal facets (deep-cross-research)
Phase 2: LENS CASCADE — Apply N cognitive lenses in parallel (autopoietic-inquiry)
Phase 3: CONVERGE — Detect cross-lens patterns, flag contradictions (advanced-evaluation)
Phase 4: CAUSAL MAP — Build temporal dependency graph (temporal-causality-engine)
Phase 5: SPLINTER SCAN — Find anomalously good/bad sub-components (splinter-analysis)
Phase 6: ARCHITECTURE DEEPEN — Surface shallow modules, propose deepening (improve-architecture)
Phase 7: CODE PHILOSOPHY CHECK — Verify: early exit, parse don't validate, atomic predictability, fail fast, intentional naming
Phase 8: VERIFICATION — Quality gates, constraint checks, safety guard (verification, safety-guard)
Phase 9: META-ASSESSMENT — Apply the skill to itself: what did we miss? (self-modifying-agent)
Phase 10: SYNTHESIS — Produce unified output with confidence-weighted recommendations
```

**Key Innovation — Lens Cascade:**

Instead of 9 independent lenses, use a **stacked filter**: each lens receives the OUTPUT of the previous lens, not the raw findings. This creates a processing pipeline:

```
Raw Findings → ANALOGICAL → DIALECTICAL → BLENDING → SYSTEMS → 
ABDUCTIVE → TRAJECTORY → METACOGNITIVE → INSPIRATION → ADVERSARIAL → Synthesized

Each lens:
  1. Takes prior lens output as input
  2. Applies its transformation
  3. Adds a "blind spot alert" — what did the prior lens miss?
```

This is more powerful than parallel lenses because later lenses can catch blind spots introduced by earlier ones. The cost: sequential, not parallel.

**Skill Consolidation Rules:**
1. If two skills prescribe conflicting protocols → the MORE RESTRICTIVE one wins (safety-first)
2. If a skill has a quantitative threshold → test it, don't tune it (null hypothesis gate from splinter-analysis)
3. Every output must pass a verification quality gate (from verification skill)
4. Every component interface must satisfy the code-philosophy 5 laws
5. The skill applies to itself on every 5th invocation (recursive self-assessment)

### 2.2 Improvements from This Session

| Observation | Improvement |
|-------------|-------------|
| Disconnected component evaluation hid system-level failures | Add "cross-component consistency" check: all modules must agree on the same objective metric |
| Data scarcity blamed but information-theoretic insufficiency was root cause | Add mutual information check: I(input_conditioning; optimal_output) before training |
| No halting criterion for iterative processes | Add convergence detector: `Δ < ε` for 3 consecutive iterations → halt |
| Lots of silent fallbacks (`ctx = torch.randn`) | Add fail-fast: crash on missing data with descriptive error |
| Flow matching on observed trajectories learns noise, not optimal paths | Add closed-form target: train velocity field toward optimal transport, not just observed steps |
| No causal tracing infrastructure | Add causal graph as core data structure, not optional add-on |
| Evaluation overfit on single metric (loss) | Add multi-dimensional evaluation suite with pairwise comparison |

### 2.3 Further Augmentations via the Skills Themselves

**Applying meta-synthesis-engine to itself recursively:**

1. **VOID** (Phase 0): Assumption that lens cascade is better than parallel — test this. Hypothesis: sequential cascade finds deeper patterns but parallel lenses have higher coverage. Optimal may be hybrid: 3 parallel groups of 3 sequential lenses.

2. **DECOMPOSE** (Phase 1): The 7-skill fusion omits (board-meeting, caveman, context-compression, diagnose). These should be evaluated for inclusion:
   - board-meeting: For multi-agent decomposition of the analysis task
   - caveman: For ultra-compressed communication between phases
   - context-compression: For long-running analyses that exceed context windows
   - diagnose: For when the analysis reveals a bug/failure pattern

3. **CONVERGE** (Phase 3): Add confidence calibration from advanced-evaluation. Every finding must have `P(true)`. Syntheses with confidence < 0.6 are "provisional" and must be flagged.

4. **SPLINTER SCAN** (Phase 5): Applied to the skill itself — which phases produce anomalously high-value insights? Track across invocations. Prune phases that never contribute.

5. **ARCHITECTURE DEEPEN** (Phase 6): The meta-skill has a shallow module: Phase 4 (Causal Map). It currently just records nodes and edges. Deepen it: add automatic counterfactual generation (from temporal-causality-engine).

6. **VERIFICATION** (Phase 8): Add constitutional-layer enforcement: the skill cannot suggest modifications to its own mutation rules. This prevents runaway self-modification.

---

## 3. MCP Server & OpenCode Plugin Proposals

### 3.1 Causal Trace Server (`causal-trace-mcp`)

Purpose: Persistent causal graph across agent sessions. Solves the "why did this happen" problem that no amount of prompting can fix.

**Protocol:**
```
// Record an action
POST /trace/action
{ agent_id, tool, params, output, state_before, state_after }

// Query causal path to an outcome
GET /trace/path-to?outcome=answer_incorrect&limit=10
→ [{ step, agent, tool, delta }]

// Counterfactual: what if this param differed?
GET /trace/counterfactual?node_id=n5&alt_param=lr:0.001
→ [{ likelihood, impact, score }]

// Checkpoint + rollback
POST /trace/checkpoint
POST /trace/rollback/{checkpoint_id}
```

**Why MCP, not a library:** The causal graph must persist across agent invocations, across sessions, and across agent boundaries. An MCP server provides a single source of truth that any agent can query.

**Integration with our stack:**
- Each reasoning step in the Latent Reasoning Engine posts to `/trace/action`
- The MetaController queries `/trace/path-to` before deciding to halt
- TaylorContribution uses `/trace/counterfactual` to estimate step importance

### 3.2 Constraint Enforcement Proxy (`constraint-proxy-mcp`)

Purpose: Hard enforcement of constitutional rules. Sits BETWEEN the agent and its tools.

**Architecture:**
```
Agent → constraint-proxy-mcp → actual tools (bash, write, edit, task)
```

Each tool call is intercepted, checked against a constraint registry, and either allowed, modified, or blocked:

```
// Register a constraint
POST /constraint
{ pattern: "write.*\.pt", action: "block", reason: "Model files are generated, not edited" }

// Intercept a tool call
POST /check
{ tool: "write", params: { path: "weight_flow_model.pt" } }
→ { decision: "block", reason: "Model files are generated, not edited", suggestion: "Use torch.save() in script instead" }
```

**Why this matters:** Our system generates model files (`weight_flow_model.pt`, `diffusion_weight_flow.pt`). These should NEVER be hand-edited. The constraint proxy prevents this structurally (code-philosophy: Make Illegal States Unrepresentable).

### 3.3 Reasoning Benchmark Orchestrator (`benchmark-orch-mcp`)

Purpose: Standardized evaluation of reasoning systems. Runs the multi-dimensional evaluation suite from the analysis.

**Features:**
- Manages a registry of reasoning benchmarks (GSM8K, MATH, BBH, etc.)
- Runs query→System→Answer→Judge pipeline
- Computes all 6 evaluation dimensions (correctness, efficiency, consistency, coherence, adaptivity, causal density)
- Stores results in versioned database for regression tracking

**Integration:**
- The Latent Reasoning Engine registers itself as a "model" with the orchestrator
- The orchestrator runs the evaluation protocol from Advanced-Evaluation (pairwise with position-swap)
- Results flow back into MetaController training as reward signal

### 3.4 Latent Visualization Server (`latent-viz-mcp`)

Purpose: Real-time visualization of the Perceiver's latent thoughts during reasoning.

**Streams:**
- Thought trajectories: `Thought_i(t)` over reasoning steps → PCA/UMAP projection
- Attention patterns: which latents attend to which input tokens
- K evolution: if dynamic K is enabled, how K changes per step
- Splinter scores: which steps are "skills" vs "deficiencies"

**Use case:** Debug why a reasoning chain failed. Visualize the exact moment the thought trajectory diverged from the correct path.

### 3.5 What Wasn't Considered: OpenCode Plugin Ideas

| Plugin | Purpose | Priority |
|--------|---------|----------|
| **Skill Dependency Resolver** | When loading a skill, automatically load its prerequisites (e.g., meta-synthesis-engine → loads all 7 sub-skills) | P0 |
| **Constraint Registry UI** | GUI for viewing/editing the constraint registry that constraint-proxy-mcp enforces | P1 |
| **Cross-Session Memory Browser** | Browse episodic memory across sessions; visualize which patterns recur | P1 |
| **Agent Budget Tracker** | Track task_budget usage across agents; detect runaway agents exceeding their budget | P0 |
| **Permission Schema Visualizer** | Visualize which agents can call which tools as a directed graph; detect unintended permission paths | P2 |
| **Failure Propagation Dashboard** | Real-time view of the Failure Propagation Protocol — see which sub-agents failed, which escalated | P1 |
| **Compaction Forecaster** | Predict when context will reach 70% threshold; suggest optimal compression strategy | P2 |
