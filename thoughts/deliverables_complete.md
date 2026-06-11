# Complete Deliverables: Dynamic K, Meta-Skill, Plugin Evaluation & MCP Design

## 1. Dynamic K — Detailed Analysis of All 6 Approaches

### Approach A: Entropy-Thresholded

**Mechanism**: Run one forward pass through the Perceiver with K_max latents. Compute attention entropy for each latent: `H_i = -Σ_j α_{ij} log α_{ij}` where α_{ij} is attention from latent i to input position j. Latents with H_i > τ (threshold) attend uniformly → useless. Drop them. Set K = number of latents with H_i ≤ τ.

**Pros**: Single forward pass, no extra training, attention entropy is already computed by the Perceiver (free signal), threshold is domain-agnostic (entropy has consistent interpretation across queries).

**Cons**: Threshold τ is a hyperparameter that needs calibration. If τ is too high, useless latents are retained (waste capacity). If τ is too low, useful latents are dropped (lose reasoning capacity). Entropy can be high for legitimate reasons (a latent attending to many relevant input regions).

**Merits**: Theoretically grounded — entropy measures specialization. A latent that attends uniformly to everything contributes nothing. Natural fit for the Perceiver architecture since attention weights are a byproduct of the forward pass.

**Risks**: Threshold calibration is brittle across domains (math queries may need different τ than creative writing). Latents may alternate between specialized and general across reasoning steps — a latent pruned at step 3 might be needed at step 7.

**Prediction**: Best overall for static K selection. For dynamic per-step adjustment, must be combined with a recovery mechanism (re-add pruned latents when needed).

### Approach B: MetaController-Predicted

**Mechanism**: Add a K-prediction head to the MetaController. Input = query encoding + current reasoning state. Output = K ∈ [K_min, K_max] (discrete or continuous + round). Train with multi-task loss: L = L_reasoning + λ_K · (K - K_optimal)² where K_optimal is found via grid search on a validation set, or via REINFORCE with reward = accuracy - α·K (penalize large K).

**Pros**: Adaptive to query difficulty, can learn query-specific K, no threshold tuning, can predict K before reasoning starts (unlike entropy which needs a forward pass).

**Cons**: Needs training data with optimal K labels, the reward signal (accuracy - α·K) is sparse and noisy, the MetaController's predictions may be inconsistent across similar queries.

**Merits**: Most flexible approach — the controller can learn complex query-K mappings. Can incorporate query metadata (length, domain, expected difficulty).

**Risks**: Reward hacking — the controller may learn to output small K for all queries to minimize α·K penalty, sacrificing accuracy. Training instability from the RL objective. Needs careful reward shaping.

**Prediction**: Best in the limit of large training data. Not suitable for low-data regimes.

### Approach C: Gradient-Based

**Mechanism**: After backward pass, compute |∂L/∂Thought_i| for each latent (gradient norm of loss w.r.t. each latent vector). Latents with near-zero gradient contribute nothing to the loss — prune them. K decreases over reasoning steps as gradient signal concentrates.

**Pros**: Directly measures which latents matter for the current objective, naturally concentrates capacity on hard reasoning steps, no thresholds (can use relative threshold like "keep top 50% by gradient norm").

**Cons**: Needs a non-zero loss (can't prune before training), backward pass is expensive, gradients may be noisy early in reasoning, requires retain_graph=True for multiple latent gradients.

**Merits**: The gradient is the most direct measure of relevance — if ∂L/∂Thought_i = 0, changing Thought_i doesn't change the answer. This is the gold standard for importance.

**Risks**: Zero gradients may indicate the latent is already optimal (no change needed) rather than useless. Premature pruning may eliminate latents that would become useful later. Vanishing gradients in deep reasoning chains.

**Prediction**: Best as a secondary signal (verify entropy-based pruning) rather than primary. Combine with approach A: entropy-prune first, then gradient-verify.

### Approach D: Spectral (SVD-Based)

**Mechanism**: After each reasoning step, compute thought covariance: C = Thoughts^T · Thoughts ∈ ℝ^{K×K}. Compute SVD: C = UΣV^T. Set K = number of singular values σ_i where σ_i > τ·σ_max (τ = 0.01-0.1). This measures the effective rank of the thought representation.

**Pros**: Theoretically optimal — the effective rank of the covariance matrix is the intrinsic dimensionality of the thought representation. No thresholds if using elbow detection. Works with any number of latents.

**Cons**: SVD is O(K²d) per step (K up to 64, so negligible). Covariance estimation needs enough samples (need multiple forward passes or a batch of inputs). If K is small (<16), SVD may be unreliable.

**Merits**: Directly measures whether the latent space is fully utilized. If thoughts span only 3 dimensions of a 16-dimensional latent space, K can be reduced to 3 without loss.

**Risks**: Elbow detection is not robust with noisy singular values. Covariance may be rank-deficient early in reasoning (all thoughts similar). SVD adds latency per step.

**Prediction**: Best for analyzing the representation post-hoc, not for real-time adjustment. Use as a diagnostic tool, not a runtime K regulator.

### Approach E: Consistency-Ensemble

**Mechanism**: Run reasoning with K ∈ {4, 8, 16, 32} in parallel (4 separate reasoning chains). Measure answer consistency: does K=4 agree with K=32? Pick smallest K that agrees with largest K on the final answer.

**Pros**: Highly robust — uses agreement as a natural stopping criterion. No thresholds, no training. The ensemble provides uncertainty estimates (if all K agree, confidence is high).

**Cons**: 4× inference cost (running 4 separate reasoning chains). May overfit to the largest K (assumes K=32 is always best, which may not hold). On ambiguous queries, small K and large K may disagree pathologically.

**Merits**: Gold standard for reliability — if K=4 gives the same answer as K=32, we can be confident K=4 is sufficient. The ensemble itself provides valuable diversity.

**Risks**: 4× cost is prohibitive for latency-sensitive applications. For very hard problems, K=32 may also be insufficient, giving false confidence. Parallel chains may interfere through shared randomness.

**Prediction**: Best for high-stakes applications where cost is secondary to reliability. Use as a validation step rather than runtime regulation.

### Approach F: Adaptive Growth

**Mechanism**: Start with K=1. After each reasoning step, compute Δthought = |Thought_step - Thought_{step-1}|. If Δthought < ε for 3 consecutive steps (reasoning has stalled), add a new latent (K+=1). Initialize from mean of existing latents + noise. Continue until convergence or K_max.

**Pros**: Minimal capacity at start, grows only when needed. Intuitive — "start simple, elaborate when stuck." Mirrors human problem-solving. No thresholds (ε can be set to a fraction of initial thought norm).

**Cons**: Can grow unbounded on very hard problems. Adding latents mid-reasoning may destabilize the chain (new latent hasn't been trained on earlier steps). Sequential addition is costly (each addition requires forward pass).

**Merits**: Biologically inspired (humans start with a simple thought and elaborate). Avoids wasted capacity on easy problems. The growth rate itself is a signal of problem difficulty.

**Risks**: Monotonic growth — once added, latents are never removed (combinatorial explosion on hard inputs). The initialization from existing latents may create redundant copies. Runtime is unpredictable (worst case: K_max steps with K_max additions).

**Prediction**: Dark horse — elegant in theory but risky in practice. Best for problems with variable difficulty where easy problems vastly outnumber hard ones.

## 2. Meta-Synthesis-Engine Skill

```markdown
# Meta-Synthesis-Engine: Unified Analysis & Design Meta-Skill

**Version**: 1.0.0  
**Derived from**: Fusion of 7 primary skills (autopoietic-inquiry-engine, advanced-evaluation, splinter-analysis, improve-codebase-architecture, code-philosophy, deep-cross-research, temporal-causality-engine) + 7 auxiliary skills (context-fundamentals, multi-agent-patterns, self-modifying-agent, verification, tool-design, safety-guard, constitutional-layer)  
**Purpose**: Maximum-depth analysis via recursive self-critique, cross-domain pattern detection, and architectural constraint propagation

## Core Architecture: 10-Phase Lens Cascade

```
Phase 0: VOID → Suspend all assumptions
Phase 1: DECOMPOSE → Orthogonal facets
Phase 2: LENS CASCADE → Sequential, stacked filters
Phase 3: CONVERGE → Cross-lens patterns
Phase 4: CAUSAL MAP → Temporal dependency graph
Phase 5: SPLINTER SCAN → Anomaly detection
Phase 6: DEEPEN → Architecture improvement
Phase 7: VERIFY → Quality gates + constraints
Phase 8: METRICS → Multi-dimensional evaluation
Phase 9: SYNTHESIZE → Confidence-weighted output
Phase 10: RECURSE → Apply skill to itself
```

### Phase 0: VOID
Document all assumptions explicitly. Prepend "What if ¬[assumption]?" to each.
Examples from this session: "What if our evaluation metrics DON'T capture reasoning quality?" → we pivoted to 6-dimensional metrics.

### Phase 1: DECOMPOSE  
Break into 5-8 orthogonal facets using deep-cross-research methodology. For each facet, generate 3-5 search queries targeting foundational results, recent advances, specific mechanisms, and known limitations.

### Phase 2: LENS CASCADE (Key Innovation)
Unlike parallel lenses (deep-cross-research v3), use STACKED FILTERS. Each lens receives the OUTPUT of the previous lens plus a "blind spot alert":

````
Raw → ANALOGICAL → DIALECTICAL → BLENDING → SYSTEMS → ABDUCTIVE → TRAJECTORY → METACOGNITIVE → INSPIRATION → ADVERSARIAL → Output
````

Each lens:
1. Takes prior lens output + "what the prior lens missed" as input
2. Applies its transformation
3. Appends its own "blind spot alert" for the next lens

Cost: Sequential (9× forward passes). Benefit: Later lenses catch blind spots of earlier ones.

### Phase 3: CONVERGE
Detect patterns across lenses. Flag contradictions (lens A says X, lens B says ¬X). Weight: if ≥5 lenses agree, confidence = HIGH. If 3 disagree, flag for re-investigation.

### Phase 4: CAUSAL MAP
Build DAG of (action → observation → state change) using temporal-causality-engine. Identify branching points (where did actual diverge from expected?). Generate counterfactuals.

### Phase 5: SPLINTER SCAN
Apply splinter-analysis VADT detection to find anomalously good/bad sub-components. Null hypothesis gate: verify reproducibility before claiming improvement.

### Phase 6: DEEPEN
Apply improve-codebase-architecture: deletion test on each module. Surface shallow modules. Propose deepening.

### Phase 7: VERIFY
Check against verification skill quality gates. Check safety-guard constraints. Check constitutional-layer mutation rules. Check code-philosophy 5 laws.

### Phase 8: METRICS
Evaluate using advanced-evaluation multi-dimensional framework: correctness, efficiency, consistency, coherence, adaptivity, causal density. Use pairwise comparison with position-swap.

### Phase 9: SYNTHESIZE
Produce confidence-weighted recommendations. Each output must have P(true). Flag provisional findings (confidence < 0.6). Sort by expected value.

### Phase 10: RECURSE
Apply phases 0-9 to the skill's own output. Track which phases generate most value. Prune phases that never contribute after 3 invocations.

## Consolidation Rules

1. **Safety-first**: If two skills prescribe conflicting protocols → the MORE RESTRICTIVE one wins
2. **Test before tune**: If a skill has a quantitative threshold → test it empirically, don't assume it's correct
3. **Quality gate**: Every output must pass verification before acceptance
4. **5 Laws compliance**: Every component interface must satisfy code-philosophy (guard clauses, parse don't validate, atomic predictability, fail fast, intentional naming)
5. **Self-assessment**: Every 5th invocation applies the skill to itself recursively
6. **Channel routing**: Every finding must be tagged with its destination channel (codebase, skill, doc, config, theory)
7. **Negative space**: Explicitly document what was NOT found and why
```

## 3. Plugin Evaluation — Merits, Shortcomings & Extensions

### Plugin Inventory Analysis

| Plugin | Lines | Type | Health | Missing |
|--------|-------|------|--------|---------|
| batch-deploy.ts | 127 | Tool | ✅ Works | No retry logic, no agent-type validation against registry |
| batch_deploy_server.py | 861 | MCP Server | ⚠️ Overengineered | MCP protocol spec drift, hardcoded ports, fragile threading |
| cognitive-friction.ts | 147 | Monitor | ✅ Novel | No per-tool friction weighting, HJB params not configurable |
| context-budget.ts | 80 | Monitor | ⚠️ Underestimates | Token count heuristic is crude (÷4), no actual context window introspection |
| dmn-idle-consolidation.ts | 69 | Background | ❌ Shallow | No actual memory operations, just system prompt injection |
| failure-propagation-enforcer.ts | 318 | Enforcer | ✅ Robust | No BYZANTINE failure detection, no recovery suggestions |
| goal-drift-index.ts | — | Monitor | — | Not reviewed |
| goal-homeostasis.ts | — | Monitor | — | Not reviewed |
| lazy-mcp.ts | — | Loader | — | Not reviewed |
| lazy-skills.ts | — | Loader | — | Not reviewed |
| metacognitive-accumulator.ts | — | Monitor | — | Not reviewed |
| plugin-event-bus.ts | — | Core | — | Not reviewed |
| prompt-compressor.ts | — | Tool | — | Not reviewed |
| ptr-bounded-execution.ts | — | Enforcer | — | Not reviewed |
| reliability-tracker.ts | — | Monitor | — | Not reviewed |
| self-compress.ts | — | Tool | — | Not reviewed |
| verbal-confidence.ts | — | Monitor | — | Not reviewed |

### Deep Evaluation of Key Plugins

#### batch-deploy (ts + py)
**Merits**: Dual-path dispatch (fast local LLM + fallback CLI) is clever and robust. The TypeScript plugin properly uses OpenCode's native spawnTask API. The MCP server provides a fallback when the TS plugin can't run.

**Shortcomings**: 
- The MCP server (`batch_deploy_server.py`) tries to call the LLM directly via HTTP, bypassing OpenCode's provider configuration entirely. This means it uses a hardcoded model, temperature, and URL — any provider configuration in opencode.json is ignored.
- 861 lines of Python for what should be a simple dispatcher. Over 50% is error handling for edge cases that rarely occur.
- The system prompt mapping (_AGENT_SYSTEM_PROMPTS) hardcodes prompts that duplicate what's already in the agent definitions. If an agent's system prompt changes in opencode.json, this plugin won't reflect it.
- No agent-type validation against the actual registry — it accepts any string as subagent_type, even if the type doesn't exist.

**Room for Improvement**: 
1. Remove the MCP server entirely — the TS plugin handles all functionality via spawnTask. The Python MCP server was a workaround for when the TS plugin didn't work.
2. Add agent registry validation: before deploying, check subagent_type against the actual configured agents in opencode.json.
3. Add retry logic: if an agent fails, retry once with a different path (e.g., if direct_llm failed, retry via fallback_cli).
4. Support custom provider configuration: read from opencode.json instead of hardcoding.

#### cognitive-friction.ts
**Merits**: Novel concept — HJB optimal stopping from control theory applied to agent reasoning. The three friction dimensions (spatial, temporal, epistemic) are well-chosen. The stop signal injection is non-intrusive (appends to system prompt, doesn't interrupt execution).

**Shortcomings**:
- VOI (Value of Information) computation is naive: `1.0 / (1.0 + toolCalls * 0.1)`. This assumes diminishing returns are uniform across all tools, which is false — a single well-placed web search can have much higher VOI than 10 tool calls. 
- No per-tool friction weighting: calling `task` (heavy) and `read` (light) incur the same friction increment.
- The friction thresholds (MAX_SPATIAL=5, MAX_TEMPORAL=300s, MAX_EPISTEMIC=50000) are hardcoded guesses, not learned from data.
- Only injects into system prompt — the agent can ignore the signal with no consequence.

**Extensions**:
1. Make VOI model-aware: track which tools historically produced high-value outputs and weight them higher.
2. Add enforcement mode: when friction exceeds VOI by >2×, force-stop the tool call (not just suggest).
3. Make all thresholds configurable via opencode.json plugin config.
4. Add per-session friction visualization to the event bus for dashboard integration.

#### context-budget.ts
**Merits**: Essential functionality — agents perform better when aware of their context budget. The urgency levels (info/warning/critical) are well-calibrated.

**Shortcomings**:
- Token estimation uses a crude heuristic (÷4 chars per token), which can be off by 2-3× depending on the model's tokenizer.
- No actual introspection of the model's context window — it assumes a hardcoded 720K default.
- The 60-second throttle means a rapid sequence of tool calls won't get timely budget updates.
- When critical, it only suggests compression — doesn't automatically trigger it.

**Extensions**:
1. Use the model's actual tokenizer for estimation (requires API access, but much more accurate).
2. Query the model's actual max context length from the provider configuration (open code.json).
3. Add automatic compression trigger at critical threshold: load context-compression skill automatically.
4. Report budget to the event bus for cross-plugin coordination (e.g., cognitive-friction can use budget data).

#### dmn-idle-consolidation.ts
**Merits**: Interesting concept — idle-time consolidation inspired by neuroscience. The three-tier memory model (hot/warm/cold) is well-structured.

**Shortcomings**: 
- **Does nothing**: The plugin only injects a prompt suggestion during idle time. It doesn't actually consolidate memory, extract patterns, or store anything. It's a toy implementation of a potentially powerful idea.
- No integration with the actual memory system (episodic memory, memory_search, memory_store).
- The 3-consolidation limit is arbitrary and not based on any theory.
- No way to retrieve past consolidations across sessions.

**Extensions**: 
1. Implement actual memory operations: use memory_store to persist consolidations, use memory_search to retrieve relevant past consolidations.
2. Add consolidation quality metric: not all idle periods produce valuable consolidations. Track which consolidations were later useful.
3. Add crossing consolidation: compare patterns across sessions, not just within a session.
4. Add active retrieval: when a new task begins, check if DMN has relevant consolidations to inject.

## 4. MCP Server & Plugin Design Document (for Handoff)

### Overview

This document specifies 5 MCP servers and 3 OpenCode plugins for the agentic regulation infrastructure. Each specification includes: purpose, protocol, API surface, data model, integration points, error handling, and testing strategy.

---

### MCP Server 1: `causal-trace-mcp`

**Purpose**: Persistent causal graph across agent sessions. Enables "why did this happen?" queries and counterfactual analysis.

**Protocol**: JSON-RPC over stdio (MCP standard transport).

**Data Model**:
```
Node:
  id: UUID
  timestamp: ISO-8601
  agent_id: string
  session_id: string
  tool: string          (tool name, e.g., "task", "bash", "write")
  input_params: dict    (redacted for sensitive fields — add redaction list)
  output_summary: string (first 500 chars, full output stored in file)
  state_delta: dict     (keys that changed: { key: { before, after } })
  kind: enum(action | observation | state_change)

Edge:
  from_node: UUID
  to_node: UUID
  label: enum(caused | enabled | influenced)
  confidence: float [0,1]

Checkpoint:
  id: UUID
  timestamp: ISO-8601
  node_snapshot: [UUID]  (list of node IDs at this checkpoint)
  description: string
```

**API Surface**:
```
POST /trace/action
  Creates a causal node from an agent action.
  Body: { agent_id, tool, params, output, state_before, state_after }
  Returns: { node_id }

POST /trace/link
  Creates a causal edge between two nodes.
  Body: { from_node, to_node, label, confidence }
  Returns: { edge_id }

GET /trace/path-to?outcome=X&limit=10
  Walks backward from outcome node through causal edges.
  Returns: [{ step, agent, tool, state_delta_before, state_delta_after }]

GET /trace/counterfactual?node_id=X&alt_param=Y:Z
  Simulates what would happen if node X's param Y had value Z instead.
  Returns: { likelihood, impact, score, alternative_chain: [...] }

POST /trace/checkpoint
  Creates a checkpoint of current graph state.
  Body: { description }
  Returns: { checkpoint_id }

POST /trace/rollback
  Restores graph to a prior checkpoint.
  Body: { checkpoint_id }
  Returns: { restored_node_count, removed_node_count }

GET /trace/summary?session_id=X
  Returns aggregate statistics for a session.
  Returns: { total_nodes, total_edges, branching_points: [...], 
             bottlenecks: [tool_names], avg_chain_length }
```

**Integration Points**:
- agent: hooks into `tool.execute.after` to auto-record actions
- MetaController: queries `/trace/path-to` before deciding to halt
- TaylorContribution: uses `/trace/counterfactual` for step importance estimation
- Latent Reasoning Engine: posts each reasoning step as a causal node

**Edge Cases**:
- Node with no edges: orphan node, flagged in /trace/summary
- Cycles: reject edge creation that would create a cycle (topological sort check)
- Missing state_before: use None, mark node as "partial fidelity"
- Concurrent sessions: nodes from different sessions are stored separately (session_id index)

**Testing Strategy**:
1. Unit: create nodes, create edges, verify graph is acyclic
2. Unit: counterfactual returns correct likelihood/impact for known chain
3. Integration: hook into a real session, verify nodes are created for each tool call
4. Stress: 1000 nodes, 2000 edges, verify query latency < 100ms

---

### MCP Server 2: `constraint-proxy-mcp`

**Purpose**: Hard enforcement of constitutional rules. Sits BETWEEN the agent and its tools. Intercepts, checks, allows/modifies/blocks each tool call.

**Protocol**: JSON-RPC over stdio. Agent → constraint-proxy → actual tool.

**Data Model**:
```
Constraint:
  id: UUID
  pattern: string        (glob or regex matching tool call pattern)
  tool: string           (target tool: "write", "bash", "edit", "task", etc.)
  action: enum(allow | block | modify | warn)
  reason: string         (human-readable explanation)
  modification: dict     (how to modify the call if action=modify)
  priority: int          (higher = applied first, conflicts resolved by highest priority)
  expires_at: datetime   (optional auto-expiry for temporary constraints)

ConstraintRegistry:
  version: int
  constraints: [Constraint]
  invariants: [string]   (immutable rules — cannot be modified through this API, only via constitutional-layer)
```

**API Surface**:
```
POST /check
  Intercepts a tool call. Checks against all active constraints.
  Body: { tool, params, agent_id, session_id }
  Returns: { decision: "allow" | "block" | "modify" | "warn",
             reason: "...", 
             modified_params: {...} | null,
             matched_constraint: constraint_id | null }

POST /constraint/register
  Registers a new constraint (must not conflict with invariants).
  Body: { pattern, tool, action, reason, modification, priority, expires_at }
  Returns: { constraint_id }

GET /constraint/list
  Lists all active constraints.
  Returns: { constraints: [...], invariants: [...] }

POST /constraint/update/{id}
  Updates a constraint (priority, action, pattern, expiry).
  Body: { ... }
  Returns: { constraint_id }

DELETE /constraint/{id}
  Removes a constraint.

POST /audit/check-all
  Runs ALL active constraints against ALL registered invariants.
  Returns: { passed: bool, conflicts: [{ invariant, constraint, reason }] }
```

**Integration Points**:
- The OpenCode runtime tool execution pipeline registers the proxy as a middleware
- All `write`, `edit`, `bash`, `task` calls pass through /check before execution
- The constitutional-layer skill seeds the registry with immutable invariants

**Edge Cases**:
- Multiple constraints match → highest priority wins. If tie, most restrictive action wins (block > modify > warn > allow).
- Proxy goes down → agent should either block all calls (safe) or pass through (unsafe). Configuration choice.
- Self-modification attempt → invariant-level constraints block changes to invariants. Only constitutional-layer can modify.
- Circular constraint → constraint trying to modify the constraint registry itself. Detection: if tool == "/check" or "/constraint/*", apply a special system-level handler.

**Testing Strategy**:
1. Unit: register constraint, call /check, verify decision matches
2. Unit: two overlapping constraints, verify priority resolution
3. Integration: try to modify an invariant via /constraint/register → verify rejection
4. Integration: proxy failure → verify safe-fail behavior

---

### MCP Server 3: `benchmark-orch-mcp`

**Purpose**: Standardized reasoning evaluation across 6 dimensions with pairwise comparison.

**Protocol**: JSON-RPC over stdio.

**Data Model**:
```
Benchmark:
  id: UUID
  name: string          ("gsm8k", "math", "bbh", etc.)
  version: string
  config: dict          (split, metrics, num_samples)

EvaluationRun:
  id: UUID
  benchmark_id: UUID
  model_name: string
  timestamp: ISO-8601
  status: enum(running | completed | failed)
  results: dict         (
    correctness: float,
    efficiency: dict(steps: int, tokens: int),
    consistency: dict(score: float, method: "pairwise_swap"),
    coherence: dict(score: float, judge: "gpt-4"),
    adaptivity: dict(loss_reduction: float, experts_trained: int),
    causal_density: dict(mutual_info: float, top_k_steps: [int])
  )
  raw_outputs: [dict]   (query, system_response, ground_truth, reasoning_trace)
```

**API Surface**:
```
POST /benchmark/run
  Runs a benchmark evaluation.
  Body: { benchmark_id, model_endpoint, config_overrides }
  Returns: { run_id, status, progress_token }

GET /benchmark/status/{run_id}
  Returns status and partial results (if streaming).

POST /benchmark/compare
  Runs pairwise comparison between two model outputs.
  Body: { run_a_id, run_b_id, swap_positions: true }
  Returns: { winner, confidence, per_criterion_scores }

GET /benchmark/results/{run_id}
  Returns complete evaluation results for a run.

POST /benchmark/regression
  Compares a new run against historical runs.
  Body: { run_id, baseline_run_ids: [...] }
  Returns: { regression: bool, delta: dict, regressed_metrics: [str] }

POST /benchmark/register
  Registers a new benchmark.
  Body: { name, version, dataset_path, split_config, metric_config }
  Returns: { benchmark_id }
```

**Integration Points**:
- Latent Reasoning Engine registers as a model endpoint
- Advanced-evaluation skill uses pairwise comparison endpoint
- Context-budget plugin feeds into efficiency metrics
- Cognitive-friction plugin feeds into adaptivity metrics

**Edge Cases**:
- Benchmark dataset not cached → download and cache locally, report progress
- Model endpoint unreachable → retry with configurable backoff, fail after N attempts
- Pairwise swap inconsistency → confidence = 0.5, verdict = TIE, flag for human review
- Run timeout → partial results saved, run marked as "timeout" with partial metrics

**Testing Strategy**:
1. Unit: register benchmark, run single query, verify result format
2. Integration: run full benchmark on a known model, compare against published scores
3. Integration: pairwise comparison with known identical outputs → verify TIE
4. Regression: run same model twice, verify regression delta ≈ 0

---

### MCP Server 4: `latent-viz-mcp`

**Purpose**: Real-time streaming visualization of Perceiver latent thoughts during reasoning.

**Protocol**: Server-Sent Events (SSE) over HTTP for streaming. JSON-RPC over stdio for control.

**Data Model**:
```
ThoughtSnapshot:
  step: int
  latents: [[float]]    (K × d, PCA-projected to 2D or 3D)
  attention: [[float]]  (K × input_len, attention weights)
  timestamp: float      (ms since start)
  entropy: [float]      (per-latent attention entropy)
  k_current: int        (if dynamic K is enabled)

VisualizationState:
  snapshots: [ThoughtSnapshot]
  pca_model: dict       (fitted PCA: mean, components, explained_variance)
  selected_run: UUID
  status: enum(recording | paused | stopped)
```

**API Surface**:
```
POST /viz/start
  Starts recording a reasoning session.
  Body: { model_endpoint, query }
  Returns: { session_id, sse_url }

POST /viz/stop/{session_id}
  Stops recording.

GET /viz/sse/{session_id}
  SSE stream: receives ThoughtSnapshot as events.

GET /viz/trajectory/{session_id}
  Returns complete trajectory as JSON (all snapshots).

POST /viz/animate
  Generates a video/animation of the reasoning trajectory.
  Body: { session_id, format: "mp4" | "gif" | "json" }
  Returns: { url, format, duration_seconds }

POST /viz/analyze
  Analyzes a completed session for insights.
  Body: { session_id }
  Returns: { convergence_rate, attention_entropy_trend, 
             latent_specialization, bottleneck_utilization }
```

**Integration Points**:
- WeightDiffusion streams thought states during denoising
- MetaController streams flags and temperature per step
- Entropy-thresholded Dynamic K (Approach A) sends K decisions

**Edge Cases**:
- High-dimensional latents (K=64, d=256) → PCA projection on client, fit incrementally
- Streaming buffer overflow → sliding window of last 100 snapshots, older snapshots persisted to disk
- No browser client → SSE works with any HTTP client, can also output to file

---

### MCP Server 5: `agent-registry-mcp`

**Purpose**: Central registry of agent types, their permissions, and their capabilities. Enables validation and discovery.

**Data Model**:
```
AgentType:
  name: string
  description: string
  task_budget: int
  mode: enum(agent | subagent)
  tools: [string]            (list of permitted tools)
  tool_permissions: dict     (tool → "allow" | "deny")
  task_permissions: dict     (agent_type → "allow" | "deny")
  hidden: bool
  model: string | null       (specific model, if overridden)
  parent: string | null      (for subagents: which agent can spawn this)
```

**API Surface**:
```
GET /registry/list
  Lists all registered agent types with their metadata.
  Returns: { agents: [AgentType] }

GET /registry/validate?agent_type=X
  Validates that an agent type exists and is not hidden.
  Returns: { valid: bool, agent: AgentType | null, error: string | null }

GET /registry/permissions?agent_type=X&tool=Y
  Checks whether agent X is permitted to use tool Y.
  Returns: { permitted: bool, reason: "allow" | "deny" | "not_found" }

GET /registry/graph
  Returns the agent hierarchy as a directed graph (who can spawn whom).
  Returns: { nodes: [{name, tools, budget}], edges: [{from, to}] }
```

---

### OpenCode Plugin Specifications

#### Plugin 1: `skill-dependency-resolver.ts`

**Purpose**: When loading a skill (`skill()` tool call), automatically load ALL prerequisite skills. Prevents "skill not loaded" errors in complex pipelines.

**Mechanism**: Hook into `experimental.chat.system.transform`. When a skill is loaded via `skill()` tool, check a dependency registry, recursively load all prerequisites.

**Dependency Registry** (in-code):
```
meta-synthesis-engine → [autopoietic-inquiry-engine, advanced-evaluation, splinter-analysis, 
                         improve-codebase-architecture, code-philosophy, deep-cross-research, 
                         temporal-causality-engine]
deep-cross-research → [autopoietic-inquiry-engine]
splinter-analysis → [deep-cross-research, self-modifying-agent, meta-evolution]
```

**Implementation Sketch** (50-80 lines):
- Load dependency map from a JSON file in ~/.config/opencode/skills/
- Intercept `skill()` tool calls
- For each skill, check if prerequisites are already loaded (via a Set)
- If not, call `skill()` recursively for each prerequisite
- Guard against circular dependencies (depth counter, max depth = 5)

**Integration**: Uses the `tool.execute.before` hook (if available) or `experimental.chat.system.transform`.

#### Plugin 2: `agent-budget-tracker.ts`

**Purpose**: Track task_budget consumption across agents in real-time. Warn when an agent is approaching its budget. Detect runaway agents.

**Mechanism**: Hook into `tool.execute.after`, track cumulative task_budget for each agent. When approaching limit (80%), inject warning. When exceeded, inject stop command.

**Budget Sources**: Read from opencode.json agent configuration. Fallback to defaults if not configured.

**Implementation Sketch** (60-90 lines):
- Parse agent config from opencode.json on initialization
- Maintain a Map<agent_id, {budget_consumed, budget_total, tool_calls}>
- On each tool call, increment budget_consumed by estimated cost (~1 per call, ~5 per task)
- At 80%: inject warning to system prompt
- At 100%: inject "STOP — budget exhausted" — the agent should stop delegating to this subagent

#### Plugin 3: `failure-dashboard.ts`

**Purpose**: Real-time dashboard for the Failure Propagation Protocol. Track failures across agents, visualize escalation chains, suggest recovery strategies.

**Mechanism**: Read from the failure-logs directory (populated by failure-propagation-enforcer.ts). Expose via HTTP or file-based dashboard.

**Implementation Sketch** (100-150 lines):
- Watch the failure-logs directory for new entries
- Parse JSONL entries into in-memory store
- Group by session_id, agent_id, severity
- Compute aggregate metrics: BLOCKER rate, WARNING rate, most failure-prone tools, most failure-prone agent types
- Expose via simple HTTP server on localhost:9876
- Optional: inject summary into system prompt every N failures

---

### Consolidation & Handoff Checklist

- [ ] MCP server specs: 5 servers defined with API surface, data model, edge cases, testing strategy
- [ ] Plugin specs: 3 plugins defined with mechanism, implementation sketch, integration points
- [ ] All specs include error handling and edge cases
- [ ] Integration points specify which existing components hook into which servers
- [ ] Testing strategy defined for each server
- [ ] Priority order: causal-trace-mcp (P0) > constraint-proxy-mcp (P0) > agent-registry-mcp (P1) > benchmark-orch-mcp (P1) > latent-viz-mcp (P2)
