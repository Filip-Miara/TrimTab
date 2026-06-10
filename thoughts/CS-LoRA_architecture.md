# CS-LoRA: Cross-Slot LoRA — Cross-Layer Iterative Expert Binding via Persistent Slot Vectors

## Abstract

We fuse **SlotLoRA** (slot-attention-based expert binding within a layer) with **X-LoRA** (cross-layer attention between LoRA modules at different depths) into a single architecture where **K persistent slot vectors** are shared across all L transformer layers. Slots represent functional roles (style, content, syntax, etc.) that bind to different LoRA experts at each layer, and propagate their state forward via gated cross-layer attention. This turns the L layers themselves into the iterative refinement loop (replacing SlotLoRA's T per-layer iterations) while enabling each layer's slots to leverage binding patterns from earlier layers.

---

## 1. Motivation & Design Goals

| Goal | SlotLoRA contribution | X-LoRA contribution |
|------|----------------------|---------------------|
| Expert specialization | Slots compete for experts via softmax binding | — |
| Cross-layer communication | — | Causal attention over cached LoRA outputs |
| Iterative refinement | T rounds of GRU within layer | L layers as natural iterations |
| Functional role persistence | Slots capture roles per token | Roles should persist across depth |
| Efficiency | O(TKN) per layer | O(L²) total |

**Key insight:** If slot vectors represent persistent functional roles, then at layer l the slots should already have information about what they bound to at layers 1..l-1. The L transformer layers become the natural iterative refinement process — no need for T internal iterations.

**Central tension resolved:** SlotLoRA rebinds from scratch each layer. CS-LoRA propagates binding memory across layers via slot state caching + gated cross-attention, initialized to standard LoRA behavior (all gates = 0).

---

## 2. Architecture Overview

```
Layer 1                    Layer 2                    Layer L
┌──────────────┐          ┌──────────────┐            ┌──────────────┐
│  Slot←Expert │          │  Slot←Expert │            │  Slot←Expert │
│   Binding    │          │   Binding    │     ╱╲     │   Binding    │
│      ↓       │   ───►   │      ↓       │  ╱╱╲╲╲    │      ↓       │
│ GRU(S, ΔS₁)  │          │ GRU(S, ΔS₂)  │ ╱╱╱╲╲╲►   │ GRU(S, ΔS_L) │
│      ↓       │          │      ↓       │  ╲╲╱╱╱    │      ↓       │
│  Slot→Cache  │          │  Slot→Cache  │   ╲╲╱╱    │  Slot→Cache  │
└──────┼───────┘          └──────┼───────┘   ╲╱     └──────┼───────┘
       │                         │                         │
       └─────────────────────────┴─────────────────────────┘
                           ↓
              Cross-layer slot attention
              (each layer attends to cached
               slot states from prior layers)
```

**Core components:**
- **K persistent slot vectors** S ∈ ℝ^{K×d} — shared across all layers
- **Per-layer expert binding** B_l ∈ ℝ^{K×N} — soft assignment of experts to slots
- **Gated cross-layer propagation** — slots attend to their own history via causal mask
- **Register slots** R ∈ ℝ^{R×d} — absorb layer-identity information via learned layer embeddings
- **Running EMA cache** — avoids O(L) storage for slot history

---

## 3. Mathematical Formulation

### 3.1 Notation

| Symbol | Meaning |
|--------|---------|
| L | Number of transformer layers |
| N_l | Number of LoRA experts at layer l (assumed = N for simplicity) |
| K | Number of slot vectors |
| d | Hidden dimension |
| S^{(l)} ∈ ℝ^{K×d} | Slot states after processing layer l |
| B_l ∈ ℝ^{K×N} | Binding matrix at layer l (softmax over slots) |
| E_l ∈ ℝ^{N×d} | Expert outputs at layer l |
| τ | Binding temperature |
| γ_l | Cross-layer gate at layer l (initialized 0) |
| β_l | Register modulation gate (initialized 0) |
| R ∈ ℝ^{R×d} | Register slot vectors |
| emb_l ∈ ℝ^{d} | Learned embedding for layer l |

### 3.2 Per-Layer Processing (Layer l)

**Step 1: Expert Computation**

Each expert i at layer l produces an output via standard LoRA:
```
e_{l,i} = B_i · A_i · x_l    for i = 1..N
E_l = [e_{l,1}; ...; e_{l,N}] ∈ ℝ^{N×d}
```

**Step 2: Slot-Expert Binding (within-layer)**

Each expert distributes its "membership probability" across the K slots. The binding matrix B_l captures which functional roles each expert contributes to:

```
B_l = softmax_{slots}(S^{(l-1)} · E_l^T / (τ · √d)) ∈ ℝ^{K×N}
```

where `softmax_{slots}` applies softmax along dimension 0 (slots), so:
```
B_l[k,i] = exp(s_k · e_{l,i} / τ) / Σ_{k'} exp(s_{k'} · e_{l,i} / τ)
```

Each column (expert) sums to 1 over slots. This is **expert competition**: each expert picks which slots it belongs to.

**Step 3: Slot Context from Experts**

Each slot aggregates expert outputs weighted by how much that expert belongs to it:
```
ΔS_l = B_l · E_l ∈ ℝ^{K×d}
ΔS_l[k] = Σ_i B_l[k,i] · e_{l,i}
```

**Step 4: Cross-Layer Slot Context**

Rather than storing all L previous slot states (O(LKd) per token), we maintain a **running EMA** of token-averaged slot states:

```
S_ema^{(l)} = α · S_ema^{(l-1)} + (1-α) · mean_tokens(S^{(l)})
```

At layer l, the cached history is H_l = [S_ema^{(1)}; S_ema^{(2)}; ...; S_ema^{(l-1)}] ∈ ℝ^{(l-1)×K×d}. We reshape to a single sequence of (l-1)K vectors of dimension d.

```
Q_l = reshape(S^{(l-1)}, (K, d)) · W_Q ∈ ℝ^{K×d}
K_l = reshape(H_l, ((l-1)K, d)) · W_K ∈ ℝ^{(l-1)K×d}
V_l = reshape(H_l, ((l-1)K, d)) · W_V ∈ ℝ^{(l-1)K×d}

A_l = softmax(Q_l · K_l^T / √d) ∈ ℝ^{K×(l-1)K}    (causal: only l-1 prior layers)
C_l = A_l · V_l ∈ ℝ^{K×d}
```

This gives each slot a **cross-layer context vector** summarizing what it has been binding to at previous layers.

**Step 5: Combined Slot Update**

```
Δ_total = LayerNorm(ΔS_l + γ_l · C_l)
S^{(l)} = GRU(S^{(l-1)}, Δ_total)    (shared GRUCell across all layers)
```

The gate γ_l ∈ [0,1] is a learned scalar initialized to 0. At initialization, γ_l = 0 → C_l is suppressed → behavior equals standard LoRA with per-layer slot binding but no cross-layer flow. During training, γ_l can grow, allowing cross-layer slot communication.

**Step 6: Register Slot Modulation**

Register slots (from CODA/ SlotLoRA) absorb layer-identity information. Each layer embeds its position into a learned vector, and register slots attend to it:

```
emb_l = LayerEmbedding(l) ∈ ℝ^{d}
reg_attn = softmax(R · emb_l / √d) ∈ ℝ^{R}
reg_out = Σ_j reg_attn[j] · R_j ∈ ℝ^{d}
S^{(l)} = S^{(l)} + β_l · reg_out    (broadcast to all slots)
```

β_l is initialized to 0. This lets the model learn "at layer 7, slot 3 specializes differently than at layer 12" without hard-coding layer-specific behavior.

**Step 7: Layer Output**

Expert outputs are combined weighted by slot binding strength:

```
g_l = softmax(Σ_k B_l[k,:] / τ) ∈ ℝ^{N}
output_l = Σ_i g_l[i] · e_{l,i}
```

The gating aggregates binding mass across slots per expert: experts strongly bound to one or more slots get higher weight.

---

## 4. Pseudocode

```
# Initialization
S = randn(K, d) * 0.02              # Slot vectors (learned)
S_prev = S                           # [K, d]
S_ema = 0                            # Running EMA
gamma = zeros(L)                     # Cross-layer gates (init 0)
beta = zeros(L)                      # Register gates (init 0)
R = randn(R, d) * 0.02              # Register slots (learned)
W_Q, W_K, W_V = Linear(d,d)          # Cross-attention projections
GRU_cell = GRUCell(d, d)             # Shared slot update
LayerEmb = Embedding(L, d)           # Layer embeddings (learned)

# Forward pass
slot_cache = []                      # For cross-layer attention
outputs = []

for l in 0..L-1:
    # 1. Compute expert outputs E_l  [N, d]
    E_l = compute_experts(l, x_l)

    # 2. Slot-expert binding
    sim = S_prev @ E_l.T / (tau * sqrt(d))     # [K, N]
    B_l = softmax(sim, dim=0)                   # softmax over slots

    # 3. Within-layer slot context
    delta_S = B_l @ E_l                         # [K, d]

    # 4. Cross-layer slot context
    if l > 0:
        # Build key-value from cached slot EMAs
        H = stack(slot_cache)                   # [(l), K, d] → [(l*K), d]
        Q = W_Q(S_prev)                         # [K, d]
        K = W_K(H)                              # [(l*K), d]
        V = W_V(H)                              # [(l*K), d]
        attn = softmax(Q @ K.T / sqrt(d), dim=-1) # [K, l*K]
        C_l = attn @ V                          # [K, d]
    else:
        C_l = 0

    # 5. Combined update
    delta_total = LayerNorm(delta_S + gamma[l] * C_l)
    S_new = GRU_cell(S_prev, delta_total)       # [K, d]

    # 6. Register modulation
    emb_l = LayerEmb(l)                         # [d]
    reg_sim = R @ emb_l / sqrt(d)              # [R]
    reg_attn = softmax(reg_sim, dim=0)          # [R]
    reg_out = reg_attn @ R                     # [d]
    S_new = S_new + beta[l] * reg_out

    # 7. Update cache
    S_bar = mean_over_tokens(S_new)             # [K, d] (or just S_new if not per-token)
    S_ema = alpha * S_ema + (1-alpha) * S_bar
    slot_cache.append(S_ema)

    # 8. Layer output
    g_l = softmax(sum(B_l, dim=0) / tau)        # [N]
    output_l = g_l @ E_l                        # [d]

    S_prev = S_new
    outputs.append(output_l)

return outputs
```

---

## 5. Implementation Sketch (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossLayerSlotBVoRAN(LowRankAdapter):
    """
    CS-LoRA implemented on top of BVoRAN expert modules.

    K slot vectors are shared across all L layers. At each layer,
    slots bind to N LoRA experts, update via GRU, and receive
    cross-layer context from prior slot states via causal attention.

    This file implements the per-layer component. A wrapper
    (CrossLayerSlotModel) orchestrates L layers with shared slots.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: AdapterConfig,
    ):
        super().__init__(in_features, out_features, config)

        # --- Expert parameters (N=1 for simplicity; expand for N>1) ---
        self.lora_A = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))

        ve_rank = config.dvora_ve_rank or max(1, config.r // 2)
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))

        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A.numel() + self.lora_B.numel()
                + self.magnitude.numel() + self.ve_lambda.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        """Standard BVoRAN expert forward — produces one expert output."""
        base_weight = base_weight.to(x.dtype)

        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        delta = adapted - base_weight

        out = F.linear(x, delta)
        out = self.norm(out)
        return out  # [B, T, d]


class SlotBridge(nn.Module):
    """
    Cross-Layer Slot Bridge orchestrator.

    Manages K shared slot vectors, cross-layer attention cache,
    GRU update, register modulation, and expert gating over L layers.

    Usage:
        bridge = SlotBridge(d_model=2048, num_experts=4, num_slots=8, num_layers=24)
        outputs = bridge(x, expert_fns)   # expert_fns: list of L callables
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_slots: int,
        num_layers: int,
        num_registers: int = 4,
        temperature: float = 1.0,
        ema_alpha: float = 0.9,
        max_cross_window: int = 8,  # attend only to last W layers' EMA states
    ):
        super().__init__()
        self.d = d_model
        self.N = num_experts
        self.K = num_slots
        self.L = num_layers
        self.R = num_registers
        self.tau = temperature
        self.alpha = ema_alpha
        self.W = max_cross_window

        # Slot vectors [K, d] — shared across all layers
        self.slot_init = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)

        # Register slots [R, d]
        self.registers = nn.Parameter(torch.randn(num_registers, d_model) * 0.02)

        # Per-layer embeddings [L, d]
        self.layer_emb = nn.Embedding(num_layers, d_model)

        # Cross-attention projections
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Per-layer gates (initialized 0 → starts as standard per-layer binding)
        self.gamma = nn.Parameter(torch.zeros(num_layers))

        # Register modulation gates
        self.beta = nn.Parameter(torch.zeros(num_layers))

        # Shared GRU cell for slot update across all layers
        self.gru = nn.GRUCell(d_model, d_model)

        # LayerNorm for update stability
        self.norm = nn.LayerNorm(d_model)

        # Output projection (expert gating MLP)
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_experts),
        )

    def forward(self, expert_outputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            expert_outputs: list of L tensors, each [B, T, N, d]
        Returns:
            outputs: list of L tensors, each [B, T, d]
        """
        B, T, N, D = expert_outputs[0].shape
        device = expert_outputs[0].device

        # Initialize slot state: [B, K, d] — broadcast over batch
        S_prev = self.slot_init.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, K, d]

        # Running EMA of slot states (per-batch-item, summarized over tokens)
        S_ema = torch.zeros(B, self.K, D, device=device)

        # Cached EMA states for cross-layer attention
        slot_cache = []

        outputs = []

        for l in range(self.L):
            E_l = expert_outputs[l]  # [B, T, N, d]

            # --- Step 1: Slot-Expert Binding ---
            # For each token, compute slot-expert similarity
            # S_prev: [B, K, d] → unsqueeze for token dim
            S_t = S_prev.unsqueeze(1)  # [B, 1, K, d]
            sim = torch.einsum('btkd,btn d->btkn', S_t, E_l)  # [B, T, K, N]
            # Temperature-scaled softmax over slots (dim=2)
            B_l = F.softmax(sim / (self.tau * math.sqrt(self.d)), dim=2)  # [B, T, K, N]

            # --- Step 2: Within-layer slot context ---
            # For each token, each slot aggregates expert outputs weighted by binding
            delta_S = torch.einsum('btkn,btnd->btkd', B_l, E_l)  # [B, T, K, d]

            # Average over tokens to get per-batch-item slot update
            delta_S_pooled = delta_S.mean(dim=1)  # [B, K, d]

            # --- Step 3: Cross-layer slot context ---
            cross_ctx = torch.zeros(B, self.K, D, device=device)

            if len(slot_cache) > 0:
                # Stack cached EMAs: list of [B, K, d] → [B, C, K, d] where C = len(cache)
                cache_tensor = torch.stack(slot_cache, dim=1)  # [B, C, K, d]

                # Window: only last W layers
                if cache_tensor.shape[1] > self.W:
                    cache_tensor = cache_tensor[:, -self.W:]

                # Reshape to [B, C*K, d]
                C = cache_tensor.shape[1]
                H = cache_tensor.reshape(B, C * self.K, D)  # [B, C*K, d]

                # Q: current slot state, K/V: cached slot EMAs
                Q = self.W_Q(S_prev)  # [B, K, d]
                K = self.W_K(H)       # [B, C*K, d]
                V = self.W_V(H)       # [B, C*K, d]

                attn = torch.softmax(
                    torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d),
                    dim=-1
                )  # [B, K, C*K]

                cross_ctx = torch.bmm(attn, V)  # [B, K, d]

            # --- Step 4: Combined slot update via shared GRU ---
            # Gate cross-layer contribution
            update = delta_S_pooled + self.gamma[l] * cross_ctx  # [B, K, d]
            update = self.norm(update)

            # GRU update (shared parameters)
            S_prev_flat = S_prev.reshape(-1, D)   # [B*K, d]
            update_flat = update.reshape(-1, D)   # [B*K, d]
            S_new_flat = self.gru(update_flat, S_prev_flat)
            S_new = S_new_flat.reshape(B, self.K, D)  # [B, K, d]

            # --- Step 5: Register modulation ---
            emb = self.layer_emb(torch.tensor([l], device=device))  # [1, d]
            reg_sim = torch.einsum('brd,bd->br', self.registers.unsqueeze(0), emb) / math.sqrt(self.d)
            reg_attn = F.softmax(reg_sim, dim=-1)  # [1, R]
            reg_out = torch.einsum('br,brd->bd', reg_attn, self.registers.unsqueeze(0))  # [1, d]
            S_new = S_new + self.beta[l] * reg_out  # [B, K, d]

            # --- Step 6: Expert gating and layer output ---
            # Method A: Simple binding-mass gating
            # g_l = B_l.sum(dim=2) / self.tau → [B, T, N] → softmax
            expert_mass = B_l.sum(dim=2)  # [B, T, N] — total binding per expert
            g_l = F.softmax(expert_mass / self.tau, dim=-1)

            # Method B: Slot-state-conditioned gating (more expressive)
            # g_l = self.gate_proj(S_new.unsqueeze(1).expand(-1, T, -1, -1)) → [B, T, K, N]
            # But Method A is simpler and captures the slot-expert binding directly.
            # We use a hybrid: project S_new to expert logits per token.
            # S_new: [B, K, d] → expert_logits: [B, T, N]
            expert_logits = torch.einsum(
                'bkd,btn d->btkn',
                S_new,
                E_l
            ).sum(dim=2)  # [B, T, N] — similarity of slots to experts
            g_l = F.softmax(expert_logits / self.tau, dim=-1)

            output_l = torch.einsum('btn,btnd->btd', g_l, E_l)  # [B, T, d]

            # --- Cache update ---
            # Update EMA: aggregate slot state over tokens
            # S_new is [B, K, d] (global per-batch, not per-token)
            # For EMA: use S_new directly since we already pooled
            S_ema = self.alpha * S_ema + (1 - self.alpha) * S_new
            slot_cache.append(S_ema.detach())  # detach to prevent gradient through time
            S_prev = S_new

            outputs.append(output_l)

        return outputs


class SlotBridgeModelWrapper(nn.Module):
    """
    Wraps a transformer model with CS-LoRA.

    Replaces specified linear layers with CrossLayerSlotBVoRAN adapters,
    and inserts SlotBridge orchestrator for cross-layer slot coordination.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: AdapterConfig,
        num_slots: int = 8,
        num_registers: int = 4,
        num_experts: int = 4,
        cross_window: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_layers = len(self._find_layers())
        self.d_model = self._infer_d_model()

        self.bridge = SlotBridge(
            d_model=self.d_model,
            num_experts=num_experts,
            num_slots=num_slots,
            num_layers=self.num_layers,
            num_registers=num_registers,
            temperature=temperature,
            max_cross_window=cross_window,
        )

        # Replace each target linear layer with N experts
        self.expert_groups = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            experts = nn.ModuleList([
                CrossLayerSlotBVoRAN(
                    self.d_model, self.d_model, config
                ) for _ in range(num_experts)
            ])
            self.expert_groups.append(experts)

    def _find_layers(self):
        """Find transformer layers in base model. Model-specific."""
        # Simplified: assumes base_model has a .layers or .h attribute
        if hasattr(self.base_model, 'layers'):
            return self.base_model.layers
        elif hasattr(self.base_model, 'h'):
            return self.base_model.h
        elif hasattr(self.base_model, 'transformer'):
            return self.base_model.transformer.h
        raise AttributeError("Cannot find transformer layers")

    def _infer_d_model(self):
        """Infer hidden dimension from first linear layer weight."""
        sample_layer = self._find_layers()[0]
        for name, mod in sample_layer.named_modules():
            if isinstance(mod, nn.Linear):
                return mod.out_features
        return 2048  # fallback

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = self._find_layers()

        # Collect expert outputs for each layer
        expert_outputs = []
        h = x

        for layer_idx, layer in enumerate(layers):
            # Forward through base layer (attention + FFN)
            # Intercept at each target module and run experts
            experts = self.expert_groups[layer_idx]

            # Simplified: one target module per layer
            # In practice, replace each nn.Linear in target_modules
            layer_out = layer(h)

            # Compute expert outputs for this layer
            E_l = []
            for expert in experts:
                e_out = expert(h, layer_out)  # simplified
                E_l.append(e_out)
            E_l = torch.stack(E_l, dim=2)  # [B, T, N, d]

            expert_outputs.append(E_l)
            h = layer_out

        # Run SlotBridge to produce gated outputs
        bridged = self.bridge(expert_outputs)

        # Apply bridged outputs back to residual stream
        h = x
        for layer_idx, layer in enumerate(layers):
            # Simplified: bridged output replaces adapter contribution
            h = layer(h) + bridged[layer_idx]

        return h
```

---

## 6. Complexity Analysis

### 6.1 Per-Layer Cost (assuming N experts, K slots, d hidden dim)

| Operation | Complexity | Dominant? |
|-----------|-----------|-----------|
| Expert computation | O(N·d²) | ✓ (same as standard LoRA) |
| Slot-Expert binding | O(K·N·d) | ✓ when K,N large |
| Within-layer context | O(K·N·d) | ✓ when K,N large |
| Cross-layer attention | O(K²·W·d) | small (W=window, K≪d) |
| GRU update | O(K·d²) | moderate |
| Register modulation | O(R·d) | negligible |
| Expert gating | O(K·N·d) | moderate |

### 6.2 Total (L layers)

| Architecture | Complexity | Scaling |
|-------------|------------|---------|
| Standard LoRA | O(L·N·d²) | Linear in L,N |
| SlotLoRA (T rounds) | O(T·L·K·N·d) | T× multiplier |
| X-LoRA | O(L·N·d² + L²·N²) | Quadratic in L |
| **CS-LoRA (ours)** | **O(L·N·d² + L·K·N·d + L·K²·W)** | **Near-linear** |

### 6.3 Practical savings

For a model with L=24, N=4, K=8, W=8, d=2048:
- SlotLoRA (T=3): ~3 × binding cost
- X-LoRA: ~24 × 24 × 16 = 9216 cross-layer pairs
- CS-LoRA: 8² × 8 = 512 cross-layer slot-EMA pairs (using window of 8)

CS-LoRA adds ≈ 12–20% overhead vs standard LoRA, compared to 3× for SlotLoRA or quadratic blowup for X-LoRA.

---

## 7. Design Decisions & Rationale

### 7.1 Why token-averaged slot states for cross-layer?

**Problem:** Storing [B, T, K, d] × L slot states is O(L·B·T·K·d) — prohibitive.

**Solution:** Cross-layer communication uses **token-averaged slot prototypes** (pooled over the T dimension). The intuition: slot k at layer l cares about what slot k has been doing on *average* across tokens at previous layers, not about every single token's slot assignment. Per-token specificity is preserved in the within-layer binding step, which IS token-specific.

**Trade-off:** Some cross-layer token-specific signal is lost. But the alternative (full per-token caching) is not practical for long sequences.

### 7.2 Running EMA vs. full history cache

**Choice:** EMA with windowed attention to last W cached states.

**Rationale:** Rather than attending to all L-1 previous slot states (cost O(K²L)), we maintain a running EMA that compresses the entire slot history. The windowed cache allows attention to recent fluctuations while the EMA captures long-term trends. With W=8, the cost is O(K²W) independent of L.

**Alternative considered:** Compressive memory (see Section 9).

### 7.3 Causal cross-attention masking

**Choice:** Causal (layer l attends only to layers < l).

**Rationale:** For autoregressive decoding, future-layer information is unavailable. Unlike SlotLoRA's symmetric within-layer binding (all experts compete simultaneously), cross-layer must respect the sequential processing order. The binding matrix B_l remains symmetric within a layer (all N experts at layer l compete for slots), matching SlotLoRA's design.

### 7.4 Shared GRU across layers

**Choice:** One GRUCell shared by all L layers.

**Rationale:** Forces all layers to use the same slot update dynamics, which encourages slot vectors to maintain consistent semantics across depth. Per-layer GRUs would allow drift where "slot 1" means syntax at layer 3 but semantics at layer 12.

**Mitigation:** The per-layer gates γ_l and β_l, plus the layer embedding in register modulation, provide layer-specific flexibility on top of the shared dynamics.

### 7.5 Register slots for layer identity

**Choice:** Register slots attend to learned layer embeddings.

**Rationale:** Layer identity is a global property (not token-specific). Register slots, being global, naturally capture this. The CODA-inspired mechanism lets registers specialize to different "layer archetypes" (early = syntactic, mid = semantic, deep = task-specific) without manual assignment.

### 7.6 Gate initialization to 0

**Choice:** γ_l = 0, β_l = 0 at initialization.

**Rationale:** Same as X-LoRA. The model starts as standard per-layer slot-expert binding (no cross-layer flow, no register modulation). During fine-tuning, gradients can activate these gates if cross-layer communication is beneficial. This ensures training stability and prevents the new components from disrupting the pre-trained initialization.

---

## 8. Relationship to Existing Adapters

### 8.1 Similarity to KnitBVoRAN

The existing `KnitBVoRAN` already implements a form of cross-layer sharing via a global registry that accumulates hidden states. CS-LoRA generalizes this:
- Knit: simple additive skip connections (prev_x + x)
- CS-LoRA: learned attention-weighted cross-layer slot context with gating

CS-LoRA subsumes Knit as a special case: with K=1, no registers, γ_l=1, and W=0 (only EMA), the cross-layer context becomes a learned weighted sum of prior slot states, similar to Knit's additive accumulation.

### 8.2 Composition with other techniques

CS-LoRA is orthogonal to AFA (attention feature adaptation), SR (spectral regularization), GA (gated aggregation), and EVA (eigenvalue adaptation). These can be composed:
- **AFA + CS-LoRA**: Expert outputs are AFA-modulated before slot binding
- **SR + CS-LoRA**: Slot vectors are spectral-regularized for diversity
- **GA + CS-LoRA**: The expert gating step uses GA's learned gating instead of simple softmax
- **EVA + CS-LoRA**: Eigenvalue normalization applied to slot update

### 8.3 Composing CS-LoRA with Knit

CS-LoRA and Knit address different cross-layer mechanisms:
- CS-LoRA: slot state flows across layers (functional role persistence)
- Knit: hidden states accumulate across layers (feature reuse)
These are complementary and can be combined.

---

## 9. Limitations & Future Work

### 9.1 Token-independence assumption in cross-layer

The token-averaged slot prototype loses per-token cross-layer information. A potential extension uses **adaptive pooling** where slot states are summarized per "binding cluster" (groups of tokens with similar slot assignment) rather than global average.

### 9.2 EMA information bottleneck

The running EMA compresses all prior layer information into a single vector per slot. For very deep models (L > 48), this may be insufficient. **Hierarchical EMA** (multiple time-scale EMAs) or **compressive transformers** (trained compressed memory) could extend this.

### 9.3 GRU parameter sharing

Shared GRU forces all layers to use identical slot dynamics. Per-layer GRU with **weight sharing via hypernetwork** (generating GRU weights from layer embedding) would allow layer-specific dynamics while maintaining parameter efficiency.

### 9.4 Binding interpretability

The binding matrices B_l provide interpretable slot-expert assignments. Future work could analyze whether slots consistently capture linguistic properties (POS, syntax, semantics) across layers, and whether this leads to more controllable fine-tuning.

### 9.5 Integration with the project's benchmarking framework

CS-LoRA should be benchmarked using the existing `run_exp.py` pipeline. Key metrics vs. the top-ranked variant (sr_afa_bvoran, PPL=1.16):
- Does cross-layer slot propagation improve PPL beyond per-layer binding?
- Does the interpretable binding matrix enable analysis that other variants don't?
- What is the wall-clock overhead of the slot attention mechanism?

---

## Appendix A: Addressing the Key Tensions

| Tension | Resolution |
|---------|-----------|
| SlotLoRA per-layer vs. X-LoRA cross-layer | **Layers as iterations.** Slots persist across layers; each layer is one refinement step. T=1 internal iterations. |
| Slot binding propagation | Slot state S^{(l-1)} initializes S^{(l)} via GRU. Cross-layer attention adds context from prior states as a residual. |
| Causal vs. symmetric binding | **Hybrid:** Within-layer = symmetric (all N experts compete). Cross-layer = causal (only prior layers). |
| Functional role persistence | Slots maintain identity across layers via shared GRU + EMA cache. Role drift is controlled by per-layer gating. |
| Register slots for layer-id | Registers attend to learned layer embeddings, producing layer-specific modulation of slot states. |
| Complexity | **O(L·K·N·d + L·K²·W)** vs. O(T·L·K·N·d) for SlotLoRA or O(L²·N²) for X-LoRA. EMA + windowed attention keep cross-layer cost constant in L. |
