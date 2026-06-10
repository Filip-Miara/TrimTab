# SCOPE-LoRA: Streaming Compositional Overlap-Pool Ensemble

## 1. Design Rationale & Key Decisions

### Granularity: Rank-1 Components
**Decision**: Rank-1 atomic units (u_i, v_i), grouped dynamically for inference.

**Why**: ContOverlap's rank-1 sharing is the source of its O(1) forgetting bound — finer granularity means more sharing opportunities. SOLE's full rank-r LoRAs are convenient but coarser. SCOPE bridges them: components are rank-1 atoms, but the ISAB layer groups them into effective rank-r blocks during forward passes, recovering SOLE's practical performance.

### Alignment Signal: Two-Tier
**Decision**: ISAB attention as **fast proxy** (every step) + gradient covariance as **calibration signal** (periodic).

**Why**: Covariance estimation costs 50 steps — unacceptable in streaming. ISAB attention is O(K²) and available every step. But ISAB may drift. Solution: use alignment scores from covariance estimation as a *regularization target* for ISAB attention during calibration phases.

### n_i → Dual Importance
**Decision**: n_i (task count) governs **protection**; α_i (learned scalar) governs **contribution**.

**Why**: Separates concerns. n_i prevents forgetting (regularization strength); α_i captures how useful the component actually is (learned via gradients). This avoids ContOverlap's coupling where a component shared by many tasks might not actually be useful for all of them.

### ISAB Inducing Points as Compressed Pool Index
**Decision**: Each inducing point q_k learns to represent a "neighborhood" in component space — components cluster around the inducing point they align with best.

**Why**: The K inducing points serve as a content-addressable hash table. A new task queries ISAB → gets weights over K neighborhoods → retrieves components within those neighborhoods. This reduces selection from O(M) to O(K + M/K) via clustering.

---

## 2. Architecture

### Component Pool

```
P = { (u_i, v_i, α_i, n_i, k_i) }ⁱ  for i = 1..M
```

Where:
- `u_i ∈ ℝʳ`, `v_i ∈ ℝᵈ` — rank-1 factor pair (W_i = v_i ⊗ u_i)
- `α_i ∈ ℝ⁺` — learned importance weight (scales contribution)
- `n_i ∈ ℕ` — task affiliation count (for graded protection)
- `k_i ∈ {1..K}` — inducing point affiliation (which cluster it belongs to)

### Inducing Points

```
Q ∈ ℝᴷˣʳ
```

Each `q_k` is a learned prototype in the input-direction space. Components affiliate with the nearest `q_k` via cosine similarity.

### ISAB Module

Two-layer Set Attention Block (SAB) from Set Transformers, modified for streaming:

```
z_t = MAB(MAB(g_t_proj, Q, Q), Q, Q)
s_t = softmax(FFN(z_t))  →  ℝᴷ   (inducing point weights)
```

Where `g_t_proj` is the task gradient projected to rank r.

### Overall Forward Pass

```
1.  Probe:     s_t = ISAB_encode(g_t)                    # ℝᴷ
2.  Retrieve:  for each k: C_k = {i | k_i = k}           # components in cluster
               w_k = Σ_{i∈C_k} α_i · sim(u_i, q_k) · s_t[k]   # scalar
3.  Compose:   ΔW_t = Σ_k w_k · (v_k ⊗ u_k)              # rank-r update
               y = Wx + ΔW_t x
```

### Streaming Update

```
def stream_update(g_t, loss_t, step):
    # Phase 0: Fast probe
    s_t = ISAB_encode(g_t)
    
    # Phase 1: Component selection via routing
    weights = {}  # component → score
    for k where s_t[k] > τ_min:
        C_k = components_with[k]
        for i in C_k:
            score = α_i · s_t[k] · cos_sim(u_i, g_t_proj)
            if score > τ_select:
                weights[i] = score
    
    # Phase 2: Composition
    ΔW = Σ_i weights[i] · v_i ⊗ u_i
    
    # Phase 3: Residual coverage
    residual = g_t - ΔW
    if ||residual||_F > τ_res:
        u_new, v_new = low_rank_approx(residual)
        P.add(u_new, v_new, α=1.0, n=1, 
              k = argmax_k cos_sim(u_new, q_k))
    
    # Phase 4: Update task affiliation
    for i in weights:
        n_i += 1
    
    # Phase 5: Fine-tune module
    ℒ_isab = MSE(ΔW, g_t_proj) + λ Σ_i n_i · ||v_i||²
    optimizer.step(ℒ_isab)
    
    # Phase 6: Background calibration (every C steps)
    if step % C == 0:
        calibrate()
```

### Calibration (Background)

```
def calibrate():
    # Gradient covariance estimation
    G = 1/N Σ ∇L_t ∇L_tᵀ
    
    # Update α_i with alignment scores (running average)
    for i in 1..M:
        a_i = u_iᵀ G u_i / (||u_i||² ||G||_F)
        α_i ← β · α_i + (1-β) · a_i
    
    # Update ISAB to match alignment (distill)
    ℒ_distill = KL(s_t || softmax(a / τ))  
    
    # Merge: components in same cluster with near-identical u_i
    for k in 1..K:
        C_k = components_with[k]
        for i,j in pairs(C_k):
            if cos_sim(u_i, u_j) > τ_merge:
                P.merge(i, j)  # v_new = (n_i·v_i + n_j·v_j)/(n_i+n_j)
                                # n_new = n_i + n_j
                                # α_new = max(α_i, α_j)
                                # keep u_i, discard u_j
```

---

## 3. Mathematical Properties

### Forgetting Bound

ContOverlap's bound carries over directly since the core mechanism (graded protection via n_i) is preserved:

```
ℒ(W_t) - ℒ(W*_t) ≤ ε₀ + c · exp(-λ · Σ_{shared} n_i · α_i)
```

The ISAB routing adds a routing error term bounded by the KL divergence:

```
error_routing ≤ O(√(KL(s_t || s*_t) / 2))
```

Where `s*_t` is the optimal alignment-based selection. The calibration step ensures this stays bounded.

### Growth Rate

Components grow only when residual exceeds τ_res, which depends on how well existing components cover the new task:

```
M(T) ≤ M₀ + T · P(||residual||_F > τ_res)
```

The cluster-aware selection reduces residual probability vs random selection.

### Complexity

| Phase | Cost | Notes |
|-------|------|-------|
| ISAB encode | O(K² + Kr) | Fast |
| Component selection | O(M/K · K) = O(M) | Clustered retrieval |
| Composition | O(Mr) | Sum of active components |
| Calibration | O(Mr² + K²) | Background, amortized |

---

## 4. Pseudocode

```python
class SCOPE_LoRA:
    def __init__(self, d_model, r=8, K=32, M_max=500):
        self.d = d_model
        self.r = r              # component rank
        self.K = K              # inducing points
        self.M_max = M_max      # max pool size

        # Pool
        self.U = Parameter(randn(M_max, r))     # input directions
        self.V = Parameter(randn(M_max, d))     # output directions
        self.alpha = Parameter(ones(M_max))     # importance
        self.n = zeros(M_max, dtype=int)        # task count
        self.k = zeros(M_max, dtype=int)        # cluster id

        self.ptr = 0  # next free slot
        self.mask = [False] * M_max

        # Inducing points
        self.Q = Parameter(randn(K, r))

        # ISAB layers
        self.W1 = Parameter(randn(r, r))
        self.W2 = Parameter(randn(r, r))
        self.W_out = Parameter(randn(r, K))

    def encode(self, g_t):
        """g_t: (d,) gradient → (K,) inducing point weights"""
        g = g_t[:self.r] if g_t.shape[0] > self.r else zero_pad(g_t, self.r)
        g = g @ self.W1
        h = g + MAB(g.unsqueeze(0), self.Q, self.Q).squeeze(0)
        h = h + MAB(h.unsqueeze(0), self.Q, self.Q).squeeze(0)
        s = softmax(h @ self.W_out, dim=-1)
        return s

    def select(self, s_t, g_t, tau=0.1):
        """Returns component indices and weights"""
        g_proj = g_t[:self.r]
        idx, scores = [], []
        for i in range(self.ptr):
            if self.alpha[i] * s_t[self.k[i]] * cos_sim(self.U[i], g_proj) > tau:
                idx.append(i)
                scores.append(self.alpha[i] * s_t[self.k[i]])
        if not idx:
            return [], []
        weights = softmax(tensor(scores), dim=0)
        return idx, weights

    def forward(self, x, g_t=None):
        if g_t is None or not any(self.mask):
            return x  # passthrough during inference

        s_t = self.encode(g_t)
        idx, w = self.select(s_t, g_t)
        if not idx:
            return x

        delta = sum(w[j] * outer(self.V[i], self.U[i]) for j, i in enumerate(idx))
        return x + x @ delta.T

    def stream_update(self, g_t, lr=0.01):
        s_t = self.encode(g_t)
        idx, w = self.select(s_t, g_t)

        # Compose delta_W
        delta_W = sum(w[j] * outer(self.V[i], self.U[i]) for j, i in enumerate(idx))

        # Residual
        residual = g_t[:self.r] - (delta_W @ g_t[:self.r])
        if norm(residual) > 0.1 and self.ptr < self.M_max:
            u_new = residual / norm(residual)
            v_new = g_t[self.r:2*self.r] if len(g_t) > self.r else g_t[:self.r]
            k_new = argmax(cos_sim(u_new, self.Q))
            self.U[self.ptr] = u_new
            self.V[self.ptr] = v_new
            self.alpha[self.ptr] = 1.0
            self.n[self.ptr] = 1
            self.k[self.ptr] = k_new
            self.mask[self.ptr] = True
            self.ptr += 1

        # Update n_i for selected
        for i in idx:
            self.n[i] += 1

        # ISAB fine-tune
        recon_loss = mse(delta_W @ g_t[:self.r], g_t[:self.r])
        reg = sum(self.n[i] * norm(self.V[i])**2 for i in idx)
        loss = recon_loss + 1e-4 * reg
        loss.backward()
        # ... optimizer step for ISAB parameters

    def calibrate(self, dataloader):
        """Background calibration: update α via covariance, merge clusters"""
        G = estimate_grad_covariance(dataloader)  # (r, r)

        for i in range(self.ptr):
            a_i = (self.U[i].T @ G @ self.U[i]) / (norm(self.U[i])**2 * norm(G, 'fro'))
            self.alpha[i] = 0.9 * self.alpha[i] + 0.1 * a_i

        # Update inducing points
        for k in range(self.K):
            comps = [i for i in range(self.ptr) if self.k[i] == k]
            if comps:
                self.Q[k] = mean([self.U[i] for i in comps], dim=0)

        # Merge
        for k in range(self.K):
            comps = [i for i in range(self.ptr) if self.k[i] == k]
            for i in comps:
                for j in comps:
                    if i < j and cos_sim(self.U[i], self.U[j]) > 0.95:
                        self._merge(i, j)

    def _merge(self, i, j):
        """Merge component j into i, delete j"""
        n_total = self.n[i] + self.n[j]
        self.V[i] = (self.n[i] * self.V[i] + self.n[j] * self.V[j]) / n_total
        self.n[i] = n_total
        self.alpha[i] = max(self.alpha[i], self.alpha[j])
        # Delete j: swap with last
        last = self.ptr - 1
        if j != last:
            self.U[j] = self.U[last]
            self.V[j] = self.V[last]
            self.alpha[j] = self.alpha[last]
            self.n[j] = self.n[last]
            self.k[j] = self.k[last]
        self.mask[last] = False
        self.ptr -= 1
```

---

## 5. Implementation Sketch (PyTorch)

The full implementation would include:
- `ComponentPool` dataclass managing the growing buffer
- `ISABEncoder` module with 2 MAB layers
- `StreamingTrainer` with the update loop
- Background calibration thread/process
- Inducing-point-aware MERGE scheduler

Key hyperparameters:
- `r=8` (component rank)
- `K=32` (inducing points)
- `M_max=500` (max components)
- `τ_select=0.1` (selection threshold)
- `τ_res=0.1` (residual threshold)
- `τ_merge=0.95` (merge cosine threshold)
- `C=50` (calibration interval)
- `λ_reg=1e-4` (n_i-weighted regularization)

---

## 6. Failure Modes & Mitigations

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Pool saturation | Too many tasks with high residual | Raise τ_res, increase M_max, more aggressive merge |
| Cluster collapse | All components map to 1-2 inducing points | Repulsion regularizer on Q |
| ISAB-Probe drift | Fast probe diverges from true alignment | Periodic distill from covariance |
| Catastrophic overwrite | α_i grows unbounded for old components | Cap α_i ≤ 1.0, normalize by max n_i |
