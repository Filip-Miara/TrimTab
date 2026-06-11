# Isocline Adaptation (DiagLoRA): Bi-Diagonal Magnitude Decomposition

## Abstract

BoRA applies learnable magnitude scaling along axes-aligned groups (full columns, full rows). We generalize this to magnitude scaling along **diagonal groups** — constant-offset (Toeplitz) and constant-sum (Hankel) families — forming a bi-orthogonal decomposition of the weight matrix. This reparameterization learns a **frequency-domain mask** on the LoRA update, enabling adaptive frequency-selective filtering of the weight space.

---

## 1. Motivation

BoRA's row+column scaling treats every row and column independently. But there is no fundamental reason magnitude should decompose along axis-aligned lines. Consider:

- **Off-diagonal coupling** captures cross-feature interactions — e.g., adjacent tokens in a sequence, adjacent pixels, or frequency bands
- **Diagonal structure** corresponds to specific 2D frequency components in the weight matrix (the 2D DFT basis decomposes into diagonals)
- Tasks may require different frequency biases — high-frequency detail (vision), low-frequency trends (time series), band-pass (certain NLP tasks)

**Core claim:** Learning magnitude along diagonals is strictly more expressive than axis-aligned scaling for a given parameter budget, because each diagonal scalar interacts with both row and column indices simultaneously.

---

## 2. Formulation

### 2.1 Definitions

Given `W₀ ∈ ℝ^{hr×hc}` and LoRA matrices `A ∈ ℝ^{hr×r}`, `B ∈ ℝ^{r×hc}`:

**Main-diagonal scaling (315°).** Index `k = j - l` ranging `-(hc-1)` to `(hr-1)`. Scalar `m^d_k` weights all `(j,l)` with `j - l = k`:

`V^(d) = m^d ⊙ (W₀ + AB) / ||(W₀ + AB)^{(j-l=k)}||`

where normalization is per-diagonal (each diagonal scaled to unit norm).

**Anti-diagonal scaling (135°).** Index `s = j + l` ranging `0` to `hr+hc-2`. Scalar `m^a_s` weights all `(j,l)` with `j + l = s`:

`W = m^a ⊙ V^(d) / ||V^{(j+l=s)}||`

### 2.2 Sequentially Applied (BoRA-style)

Following BoRA's two-step pattern:

```
Step 1:  V^d = normalize_diag(W₀ + AB, direction=main)
         V_diag = m^d ⊙ V^d                     (learn diagonal magnitudes)

Step 2:  H^a = normalize_anti(V_diag, direction=anti)
         W = m^a ⊙ H^a                          (learn anti-diagonal magnitudes)
```

where `normalize_diag(M, main)` divides each diagonal `j-l=k` by its L2 norm, and `normalize_anti(M, anti)` divides each anti-diagonal `j+l=s` by its L2 norm.

### 2.3 Parameter Count

| Method | Extra params per layer |
|--------|----------------------|
| DoRA   | `hc` |
| BoRA   | `hr + hc` |
| DiagLoRA | `(hr+hc-1) + (hr+hc-1) = 2(hr+hc-1)` |

For a typical dense layer (hr=hc=d), DoRA adds d, BoRA adds 2d, DiagLoRA adds ~4d — about 2× BoRA.

### 2.4 Low-Rank Variant

If 4d parameters are too many, we can parameterize `m^d` and `m^a` through a learned low-dimensional embedding:

`m^d = MLP_small(z_d)`, `z_d ∈ ℝ^r'`, `r' ≪ hr+hc`

This learns a structured mapping from a compact latent code to diagonal magnitude profiles.

---

## 3. Interpretation

### 3.1 Frequency-Domain View

Consider the 2D DFT of `W`:

`Ŵ(ω_j, ω_l) = Σ_{j,l} W_{j,l} exp(-i(ω_j·j + ω_l·l))`

- Main-diagonal offset `k` corresponds to frequency energy along the line `ω_j = ω_l + const`
- Anti-diagonal sum `s` corresponds to frequency energy along `ω_j = -ω_l + const`

DiagLoRA learns a **learnable band-pass filter** in the 2D frequency domain of the weight matrix, where the filter is constrained to have diagonal support.

### 3.2 Algebraic View (Toeplitz + Hankel)

- `m^d` scaling makes the update tend toward a **Toeplitz matrix** (constant along diagonals)
- `m^a` scaling makes the update tend toward a **Hankel matrix** (constant along anti-diagonals)
- Together they span the space of matrices with multiplicative rank-1 structure in the (j+l, j-l) coordinate system

### 3.3 Information Flow View

- Column scaling (DoRA): weights an entire input feature uniformly across all outputs
- Row scaling (BoRA): weights an entire output feature uniformly across all inputs
- **Diagonal scaling**: weights specific **input-output offset relationships** — e.g., how much adjacent features interact vs. distant features. This is akin to learning a **distance-dependent kernel**.

---

## 4. Implementation Considerations

### 4.1 Scatter/Gather Efficiency

Row/column ops map to BLAS-2 (GEMV). Diagonal gather/scatter requires:
- Index computation: `indices = j * hc + l` for each diagonal
- `torch.scatter_reduce` or custom CUDA kernel

**Potential optimization:** Represent the diagonal gather as a `nn.Conv2d` with dilated kernels (one kernel per offset k). This maps the diagonal masking to a group convolution.

### 4.2 Normalization Stability

Diagonal normalization over varying lengths (short diagonals at corners, long in middle) may produce uneven gradient signals. Consider:
- **Group normalization** across all diagonal norms instead of per-diagonal division
- **Weight-tying** short diagonals with longer ones to stabilize

### 4.3 Initialization

- `m^d_k = 1` for all k → identity initialization (no net effect)
- `m^a_s = 1` for all s → identity initialization
- `A` and `B` initialized as standard LoRA (zero init for B)
- This ensures DiagLoRA starts as the base model and smoothly departs during training

---

## 5. Potential Applications

| Domain | Why DiagLoRA fits |
|--------|-------------------|
| Vision (ViT) | ConvNets have diagonal structure in patches; DiagLoRA can learn patch-offset interactions |
| Time-series | Temporal offsets are naturally diagonal; different lags can get different magnitudes |
| LLMs (attention) | Attention logits have token-offset structure; DiagLoRA can weight near/far token pairs differently |
| Spectrogram processing | 2D frequency×time structure maps to diagonal bands |

---

## 6. Research Questions

1. **Expressivity gap.** Is DiagLoRA strictly more expressive than BoRA? (Conjecture: yes, because diagonal basis spans a different subspace of ℝ^{hr×hc} than row/column basis, and the union of both subspaces has higher dimension.)

2. **Quantization synergy.** Diagonals collapse one spatial dimension. Does this make magnitude scaling more or less sensitive to quantization / NF4?

3. **Merge behavior.** Can `m^d`, `m^a`, A, B be merged into `W₀` for zero-inference-overhead? (Yes — same as DoRA/BoRA — all are linear operations on W₀.)

4. **Diagonal dropout.** If some diagonals are naturally more important than others (e.g., near-diagonals dominate in attention), can we learn to prune whole diagonal groups?

5. **Extension to higher-order.** For 3D weight tensors (e.g., Conv2D kernels), diagonal groups become **planar diagonals**. Does this generalize naturally?

---

## 7. Relation to Existing Work

| Method | Grouping | Dims | Extra params / layer |
|--------|----------|------|---------------------|
| LoRA | None | — | 0 |
| DoRA | Column | 1D | hc |
| BoRA | Row + Column | 2D (axis) | hr + hc |
| DiagLoRA | Diagonal + Anti-diagonal | 2D (rotated) | 2(hr+hc-1) |
| VeRA | Shared random projection | 1D | d (learnable scaling) |
| AdaLoRA | SVD-based importance scoring | — | dynamic |
| FFTA (speculative) | DFT-basis magnitude | 2D (frequency) | hr·hc |

DiagLoRA occupies a middle ground between axis-aligned magnitude (BoRA) and full frequency-domain decomposition (FFTA), offering more expressivity than BoRA without the full parameter cost of unrestricted frequency masks.

---

## Open Questions / Next Steps

- Derive the exact optimization landscape for diagonal-normalized vs row-normalized gradients
- Prototype a single-layer DiagLoRA in PyTorch and compare loss landscape curvature to BoRA (small-scale, 2D regression)
- If promising, design ablation: DiagLoRA vs BoRA vs DoRA on a vision task (e.g., VTAB-1k with ViT-B/16) controlling for total trainable parameters
