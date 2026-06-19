# TrajectoryTransformer Training Pipeline

## Overview

The TrajectoryTransformer (TT) is a 48M-parameter transformer that predicts hidden state velocity vectors across a frozen language model's 28 layers. The predicted velocities are used for KV-cache steering: multiplying by α=0.1 and adding to the hidden state before recomputing K/V projections achieves significant accuracy improvements on GSM8K (+20pp at best trim-tab layers).

## 1. Data Collection

Trajectories are collected by running the language model (Qwen2.5-7B-Instruct) on GSM8K problems with greedy decoding:

```
For each generation step (prompt + generated tokens):
  1. Forward pass through all 28 layers
  2. Record hidden states h_l at each layer l (shape: [28, 3584])
  3. Compute velocity targets: v_l = h_{l+1} - h_l (layer-to-layer delta)
  4. Save (h, v) pairs as .pt files
```

84 batch files of ~1100 trajectories each ≈ 90K total trajectories.
Each trajectory: hidden_seqs [28, 3584], velocity_targets [28, 3584], dtype=float16.
File size: ~422MB each, ~35GB total.

### AWQ trajectories
16 batch files of ~3000 trajectories each ≈ 49K total.
Same format but collected via Google Colab (different GPU environment).
Notably different velocity distribution: AWQ mid-layer velocities are ~2× larger than BnB's, while L0 is near-zero.

## 2. Data Preprocessing

### Memory-mapped storage (mmap)
All .pt files are merged into two memory-mapped files for efficient random access:

```
train_h.bin: float16, shape [90K, 28, 3584], ~17GB
train_v.bin: float16, shape [90K, 28, 3584], ~17GB
val_h.bin:   float16, shape [1.1K, 28, 3584], ~211MB
val_v.bin:   float16, shape [1.1K, 28, 3584], ~211MB
```

### Normalization
Global normalization across samples AND layers:
```python
v_mean = velocities.mean(dim=(0, 1), keepdim=True)  # shape [1, 1, 3584]
v_std = velocities.std(dim=(0, 1), keepdim=True) + 1e-8
```

This preserves cross-layer velocity magnitude ratios (layer 0 changes 3× more than layer 27).

### Validation
First 500 trajectories from the validation split.
Used only for monitoring — never for training decisions.

## 3. Model Architecture

```python
class TrajectoryTransformer(nn.Module):
    d_model=768
    n_layers=6
    n_heads=8
    d_ff=3072  # d_model * 4
    n_positions=28
    d_input=3584

    Input:  Linear(3584 → 768) + LayerNorm
    Pos:    Learned embedding (28 positions)
    Encoder: 6× TransformerBlock(Pre-LN SelfAttention + FFN(GELU))
    Output: LayerNorm + Linear(768 → 3584)
```

48,061,184 total parameters.
Bidirectional self-attention (causal=False) — each layer attends to all others.
Learned positional embeddings per layer index.

## 4. Training Pipeline

### VRAM Double-Buffer
Two pre-allocated GPU buffers (float16, shape [4200, 28, 3584]) for sliding-window data loading:
- GPU buffer A: current chunk being processed
- GPU buffer B: next chunk being asynchronously transferred (CUDA stream)
- Background thread prefetches next chunk from mmap (CPU) while GPU computes

Per-minibatch: convert from float16 GPU buffer → float32 for model input.

### Optimizations
```python
torch.set_float32_matmul_precision('high')         # TF32 (~2× FP32 matmul speed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### Hyperparameters
- Batch size: 64
- Learning rate: 3e-4 (AdamW, weight_decay=1e-4)
- Epochs: 30-50
- Gradient clipping: 1.0
- Chunk size: 4000 trajectories

### Loss Function
Default: MSE only
```python
loss = (layer_w * (tt(hb) - vn).pow(2).mean(dim=-1)).sum(dim=-1).mean()
```

With cosine loss (experimental):
```python
mse = (pred - vn).pow(2).mean(dim=-1)
cos = (pred * vn).sum(dim=-1) / (pred.norm(dim=-1) * vn.norm(dim=-1) + 1e-8)
loss = (layer_w * (mse + λ * (1 - cos))).sum(dim=-1).mean()  # λ=0.5 tested
```

## 5. Evaluation Metrics

### R² (coefficient of determination)
```python
r2 = 1 - MSE / Var(y_true)
```
Measures magnitude accuracy — fraction of velocity variance explained.
R²=0.85 means 85% of velocity magnitude variance is captured.

### Cosine similarity
```python
cos = (v_pred · v_true) / (|v_pred| × |v_true|)
```
Measures directional alignment only, ignoring magnitude.
cos=0.77 means average angle of ~40° between predicted and actual velocities.

Both metrics computed on DENORMALIZED (original scale) velocities.

## 6. Steering Application

Once trained, the TT is used for KV-cache steering:

```
For each generation step (after first token):
  1. Get hidden states from forward pass: hs[0..28]
  2. Stack last-token hidden states: h_stack = stack(hs[l][:, -1, :])
  3. Predict velocities: v = TT(h_stack.unsqueeze(0))
  4. At target layer l_t:
     h_steered = hs[l_t+1][:, -1, :] + α * v[:, l_t, :]
     k_new = k_proj(h_steered)
     v_new = v_proj(h_steered)
     KV_cache[l_t][:, :, -1:, :] = k_new
```

α=0.1 typically used. Multiple α values can be swept.
Different layers produce different effects: "trim-tab" layers (+10-20pp), "death" layers (-23pp).

## 7. Current Best Results

| Configuration | R² | Cos | Notes |
|--------------|:--:|:---:|-------|
| BnB data, baseline | 0.848 | 0.770 | Standard bidirectional, MSE only |
| BnB data, +cosine λ=0.5 | 0.838 | 0.709 | Epoch 3, still climbing |
| AWQ data, scratch | 0.450 | 0.524 | AWQ → AWQ, limited by data quality |
| BnB → AWQ finetune | 0.347 | 0.439 | Catastrophic forgetting of BnB |

## 8. Known Issues

1. **AWQ transfer**: BnB TT doesn't transfer to AWQ model (different hidden state distribution). AWQ-trained TT plateaus at low R² (0.45 vs 0.85).
2. **Catastrophic forgetting**: Fine-tuning BnB TT on AWQ data destroys BnB performance.
3. **GPU stability**: CUDA crashes under sustained load (thermal/power). Need conservative settings.
4. **Causal attention**: Performs worse than bidirectional (R²=0.812 vs 0.840 at epoch 2).
5. **Data I/O bound**: Training on USB HDD is slow. SSD needed for optimal throughput.
