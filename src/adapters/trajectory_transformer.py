"""TrajectoryTransformer: direct transformer over thought trajectories.

Bypasses the Perceiver bottleneck by processing all 23 hidden states at
full 2048-dim through a transformer encoder with self-attention.

Architecture:
  Input:  (B, 23, 2048)  — hidden states at layers 0..22
  Proj:   Linear(2048 → d_model) + LayerNorm
  Pos:    Learned position embedding per layer (0..22)
  Encoder: N x TransformerBlock (Pre-LN self-attention + FFN)
  Output: Linear(d_model → 2048)
  Output: (B, 23, 2048)  — predicted velocities v[0..22]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        if causal:
            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal=causal)
        x = x + self.ffn(self.ln2(x))
        return x


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_positions: int = 256,  # support up to 256 reasoning steps
        d_input: int = 2048,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Embedding(n_positions, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_input)

    def forward(self, hidden_seq: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, L, D = hidden_seq.shape
        x = self.input_norm(self.input_proj(hidden_seq))
        pos = self.pos_embed(torch.arange(L, device=hidden_seq.device)).unsqueeze(0)
        x = x + pos

        for block in self.blocks:
            x = block(x, causal=causal)

        x = self.output_norm(x)
        velocity = self.output_proj(x)
        return velocity


class BottleneckTrajectoryTransformer(nn.Module):
    """TT with progressive bottleneck: wide → narrow → bottleneck → narrow → wide.

    dims: list of widths for each processing layer, e.g. [1536, 1280, 768, 1280, 1536]
    """
    def __init__(self, dims: list, n_heads: int = 8, d_ff_ratio: int = 4,
                 n_positions: int = 256, d_input: int = 3584):
        super().__init__()
        self.input_proj = nn.Linear(d_input, dims[0])
        self.input_norm = nn.LayerNorm(dims[0])
        self.pos_embed = nn.Embedding(n_positions, dims[0])
        self.dims = dims

        self.blocks = nn.ModuleList()
        for i in range(len(dims)):
            block = TransformerBlock(dims[i], min(n_heads, dims[i] // 64), dims[i] * d_ff_ratio)
            self.blocks.append(block)
            if i < len(dims) - 1 and dims[i] != dims[i + 1]:
                self.blocks.append(nn.Linear(dims[i], dims[i + 1]))

        self.output_norm = nn.LayerNorm(dims[-1])
        self.output_proj = nn.Linear(dims[-1], d_input)

    def forward(self, hidden_seq: torch.Tensor, causal: bool = False) -> torch.Tensor:
        x = self.input_norm(self.input_proj(hidden_seq))
        pos = self.pos_embed(torch.arange(x.shape[1], device=hidden_seq.device)).unsqueeze(0)
        x = x + pos
        for block in self.blocks:
            x = block(x, causal=causal) if isinstance(block, TransformerBlock) else block(x)
        x = self.output_norm(x)
        return self.output_proj(x)
