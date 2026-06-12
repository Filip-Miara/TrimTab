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


class TrajectoryTrainer:
    def __init__(self, model: TrajectoryTransformer, lr: float = 3e-4, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def train_step(self, hidden_seq: torch.Tensor, velocity_target: torch.Tensor) -> float:
        v_pred = self.model(hidden_seq.to(self.device))
        loss = F.mse_loss(v_pred, velocity_target.to(self.device))
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return loss.item()
