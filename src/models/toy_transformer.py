from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyConfig:
    vocab_size: int = 10000
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    intermediate_dim: int = 1024
    max_seq_len: int = 128
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5


class Attention(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(mask[:, None, None, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.attn = Attention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        mask = torch.ones((B, T), device=input_ids.device)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
            return loss
        return logits
