"""ThoughtDiffusion: flow matching over thought trajectories.

Perceiver-based architecture processing the full trajectory sequence.
Given [h0, h1, ..., h_{L-1}], predicts all velocities v_l = h_{l+1} - h_l.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverBlock(nn.Module):
    def __init__(self, d_latent: int, n_heads: int):
        super().__init__()
        self.cross_in = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_in = nn.LayerNorm(d_latent)
        self.self_attn = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_sa = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, 4 * d_latent), nn.GELU(), nn.Linear(4 * d_latent, d_latent),
        )
        self.norm_ff = nn.LayerNorm(d_latent)

    def forward(self, Z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Z_in, _ = self.cross_in(Z, x, x, need_weights=False)
        Z = self.norm_in(Z + Z_in)
        Z_sa, _ = self.self_attn(Z, Z, Z, need_weights=False)
        Z = self.norm_sa(Z + Z_sa)
        Z = self.norm_ff(Z + self.ffn(Z))
        return Z


class ThoughtDiffusion(nn.Module):
    """Perceiver-based flow matching over full thought trajectory.

    Input: [h_0, h_1, ..., h_{L-1}] — sequence of hidden states (B, L, D)
    Output: [v_0, v_1, ..., v_{L-1}] — per-layer velocities (B, L, D)
    """

    def __init__(
        self,
        d_model: int = 2048,
        n_layers: int = 24,
        d_latent: int = 64,
        n_latents: int = 16,
        n_heads: int = 4,
        d_text_ctx: int = 128,
        n_perceiver_blocks: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_latent = d_latent
        self.n_latents = n_latents

        # Project hidden states to latent dimension
        self.thought_proj = nn.Linear(d_model, d_latent)
        self.layer_embed = nn.Embedding(n_layers, d_latent)
        self.text_proj = nn.Linear(d_text_ctx, d_latent)

        self.latents = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)
        self.blocks = nn.ModuleList([
            PerceiverBlock(d_latent, n_heads) for _ in range(n_perceiver_blocks)
        ])

        self.out_proj = nn.Linear(d_latent, d_latent)
        self.cross_out = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(d_latent)

        # Decode velocity at each layer position
        self.decode = nn.Sequential(
            nn.Linear(d_latent, 2 * d_latent),
            nn.GELU(),
            nn.Linear(2 * d_latent, d_model),
        )

    def forward(
        self,
        hidden_seq: torch.Tensor,
        text_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocities for full trajectory.

        Args:
            hidden_seq: (B, L, d_model) hidden states across L layers
            text_ctx: (B, d_text_ctx) text conditioning
        Returns:
            velocity: (B, L, d_model) predicted velocity at each layer
        """
        B, L, D = hidden_seq.shape

        h_emb = self.thought_proj(hidden_seq)
        layer_pos = self.layer_embed(torch.arange(L, device=hidden_seq.device)).unsqueeze(0)
        h_emb = h_emb + layer_pos
        h_emb = h_emb + self.text_proj(text_ctx).unsqueeze(1).expand(-1, L, -1)

        Z = self.latents.unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            Z = block(Z, h_emb)

        h_query = self.out_proj(h_emb)
        h_out, _ = self.cross_out(h_query, Z, Z, need_weights=False)
        h_out = self.norm_out(h_out)

        velocity = self.decode(h_out)
        return velocity


class ThoughtFlowTrainer:
    def __init__(
        self,
        model: ThoughtDiffusion,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def train_step(
        self,
        hidden_seq: torch.Tensor,
        velocity_target: torch.Tensor,
        text_ctx: torch.Tensor,
    ) -> float:
        v_pred = self.model(hidden_seq.to(self.device), text_ctx.to(self.device))
        target = velocity_target.to(self.device)
        loss = F.mse_loss(v_pred, target)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        return loss.item()
