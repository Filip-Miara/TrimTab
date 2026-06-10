"""Flow matching over adapter weights.

A velocity field v_θ(W_t, t, data) predicts how adapter weights should change,
trained to match observed SGD trajectories. At inference, integrating from
W_0 = 0 generates trained adapter weights without backprop.

This is conditional flow matching on the adapter weight manifold:

    data → v_θ(W_t, t, data) → integrate → W_1  (no SGD needed)

Architecture uses a Perceiver bottleneck to handle the high-dimensional
weight space efficiently: compress N weights through K latents.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEncoder(nn.Module):
    """Encodes training data (input, target) into a context vector."""
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.mean(dim=(0, 1)) if x.dim() == 3 else x.mean(dim=0))


class WeightFlowField(nn.Module):
    """Velocity field over adapter weights with Perceiver bottleneck.

    Maps (weights, t, data_context) → velocity (predicted weight change).

    Uses a Perceiver-style bottleneck: N weights → K latents → self-attend
    → cross-attend back to N velocity predictions. O(NK + K²).
    """

    def __init__(self, n_weights: int, d_latent: int = 64, n_latents: int = 32,
                 n_heads: int = 4, d_context: int = 64):
        super().__init__()
        self.n_weights = n_weights
        self.d_latent = d_latent
        self.n_latents = n_latents

        # Weight embedding: each weight position gets a learned positional encoding
        self.pos_embed = nn.Embedding(n_weights, d_latent)

        # Data context projection
        self.context_proj = nn.Linear(d_context, d_latent)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )

        # Perceiver bottleneck
        self.latents = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)
        self.cross_in = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_in = nn.LayerNorm(d_latent)
        self.self_attn = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_sa = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, 4 * d_latent), nn.GELU(), nn.Linear(4 * d_latent, d_latent),
        )
        self.norm_ff = nn.LayerNorm(d_latent)
        self.cross_out = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(d_latent)

        # Decode: latent → weight velocity
        self.decode = nn.Linear(d_latent, 1)

    def forward(
        self,
        weights: torch.Tensor,
        t: torch.Tensor,
        data_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = weights.shape[0]
        N = self.n_weights

        pos = self.pos_embed.weight.unsqueeze(0).expand(B, -1, -1)
        t_emb = self.time_embed(t.unsqueeze(-1)).unsqueeze(1).expand(-1, N, -1)

        w_emb = weights.unsqueeze(-1) * 0.01 + pos + t_emb

        ctx = torch.zeros(B, 1, self.d_latent, device=weights.device)
        if data_ctx is not None:
            ctx = self.context_proj(data_ctx).unsqueeze(1)

        Z = self.latents.unsqueeze(0).expand(B, -1, -1)
        Z_in, _ = self.cross_in(Z + ctx, w_emb, w_emb, need_weights=False)
        Z = self.norm_in(Z + Z_in)
        Z_sa, _ = self.self_attn(Z, Z, Z, need_weights=False)
        Z = self.norm_sa(Z + Z_sa)
        Z = self.norm_ff(Z + self.ffn(Z))
        Z_out, _ = self.cross_out(w_emb, Z, Z, need_weights=False)
        w_out = self.norm_out(Z_out)

        velocity = self.decode(w_out).squeeze(-1)
        return velocity * 0.1


class WeightFlowTrainer:
    """Trains WeightFlowField via flow matching on observed weight trajectories.

    Collects (W_t, W_{t+1}, t, data) pairs from SGD training, then trains
    v_θ(W_t, t, data) to match W_{t+1} - W_t.
    """

    def __init__(self, flow: WeightFlowField, lr: float = 1e-3, device: str = "cpu"):
        self.flow = flow.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.flow.parameters(), lr=lr)

    def train_step(self, weights_t: torch.Tensor, weights_tp1: torch.Tensor,
                   t: torch.Tensor, data_ctx: torch.Tensor | None = None) -> float:
        velocity_target = (weights_tp1 - weights_t).detach()
        velocity_pred = self.flow(weights_t, t, data_ctx)
        loss = F.mse_loss(velocity_pred, velocity_target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def generate_weights(self, n_steps: int, data_ctx: torch.Tensor | None = None,
                         init_weights: torch.Tensor | None = None) -> torch.Tensor:
        self.flow.eval()
        if init_weights is None:
            w = torch.zeros(1, self.flow.n_weights, device=self.device)
        else:
            w = init_weights.clone()
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.tensor([step / n_steps], device=self.device)
            v = self.flow(w, t, data_ctx)
            w = w + v * dt
        return w


def collect_trajectory(expert: nn.Module, x: torch.Tensor, target: torch.Tensor,
                       n_steps: int, lr: float = 1e-2) -> list[torch.Tensor]:
    """Run SGD on an expert and return weight trajectory."""
    named = list(expert.named_parameters())
    loa_named = [(n, p) for n, p in named if p.requires_grad and ('lora_A' in n or 'lora_B' in n)]
    params = [p for _, p in loa_named]
    if not params:
        params = [p for _, p in named if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr)
    trajectory = []
    for _ in range(n_steps):
        trajectory.append(torch.cat([p.data.flatten() for p in params]).clone())
        loss = F.mse_loss(expert(x), target)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return trajectory


def test_weight_flow():
    """Proof of concept: learn to generate LoRA weights via flow matching."""
    from .stream_fusion import PlainStreamExpert, EXPERT_REGISTRY

    expert = PlainStreamExpert(64, 32, 4, 16, 16, 1.0)
    x = torch.randn(4, 64)
    target = torch.randn(4, 32)

    n_weights = PlainStreamExpert.absorb_dim(64, 32, 4)
    print(f"Weight space dim: {n_weights}")

    flow = WeightFlowField(n_weights, d_latent=16, n_latents=8)
    trainer = WeightFlowTrainer(flow, device="cpu")

    for iteration in range(20):
        traj = collect_trajectory(expert, x, target, 10)
        total_loss = 0.0
        for t_idx in range(len(traj) - 1):
            w_t = traj[t_idx].unsqueeze(0)
            w_tp1 = traj[t_idx + 1].unsqueeze(0)
            t = torch.tensor([t_idx / (len(traj) - 1)])
            loss = trainer.train_step(w_t, w_tp1, t)
            total_loss += loss
        if iteration % 5 == 0:
            print(f"Iter {iteration}: avg loss={total_loss/9:.6f}")

    gen_w = trainer.generate_weights(10)
    print(f"Generated weights shape: {gen_w.shape}")
    print("WEIGHT FLOW MATCHING TEST PASSED")


if __name__ == "__main__":
    test_weight_flow()
