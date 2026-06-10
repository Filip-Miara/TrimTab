"""Composed flows: diffusion + flow matching over adapter weights.

Architecture:
  MetaController (flag-flow)
      │
      ├── flags: architecture configuration [bi, vec, norm, gate, ...]
      │
      └── conditions WeightDiffusion
                │
                ├── input: W_t, t_noise, flags, data_ctx
                ├── output: ε_pred (denoising) + velocity_pred (flow)
                ├── training: L = λ_diff · MSE(ε, ε_gt) + λ_flow · MSE(v, v_gt)
                └── data: augmented trajectories (K noise levels × 20 steps)

The diffusion objective provides vastly more training signal — each clean
weight can be corrupted at many noise levels, giving ~infinite data from
a few trajectories. The flow objective anchors predictions to real SGD dynamics.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stream_fusion import FLAG_NAMES, N_FLAGS


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """Cosine noise schedule for diffusion."""
    return torch.cos((t * math.pi / 2).clamp(max=math.pi / 2)) ** 2


def add_noise(weights: torch.Tensor, t: torch.Tensor, sqrt_alpha_bar: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Add noise at level t. Returns (noisy_weights, noise)."""
    noise = torch.randn_like(weights)
    if sqrt_alpha_bar is None:
        sqrt_alpha_bar = cosine_schedule(t).to(weights.device)
    noisy = sqrt_alpha_bar * weights + (1 - sqrt_alpha_bar).clamp(min=0).sqrt() * noise
    return noisy, noise


class PerceiverBlock(nn.Module):
    """Single Perceiver block: cross-attend to inputs, self-attend."""

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


class WeightDiffusion(nn.Module):
    """Composed diffusion + flow matching over adapter weights.

    Perceiver-bottleneck architecture conditioned on flags and data context.
    Predicts both noise (denoising) and velocity (flow matching).

    At inference: start from noise, iteratively denoise, then apply velocity.
    """

    def __init__(
        self,
        n_weights: int,
        d_latent: int = 64,
        n_latents: int = 16,
        n_heads: int = 4,
        d_ctx: int = 5120,
        n_perceiver_blocks: int = 2,
    ):
        super().__init__()
        self.n_weights = n_weights
        self.d_latent = d_latent
        self.n_latents = n_latents

        # Embeddings
        self.pos_embed = nn.Embedding(n_weights, d_latent)
        self.weight_proj = nn.Linear(2, d_latent)  # weight + noise level
        self.flag_proj = nn.Linear(N_FLAGS, d_latent)
        self.ctx_proj = nn.Linear(d_ctx, d_latent)

        # Time embedding (diffusion time + flow time)
        self.time_embed = nn.Sequential(
            nn.Linear(2, d_latent), nn.GELU(), nn.Linear(d_latent, d_latent),
        )

        # Perceiver blocks
        self.latents = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)
        self.blocks = nn.ModuleList([
            PerceiverBlock(d_latent, n_heads) for _ in range(n_perceiver_blocks)
        ])

        # Cross-attend: weight positions → latents → weight positions
        self.w_emb_proj = nn.Linear(d_latent, d_latent)
        self.cross_out = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(d_latent)
        self.decode = nn.Linear(d_latent, 2)

    def forward(
        self,
        weights: torch.Tensor,
        t_noise: torch.Tensor,
        t_flow: torch.Tensor,
        flags: torch.Tensor,
        data_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = weights.shape

        w_inp = torch.stack([weights, torch.full_like(weights, t_noise.mean().item())], dim=-1)
        w_emb = self.weight_proj(w_inp)
        w_emb = w_emb + self.pos_embed.weight.unsqueeze(0)
        w_emb = w_emb + self.time_embed(torch.cat([t_noise, t_flow], dim=-1)).unsqueeze(1)

        flag_emb = self.flag_proj(flags).unsqueeze(1)
        ctx_emb = self.ctx_proj(data_ctx).unsqueeze(1)

        Z = self.latents.unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            Z = block(Z, w_emb + flag_emb + ctx_emb)

        # Cross-attend back to weight positions
        w_query = self.w_emb_proj(w_emb)
        w_out, _ = self.cross_out(w_query, Z, Z, need_weights=False)
        w_out = self.norm_out(w_out)

        out = self.decode(w_out)  # (B, N, 2)
        noise_pred = out[:, :, 0]  # (B, N)
        velocity_pred = out[:, :, 1] * 0.1
        return noise_pred, velocity_pred


class DiffusionFlowTrainer:
    """Trains WeightDiffusion with combined denoising + flow matching loss.

    Generates augmented data: each clean weight produces K noisy versions
    at different noise levels, giving O(K × N) training samples from N trajectories.
    """

    def __init__(self, model: WeightDiffusion, lr: float = 1e-3, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.lambda_diff = 1.0
        self.lambda_flow = 0.1

    def train_step(
        self,
        clean_weights: torch.Tensor,  # (B, N)
        next_weights: torch.Tensor,   # (B, N) — for flow target
        flags: torch.Tensor,          # (B, N_FLAGS)
        data_ctx: torch.Tensor,       # (B, d_ctx)
        t_noise: torch.Tensor,        # (B, 1) — random noise levels
        t_flow: torch.Tensor,         # (B, 1) — flow time
    ) -> tuple[float, float, float]:
        noisy, noise = add_noise(clean_weights, t_noise)
        noisy, noise = noisy.to(self.device), noise.to(self.device)

        noise_pred, velocity_pred = self.model(
            noisy, t_noise.to(self.device), t_flow.to(self.device),
            flags.to(self.device), data_ctx.to(self.device),
        )

        target_velocity = (next_weights - clean_weights).to(self.device)

        loss_diff = F.mse_loss(noise_pred.squeeze(-1), noise)
        loss_flow = F.mse_loss(velocity_pred.squeeze(-1), target_velocity)
        loss = self.lambda_diff * loss_diff + self.lambda_flow * loss_flow

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        return loss.item(), loss_diff.item(), loss_flow.item()

    @torch.no_grad()
    def generate_weights(
        self,
        n_steps: int,
        data_ctx: torch.Tensor,
        flags: torch.Tensor,
        init_weights: torch.Tensor | None = None,
        ddim_steps: int = 10,
    ) -> torch.Tensor:
        """Generate weights via DDIM denoising + flow integration."""
        self.model.eval()
        if init_weights is None:
            w = torch.randn(1, self.model.n_weights, device=self.device)
        else:
            w = init_weights.clone().to(self.device)

        # DDIM denoising
        step_size = 1.0 / ddim_steps
        for s in range(ddim_steps):
            t = torch.tensor([[1.0 - s * step_size]], device=self.device)
            noise_pred, _ = self.model(w, t, torch.zeros(1, 1, device=self.device), flags.to(self.device), data_ctx.to(self.device))
            alpha = cosine_schedule(t)
            w = (w - (1 - alpha).clamp(min=0).sqrt() * noise_pred) / alpha.clamp(min=1e-8).sqrt()

        # Flow integration (starting from denoised weights)
        for s in range(n_steps):
            t = torch.tensor([[s / n_steps]], device=self.device)
            _, velocity_pred = self.model(
                w, torch.zeros(1, 1, device=self.device), t,
                flags.to(self.device), data_ctx.to(self.device),
            )
            w = w + velocity_pred / n_steps

        return w.cpu()


def augment_trajectories(
    trajectories: list,
    n_noise_levels: int = 5,
) -> list[dict]:
    """Augment trajectories with multiple noise levels for diffusion training.

    Each (clean_weight, next_weight, ctx) pair becomes n_noise_levels
    training examples at different noise levels.
    """
    augmented = []
    for weights, ctxs in trajectories:
        for t_idx in range(len(weights) - 1):
            w_t = weights[t_idx]
            w_tp1 = weights[t_idx + 1]
            ctx = ctxs[t_idx] if t_idx < len(ctxs) else ctxs[-1]
            for k in range(n_noise_levels):
                t_noise_val = k / max(n_noise_levels - 1, 1)
                augmented.append({
                    "clean": w_t, "next": w_tp1,
                    "t_noise": t_noise_val, "t_flow": t_idx / max(len(weights) - 2, 1),
                    "ctx": ctx,
                    "flags": torch.zeros(N_FLAGS),  # placeholder — will be filled by MetaController
                })
    return augmented
