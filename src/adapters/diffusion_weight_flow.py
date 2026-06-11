"""Composed flows with proper denoising architecture.

Key improvements over v1:
1. Weight normalization: normalize→add noise→predict noise→unnormalize
   Ensures noise scale matches weight scale so MSE < 1.0 is meaningful.
2. Separate output heads: shared Perceiver backbone, independent decode_noise + decode_flow
3. Curriculum noise: start with low noise, gradually increase during training
4. Proper noise scaling: ε ~ N(0, σ_W) not N(0, 1)
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stream_fusion import FLAG_NAMES, N_FLAGS


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """Cosine noise schedule: ᾱ(t) = cos(t·π/2)²"""
    return torch.cos((t * math.pi / 2).clamp(max=math.pi / 2)) ** 2


class WeightNormalizer:
    """Tracks running statistics of weight vectors for proper noise scaling.

    Weights from SGD trajectories have variance ≈ 0.01-0.1.
    Without normalization, noise with variance 1.0 dominates and denoising
    MSE of 1.0 (predicting zero) is the optimal strategy.
    With normalization, noise scale matches weight scale, making denoising meaningful.
    """

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def update(self, weights: torch.Tensor):
        w = weights.flatten()
        m = w.mean()
        s = w.std() + 1e-8
        if self.mean is None:
            self.mean = m
            self.std = s
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * m
            self.std = self.momentum * self.std + (1 - self.momentum) * s

    def normalize(self, weights: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return weights
        return (weights - self.mean) / self.std

    def unnormalize(self, weights: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return weights
        return weights * self.std + self.mean

    def denormalize_noise(self, noise_pred: torch.Tensor) -> torch.Tensor:
        """Scale noise prediction back to original weight scale."""
        if self.std is None:
            return noise_pred
        return noise_pred * self.std

    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Return appropriate noise scale for timestep t."""
        if self.std is None:
            return torch.ones_like(t)
        return self.std.expand_as(t)


def add_noise_scaled(
    weights: torch.Tensor,
    t: torch.Tensor,
    normalizer: WeightNormalizer,
    sqrt_alpha_bar: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add noise scaled to match weight distribution.

    Normalizes weights, adds unit-variance noise, scales back.
    This ensures the denoising objective has a meaningful baseline:
    MSE=1.0 = random guess, MSE<1.0 = learning.
    """
    w_norm = normalizer.normalize(weights)
    noise = torch.randn_like(w_norm)
    if sqrt_alpha_bar is None:
        sqrt_alpha_bar = cosine_schedule(t).to(weights.device)
    noisy_norm = sqrt_alpha_bar * w_norm + (1 - sqrt_alpha_bar).clamp(min=0).sqrt() * noise
    noisy = normalizer.unnormalize(noisy_norm)
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
    """Diffusion + flow matching with proper denoising architecture.

    Key design:
    - Shared Perceiver backbone
    - Separate decode_noise and decode_flow heads (no competition)
    - Weight normalization for well-conditioned denoising
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
        self.normalizer = WeightNormalizer()

        # Weight position embedding (learned, shared across all computations)
        self.pos_embed = nn.Embedding(n_weights, d_latent)

        # Input projection: concatenate [weight_val, noise_level] → d_latent
        self.weight_proj = nn.Linear(2, d_latent)

        # Conditioners
        self.flag_proj = nn.Linear(N_FLAGS, d_latent)
        self.ctx_proj = nn.Linear(d_ctx, d_latent)

        # Time embedding: [t_noise, t_flow] → d_latent
        self.time_embed = nn.Sequential(
            nn.Linear(2, d_latent), nn.GELU(), nn.Linear(d_latent, d_latent),
        )

        # Perceiver blocks (shared)
        self.latents = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)
        self.blocks = nn.ModuleList([
            PerceiverBlock(d_latent, n_heads) for _ in range(n_perceiver_blocks)
        ])

        # Cross-attention back to weight positions (shared)
        self.w_emb_proj = nn.Linear(d_latent, d_latent)
        self.cross_out = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(d_latent)

        # SEPARATE output heads (no parameter sharing between noise and velocity)
        self.decode_noise = nn.Linear(d_latent, 1)  # ε prediction
        self.decode_flow = nn.Linear(d_latent, 1)   # velocity prediction

    def forward(
        self,
        weights: torch.Tensor,
        t_noise: torch.Tensor,
        t_flow: torch.Tensor,
        flags: torch.Tensor,
        data_ctx: torch.Tensor,
        noise_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict noise and velocity with separate heads.

        Args:
            weights: (B, N) weights (may be noisy)
            t_noise: (B, 1) diffusion time
            t_flow: (B, 1) flow time
            flags: (B, N_FLAGS)
            data_ctx: (B, d_ctx)
            noise_scale: (B, 1) standard deviation of weight distribution
        Returns:
            noise_pred: (B, N) predicted noise (unit variance, not weight-scaled)
            velocity_pred: (B, N) predicted weight change
        """
        B, N = weights.shape

        # Per-weight embedding: [weight_value, noise_level]
        noise_level = t_noise.mean().item()
        w_inp = torch.stack([weights, torch.full_like(weights, noise_level)], dim=-1)
        w_emb = self.weight_proj(w_inp)
        w_emb = w_emb + self.pos_embed.weight.unsqueeze(0)

        # Add time conditioning
        w_emb = w_emb + self.time_embed(torch.cat([t_noise, t_flow], dim=-1)).unsqueeze(1)

        # Add conditioner embeddings
        flag_emb = self.flag_proj(flags).unsqueeze(1)
        ctx_flat = data_ctx.squeeze(1) if data_ctx.dim() == 3 else data_ctx
        ctx_emb = self.ctx_proj(ctx_flat).unsqueeze(1)
        cond = w_emb + flag_emb + ctx_emb

        # Perceiver processing
        Z = self.latents.unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            Z = block(Z, cond)

        # Cross-attend back to weight positions
        w_query = self.w_emb_proj(w_emb)
        w_out, _ = self.cross_out(w_query, Z, Z, need_weights=False)
        w_out = self.norm_out(w_out)

        # Separate heads — no parameter sharing
        noise_pred = self.decode_noise(w_out).squeeze(-1)  # (B, N)
        velocity_pred = self.decode_flow(w_out).squeeze(-1) * 0.1  # (B, N)

        # If noise_scale provided, scale noise prediction to match weight scale
        if noise_scale is not None:
            noise_pred = noise_pred * noise_scale.mean().item()

        return noise_pred, velocity_pred


class DiffusionFlowTrainer:
    """Trains WeightDiffusion with properly scaled denoising + flow matching.

    Weight normalization ensures denoising has a meaningful baseline:
    - MSE=1.0 means "predicting zero" (random guess)
    - MSE<1.0 means actual learning

    Curriculum noise linearly ramps max noise from 0.3 to 1.0 over epochs.
    """

    def __init__(
        self,
        model: WeightDiffusion,
        lr: float = 1e-3,
        device: str = "cpu",
        lambda_diff: float = 1.0,
        lambda_flow: float = 0.1,
        lambda_optimal: float = 0.5,
        curriculum_epochs: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.lambda_diff = lambda_diff
        self.lambda_flow = lambda_flow
        self.lambda_optimal = lambda_optimal
        self.curriculum_epochs = curriculum_epochs
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _get_max_noise(self) -> float:
        """Curriculum: linearly increase max noise from 0.3 to 1.0."""
        progress = min(self._epoch / max(self.curriculum_epochs, 1), 1.0)
        return 0.3 + 0.7 * progress

    def train_step(
        self,
        clean_weights: torch.Tensor,
        next_weights: torch.Tensor,
        flags: torch.Tensor,
        data_ctx: torch.Tensor,
        t_noise: torch.Tensor,
        t_flow: torch.Tensor,
        optimal_target: torch.Tensor | None = None,
    ) -> tuple[float, float, float, float]:
        """Single training step with normalized denoising.

        Noise is scaled to match the weight distribution's standard deviation.
        This makes denoising well-conditioned: MSE=1.0 = zero prediction,
        MSE < 1.0 = actual learning.
        """
        # Update normalizer with current weights
        self.model.normalizer.update(clean_weights)

        # Add noise with proper scaling
        noisy, noise = add_noise_scaled(clean_weights, t_noise, self.model.normalizer)
        noisy, noise = noisy.to(self.device), noise.to(self.device)

        # Get weight std for output scaling
        w_std = self.model.normalizer.std.clone().detach() if self.model.normalizer.std is not None else torch.tensor(1.0)

        noise_pred, velocity_pred = self.model(
            noisy, t_noise.to(self.device), t_flow.to(self.device),
            flags.to(self.device), data_ctx.to(self.device),
            noise_scale=w_std.unsqueeze(0).to(self.device),
        )

        target_velocity = (next_weights - clean_weights).to(self.device)

        # Loss components
        loss_diff = F.mse_loss(noise_pred, noise)
        loss_flow = F.mse_loss(velocity_pred, target_velocity)
        loss = self.lambda_diff * loss_diff + self.lambda_flow * loss_flow

        if optimal_target is not None and self.lambda_optimal > 0:
            loss_opt = F.mse_loss(velocity_pred, optimal_target.to(self.device))
            loss = loss + self.lambda_optimal * loss_opt
        else:
            loss_opt = 0.0

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        return loss.item(), loss_diff.item(), loss_flow.item(), loss_opt if isinstance(loss_opt, float) else loss_opt.item()

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

        w_std = self.model.normalizer.std.clone() if self.model.normalizer.std is not None else torch.tensor(1.0)

        # DDIM denoising (in normalized space)
        step_size = 1.0 / ddim_steps
        for s in range(ddim_steps):
            t = torch.tensor([[1.0 - s * step_size]], device=self.device)
            noise_pred, _ = self.model(
                w, t, torch.zeros(1, 1, device=self.device),
                flags.to(self.device), data_ctx.to(self.device),
                noise_scale=w_std.unsqueeze(0).to(self.device),
            )
            alpha = cosine_schedule(t).to(self.device)
            w = (w - (1 - alpha).clamp(min=0).sqrt() * noise_pred) / alpha.clamp(min=1e-8).sqrt()

        # Flow integration
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
    """Augment trajectories with multiple noise levels."""
    augmented = []
    for item in trajectories:
        if len(item) == 3:
            weights, ctxs, grads = item
        else:
            weights, ctxs = item
            grads = None
        for t_idx in range(len(weights) - 1):
            w_t = weights[t_idx]
            w_tp1 = weights[t_idx + 1]
            ctx = ctxs[t_idx] if t_idx < len(ctxs) else ctxs[-1]

            opt_target = None
            if grads is not None and t_idx < len(grads):
                g = grads[t_idx]
                g_norm = g.norm()
                if g_norm > 1e-8:
                    opt_target = -(g / g_norm)

            for k in range(n_noise_levels):
                t_noise_val = k / max(n_noise_levels - 1, 1)
                augmented.append({
                    "clean": w_t, "next": w_tp1,
                    "t_noise": t_noise_val, "t_flow": t_idx / max(len(weights) - 2, 1),
                    "ctx": ctx,
                    "flags": torch.zeros(N_FLAGS),
                    "optimal_target": opt_target,
                })
    return augmented
