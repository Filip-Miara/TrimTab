"""Dynamic K: entropy-thresholded and adaptive growth for Perceiver latents.

Wraps a PerceiverFusion to support varying the number of latent slots at runtime.

Approach A — Entropy-Thresholded:
    After cross-attention, compute attention entropy H_i for each latent.
    Keep latents with H_i < τ·H_max. Drop high-entropy (uniformly attending) latents.

Approach F — Adaptive Growth:
    Start with K=1. Track thought change Δ = ||z_t - z_{t-1}||. If Δ < ε for 3
    consecutive steps, add a new latent initialized from mean(existing) + noise.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


DynamicKMode = Literal["fixed", "entropy", "growth"]


class DynamicPerceiverWrapper(nn.Module):
    """Wraps a PerceiverFusion-style model with dynamic K regulation.

    Usage:
        model = PerceiverFusion(...)
        dyn = DynamicPerceiverWrapper(model, mode="entropy", K_max=32, K_min=4)
        output = dyn(h, embeddings, keys)
        # dyn.current_k -> number of active latents used
    """

    def __init__(
        self,
        perceiver: nn.Module,
        mode: DynamicKMode = "entropy",
        K_max: int = 32,
        K_min: int = 2,
        entropy_threshold: float = 0.7,
        growth_delta: float = 0.05,
        growth_patience: int = 3,
    ):
        super().__init__()
        self.perceiver = perceiver
        self.mode = mode
        self.K_max = K_max
        self.K_min = K_min
        self.entropy_threshold = entropy_threshold  # fraction of max entropy
        self.growth_delta = growth_delta
        self.growth_patience = growth_patience

        self.current_k: int = K_min if mode == "growth" else K_max
        init_active = torch.zeros(K_max, dtype=torch.bool)
        init_active[:self.current_k] = True
        self.register_buffer("_active_mask", init_active)
        self._growth_counter = 0
        self._prev_thought: torch.Tensor | None = None
        self._entropy_history: list[float] = []

    def _entropy_selection(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Select latents based on attention entropy.

        Args:
            attn_weights: (B, n_latents, N_key) attention weights from cross-attention
        Returns:
            mask: (n_latents,) True for latents to keep
        """
        B, K, N = attn_weights.shape
        # Entropy per latent: H_i = -Σ_j α_ij log α_ij
        eps = 1e-8
        entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # (B, K)
        entropy = entropy.mean(dim=0) / math.log(max(N, 2))  # normalize by max entropy

        self._entropy_history = entropy.detach().cpu().tolist()

        max_entropy = entropy.max()
        threshold = self.entropy_threshold * max_entropy if max_entropy > 0 else 1.0
        mask = entropy < threshold  # (K,) — keep low-entropy (specialized) latents

        # Ensure at least K_min latents
        n_keep = mask.sum().item()
        if n_keep < self.K_min:
            # Fall back to top-K_min by lowest entropy
            _, idx = entropy.topk(self.K_min, largest=False)
            mask = torch.zeros(K, dtype=torch.bool, device=entropy.device)
            mask[idx] = True

        self.current_k = max(int(mask.sum().item()), self.K_min)
        return mask

    def _growth_step(self, thought: torch.Tensor) -> torch.Tensor:
        """Adaptive growth: add a latent if thought has converged.

        Args:
            thought: (B, K_max, d) current thought latents
        Returns:
            updated thought: (B, K_max, d) with possibly more active latents
        """
        if self._prev_thought is None:
            self._prev_thought = thought.mean(dim=0, keepdim=True).detach()
            return thought

        delta = (thought.mean(dim=0, keepdim=True) - self._prev_thought).norm().item()
        self._prev_thought = thought.mean(dim=0, keepdim=True).detach()

        if delta < self.growth_delta:
            self._growth_counter += 1
        else:
            self._growth_counter = 0

        if self._growth_counter >= self.growth_patience:
            self._growth_counter = 0
            # Find an inactive slot to activate
            if self.current_k < self.K_max:
                inactive = (~self._active_mask).nonzero(as_tuple=True)[0]
                if len(inactive) > 0:
                    slot = inactive[0].item()
                    # Initialize from mean of active latents + noise
                    active_latents = thought[:, self._active_mask]
                    mean_latent = active_latents.mean(dim=1, keepdim=True)
                    noise = torch.randn_like(mean_latent) * 0.01
                    thought[:, slot:slot + 1] = mean_latent + noise
                    self._active_mask[slot] = True
                    self.current_k += 1

        return thought

    def forward(
        self, h: torch.Tensor, embeddings: torch.Tensor, keys: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass with dynamic K regulation.

        Returns:
            delta: (B, 1, d_out) fused output
            info: dict with {"k_used": int, "entropy": list[float] | None, ...}
        """
        info = {"k_used": self.current_k}

        if self.mode == "fixed":
            return self.perceiver(h, embeddings, keys), info

        # For entropy mode, we need attention weights. Patch cross_in to capture them.
        if self.mode == "entropy":
            attn_weights = self._capture_attention(h, embeddings, keys)
            mask = self._entropy_selection(attn_weights)
            self._active_mask = mask
            info["entropy"] = self._entropy_history
            info["k_used"] = self.current_k
            info["k_masked"] = int((~mask).sum().item())

            # Forward with masked latents: replace high-entropy latents with zeros
            # The Perceiver's self-attention will effectively ignore them
            return self._forward_masked(h, embeddings, keys, mask), info

        if self.mode == "growth":
            # Mask out inactive latents before forward
            mask = self._active_mask.clone()
            output = self._forward_masked(h, embeddings, keys, mask)
            # After forward, check if we should grow
            if hasattr(self.perceiver, '_last_Z'):
                self.perceiver._last_Z = self._growth_step(self.perceiver._last_Z)
            info["k_used"] = self.current_k
            info["growth_counter"] = self._growth_counter
            return output, info

        return self.perceiver(h, embeddings, keys), info

    def _capture_attention(
        self, h: torch.Tensor, embeddings: torch.Tensor, keys: torch.Tensor | None
    ) -> torch.Tensor:
        """Do a forward pass through cross_in only, capture attention weights."""
        Z = self.perceiver.init_latents(h, keys)
        if embeddings.shape[1] > 0:
            emb_proj = self.perceiver.proj_emb(embeddings)
            # Use a temporary forward with need_weights=True
            _, attn = self.perceiver.cross_in(Z, emb_proj, emb_proj, need_weights=True)
            return attn
        return torch.zeros(Z.shape[0], Z.shape[1], 1)

    def _forward_masked(
        self, h: torch.Tensor, embeddings: torch.Tensor, keys: torch.Tensor | None,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with masked (zeroed) high-entropy latents."""
        B = h.shape[0]
        Z = self.perceiver.init_latents(h, keys)

        # Zero out high-entropy latents
        drop_mask = ~mask
        Z[:, drop_mask] = 0.0

        if embeddings.shape[1] > 0:
            emb_proj = self.perceiver.proj_emb(embeddings)
            Z_in, _ = self.perceiver.cross_in(Z, emb_proj, emb_proj, need_weights=False)
            Z = self.perceiver.norm_in(Z + Z_in)

        Z_sa, _ = self.perceiver.self_attn(Z, Z, Z, need_weights=False)
        Z = self.perceiver.norm_sa(Z + Z_sa)
        Z = self.perceiver.norm_ff(Z + self.perceiver.ffn(Z))

        h_q = self.perceiver.cross_out_q(h).unsqueeze(1)
        Z_out, _ = self.perceiver.cross_out(h_q, Z, Z, need_weights=False)
        fused = self.perceiver.norm_out(Z_out)

        return self.perceiver.out_proj(fused)

    def reset(self):
        """Reset dynamic K state (call between reasoning chains)."""
        self.current_k = self.K_max
        self._active_mask = torch.ones(self.K_max, dtype=torch.bool, device=self._active_mask.device)
        self._growth_counter = 0
        self._prev_thought = None
        self._entropy_history = []
