from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class SRBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))

        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))

        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))

        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A_fwd, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_fwd)
        nn.init.kaiming_uniform_(self.lora_A_bwd, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_bwd)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            rank_scores = grad.norm(p=2, dim=1)
            self.importance_fwd.copy_(rank_scores)

            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            rank_scores_bwd = grad_bwd.norm(p=2, dim=1)
            self.importance_bwd.copy_(rank_scores_bwd)

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)

            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]

            if len(fuse_idx) > 0:
                A_keep = self.lora_A_fwd[keep_idx]
                B_keep = self.lora_B_fwd[:, keep_idx]
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse

            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)

            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight

        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T

        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)

        self._step_counter += 1
        return out
