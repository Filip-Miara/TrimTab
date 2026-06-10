from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig
from .spectral_utils import SeLoRALoRA, spectral_transform


class SeBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        sparse_ratio = config.extra_kwargs.get("selora_sparse_ratio", 0.4)
        spec_type = config.extra_kwargs.get("selora_spectral_type", "wavelet")

        self.se_A_fwd = SeLoRALoRA(config.r, in_features, sparse_ratio, spec_type)
        self.se_B_fwd = SeLoRALoRA(out_features, config.r, sparse_ratio, spec_type)
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))

        self.se_A_bwd = SeLoRALoRA(config.r, out_features, sparse_ratio, spec_type)
        self.se_B_bwd = SeLoRALoRA(in_features, config.r, sparse_ratio, spec_type)
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))

        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        self.se_A_fwd.reset_parameters()
        self.se_B_fwd.sparse_F.data.zero_()
        self.se_A_bwd.reset_parameters()
        self.se_B_bwd.sparse_F.data.zero_()
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)

    @property
    def trainable_params(self) -> int:
        return (self.se_A_fwd.sparse_F.numel() + self.se_B_fwd.sparse_F.numel()
                + self.magnitude_fwd.numel()
                + self.se_A_bwd.sparse_F.numel() + self.se_B_bwd.sparse_F.numel()
                + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        A_tilde_fwd = self.se_A_fwd.get_matrix()
        B_tilde_fwd = self.se_B_fwd.get_matrix()
        BA_fwd = (B_tilde_fwd @ A_tilde_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight

        A_tilde_bwd = self.se_A_bwd.get_matrix()
        B_tilde_bwd = self.se_B_bwd.get_matrix()
        BA_bwd = (B_tilde_bwd @ A_tilde_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T

        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        return out
