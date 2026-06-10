from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig
from .spectral_utils import SeLoRALoRA


class ESeBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        sparse_ratio = config.extra_kwargs.get("selora_sparse_ratio", 0.4)
        spec_type = config.extra_kwargs.get("selora_spectral_type", "wavelet")

        self.se_A_fwd = SeLoRALoRA(config.r, in_features, sparse_ratio, spec_type)
        self.se_B_fwd = SeLoRALoRA(out_features, config.r, sparse_ratio, spec_type)
        self.se_A_bwd = SeLoRALoRA(config.r, out_features, sparse_ratio, spec_type)
        self.se_B_bwd = SeLoRALoRA(in_features, config.r, sparse_ratio, spec_type)

        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))

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
                + self.se_A_bwd.sparse_F.numel() + self.se_B_bwd.sparse_F.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        BA_fwd = (self.se_B_fwd.get_matrix() @ self.se_A_fwd.get_matrix()) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight

        BA_bwd = (self.se_B_bwd.get_matrix() @ self.se_A_bwd.get_matrix()) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T

        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        return out
