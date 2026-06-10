from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class BVAuroRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))

        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))

        hidden_factor = config.extra_kwargs.get("aurora_hidden_factor", 2)
        hidden_r = max(1, config.r * hidden_factor)
        self.ANL_fwd = nn.Sequential(
            nn.Linear(config.r, hidden_r),
            nn.GELU(),
            nn.Linear(hidden_r, config.r),
        )
        self.ANL_bwd = nn.Sequential(
            nn.Linear(config.r, hidden_r),
            nn.GELU(),
            nn.Linear(hidden_r, config.r),
        )

        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A_fwd, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_fwd)
        nn.init.kaiming_uniform_(self.lora_A_bwd, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_bwd)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for m in [self.ANL_fwd, self.ANL_bwd]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.ANL_fwd.parameters())
                + sum(p.numel() for p in self.ANL_bwd.parameters())
                + sum(p.numel() for p in self.norm.parameters()))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        A_T_fwd = self.lora_A_fwd.T
        A_nl_fwd = self.ANL_fwd(A_T_fwd).T
        BA_fwd = (self.lora_B_fwd @ A_nl_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight

        A_T_bwd = self.lora_A_bwd.T
        A_nl_bwd = self.ANL_bwd(A_T_bwd).T
        BA_bwd = (self.lora_B_bwd @ A_nl_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T

        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        return out
