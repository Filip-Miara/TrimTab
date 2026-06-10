from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig
from .knit_bvoran import _get_knit_stack


class KnitEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))

        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))

        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank

        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))

        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x

        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight

        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T

        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        return out
