from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class DVoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))

        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank

        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))

        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
                + self.ve_lambda.numel())

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve_update = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve_update

        column_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (column_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        return F.linear(x, adapted - base_weight)
