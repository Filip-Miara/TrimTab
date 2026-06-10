from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class EDoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, config.r))

        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups = math.ceil(out_features / group_size)
        self.magnitude = nn.Parameter(torch.empty(n_groups))

        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        column_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (column_norm + self.eps)

        mag = self.magnitude.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted = mag * normalized
        return F.linear(x, adapted - base_weight)
