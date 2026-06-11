from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class PlainLoRA(LowRankAdapter):
    """Vanilla LoRA — no magnitude scaling, no norm. Just BAΔ."""
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, config.r))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        BA = (self.lora_B @ self.lora_A) * self.scaling
        return F.linear(x, BA)
