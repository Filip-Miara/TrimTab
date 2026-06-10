from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class BNBQuantLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        import bitsandbytes as bnb
        self.out_features, self.in_features = weight.shape
        w_4bit = bnb.nn.Linear4bit(
            self.in_features, self.out_features, bias=False,
            compute_dtype=torch.bfloat16, quant_type="nf4",
        )
        w_4bit.weight.data = weight
        w_4bit = w_4bit.to("cuda")
        self.weight = w_4bit.weight
        self.bias = None
        if bias is not None:
            self.bias = nn.Parameter(bias.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias)


class QDoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.eps = config.dora_eps
        self.quantized = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    def quantize_base(self, base_weight: torch.Tensor):
        self.register_buffer("_base_weight_q", base_weight.cpu())
        try:
            import bitsandbytes as bnb
            w_4bit = bnb.nn.Linear4bit(
                base_weight.shape[1], base_weight.shape[0], bias=False,
                compute_dtype=torch.bfloat16, quant_type="nf4",
            )
            w_4bit.weight.data = base_weight.contiguous()
            w_4bit = w_4bit.to(base_weight.device)
            self._quantized_weight = w_4bit.weight
            self.quantized = True
        except Exception:
            self._quantized_weight = base_weight
            self.quantized = True

    @property
    def trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        column_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (column_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        return F.linear(x, adapted - base_weight)
