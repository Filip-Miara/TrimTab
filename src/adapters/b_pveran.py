from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class BPVERAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        poly_order = config.extra_kwargs.get("pera_poly_order", 2)
        self.poly_order = poly_order
        eff_r = config.r * poly_order

        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.poly_coeff_fwd = nn.Parameter(torch.randn(poly_order) * 0.01)
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))

        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.poly_coeff_bwd = nn.Parameter(torch.randn(poly_order) * 0.01)
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))

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

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.poly_coeff_fwd.numel()
                + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.poly_coeff_bwd.numel()
                + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _poly_product(self, A: torch.Tensor, B: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        BA = B @ A
        result = coeffs[0] * BA
        for order in range(1, len(coeffs)):
            result = result + coeffs[order] * (BA ** (order + 1))
        return result

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        BA_fwd = self._poly_product(self.lora_A_fwd, self.lora_B_fwd, self.poly_coeff_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight

        BA_bwd = self._poly_product(self.lora_A_bwd, self.lora_B_bwd, self.poly_coeff_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T

        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        return out
