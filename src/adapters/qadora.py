from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_bits, sym, per_channel, dim):
        if per_channel:
            xmin = x.amin(dim=dim, keepdim=True)
            xmax = x.amax(dim=dim, keepdim=True)
        else:
            xmin = x.min()
            xmax = x.max()
        if sym:
            xmax = x.abs().max()
            xmin = -xmax
        scale = (xmax - xmin) / (2 ** n_bits - 1)
        scale = scale.clamp(min=1e-10)
        zero_point = (-xmin / scale).round().clamp(0, 2 ** n_bits - 1)
        x_q = (x / scale + zero_point).round().clamp(0, 2 ** n_bits - 1)
        x_dq = (x_q - zero_point) * scale
        ctx.save_for_backward(scale, zero_point)
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def fake_quantize(x, n_bits=8, sym=False, per_channel=True, dim=1):
    return FakeQuantize.apply(x, n_bits, sym, per_channel, dim)


class QADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))

        self.qa_n_bits = config.qa_quant_bits
        self.qa_sym = config.qa_quant_sym
        self.qa_per_channel = config.qa_quant_per_channel

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
        merged_q = fake_quantize(merged, self.qa_n_bits, self.qa_sym, self.qa_per_channel, dim=1)
        column_norm = merged_q.norm(p=2, dim=1, keepdim=True)
        normalized = merged_q / (column_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        return F.linear(x, adapted - base_weight)
