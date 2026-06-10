from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class EVABVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))

        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))

        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0

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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def collect_activation(self, x: torch.Tensor):
        with torch.no_grad():
            x_flat = x.view(-1, self.in_features)
            if self._buffer_filled == 0:
                self.activation_buffer = x_flat[:self.buffer_size].clone()
                self._buffer_filled = x_flat.shape[0]
            elif self._buffer_filled < self.buffer_size:
                remaining = self.buffer_size - self._buffer_filled
                take = min(remaining, x_flat.shape[0])
                self.activation_buffer = torch.cat(
                    [self.activation_buffer[:self._buffer_filled], x_flat[:take]]
                )
                self._buffer_filled += take

    def eva_init_from_buffer(self):
        if self._buffer_filled < 2:
            return
        with torch.no_grad():
            x_centered = self.activation_buffer[:self._buffer_filled] - self.activation_buffer[:self._buffer_filled].mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)

            y = self.activation_buffer[:self._buffer_filled]
            y_centered = y - y.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)

            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight

        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T

        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        return out
