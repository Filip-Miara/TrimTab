from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class CycledBoRAN(LowRankAdapter):
    """CycledBoRAN (formerly CycledBVoRAN): Bidirectional adapter that alternates between forward and backward
    branches rather than applying both simultaneously.

    Every `cycle_interval` training steps the active branch switches, so
    only one direction contributes to the gradient at any given step.
    This remediates the conflicting-gradient-signal issue observed in
    standard bidirectional adapters.

    Reference: AltLoRA (arXiv:2505.12455) alternates between A/B matrices;
    this extends the idea to alternation along the fwd/bwd information
    flow direction.
    """

    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))

        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))

        self.eps = config.dora_eps
        self.has_norm = config.extra_kwargs.get("has_norm", True)
        if self.has_norm:
            self.norm = nn.LayerNorm(out_features)

        self.cycle_interval = config.extra_kwargs.get("cycle_interval", 10)
        self.register_buffer("_cycle_step", torch.tensor(0, dtype=torch.long))
        self._active_dir = 0  # 0 = fwd, 1 = bwd

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
        n = (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
             + self.magnitude_fwd.numel()
             + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
             + self.magnitude_bwd.numel())
        if self.has_norm:
            n += sum(p.numel() for p in self.norm.parameters())
        return n

    @property
    def active_params(self) -> int:
        """Number of trainable params that are contributing this step."""
        if self._active_dir == 0:
            return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                    + self.magnitude_fwd.numel())
        else:
            return (self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                    + self.magnitude_bwd.numel())

    def _make_delta(self, A, B, mag, base_w):
        BA = (B @ A) * self.scaling
        merged = base_w + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = mag[:, None] * normalized
        return adapted - base_w

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        # Toggle direction at cycle boundaries
        if self._cycle_step % self.cycle_interval == 0:
            self._active_dir = int(self._cycle_step // self.cycle_interval) % 2

        if self._active_dir == 0:
            delta = self._make_delta(self.lora_A_fwd, self.lora_B_fwd,
                                     self.magnitude_fwd, base_weight)
            out = F.linear(x, delta)
        else:
            delta = self._make_delta(self.lora_A_bwd, self.lora_B_bwd,
                                     self.magnitude_bwd, base_weight.T)
            out = F.linear(x, delta.T)

        if self.has_norm:
            out = self.norm(out)
        self._cycle_step += 1
        return out
