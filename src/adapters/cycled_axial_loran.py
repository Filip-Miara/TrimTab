from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class CycledAxialBoRA(LowRankAdapter):
    """Single-direction adapter that cycles between column-axis and row-axis
    magnitude scaling rather than applying both simultaneously.

    BoRA applies both column scaling (per-output) and row scaling (per-input)
    in the same step, which creates conflicting gradient signals — both
    axes pull the weight update in different directions simultaneously.

    This variant trains only ONE axis at a time, cycling every N steps.
    When column-scale is active, magnitude_col is trained (magnitude_row
    frozen at 1.0). When row-scale is active, magnitude_row is trained.

    Unlike CycledBoRAN which cycles between fwd/bwd *branches*, this
    cycles between scaling *axes* within a single direction, isolating
    the gradient conflict that arises from simultaneous multi-axis scaling.
    """

    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_col = nn.Parameter(torch.empty(out_features))
        self.magnitude_row = nn.Parameter(torch.empty(in_features))

        self.eps = config.dora_eps
        self.has_norm = config.extra_kwargs.get("has_norm", True)
        if self.has_norm:
            self.norm = nn.LayerNorm(out_features)

        self.cycle_interval = config.extra_kwargs.get("cycle_interval", 10)
        self.register_buffer("_cycle_step", torch.tensor(0, dtype=torch.long))
        self._active_axis = 0  # 0 = column, 1 = row

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude_col)
        nn.init.ones_(self.magnitude_row)

    @property
    def trainable_params(self) -> int:
        n = (self.lora_A.numel() + self.lora_B.numel()
             + self.magnitude_col.numel() + self.magnitude_row.numel())
        if self.has_norm:
            n += sum(p.numel() for p in self.norm.parameters())
        return n

    @property
    def active_params(self) -> int:
        if self._active_axis == 0:
            return self.magnitude_col.numel()
        return self.magnitude_row.numel()

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        if self._cycle_step % self.cycle_interval == 0:
            self._active_axis = int(self._cycle_step // self.cycle_interval) % 2

        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA

        if self._active_axis == 0:
            # Column-scale (DoRA-style): per-output magnitude
            col_norm = merged.norm(p=2, dim=1, keepdim=True)
            normalized = merged / (col_norm + self.eps)
            adapted = self.magnitude_col[:, None] * normalized
        else:
            # Row-scale (per-input magnitude)
            row_norm = merged.norm(p=2, dim=0, keepdim=True)
            normalized = merged / (row_norm + self.eps)
            adapted = self.magnitude_row[None, :] * normalized

        out = F.linear(x, adapted - base_weight)
        if self.has_norm:
            out = self.norm(out)
        self._cycle_step += 1
        return out
