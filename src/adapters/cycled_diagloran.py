from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


def _round_half_even(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x + 1e-8)


class CycledDiagLoRA(LowRankAdapter):
    """Single-direction DiagLoRA that alternates between main-diagonal and
    anti-diagonal band scaling rather than applying both simultaneously.

    When diag-active: only mag_diag contributes (mag_anti frozen at 1.0).
    When anti-active: only mag_anti contributes (mag_diag frozen at 1.0).
    """

    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.mag_diag = nn.Parameter(torch.empty(out_features + in_features - 1))
        self.mag_anti = nn.Parameter(torch.empty(out_features + in_features - 1))

        self.has_norm = config.extra_kwargs.get("has_norm", True)
        if self.has_norm:
            self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps

        self.cycle_interval = config.extra_kwargs.get("cycle_interval", 10)
        self.register_buffer("_cycle_step", torch.tensor(0, dtype=torch.long))
        self._active_band = 0  # 0 = main diag, 1 = anti diag

        # Precompute band maps
        j = torch.arange(out_features, dtype=torch.float).view(-1, 1)
        l = torch.arange(in_features, dtype=torch.float).view(1, -1)
        diag_band = _round_half_even(j - l).long()
        diag_band = diag_band - diag_band.min()
        anti_band = _round_half_even(j + l).long()
        anti_band = anti_band - anti_band.min()
        self.register_buffer("_diag_map", diag_band)
        self.register_buffer("_anti_map", anti_band)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.ones_(self.mag_diag)
        nn.init.ones_(self.mag_anti)

    @property
    def trainable_params(self) -> int:
        n = (self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
             + self.mag_diag.numel() + self.mag_anti.numel())
        if self.has_norm:
            n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _normalize_and_scale(self, M, band_map, mag):
        sq = torch.zeros(mag.numel(), device=M.device, dtype=M.dtype)
        sq.scatter_add_(0, band_map.ravel(), (M ** 2).ravel())
        norm = sq.sqrt().clamp(min=self.eps)
        return M / norm[band_map] * mag[band_map]

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        if self._cycle_step % self.cycle_interval == 0:
            self._active_band = int(self._cycle_step // self.cycle_interval) % 2

        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA

        if self._active_band == 0:
            V = self._normalize_and_scale(merged, self._diag_map, self.mag_diag)
        else:
            V = self._normalize_and_scale(merged, self._anti_map, self.mag_anti)

        col_norm = V.norm(p=2, dim=1, keepdim=True)
        normalized = V / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        if self.has_norm:
            out = self.norm(out)
        self._cycle_step += 1
        return out
