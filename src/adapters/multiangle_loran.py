from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class MultiAngleLoRAN(LowRankAdapter):
    """Multi-angle diagonal magnitude decomposition.

    Applies learnable band-pass magnitude scaling along N projection
    angles.  Each angle θ defines bands via banker's-rounding of
    j·cos(θ) + l·sin(θ), with per-band magnitude and L2 normalization.

    Default angles: 0° (rows), 45° (main diag), 90° (cols), 135° (anti diag).
    """

    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))

        angles_deg = config.extra_kwargs.get("multiangle_angles", [0, 45, 90, 135])
        self.n_angles = len(angles_deg)
        self.angles_deg = angles_deg

        j = torch.arange(out_features, dtype=torch.float).view(-1, 1)
        l = torch.arange(in_features, dtype=torch.float).view(1, -1)

        band_maps = []
        for ang in angles_deg:
            rad = math.radians(ang)
            c = math.cos(rad)
            s = math.sin(rad)
            raw = j * c + l * s
            band = torch.round(raw).long()
            band = band - band.min()
            band_maps.append(band)

        self.register_buffer("band_maps", torch.stack(band_maps))
        band_sizes = [b.max().item() + 1 for b in band_maps]
        for i, n in enumerate(band_sizes):
            self.register_parameter(f"mag_angle_{i}", nn.Parameter(torch.empty(n)))

        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        for i in range(self.n_angles):
            getattr(self, f"mag_angle_{i}").data.fill_(1.0)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        for i in range(self.n_angles):
            n += getattr(self, f"mag_angle_{i}").numel()
        return n

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        M = base_weight + BA

        for i in range(self.n_angles):
            bm = self.band_maps[i]
            mag = getattr(self, f"mag_angle_{i}")
            sq = torch.zeros(mag.numel(), device=M.device, dtype=M.dtype)
            sq.scatter_add_(0, bm.ravel(), (M ** 2).ravel())
            norm = sq.sqrt().clamp(min=self.eps)
            M = M / norm[bm] * mag[bm]

        col_norm = M.norm(p=2, dim=1, keepdim=True)
        normalized = M / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        return F.linear(x, adapted - base_weight)
