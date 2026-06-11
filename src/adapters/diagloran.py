from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


class DiagLoRAN(LowRankAdapter):
    """DiagLoRA: Bi-diagonal magnitude decomposition.

    Applies magnitude scaling along main-diagonal (Toeplitz) and
    anti-diagonal (Hankel) groups, forming a frequency-domain mask
    on the LoRA update.  The two-pass pattern mirrors BoRA:

      V  = m^d ⊙ normalize_diag(W₀ + AB)     main-diagonal
      W  = m^a ⊙ normalize_anti(V)           anti-diagonal
    """

    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)

        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))

        n_diags = out_features + in_features - 1
        self.mag_diag_fwd = nn.Parameter(torch.empty(n_diags))
        self.mag_anti_fwd = nn.Parameter(torch.empty(n_diags))
        self.mag_diag_bwd = nn.Parameter(torch.empty(n_diags))
        self.mag_anti_bwd = nn.Parameter(torch.empty(n_diags))

        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps

        self.register_buffer("_j_idx", torch.arange(max(in_features, out_features)).view(-1, 1))
        self.register_buffer("_l_idx", torch.arange(max(in_features, out_features)).view(1, -1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A_fwd, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_fwd)
        nn.init.kaiming_uniform_(self.lora_A_bwd, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_bwd)
        nn.init.ones_(self.mag_diag_fwd)
        nn.init.ones_(self.mag_anti_fwd)
        nn.init.ones_(self.mag_diag_bwd)
        nn.init.ones_(self.mag_anti_bwd)

    @property
    def trainable_params(self) -> int:
        n_diags = self.out_features + self.in_features - 1
        n = (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
             + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
             + 4 * n_diags
             + sum(p.numel() for p in self.norm.parameters()))
        return n

    @staticmethod
    def _diag_gathersq(M, k_map, n_diags):
        sq = torch.zeros(n_diags, device=M.device, dtype=M.dtype)
        sq.scatter_add_(0, k_map.ravel(), (M ** 2).ravel())
        return sq.sqrt().clamp(min=1e-8)

    def _apply_transform(self, M, mag_diag, mag_anti):
        out, inn = M.shape
        offset = inn - 1
        j = self._j_idx[:out, :inn]
        l = self._l_idx[:out, :inn]
        k_map = j - l + offset
        s_map = j + l
        n = out + inn - 1

        # Main-diagonal normalise & scale
        d_norm = self._diag_gathersq(M, k_map, n)
        V = M / d_norm[k_map]
        V = V * mag_diag[k_map]

        # Anti-diagonal normalise & scale
        a_norm = self._diag_gathersq(V, s_map, n)
        H = V / a_norm[s_map]
        H = H * mag_anti[s_map]
        return H

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)

        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        adapted_fwd = self._apply_transform(base_weight + BA_fwd, self.mag_diag_fwd, self.mag_anti_fwd)

        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        adapted_bwd = self._apply_transform(base_weight.T + BA_bwd, self.mag_diag_bwd, self.mag_anti_bwd)

        out = F.linear(x, adapted_fwd - base_weight) + F.linear(x, (adapted_bwd - base_weight.T).T)
        out = self.norm(out)
        return out
