"""DoRA-style combinatoric adapters (single-direction column magnitude)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig
from .knit_bvoran import _get_knit_stack


class GenDDoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        return out

class GenDDoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        return out

class GenDDoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        return out

class GenDDoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        return out

class GenDEVADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        return out

class GenDEVADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        return out

class GenDEVADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        return out

class GenDEVADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        return out

class GenDAFADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        return out

class GenDAFADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        return out

class GenDAFADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        return out

class GenDAFADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        return out

class GenDAFAEVADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        return out

class GenDAFAEVADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        return out

class GenDAFAEVADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        return out

class GenDAFAEVADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        return out

class GenDSRDoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self._step_counter += 1
        return out

class GenDSRDoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self._step_counter += 1
        return out

class GenDSRDoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self._step_counter += 1
        return out

class GenDSRDoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self._step_counter += 1
        return out

class GenDSREVADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self._step_counter += 1
        return out

class GenDSREVADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self._step_counter += 1
        return out

class GenDSREVADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self._step_counter += 1
        return out

class GenDSREVADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA = (self.lora_B @ self.lora_A) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self._step_counter += 1
        return out

class GenDSRAFADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        self._step_counter += 1
        return out

class GenDSRAFADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out

class GenDSRAFADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        self._step_counter += 1
        return out

class GenDSRAFADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out

class GenDSRAFAEVADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        self._step_counter += 1
        return out

class GenDSRAFAEVADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out

class GenDSRAFAEVADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        self._step_counter += 1
        return out

class GenDSRAFAEVADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        merged = base_weight + BA
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out

class GenDKnitDoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        return out

class GenDKnitDoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        return out

class GenDKnitDoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        return out

class GenDKnitDoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        return out

class GenDKnitEVADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        return out

class GenDKnitEVADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        return out

class GenDKnitEVADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        return out

class GenDKnitEVADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        return out

class GenDKnitAFADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        return out

class GenDKnitAFADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        return out

class GenDKnitAFADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        return out

class GenDKnitAFADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        return out

class GenDKnitAFAEVADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        return out

class GenDKnitAFAEVADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        return out

class GenDKnitAFAEVADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        return out

class GenDKnitAFAEVADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        return out

class GenDKnitSRDoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self._step_counter += 1
        return out

class GenDKnitSRDoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self._step_counter += 1
        return out

class GenDKnitSRDoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self._step_counter += 1
        return out

class GenDKnitSRDoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self._step_counter += 1
        return out

class GenDKnitSREVADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self._step_counter += 1
        return out

class GenDKnitSREVADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self._step_counter += 1
        return out

class GenDKnitSREVADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self._step_counter += 1
        return out

class GenDKnitSREVADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B @ self.lora_A) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        BA = (self.lora_B @ self.lora_A) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self._step_counter += 1
        return out

class GenDKnitSRAFADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        self._step_counter += 1
        return out

class GenDKnitSRAFADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out

class GenDKnitSRAFADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        self._step_counter += 1
        return out

class GenDKnitSRAFADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out

class GenDKnitSRAFAEVADoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        self._step_counter += 1
        return out

class GenDKnitSRAFAEVADoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out

class GenDKnitSRAFAEVADoRAGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        self.step += 1
        self._step_counter += 1
        return out

class GenDKnitSRAFAEVADoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty(self.config.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.config.r))
        self.magnitude = nn.Parameter(torch.empty(out_features))
        self.norm = nn.LayerNorm(out_features)
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance", torch.ones(self.config.r))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, self.config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self._ga_initialized = False
        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
        n += self.ve_lambda.numel()
        n += sum(p.numel() for p in self.norm.parameters())
        return n

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A.data = Vh[:self.config.r].to(self.lora_A.dtype)
            nn.init.zeros_(self.lora_B)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance, descending=True)
            keep_r = max(1, torch.sum(self.importance > 0.1 * self.importance.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B[:, fuse_idx] @ self.lora_A[fuse_idx]) * self.scaling
                W_eff = W_eff - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A.data = A_new.to(self.lora_A.dtype)
            self.lora_B.data = B_new.to(self.lora_B.dtype)
            self.importance = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out
