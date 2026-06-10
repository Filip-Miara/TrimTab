"""Combinatoric hybrid adapters combining techniques from multiple papers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig
from .knit_bvoran import _get_knit_stack



class GenBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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




class GenBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self._ga_initialized = False
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

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

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




class GenEVABVoRAN(LowRankAdapter):
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
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




class GenEVABVoRANGA(LowRankAdapter):
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
        self._ga_initialized = False
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

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




class GenAFABVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
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

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenAFABVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self._ga_initialized = False
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

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenAFAEVABVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenAFAEVABVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self._ga_initialized = False
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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenSRBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
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

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

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
        self._step_counter += 1
        return out




class GenSRBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
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

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

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
        self._step_counter += 1
        return out




class GenSREVABVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

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
        self._step_counter += 1
        return out




class GenSREVABVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

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
        self._step_counter += 1
        return out




class GenSRAFABVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
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

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenSRAFABVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
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

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenSRAFAEVABVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenSRAFAEVABVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenKnitBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenKnitBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenKnitEVABVoRAN(LowRankAdapter):
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
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenKnitEVABVoRANGA(LowRankAdapter):
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
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenKnitAFABVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenKnitAFABVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenKnitAFAEVABVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenKnitAFAEVABVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenKnitSRBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenKnitSRBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenKnitSREVABVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenKnitSREVABVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenKnitSRAFABVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenKnitSRAFABVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenKnitSRAFAEVABVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenKnitSRAFAEVABVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self._ga_initialized = False
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenEVAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
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
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenEVAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self._ga_initialized = False
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenAFAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenAFAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self._ga_initialized = False
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenAFAEVAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenAFAEVAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self._ga_initialized = False
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenSREBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenSREBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenSREVAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenSREVAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenSRAFAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenSRAFAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenSRAFAEVAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenSRAFAEVAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
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
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        merged_fwd = base_weight + BA_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        merged_bwd = base_weight.T + BA_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(x, fwd_delta) + F.linear(x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenKnitEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenKnitEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenKnitEVAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenKnitEVAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        return out




class GenKnitAFAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenKnitAFAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenKnitAFAEVAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenKnitAFAEVAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        return out




class GenKnitSREBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenKnitSREBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenKnitSREVAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenKnitSREVAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
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
            buf = self.activation_buffer[:self._buffer_filled]
            x_centered = buf - buf.mean(dim=0, keepdim=True)
            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)
            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self.lora_A_fwd[fuse_idx]) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self.lora_A_bwd[fuse_idx_bwd]) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        BA_fwd = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self._step_counter += 1
        return out




class GenKnitSRAFAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenKnitSRAFAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenKnitSRAFAEVAEBVoRAN(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out




class GenKnitSRAFAEVAEBVoRANGA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))
        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))
        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))
        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))
        group_size = config.edora_group_size
        if group_size <= 0:
            group_size = max(1, out_features // 4)
        self.group_size = group_size
        n_groups_fwd = math.ceil(out_features / group_size)
        n_groups_bwd = math.ceil(in_features / group_size)
        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))
        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))
        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("importance_fwd", torch.ones(config.r))
        self.register_buffer("importance_bwd", torch.ones(config.r))
        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, config.r // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A_fwd", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B_fwd", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))
        self.register_buffer("ve_A_bwd", torch.randn(ve_rank, out_features) / math.sqrt(out_features))
        self.register_buffer("ve_B_bwd", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))
        self.norm = nn.LayerNorm(out_features)
        self.eps = config.dora_eps
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
        self._ga_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        for A in [self.lora_A_fwd, self.lora_A_bwd]:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in [self.lora_B_fwd, self.lora_B_bwd]:
            nn.init.zeros_(B)
        nn.init.ones_(self.magnitude_fwd)
        nn.init.ones_(self.magnitude_bwd)
        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:
            nn.init.normal_(v, mean=0.0, std=0.01)

    @property
    def trainable_params(self) -> int:
        return (self.lora_A_fwd.numel() + self.lora_B_fwd.numel()
                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()
                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()
                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()
                + sum(p.numel() for p in self.norm.parameters()))

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
            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)
            nn.init.zeros_(self.lora_B_fwd)
            y_centered = buf - buf.mean(dim=0, keepdim=True)
            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)
            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)
            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)
            nn.init.zeros_(self.lora_B_bwd)
            self._buffer_filled = 0

    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G_fwd = grad_output.T @ x
            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)
            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)
            G_bwd = x.T @ grad_output
            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)
            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)

    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
            act = self._get_alpha()
            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling
            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]
            self.importance_fwd.copy_(grad.norm(p=2, dim=1))
            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling
            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]
            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A_fwd.device
            alpha = self._get_alpha()
            W_eff_fwd = base_weight + (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
            sorted_idx = torch.argsort(self.importance_fwd, descending=True)
            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())
            keep_r = min(keep_r, self.config.r)
            keep_idx = sorted_idx[:keep_r]
            fuse_idx = sorted_idx[keep_r:]
            if len(fuse_idx) > 0:
                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ self._apply_activation(self.lora_A_fwd[fuse_idx], alpha)) * self.scaling
                W_eff_fwd = W_eff_fwd - delta_fuse
            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)
            A_new = Vh[:self.config.r].contiguous().to(device)
            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)
            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)
            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)
            self.importance_fwd = torch.ones(self.config.r, device=device)
            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)
            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())
            keep_r_bwd = min(keep_r_bwd, self.config.r)
            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]
            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]
            if len(fuse_idx_bwd) > 0:
                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ self._apply_activation(self.lora_A_bwd[fuse_idx_bwd], alpha)) * self.scaling
                W_eff_bwd = W_eff_bwd - delta_fuse_bwd
            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)
            A_new_b = Vh_b[:self.config.r].contiguous().to(device)
            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)
            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)
            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)
            self.importance_bwd = torch.ones(self.config.r, device=device)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            knit_x = x
            for prev_x in stack:
                knit_x = knit_x + prev_x.detach()
            stack.append(x.detach())
        else:
            knit_x = x
        alpha = self._get_alpha()
        BA_fwd = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, alpha)) * self.scaling
        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)
        merged_fwd = base_weight + BA_fwd + ve_fwd
        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)
        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)
        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]
        adapted_fwd = mag_fwd * normalized_fwd
        fwd_delta = adapted_fwd - base_weight
        BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, alpha)) * self.scaling
        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)
        merged_bwd = base_weight.T + BA_bwd + ve_bwd
        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)
        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)
        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]
        adapted_bwd = mag_bwd * normalized_bwd
        bwd_delta = adapted_bwd - base_weight.T
        out = F.linear(knit_x, fwd_delta) + F.linear(knit_x, bwd_delta.T)
        out = self.norm(out)
        self.step += 1
        self._step_counter += 1
        return out


