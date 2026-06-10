from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32


@dataclass
class AdapterConfig:
    r: int = 8
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    init_type: Literal["kaiming", "gaussian", "zeros"] = "kaiming"
    use_rslora: bool = False

    dora_eps: float = 1e-5
    bora_shared_magnitude: bool = False
    edora_group_size: int = 0
    dvora_ve_rank: int = 0
    qa_quant_bits: int = 8
    qa_quant_sym: bool = False
    qa_quant_per_channel: bool = True

    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class LowRankAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.r = config.r
        self.model_dtype = DTYPE
        self.scaling = config.lora_alpha / config.r if not config.use_rslora else config.lora_alpha / math.sqrt(config.r)
        self.dropout = nn.Dropout(config.lora_dropout) if config.lora_dropout > 0 else nn.Identity()

    @property
    def trainable_params(self) -> int:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def adapt_linear_layer(
    layer: nn.Linear,
    adapter_cls: type[LowRankAdapter],
    config: AdapterConfig,
    weight: torch.Tensor | None = None,
) -> nn.Module:
    adapter = adapter_cls(layer.in_features, layer.out_features, config)
    w = weight if weight is not None else layer.weight.data.clone()
    adapter = adapter.to(dtype=w.dtype)
    adapter.model_dtype = w.dtype
    wrapped = AdapterWrappedLinear(layer, adapter, w)
    return wrapped


class AdapterWrappedLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, adapter: LowRankAdapter, frozen_weight: torch.Tensor):
        super().__init__()
        self.base_linear = base_linear
        self.adapter = adapter
        self.frozen_weight = nn.Parameter(frozen_weight, requires_grad=False)
        self.base_linear.weight.requires_grad = False
        if hasattr(self.base_linear, "bias") and self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False
        # Free original weight data — we use frozen_weight for compute
        try:
            self.base_linear.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        except Exception:
            pass

    @property
    def trainable_params(self) -> int:
        return self.adapter.trainable_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fw = self.frozen_weight.to(x.dtype)
        base_out = F.linear(x, fw, self.base_linear.bias)
        adapter_out = self.adapter(x, fw)
        return base_out + adapter_out
