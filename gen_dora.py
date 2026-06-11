"""Generate combinatoric DoRA-style (single-direction) variants ±LayerNorm."""
from __future__ import annotations

import os
import math


def cls(has_afa, has_ga, has_sr, has_knit, has_eva, has_norm):
    base = "DoRAN" if has_norm else "DoRA"
    parts = []
    if has_knit: parts.append("Knit")
    if has_sr: parts.append("SR")
    if has_afa: parts.append("AFA")
    if has_eva: parts.append("EVA")
    parts.append(base)
    if has_ga: parts.append("GA")

    class_name = "GenD" + "".join(parts)
    r_ = "self.config.r"

    code = f"""
class {class_name}(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        self.lora_A = nn.Parameter(torch.empty({r_}, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, {r_}))
        self.magnitude = nn.Parameter(torch.empty(out_features))
"""
    if has_norm:
        code += f"""        self.norm = nn.LayerNorm(out_features)
"""
    if has_afa:
        code += f"""        self.anneal_rate = config.extra_kwargs.get("afa_anneal_rate", 0.05)
        self.anneal_start = config.extra_kwargs.get("afa_anneal_start", 0)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
"""
    if has_sr:
        code += f"""        self.register_buffer("importance", torch.ones({r_}))
        self.sr_step = config.extra_kwargs.get("sr_recompose_step", 50)
        self._step_counter = 0
"""
    if has_knit:
        code += f"""        ve_rank = config.dvora_ve_rank
        if ve_rank <= 0:
            ve_rank = max(1, {r_} // 2)
        self.ve_rank = ve_rank
        self.register_buffer("ve_A", torch.randn(ve_rank, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))
        self.ve_lambda = nn.Parameter(torch.empty(ve_rank))
        self.module_path = ""
        self.knit_mode = config.extra_kwargs.get("knit_mode", "in_dense")
"""
    if has_eva:
        code += f"""        self.register_buffer("activation_buffer", torch.empty(0, in_features))
        self.buffer_size = config.extra_kwargs.get("eva_buffer_size", 128)
        self._buffer_filled = 0
"""
    if has_ga:
        code += f"""        self._ga_initialized = False
"""
    code += f"""        self.eps = config.dora_eps
        self.reset_parameters()

    def reset_parameters(self):
"""
    if has_knit:
        code += f"""        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
        nn.init.normal_(self.ve_lambda, mean=0.0, std=0.01)
"""
    else:
        code += f"""        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.ones_(self.magnitude)
"""
    code += f"""
    @property
    def trainable_params(self) -> int:
        n = self.lora_A.numel() + self.lora_B.numel() + self.magnitude.numel()
"""
    if has_knit:
        code += f"""        n += self.ve_lambda.numel()
"""
    if has_norm:
        code += f"""        n += sum(p.numel() for p in self.norm.parameters())
"""
    code += f"""        return n
"""
    if has_afa:
        code += f"""
    def _get_alpha(self) -> float:
        t = max(0, self.step.item() - self.anneal_start)
        return math.exp(-self.anneal_rate * t)

    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * x + alpha * torch.tanh(x)
"""
    if has_eva:
        code += f"""
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
"""
    if has_ga:
        code += f"""
    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):
        with torch.no_grad():
            G = grad_output.T @ x
            U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
            scale = (S[:self.config.r].mean() / self.config.r) ** 0.5
            self.lora_A.data = (Vh[:self.config.r] * scale).to(self.lora_A.dtype)
            self.lora_B.data = (U[:, :self.config.r] * scale).to(self.lora_B.dtype)
"""
    if has_sr:
        code += f"""
    def update_importance(self, x: torch.Tensor):
        with torch.no_grad():
"""
        if has_afa:
            code += f"""            alpha = self._get_alpha()
            BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
"""
        else:
            code += f"""            BA = (self.lora_B @ self.lora_A) * self.scaling
"""
        code += f"""            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A, retain_graph=True)[0]
            self.importance.copy_(grad.norm(p=2, dim=1))

    def recompose(self, base_weight: torch.Tensor):
        with torch.no_grad():
            device = self.lora_A.device
"""
        if has_afa:
            code += f"""            alpha = self._get_alpha()
            W_eff = base_weight + (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
"""
        else:
            code += f"""            W_eff = base_weight + (self.lora_B @ self.lora_A) * self.scaling
"""
        code += f"""            sorted_idx = torch.argsort(self.importance, descending=True)
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
"""
    code += f"""
    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        base_weight = base_weight.to(x.dtype)
"""
    if has_knit:
        code += f"""        stack = _get_knit_stack(self.module_path)
        if self.knit_mode == "in_dense":
            for prev_x in stack:
                x = x + prev_x.detach()
            stack.append(x.detach())
"""
    if has_afa:
        code += f"""        alpha = self._get_alpha()
        BA = (self.lora_B @ self._apply_activation(self.lora_A, alpha)) * self.scaling
"""
    else:
        code += f"""        BA = (self.lora_B @ self.lora_A) * self.scaling
"""
    if has_knit:
        code += f"""        ve = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = base_weight + BA + ve
"""
    else:
        code += f"""        merged = base_weight + BA
"""
    code += f"""        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        out = F.linear(x, adapted - base_weight)
"""
    if has_norm:
        code += f"""        out = self.norm(out)
"""
    if has_afa:
        code += f"""        self.step += 1
"""
    if has_sr:
        code += f"""        self._step_counter += 1
"""
    code += f"""        return out
"""
    return class_name, code


# All 2^6 = 64 flag combinations (5 technique flags + norm)
variants = []
for has_knit in (False, True):
    for has_sr in (False, True):
        for has_afa in (False, True):
            for has_eva in (False, True):
                for has_ga in (False, True):
                    for has_norm in (False, True):
                        base = "doran" if has_norm else "dora"
                        parts = []
                        if has_knit: parts.append("knit")
                        if has_sr: parts.append("sr")
                        if has_afa: parts.append("afa")
                        if has_eva: parts.append("eva")
                        parts.append(base)
                        if has_ga: parts.append("ga")
                        suffix = "_".join(parts)
                        variants.append((suffix, has_afa, has_ga, has_sr, has_knit, has_eva, has_norm))

out_path = os.path.join(os.path.dirname(__file__), "src", "adapters", "dora_combo_adapters.py")
with open(out_path, "w") as f:
    f.write('"""DoRA-style combinatoric adapters (single-direction column magnitude)."""\n\n')
    f.write("from __future__ import annotations\n\n")
    f.write("import math\n\n")
    f.write("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n")
    f.write("from .base import LowRankAdapter, AdapterConfig\n")
    f.write("from .knit_bvoran import _get_knit_stack\n\n")
    for idx, (suffix, has_afa, has_ga, has_sr, has_knit, has_eva, has_norm) in enumerate(variants):
        class_name, code = cls(has_afa, has_ga, has_sr, has_knit, has_eva, has_norm)
        f.write(code)
        print(f"  [{idx+1}/{len(variants)}] Generated {class_name}")

print(f"\nWritten {len(variants)} classes to {out_path}")
print(f"File size: {os.path.getsize(out_path):,} bytes")
