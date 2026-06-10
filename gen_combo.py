import os

def esc(s):
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")

CLASSES = []

def cls(name, is_eb, has_afa, has_ga, has_sr, has_knit, has_eva):
    base_name = f"EBVoRAN" if is_eb else "BVoRAN"
    
    # Determine features from name parts
    parts = []
    if has_knit: parts.append("Knit")
    if has_sr: parts.append("SR")
    if has_afa: parts.append("AFA")
    if has_eva: parts.append("EVA")
    parts.append(base_name)
    if has_ga: parts.append("GA")
    
    class_name = "Gen" + "".join(parts)
    
    # Class definition
    code = f"\n\nclass {class_name}(LowRankAdapter):\n"
    
    # __init__
    code += f"    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):\n"
    code += f"        super().__init__(in_features, out_features, config)\n"
    
    # Parameters - forward path
    code += f"        self.lora_A_fwd = nn.Parameter(torch.empty(config.r, in_features))\n"
    code += f"        self.lora_B_fwd = nn.Parameter(torch.empty(out_features, config.r))\n"
    
    if is_eb:
        code += f"        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))\n"
        code += f"        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))\n"
        code += f"        group_size = config.edora_group_size\n"
        code += f"        if group_size <= 0:\n"
        code += f"            group_size = max(1, out_features // 4)\n"
        code += f"        self.group_size = group_size\n"
        code += f"        n_groups_fwd = math.ceil(out_features / group_size)\n"
        code += f"        n_groups_bwd = math.ceil(in_features / group_size)\n"
        code += f"        self.magnitude_fwd = nn.Parameter(torch.empty(n_groups_fwd))\n"
        code += f"        self.magnitude_bwd = nn.Parameter(torch.empty(n_groups_bwd))\n"
    else:
        code += f"        self.magnitude_fwd = nn.Parameter(torch.empty(out_features))\n"
        code += f"        self.lora_A_bwd = nn.Parameter(torch.empty(config.r, out_features))\n"
        code += f"        self.lora_B_bwd = nn.Parameter(torch.empty(in_features, config.r))\n"
        code += f"        self.magnitude_bwd = nn.Parameter(torch.empty(in_features))\n"
    
    # AFA params
    if has_afa:
        code += f"        self.anneal_rate = config.extra_kwargs.get(\"afa_anneal_rate\", 0.05)\n"
        code += f"        self.anneal_start = config.extra_kwargs.get(\"afa_anneal_start\", 0)\n"
        code += f"        self.register_buffer(\"step\", torch.tensor(0, dtype=torch.long))\n"
    
    # SR params
    if has_sr:
        code += f"        self.register_buffer(\"importance_fwd\", torch.ones(config.r))\n"
        code += f"        self.register_buffer(\"importance_bwd\", torch.ones(config.r))\n"
    
    # EVA params
    if has_eva:
        code += f"        self.register_buffer(\"activation_buffer\", torch.empty(0, in_features))\n"
        code += f"        self.buffer_size = config.extra_kwargs.get(\"eva_buffer_size\", 128)\n"
        code += f"        self._buffer_filled = 0\n"
    
    # Knit params (VE)
    if has_knit:
        code += f"        ve_rank = config.dvora_ve_rank\n"
        code += f"        if ve_rank <= 0:\n"
        code += f"            ve_rank = max(1, config.r // 2)\n"
        code += f"        self.ve_rank = ve_rank\n"
        code += f"        self.register_buffer(\"ve_A_fwd\", torch.randn(ve_rank, in_features) / math.sqrt(in_features))\n"
        code += f"        self.register_buffer(\"ve_B_fwd\", torch.randn(out_features, ve_rank) / math.sqrt(ve_rank))\n"
        code += f"        self.ve_lambda_fwd = nn.Parameter(torch.empty(ve_rank))\n"
        code += f"        self.register_buffer(\"ve_A_bwd\", torch.randn(ve_rank, out_features) / math.sqrt(out_features))\n"
        code += f"        self.register_buffer(\"ve_B_bwd\", torch.randn(in_features, ve_rank) / math.sqrt(ve_rank))\n"
        code += f"        self.ve_lambda_bwd = nn.Parameter(torch.empty(ve_rank))\n"
    
    # Common params
    code += f"        self.norm = nn.LayerNorm(out_features)\n"
    code += f"        self.eps = config.dora_eps\n"
    
    if has_knit:
        code += f"        self.module_path = \"\"\n"
        code += f"        self.knit_mode = config.extra_kwargs.get(\"knit_mode\", \"in_dense\")\n"
    
    if has_sr:
        code += f"        self.sr_step = config.extra_kwargs.get(\"sr_recompose_step\", 50)\n"
        code += f"        self._step_counter = 0\n"
    
    if has_ga:
        code += f"        self._ga_initialized = False\n"
    
    code += f"        self.reset_parameters()\n\n"
    
    # reset_parameters
    code += f"    def reset_parameters(self):\n"
    if has_knit:
        code += f"        for A in [self.lora_A_fwd, self.lora_A_bwd]:\n"
        code += f"            nn.init.kaiming_uniform_(A, a=math.sqrt(5))\n"
        code += f"        for B in [self.lora_B_fwd, self.lora_B_bwd]:\n"
        code += f"            nn.init.zeros_(B)\n"
    else:
        code += f"        nn.init.kaiming_uniform_(self.lora_A_fwd, a=math.sqrt(5))\n"
        code += f"        nn.init.zeros_(self.lora_B_fwd)\n"
        code += f"        nn.init.kaiming_uniform_(self.lora_A_bwd, a=math.sqrt(5))\n"
        code += f"        nn.init.zeros_(self.lora_B_bwd)\n"
    code += f"        nn.init.ones_(self.magnitude_fwd)\n"
    code += f"        nn.init.ones_(self.magnitude_bwd)\n"
    if has_knit:
        code += f"        for v in [self.ve_lambda_fwd, self.ve_lambda_bwd]:\n"
        code += f"            nn.init.normal_(v, mean=0.0, std=0.01)\n"
    code += "\n"
    
    # trainable_params
    code += f"    @property\n"
    code += f"    def trainable_params(self) -> int:\n"
    if is_eb:
        n = ("self.lora_A_fwd.numel() + self.lora_B_fwd.numel()\n"
             "                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel()\n"
             "                + self.magnitude_fwd.numel() + self.magnitude_bwd.numel()")
    else:
        n = ("self.lora_A_fwd.numel() + self.lora_B_fwd.numel() + self.magnitude_fwd.numel()\n"
             "                + self.lora_A_bwd.numel() + self.lora_B_bwd.numel() + self.magnitude_bwd.numel()")
    if has_knit:
        n += "\n                + self.ve_lambda_fwd.numel() + self.ve_lambda_bwd.numel()"
    n += "\n                + sum(p.numel() for p in self.norm.parameters())"
    code += f"        return ({n})\n\n"
    
    # AFA methods
    if has_afa:
        code += f"    def _get_alpha(self) -> float:\n"
        code += f"        t = max(0, self.step.item() - self.anneal_start)\n"
        code += f"        return math.exp(-self.anneal_rate * t)\n\n"
        code += f"    def _apply_activation(self, x: torch.Tensor, alpha: float) -> torch.Tensor:\n"
        code += f"        return (1 - alpha) * x + alpha * torch.tanh(x)\n\n"
    
    # EVA methods
    if has_eva:
        code += f"    def collect_activation(self, x: torch.Tensor):\n"
        code += f"        with torch.no_grad():\n"
        code += f"            x_flat = x.view(-1, self.in_features)\n"
        code += f"            if self._buffer_filled == 0:\n"
        code += f"                self.activation_buffer = x_flat[:self.buffer_size].clone()\n"
        code += f"                self._buffer_filled = x_flat.shape[0]\n"
        code += f"            elif self._buffer_filled < self.buffer_size:\n"
        code += f"                remaining = self.buffer_size - self._buffer_filled\n"
        code += f"                take = min(remaining, x_flat.shape[0])\n"
        code += f"                self.activation_buffer = torch.cat(\n"
        code += f"                    [self.activation_buffer[:self._buffer_filled], x_flat[:take]]\n"
        code += f"                )\n"
        code += f"                self._buffer_filled += take\n\n"
        code += f"    def eva_init_from_buffer(self):\n"
        code += f"        if self._buffer_filled < 2:\n"
        code += f"            return\n"
        code += f"        with torch.no_grad():\n"
        code += f"            buf = self.activation_buffer[:self._buffer_filled]\n"
        code += f"            x_centered = buf - buf.mean(dim=0, keepdim=True)\n"
        code += f"            cov = (x_centered.T @ x_centered) / (self._buffer_filled - 1)\n"
        code += f"            _, _, Vh = torch.linalg.svd(cov.float(), full_matrices=False)\n"
        code += f"            self.lora_A_fwd.data = Vh[:self.config.r].to(self.lora_A_fwd.dtype)\n"
        code += f"            nn.init.zeros_(self.lora_B_fwd)\n"
        code += f"            y_centered = buf - buf.mean(dim=0, keepdim=True)\n"
        code += f"            cov_bwd = (y_centered.T @ y_centered) / (self._buffer_filled - 1)\n"
        code += f"            _, _, Vh_b = torch.linalg.svd(cov_bwd.float(), full_matrices=False)\n"
        code += f"            self.lora_A_bwd.data = Vh_b[:self.config.r].to(self.lora_A_bwd.dtype)\n"
        code += f"            nn.init.zeros_(self.lora_B_bwd)\n"
        code += f"            self._buffer_filled = 0\n\n"
    
    # GA method
    if has_ga:
        code += f"    def gradient_align_init(self, x: torch.Tensor, grad_output: torch.Tensor):\n"
        code += f"        with torch.no_grad():\n"
        code += f"            G_fwd = grad_output.T @ x\n"
        code += f"            U_fwd, S_fwd, Vh_fwd = torch.linalg.svd(G_fwd.float(), full_matrices=False)\n"
        code += f"            scale_fwd = (S_fwd[:self.config.r].mean() / self.config.r) ** 0.5\n"
        code += f"            self.lora_A_fwd.data = (Vh_fwd[:self.config.r] * scale_fwd).to(self.lora_A_fwd.dtype)\n"
        code += f"            self.lora_B_fwd.data = (U_fwd[:, :self.config.r] * scale_fwd).to(self.lora_B_fwd.dtype)\n"
        code += f"            G_bwd = x.T @ grad_output\n"
        code += f"            U_bwd, S_bwd, Vh_bwd = torch.linalg.svd(G_bwd.float(), full_matrices=False)\n"
        code += f"            scale_bwd = (S_bwd[:self.config.r].mean() / self.config.r) ** 0.5\n"
        code += f"            self.lora_A_bwd.data = (Vh_bwd[:self.config.r] * scale_bwd).to(self.lora_A_bwd.dtype)\n"
        code += f"            self.lora_B_bwd.data = (U_bwd[:, :self.config.r] * scale_bwd).to(self.lora_B_bwd.dtype)\n\n"
    
    # SR methods
    if has_sr:
        code += f"    def update_importance(self, x: torch.Tensor):\n"
        code += f"        with torch.no_grad():\n"
        if has_afa:
            code += f"            act = self._get_alpha()\n"
            code += f"            BA = (self.lora_B_fwd @ self._apply_activation(self.lora_A_fwd, act)) * self.scaling\n"
        else:
            code += f"            BA = (self.lora_B_fwd @ self.lora_A_fwd) * self.scaling\n"
        code += f"            grad = torch.autograd.grad(BA.pow(2).sum(), self.lora_A_fwd, retain_graph=True)[0]\n"
        code += f"            self.importance_fwd.copy_(grad.norm(p=2, dim=1))\n"
        if has_afa:
            code += f"            BA_bwd = (self.lora_B_bwd @ self._apply_activation(self.lora_A_bwd, act)) * self.scaling\n"
        else:
            code += f"            BA_bwd = (self.lora_B_bwd @ self.lora_A_bwd) * self.scaling\n"
        code += f"            grad_bwd = torch.autograd.grad(BA_bwd.pow(2).sum(), self.lora_A_bwd, retain_graph=True)[0]\n"
        code += f"            self.importance_bwd.copy_(grad_bwd.norm(p=2, dim=1))\n\n"
        
        code += f"    def recompose(self, base_weight: torch.Tensor):\n"
        code += f"        with torch.no_grad():\n"
        code += f"            device = self.lora_A_fwd.device\n"
        if has_afa:
            code += f"            alpha = self._get_alpha()\n"
            a_apply = lambda x: f"self._apply_activation({x}, alpha)"
        else:
            a_apply = lambda x: x
        code += f"            W_eff_fwd = base_weight + (self.lora_B_fwd @ {a_apply('self.lora_A_fwd')}) * self.scaling\n"
        code += f"            sorted_idx = torch.argsort(self.importance_fwd, descending=True)\n"
        code += f"            keep_r = max(1, torch.sum(self.importance_fwd > 0.1 * self.importance_fwd.max()).int().item())\n"
        code += f"            keep_r = min(keep_r, self.config.r)\n"
        code += f"            keep_idx = sorted_idx[:keep_r]\n"
        code += f"            fuse_idx = sorted_idx[keep_r:]\n"
        code += f"            if len(fuse_idx) > 0:\n"
        code += f"                delta_fuse = (self.lora_B_fwd[:, fuse_idx] @ {a_apply('self.lora_A_fwd[fuse_idx]')}) * self.scaling\n"
        code += f"                W_eff_fwd = W_eff_fwd - delta_fuse\n"
        code += f"            U, S, Vh = torch.linalg.svd(W_eff_fwd.float(), full_matrices=False)\n"
        code += f"            A_new = Vh[:self.config.r].contiguous().to(device)\n"
        code += f"            B_new = (U[:, :self.config.r] * S[:self.config.r, None]).contiguous().to(device)\n"
        code += f"            self.lora_A_fwd.data = A_new.to(self.lora_A_fwd.dtype)\n"
        code += f"            self.lora_B_fwd.data = B_new.to(self.lora_B_fwd.dtype)\n"
        code += f"            self.importance_fwd = torch.ones(self.config.r, device=device)\n"
        code += f"            W_eff_bwd = base_weight.T + (self.lora_B_bwd @ {a_apply('self.lora_A_bwd')}) * self.scaling\n"
        code += f"            sorted_idx_bwd = torch.argsort(self.importance_bwd, descending=True)\n"
        code += f"            keep_r_bwd = max(1, torch.sum(self.importance_bwd > 0.1 * self.importance_bwd.max()).int().item())\n"
        code += f"            keep_r_bwd = min(keep_r_bwd, self.config.r)\n"
        code += f"            keep_idx_bwd = sorted_idx_bwd[:keep_r_bwd]\n"
        code += f"            fuse_idx_bwd = sorted_idx_bwd[keep_r_bwd:]\n"
        code += f"            if len(fuse_idx_bwd) > 0:\n"
        code += f"                delta_fuse_bwd = (self.lora_B_bwd[:, fuse_idx_bwd] @ {a_apply('self.lora_A_bwd[fuse_idx_bwd]')}) * self.scaling\n"
        code += f"                W_eff_bwd = W_eff_bwd - delta_fuse_bwd\n"
        code += f"            U_b, S_b, Vh_b = torch.linalg.svd(W_eff_bwd.float(), full_matrices=False)\n"
        code += f"            A_new_b = Vh_b[:self.config.r].contiguous().to(device)\n"
        code += f"            B_new_b = (U_b[:, :self.config.r] * S_b[:self.config.r, None]).contiguous().to(device)\n"
        code += f"            self.lora_A_bwd.data = A_new_b.to(self.lora_A_bwd.dtype)\n"
        code += f"            self.lora_B_bwd.data = B_new_b.to(self.lora_B_bwd.dtype)\n"
        code += f"            self.importance_bwd = torch.ones(self.config.r, device=device)\n\n"
    
    # forward
    code += f"    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:\n"
    code += f"        base_weight = base_weight.to(x.dtype)\n"
    
    if has_knit:
        code += f"        stack = _get_knit_stack(self.module_path)\n"
        code += f"        if self.knit_mode == \"in_dense\":\n"
        code += f"            knit_x = x\n"
        code += f"            for prev_x in stack:\n"
        code += f"                knit_x = knit_x + prev_x.detach()\n"
        code += f"            stack.append(x.detach())\n"
        code += f"        else:\n"
        code += f"            knit_x = x\n"
        input_var = "knit_x"
    else:
        input_var = "x"
    
    if has_afa:
        code += f"        alpha = self._get_alpha()\n"
        ba = lambda x: f"self._apply_activation({x}, alpha)"
    else:
        ba = lambda x: x
    
    # BA computation
    code += f"        BA_fwd = (self.lora_B_fwd @ {ba('self.lora_A_fwd')}) * self.scaling\n"
    if has_knit:
        code += f"        ve_fwd = self.ve_B_fwd @ (self.ve_lambda_fwd[:, None] * self.ve_A_fwd)\n"
        code += f"        merged_fwd = base_weight + BA_fwd + ve_fwd\n"
    else:
        code += f"        merged_fwd = base_weight + BA_fwd\n"
    
    code += f"        col_norm_fwd = merged_fwd.norm(p=2, dim=1, keepdim=True)\n"
    code += f"        normalized_fwd = merged_fwd / (col_norm_fwd + self.eps)\n"
    
    if is_eb:
        code += f"        mag_fwd = self.magnitude_fwd.repeat_interleave(self.group_size)[:self.out_features, None]\n"
        code += f"        adapted_fwd = mag_fwd * normalized_fwd\n"
    else:
        code += f"        adapted_fwd = self.magnitude_fwd[:, None] * normalized_fwd\n"
    
    code += f"        fwd_delta = adapted_fwd - base_weight\n"
    
    code += f"        BA_bwd = (self.lora_B_bwd @ {ba('self.lora_A_bwd')}) * self.scaling\n"
    if has_knit:
        code += f"        ve_bwd = self.ve_B_bwd @ (self.ve_lambda_bwd[:, None] * self.ve_A_bwd)\n"
        code += f"        merged_bwd = base_weight.T + BA_bwd + ve_bwd\n"
    else:
        code += f"        merged_bwd = base_weight.T + BA_bwd\n"
    
    code += f"        col_norm_bwd = merged_bwd.norm(p=2, dim=1, keepdim=True)\n"
    code += f"        normalized_bwd = merged_bwd / (col_norm_bwd + self.eps)\n"
    
    if is_eb:
        code += f"        mag_bwd = self.magnitude_bwd.repeat_interleave(self.group_size)[:self.in_features, None]\n"
        code += f"        adapted_bwd = mag_bwd * normalized_bwd\n"
    else:
        code += f"        adapted_bwd = self.magnitude_bwd[:, None] * normalized_bwd\n"
    
    code += f"        bwd_delta = adapted_bwd - base_weight.T\n"
    code += f"        out = F.linear({input_var}, fwd_delta) + F.linear({input_var}, bwd_delta.T)\n"
    code += f"        out = self.norm(out)\n"
    
    if has_afa:
        code += f"        self.step += 1\n"
    if has_sr:
        code += f"        self._step_counter += 1\n"
    
    code += f"        return out\n\n\n"
    
    return class_name, code

# Generate all 64 combinatoric variants (2^6)
variants = []
for is_eb in [False, True]:
    for has_knit in [False, True]:
        for has_sr in [False, True]:
            for has_afa in [False, True]:
                for has_eva in [False, True]:
                    for has_ga in [False, True]:
                        parts = []
                        if has_knit: parts.append("knit")
                        if has_sr: parts.append("sr")
                        if has_afa: parts.append("afa")
                        if has_eva: parts.append("eva")
                        parts.append("ebvoran" if is_eb else "bvoran")
                        if has_ga: parts.append("ga")
                        suffix = "_".join(parts)
                        variants.append((suffix, is_eb, has_afa, has_ga, has_sr, has_knit, has_eva))

out_path = "/home/filip/Projects/Personal/AI/RankAdaptation/src/adapters/combo_adapters.py"
with open(out_path, "w") as f:
    f.write('"""Combinatoric hybrid adapters combining techniques from multiple papers."""\n\n')
    f.write("from __future__ import annotations\n\n")
    f.write("import math\n\n")
    f.write("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n")
    f.write("from .base import LowRankAdapter, AdapterConfig\n")
    f.write("from .knit_bvoran import _get_knit_stack\n\n")
    for idx, (suffix, is_eb, has_afa, has_ga, has_sr, has_knit, has_eva) in enumerate(variants):
        class_name, code = cls(suffix, is_eb, has_afa, has_ga, has_sr, has_knit, has_eva)
        f.write(code)
        print(f"  [{idx+1}/{len(variants)}] Generated {class_name}")

print(f"\nWritten {len(variants)} classes to {out_path}")
print(f"File size: {os.path.getsize(out_path):,} bytes")
