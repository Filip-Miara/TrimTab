"""Adapter evolution: meta-controller that learns adapter lifecycles.

A transformer reads the history of past adapter configurations and their
outcomes, then outputs the optimal configuration for the next segment.

Uses evolution strategies (ES) for training — maintains a population of
controller variants, evaluates them on segments, and breeds the fittest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


N_FLAGS = 7
FLAG_NAMES = [
    "bidirectional", "use_vectors", "use_norm", "use_gate",
    "use_activation", "use_autoencoder", "use_polynomial",
]


@dataclass
class AdapterState:
    """Immutable record of one adapter's lifecycle."""
    flags: dict[str, bool] = field(default_factory=dict)
    poly_order: int = 2
    loss_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    taylor_contribution: float = 0.0
    age_steps: int = 0
    eval_ppl_before: float = 0.0
    eval_ppl_after: float = 0.0
    improvement: float = 0.0

    def to_vector(self, max_history: int = 5) -> torch.Tensor:
        d = [float(self.flags.get(k, False)) for k in FLAG_NAMES]
        d.append(float(self.poly_order) / 4.0)
        hist = self.loss_history[-max_history:]
        d += hist + [0.0] * (max_history - len(hist))
        ghist = self.grad_norm_history[-max_history:]
        d += ghist + [0.0] * (max_history - len(ghist))
        d.append(float(self.taylor_contribution))
        d.append(float(self.age_steps) / 100.0)
        d.append(float(self.improvement) / 1000.0)
        return torch.tensor(d, dtype=torch.float32)

    @classmethod
    def vec_dim(cls, max_history: int = 5) -> int:
        return N_FLAGS + 1 + max_history * 2 + 3

    @classmethod
    def flags_from_vec(cls, vec: torch.Tensor, temperature: float = 1.0) -> dict[str, bool]:
        raw = vec[:N_FLAGS]
        probs = torch.sigmoid(raw / temperature)
        return {FLAG_NAMES[i]: bool((probs[i] > 0.5).item()) for i in range(N_FLAGS)}

    @classmethod
    def soft_flags_from_vec(cls, vec: torch.Tensor, temperature: float = 1.0) -> dict[str, float]:
        raw = vec[:N_FLAGS]
        probs = torch.sigmoid(raw / temperature)
        return {FLAG_NAMES[i]: float(probs[i].item()) for i in range(N_FLAGS)}


class MetaController(nn.Module):
    """Transformer that reads adapter history → outputs next config.

    Architecture:
        Embed each AdapterState → d_model
        Positional encoding (segment index)
        TransformerEncoder (self-attention over history)
        MLP decoder → flag logits + poly_order
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        max_history: int = 5,
        max_segments: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_history = max_history
        self.input_dim = AdapterState.vec_dim(max_history)

        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.pos_encoding = nn.Embedding(max_segments, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)

        self.output_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, N_FLAGS + 1),
        )

    def forward(
        self,
        history: list[AdapterState],
        return_attention: bool = False,
        temperature: float = 1.0,
    ) -> tuple[dict[str, float], int, torch.Tensor | None]:
        if not history:
            flags = {k: False for k in FLAG_NAMES}
            flags["use_polynomial"] = True
            return flags, 2, None

        vectors = [s.to_vector(self.max_history) for s in history]
        x = torch.stack(vectors).unsqueeze(0)

        x = self.input_proj(x)
        seg_ids = torch.arange(len(history), device=x.device).unsqueeze(0)
        x = x + self.pos_encoding(seg_ids)

        x = self.transformer(x)

        last = x[:, -1, :]
        logits = self.output_net(last).squeeze(0)

        flag_logits = logits[:N_FLAGS]
        poly_logit = logits[N_FLAGS:]

        raw_flags = AdapterState.soft_flags_from_vec(flag_logits, temperature)
        # Round to binary for the actual config
        flags = {k: v > 0.5 for k, v in raw_flags.items()}

        poly_order = max(1, min(4, int(torch.round(torch.sigmoid(poly_logit) * 3 + 1).item())))

        attn = None
        if return_attention:
            attn = x  # simplified; real attn weights need hook

        return flags, poly_order, attn


class AdapterEvolution:
    """Evolution-strategy training for MetaController.

    Maintains a population of controller parameter vectors.
    Each "individual" is evaluated by running it for N segments and
    measuring cumulative improvement. Best individuals breed via
    crossover + mutation.
    """

    def __init__(
        self,
        controller: MetaController,
        pop_size: int = 16,
        elite_frac: float = 0.25,
        mutation_std: float = 0.1,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.controller = controller.to(device)
        self.pop_size = pop_size
        self.elite_count = max(2, int(pop_size * elite_frac))
        self.mutation_std = mutation_std
        self.lr = lr
        self.device = device

        base_params = torch.cat([p.data.flatten() for p in controller.parameters()])
        self.base = base_params.clone()
        self.population = torch.randn(pop_size, base_params.shape[0], device=device) * 0.01
        self.fitness = torch.full((pop_size,), -float("inf"), device=device)

    def _set_params(self, params_vec: torch.Tensor):
        idx = 0
        for p in self.controller.parameters():
            n = p.numel()
            p.data.copy_(params_vec[idx:idx + n].reshape(p.shape))
            idx += n

    def _get_params(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in self.controller.parameters()])

    def suggest_config(self, history: list[AdapterState], temperature: float = 1.0) -> tuple[dict[str, bool], int]:
        flags, poly, _ = self.controller(history, temperature=temperature)
        return flags, poly

    def record_fitness(self, individual_idx: int, fitness: float):
        self.fitness[individual_idx] = fitness

    def evolve(self):
        """Breed top-k individuals via crossover + mutation."""
        sorted_idx = torch.argsort(self.fitness, descending=True)
        elite_idx = sorted_idx[:self.elite_count]

        elite_params = self.population[elite_idx]
        elite_fitness = self.fitness[elite_idx]

        best_idx = elite_idx[0]
        best_params = self.population[best_idx].clone()
        best_fitness = self.fitness[best_idx].item()

        new_pop = []
        for i in range(self.pop_size):
            if i < self.elite_count:
                new_pop.append(elite_params[i].clone())
            else:
                p1, p2 = elite_params[torch.randint(0, self.elite_count, (2,))]
                mask = torch.rand_like(p1) > 0.5
                child = torch.where(mask, p1, p2)
                child = child + torch.randn_like(child) * self.mutation_std
                new_pop.append(child)

        self.population = torch.stack(new_pop)
        self.fitness = torch.full((self.pop_size,), -float("inf"), device=self.device)

        return best_fitness, best_params

    def deploy_best(self):
        """Load best-found parameters into controller."""
        best_idx = torch.argmax(self.fitness)
        self._set_params(self.population[best_idx])


@dataclass
class LifecycleConfig:
    """Full config for how to generate adapter lifecycles."""
    use_metacontroller: bool = False
    meta_d_model: int = 64
    meta_nhead: int = 4
    meta_layers: int = 3
    meta_pop_size: int = 16
    meta_elite_frac: float = 0.25
    meta_mutation_std: float = 0.1
    meta_lr: float = 1e-3
    meta_temperature: float = 1.0
    meta_evolve_every: int = 5
    # Fallback schedule if not using metacontroller
    schedule_type: str = "constant"
    # "constant": keep same flags throughout
    # "linear": linearly interpolate from start to end flags
    # "threshold": switch based on PPL improvement rate
    start_flags: dict[str, bool] = field(default_factory=lambda: {"use_polynomial": True})
    end_flags: dict[str, bool] = field(default_factory=lambda: {
        "bidirectional": True, "use_vectors": True, "use_norm": True,
    })
    switch_segment: int = 3
