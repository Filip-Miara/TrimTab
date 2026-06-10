"""Flow matching over adapter lifecycle trajectories.

Represents adapter configurations as points in a continuous 7D flag space.
Lifecycles are trajectories through this space. A velocity field v_θ(f, t, ctx)
is trained via flow matching to predict the optimal direction of change.

At inference: start from an initial configuration, integrate the velocity
field to generate a full lifecycle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter_evolution import AdapterState, FLAG_NAMES, N_FLAGS


class LifecycleFlow(nn.Module):
    """Velocity field for adapter configuration flow.

    v_θ: (flag_7D, t, context) → δflag (direction of optimal change)

    Conditioned on adapter history via a small transformer encoder.
    The velocity field defines how to move through architectural space.
    """

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 3, max_history: int = 5):
        super().__init__()
        self.d_model = d_model
        self.max_history = max_history

        self.input_dim = N_FLAGS + 1  # flags + time t
        self.input_proj = nn.Linear(self.input_dim, d_model)

        ctx_dim = AdapterState.vec_dim(max_history)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(ctx_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.flow_net = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, N_FLAGS),
        )

    def forward(
        self,
        flags: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity δflags at current position.

        Args:
            flags: (B, N_FLAGS) current soft flag values in [0, 1]
            t: (B, 1) time in [0, 1] (normalized segment progress)
            context: (B, ctx_dim) optional adapter history encoding
        Returns:
            velocity: (B, N_FLAGS) predicted direction of change
        """
        inp = torch.cat([flags, t], dim=-1)
        x = self.input_proj(inp)

        if context is not None:
            ctx = self.ctx_encoder(context)
            x = torch.cat([x, ctx], dim=-1)
        else:
            x = torch.cat([x, torch.zeros_like(x)], dim=-1)

        v = self.flow_net(x)
        return torch.tanh(v) * 0.5  # bound velocity to [-0.5, 0.5] per step


class FlowMatchingTrainer:
    """Trains LifecycleFlow via flow matching on observed trajectories.

    Loss: MSE(v_θ(f_t, t, ctx), f_{t+1} - f_t) over all observed transitions.
    Optionally also predict the eval PPL delta (auxiliary loss).
    """

    def __init__(self, flow_model: LifecycleFlow, lr: float = 1e-3, device: str = "cpu"):
        self.flow = flow_model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.flow.parameters(), lr=lr)
        self.loss_history: list[float] = []

    def train_step(
        self,
        trajectories: list[list[dict[str, Any]]],
        histories: list[list[AdapterState]],
    ) -> float:
        """One training step over a batch of observed trajectories.

        trajectories: list of per-segment dicts with 'flags' keys
        histories: list of AdapterState lists (one per segment)
        """
        total_loss = torch.tensor(0.0, device=self.device)
        n_transitions = 0

        for traj, hist in zip(trajectories, histories):
            for seg_idx in range(len(traj) - 1):
                curr = traj[seg_idx]
                next_ = traj[seg_idx + 1]

                f_t = torch.tensor(
                    [float(curr["flags"].get(k, 0.0)) for k in FLAG_NAMES],
                    device=self.device,
                ).unsqueeze(0)

                f_next = torch.tensor(
                    [float(next_["flags"].get(k, 0.0)) for k in FLAG_NAMES],
                    device=self.device,
                ).unsqueeze(0)

                target_velocity = (f_next - f_t).detach()

                t_val = seg_idx / max(len(traj) - 1, 1)
                t = torch.tensor([[t_val]], device=self.device)

                ctx_vec = None
                if hist and seg_idx < len(hist):
                    ctx_vec = hist[seg_idx].to_vector().unsqueeze(0).to(self.device)

                pred_v = self.flow(f_t, t, ctx_vec)
                loss = F.mse_loss(pred_v, target_velocity)
                total_loss = total_loss + loss
                n_transitions += 1

        if n_transitions == 0:
            return 0.0

        total_loss = total_loss / n_transitions
        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
        self.opt.step()

        loss_val = total_loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def generate_lifecycle(
        self,
        n_segments: int,
        steps_per_segment: int,
        start_flags: dict[str, float] | None = None,
        history: list[AdapterState] | None = None,
        temperature: float = 1.0,
    ) -> list[dict[str, float]]:
        """Generate a lifecycle by integrating the velocity field.

        Starts from start_flags, applies v_θ at each step to morph.
        """
        self.flow.eval()

        if start_flags is None:
            flags = {k: 0.0 for k in FLAG_NAMES}
            flags["use_polynomial"] = 1.0
        else:
            flags = start_flags.copy()

        trajectory = [flags.copy()]

        with torch.no_grad():
            for seg_idx in range(n_segments):
                for step in range(steps_per_segment):
                    f_t = torch.tensor(
                        [flags.get(k, 0.0) for k in FLAG_NAMES],
                        device=self.device,
                    ).unsqueeze(0)

                    progress = step / max(steps_per_segment - 1, 1)
                    t = torch.tensor([[progress]], device=self.device)

                    ctx_vec = None
                    if history and seg_idx < len(history):
                        ctx_vec = history[seg_idx].to_vector().unsqueeze(0).to(self.device)

                    v = self.flow(f_t, t, ctx_vec)
                    delta = v.squeeze(0).cpu().tolist()

                    for i, k in enumerate(FLAG_NAMES):
                        flags[k] = max(0.0, min(1.0, flags[k] + delta[i] / steps_per_segment))

                trajectory.append(flags.copy())

        return trajectory


@dataclass
class FlowLifecycleConfig:
    """Configuration for flow-based lifecycle generation."""
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    lr: float = 1e-3
    train_steps: int = 100
    temperature: float = 1.0
