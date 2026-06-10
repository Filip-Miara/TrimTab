from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel


@dataclass
class TrainMetrics:
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    step_times: list[float] = field(default_factory=list)
    total_trainable: int = 0
    total_params: int = 0
    memory_peak_mb: float = 0.0
    memory_allocated_mb: float = 0.0


def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        adapter_modules: list[nn.Module],
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        num_epochs: int = 1,
        max_steps: int | None = None,
        log_interval: int = 10,
        eval_interval: int = 50,
        device: str = "cuda",
        warmup_steps: int = 0,
    ):
        self.model = model
        self.adapter_modules = adapter_modules
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.device = device
        self.warmup_steps = warmup_steps
        self.metrics = TrainMetrics()

        trainable_params = [p for m in adapter_modules for p in m.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.metrics.total_params = total
        self.metrics.total_trainable = trainable

    def train(self) -> TrainMetrics:
        self.model.train()
        self.model.to(self.device)

        global_step = 0
        best_eval_loss = float("inf")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.metrics.memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024

        for epoch in range(self.num_epochs):
            for batch in self.train_loader:
                if self.max_steps is not None and global_step >= self.max_steps:
                    break

                t0 = time.time()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if self.warmup_steps > 0 and global_step < self.warmup_steps:
                    lr_scale = min(1.0, (global_step + 1) / self.warmup_steps)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = pg["lr"] * lr_scale

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()

                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for m in self.adapter_modules for p in m.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )
                else:
                    grad_norm = torch.tensor(0.0)

                self.optimizer.step()

                step_time = time.time() - t0
                current_lr = self.optimizer.param_groups[0]["lr"]

                self.metrics.losses.append(loss.item())
                self.metrics.grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                self.metrics.learning_rates.append(current_lr)
                self.metrics.step_times.append(step_time)

                if global_step % self.log_interval == 0:
                    ppl = compute_perplexity(loss.item())
                    print(f"  Step {global_step:>5} | Loss {loss.item():.4f} | PPL {ppl:.2f} | GradNorm {grad_norm:.4f} | LR {current_lr:.2e} | {step_time*1000:.0f}ms/step")

                global_step += 1

            if self.max_steps is not None and global_step >= self.max_steps:
                break

        if torch.cuda.is_available():
            self.metrics.memory_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        return self.metrics
