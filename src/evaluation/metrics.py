from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def measure_inference_speed(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    num_trials: int = 20,
    num_warmup: int = 5,
    max_new_tokens: int = 20,
    device: str = "cuda",
) -> dict[str, float]:
    model.eval()
    model.to(device)

    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    tokens_generated = 0

    for _ in range(num_trials):
        start.record()
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        times.append(elapsed)
        tokens_generated += out.shape[1] - input_ids.shape[1]

    avg_time = sum(times) / len(times)
    tokens_per_sec = tokens_generated / (avg_time / 1000) / num_trials

    return {
        "avg_latency_ms": avg_time,
        "tokens_per_second": tokens_per_sec,
        "std_latency_ms": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5,
    }


def measure_memory(model: torch.nn.Module, device: str = "cuda") -> dict[str, float]:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 ** 2)
        return {
            "param_memory_mb": param_memory,
            "buffer_memory_mb": buffer_memory,
            "total_model_memory_mb": param_memory + buffer_memory,
        }
    return {}


def compute_frobenius_norm(weight: torch.Tensor, delta: torch.Tensor) -> float:
    return (delta.norm(p="fro") / weight.norm(p="fro")).item()


def compute_effective_rank(weight: torch.Tensor, threshold: float = 0.99) -> int:
    _, S, _ = torch.svd(weight.float())
    cumsum = S.cumsum(0)
    threshold_val = cumsum[-1] * threshold
    return int((cumsum < threshold_val).sum().item()) + 1
