from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
from transformers import PreTrainedModel

from .metrics import count_parameters, measure_inference_speed, measure_memory


@dataclass
class BenchmarkResult:
    variant_name: str
    config: dict[str, Any]
    trainable_params: int
    total_params: int
    param_efficiency: float
    training_loss: float
    training_perplexity: float
    final_loss: float
    final_perplexity: float
    memory_peak_mb: float
    memory_allocated_mb: float
    total_model_memory_mb: float
    avg_latency_ms: float
    tokens_per_second: float
    grad_norm_avg: float
    step_time_avg_ms: float
    total_training_steps: int
    frobenius_ratio: float = 0.0
    eval_loss: float = 0.0
    eval_perplexity: float = 0.0
    status: str = "completed"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchmarkResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ComparisonReport:
    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = ""

    def add(self, result: BenchmarkResult):
        for i, r in enumerate(self.results):
            if r.variant_name == result.variant_name:
                self.results[i] = result
                return
        self.results.append(result)

    def sort_by(self, metric: str, reverse: bool = False):
        self.results.sort(key=lambda r: getattr(r, metric, 0), reverse=reverse)

    def summary_table(self) -> str:
        header = f"{'Variant':<25} {'Params':>10} {'TrainLoss':>10} {'EvalLoss':>10} {'EvalPPL':>10} {'PeakMem':>10} {'Steps':>8}"
        sep = "-" * len(header)
        rows = [header, sep]
        for r in sorted(self.results, key=lambda x: x.eval_loss if x.eval_loss > 0 else 999):
            rows.append(
                f"{r.variant_name:<25} {r.trainable_params:>10,} {r.training_loss:>10.4f} {r.eval_loss:>10.4f} {r.eval_perplexity:>10.2f} {r.memory_peak_mb:>10.1f} {r.total_training_steps:>8}"
            )
        return "\n".join(rows)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump({"timestamp": self.timestamp, "results": [r.to_dict() for r in self.results]}, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ComparisonReport":
        with open(path) as f:
            data = json.load(f)
        report = cls(timestamp=data.get("timestamp", ""))
        report.results = [BenchmarkResult.from_dict(r) for r in data.get("results", [])]
        return report
