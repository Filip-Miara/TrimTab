#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.knit_bvoran import clear_knit_registry
from src.adapters import (
    AdapterConfig, AdapterWrappedLinear, LowRankAdapter, adapt_linear_layer,
    DoRA, BoRA, EDoRA, DVoRA, DoRAN, QDoRA, QADoRA,
    EDoRAN, EBoRA, EBoRAN, BVoRA, DiagLoRAN, MultiAngleLoRAN, CycledBoRAN,
    CycledAxialBoRA, CycledDiagLoRA, PlainLoRA,
    BVoRAN, EBVoRAN,
    SeBVoRAN, ESeBVoRAN,
    KnitBVoRAN, KnitEBVoRAN,
    SRBVoRAN, ESRBVoRAN,
    BVoRANGA, EBVoRANGA,
    EVABVoRAN, EVAEBVoRAN,
    BPVERAN, EBPVERAN,
    BVAuroRAN, EBVAuroRAN,
    StreamFusionLoRA,
    AFABVoRAN, AFAEBVoRAN,
    GenBVoRAN,
    GenBVoRANGA,
    GenEVABVoRAN,
    GenEVABVoRANGA,
    GenAFABVoRAN,
    GenAFABVoRANGA,
    GenAFAEVABVoRAN,
    GenAFAEVABVoRANGA,
    GenSRBVoRAN,
    GenSRBVoRANGA,
    GenSREVABVoRAN,
    GenSREVABVoRANGA,
    GenSRAFABVoRAN,
    GenSRAFABVoRANGA,
    GenSRAFAEVABVoRAN,
    GenSRAFAEVABVoRANGA,
    GenKnitBVoRAN,
    GenKnitBVoRANGA,
    GenKnitEVABVoRAN,
    GenKnitEVABVoRANGA,
    GenKnitAFABVoRAN,
    GenKnitAFABVoRANGA,
    GenKnitAFAEVABVoRAN,
    GenKnitAFAEVABVoRANGA,
    GenKnitSRBVoRAN,
    GenKnitSRBVoRANGA,
    GenKnitSREVABVoRAN,
    GenKnitSREVABVoRANGA,
    GenKnitSRAFABVoRAN,
    GenKnitSRAFABVoRANGA,
    GenKnitSRAFAEVABVoRAN,
    GenKnitSRAFAEVABVoRANGA,
    GenEBVoRAN,
    GenEBVoRANGA,
    GenEVAEBVoRAN,
    GenEVAEBVoRANGA,
    GenAFAEBVoRAN,
    GenAFAEBVoRANGA,
    GenAFAEVAEBVoRAN,
    GenAFAEVAEBVoRANGA,
    GenSREBVoRAN,
    GenSREBVoRANGA,
    GenSREVAEBVoRAN,
    GenSREVAEBVoRANGA,
    GenSRAFAEBVoRAN,
    GenSRAFAEBVoRANGA,
    GenSRAFAEVAEBVoRAN,
    GenSRAFAEVAEBVoRANGA,
    GenKnitEBVoRAN,
    GenKnitEBVoRANGA,
    GenKnitEVAEBVoRAN,
    GenKnitEVAEBVoRANGA,
    GenKnitAFAEBVoRAN,
    GenKnitAFAEBVoRANGA,
    GenKnitAFAEVAEBVoRAN,
    GenKnitAFAEVAEBVoRANGA,
    GenKnitSREBVoRAN,
    GenKnitSREBVoRANGA,
    GenKnitSREVAEBVoRAN,
    GenKnitSREVAEBVoRANGA,
    GenKnitSRAFAEBVoRAN,
    GenKnitSRAFAEBVoRANGA,
    GenKnitSRAFAEVAEBVoRAN,
    GenKnitSRAFAEVAEBVoRANGA,
)
from src.training.trainer import Trainer, compute_perplexity
from src.evaluation.benchmark import BenchmarkResult, ComparisonReport
from src.evaluation.metrics import measure_inference_speed, measure_memory, count_parameters


# ── Model ──────────────────────────────────────────────────────────
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

# Fallback: load from external HDD if not on SSD
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"

# ── Data ───────────────────────────────────────────────────────────
ARXIV_PARQUET_DIR = "/run/media/filip/B522-875D/Datasets/hub/datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section"

NUM_TRAIN = 1000
NUM_EVAL = 200


def load_arxiv_data(num_train: int = NUM_TRAIN, num_eval: int = NUM_EVAL):
    """Load arxiv articles as train/eval text lists (streaming, stops early)."""
    parq_files = sorted(
        os.path.join(ARXIV_PARQUET_DIR, f)
        for f in os.listdir(ARXIV_PARQUET_DIR)
        if f.endswith(".parquet")
    )
    needed = num_train + num_eval
    all_texts = []
    for pf in parq_files:
        tbl = pq.read_table(pf, columns=["article"])
        for art in tbl.column("article"):
            t = art.as_py()
            if len(t) > 500:
                all_texts.append(t)
                if len(all_texts) >= needed:
                    return all_texts[:num_train], all_texts[num_train:num_train + num_eval]
    train_texts = all_texts[:num_train]
    eval_texts = all_texts[num_train : num_train + num_eval]
    return train_texts, eval_texts


# ── Adapter variants ────────────────────────────────────────────────
# All 64 Gen variants (full flag grid)
ADAPTER_VARIANTS: dict[str, type[LowRankAdapter]] = {
    "gen_bvoran": GenBVoRAN,
    "gen_bvoran_ga": GenBVoRANGA,
    "gen_eva_bvoran": GenEVABVoRAN,
    "gen_eva_bvoran_ga": GenEVABVoRANGA,
    "gen_afa_bvoran": GenAFABVoRAN,
    "gen_afa_bvoran_ga": GenAFABVoRANGA,
    "gen_afa_eva_bvoran": GenAFAEVABVoRAN,
    "gen_afa_eva_bvoran_ga": GenAFAEVABVoRANGA,
    "gen_sr_bvoran": GenSRBVoRAN,
    "gen_sr_bvoran_ga": GenSRBVoRANGA,
    "gen_sr_eva_bvoran": GenSREVABVoRAN,
    "gen_sr_eva_bvoran_ga": GenSREVABVoRANGA,
    "gen_sr_afa_bvoran": GenSRAFABVoRAN,
    "gen_sr_afa_bvoran_ga": GenSRAFABVoRANGA,
    "gen_sr_afa_eva_bvoran": GenSRAFAEVABVoRAN,
    "gen_sr_afa_eva_bvoran_ga": GenSRAFAEVABVoRANGA,
    "gen_knit_bvoran": GenKnitBVoRAN,
    "gen_knit_bvoran_ga": GenKnitBVoRANGA,
    "gen_knit_eva_bvoran": GenKnitEVABVoRAN,
    "gen_knit_eva_bvoran_ga": GenKnitEVABVoRANGA,
    "gen_knit_afa_bvoran": GenKnitAFABVoRAN,
    "gen_knit_afa_bvoran_ga": GenKnitAFABVoRANGA,
    "gen_knit_afa_eva_bvoran": GenKnitAFAEVABVoRAN,
    "gen_knit_afa_eva_bvoran_ga": GenKnitAFAEVABVoRANGA,
    "gen_knit_sr_bvoran": GenKnitSRBVoRAN,
    "gen_knit_sr_bvoran_ga": GenKnitSRBVoRANGA,
    "gen_knit_sr_eva_bvoran": GenKnitSREVABVoRAN,
    "gen_knit_sr_eva_bvoran_ga": GenKnitSREVABVoRANGA,
    "gen_knit_sr_afa_bvoran": GenKnitSRAFABVoRAN,
    "gen_knit_sr_afa_bvoran_ga": GenKnitSRAFABVoRANGA,
    "gen_knit_sr_afa_eva_bvoran": GenKnitSRAFAEVABVoRAN,
    "gen_knit_sr_afa_eva_bvoran_ga": GenKnitSRAFAEVABVoRANGA,
    "gen_ebvoran": GenEBVoRAN,
    "gen_ebvoran_ga": GenEBVoRANGA,
    "gen_eva_ebvoran": GenEVAEBVoRAN,
    "gen_eva_ebvoran_ga": GenEVAEBVoRANGA,
    "gen_afa_ebvoran": GenAFAEBVoRAN,
    "gen_afa_ebvoran_ga": GenAFAEBVoRANGA,
    "gen_afa_eva_ebvoran": GenAFAEVAEBVoRAN,
    "gen_afa_eva_ebvoran_ga": GenAFAEVAEBVoRANGA,
    "gen_sr_ebvoran": GenSREBVoRAN,
    "gen_sr_ebvoran_ga": GenSREBVoRANGA,
    "gen_sr_eva_ebvoran": GenSREVAEBVoRAN,
    "gen_sr_eva_ebvoran_ga": GenSREVAEBVoRANGA,
    "gen_sr_afa_ebvoran": GenSRAFAEBVoRAN,
    "gen_sr_afa_ebvoran_ga": GenSRAFAEBVoRANGA,
    "gen_sr_afa_eva_ebvoran": GenSRAFAEVAEBVoRAN,
    "gen_sr_afa_eva_ebvoran_ga": GenSRAFAEVAEBVoRANGA,
    "gen_knit_ebvoran": GenKnitEBVoRAN,
    "gen_knit_ebvoran_ga": GenKnitEBVoRANGA,
    "gen_knit_eva_ebvoran": GenKnitEVAEBVoRAN,
    "gen_knit_eva_ebvoran_ga": GenKnitEVAEBVoRANGA,
    "gen_knit_afa_ebvoran": GenKnitAFAEBVoRAN,
    "gen_knit_afa_ebvoran_ga": GenKnitAFAEBVoRANGA,
    "gen_knit_afa_eva_ebvoran": GenKnitAFAEVAEBVoRAN,
    "gen_knit_afa_eva_ebvoran_ga": GenKnitAFAEVAEBVoRANGA,
    "gen_knit_sr_ebvoran": GenKnitSREBVoRAN,
    "gen_knit_sr_ebvoran_ga": GenKnitSREBVoRANGA,
    "gen_knit_sr_eva_ebvoran": GenKnitSREVAEBVoRAN,
    "gen_knit_sr_eva_ebvoran_ga": GenKnitSREVAEBVoRANGA,
    "gen_knit_sr_afa_ebvoran": GenKnitSRAFAEBVoRAN,
    "gen_knit_sr_afa_ebvoran_ga": GenKnitSRAFAEBVoRANGA,
    "gen_knit_sr_afa_eva_ebvoran": GenKnitSRAFAEVAEBVoRAN,
    "gen_knit_sr_afa_eva_ebvoran_ga": GenKnitSRAFAEVAEBVoRANGA,
}

# Unique standalone architectures (not covered by Gen variants)
UNIQUE_STANDALONE: dict[str, type[LowRankAdapter]] = {
    "dora": DoRA,
    "bora": BoRA,
    "edora": EDoRA,
    "dvora": DVoRA,
    "doran": DoRAN,
    "qdora": QDoRA,
    "qadora": QADoRA,
    "edoran": EDoRAN,
    "ebora": EBoRA,
    "eboran": EBoRAN,
    "bvora": BVoRA,
    "se_bvoran": SeBVoRAN,
    "ese_bvoran": ESeBVoRAN,
    "esr_bvoran": ESRBVoRAN,
    "b_pveran": BPVERAN,
    "eb_pveran": EBPVERAN,
    "bv_auroran": BVAuroRAN,
    "ebv_auroran": EBVAuroRAN,
    "diag_loran": DiagLoRAN,
    "multiangle_loran": MultiAngleLoRAN,
    "cycled_bvoran": CycledBoRAN,
    "cycled_axial_loran": CycledAxialBoRA,
    "plain_lora": PlainLoRA,
    "cycled_diag_lora": CycledDiagLoRA,
    "stream_fusion": StreamFusionLoRA,
}

ADAPTER_VARIANTS.update(UNIQUE_STANDALONE)

# Cycling variants with and without LayerNorm
def _cycled_wrapper(cls, has_norm):
    class Wrapped(cls):
        def __init__(self, in_f, out_f, config):
            config.extra_kwargs["has_norm"] = has_norm
            super().__init__(in_f, out_f, config)
    Wrapped.__name__ = f"{cls.__name__}_{'norm' if has_norm else 'nonorm'}"
    return Wrapped

for suffix, cls, has_norm in [
    ("cycled_bora", CycledBoRAN, False),
    ("cycled_axial_bora", CycledAxialBoRA, False),
    ("cycled_boran", CycledBoRAN, True),
    ("cycled_axial_boran", CycledAxialBoRA, True),
]:
    ADAPTER_VARIANTS[f"gend_{suffix}"] = _cycled_wrapper(cls, has_norm)

# Zipped Knit variants for SmolLM test
ADAPTER_VARIANTS["gend_knit_dora"] = _cycled_wrapper(
    __import__("src.adapters.dora_combo_adapters", fromlist=["GenDKnitDoRA"]).GenDKnitDoRA, False
)
ADAPTER_VARIANTS["gend_knit_doran"] = _cycled_wrapper(
    __import__("src.adapters.dora_combo_adapters", fromlist=["GenDKnitDoRAN"]).GenDKnitDoRAN, True
)

# DoRA-style (single-direction) combinatoric variants, ±LayerNorm
from src.adapters.dora_combo_adapters import (GenDDoRA, GenDDoRAN, GenDDoRAGA, GenDDoRANGA,
    GenDEVADoRA, GenDEVADoRAN, GenDEVADoRAGA, GenDEVADoRANGA,
    GenDAFADoRA, GenDAFADoRAN, GenDAFADoRAGA, GenDAFADoRANGA,
    GenDAFAEVADoRA, GenDAFAEVADoRAN, GenDAFAEVADoRAGA, GenDAFAEVADoRANGA,
    GenDSRDoRA, GenDSRDoRAN, GenDSRDoRAGA, GenDSRDoRANGA,
    GenDSREVADoRA, GenDSREVADoRAN, GenDSREVADoRAGA, GenDSREVADoRANGA,
    GenDSRAFADoRA, GenDSRAFADoRAN, GenDSRAFADoRAGA, GenDSRAFADoRANGA,
    GenDSRAFAEVADoRA, GenDSRAFAEVADoRAN, GenDSRAFAEVADoRAGA, GenDSRAFAEVADoRANGA,)
DORA_COMBO: dict[str, type[LowRankAdapter]] = {}
for suffix, cls in [
    ("dora", GenDDoRA), ("dora_ga", GenDDoRAGA),
    ("eva_dora", GenDEVADoRA), ("eva_dora_ga", GenDEVADoRAGA),
    ("afa_dora", GenDAFADoRA), ("afa_dora_ga", GenDAFADoRAGA),
    ("afa_eva_dora", GenDAFAEVADoRA), ("afa_eva_dora_ga", GenDAFAEVADoRAGA),
    ("sr_dora", GenDSRDoRA), ("sr_dora_ga", GenDSRDoRAGA),
    ("sr_eva_dora", GenDSREVADoRA), ("sr_eva_dora_ga", GenDSREVADoRAGA),
    ("sr_afa_dora", GenDSRAFADoRA), ("sr_afa_dora_ga", GenDSRAFADoRAGA),
    ("sr_afa_eva_dora", GenDSRAFAEVADoRA), ("sr_afa_eva_dora_ga", GenDSRAFAEVADoRAGA),
    ("doran", GenDDoRAN), ("doran_ga", GenDDoRANGA),
    ("eva_doran", GenDEVADoRAN), ("eva_doran_ga", GenDEVADoRANGA),
    ("afa_doran", GenDAFADoRAN), ("afa_doran_ga", GenDAFADoRANGA),
    ("afa_eva_doran", GenDAFAEVADoRAN), ("afa_eva_doran_ga", GenDAFAEVADoRANGA),
    ("sr_doran", GenDSRDoRAN), ("sr_doran_ga", GenDSRDoRANGA),
    ("sr_eva_doran", GenDSREVADoRAN), ("sr_eva_doran_ga", GenDSREVADoRANGA),
    ("sr_afa_doran", GenDSRAFADoRAN), ("sr_afa_doran_ga", GenDSRAFADoRANGA),
    ("sr_afa_eva_doran", GenDSRAFAEVADoRAN), ("sr_afa_eva_doran_ga", GenDSRAFAEVADoRANGA),
]:
    DORA_COMBO[f"gend_{suffix}"] = cls
ADAPTER_VARIANTS.update(DORA_COMBO)


def find_linear_layers(model: nn.Module, target_modules: tuple[str, ...]) -> dict[str, nn.Linear]:
    layers = {}
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in target_modules) and isinstance(module, nn.Linear):
            layers[name] = module
    return layers


def inject_adapters(
    model: PreTrainedModel,
    adapter_cls: type[LowRankAdapter],
    config: AdapterConfig,
) -> list[nn.Module]:
    adapters = []
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in config.target_modules) and isinstance(module, nn.Linear):
            wrapped = adapt_linear_layer(module, adapter_cls, config)
            parent_name, _, child_name = name.rpartition(".")
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, wrapped)
            if hasattr(wrapped.adapter, "module_path"):
                wrapped.adapter.module_path = name
            adapters.append(wrapped.adapter)
    return adapters


def restore_original_layers(
    model: PreTrainedModel,
    target_modules: tuple[str, ...],
    saved_weights: dict[str, torch.Tensor],
):
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in target_modules) and isinstance(module, AdapterWrappedLinear):
            parent_name, _, child_name = name.rpartition(".")
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            w = saved_weights.get(name)
            if w is not None:
                bias = module.base_linear.bias
                new_linear = nn.Linear(w.shape[1], w.shape[0], bias=(bias is not None), device=device, dtype=w.dtype)
                new_linear.weight.data.copy_(w.to(device))
                if bias is not None:
                    new_linear.bias.data.copy_(bias.data.to(device))
                setattr(parent, child_name, new_linear)


def count_all_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_variant(
    variant_name: str,
    adapter_cls: type[LowRankAdapter],
    config: AdapterConfig,
    model: PreTrainedModel,
    tokenizer,
    saved_weights: dict[str, torch.Tensor],
    train_texts: list[str],
    eval_texts: list[str],
    device: str = "cuda",
    max_steps: int = 100,
    lr: float = 2e-4,
    batch_size: int = 2,
    max_length: int = 512,
    num_trials_speed: int = 10,
) -> BenchmarkResult:
    print(f"\n{'='*60}")
    print(f"  Running variant: {variant_name}")
    print(f"  Config: r={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"  Train samples: {len(train_texts)}, Eval samples: {len(eval_texts)}")
    print(f"{'='*60}")

    model.requires_grad_(False)
    restore_original_layers(model, config.target_modules, saved_weights)
    gc.collect()
    torch.cuda.empty_cache()

    print("  Injecting adapters...")
    adapter_modules = inject_adapters(model, adapter_cls, config)
    trainable_params = count_all_trainable(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
    param_efficiency = trainable_params / total_params

    from src.training.data import InstructionDataset
    train_dataset = InstructionDataset(train_texts, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = InstructionDataset(eval_texts, tokenizer, max_length)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    memory_stats = measure_memory(model, device)

    print("  Training...")
    epochs = max(1, max_steps // max(len(train_loader), 1))
    trainer = Trainer(
        model=model,
        adapter_modules=adapter_modules,
        train_loader=train_loader,
        lr=lr,
        max_steps=max_steps,
        num_epochs=epochs,
        log_interval=10,
        device=device,
    )
    metrics = trainer.train()
    final_loss = metrics.losses[-1] if metrics.losses else float("inf")
    training_loss = sum(metrics.losses[-10:]) / max(len(metrics.losses[-10:]), 1)

    print(f"  Final loss: {final_loss:.4f}, Perplexity: {compute_perplexity(final_loss):.2f}")

    # Eval on held-out set
    print("  Evaluating on held-out set...")
    model.eval()
    eval_losses = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_losses.append(outputs.loss.item())
    eval_loss = sum(eval_losses) / max(len(eval_losses), 1)
    eval_ppl = compute_perplexity(eval_loss)
    print(f"  Eval loss: {eval_loss:.4f}, Eval PPL: {eval_ppl:.2f}")

    # Inference speed
    input_ids = torch.randint(0, 1000, (1, 32)).to(device)
    attention_mask = torch.ones_like(input_ids)
    clear_knit_registry()
    speed_metrics = {"avg_latency_ms": 0.0, "tokens_per_second": 0.0, "std_latency_ms": 0.0}
    try:
        speed_metrics = measure_inference_speed(
            model, input_ids, attention_mask,
            num_trials=num_trials_speed, max_new_tokens=10,
            device=device,
        )
    except Exception as e:
        print(f"  Warning: inference speed test failed: {e}")
    torch.cuda.empty_cache()

    # Frobenius ratio (handle both lora_A/lora_B and lora_A_fwd/lora_B_fwd naming)
    total_delta_norm = 0.0
    total_weight_norm = 0.0
    for m in adapter_modules:
        if hasattr(m, "frozen_weight"):
            fw = m.frozen_weight
            if hasattr(m.adapter, "lora_B"):
                ba = (m.adapter.lora_B @ m.adapter.lora_A) * m.adapter.scaling
                total_delta_norm += ba.norm(p="fro").item()
            elif hasattr(m.adapter, "lora_B_fwd"):
                ba_fwd = (m.adapter.lora_B_fwd @ m.adapter.lora_A_fwd) * m.adapter.scaling
                total_delta_norm += ba_fwd.norm(p="fro").item()
            total_weight_norm += fw.norm(p="fro").item()
    frobenius_ratio = total_delta_norm / total_weight_norm if total_weight_norm > 0 else 0.0

    cleanup_variant(model, config, saved_weights)
    del trainer, train_loader, eval_loader, train_dataset, eval_dataset, adapter_modules
    result = BenchmarkResult(
        variant_name=variant_name,
        config={
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
        },
        trainable_params=trainable_params,
        total_params=total_params,
        param_efficiency=param_efficiency,
        training_loss=training_loss,
        training_perplexity=compute_perplexity(training_loss),
        final_loss=final_loss,
        final_perplexity=compute_perplexity(final_loss),
        eval_loss=eval_loss,
        eval_perplexity=eval_ppl,
        memory_peak_mb=metrics.memory_peak_mb,
        memory_allocated_mb=metrics.memory_allocated_mb,
        total_model_memory_mb=memory_stats.get("total_model_memory_mb", 0),
        avg_latency_ms=speed_metrics["avg_latency_ms"],
        tokens_per_second=speed_metrics["tokens_per_second"],
        grad_norm_avg=sum(metrics.grad_norms) / max(len(metrics.grad_norms), 1),
        step_time_avg_ms=sum(metrics.step_times) / max(len(metrics.step_times), 1) * 1000,
        total_training_steps=len(metrics.losses),
        frobenius_ratio=frobenius_ratio,
    )

    return result


def cleanup_variant(model, config, saved_weights):
    try:
        restore_original_layers(model, config.target_modules, saved_weights)
    except Exception as e:
        print(f"  [cleanup] restore error: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="RankAdaptation: Qwen3.5-2B sweep")
    parser.add_argument("--variants", nargs="+", default=list(ADAPTER_VARIANTS.keys()),
                        help="Variants to run (default: all)")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=8.0, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--max-steps", type=int, default=100, help="Training steps per variant")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default="results_2b.json", help="Output JSON path")
    parser.add_argument("--num-train", type=int, default=1000, help="Number of training texts")
    parser.add_argument("--num-eval", type=int, default=200, help="Number of eval texts")
    parser.add_argument("--edora-group-size", type=int, default=0, help="EDoRA group size (0=auto)")
    parser.add_argument("--resume", type=str, default="", help="Resume from existing results JSON")
    args = parser.parse_args()

    # Load data
    print("  Loading arxiv data...")
    train_texts, eval_texts = load_arxiv_data(args.num_train, args.num_eval)
    print(f"  Train: {len(train_texts)} texts, Eval: {len(eval_texts)} texts")

    config = AdapterConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        edora_group_size=args.edora_group_size,
    )

    report = ComparisonReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))

    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            existing = json.load(f)
        completed_names = {r["variant_name"] for r in existing.get("results", []) if r.get("status") == "completed"}
        for r_data in existing.get("results", []):
            report.add(BenchmarkResult(**r_data))
        variants_to_run = [v for v in args.variants if v not in completed_names]
        if not variants_to_run:
            print("All variants already completed!")
            print(report.summary_table())
            return
        print(f"  Skipping {len(completed_names)} completed, queued {len(variants_to_run)} remaining")
    else:
        variants_to_run = args.variants

    print(f"  Loading model ({MODEL_PATH})...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=args.device if args.device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Store weight TENSORS (not full modules) for O(4GB) memory savings
    saved_weights: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in config.target_modules) and isinstance(module, nn.Linear):
            saved_weights[name] = module.weight.data.clone().cpu()

    for vname in variants_to_run:
        if "Knit" in ADAPTER_VARIANTS[vname].__name__:
            clear_knit_registry()
        try:
            result = run_variant(
                variant_name=vname,
                adapter_cls=ADAPTER_VARIANTS[vname],
                config=config,
                model=model,
                tokenizer=tokenizer,
                saved_weights=saved_weights,
                train_texts=train_texts,
                eval_texts=eval_texts,
                device=args.device,
                max_steps=args.max_steps,
                lr=args.lr,
                batch_size=args.batch_size,
            )
            report.add(result)
            report.to_json(args.output)
            print(f"  Saved to {args.output}")
        except Exception as e:
            print(f"  ERROR running {vname}: {e}")
            import traceback
            traceback.print_exc()
            cleanup_variant(model, config, saved_weights)
            report.add(BenchmarkResult(
                variant_name=vname,
                config={"r": config.r, "lora_alpha": config.lora_alpha, "lora_dropout": config.lora_dropout},
                trainable_params=0, total_params=0, param_efficiency=0,
                training_loss=0, training_perplexity=0, final_loss=0, final_perplexity=0,
                memory_peak_mb=0, memory_allocated_mb=0, total_model_memory_mb=0,
                avg_latency_ms=0, tokens_per_second=0, grad_norm_avg=0, step_time_avg_ms=0,
                total_training_steps=0, frobenius_ratio=0,
                status="failed", error=str(e),
            ))
            report.to_json(args.output)
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(report.summary_table())
    report.to_json(args.output)
    print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
