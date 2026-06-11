#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters import (
    AdapterConfig, AdapterWrappedLinear, LowRankAdapter, adapt_linear_layer,
    DoRA, EDoRA, DVoRA, DoRAN, QDoRA, QADoRA, EDoRAN,
    DiagLoRAN, MultiAngleLoRAN, CycledBVoRAN,
    GenDDoRAN, GenDDoRANGA, GenDEVADoRAN, GenDEVADoRANGA,
    GenDAFADoRAN, GenDAFADoRANGA, GenDAFAEVADoRAN, GenDAFAEVADoRANGA,
    GenDSRDoRAN, GenDSRDoRANGA, GenDSREVADoRAN, GenDSREVADoRANGA,
    GenDSRAFADoRAN, GenDSRAFADoRANGA, GenDSRAFAEVADoRAN, GenDSRAFAEVADoRANGA,
    GenDKnitDoRAN, GenDKnitDoRANGA, GenDKnitEVADoRAN, GenDKnitEVADoRANGA,
    GenDKnitAFADoRAN, GenDKnitAFADoRANGA, GenDKnitAFAEVADoRAN, GenDKnitAFAEVADoRANGA,
    GenDKnitSRDoRAN, GenDKnitSRDoRANGA, GenDKnitSREVADoRAN, GenDKnitSREVADoRANGA,
    GenDKnitSRAFADoRAN, GenDKnitSRAFADoRANGA, GenDKnitSRAFAEVADoRAN, GenDKnitSRAFAEVADoRANGA,
)
from src.models.toy_transformer import ToyTransformer, ToyConfig
from src.adapters.knit_bvoran import clear_knit_registry


@dataclass
class ToyResult:
    variant_name: str
    config: dict[str, Any]
    trainable_params: int
    total_params: int
    param_efficiency: float
    train_loss: float
    eval_loss: float
    eval_perplexity: float
    total_steps: int
    step_time_ms: float
    status: str = "completed"
    error: str | None = None


VARIANTS: dict[str, type[LowRankAdapter]] = dict(
    dora=DoRA, edora=EDoRA, dvora=DVoRA, doran=DoRAN,
    qdora=QDoRA, qadora=QADoRA, edoran=EDoRAN,
    diag_loran=DiagLoRAN, multiangle_loran=MultiAngleLoRAN,
    cycled_bvoran=CycledBVoRAN,
    **{f"gend_{s}": c for s, c in [
        ("doran", GenDDoRAN), ("doran_ga", GenDDoRANGA),
        ("eva_doran", GenDEVADoRAN), ("eva_doran_ga", GenDEVADoRANGA),
        ("afa_doran", GenDAFADoRAN), ("afa_doran_ga", GenDAFADoRANGA),
        ("afa_eva_doran", GenDAFAEVADoRAN), ("afa_eva_doran_ga", GenDAFAEVADoRANGA),
        ("sr_doran", GenDSRDoRAN), ("sr_doran_ga", GenDSRDoRANGA),
        ("sr_eva_doran", GenDSREVADoRAN), ("sr_eva_doran_ga", GenDSREVADoRANGA),
        ("sr_afa_doran", GenDSRAFADoRAN), ("sr_afa_doran_ga", GenDSRAFADoRANGA),
        ("sr_afa_eva_doran", GenDSRAFAEVADoRAN), ("sr_afa_eva_doran_ga", GenDSRAFAEVADoRANGA),
        ("knit_doran", GenDKnitDoRAN), ("knit_doran_ga", GenDKnitDoRANGA),
        ("knit_eva_doran", GenDKnitEVADoRAN), ("knit_eva_doran_ga", GenDKnitEVADoRANGA),
        ("knit_afa_doran", GenDKnitAFADoRAN), ("knit_afa_doran_ga", GenDKnitAFADoRANGA),
        ("knit_afa_eva_doran", GenDKnitAFAEVADoRAN), ("knit_afa_eva_doran_ga", GenDKnitAFAEVADoRANGA),
        ("knit_sr_doran", GenDKnitSRDoRAN), ("knit_sr_doran_ga", GenDKnitSRDoRANGA),
        ("knit_sr_eva_doran", GenDKnitSREVADoRAN), ("knit_sr_eva_doran_ga", GenDKnitSREVADoRANGA),
        ("knit_sr_afa_doran", GenDKnitSRAFADoRAN), ("knit_sr_afa_doran_ga", GenDKnitSRAFADoRANGA),
        ("knit_sr_afa_eva_doran", GenDKnitSRAFAEVADoRAN), ("knit_sr_afa_eva_doran_ga", GenDKnitSRAFAEVADoRANGA),
    ]},
)


def find_linear_layers(model, target_modules=("q_proj", "k_proj", "v_proj", "o_proj")):
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in target_modules) and isinstance(module, nn.Linear):
            yield name, module


def inject_adapters(model, adapter_cls, config, device="cuda"):
    modules = []
    for name, module in find_linear_layers(model):
        wrapped = adapt_linear_layer(module, adapter_cls, config)
        wrapped = wrapped.to(device)
        parent_name, _, child = name.rpartition(".")
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        setattr(parent, child, wrapped)
        if hasattr(wrapped.adapter, "module_path"):
            wrapped.adapter.module_path = name
        modules.append(wrapped.adapter)
    return modules


def restore_originals(model, saved_weights):
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if isinstance(module, AdapterWrappedLinear):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            w = saved_weights.get(name)
            if w is not None:
                bias = module.base_linear.bias
                new_lin = nn.Linear(w.shape[1], w.shape[0], bias=(bias is not None), device=device, dtype=w.dtype)
                new_lin.weight.data.copy_(w.to(device, non_blocking=True))
                if bias is not None:
                    new_lin.bias.data.copy_(bias.data.to(device, non_blocking=True))
                setattr(parent, child, new_lin)


def compute_ppl(loss):
    return math.exp(min(loss, 20.0))


def run_variant(name, adapter_cls, config, model, saved_weights, train_loader, eval_loader,
                device, max_steps, lr):
    model.requires_grad_(False)
    restore_originals(model, saved_weights)
    gc.collect()
    torch.cuda.empty_cache()

    modules = inject_adapters(model, adapter_cls, config, device=device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(
        [p for m in modules for p in m.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    model.train()
    losses = []
    step = 0
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            t0 = time.time()
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            loss = model(input_ids=input_ids, labels=labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_time = time.time() - t0
            losses.append(loss.item())
            if step % 10 == 0:
                print(f"  Step {step:>4} | Loss {loss.item():.4f} | {step_time*1000:.0f}ms")
            step += 1

    train_loss = sum(losses[-10:]) / max(len(losses[-10:]), 1)
    final_loss = losses[-1] if losses else float("inf")

    model.eval()
    eval_losses = []
    with torch.no_grad():
        for batch in eval_loader:
            loss = model(input_ids=batch[0].to(device), labels=batch[1].to(device))
            eval_losses.append(loss.item())
    eval_loss = sum(eval_losses) / max(len(eval_losses), 1)

    restore_originals(model, saved_weights)
    del modules, opt, losses
    gc.collect()
    torch.cuda.empty_cache()

    params = trainable
    if hasattr(adapter_cls, "active_params") and not isinstance(adapter_cls.active_params, property):
        pass

    return ToyResult(
        variant_name=name,
        config={"r": config.r, "alpha": config.lora_alpha},
        trainable_params=trainable,
        total_params=total,
        param_efficiency=trainable / max(total, 1),
        train_loss=train_loss,
        eval_loss=eval_loss,
        eval_perplexity=compute_ppl(eval_loss),
        total_steps=step,
        step_time_ms=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results_toy.json")
    parser.add_argument("--num-train", type=int, default=2048)
    parser.add_argument("--num-eval", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    print("  Creating toy transformer...")
    model_cfg = ToyConfig()
    model = ToyTransformer(model_cfg).to(args.device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    print("  Generating data...")
    torch.manual_seed(42)
    train_ids = torch.randint(1, model_cfg.vocab_size, (args.num_train, args.seq_len))
    train_targets = torch.randint(1, model_cfg.vocab_size, (args.num_train, args.seq_len))
    eval_ids = torch.randint(1, model_cfg.vocab_size, (args.num_eval, args.seq_len))
    eval_targets = torch.randint(1, model_cfg.vocab_size, (args.num_eval, args.seq_len))
    train_loader = DataLoader(TensorDataset(train_ids, train_targets), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(eval_ids, eval_targets), batch_size=args.batch_size)

    saved_weights = {}
    for name, module in find_linear_layers(model):
        saved_weights[name] = module.weight.data.clone().cpu()

    results = []
    to_run = [v for v in args.variants if v in VARIANTS]
    print(f"\n  Running {len(to_run)} variants\n")

    for vname in to_run:
        print(f"\n{'='*50}")
        print(f"  {vname}")
        print(f"{'='*50}")
        try:
            result = run_variant(
                vname, VARIANTS[vname], AdapterConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout),
                model, saved_weights, train_loader, eval_loader,
                args.device, args.max_steps, args.lr,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append(ToyResult(
                variant_name=vname, config={}, trainable_params=0, total_params=0,
                param_efficiency=0, train_loss=0, eval_loss=0, eval_perplexity=0,
                total_steps=0, step_time_ms=0, status="failed", error=str(e),
            ))
        # Save intermediate
        with open(args.output, "w") as f:
            json.dump({"results": [asdict(r) for r in results]}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  RANKING")
    print(f"{'='*60}")
    sorted_r = sorted(results, key=lambda r: r.eval_loss)
    print(f"  {'Rank':<5} {'Variant':<30} {'EvalLoss':<10} {'EvalPPL':<10} {'Params':<10}")
    print(f"  {'-'*65}")
    for i, r in enumerate(sorted_r[:20], 1):
        print(f"  {i:<5} {r.variant_name:<30} {r.eval_loss:<10.4f} {r.eval_perplexity:<10.2f} {r.trainable_params:<10,}")

    with open(args.output, "w") as f:
        json.dump({"results": [asdict(r) for r in sorted_r]}, f, indent=2)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
