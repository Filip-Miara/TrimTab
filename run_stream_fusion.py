#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters import (
    AdapterConfig, AdapterWrappedLinear, adapt_linear_layer, LowRankAdapter,
    StreamFusionLoRA, StreamFusionConfig,
)
from src.training.trainer import compute_perplexity
from src.evaluation.metrics import measure_memory

MODEL_PATH = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub"),
    "models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f729f269f5fb618e03d52d7c",
)
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"

ARXIV_PARQUET_DIR = "/run/media/filip/B522-875D/Datasets/hub/datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section"


def find_linear_layers(model, target_modules):
    layers = {}
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in target_modules) and isinstance(module, nn.Linear):
            layers[name] = module
    return layers


def inject_adapters(model, adapter_cls, config):
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


def restore_original_layers(model, target_modules, saved_weights):
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


def load_arxiv_data(num_texts=200):
    parq_files = sorted(
        os.path.join(ARXIV_PARQUET_DIR, f)
        for f in os.listdir(ARXIV_PARQUET_DIR)
        if f.endswith(".parquet")
    )
    all_texts = []
    for pf in parq_files:
        pf_path = os.path.join(ARXIV_PARQUET_DIR, pf) if not pf.startswith("/") else pf
        tbl = pq.read_table(pf_path, columns=["article"])
        for art in tbl.column("article"):
            t = art.as_py()
            if len(t) > 500:
                all_texts.append(t)
                if len(all_texts) >= num_texts:
                    return all_texts
    return all_texts


def tokenize_texts(tokenizer, texts, max_length=512):
    return tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
    )


@torch.no_grad()
def eval_perplexity(model, eval_loader, device):
    model.eval()
    losses = []
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(outputs.loss.item())
    avg = sum(losses) / max(len(losses), 1)
    return avg, compute_perplexity(avg)


def main():
    parser = argparse.ArgumentParser(description="StreamFusion-LoRA: Online Continual Learning")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--n-latents", type=int, default=32)
    parser.add_argument("--d-latent", type=int, default=64)
    parser.add_argument("--d-key", type=int, default=64)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--n-segments", type=int, default=10)
    parser.add_argument("--steps-per-segment", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-texts", type=int, default=100)
    parser.add_argument("--expert-variant", type=str, default="plain", choices=["plain", "dora", "bora", "vera", "hybrid"])
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--use-vectors", action="store_true")
    parser.add_argument("--use-norm", action="store_true")
    parser.add_argument("--use-gate", action="store_true")
    parser.add_argument("--use-activation", action="store_true")
    parser.add_argument("--use-autoencoder", action="store_true")
    parser.add_argument("--use-polynomial", action="store_true")
    parser.add_argument("--poly-order", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="stream_fusion_results.json")
    args = parser.parse_args()

    if args.seed != 0:
        torch.manual_seed(args.seed)
        import random
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)

    print("=" * 60)
    print("  StreamFusion-LoRA: Online Continual Learning")
    print("=" * 60)

    print(f"  Loading model ({MODEL_PATH})...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=args.device if args.device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.requires_grad_(False)
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

    saved_weights = {}
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in target_modules) and isinstance(module, nn.Linear):
            saved_weights[name] = module.weight.data.clone().cpu()

    config = AdapterConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.0,
        target_modules=target_modules,
        extra_kwargs={
            "expert_variant": args.expert_variant,
            "bidirectional": args.bidirectional,
            "use_vectors": args.use_vectors,
            "use_norm": args.use_norm,
            "use_gate": args.use_gate,
            "use_activation": args.use_activation,
            "use_autoencoder": args.use_autoencoder,
            "use_polynomial": args.use_polynomial,
            "poly_order": args.poly_order,
            "n_latents": args.n_latents,
            "d_latent": args.d_latent,
            "d_key": args.d_key,
            "top_m": args.top_m,
            "max_experts": 200,
            "abs_steps": 4,
        },
    )

    print("  Injecting StreamFusion adapters...")
    adapter_modules = inject_adapters(model, StreamFusionLoRA, config)
    stream_fusion_adapters = [m for m in adapter_modules if isinstance(m, StreamFusionLoRA)]
    print(f"  {len(stream_fusion_adapters)} adapters installed")

    model.to(args.device)

    def count_trainable():
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Trainable params: {count_trainable():,}")
    print(f"  Loading {args.num_texts} arxiv texts...")
    texts = load_arxiv_data(args.num_texts)
    print(f"  Loaded {len(texts)} texts")

    segments = texts[:args.n_segments] if args.n_segments <= len(texts) else texts
    n_segments = len(segments)
    print(f"  {n_segments} streaming segments")

    from torch.utils.data import DataLoader, TensorDataset

    results = {
        "config": {
            "r": args.r, "alpha": args.alpha,
            "n_latents": args.n_latents, "d_latent": args.d_latent,
            "d_key": args.d_key, "top_m": args.top_m,
        },
        "segments": [],
        "final_eval": {},
    }

    for seg_idx, seg_text in enumerate(segments):
        print(f"\n{'─' * 50}")
        print(f"  Segment {seg_idx + 1}/{n_segments}")
        print(f"{'─' * 50}")

        for adapter in stream_fusion_adapters:
            adapter.add_expert()

        expert_count = stream_fusion_adapters[0].experts.__len__()
        trainable = count_trainable()
        print(f"  Experts per layer: {expert_count}, Trainable: {trainable:,}")

        enc = tokenize_texts(tokenizer, [seg_text])
        seg_ids = enc["input_ids"]
        seg_mask = enc["attention_mask"]
        seg_labels = seg_ids.clone()

        model.train()
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
        )

        total_loss = 0.0
        for step in range(args.steps_per_segment):
            input_ids = seg_ids.to(args.device)
            attention_mask = seg_mask.to(args.device)
            labels = seg_labels.to(args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            opt.step()
            opt.zero_grad()

            del outputs, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / args.steps_per_segment
        ppl = compute_perplexity(avg_loss)

        for adapter in stream_fusion_adapters:
            adapter.absorb()
            adapter.prune(threshold=0.02)

        seg_result = {
            "segment": seg_idx + 1,
            "training_loss": avg_loss,
            "training_ppl": ppl,
            "expert_count": expert_count,
        }

        if seg_idx % 1 == 0:
            eval_texts_subset = texts[max(0, n_segments - 10):n_segments] if n_segments > 10 else texts[:min(10, len(texts))]
            eval_ids = tokenize_texts(tokenizer, eval_texts_subset)["input_ids"]
            with torch.no_grad():
                model.eval()
                eval_losses = []
                for i in range(0, len(eval_ids), args.batch_size):
                    batch_ids = eval_ids[i:i + args.batch_size].to(args.device)
                    batch_mask = torch.ones_like(batch_ids)
                    outs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
                    eval_losses.append(outs.loss.item())
                avg_eval = sum(eval_losses) / max(len(eval_losses), 1)
                eval_ppl = compute_perplexity(avg_eval)
            model.train()
            print(f"  Train loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Eval PPL: {eval_ppl:.2f} | Experts: {expert_count}")
            seg_result["eval_loss"] = avg_eval
            seg_result["eval_ppl"] = eval_ppl
        else:
            print(f"  Train loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Experts: {expert_count}")

        results["segments"].append(seg_result)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        del enc, seg_ids, seg_mask, seg_labels, opt
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"  Total segments: {n_segments}")
    print(f"  Final Expert Count: {stream_fusion_adapters[0].experts.__len__()}")
    print(f"  Results saved to {args.output}")
    print(f"{'=' * 60}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    model.requires_grad_(False)
    restore_original_layers(model, target_modules, saved_weights)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
