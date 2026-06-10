#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import pyarrow.parquet as pq
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters import AdapterConfig, AdapterWrappedLinear, adapt_linear_layer
from src.training.trainer import compute_perplexity

MODEL_PATH = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub"),
    "models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f729f269f5fb618e03d52d7c",
)
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"

ARXIV_PARQUET_DIR = "/run/media/filip/B522-875D/Datasets/hub/datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section"


class PlainLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, scaling):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / rank ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = scaling

    def forward(self, x):
        return self.scaling * (x @ self.lora_A.T @ self.lora_B.T)


def inject_lora(model, rank, alpha, target_modules, device):
    scaling = alpha / rank
    adapters = []
    targets = []
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in target_modules) and isinstance(module, nn.Linear):
            targets.append((name, module))
    for name, module in targets:
        lora = PlainLoRA(module.in_features, module.out_features, rank, scaling)
        lora.to(device, dtype=next(model.parameters()).dtype)
        parent_name, _, child_name = name.rpartition(".")
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        setattr(parent, f"lora_{child_name}", lora)
        adapters.append(lora)

        orig_forward = module.forward
        module._original_forward = orig_forward
        module.forward = lambda x, l=lora, of=orig_forward: of(x) + l(x)
    return adapters


def restore_forward(model, target_modules):
    for module in model.modules():
        if hasattr(module, '_original_forward'):
            module.forward = module._original_forward


def load_arxiv_data(num_texts=200):
    parq_files = sorted(
        os.path.join(ARXIV_PARQUET_DIR, f)
        for f in os.listdir(ARXIV_PARQUET_DIR)
        if f.endswith(".parquet")
    )
    all_texts = []
    for pf in parq_files:
        tbl = pq.read_table(pf, columns=["article"])
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


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Plain LoRA Baseline")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-texts", type=int, default=30)
    parser.add_argument("--output", type=str, default="lora_baseline.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  LoRA Baseline")
    print("=" * 60)

    print(f"  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=args.device if args.device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

    print(f"  Injecting LoRA (r={args.r}, alpha={args.alpha})...")
    adapters = inject_lora(model, args.r, args.alpha, target_modules, args.device)
    print(f"  {len(adapters)} adapters installed, {count_trainable(model):,} trainable params")

    print(f"  Loading {args.num_texts} texts...")
    texts = load_arxiv_data(args.num_texts)
    print(f"  Loaded {len(texts)} texts")

    split = int(args.num_texts * 0.8)
    train_texts = texts[:split]
    eval_texts = texts[split:]

    train_enc = tokenize_texts(tokenizer, train_texts)
    eval_enc = tokenize_texts(tokenizer, eval_texts)

    total_steps = args.steps
    steps_per_text = max(1, total_steps // len(train_texts))

    print(f"  {len(train_texts)} train / {len(eval_texts)} eval texts, {steps_per_text} steps/text = {total_steps} total steps")

    results = {
        "config": {"r": args.r, "alpha": args.alpha, "steps": total_steps, "lr": args.lr},
        "metrics": [],
    }

    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    step = 0
    for epoch in range(max(1, total_steps // (len(train_texts) * steps_per_text) + 1)):
        for ti, text in enumerate(train_texts):
            if step >= total_steps:
                break
            enc = tokenize_texts(tokenizer, [text])
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)
            labels = enc["input_ids"].clone().to(args.device)

            for s in range(steps_per_text):
                if step >= total_steps:
                    break
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                opt.step()
                opt.zero_grad()
                step += 1

                if step % 20 == 0 or step == total_steps:
                    ppl = compute_perplexity(loss.item())
                    print(f"  Step {step}/{total_steps} | loss: {loss.item():.4f} | ppl: {ppl:.2f}")
                    results["metrics"].append({"step": step, "loss": loss.item(), "ppl": ppl})

            if step >= total_steps:
                break
        if step >= total_steps:
            break

    model.eval()
    eval_losses = []
    with torch.no_grad():
        for i in range(0, len(eval_enc["input_ids"]), args.batch_size):
            ids = eval_enc["input_ids"][i:i + args.batch_size].to(args.device)
            mask = eval_enc["attention_mask"][i:i + args.batch_size].to(args.device)
            labels = eval_enc["input_ids"][i:i + args.batch_size].clone().to(args.device)
            outs = model(input_ids=ids, attention_mask=mask, labels=labels)
            eval_losses.append(outs.loss.item())
    avg_eval = sum(eval_losses) / max(len(eval_losses), 1)
    eval_ppl = compute_perplexity(avg_eval)
    print(f"\n  Final eval: loss={avg_eval:.4f} ppl={eval_ppl:.2f}")
    results["eval_loss"] = avg_eval
    results["eval_ppl"] = eval_ppl
    results["final_train_ppl"] = compute_perplexity(results["metrics"][-1]["loss"]) if results["metrics"] else 0

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
