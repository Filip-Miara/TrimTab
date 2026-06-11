#!/usr/bin/env python3
"""Evaluate WeightDiffusion vs SGD vs zero on real LM loss."""
from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.stream_fusion import PlainStreamExpert, N_FLAGS
from src.adapters.diffusion_weight_flow import WeightDiffusion, DiffusionFlowTrainer

MODEL_PATH = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub"),
    "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17",
)
ARXIV_DIR = "/run/media/filip/B522-875D/Datasets/hub/datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section"
DEVICE = "cuda"
RANK = 8
N_TEST = 5
STEPS = 20


def load_test_texts(n: int, offset: int) -> list[str]:
    parq_files = sorted(os.path.join(ARXIV_DIR, f) for f in os.listdir(ARXIV_DIR) if f.endswith(".parquet"))
    texts = []
    for pf in parq_files:
        tbl = pq.read_table(pf, columns=["article"])
        for art in tbl.column("article"):
            t = art.as_py()
            if len(t) > 500:
                texts.append(t)
                if len(texts) >= offset + n:
                    return texts[offset:offset + n]
    return texts[offset:offset + n]


@torch.no_grad()
def compute_loss(model, input_ids):
    return model(input_ids=input_ids, labels=input_ids.clone()).loss.item()


def main():
    print("=" * 60)
    print("  WeightDiffusion Evaluation on Qwen3.5-0.8B")
    print("=" * 60)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)

    target_name = "model.layers.3.self_attn.q_proj"
    target_module = model.get_submodule(target_name)
    orig_forward = target_module.forward
    d_in, d_out = target_module.in_features, target_module.out_features
    n_weights = PlainStreamExpert.absorb_dim(d_in, d_out, RANK)
    d_ctx = d_in + d_out
    print(f"Target: {target_name} ({d_in}→{d_out}), weights: {n_weights}")

    # Load diffusion model
    flow = WeightDiffusion(n_weights, d_latent=64, n_latents=16, d_ctx=d_ctx)
    sd = torch.load("diffusion_weight_flow.pt", map_location="cpu")
    # Handle architecture migration: old single decode → new separate decode_noise/decode_flow
    if "decode.weight" in sd and "decode_noise.weight" not in sd:
        sd["decode_noise.weight"] = sd["decode.weight"][:1]   # first row
        sd["decode_noise.bias"] = sd["decode.bias"][:1]
        sd["decode_flow.weight"] = sd["decode.weight"][1:]    # second row
        sd["decode_flow.bias"] = sd["decode.bias"][1:]
        del sd["decode.weight"], sd["decode.bias"]
    flow.load_state_dict(sd, strict=False)
    flow.to(DEVICE).eval()
    print(f"Diffusion model loaded (weights: {sum(p.numel() for p in flow.parameters()):,})")

    print(f"\nLoading {N_TEST} test texts...")
    texts = load_test_texts(N_TEST, 15)
    print(f"  Loaded {len(texts)} texts")

    results = []
    for i, text in enumerate(texts):
        print(f"\n--- Test {i + 1}/{N_TEST} ---")

        # Tokenize
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        labels = input_ids.clone()

        # Get hidden state context
        cache = {}
        def fwd_hook(m, inp, out):
            cache["x"] = inp[0].detach().float().mean(dim=1)
        handle = target_module.register_forward_hook(fwd_hook)
        model(input_ids=input_ids)
        handle.remove()
        ctx_hidden = cache.get("x", torch.zeros(1, d_in))

        # === 1. SGD trajectory ===
        adapter = PlainStreamExpert(d_in, d_out, RANK, 64, 64, 8.0 / RANK).to(DEVICE, dtype=torch.bfloat16)
        def fwd_sgd(x): return orig_forward(x) + adapter(x)
        target_module.forward = fwd_sgd

        grad_ctx = torch.zeros(1, d_out)
        opt = torch.optim.Adam([p for n,p in adapter.named_parameters() if 'lora_A' in n or 'lora_B' in n], lr=1e-3)
        for step in range(STEPS):
            loss = model(input_ids=input_ids, labels=labels).loss
            opt.zero_grad(); loss.backward(); opt.step()
            if step == 0:
                for n, p in adapter.named_parameters():
                    if 'lora_B' in n and p.grad is not None:
                        grad_ctx = p.grad.float().mean(dim=1).unsqueeze(0).cpu()
                        break
        sgd_loss = compute_loss(model, input_ids)
        print(f"  SGD:  {sgd_loss:.4f}")
        target_module.forward = orig_forward

        # === 2. WeightDiffusion (flow-only: start from zero, skip DDIM) ===
        ctx = torch.cat([ctx_hidden.cpu(), grad_ctx], dim=-1).to(DEVICE)
        flags = torch.zeros(1, N_FLAGS, device=DEVICE)
        with torch.no_grad():
            w_gen = torch.zeros(1, n_weights, device=DEVICE)
            for s in range(STEPS):
                _, v = flow(w_gen, torch.zeros(1, 1, device=DEVICE),
                           torch.tensor([[s / STEPS]], device=DEVICE),
                           flags, ctx)
                w_gen = w_gen + v / STEPS

        adapter_flow = PlainStreamExpert(d_in, d_out, RANK, 64, 64, 8.0 / RANK).to(DEVICE, dtype=torch.bfloat16)
        sidx = 0
        for n, p in adapter_flow.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:
                ne = p.numel()
                p.data.copy_(w_gen[0, sidx:sidx+ne].reshape(p.shape).to(torch.bfloat16))
                sidx += ne

        def fwd_flow(x): return orig_forward(x) + adapter_flow(x)
        target_module.forward = fwd_flow
        flow_loss = compute_loss(model, input_ids)
        print(f"  Flow: {flow_loss:.4f}")
        target_module.forward = orig_forward

        # === 3. Zero weights ===
        adapter_zero = PlainStreamExpert(d_in, d_out, RANK, 64, 64, 8.0 / RANK).to(DEVICE, dtype=torch.bfloat16)
        for p in adapter_zero.parameters():
            if p.requires_grad: p.data.zero_()
        def fwd_zero(x): return orig_forward(x) + adapter_zero(x)
        target_module.forward = fwd_zero
        zero_loss = compute_loss(model, input_ids)
        print(f"  Zero: {zero_loss:.4f}")
        target_module.forward = orig_forward

        results.append({"sgd": sgd_loss, "flow": flow_loss, "zero": zero_loss})
        del adapter, adapter_flow, adapter_zero, opt; gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for method, key in [("SGD", "sgd"), ("Diffusion", "flow"), ("Zero", "zero")]:
        vals = [r[key] for r in results]
        print(f"  {method:>10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    zero_mean = np.mean([r["zero"] for r in results])
    flow_mean = np.mean([r["flow"] for r in results])
    sgd_mean = np.mean([r["sgd"] for r in results])
    print(f"\n  Flow gain vs Zero: {zero_mean - flow_mean:.4f} ({((zero_mean - flow_mean)/zero_mean*100):.1f}%)")
    print(f"  Flow/SGD ratio: {flow_mean/sgd_mean:.2f}x")
    print(f"  Flow < Zero: {sum(r['flow'] < r['zero'] for r in results)}/{len(results)}")

    with open("diffusion_flow_eval.json", "w") as f:
        json.dump({"results": results, "summary": {
            "sgd_mean": sgd_mean, "flow_mean": flow_mean, "zero_mean": zero_mean,
        }}, f, indent=2)


if __name__ == "__main__":
    main()
