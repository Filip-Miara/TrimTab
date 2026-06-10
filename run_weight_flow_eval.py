#!/usr/bin/env python3
"""Full evaluation: compare flow-generated weights vs SGD vs zero on real LM loss."""
from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.stream_fusion import PlainStreamExpert
from src.adapters.weight_flow import WeightFlowField

MODEL_PATH = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub"),
    "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17",
)
ARXIV_DIR = "/run/media/filip/B522-875D/Datasets/hub/datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section"

DEVICE = "cuda"
RANK = 8
N_TEST = 5
TRAIN_OFFSET = 15  # first 15 were training, next 5 are test
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
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss.item()


def eval_one_trajectory(model, text, target_module_name, orig_forward, device="cuda"):
    """Test SGD vs Flow vs Zero on one text. Returns dict of losses."""
    from src.adapters.stream_fusion import PlainStreamExpert
    in_f = orig_forward.__self__.in_features if hasattr(orig_forward, '__self__') else 1024
    out_f = orig_forward.__self__.out_features if hasattr(orig_forward, '__self__') else 4096
    n_weights = PlainStreamExpert.absorb_dim(in_f, out_f, RANK)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Get data context (mean hidden state + gradient)
    ctx_hidden = None
    cache = {}
    def hook_fn(m, inp, out):
        cache["x"] = inp[0].detach().float().mean(dim=1)  # (B, d)
    handle = orig_forward.__self__.register_forward_hook(hook_fn)
    model(input_ids=input_ids)
    handle.remove()
    ctx_hidden = cache.get("x", torch.zeros(1, in_f))

    # 1. SGD trajectory — capture gradient context from first step
    adapter = PlainStreamExpert(in_f, out_f, RANK, 64, 64, 8.0 / RANK).to(device, dtype=torch.bfloat16)
    def fwd_sgd(x):
        return orig_forward(x) + adapter(x)
    orig_forward.__self__.forward = fwd_sgd

    opt = torch.optim.Adam([p for n,p in adapter.named_parameters() if p.requires_grad and ('lora_A' in n or 'lora_B' in n)], lr=1e-3)
    def _lora_traj(mod):
        return torch.cat([p.data.flatten() for n,p in mod.named_parameters() if 'lora_A' in n or 'lora_B' in n]).float().cpu()
    sgd_traj = [_lora_traj(adapter)]
    grad_ctx = None
    for step in range(STEPS):
        outputs = model(input_ids=input_ids, labels=input_ids.clone())
        loss = outputs.loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0:
            for n, p in adapter.named_parameters():
                if 'lora_B' in n and p.grad is not None:
                    grad_ctx = p.grad.float().mean(dim=1).unsqueeze(0).cpu()
                    break
            if grad_ctx is None:
                grad_ctx = torch.zeros(1, out_f)
        flat = _lora_traj(adapter)
        sgd_traj.append(flat)

    # Build combined context
    ctx = torch.cat([ctx_hidden.cpu(), grad_ctx], dim=-1)  # (1, d_in + d_out)
    ctx_b = ctx.to(device)

    # Measure SGD final loss
    sgd_loss = compute_loss(model, input_ids)
    print(f"    SGD loss: {sgd_loss:.4f}")

    # Restore original forward
    orig_forward.__self__.forward = orig_forward

    # 2. Flow-generated weights
    d_ctx = in_f + out_f
    flow = WeightFlowField(n_weights, d_latent=64, n_latents=16, d_context=d_ctx)
    flow.load_state_dict(torch.load("weight_flow_model.pt", map_location=device))
    flow.to(device).eval()

    with torch.no_grad():
        w = torch.zeros(1, n_weights, device=device)
        for s in range(STEPS):
            t = torch.tensor([s / (STEPS - 1)], device=device)
            v = flow(w, t, ctx_b)
            w = w + v / STEPS

    # Inject flow weights
    adapter_flow = PlainStreamExpert(in_f, out_f, RANK, 64, 64, 8.0 / RANK).to(device, dtype=torch.bfloat16)
    sidx = 0
    for n, p in adapter_flow.named_parameters():
        if 'lora_A' in n or 'lora_B' in n:
            ne = p.numel()
            p.data.copy_(w[0, sidx:sidx+ne].reshape(p.shape).to(torch.bfloat16))
            sidx += ne

    def fwd_flow(x):
        return orig_forward(x) + adapter_flow(x)
    orig_forward.__self__.forward = fwd_flow
    flow_loss = compute_loss(model, input_ids)
    orig_forward.__self__.forward = orig_forward
    print(f"    Flow loss: {flow_loss:.4f}")

    # 3. Zero weights
    adapter_zero = PlainStreamExpert(in_f, out_f, RANK, 64, 64, 8.0 / RANK).to(device, dtype=torch.bfloat16)
    for p in adapter_zero.parameters():
        if p.requires_grad:
            p.data.zero_()
    def fwd_zero(x):
        return orig_forward(x) + adapter_zero(x)
    orig_forward.__self__.forward = fwd_zero
    zero_loss = compute_loss(model, input_ids)
    orig_forward.__self__.forward = orig_forward
    print(f"    Zero loss: {zero_loss:.4f}")

    # Also compute SGD trajectory loss (loss achieved during training)
    sgd_train_losses = []
    orig_forward.__self__.forward = fwd_sgd
    for step in range(STEPS):
        sidx2 = 0
        for n, p in adapter.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:
                ne = p.numel()
                p.data.copy_(sgd_traj[step][sidx2:sidx2+ne].reshape(p.shape).to(torch.bfloat16))
                sidx2 += ne
        sgd_train_losses.append(compute_loss(model, input_ids))
    orig_forward.__self__.forward = orig_forward

    del adapter, adapter_flow, adapter_zero, flow, opt
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "sgd_final": sgd_loss,
        "flow": flow_loss,
        "zero": zero_loss,
        "sgd_trajectory": sgd_train_losses,
        "sgd_first": sgd_train_losses[0] if sgd_train_losses else 0,
    }


def main():
    print("=" * 60)
    print("  Weight Flow Evaluation on Qwen3.5-0.8B")
    print("=" * 60)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.to(DEVICE)

    # Find target layer
    target_name = "model.layers.3.self_attn.q_proj"
    target_module = model.get_submodule(target_name)
    orig_forward = target_module.forward
    print(f"Target: {target_name} ({target_module})")

    print(f"\nLoading {N_TEST} test texts...")
    texts = load_test_texts(N_TEST, TRAIN_OFFSET)
    print(f"  Loaded {len(texts)} texts")

    results = []
    for i, text in enumerate(texts):
        print(f"\n--- Test {i + 1}/{N_TEST} ---")
        r = eval_one_trajectory(model, text, target_name, orig_forward, DEVICE)
        r["text_id"] = TRAIN_OFFSET + i
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':>10s} {'Mean Loss':>10s} {'Std':>10s}")
    print(f"  {'-'*32}")

    for method, key in [("SGD final", "sgd_final"), ("Flow", "flow"), ("Zero", "zero"), ("SGD first", "sgd_first")]:
        vals = [r[key] for r in results]
        print(f"  {method:>10s}: {np.mean(vals):>8.4f} ± {np.std(vals):.4f}")

    # Flow improvement over zero
    zero_mean = np.mean([r["zero"] for r in results])
    flow_mean = np.mean([r["flow"] for r in results])
    sgd_mean = np.mean([r["sgd_final"] for r in results])
    print(f"\n  Flow improvement vs Zero: {zero_mean - flow_mean:.4f} ({((zero_mean - flow_mean)/zero_mean*100):.1f}%)")
    print(f"  Flow efficiency vs SGD:  {flow_mean / sgd_mean:.2f}x loss")
    print(f"  Flow beats zero: {sum(r['flow'] < r['zero'] for r in results)}/{len(results)}")

    # Save
    output = {
        "config": {"rank": RANK, "steps": STEPS, "n_test": N_TEST},
        "per_text": results,
        "summary": {
            "sgd_mean": sgd_mean,
            "flow_mean": flow_mean,
            "zero_mean": zero_mean,
        },
    }
    with open("weight_flow_eval.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to weight_flow_eval.json")


if __name__ == "__main__":
    main()
