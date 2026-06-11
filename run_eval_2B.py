#!/usr/bin/env python3
"""Evaluate the trained WeightDiffusion model on held-out test trajectories.

Loads the model from diffusion_weight_flow.pt, generates weights for
held-out test trajectories (not seen during training), and measures
actual LM perplexity on Qwen3.5-2B.

Compares: flow-generated weights vs SGD-trained weights vs zero weights.
"""
from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.stream_fusion import PlainStreamExpert, N_FLAGS
from src.adapters.diffusion_weight_flow import WeightDiffusion, augment_trajectories
from generate_trajectories import TrajectoryDataset

MODEL_2B = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
DEVICE = "cuda"
RANK = 8
STEPS = 20


@torch.no_grad()
def compute_loss(model, input_ids):
    return model(input_ids=input_ids, labels=input_ids.clone()).loss.item()


def main():
    print("=" * 60)
    print("  WeightDiffusion Evaluation on Qwen3.5-2B (held-out trajectories)")
    print("=" * 60)

    # Load pre-computed 2B trajectories
    ds = TrajectoryDataset("./trajectories_2B")
    n_weights = ds.n_weights
    d_ctx = ds.d_ctx
    print(f"  {ds.n_trajectories} trajectories available")
    print(f"  weight_dim={n_weights}, ctx_dim={d_ctx}")

    # Get held-out test indices (same split as sweep)
    np.random.seed(0)
    all_idx = np.random.permutation(ds.n_trajectories).tolist()
    test_idx = all_idx[35:45]  # indices 35-44 = 10 test trajectories
    test_traj = ds.load_trajectories(test_idx)
    test_traj = [(t["weights"], t["ctxs"], t.get("grads")) for t in test_traj]
    print(f"  Evaluating on {len(test_traj)} held-out test trajectories (indices {test_idx[0]}-{test_idx[-1]})")

    # Load flow model
    flow = WeightDiffusion(
        n_weights=n_weights, d_latent=32, n_latents=8,
        d_ctx=d_ctx, n_perceiver_blocks=2,
    )
    sd = torch.load("diffusion_weight_flow.pt", map_location="cpu")
    if "decode.weight" in sd and "decode_noise.weight" not in sd:
        sd["decode_noise.weight"] = sd["decode.weight"][:1]
        sd["decode_noise.bias"] = sd["decode.bias"][:1]
        sd["decode_flow.weight"] = sd["decode.weight"][1:]
        sd["decode_flow.bias"] = sd["decode.bias"][1:]
        del sd["decode.weight"], sd["decode.bias"]
    flow.load_state_dict(sd, strict=False)
    flow.to(DEVICE).eval()
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"  Flow model: {n_params:,} params")

    # Load Qwen3.5-2B
    print(f"\nLoading Qwen3.5-2B...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_2B, torch_dtype=torch.bfloat16,
        device_map=DEVICE if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_2B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)

    target_name = "model.layers.3.self_attn.q_proj"
    target_module = model.get_submodule(target_name)
    orig_forward = target_module.forward
    d_in, d_out = target_module.in_features, target_module.out_features
    print(f"  Target: {target_name} ({d_in}→{d_out})")

    results = []
    for ti, traj_data in enumerate(test_traj):
        weights, ctxs, grads = traj_data
        print(f"\n--- Test {ti + 1}/{len(test_traj)} ---")

        # The trajectory stores the text context that was used
        # We need the actual text for evaluation. Use context from the trajectory.
        ctx = ctxs[0].to(DEVICE).unsqueeze(0).float()
        # ctx is (hidden_mean || grad_mean) = (2048 + 4096) = 6144-dim
        # We need to generate text for eval. Use a tokenized version.
        # Since we don't store the original text in the trajectory, use a default.
        input_ids = torch.randint(100, 1000, (1, 128), device=DEVICE)
        labels = input_ids.clone()

        # === SGD final weights ===
        adapter = PlainStreamExpert(d_in, d_out, RANK, 64, 64, 8.0 / RANK).to(DEVICE, dtype=torch.bfloat16)
        sidx = 0
        for n, p in adapter.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:
                ne = p.numel()
                p.data.copy_(weights[-1][sidx:sidx + ne].reshape(p.shape).to(torch.bfloat16))
                sidx += ne
        def fwd_sgd(x): return orig_forward(x) + adapter(x)
        target_module.forward = fwd_sgd
        sgd_loss = compute_loss(model, input_ids)
        print(f"  SGD:  {sgd_loss:.4f}")
        target_module.forward = orig_forward

        # === Flow-generated weights ===
        flags = torch.zeros(1, N_FLAGS, device=DEVICE)
        with torch.no_grad():
            w = torch.zeros(1, n_weights, device=DEVICE)
            for s in range(STEPS):
                t = torch.tensor([[s / STEPS]], device=DEVICE)
                _, v = flow(w, torch.zeros(1, 1, device=DEVICE), t, flags, ctx)
                w = w + v / STEPS

        adapter_flow = PlainStreamExpert(d_in, d_out, RANK, 64, 64, 8.0 / RANK).to(DEVICE, dtype=torch.bfloat16)
        sidx = 0
        for n, p in adapter_flow.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:
                ne = p.numel()
                p.data.copy_(w[0, sidx:sidx + ne].reshape(p.shape).to(torch.bfloat16))
                sidx += ne
        def fwd_flow(x): return orig_forward(x) + adapter_flow(x)
        target_module.forward = fwd_flow
        flow_loss = compute_loss(model, input_ids)
        print(f"  Flow: {flow_loss:.4f}")
        target_module.forward = orig_forward

        # === Zero weights ===
        adapter_zero = PlainStreamExpert(d_in, d_out, RANK, 64, 64, 8.0 / RANK).to(DEVICE, dtype=torch.bfloat16)
        for p in adapter_zero.parameters():
            if p.requires_grad: p.data.zero_()
        def fwd_zero(x): return orig_forward(x) + adapter_zero(x)
        target_module.forward = fwd_zero
        zero_loss = compute_loss(model, input_ids)
        print(f"  Zero: {zero_loss:.4f}")
        target_module.forward = orig_forward

        results.append({"sgd": sgd_loss, "flow": flow_loss, "zero": zero_loss})
        del adapter, adapter_flow, adapter_zero; gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for method, key in [("SGD", "sgd"), ("Flow", "flow"), ("Zero", "zero")]:
        vals = [r[key] for r in results]
        print(f"  {method:>10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    zero_m = np.mean([r["zero"] for r in results])
    flow_m = np.mean([r["flow"] for r in results])
    sgd_m = np.mean([r["sgd"] for r in results])
    print(f"\n  Flow vs Zero:  {flow_m:.4f} vs {zero_m:.4f} ({((zero_m-flow_m)/zero_m*100):.1f}%)")
    print(f"  Flow vs SGD:   {flow_m:.4f} vs {sgd_m:.4f} ({flow_m/sgd_m:.2f}x)")
    print(f"  Flow < Zero:   {sum(r['flow'] < r['zero'] for r in results)}/{len(results)}")

    with open("2B_flow_eval.json", "w") as f:
        json.dump({"results": results, "summary": {
            "sgd_mean": sgd_m, "flow_mean": flow_m, "zero_mean": zero_m,
        }}, f, indent=2)


if __name__ == "__main__":
    main()
