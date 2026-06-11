#!/usr/bin/env python3
"""Compare flow-predicted velocity vs SVD-computed optimal update vs SGD."""
from __future__ import annotations

import gc
import json
import sys
import os

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.stream_fusion import PlainStreamExpert, N_FLAGS
from src.adapters.diffusion_weight_flow import WeightDiffusion, DiffusionFlowTrainer
from src.adapters.flow_weight_expert import compute_closed_form_lora

MODEL_PATH = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub"),
    "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17",
)
ARXIV_DIR = "/run/media/filip/B522-875D/Datasets/hub/datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section"
DEVICE = "cuda"
RANK = 8
STEPS = 20


def load_test_texts(n: int) -> list[str]:
    texts = []
    for pf in sorted(os.listdir(ARXIV_DIR))[:5]:
        tbl = pq.read_table(os.path.join(ARXIV_DIR, pf), columns=["article"])
        for art in tbl.column("article"):
            t = art.as_py()
            if len(t) > 500:
                texts.append(t[:2000])
                if len(texts) >= n:
                    return texts
    return texts


def main():
    print("=" * 60)
    print("  Flow vs SVD vs SGD: Weight Velocity Comparison")
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
    n_weights_flat = d_in * d_out  # for SVD (full matrix, not factorized)

    # Load diffusion model (trained without optimal target, for comparison)
    flow = WeightDiffusion(n_weights, d_latent=64, n_latents=16, d_ctx=d_ctx)
    flow.load_state_dict(torch.load("diffusion_weight_flow.pt", map_location="cpu"))
    flow.to(DEVICE).eval()

    print(f"Target: {target_name} ({d_in}→{d_out})")
    print(f"  Weight dim (factorized): {n_weights}")
    print(f"  Weight dim (full SVD):   {n_weights_flat}")

    texts = load_test_texts(5)
    results = []

    for ti, text in enumerate(texts):
        print(f"\n--- Text {ti + 1} ---")
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        labels = input_ids.clone()

        # Capture X (hidden states at q_proj input)
        cache = {}
        def fwd_hook(m, inp, out): cache["x"] = inp[0].detach().float().mean(dim=1)
        handle = target_module.register_forward_hook(fwd_hook)
        model(input_ids=input_ids)
        handle.remove()
        X_mean = cache.get("x", torch.zeros(1, d_in))

        # Run SGD, compare velocities at each step
        adapter = PlainStreamExpert(d_in, d_out, RANK, 64, 64, 8.0 / RANK).to(DEVICE, dtype=torch.bfloat16)
        def fwd_sgd(x): return orig_forward(x) + adapter(x)
        target_module.forward = fwd_sgd

        params = [p for n, p in adapter.named_parameters() if 'lora_A' in n or 'lora_B' in n]
        opt = torch.optim.Adam(params, lr=1e-3)

        for step in range(STEPS):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            # ---- Collect velocities from 3 methods ----

            # 1. SGD velocity (actual update applied)
            with torch.no_grad():
                B, A = params[1].data, params[0].data  # B_fwd, A_fwd
                sgd_update = torch.cat([p.flatten() for p in params])
                # Also compute full-matrix SGD update: ΔW = B·A - previous_BA
                sgd_delta_full = (B @ A).flatten()

            # 2. SVD optimal update: SVD(R·X⁺) where R = -gradient signal
            # We need X (not mean) for the full SVD. Capture it.
            cache_x_full = {}
            def hook_x(m, inp, out): cache_x_full["x"] = inp[0].detach().float()
            handle_x = target_module.register_forward_hook(hook_x)
            model(input_ids=input_ids)
            handle_x.remove()
            X_full = cache_x_full.get("x", torch.randn(1, 512, d_in))

            # Target Y for q_proj: the output WITHOUT adapter
            with torch.no_grad():
                Y_base = orig_forward(X_full.to(device=DEVICE, dtype=torch.bfloat16))

            # Compute closed-form optimal rank-r update
            A_opt, B_opt = compute_closed_form_lora(
                X_full.float(), Y_base.float(),
                target_module.weight.float() if hasattr(target_module, 'weight') else torch.zeros(d_out, d_in),
                RANK,
            )
            svd_optimal_full = torch.cat([A_opt.flatten(), B_opt.flatten()])

            # 3. Flow-predicted velocity
            with torch.no_grad():
                # Build context
                grad_b = torch.zeros(1, d_out)
                for n, p in adapter.named_parameters():
                    if 'lora_B' in n and p.grad is not None:
                        grad_b = p.grad.float().mean(dim=1).unsqueeze(0).cpu()
                        break
                ctx = torch.cat([X_mean.cpu(), grad_b], dim=-1).float().to(DEVICE)
                flags = torch.zeros(1, N_FLAGS, device=DEVICE)
                current_w = torch.cat([p.data.flatten() for p in params]).unsqueeze(0).float().to(DEVICE)
                t = torch.tensor([[step / max(STEPS - 1, 1)]], device=DEVICE)
                _, v_pred = flow(current_w, torch.zeros(1, 1, device=DEVICE), t, flags, ctx)
                flow_velocity = v_pred[0]

            # Compare cosine similarities
            def cos(a, b):
                a_f, b_f = a.float().flatten(), b.float().flatten()
                return (a_f @ b_f / (a_f.norm() * b_f.norm() + 1e-8)).item()

            from time import sleep
            sleep(0.001)  # yield

            if step in [0, 5, 10, 19]:
                print(f"  Step {step}:")
                print(f"    SGD→SVD cos: {cos(sgd_update, svd_optimal_full):.4f}")
                print(f"    Flow→SVD cos: {cos(flow_velocity, svd_optimal_full):.4f}")
                print(f"    Flow→SGD cos: {cos(flow_velocity, sgd_update):.4f}")

        target_module.forward = orig_forward
        del adapter, opt; gc.collect(); torch.cuda.empty_cache()
        break  # just one text for now

    print("\n  Comparison complete.")


if __name__ == "__main__":
    main()
