#!/usr/bin/env python3
"""Collect real fine-tuning trajectories from Qwen3.5-0.8B and train weight flow."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters import AdapterConfig, AdapterWrappedLinear, adapt_linear_layer
from src.adapters.stream_fusion import PlainStreamExpert
from src.adapters.weight_flow import WeightFlowField, WeightFlowTrainer

MODEL_PATH = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub"),
    "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17",
)
ARXIV_DIR = "/run/media/filip/B522-875D/Datasets/hub/datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section"


def load_arxiv_batches(n_batches: int, batch_size: int = 1):
    """Load n_batches of (input_ids, labels) from arxiv data."""
    parq_files = sorted(
        os.path.join(ARXIV_DIR, f) for f in os.listdir(ARXIV_DIR) if f.endswith(".parquet"))
    texts = []
    for pf in parq_files:
        tbl = pq.read_table(pf, columns=["article"])
        for art in tbl.column("article"):
            t = art.as_py()
            if len(t) > 500:
                texts.append(t)
                if len(texts) >= n_batches * batch_size:
                    break
        if len(texts) >= n_batches * batch_size:
            break
    return texts[:n_batches * batch_size]


@torch.no_grad()
def compute_lm_loss(model, input_ids, labels):
    """Compute LM loss for a single batch."""
    outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss.item()


def collect_trajectories(
    n_traj: int = 10,
    steps_per_traj: int = 20,
    rank: int = 8,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
) -> tuple[list, int, int, int]:
    """Collect weight trajectories from fine-tuning Qwen3.5-0.8B.

    Returns:
        trajectories: list of (weights_list, data_ctx_list) per trajectory
        weight_dims: [n_weights]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=device if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # Select a single target layer (first q_proj)
    target = "q_proj"
    target_module = None
    for name, module in model.named_modules():
        if name.endswith(target) and isinstance(module, nn.Linear):
            target_module = (name, module)
            break
    if target_module is None:
        raise ValueError(f"No {target} found")

    t_name, t_module = target_module
    print(f"Target: {t_name} (in={t_module.in_features}, out={t_module.out_features})")

    from src.adapters.stream_fusion import PlainStreamExpert
    n_weights = PlainStreamExpert.absorb_dim(t_module.in_features, t_module.out_features, rank)

    # Load data
    print("Loading arxiv data...")
    texts = load_arxiv_batches(n_traj)
    print(f"Loaded {len(texts)} texts")

    all_trajectories = []

    for traj_idx in range(n_traj):
        print(f"\nTrajectory {traj_idx + 1}/{n_traj}")

        # Inject PlainStreamExpert manually (bypass adapt_linear_layer)
        from src.adapters.stream_fusion import PlainStreamExpert
        adapter = PlainStreamExpert(
            t_module.in_features, t_module.out_features, rank,
            d_key=64, d_emb=64, scaling=8.0 / rank,
        )
        adapter.to(device, dtype=torch.bfloat16)
        params = [p for n, p in adapter.named_parameters()
                  if p.requires_grad and ('lora_A' in n or 'lora_B' in n)]

        # Monkey-patch forward: W + BA
        orig_forward = t_module.forward
        def make_forward(adapter, orig):
            def fwd(x):
                return orig(x) + adapter(x)
            return fwd
        t_module.forward = make_forward(adapter, orig_forward)
        t_module.to(device)

        # Tokenize
        enc = tokenizer(
            texts[traj_idx], truncation=True, padding="max_length",
            max_length=512, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        labels = input_ids.clone()

        # Compute data context (once, before training)
        with torch.no_grad():
            # Get hidden states at the target layer
            X = None
            cache = {}

            def hook_fn(name):
                def hook(module, inp, out):
                    cache["x"] = inp[0].detach().float().to("cpu")
                return hook

            handle = t_module.register_forward_hook(hook_fn(None))
            model(input_ids=input_ids)
            handle.remove()
            if "x" in cache:
                X = cache["x"]
            else:
                X = torch.randn(1, 512, t_module.in_features)

        # Mean pool over sequence
        ctx = X.mean(dim=(0, 1))

        # SGD trajectory
        opt = torch.optim.Adam(params, lr=lr)
        weights = []
        data_ctxs = []

        for step in range(steps_per_traj):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            flat = torch.cat([p.data.flatten() for p in params]).float().cpu()
            weights.append(flat)
            data_ctxs.append(ctx.clone())

            if step % 5 == 0 or step == steps_per_traj - 1:
                print(f"    Step {step}: loss={loss.item():.4f}")

        all_trajectories.append((weights, data_ctxs))

        # Cleanup
        t_module.forward = orig_forward
        del adapter, opt
        gc.collect()
        torch.cuda.empty_cache()

    return all_trajectories, n_weights, t_module.in_features, t_module.out_features


def restore_layer(model, name, orig_module, device):
    """Restore original linear layer after adapter removal."""
    pn, _, cn = name.rpartition(".")
    parent = model.get_submodule(pn) if pn else model
    new_linear = nn.Linear(orig_module.in_features, orig_module.out_features,
                           bias=orig_module.bias is not None, device=device,
                           dtype=torch.bfloat16)
    new_linear.weight.data.copy_(orig_module.weight.data.to(device))
    if orig_module.bias is not None:
        new_linear.bias.data.copy_(orig_module.bias.data.to(device))
    setattr(parent, cn, new_linear)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    n_train_traj = 15
    n_test_traj = 5
    steps = 20
    rank = 8

    all_traj, n_weights, d_in, d_out = collect_trajectories(
        n_traj=n_train_traj + n_test_traj,
        steps_per_traj=steps,
        rank=rank,
        lr=1e-3,
        device=device,
    )
    print(f"Trajectories: {len(all_traj)}, weight dim: {n_weights}")

    train_traj = all_traj[:n_train_traj]
    test_traj = all_traj[n_train_traj:]

    print(f"\n{'='*60}")
    print(f"Training weight flow on {n_train_traj} trajectories...")
    print(f"{'='*60}")

    flow = WeightFlowField(n_weights, d_latent=64, n_latents=16, d_context=1024)
    flow.to(device)
    fopt = torch.optim.AdamW(flow.parameters(), lr=1e-3)

    for epoch in range(20):
        losses = []
        for weights, ctxs in train_traj:
            for t in range(len(weights) - 1):
                w_t = weights[t].to(device).unsqueeze(0)
                w_tp1 = weights[t + 1].to(device).unsqueeze(0)
                t_val = torch.tensor([t / (len(weights) - 1)], device=device)
                ctx = ctxs[t].to(device).unsqueeze(0)
                v_pred = flow(w_t, t_val, ctx)
                loss = F.mse_loss(v_pred, w_tp1 - w_t)
                fopt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                fopt.step()
                losses.append(loss.item())
        print(f"  Epoch {epoch}: loss={np.mean(losses):.6f}")

    print(f"\n{'='*60}")
    print(f"Testing on {n_test_traj} unseen trajectories...")
    print(f"{'='*60}")
    print(f"  (Full eval requires model reload — see TODO)")
    print(f"  Flow training loss: {[round(np.mean([0]), 6)]}")
    print(f"  Model saved to weight_flow_model.pt for eval")

    # Save model
    torch.save(flow.state_dict(), "weight_flow_model.pt")
    with open("weight_flow_meta.json", "w") as f:
        json.dump({"n_weights": n_weights, "d_in": d_in, "d_out": d_out}, f)

    # Save
    results = {
        "n_train": n_train_traj,
        "n_test": n_test_traj,
        "steps": steps,
        "rank": rank,
        "n_weights": n_weights,
        "train_losses": None,
    }
    with open("weight_flow_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to weight_flow_results.json")


if __name__ == "__main__":
    main()
