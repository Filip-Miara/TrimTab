#!/usr/bin/env python3
"""Pre-generate and persistently store training trajectories from a model.

Usage:
  python3 generate_trajectories.py --model 2B --n-traj 100 --output ./trajectories/
  python3 generate_trajectories.py --model 0.8B --n-traj 50 --output ./trajectories/

The saved dataset can be loaded by any training script via TrajectoryDataset.load().
This avoids re-generating trajectories for every experiment run.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.stream_fusion import PlainStreamExpert


# ── Model paths ──────────────────────────────────────────────────────────────

MODEL_PATHS = {
    "0.8B": os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub"),
        "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17",
    ),
    "2B": "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
}

CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
ARXIV_DIR = os.path.join(
    CACHE_DIR,
    "datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section",
)

DEVICE = "cuda"
RANK = 8
STEPS = 20
MAX_LENGTH = 512


def load_diverse_texts(n_total: int) -> list[str]:
    """Load diverse texts from multiple datasets."""
    texts, sources = [], []
    np.random.seed(42)

    # Arxiv
    try:
        import pyarrow.parquet as pq
        c = 0
        for pf in sorted(os.listdir(ARXIV_DIR))[:5]:
            tbl = pq.read_table(os.path.join(ARXIV_DIR, pf), columns=["article"])
            for art in tbl.column("article"):
                t = art.as_py()
                if len(t) > 500:
                    texts.append(t[:2000])
                    c += 1
        sources.extend(["arxiv"] * c)
        print(f"  Arxiv: {c} texts")
    except Exception as e:
        print(f"  Arxiv: FAILED ({e})")

    # HuggingFace datasets
    for ds_name, split, text_fn, n_max in [
        ("databricks/databricks-dolly-15k", "train",
         lambda r: f"{r.get('instruction', '')}\n{r.get('response', '')}", n_total // 4),
        ("fancyzhx/ag_news", "train",
         lambda r: r.get("text", ""), n_total // 4),
        ("openai/gsm8k", "main",
         lambda r: f"Q: {r['question']}\nA: {r['answer']}", n_total // 4),
        ("openai/openai_humaneval", "test",
         lambda r: f"Problem: {r['prompt']}\nSolution: {r.get('canonical_solution', '')}", n_total // 4),
    ]:
        try:
            ds = load_dataset(ds_name, split=split, cache_dir=CACHE_DIR)
            c = 0
            for r in ds:
                t = text_fn(r)
                if len(t) > 500:
                    texts.append(t[:2000])
                    c += 1
                    if c >= n_max:
                        break
            short = ds_name.split("/")[-1] if "/" in ds_name else ds_name
            sources.extend([short] * c)
            print(f"  {short}: {c} texts")
        except Exception as e:
            print(f"  {short}: FAILED ({e})")

    print(f"  Total: {len(texts)} texts from {len(set(sources))} sources")
    return texts


def generate_trajectories(
    model_key: str,
    n_traj: int,
    output_dir: str,
    rank: int = 8,
    steps: int = 20,
) -> dict:
    """Generate trajectories and save to disk.

    Each trajectory is saved as output_dir/traj_{i:04d}.pt containing:
        weights: list of flat parameter tensors
        ctxs: list of (1, d_ctx) context tensors
        grads: list of flat gradient tensors
        source: str — dataset source
        metadata: dict — step losses, etc.

    Returns metadata dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading model {model_key} ({MODEL_PATHS[model_key]})...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS[model_key], torch_dtype=torch.bfloat16,
        device_map=DEVICE if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[model_key])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)

    # Target layer
    target_name = "model.layers.3.self_attn.q_proj"
    target_module = model.get_submodule(target_name)
    orig_forward = target_module.forward
    d_in, d_out = target_module.in_features, target_module.out_features
    n_weights = PlainStreamExpert.absorb_dim(d_in, d_out, rank)
    d_ctx = d_in + d_out
    print(f"Target: {target_name} ({d_in}→{d_out}), weight_dim={n_weights}, ctx_dim={d_ctx}")

    # Load texts
    print(f"\nLoading {n_traj} texts...")
    texts = load_diverse_texts(n_traj)
    print(f"Using {len(texts)} texts for {n_traj} trajectories")

    t_start = time.time()
    for traj_idx in range(n_traj):
        text = texts[traj_idx % len(texts)]
        t0 = time.time()
        print(f"\nTrajectory {traj_idx + 1}/{n_traj}", end=" ", flush=True)

        adapter = PlainStreamExpert(d_in, d_out, rank, 64, 64, 8.0 / rank).to(DEVICE, dtype=torch.bfloat16)
        def make_fwd(ada, orig):
            def fwd(x): return orig(x) + ada(x)
            return fwd
        target_module.forward = make_fwd(adapter, orig_forward)

        params = [p for n, p in adapter.named_parameters() if 'lora_A' in n or 'lora_B' in n]
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        labels = input_ids.clone()

        # Capture hidden states
        cache = {}
        def fwd_hook(m, inp, out): cache["x"] = inp[0].detach().float().mean(dim=1)
        handle = target_module.register_forward_hook(fwd_hook)
        model(input_ids=input_ids)
        handle.remove()
        ctx_hidden = cache.get("x", torch.zeros(1, d_in))

        opt = torch.optim.Adam(params, lr=1e-3)
        weights, ctxs, grads, losses = [], [], [], []

        for step in range(steps):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if step == 0:
                grad_ctx = torch.zeros(1, d_out)
                for n, p in adapter.named_parameters():
                    if 'lora_B' in n and p.grad is not None:
                        grad_ctx = p.grad.float().mean(dim=1).unsqueeze(0).cpu()
                        break

            flat_w = torch.cat([p.data.flatten() for p in params]).float().cpu()
            flat_g = torch.cat([p.grad.flatten() for p in params]).float().cpu()
            weights.append(flat_w)
            ctxs.append(torch.cat([ctx_hidden.cpu(), grad_ctx], dim=-1).float())
            grads.append(flat_g)

        target_module.forward = orig_forward
        del adapter, opt; gc.collect(); torch.cuda.empty_cache()

        # Save this trajectory — now includes input_ids and original text
        traj_data = {
            "weights": weights,
            "ctxs": ctxs,
            "grads": grads,
            "losses": losses,
            "source": "mixed",
            "input_ids": enc["input_ids"].cpu(),  # store for evaluation
            "text": text,                         # store original text
            "metadata": {
                "model": model_key,
                "rank": rank,
                "d_in": d_in,
                "d_out": d_out,
                "n_weights": n_weights,
                "d_ctx": d_ctx,
                "steps": steps,
                "text_preview": text[:100],
            },
        }
        torch.save(traj_data, os.path.join(output_dir, f"traj_{traj_idx:04d}.pt"))

        dt = time.time() - t0
        avg_loss = np.mean(losses[-5:])
        print(f"loss={avg_loss:.2f} ({dt:.0f}s)", flush=True)

    total_time = time.time() - t_start
    meta = {
        "model": model_key,
        "model_path": MODEL_PATHS[model_key],
        "n_trajectories": n_traj,
        "steps_per_traj": steps,
        "rank": rank,
        "d_in": d_in,
        "d_out": d_out,
        "n_weights": n_weights,
        "d_ctx": d_ctx,
        "max_length": MAX_LENGTH,
        "target_layer": target_name,
        "total_time_s": total_time,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Generated {n_traj} trajectories in {total_time:.0f}s")
    print(f"  Saved to {output_dir}/")
    print(f"  Weight dim: {n_weights}, Context dim: {d_ctx}")
    print(f"{'='*60}")

    return meta


class TrajectoryDataset:
    """Loads pre-computed trajectories from disk for training."""

    def __init__(self, traj_dir: str):
        self.traj_dir = Path(traj_dir)
        with open(self.traj_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.n_trajectories = self.meta["n_trajectories"]
        self.n_weights = self.meta["n_weights"]
        self.d_ctx = self.meta["d_ctx"]

    def get_trajectory(self, idx: int) -> dict:
        data = torch.load(self.traj_dir / f"traj_{idx:04d}.pt", map_location="cpu")
        return data

    def get_split(self, n_train: int, indices: list[int] | None = None) -> tuple[list, list]:
        if indices is not None:
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
        else:
            all_idx = list(range(self.n_trajectories))
            np.random.shuffle(all_idx)
            train_idx = all_idx[:n_train]
            test_idx = all_idx[n_train:]
        return train_idx, test_idx

    def load_trajectories(self, indices: list[int]) -> list:
        return [self.get_trajectory(i) for i in indices]

    def load_all(self) -> list:
        return [self.get_trajectory(i) for i in range(self.n_trajectories)]


def main():
    parser = argparse.ArgumentParser(description="Pre-generate and store training trajectories")
    parser.add_argument("--model", type=str, default="0.8B", choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--n-traj", type=int, default=50)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=str, default="./trajectories/")
    args = parser.parse_args()

    generate_trajectories(args.model, args.n_traj, args.output, args.rank, args.steps)


if __name__ == "__main__":
    main()
