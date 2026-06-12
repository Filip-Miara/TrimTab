#!/usr/bin/env python3
"""Generate layer-to-layer thought trajectories (single forward pass).

Records hidden states at each layer for last token during model forward pass.
Fast: processes texts in batches for efficiency.
~60 trajectories/second on RTX 4060 (after model load).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATHS = {
    "0.8B": os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
        "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"),
    "2B": "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
}
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
ARXIV_DIR = os.path.join(CACHE_DIR,
    "datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section")
BATCH_SIZE = 8
MAX_LENGTH = 512


def load_diverse_texts(n_total: int) -> list[str]:
    texts = []
    np.random.seed(42)
    try:
        import pyarrow.parquet as pq
        for pf in sorted(os.listdir(ARXIV_DIR))[:10]:
            tbl = pq.read_table(os.path.join(ARXIV_DIR, pf), columns=["article"])
            for art in tbl.column("article"):
                t = art.as_py()
                if len(t) > 500:
                    texts.append(t[:2000])
        print(f"  Arxiv: {len(texts)} texts")
    except Exception as e:
        print(f"  Arxiv: FAIL ({e})")

    for ds_name, split, text_fn, n_max in [
        ("databricks/databricks-dolly-15k", "train",
         lambda r: f"{r.get('instruction', '')}\n{r.get('response', '')}", n_total // 4),
        ("openai/gsm8k", "main",
         lambda r: f"Q: {r['question']}\nA: {r['answer']}", n_total // 4),
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
            print(f"  {ds_name.split('/')[-1]}: {c} texts")
        except Exception as e:
            print(f"  {ds_name.split('/')[-1]}: FAIL ({e})")

    np.random.shuffle(texts)
    print(f"  Total: {len(texts)} texts available")
    return texts[:n_total]


def main():
    parser = argparse.ArgumentParser(description="Generate layer-to-layer thought trajectories")
    parser.add_argument("--model", choices=["0.8B", "2B"], default="2B")
    parser.add_argument("--n-traj", type=int, default=5000)
    parser.add_argument("--output", type=str, default="./thought_trajs_5k/")
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model]
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    n_texts = int(args.n_traj * 1.2) + 10
    print(f"Loading {n_texts} texts...")
    texts = load_diverse_texts(n_texts)

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    text_cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    n_layers = getattr(text_cfg, 'num_hidden_layers', 24)
    d_model = getattr(text_cfg, 'hidden_size', 2048)
    print(f"  Layers: {n_layers}, d_model: {d_model}")

    # Process texts in batches
    trajectories = []
    t0 = time.time()
    batch_texts = []

    for idx, text in enumerate(texts):
        if len(trajectories) >= args.n_traj:
            break

        batch_texts.append(text)

        if len(batch_texts) < BATCH_SIZE and idx < len(texts) - 1:
            continue

        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                          max_length=MAX_LENGTH, padding=True).to("cuda")

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        hs = out.hidden_states
        for b in range(inputs.input_ids.shape[0]):
            seq_len = (inputs.attention_mask[b] != 0).sum().item()
            if seq_len < 10:
                continue
            last_pos = seq_len - 1
            traj = torch.stack([h[b, last_pos, :].cpu().float() for h in hs], dim=0)
            trajectories.append(traj)

        batch_texts = []

        if len(trajectories) % 500 == 0 and len(trajectories) > 0:
            elapsed = time.time() - t0
            rate = len(trajectories) / elapsed
            eta = (args.n_traj - len(trajectories)) / rate if rate > 0 else 0
            print(f"  {len(trajectories)}/{args.n_traj} | {elapsed:.0f}s | "
                  f"{rate:.1f} traj/s | ETA {eta:.0f}s")
            gc.collect()
            torch.cuda.empty_cache()

    # Save
    meta = {
        "n_trajectories": len(trajectories),
        "n_layers": n_layers,
        "d_model": d_model,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": round(time.time() - t0, 2),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for i, traj in enumerate(trajectories):
        torch.save(traj, os.path.join(output_dir, f"traj_{i:04d}.pt"))

    elapsed = time.time() - t0
    print(f"\nDone: {len(trajectories)} trajectories in {elapsed:.1f}s ({len(trajectories)/elapsed:.1f}/s)")
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()
