#!/usr/bin/env python3
"""Generate reasoning-step thought trajectories.

A "reasoning-step trajectory" records the final-layer hidden state of the
LAST token after each generated token during model inference.

For a model generating T tokens, the trajectory = [h_token_1, ..., h_token_T]
where each h is the (d_model,) vector at the final layer for the last position.
Flow matching learns: h[t] → h[t+1] (predict next step's hidden state).

Usage:
  python3 generate_thought_trajectories.py --model 2B --n-traj 500 --output ./reasoning_trajectories/
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
    "0.8B": os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub"),
        "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17",
    ),
    "2B": "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
}

CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
ARXIV_DIR = os.path.join(CACHE_DIR,
    "datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section")

DEVICE = "cuda"
MAX_LENGTH = 256
MAX_GEN_TOKENS = 48


def load_diverse_texts(n_total: int, max_len: int = 200) -> list[str]:
    texts = []
    np.random.seed(42)
    try:
        import pyarrow.parquet as pq
        for pf in sorted(os.listdir(ARXIV_DIR))[:3]:
            tbl = pq.read_table(os.path.join(ARXIV_DIR, pf), columns=["article"])
            for art in tbl.column("article"):
                t = art.as_py()
                if 80 < len(t) < max_len:
                    texts.append(t)
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
                if 80 < len(t) < max_len:
                    texts.append(t)
                    c += 1
                    if c >= n_max:
                        break
            print(f"  {ds_name.split('/')[-1]}: {c} texts")
        except Exception as e:
            print(f"  {ds_name.split('/')[-1]}: FAIL ({e})")

    np.random.shuffle(texts)
    return texts[:n_total]


def generate_trajectories(
    model_path: str, texts: list[str], n_traj: int,
    max_length: int = 256, max_gen: int = 48,
) -> dict:
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    model.eval()

    text_cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    n_layers = getattr(text_cfg, 'num_hidden_layers', 24)
    d_model = getattr(text_cfg, 'hidden_size', 2048)
    print(f"  Layers: {n_layers}, d_model: {d_model}")

    trajectories, n_skipped = [], []
    t0 = time.time()

    for idx, text in enumerate(texts):
        if len(trajectories) >= n_traj:
            break

        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_length, padding=False).to(DEVICE)
        if inputs.input_ids.shape[1] < 5:
            continue

        # Generate tokens and record hidden states at each step
        hidden_seq = []
        input_len = inputs.input_ids.shape[1]
        past = None

        for gen_step in range(max_gen):
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=True,
                          past_key_values=past)

            # Record final-layer hidden state of last token
            hs = out.hidden_states
            h_final = hs[-1][0, -1, :].cpu().float()
            hidden_seq.append(h_final)

            # Sample next token
            next_logits = out.logits[0, -1, :]
            next_token = torch.multinomial(torch.softmax(next_logits / 0.8, dim=-1), 1)
            if next_token.item() == tokenizer.eos_token_id:
                break

            inputs = {"input_ids": next_token.unsqueeze(0)}
            past = out.past_key_values

        if len(hidden_seq) >= 5:
            trajectories.append(torch.stack(hidden_seq))
        else:
            n_skipped.append(idx)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(texts)}] {len(trajectories)}/{n_traj} "
                  f"(+{len(n_skipped)} skip) {elapsed:.1f}s")

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Done: {len(trajectories)} trajectories in {elapsed:.1f}s")
    return {"trajectories": trajectories, "n_layers": n_layers, "d_model": d_model}


def save_trajectories(result: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    meta = {
        "n_trajectories": len(result["trajectories"]),
        "d_model": result["d_model"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for i, traj in enumerate(result["trajectories"]):
        torch.save(traj, os.path.join(output_dir, f"traj_{i:04d}.pt"))
    print(f"  Saved {len(result['trajectories'])} trajectories to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["0.8B", "2B"], default="2B")
    parser.add_argument("--n-traj", type=int, default=500)
    parser.add_argument("--output", type=str, default="./reasoning_trajectories/")
    parser.add_argument("--max-gen", type=int, default=48)
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model]
    texts = load_diverse_texts(args.n_traj * 2)
    result = generate_trajectories(model_path, texts, args.n_traj,
                                   max_gen=args.max_gen)
    save_trajectories(result, args.output)


if __name__ == "__main__":
    main()
