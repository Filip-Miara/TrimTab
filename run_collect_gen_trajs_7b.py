#!/usr/bin/env python3 -u
"""Collect generation trajectories from Qwen2.5-7B-Instruct.

Architecture: 28 layers, 3584 hidden, 4 KV heads, standard MHA.
Batches written to disk every 5000 tokens to bound memory.
"""
from __future__ import annotations

import gc, glob, json, os, re, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEVICE = "cuda"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
OUTPUT_DIR = "/run/media/filip/B522-875D/Datasets/project_data"
MAX_GEN = 200
D_MODEL = 3584
N_LAYERS = 28
BATCH_SIZE = 1000
N_PROBLEMS = 500
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"So answer is\s*(-?\d+)", r"Therefore,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def save_batch(hidden_list, vel_list, token_list, correct_list, batch_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"gen_trajs_7b_batch_{batch_idx:04d}.pt")
    data = {
        "hidden_seqs": torch.stack(hidden_list).half(),
        "velocity_targets": torch.stack(vel_list).half(),
        "token_ids": torch.tensor(token_list),
        "is_correct": torch.tensor(correct_list),
    }
    torch.save(data, path)
    print(f"  Saved batch {batch_idx}: {len(hidden_list)} tokens -> {path} "
          f"({os.path.getsize(path)/1e6:.0f}MB)", flush=True)
    hidden_list.clear(); vel_list.clear(); token_list.clear(); correct_list.clear()
    gc.collect()
    return batch_idx + 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-problems", type=int, default=N_PROBLEMS)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    print("Loading Qwen2.5-7B-Instruct (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M params", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_problems]
    print(f"  {len(problems)} problems", flush=True)

    n_tokens, batch_idx, n_correct_problems = 0, 0, 0
    hidden_list, vel_list, token_list, correct_list = [], [], [], []
    t0 = time.time()

    for idx, prob in enumerate(problems):
        msgs = [{'role': 'user', 'content': f'Q: {prob["question"]}\nA:'}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        correct_ans = parse_answer(prob["answer"])
        input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids
        past = None
        generated_tokens = []

        for step in range(MAX_GEN):
            with torch.no_grad():
                fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)
            hs = fwd.hidden_states
            next_tok = fwd.logits[0, -1, :].argmax().item()
            generated_tokens.append(next_tok)

            # Record layer trajectory: h[0..27] (28 states), compute v[l] = h[l+1] - h[l]
            h_at_layers = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS + 1]], dim=0)
            hidden_seq = h_at_layers[:N_LAYERS]   # (28, 3584)
            velocity_target = h_at_layers[1:N_LAYERS + 1] - h_at_layers[:N_LAYERS]  # (28, 3584)

            hidden_list.append(hidden_seq.cpu().half())
            vel_list.append(velocity_target.cpu().half())
            token_list.append(next_tok)
            n_tokens += 1

            if next_tok == tok.eos_token_id:
                break
            past = fwd.past_key_values
            input_ids = torch.tensor([[next_tok]], device=DEVICE)

        gen_text = tok.decode(generated_tokens, skip_special_tokens=True)
        predicted = parse_answer(gen_text)
        is_correct = (predicted is not None and correct_ans is not None and predicted == correct_ans)
        if is_correct: n_correct_problems += 1
        for _ in generated_tokens: correct_list.append(is_correct)

        if len(hidden_list) >= BATCH_SIZE:
            batch_idx = save_batch(hidden_list, vel_list, token_list, correct_list, batch_idx,
                                   os.path.join(args.output_dir, "qwen25_7b_gen_trajs"))

        if (idx + 1) % 50 == 0:
            rate = n_tokens / (time.time() - t0)
            print(f"  [{idx+1}/{len(problems)}] {n_tokens} tokens ({rate:.0f} tok/s) "
                  f"acc={n_correct_problems}/{idx+1} ({100*n_correct_problems/(idx+1):.0f}%) "
                  f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    if hidden_list:
        batch_idx = save_batch(hidden_list, vel_list, token_list, correct_list, batch_idx,
                               os.path.join(args.output_dir, "qwen25_7b_gen_trajs"))

    print(f"\n{'='*60}", flush=True)
    print(f"Collection complete!", flush=True)
    print(f"  Tokens: {n_tokens}, Batches: {batch_idx}", flush=True)
    print(f"  Gen accuracy: {100*n_correct_problems/len(problems):.1f}%", flush=True)
    print(f"  Time: {time.time()-t0:.0f}s", flush=True)

    meta = {"n_tokens": n_tokens, "n_batches": batch_idx, "n_problems": len(problems),
            "d_model": D_MODEL, "n_layers": N_LAYERS, "model": MODEL_NAME,
            "accuracy": n_correct_problems / len(problems)}
    with open(os.path.join(args.output_dir, "qwen25_7b_gen_trajs_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved", flush=True)


if __name__ == "__main__":
    main()
