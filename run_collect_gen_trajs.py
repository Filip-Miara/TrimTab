#!/usr/bin/env python3 -u
"""Collect generation-time hidden state trajectories on SmolLM2-360M.

Memory-efficient: writes batches to disk periodically to avoid OOM.
Output unbuffered: (-u flag) shows progress in real-time.
"""
from __future__ import annotations

import gc, glob, json, os, re, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
OUTPUT_DIR = "/run/media/filip/B522-875D/Datasets/project_data"
MAX_GEN = 100
D_MODEL = 960
N_LAYERS = 32
BATCH_SIZE = 5000  # write to disk every 5000 tokens


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def save_batch(hidden_list, vel_list, token_list, correct_list, batch_idx, output_dir):
    """Save a batch of trajectories to disk and clear lists."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"gen_trajs_batch_{batch_idx:04d}.pt")
    data = {
        "hidden_seqs": torch.stack(hidden_list).half(),       # (B, 32, 960) float16
        "velocity_targets": torch.stack(vel_list).half(),      # (B, 32, 960) float16
        "token_ids": torch.tensor(token_list),
        "is_correct": torch.tensor(correct_list),
    }
    torch.save(data, path)
    print(f"  Saved batch {batch_idx}: {len(hidden_list)} tokens -> {path} "
          f"({os.path.getsize(path)/1e6:.0f}MB)", flush=True)
    hidden_list.clear()
    vel_list.clear()
    token_list.clear()
    correct_list.clear()
    gc.collect()
    return batch_idx + 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-problems", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    model_dir = os.path.join(CACHE_DIR, "models--HuggingFaceTB--SmolLM2-360M", "snapshots")
    snaps = glob.glob(model_dir + "/*/")
    if not snaps:
        raise FileNotFoundError(f"No SmolLM2-360M at {model_dir}")
    model_path = snaps[0]

    print("Loading SmolLM2-360M...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M params", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_problems]
    print(f"  {len(problems)} problems", flush=True)

    # Accumulate in lists, write to disk periodically
    hidden_list, vel_list = [], []
    token_list, correct_list = [], []
    n_tokens, batch_idx = 0, 0
    n_correct_problems = 0
    t0 = time.time()

    for idx, prob in enumerate(problems):
        prompt = f"Q: {prob['question']}\nA:"
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

            # Layer trajectory: (N+1, D) -> (N, D) hidden + (N, D) velocity
            h_at_layers = torch.stack([h[0, -1, :].float() for h in hs], dim=0)
            hidden_seq = h_at_layers[:N_LAYERS]  # (32, 960)
            velocity_target = h_at_layers[1:N_LAYERS+1] - h_at_layers[:N_LAYERS]

            hidden_list.append(hidden_seq.cpu())
            vel_list.append(velocity_target.cpu())
            token_list.append(next_tok)
            n_tokens += 1

            if next_tok == tok.eos_token_id:
                break

            past = fwd.past_key_values
            input_ids = torch.tensor([[next_tok]], device=DEVICE)

        # Track correctness
        gen_text = tok.decode(generated_tokens, skip_special_tokens=True)
        predicted = parse_answer(gen_text)
        is_correct = (predicted is not None and correct_ans is not None and predicted == correct_ans)
        if is_correct:
            n_correct_problems += 1
        for _ in generated_tokens:
            correct_list.append(is_correct)

        # Batch write to disk
        if len(hidden_list) >= BATCH_SIZE:
            batch_idx = save_batch(hidden_list, vel_list, token_list, correct_list, batch_idx,
                                   os.path.join(args.output_dir, "smolm2_gen_trajs"))

        if (idx + 1) % 50 == 0:
            rate = n_tokens / (time.time() - t0)
            print(f"  [{idx+1}/{len(problems)}] {n_tokens} tokens ({rate:.0f} tok/s) "
                  f"acc={n_correct_problems}/{idx+1} ({100*n_correct_problems/(idx+1):.0f}%) "
                  f"mem={torch.cuda.memory_allocated()/1e9:.2f}GB GPU", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    # Final batch
    if hidden_list:
        batch_idx = save_batch(hidden_list, vel_list, token_list, correct_list, batch_idx,
                               os.path.join(args.output_dir, "smolm2_gen_trajs"))

    print(f"\n{'='*60}", flush=True)
    print(f"Collection complete!", flush=True)
    print(f"  Problems: {len(problems)}", flush=True)
    print(f"  Tokens: {n_tokens}", flush=True)
    print(f"  Batches: {batch_idx}", flush=True)
    print(f"  Generation accuracy: {100*n_correct_problems/len(problems):.1f}%", flush=True)
    print(f"  Time: {time.time()-t0:.0f}s ({n_tokens/max(time.time()-t0, 1):.0f} tok/s)", flush=True)

    # Write metadata
    meta = {
        "n_tokens": n_tokens,
        "n_problems": len(problems),
        "n_batches": batch_idx,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "accuracy": n_correct_problems / len(problems),
        "model": "SmolLM2-360M",
        "dataset": "GSM8K train",
    }
    with open(os.path.join(args.output_dir, "smolm2_gen_trajs_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved to {os.path.join(args.output_dir, 'smolm2_gen_trajs_meta.json')}", flush=True)


if __name__ == "__main__":
    main()
