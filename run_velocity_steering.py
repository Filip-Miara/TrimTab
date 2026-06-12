#!/usr/bin/env python3
"""Velocity Steering: use Perceiver flow to steer token selection.

For each generated token, compute the Perceiver velocity for the last
hidden state, steer it, and re-sample from the steered logits. If the
steered token differs from the original, it's a "flow correction."

This tests the core hypothesis: the Perceiver's predicted velocity
points in a direction that improves token selection.

Two modes:
  --analyze:   Post-hoc analysis of generated answers (for paper)
  --steer:     Real-time steering during generation (expensive)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.thought_diffusion import ThoughtDiffusion

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
PERCEIVER_PATH = "best_perceiver.pt"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"


def parse_answer(text: str) -> str | None:
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for pat in [
        r"####\s*(-?\d+[.,]?\d*)",
        r"The answer is\s*(-?\d+[.,]?\d*)",
        r"answer is\s*(-?\d+[.,]?\d*)",
        r"=\s*(-?\d+[.,]?\d*)\s*$",
        r"Therefore,? .*? (-?\d+)",
        r"So,? .*? (-?\d+)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None


def analyze(args):
    """Post-hoc analysis: for each generated token, check if steering would change it."""
    print("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n]

    print("Loading model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
    text_cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    n_layers = getattr(text_cfg, 'num_hidden_layers', 24)
    d_model = getattr(text_cfg, 'hidden_size', 2048)

    print("Loading Perceiver...")
    perceiver = ThoughtDiffusion(d_model=d_model, n_layers=n_layers,
                                  d_latent=128, n_latents=32, d_text_ctx=d_model)
    perceiver.load_state_dict(torch.load(args.perceiver, map_location=DEVICE), strict=False)
    perceiver.to(DEVICE)
    perceiver.eval()
    for p in perceiver.parameters():
        p.requires_grad = False

    examples = (
        "Q: Janet has 5 oranges. She buys 3 more. How many oranges does she have?\n"
        "A: Step 1: Janet has 5. Step 2: She buys 3. Step 3: 5 + 3 = 8.\n"
        "So the answer is 8.\n\n"
        "Q: A bakery sells 12 croissants per hour. How many in 8 hours?\n"
        "A: Step 1: 12 per hour. Step 2: 8 hours total. Step 3: 12 × 8 = 96.\n"
        "So the answer is 96.\n\n"
    )

    total_tokens = 0
    total_divergences = 0
    total_corrections = 0
    steered_correct = 0
    steered_incorrect = 0
    results = []
    t_start = time.time()

    for idx, problem in enumerate(problems):
        question = problem["question"]
        correct_ans = parse_answer(problem["answer"])
        prompt = f"{examples}Q: {question}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=512).to(DEVICE)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_gen,
                                 do_sample=args.temperature > 0,
                                 temperature=args.temperature if args.temperature > 0 else None,
                                 pad_token_id=tokenizer.eos_token_id)

        # Single forward pass for all hidden states
        with torch.no_grad():
            fwd = model(out, output_hidden_states=True)

        hs_tuple = fwd.hidden_states
        full_logits = fwd.logits[0]  # (S, V)

        per_pos = []
        divergences = 0
        corrections = 0

        for pos in range(prompt_len - 1, out.shape[1] - 1):
            original_token = out[0, pos + 1].item()

            # Get hidden states for this position
            h_pos = torch.stack([h[0, pos, :].float() for h in hs_tuple], dim=0)
            h_final = h_pos[24]  # the one used by LM head

            # Perceiver velocity
            x = h_pos[:23].unsqueeze(0).to(DEVICE)
            ctx = h_pos[0].unsqueeze(0).to(DEVICE)
            v = perceiver(x, ctx)  # (1, 23, 2048)

            # Use v[0, 22, :] as effective velocity for last layer
            v_eff = v[0, 22, :]

            # Steered hidden state
            h_steered = h_final + args.alpha * v_eff

            steered_logits = lm_head(h_steered.to(lm_head.weight.dtype))
            steered_token = steered_logits.argmax().item()

            original_text = tokenizer.decode([original_token])
            steered_text = tokenizer.decode([steered_token])

            is_divergence = steered_token != original_token
            if is_divergence:
                divergences += 1

            per_pos.append({
                "pos": pos - prompt_len + 1,
                "original": original_token,
                "original_text": original_text,
                "steered": steered_token,
                "steered_text": steered_text,
                "diverges": is_divergence,
            })

        generated = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        predicted_ans = parse_answer(generated)
        is_correct = predicted_ans is not None and correct_ans is not None and predicted_ans == correct_ans

        total_tokens += len(per_pos)
        total_divergences += divergences
        total_corrections += corrections
        if is_correct:
            steered_correct += 1
        else:
            steered_incorrect += 1

        results.append({
            "idx": idx, "correct": is_correct, "predicted": predicted_ans,
            "expected": correct_ans, "tokens": len(per_pos),
            "divergences": divergences, "divergence_rate": divergences / max(len(per_pos), 1),
        })

        gc.collect()
        torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0:
            n_correct = sum(1 for r in results if r["correct"])
            elapsed = time.time() - t_start
            div_rate = total_divergences / max(total_tokens, 1)
            print(f"  [{idx+1}/{len(problems)}] acc={n_correct}/{idx+1} "
                  f"({100*n_correct/(idx+1):.0f}%) | div_rate={div_rate:.4f} | {elapsed:.0f}s")

    n_correct = sum(1 for r in results if r["correct"])
    div_rate = total_divergences / max(total_tokens, 1)

    print(f"\n{'='*60}")
    print(f"Velocity Steering Analysis (α={args.alpha})")
    print(f"  Problems: {len(results)}, Accuracy: {n_correct}/{len(results)} "
          f"({100*n_correct/len(results):.0f}%)")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Divergence rate: {total_divergences}/{total_tokens} ({100*div_rate:.1f}%)")
    print(f"    (how often steering changes the selected token)")
    print(f"{'='*60}")

    summary = {
        "n_problems": len(results), "accuracy": n_correct / len(results),
        "total_tokens": total_tokens, "divergence_rate": div_rate,
        "alpha": args.alpha, "per_problem": results,
    }
    with open("velocity_steering_results.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["analyze"], default="analyze")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Steering strength")
    parser.add_argument("--perceiver", type=str, default=PERCEIVER_PATH)
    parser.add_argument("--max-gen", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    analyze(args)


if __name__ == "__main__":
    main()
