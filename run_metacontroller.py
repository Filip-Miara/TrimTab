#!/usr/bin/env python3
"""MetaController: uncertainty analysis on GSM8K.

Two-pass approach:
  Pass 1: Generate answer via model.generate() (fast, KV-cached)
  Pass 2: Single full-sequence forward pass with output_hidden_states
          → Perceiver → reading head → per-token uncertainty

This gives us the uncertainty profile of each generation WITHOUT
per-step overhead. We can then analyze whether uncertainty correlates
with errors, and where interventions would be most valuable.
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.thought_diffusion import ThoughtDiffusion

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
PERCEIVER_PATH = "best_perceiver.pt"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
MAX_GEN = 256


class UncertaintyProbe(nn.Module):
    def __init__(self, n_latents=32, d_latent=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents * d_latent, 128), nn.GELU(), nn.Linear(128, 1),
        )

    def forward(self, latents):
        return self.net(latents.reshape(latents.shape[0], -1)).squeeze(-1)


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0=greedy, >0=sampling")
    args = parser.parse_args()

    print("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_problems]
    print(f"  {len(problems)} problems")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    n_layers = getattr(cfg, 'num_hidden_layers', 24)
    d_model = getattr(cfg, 'hidden_size', 2048)

    print("Loading Perceiver + reading head...")
    perceiver = ThoughtDiffusion(d_model=d_model, n_layers=n_layers,
                                  d_latent=128, n_latents=32, d_text_ctx=d_model)
    perceiver.load_state_dict(torch.load(PERCEIVER_PATH, map_location=DEVICE), strict=False)
    perceiver.to(DEVICE)
    perceiver.eval()
    for p in perceiver.parameters():
        p.requires_grad = False

    probe = UncertaintyProbe(n_latents=32, d_latent=128).to(DEVICE)
    probe.eval()

    examples = (
        "Q: Janet has 5 oranges. She buys 3 more. How many oranges does she have?\n"
        "A: Step 1: Janet has 5. Step 2: She buys 3. Step 3: 5 + 3 = 8.\n"
        "So the answer is 8.\n\n"
        "Q: A bakery sells 12 croissants per hour. How many in 8 hours?\n"
        "A: Step 1: 12 per hour. Step 2: 8 hours total. Step 3: 12 × 8 = 96.\n"
        "So the answer is 96.\n\n"
    )

    all_uncertainties = []
    correct_uncertainties = []
    incorrect_uncertainties = []
    results = []
    t_start = time.time()

    for idx, problem in enumerate(problems):
        question = problem["question"]
        correct = parse_answer(problem["answer"])
        prompt = f"{examples}Q: {question}\nA:"

        # Pass 1: Generate answer (fast, KV-cached)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=512).to(DEVICE)
        prompt_len = inputs.input_ids.shape[1]

        gen_kwargs = dict(max_new_tokens=MAX_GEN, pad_token_id=tokenizer.eos_token_id)
        if args.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=args.temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        gen_text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        predicted = parse_answer(gen_text)
        is_correct = predicted is not None and correct is not None and predicted == correct

        # Pass 2: Single forward pass for hidden states
        with torch.no_grad():
            fwd = model(out, output_hidden_states=True)

        hs_tuple = fwd.hidden_states

        # Per-token uncertainty
        uncertainties = []
        seq_len = hs_tuple[0].shape[1]

        for pos in range(seq_len - 2):
            h_pos = torch.stack([h[0, pos, :].float() for h in hs_tuple], dim=0)
            x = h_pos[:23].unsqueeze(0).to(DEVICE)
            ctx = h_pos[0].unsqueeze(0).to(DEVICE)

            _, latents = perceiver(x, ctx, return_latents=True)
            u = probe(latents).item()
            uncertainties.append(u)

        avg_u = np.mean(uncertainties) if uncertainties else 0.0

        all_uncertainties.extend(uncertainties)
        if is_correct:
            correct_uncertainties.extend(uncertainties)
        else:
            incorrect_uncertainties.extend(uncertainties)

        results.append({
            "idx": idx, "correct": is_correct, "predicted": predicted,
            "expected": correct, "gen_len": len(uncertainties),
            "uncertainty_mean": float(avg_u),
            "uncertainty_std": float(np.std(uncertainties)) if uncertainties else 0.0,
        })

        gc.collect()
        torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0:
            n_correct = sum(1 for r in results if r["correct"])
            elapsed = time.time() - t_start
            print(f"  [{idx+1}/{len(problems)}] acc={n_correct}/{idx+1} "
                  f"({100*n_correct/(idx+1):.0f}%) | {elapsed:.0f}s")

    n_correct = sum(1 for r in results if r["correct"])
    total_tokens = sum(r["gen_len"] for r in results)

    print(f"\n{'='*60}")
    print(f"GSM8K: {args.n_problems} problems, temperature={args.temperature}")
    print(f"  Accuracy: {n_correct}/{len(results)} ({100*n_correct/len(results):.0f}%)")
    print(f"  Total tokens: {total_tokens}")

    if correct_uncertainties and incorrect_uncertainties:
        c_mean = np.mean(correct_uncertainties)
        i_mean = np.mean(incorrect_uncertainties)
        print(f"  Correct: uncertainty={c_mean:.4f}")
        print(f"  Incorrect: uncertainty={i_mean:.4f}")
        diff = i_mean - c_mean
        print(f"  Difference: {diff:.4f} ({'higher for incorrect' if diff > 0 else 'higher for correct'})")

        # Per-problem uncertainty
        c_prob_means = [r["uncertainty_mean"] for r in results if r["correct"]]
        i_prob_means = [r["uncertainty_mean"] for r in results if not r["correct"]]
        print(f"  Per-problem: correct mean={np.mean(c_prob_means):.4f}, "
              f"incorrect mean={np.mean(i_prob_means):.4f}")

        # Top-10% highest uncertainty tokens — what fraction are in incorrect answers?
        sorted_uncertainties = sorted(all_uncertainties, reverse=True)
        top10_threshold = sorted_uncertainties[max(len(all_uncertainties) // 10, 1)]
        top10_tokens = sum(1 for u in all_uncertainties if u >= top10_threshold)
        top10_incorrect = sum(1 for u in incorrect_uncertainties if u >= top10_threshold)
        print(f"  Top 10% highest uncertainty tokens: {top10_incorrect}/{top10_tokens} "
              f"({100*top10_incorrect/max(top10_tokens,1):.0f}%) are from incorrect answers")
        print(f"  (Baseline: {len(incorrect_uncertainties)}/{total_tokens} = "
              f"{100*len(incorrect_uncertainties)/total_tokens:.0f}% of tokens are incorrect)")

    print(f"{'='*60}")

    summary = {
        "n_problems": len(results),
        "accuracy": n_correct / len(results),
        "total_tokens": total_tokens,
        "correct_uncertainty_mean": float(np.mean(correct_uncertainties)) if correct_uncertainties else None,
        "incorrect_uncertainty_mean": float(np.mean(incorrect_uncertainties)) if incorrect_uncertainties else None,
        "per_problem": results,
    }
    with open("metacontroller_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
