#!/usr/bin/env python3
"""Flow-Correctness Correlation Experiment.

Hypothesis: reasoning chains leading to correct answers have higher
"flow alignment" — the Perceiver's velocity predictions better match
the actual layer-to-layer hidden state evolution.

For each generated token position p:
  1. Extract hidden states across all layers: h[p, 0..22]
  2. Compute actual velocities: v = h[p, 1..23] - h[p, 0..22]
  3. Perceiver predicts: v_pred = f(h[p, 0..22])
  4. Error = MSE(v_pred, v)

Compare: average error per correct vs incorrect answer.
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
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.thought_diffusion import ThoughtDiffusion

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
PERCEIVER_PATH = "best_perceiver.pt"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
N_PROBLEMS = 50
MAX_GEN = 256


def parse_gsm8k_answer(text: str) -> str | None:
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
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return numbers[-1]
    return None


def compute_flow_alignment(
    hidden_states: tuple[torch.Tensor],
    perceiver: ThoughtDiffusion,
    n_layers: int = 24,
    gen_len: int | None = None,
) -> list[float]:
    """Compute per-position Perceiver prediction error.

    hidden_states: tuple of (B, S, D) from output_hidden_states=True
    Returns: list of MSE values, one per sequence position (excl last)
    """
    device = next(perceiver.parameters()).device
    errors = []

    seq_len = hidden_states[0].shape[1]
    start_pos = max(0, seq_len - gen_len)
    for pos in range(start_pos, seq_len - 2):
        hs = torch.stack([h[0, pos, :].float() for h in hidden_states], dim=0)
        x = hs[:n_layers - 1].unsqueeze(0)
        y = (hs[1:n_layers] - hs[:n_layers - 1]).unsqueeze(0)
        ctx = hs[0].unsqueeze(0)

        with torch.no_grad():
            vp = perceiver(x.to(device), ctx.to(device))
            mse = F.mse_loss(vp, y.to(device)).item()
        errors.append(mse)

    return errors


def main():
    print("Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:N_PROBLEMS]
    print(f"  {len(problems)} problems")

    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    model.eval()

    cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    n_layers = getattr(cfg, 'num_hidden_layers', 24)
    d_model = getattr(cfg, 'hidden_size', 2048)

    print(f"Loading Perceiver from {PERCEIVER_PATH}...")
    perceiver = ThoughtDiffusion(d_model=d_model, n_layers=n_layers,
                                  d_latent=128, n_latents=32, d_text_ctx=d_model)
    state = torch.load(PERCEIVER_PATH, map_location=DEVICE)
    perceiver.load_state_dict(state, strict=False)
    perceiver.to(DEVICE)
    perceiver.eval()
    n_params = sum(p.numel() for p in perceiver.parameters())
    print(f"  Perceiver: {n_params:,} params")

    results = []
    t_start = time.time()

    for idx, problem in enumerate(problems):
        question = problem["question"]
        correct_answer = parse_gsm8k_answer(problem["answer"])
        # Few-shot prompt
        examples = (
            "Q: Janet has 5 oranges. She buys 3 more. How many oranges does she have?\n"
            "A: Step 1: Janet has 5. Step 2: She buys 3. Step 3: 5 + 3 = 8.\n"
            "So the answer is 8.\n\n"
            "Q: A bakery sells 12 croissants per hour. How many in 8 hours?\n"
            "A: Step 1: 12 per hour. Step 2: 8 hours total. Step 3: 12 × 8 = 96.\n"
            "So the answer is 96.\n\n"
        )
        prompt = f"{examples}Q: {question}\nA:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=512).to(DEVICE)
        prompt_len = inputs.input_ids.shape[1]

        # Generate answer
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_GEN,
                do_sample=False, temperature=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        generated_text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        predicted_ans = parse_gsm8k_answer(generated_text)
        is_correct = (
            predicted_ans is not None and correct_answer is not None
            and predicted_ans == correct_answer
        )

        # Get hidden states for full sequence
        with torch.no_grad():
            fwd = model(out, output_hidden_states=True)

        hs_tuple = fwd.hidden_states  # tuple of (1, S, D) per layer

        # Compute flow alignment for the GENERATED tokens only
        gen_errors = compute_flow_alignment(hs_tuple, perceiver, n_layers, gen_len=out.shape[1] - prompt_len)

        avg_error = np.mean(gen_errors) if gen_errors else 0.0
        std_error = np.std(gen_errors) if len(gen_errors) > 1 else 0.0

        result = {
            "idx": idx,
            "correct": is_correct,
            "predicted": predicted_ans,
            "expected": correct_answer,
            "gen_len": len(gen_errors),
            "flow_error_mean": float(avg_error),
            "flow_error_std": float(std_error),
        }
        results.append(result)

        status = "✓" if is_correct else "✗"
        elapsed = time.time() - t_start
        n_correct = sum(1 for r in results if r["correct"])
        print(f"  [{idx+1}/{len(problems)}] {status} | acc={n_correct}/{idx+1} "
              f"({100*n_correct/(idx+1):.0f}%) | flow_err={avg_error:.5f} | {elapsed:.0f}s")

        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate
    correct_errors = [r["flow_error_mean"] for r in results if r["correct"]]
    incorrect_errors = [r["flow_error_mean"] for r in results if not r["correct"]]

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(problems)} GSM8K problems")
    print(f"{'='*60}")
    print(f"  Accuracy: {len(correct_errors)}/{len(results)} "
          f"({100*len(correct_errors)/len(results):.0f}%)")
    print(f"  Correct answers: mean flow error = {np.mean(correct_errors):.5f} "
          f"(std={np.std(correct_errors):.5f})" if correct_errors else "  No correct answers")
    print(f"  Incorrect answers: mean flow error = {np.mean(incorrect_errors):.5f} "
          f"(std={np.std(incorrect_errors):.5f})" if incorrect_errors else "  No incorrect answers")

    if correct_errors and incorrect_errors:
        t_stat, p_val = None, None
        try:
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(correct_errors, incorrect_errors)
            print(f"  t-test: t={t_stat:.4f}, p={p_val:.4f}")
        except ImportError:
            diff = np.mean(correct_errors) - np.mean(incorrect_errors)
            print(f"  Mean diff: {diff:.5f} ({'correct lower' if diff < 0 else 'incorrect lower'})")

        print(f"\n  Verdict: ", end="")
        if p_val is not None and p_val < 0.05:
            if np.mean(correct_errors) < np.mean(incorrect_errors):
                print("✅ Flow alignment correlates with reasoning quality!")
            else:
                print("⚠️  Counter-intuitive: incorrect answers have higher flow alignment")
        elif p_val is not None:
            print("❌ No significant correlation between flow alignment and correctness")
        else:
            print("⚠️  No statistical test available")
    print(f"{'='*60}")

    with open("flow_correctness_results.json", "w") as f:
        json.dump({
            "n_problems": len(results),
            "accuracy": len(correct_errors) / len(results),
            "correct_mean_error": float(np.mean(correct_errors)) if correct_errors else None,
            "incorrect_mean_error": float(np.mean(incorrect_errors)) if incorrect_errors else None,
            "per_problem": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
