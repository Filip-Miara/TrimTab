#!/usr/bin/env python3
"""Test reasoning-step model for token-level error detection."""
from __future__ import annotations

import gc, json, re, sys, time
import numpy as np
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
TT_PATH = "best_reasoning_transformer.pt"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
N_TEST = 30


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def main():
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:N_TEST]

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    tt = TrajectoryTransformer(d_model=512, n_layers=6, n_heads=8, d_ff=2048, n_positions=50)
    tt.load_state_dict(torch.load(TT_PATH, map_location=DEVICE))
    tt.to(DEVICE)
    tt.eval()

    examples = ("Q: Janet has 5. She buys 3. How many?\nA: Step 1: 5. Step 2: She buys 3. Step 3: 5+3=8.\n"
                "So answer is 8.\n\n"
                "Q: Bakery sells 12/hr. How many in 8h?\nA: Step 1: 12/hr. Step 2: 8 hours. Step 3: 12x8=96.\n"
                "So answer is 96.\n\n")

    per_token_errors = {True: [], False: []}
    t0 = time.time()

    for idx, prob in enumerate(problems):
        question = prob["question"]
        correct_ans = parse_answer(prob["answer"])
        prompt = f"{examples}Q: {question}\nA:"
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        plen = inp.input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)

        gen_text = tok.decode(out[0, plen:], skip_special_tokens=True)
        predicted = parse_answer(gen_text)
        is_correct = predicted is not None and correct_ans is not None and predicted == correct_ans

        with torch.no_grad():
            fwd = model(out, output_hidden_states=True)
        hs = fwd.hidden_states[-1][0]

        T = out.shape[1] - plen
        if T < 3:
            gc.collect()
            torch.cuda.empty_cache()
            continue

        for t in range(T - 1):
            h_current = hs[plen + t]
            h_next = hs[plen + t + 1]
            traj = hs[plen:plen + t + 1].unsqueeze(0).float()
            with torch.no_grad():
                v_pred = tt(traj.to(DEVICE), causal=True)[0, t]
            v_actual = h_next - h_current
            error = F.mse_loss(v_pred.cpu(), v_actual.cpu()).item()
            per_token_errors[is_correct].append(error)

        if (idx + 1) % 10 == 0:
            ce = np.mean(per_token_errors[True]) if per_token_errors[True] else 0
            ie = np.mean(per_token_errors[False]) if per_token_errors[False] else 0
            print(f"  [{idx+1}/{N_TEST}] correct_err={ce:.4f} incorrect_err={ie:.4f} {time.time()-t0:.0f}s")
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nResults: {N_TEST} problems")
    c_errs = per_token_errors[True]
    i_errs = per_token_errors[False]
    print(f"  Correct tokens:   mean err = {np.mean(c_errs):.4f} (n={len(c_errs)})")
    print(f"  Incorrect tokens: mean err = {np.mean(i_errs):.4f} (n={len(i_errs)})")
    print(f"  Gap: {np.mean(i_errs) - np.mean(c_errs):.4f}")
    if c_errs and i_errs:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(i_errs, c_errs)
        print(f"  t-test: t={t_stat:.4f}, p={p_val:.4f} {'✅' if p_val < 0.05 else '❌'}")


if __name__ == "__main__":
    main()
