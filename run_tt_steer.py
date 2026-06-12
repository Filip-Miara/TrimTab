#!/usr/bin/env python3
"""Reasoning-step TT direct steering: use TT-predicted next hidden state.

For each token step t:
  1. Forward → h[t] (final layer hidden state)
  2. TT predicts: v[t] = h[t+1]_predicted - h[t]
  3. h[t+1]_predicted = h[t] + v[t]
  4. steered_logits = LM_head(h[t] + α * v[t])  — steer toward predicted next state
  5. Sample next token from steered_logits

The TT (R²=0.75) was trained on generation data — no distribution shift.
"""
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
MAX_GEN = 100


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--mode", choices=["baseline", "steer"], default="steer")
    parser.add_argument("--alpha", type=float, default=1.0, help="Velocity steering strength")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()

    tt = TrajectoryTransformer(d_model=512, n_layers=6, n_heads=8, d_ff=2048, n_positions=50)
    tt.load_state_dict(torch.load(TT_PATH, map_location=DEVICE))
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    examples = ("Q: Janet has 5. She buys 3. How many?\nA: Step 1: 5. Step 2: She buys 3. Step 3: 5+3=8.\nSo answer is 8.\n\n"
                "Q: Bakery sells 12/hr. How many in 8h?\nA: Step 1: 12/hr. Step 2: 8 hours. Step 3: 12x8=96.\nSo answer is 96.\n\n")

    correct, total = 0, 0
    t0 = time.time()

    for idx, prob in enumerate(problems):
        question = prob["question"]
        correct_ans = parse_answer(prob["answer"])
        prompt = f"{examples}Q: {question}\nA:"

        if args.mode == "baseline":
            inp = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=MAX_GEN, do_sample=False, pad_token_id=tok.eos_token_id)
            gen_text = tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            # Step-by-step generation with TT velocity steering
            full_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE).input_ids
            plen = full_ids.shape[1]

            for step in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(full_ids, output_hidden_states=True)

                hs = fwd.hidden_states[-1][0]  # final layer, (S, 2048)
                orig_logits = fwd.logits[0, -1, :]
                actual_step = full_ids.shape[1] - plen

                if actual_step >= 2:
                    # Get trajectory so far and predict next velocity
                    n = min(actual_step, 48)
                    traj = hs[plen:plen + n].unsqueeze(0).float()
                    with torch.no_grad():
                        v_pred = tt(traj.to(DEVICE), causal=True)[0, n - 1]  # (2048,)

                    # Steered hidden state: h_steered = h_current + α * v_pred
                    h_current = hs[-1, :]  # last position's hidden state
                    h_steered = h_current.float() + args.alpha * v_pred
                    steer_logits = lm_head(h_steered.to(lm_head.weight.dtype).unsqueeze(0))[0]

                    # Sample from steered logits
                    next_token = steer_logits.argmax().item()
                else:
                    next_token = orig_logits.argmax().item()

                if next_token == tok.eos_token_id:
                    break
                full_ids = torch.cat([full_ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)

            gen_text = tok.decode(full_ids[0, plen:], skip_special_tokens=True)

        predicted = parse_answer(gen_text)
        if predicted is not None and correct_ans is not None and predicted == correct_ans:
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{args.n_test}] acc={correct}/{total} ({100*correct/total:.0f}%) {time.time()-t0:.0f}s")
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}" + (f" (α={args.alpha})" if args.mode == "steer" else ""))
    print(f"  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
