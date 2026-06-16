#!/usr/bin/env python3 -u
"""α sweep on L8: test 7 alpha values to find the functional form."""
from __future__ import annotations

import gc, json, re, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
N_LAYERS = 28
N_KV_HEADS = 4
HEAD_DIM = 128
TT_PATH = "best_gen_tt_7b.pt"
MAX_GEN = 200
ALPHAS = [-0.5, -0.2, -0.1, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

examples = ('Q: Janet has 5 ducks. She buys 3 more. How many?\nA: Step 1: 5. Step 2: buy 3. Step 3: 5+3=8. So answer is 8.\n\n'
            'Q: 12 muffins/hr, 8 hours?\nA: Step 1: 12/hr. Step 2: 8h. Step 3: 12x8=96. So answer is 96.\n\n')


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"####\s*(-?\d+)", r"So answer is\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def steer_layer(model, hs, v, pkv, alpha, li):
    h = hs[li + 1][0, -1, :] + alpha * v[0, li, :]
    l = model.model.layers[li]
    k = l.self_attn.k_proj(h.to(torch.bfloat16)).view(1, N_KV_HEADS, 1, HEAD_DIM)
    vo = l.self_attn.v_proj(h.to(torch.bfloat16)).view(1, N_KV_HEADS, 1, HEAD_DIM)
    lc = pkv.layers[li]
    lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
    lc.values[0, :, -1:, :] = vo.to(lc.values.dtype)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print(f"Loading TT from {TT_PATH}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=3584).to(DEVICE)
    tt.load_state_dict(torch.load(TT_PATH, map_location="cpu"), strict=False)
    tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    li = args.layer
    print(f"\nα sweep on L{li} ({args.n_test} problems, {len(args.alphas)} alphas)", flush=True)

    results = {a: [] for a in args.alphas}
    t0 = time.time()

    for idx, prob in enumerate(problems):
        prompt = f"{examples}Q: {prob['question']}\nA:"
        input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids

        # First, get the baseline direction for each alpha independently
        # We generate separately for each alpha (could optimize, but clear)
        for alpha in args.alphas:
            past, gens, first = None, [], True
            ids = input_ids.clone()

            for step in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(ids, past_key_values=past, use_cache=True,
                                 output_hidden_states=True)
                nt = fwd.logits[0, -1, :].argmax().item()
                if nt == tok.eos_token_id: break
                gens.append(nt)

                if not first:
                    hs = fwd.hidden_states
                    hp = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                    x = hp.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        v = tt(x)
                    steer_layer(model, hs, v, fwd.past_key_values, alpha, li)

                past = fwd.past_key_values
                ids = torch.tensor([[nt]], device=DEVICE)
                first = False

            gen = tok.decode(gens, skip_special_tokens=True)
            ca = re.search(r"####\s*(-?\d+)", prob["answer"])
            pa = extract_number(gen)
            results[alpha].append(1 if (pa and ca and pa == ca.group(1)) else 0)

        if (idx + 1) % 10 == 0:
            status = " | ".join(f"α={a}: {sum(results[a])}/{len(results[a])}" for a in args.alphas)
            print(f"  [{idx+1}/{args.n_test}] {status}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*60}", flush=True)
    print(f"α SWEEP ON L{li}", flush=True)
    print(f"{'='*60}", flush=True)
    baseline_acc = np.mean(results[0.0]) if 0.0 in results else 0
    print(f"  {'α':>6} {'Acc':>8s} {'Δ':>8s} {'CI_lo':>8s} {'CI_hi':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for a in sorted(results.keys()):
        r = np.mean(results[a])
        ci = 1.96 * np.std(results[a]) / np.sqrt(len(results[a]))
        d = r - baseline_acc if a != 0.0 else 0.0
        print(f"  {a:6.2f} {100*r:7.1f}% {100*d:+7.1f}pp [{100*(r-ci):5.1f}%,{100*(r+ci):5.1f}%]", flush=True)

    out = {"layer": li, "alphas": {str(a): float(np.mean(results[a])) for a in results},
           "baseline": float(baseline_acc), "n_test": args.n_test}
    with open(f"alpha_sweep_L{li}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to alpha_sweep_L{li}.json", flush=True)
    print(f"Time: {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
