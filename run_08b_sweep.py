#!/usr/bin/env python3 -u
"""Comprehensive per-layer sweep for Qwen3.5-0.8B.

Tests all 24 layers (6 FA + 18 GDN) with the trained TT at α=0.1.
30 problems per layer, baseline comparison, both steering surfaces.
"""
from __future__ import annotations

import gc, json, os, re, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TT_PATH = "best_tt_08b.pt"
N_LAYERS = 24
D_MODEL = 1024
FA_LAYERS = [3, 7, 11, 15, 19, 23]
N_KV_HEADS = 2
KV_HEAD_DIM = 256
MAX_GEN = 200
ALPHA = 0.1

examples = ('Q: Janet has 5 ducks. She buys 3 more. How many?\nA: Step 1: 5. Step 2: buy 3. Step 3: 5+3=8. So answer is 8.\n\n'
            'Q: 12 muffins/hr, 8 hours?\nA: Step 1: 12/hr. Step 2: 8h. Step 3: 12x8=96. So answer is 96.\n\n')


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"####\s*(-?\d+)", r"So answer is\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def steer_fa(model, h, pkv, li):
    l = model.model.layers[li]
    k = l.self_attn.k_proj(h.to(torch.bfloat16)).view(1, N_KV_HEADS, 1, KV_HEAD_DIM)
    vo = l.self_attn.v_proj(h.to(torch.bfloat16)).view(1, N_KV_HEADS, 1, KV_HEAD_DIM)
    c = pkv.layers[li]
    c.keys[0, :, -1:, :] = k.to(c.keys.dtype)
    c.values[0, :, -1:, :] = vo.to(c.values.dtype)


def steer_gdn(model, h, pkv, li):
    la = model.model.layers[li].linear_attn
    _, k, vo = la.in_proj_qkv(h.to(torch.bfloat16)).chunk(3)
    b = torch.sigmoid(la.in_proj_a(h.to(torch.bfloat16)))
    k = k.view(1, 16, 128)
    vo = vo.view(1, 16, 128)
    delta = b.view(1, 16, 1, 1) * (k.unsqueeze(-1) @ vo.unsqueeze(-2))
    pkv.layers[li].recurrent_states += delta.to(pkv.layers[li].recurrent_states.dtype)


def evaluate(tok, model, tt, problems, steer_layers=None, alpha=ALPHA):
    correct, total = 0, 0
    t0 = time.time()

    for idx, prob in enumerate(problems):
        prompt = f"{examples}Q: {prob['question']}\nA:"
        input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids

        if not steer_layers:
            out = model.generate(input_ids, max_new_tokens=MAX_GEN, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        else:
            past, gens, first = None, [], True
            for step in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(input_ids, past_key_values=past, use_cache=True,
                                 output_hidden_states=True)
                nt = fwd.logits[0, -1, :].argmax().item()
                if nt == tok.eos_token_id:
                    break
                gens.append(nt)

                if not first:
                    hs = fwd.hidden_states
                    hp = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                    x = hp.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        v = tt(x)

                    for li in steer_layers:
                        h_steer = hs[li + 1][0, -1, :] + alpha * v[0, li, :]
                        if li in FA_LAYERS:
                            steer_fa(model, h_steer, fwd.past_key_values, li)
                        else:
                            steer_gdn(model, h_steer, fwd.past_key_values, li)

                past = fwd.past_key_values
                input_ids = torch.tensor([[nt]], device=DEVICE)
                first = False

            gen = tok.decode(gens, skip_special_tokens=True)

        ca = re.search(r"####\s*(-?\d+)", prob["answer"])
        pa = extract_number(gen)
        if pa and ca and pa == ca.group(1):
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            lbl = "base" if not steer_layers else f"L{steer_layers}"
            print(f"  {lbl} [{idx+1}/{len(problems)}] acc={correct}/{total} "
                  f"({100*correct/total:.0f}%) {time.time()-t0:.0f}s", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to test (default: all 24)")
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} (bf16)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    print(f"Loading TT from {TT_PATH}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=D_MODEL).to(DEVICE)
    tt.load_state_dict(torch.load(TT_PATH, map_location="cpu"), strict=False)
    tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    layers = args.layers if args.layers else list(range(N_LAYERS))

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    print(f"\n{len(problems)} problems, {len(layers)} layers, α={args.alpha}", flush=True)

    results = {}

    # Baseline
    print(f"\n{'='*60}", flush=True)
    print("Baseline", flush=True)
    r = evaluate(tok, model, tt, problems, None)
    results["baseline"] = r
    print(f"  Baseline: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    # Per-layer sweep
    for li in layers:
        layer_type = "FA" if li in FA_LAYERS else "GDN"
        print(f"\n{'='*60}", flush=True)
        print(f"{layer_type} L{li} (α={args.alpha})", flush=True)
        r = evaluate(tok, model, tt, problems, [li], args.alpha)
        results[f"{layer_type}_L{li}"] = r
        print(f"  L{li}: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"FULL LAYER SWEEP (α={args.alpha}, {len(problems)} problems)", flush=True)
    print(f"{'='*60}", flush=True)
    ba = results["baseline"]["accuracy"]
    print(f"  {'Layer':>8} {'Type':>5} {'Acc':>7} {'Δ':>7}")
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*7}")
    print(f"  {'base':>8} {'':>5} {100*ba:6.1f}%")
    for li in layers:
        key = f"{'FA' if li in FA_LAYERS else 'GDN'}_L{li}"
        r = results[key]
        acc = r["accuracy"]
        delta = acc - ba
        marker = " ← BEST" if delta == max(results[k]["accuracy"] - ba for k in results if k != "baseline") and delta > 0 else ""
        print(f"  L{li:3d} {'FA' if li in FA_LAYERS else 'GDN':>5} {100*acc:6.1f}% {100*delta:+6.1f}pp{marker}", flush=True)

    print(f"\n  Best layer: max Δ = {max(results[k]['accuracy'] - ba for k in results if k != 'baseline'):+.1f}pp", flush=True)
    print(f"  Worst layer: min Δ = {min(results[k]['accuracy'] - ba for k in results if k != 'baseline'):+.1f}pp", flush=True)

    # Save
    out = {"alpha": args.alpha, "n_test": args.n_test,
           "results": {k: {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"]}
                       for k, v in results.items()}}
    with open("sweep_08b_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to sweep_08b_results.json", flush=True)


if __name__ == "__main__":
    main()
