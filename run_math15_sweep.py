#!/usr/bin/env python3 -u
"""Unified per-layer sweep for Math-1.5B with standard TT.

Stages:
  1. Per-layer sweep (find trim tabs)
  2. Alpha sweep on best layers
  3. Multi-layer combinations
  4. Per-layer alpha in combinations
"""
from __future__ import annotations

import gc, json, os, re, sys, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
D_MODEL = 1536
N_LAYERS = 28
N_KV_HEADS = 2
HEAD_DIM = 128
TT_PATH = "best_tt_all.pt"
MAX_GEN = 200

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


def evaluate(tok, model, tt, problems, alpha, layers):
    """layers = list of layer indices to steer. Empty list = baseline."""
    correct, total = 0, 0
    t0 = time.time()
    for idx, prob in enumerate(problems):
        prompt = f"{examples}Q: {prob['question']}\nA:"
        input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids

        if not layers or alpha == 0.0:
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
                if nt == tok.eos_token_id: break
                gens.append(nt)
                if not first:
                    hs = fwd.hidden_states
                    hp = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                    x = hp.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        v = tt(x)
                    if isinstance(alpha, dict):
                        for li in layers:
                            steer_layer(model, hs, v, fwd.past_key_values, alpha.get(li, 0.1), li)
                    else:
                        for li in layers:
                            steer_layer(model, hs, v, fwd.past_key_values, alpha, li)
                past = fwd.past_key_values
                input_ids = torch.tensor([[nt]], device=DEVICE)
                first = False
            gen = tok.decode(gens, skip_special_tokens=True)

        ca = re.search(r"####\s*(-?\d+)", prob["answer"])
        pa = extract_number(gen)
        if pa and ca and pa == ca.group(1): correct += 1
        total += 1
        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(problems)}] acc={correct}/{total} ({100*correct/total:.0f}%)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    print("Loading Math-1.5B (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print(f"Loading standard TT from {TT_PATH}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=D_MODEL).to(DEVICE)
    tt.load_state_dict(torch.load(TT_PATH, map_location="cpu"), strict=False)
    tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    results = {}

    if args.stage == 1:
        # Stage 1: Per-layer sweep
        print(f"Baseline", flush=True)
        r = evaluate(tok, model, tt, problems, 0.0, [])
        results["baseline"] = r
        print(f"  Baseline: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

        for li in range(N_LAYERS):
            print(f"Layer {li}", flush=True)
            r = evaluate(tok, model, tt, problems, args.alpha, [li])
            results[f"L{li}"] = r
            print(f"  L{li}: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

        print(f"\nLayer sweep summary:", flush=True)
        ba = results["baseline"]["accuracy"]
        for k in sorted(results):
            if k == "baseline": continue
            d = results[k]["accuracy"] - ba
            print(f"  {k}: {100*results[k]['accuracy']:5.1f}% ({100*d:+5.1f}pp)", flush=True)

    elif args.stage == 2:
        # Stage 2: Alpha sweep on best layers
        best_layers = [l for l in range(N_LAYERS)]  # will be pruned by user
        alphas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
        print(f"Baseline", flush=True)
        r = evaluate(tok, model, tt, problems, 0.0, [])
        results["baseline"] = r
        for li in best_layers:
            for a in alphas:
                lbl = f"L{li}_a{a}"
                print(f"{lbl}", flush=True)
                r = evaluate(tok, model, tt, problems, a, [li])
                results[lbl] = r

    elif args.stage == 3:
        # Stage 3: Multi-layer combinations
        print(f"Baseline", flush=True)
        r = evaluate(tok, model, tt, problems, 0.0, [])
        results["baseline"] = r
        # Test pairs and triplets of top layers from stage 1
        top = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # adjust after stage 1
        for (a, b) in [(top[i], top[j]) for i in range(len(top)) for j in range(i+1, len(top))][:20]:
            r = evaluate(tok, model, tt, problems, args.alpha, [a, b])
            results[f"L{a}+{b}"] = r

    elif args.stage == 4:
        # Stage 4: Per-layer alpha in combinations
        alpha_dict = {l: a for l, a in zip(range(28), [args.alpha]*28)}
        r = evaluate(tok, model, tt, problems, alpha_dict, list(range(28)))
        results["all_per_layer_alpha"] = r

    with open(f"math15_stage{args.stage}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved.", flush=True)


if __name__ == "__main__":
    main()
