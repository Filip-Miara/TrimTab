#!/usr/bin/env python3 -u
"""Autonomous 4-stage sweep pipeline for Math-1.5B.

Stage 1: Per-layer sweep (find trim tabs)
Stage 2: Alpha sweep on top-3 layers
Stage 3: Multi-layer combinations (pairs + triplets)
Stage 4: Per-layer alpha vector in best combo

Each stage runs sequentially, selecting optimal params for next stage.
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
N_TEST = 100  # Stage 1 uses 100 for precision; stages 2-4 can use 100 too

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
    correct, total = 0, 0
    t0 = time.time()
    for idx, prob in enumerate(problems):
        prompt = f"{examples}Q: {prob['question']}\nA:"
        input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids

        if not layers or alpha == 0.0 or (isinstance(alpha, dict) and not any(alpha.values())):
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
            print(f"  [{idx+1}/{len(problems)}] acc={correct}/{total} ({100*correct/total:.0f}%)",
                  flush=True)
            gc.collect(); torch.cuda.empty_cache()
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def print_table(results, baseline_key="baseline"):
    ba = results.get(baseline_key, {}).get("accuracy", 0)
    for k in sorted(results.keys()):
        v = results[k]
        if k == baseline_key:
            print(f"  {k:20s}: {100*v['accuracy']:5.1f}%", flush=True)
        else:
            d = v["accuracy"] - ba
            print(f"  {k:20s}: {100*v['accuracy']:5.1f}% ({100*d:+5.1f}pp)", flush=True)


def save(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {path}", flush=True)


def main():
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
    problems = [r for r in ds if len(r["question"]) > 50][:N_TEST]

    overall_t0 = time.time()
    results_all = {}

    # ===== STAGE 1: Per-layer sweep =====
    print(f"\n{'='*60}", flush=True)
    print(f"STAGE 1: Per-layer sweep ({N_TEST} problems, α=0.1)", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\nBaseline", flush=True)
    r = evaluate(tok, model, tt, problems, 0.0, [])
    results_all["baseline"] = r
    print(f"  Baseline: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    for li in range(N_LAYERS):
        print(f"\nLayer {li}", flush=True)
        r = evaluate(tok, model, tt, problems, 0.1, [li])
        results_all[f"L{li}"] = r
        print(f"  L{li}: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)
        save(results_all, "math15_stage1_results.json")

    print(f"\nSTAGE 1 summary:", flush=True)
    print_table(results_all)
    save(results_all, "math15_stage1_results.json")

    # Identify top-3 best layers and worst layers
    ba = results_all["baseline"]["accuracy"]
    layer_deltas = [(li, results_all[f"L{li}"]["accuracy"] - ba) for li in range(N_LAYERS)]
    layer_deltas.sort(key=lambda x: x[1], reverse=True)
    top3 = [ld[0] for ld in layer_deltas[:3] if ld[1] > 0]
    worst3 = [ld[0] for ld in layer_deltas[-3:] if ld[1] < 0]
    print(f"\n  Top layers: {top3}", flush=True)
    print(f"  Worst layers: {worst3}", flush=True)

    # ===== STAGE 2: Alpha sweep on top layers =====
    if top3:
        print(f"\n{'='*60}", flush=True)
        print(f"STAGE 2: Alpha sweep on top layers {top3}", flush=True)
        print(f"{'='*60}", flush=True)
        alphas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
        for li in top3:
            for a in alphas:
                print(f"\nL{li} α={a}", flush=True)
                r = evaluate(tok, model, tt, problems, a, [li])
                results_all[f"L{li}_a{a}"] = r
                save(results_all, "math15_stage2_results.json")
        print(f"\nSTAGE 2 summary:", flush=True)
        print_table(results_all, "baseline")
        save(results_all, "math15_stage2_results.json")

        # Select best alpha per layer
        best_alphas = {}
        for li in top3:
            cands = [(a, results_all.get(f"L{li}_a{a}", {}).get("accuracy", 0)) for a in alphas]
            cands.sort(key=lambda x: x[1], reverse=True)
            best_alphas[li] = cands[0][0]
        print(f"  Best alphas: {best_alphas}", flush=True)

        # ===== STAGE 3: Multi-layer combinations =====
        print(f"\n{'='*60}", flush=True)
        print(f"STAGE 3: Multi-layer combinations of {top3}", flush=True)
        print(f"{'='*60}", flush=True)
        combos = []
        for a in top3:
            for b in top3:
                if a < b:
                    combos.append((a, b))
                    for c in top3:
                        if b < c:
                            combos.append((a, b, c))

        for combo in combos:
            print(f"\nL{'+'.join(str(l) for l in combo)}", flush=True)
            r = evaluate(tok, model, tt, problems, 0.1, list(combo))
            results_all[f"L{'+'.join(str(l) for l in combo)}"] = r
            save(results_all, "math15_stage3_results.json")

        print(f"\nSTAGE 3 summary:", flush=True)
        print_table(results_all, "baseline")
        save(results_all, "math15_stage3_results.json")

        # Find best combination for Stage 4
        combo_keys = [k for k in results_all if k.startswith("L") and "+" in k]
        if combo_keys:
            best_combo = max(combo_keys, key=lambda k: results_all[k]["accuracy"])
            best_layers = [int(x) for x in best_combo.replace("L", "").split("+")]
            print(f"  Best combo: {best_combo} → layers {best_layers}", flush=True)
        else:
            best_layers = top3

        # ===== STAGE 4: Per-layer alpha in best combo =====
        print(f"\n{'='*60}", flush=True)
        print(f"STAGE 4: Per-layer alpha vector on {best_layers}", flush=True)
        print(f"{'='*60}", flush=True)
        # Use the best per-layer alphas from Stage 2 or default
        alpha_vec = {l: best_alphas.get(l, 0.1) for l in best_layers}
        print(f"  Alpha vector: {alpha_vec}", flush=True)
        r = evaluate(tok, model, tt, problems, alpha_vec, best_layers)
        lbl = f"L{'+'.join(str(l) for l in best_layers)}_alpha_vec"
        results_all[lbl] = r
        save(results_all, "math15_stage4_results.json")

    print(f"\n{'='*60}", flush=True)
    print(f"FULL PIPELINE COMPLETE", flush=True)
    print(f"  Total time: {(time.time()-overall_t0)/60:.0f} min", flush=True)
    print(f"{'='*60}", flush=True)
    print_table(results_all, "baseline")
    save(results_all, "math15_full_results.json")


if __name__ == "__main__":
    main()
