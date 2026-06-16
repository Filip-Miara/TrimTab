#!/usr/bin/env python3 -u
"""Epoch 1 protocol: 4-condition × 28-layer sweep with statistics.

Conditions:
  C1: No steering (baseline)
  C2: Random vector (same norm as TT prediction)
  C3: Standard TT prediction
  C4: Contrastive TT (v_correct - v_incorrect)

Statistics: Bonferroni correction, bootstrap confidence intervals.
"""
from __future__ import annotations

import gc, json, math, os, re, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
D_MODEL = 3584
N_LAYERS = 28
N_KV_HEADS = 4
HEAD_DIM = 128
MAX_GEN = 200
N_TEST = 50  # problems per condition
N_CONDITIONS = 4
N_TESTS = N_CONDITIONS * N_LAYERS  # 112
ALPHA_THRESH = 0.000446  # Bonferroni: 0.05 / 112

examples = ('Q: Janet has 5 ducks. She buys 3 more. How many?\nA: Step 1: 5. Step 2: buy 3. Step 3: 5+3=8. So answer is 8.\n\n'
            'Q: 12 muffins/hr, 8 hours?\nA: Step 1: 12/hr. Step 2: 8h. Step 3: 12x8=96. So answer is 96.\n\n')


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"####\s*(-?\d+)", r"So answer is\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def steer_layer(model, hs, v, pkv, li, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM):
    """Apply steering vector v (shape: (1, N_LAYERS, D)) at layer li only."""
    h = hs[li + 1][0, -1, :] + v[0, li, :].to(hs[li + 1].dtype)
    l = model.model.layers[li]
    k = l.self_attn.k_proj(h.to(torch.bfloat16)).view(1, n_kv_heads, 1, head_dim)
    vo = l.self_attn.v_proj(h.to(torch.bfloat16)).view(1, n_kv_heads, 1, head_dim)
    lc = pkv.layers[li]
    lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
    lc.values[0, :, -1:, :] = vo.to(lc.values.dtype)


def bootstrap_ci(accuracies, n_resamples=1000):
    """95% bootstrap confidence interval for accuracy."""
    accs = np.array(accuracies)
    means = [np.mean(np.random.choice(accs, len(accs), replace=True)) for _ in range(n_resamples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_condition(problems, tok, model, layer, cond, tts=None, alpha=0.1):
    """Evaluate one condition at one layer. Returns list of per-problem binaries."""
    results = []
    for idx, prob in enumerate(problems):
        prompt = f"{examples}Q: {prob['question']}\nA:"
        input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids

        if cond == "baseline":
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

                    if cond == "random":
                        v = torch.randn(1, N_LAYERS, D_MODEL, device=DEVICE, dtype=torch.float32)
                        v_norm = v.norm()
                        with torch.no_grad():
                            v_tt = tts["standard"](x)
                        scale = v_tt.norm() / (v_norm + 1e-8)
                        v = v * scale
                    elif cond == "standard":
                        with torch.no_grad():
                            v = tts["standard"](x)
                    elif cond == "contrastive":
                        with torch.no_grad():
                            v_c = tts["correct"](x)
                            v_i = tts["incorrect"](x)
                            v = v_c - v_i

                    steer_layer(model, hs, v * alpha, fwd.past_key_values, layer)

                past = fwd.past_key_values
                input_ids = torch.tensor([[nt]], device=DEVICE)
                first = False
            gen = tok.decode(gens, skip_special_tokens=True)

        ca = re.search(r"####\s*(-?\d+)", prob["answer"])
        pa = extract_number(gen)
        results.append(1 if (pa and ca and pa == ca.group(1)) else 0)

        if (idx + 1) % 10 == 0:
            print(f"  {cond} L{layer} [{idx+1}/{len(problems)}] "
                  f"acc={sum(results)}/{len(results)} ({100*sum(results)/len(results):.0f}%)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--tt-standard", type=str, default="best_gen_tt_7b.pt")
    parser.add_argument("--tt-correct", type=str, default="best_tt_correct.pt")
    parser.add_argument("--tt-incorrect", type=str, default="best_tt_incorrect.pt")
    args = parser.parse_args()

    layers = args.layers if args.layers else list(range(N_LAYERS))
    conditions = ["baseline"]  # run once
    steer_conditions = ["random", "standard", "contrastive"]
    n_tests = 1 + len(steer_conditions) * len(layers)  # 1 baseline + 3×28 = 85
    bonf_alpha = 0.05 / max(n_tests, 1)
    z_threshold = 3.33  # Bonferroni α=0.05/85 ≈ 3.33σ

    print(f"Loading Qwen2.5-7B-Instruct (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print(f"Loading TTs...", flush=True)
    ckpt_std = torch.load(args.tt_standard, map_location="cpu")
    ckpt_correct = torch.load(args.tt_correct, map_location="cpu")
    ckpt_incorrect = torch.load(args.tt_incorrect, map_location="cpu")

    def load_tt(path, ckpt):
        tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                    n_positions=N_LAYERS, d_input=D_MODEL).to(DEVICE)
        sd = ckpt.get("model_state_dict", ckpt)
        tt.load_state_dict(sd, strict=False)
        tt.eval()
        for p in tt.parameters(): p.requires_grad = False
        return tt

    tts = {
        "standard": load_tt(args.tt_standard, ckpt_std),
        "correct": load_tt(args.tt_correct, ckpt_correct),
        "incorrect": load_tt(args.tt_incorrect, ckpt_incorrect),
    }

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    overall_t0 = time.time()

    all_results = {}  # key: "{cond}_L{layer}" or "baseline", value: list of 0/1

    # Baseline — run once (layer-independent)
    print(f"\n{'='*60}", flush=True)
    print(f"  baseline (no steering)", flush=True)
    print(f"{'='*60}", flush=True)
    r = evaluate_condition(problems, tok, model, 0, "baseline", tts, args.alpha)
    all_results["baseline"] = r
    acc = np.mean(r)
    ci_lo, ci_hi = bootstrap_ci(r)
    print(f"  baseline: {sum(r)}/{len(r)} ({100*acc:.1f}%) CI=[{100*ci_lo:.1f},{100*ci_hi:.1f}]", flush=True)

    # Steering conditions — run per layer
    for cond in steer_conditions:
        for li in layers:
            key = f"{cond}_L{li}"
            print(f"\n{'='*60}", flush=True)
            print(f"  {cond} | L{li} | α={args.alpha}", flush=True)
            print(f"{'='*60}", flush=True)
            r = evaluate_condition(problems, tok, model, li, cond, tts, args.alpha)
            all_results[key] = r
            acc = np.mean(r)
            ci_lo, ci_hi = bootstrap_ci(r)
            print(f"  {key}: {sum(r)}/{len(r)} ({100*acc:.1f}%) CI=[{100*ci_lo:.1f},{100*ci_hi:.1f}]", flush=True)

    # Analysis
    print(f"\n{'='*60}", flush=True)
    print(f"EPOCH 1 — 4-CONDITION × {len(layers)}-LAYER PROTOCOL", flush=True)
    print(f"  Bonferroni α={bonf_alpha:.6f} (z_thresh={z_threshold:.2f})", flush=True)
    print(f"  Problems/condition: {args.n_test}", flush=True)
    print(f"{'='*60}", flush=True)

    baseline_accs = all_results.get("baseline", [])
    baseline_mean = np.mean(baseline_accs) if baseline_accs else 0
    baseline_se = np.std(baseline_accs) / np.sqrt(len(baseline_accs)) if baseline_accs else 0
    print(f"  Baseline: {100*baseline_mean:.1f}%", flush=True)

    summary = {"metadata": {
        "n_test": args.n_test, "alpha_steer": args.alpha,
        "bonferroni_alpha": bonf_alpha, "z_threshold": z_threshold,
        "n_tests": n_tests, "layers": layers,
        "conditions": ["baseline"] + steer_conditions,
    }, "results": {"baseline": {
        "accuracy": float(baseline_mean), "n": len(baseline_accs),
    }}, "significant": []}

    print(f"  {'Cond':20s} {'Layer':>5} {'Acc':>7} {'Δ':>7} {'z':>6} {'p_sig':>6} {'CI_lo':>6} {'CI_hi':>6}", flush=True)
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6}", flush=True)

    for cond in steer_conditions:
        for li in layers:
            key = f"{cond}_L{li}"
            if key not in all_results:
                continue
            r = all_results[key]
            acc = np.mean(r)
            se = np.std(r) / np.sqrt(len(r))
            delta = acc - baseline_mean
            z = delta / (baseline_se + 1e-8) if cond != "baseline" else 0
            p_sig = abs(z) > z_threshold
            ci_lo, ci_hi = bootstrap_ci(r)

            print(f"  {cond:20s} L{li:3d} {100*acc:6.1f}% {100*delta:+6.1f}pp {z:5.2f} "
                  f"{'*' if p_sig else '':>6s} {100*ci_lo:5.1f}% {100*ci_hi:5.1f}%", flush=True)

            entry = {"accuracy": float(acc), "delta": float(delta), "z": float(z),
                     "significant": bool(p_sig), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}
            summary["results"][key] = entry
            if p_sig and cond != "baseline":
                summary["significant"].append(key)

    print(f"\n  Significant results (Bonferroni-corrected, z>{z_threshold:.2f}):", flush=True)
    for k in summary["significant"]:
        print(f"    ✅ {k}", flush=True)
    if not summary["significant"]:
        print(f"    None — no result survives correction", flush=True)

    # Key comparison: TT > random?
    tt_better = 0
    for li in layers:
        std_key = f"standard_L{li}"
        rnd_key = f"random_L{li}"
        if std_key in all_results and rnd_key in all_results:
            if np.mean(all_results[std_key]) > np.mean(all_results[rnd_key]):
                tt_better += 1
    print(f"\n  TT beats random on {tt_better}/{len(layers)} layers", flush=True)
    summary["tt_beats_random_on"] = tt_better

    overall_time = time.time() - overall_t0
    print(f"\n  Total time: {overall_time:.0f}s ({overall_time/3600:.1f}h)", flush=True)
    summary["total_time_s"] = overall_time

    with open("epoch1_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to epoch1_results.json", flush=True)


if __name__ == "__main__":
    main()
