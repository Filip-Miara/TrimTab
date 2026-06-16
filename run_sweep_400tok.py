#!/usr/bin/env python3 -u
"""400-token verification sweep.

Re-runs baseline + top trim-tabs with MAX_GEN=400 for fair comparison.
Also adds per-problem logging to enable problem-layer affinity analysis.
"""
from __future__ import annotations

import gc, json, re, sys, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TT_PATH = "best_gen_tt_7b.pt"
N_LAYERS = 28; D_MODEL = 3584; N_KV_HEADS = 4; HEAD_DIM = 128
MAX_GEN = 400; ALPHA = 0.1
LAYERS = [2, 3, 5, 8, 10, 11, 12, 13, 14, 17, 20]

def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    print("Loading Qwen2.5-7B (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval(); tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print(f"Loading TT...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=D_MODEL).to(DEVICE)
    tt.load_state_dict(torch.load(TT_PATH, map_location="cpu"), strict=False)
    tt.eval(); [p.requires_grad_(False) for p in tt.parameters()]

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    def evaluate_one(prob):
        msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
        iids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_tensors="pt").input_ids.to(DEVICE)
        am = iids.ne(tok.pad_token_id).long()
        out = model.generate(iids, attention_mask=am, max_new_tokens=MAX_GEN,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, iids.shape[1]:], skip_special_tokens=True)
        pa = extract_number(gen)
        ca = re.search(r"####\s*(-?\d+)", prob["answer"])
        return 1 if (pa and ca and pa == ca.group(1)) else 0

    def evaluate_steered(prob, li):
        msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
        input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_tensors="pt").input_ids.to(DEVICE)
        past, generated_tokens, first_step = None, [], True
        for step in range(MAX_GEN):
            with torch.no_grad():
                fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)
            nt = fwd.logits[0, -1, :].argmax().item()
            if nt == tok.eos_token_id: break
            generated_tokens.append(nt)
            if not first_step:
                hs = fwd.hidden_states
                hp = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                x = hp.unsqueeze(0).to(DEVICE)
                with torch.no_grad(): v = tt(x)
                h_act = hs[li + 1][0, -1, :]
                h_steered = h_act + args.alpha * v[0, li, :]
                layer = model.model.layers[li]
                k = layer.self_attn.k_proj(h_steered.to(torch.bfloat16)).view(1, N_KV_HEADS, 1, HEAD_DIM)
                vo = layer.self_attn.v_proj(h_steered.to(torch.bfloat16)).view(1, N_KV_HEADS, 1, HEAD_DIM)
                pkv_l = fwd.past_key_values.layers[li]
                pkv_l.keys[0, :, -1:, :] = k.to(pkv_l.keys.dtype)
                pkv_l.values[0, :, -1:, :] = vo.to(pkv_l.values.dtype)
            past = fwd.past_key_values
            input_ids = torch.tensor([[nt]], device=DEVICE)
            first_step = False
        gen = tok.decode(generated_tokens, skip_special_tokens=True)
        pa = extract_number(gen)
        ca = re.search(r"####\s*(-?\d+)", prob["answer"])
        return 1 if (pa and ca and pa == ca.group(1)) else 0

    results = {}

    # Baseline per-problem
    print(f"\nBaseline (400 tok)...", flush=True)
    base_correct = []
    for i in range(args.n_test):
        c = evaluate_one(problems[i])
        base_correct.append(c)
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{args.n_test}] acc={sum(base_correct)}/{i+1} ({100*sum(base_correct)/(i+1):.0f}%)", flush=True)
    baseline_acc = sum(base_correct) / len(base_correct)
    results["baseline"] = {"correct": sum(base_correct), "total": len(base_correct),
                            "per_problem": base_correct, "accuracy": baseline_acc}
    print(f"  Baseline: {sum(base_correct)}/{len(base_correct)} ({100*baseline_acc:.1f}%)", flush=True)

    # Steered per-problem
    for li in LAYERS:
        print(f"\nLayer {li}...", flush=True)
        correct_list = []
        t0 = time.time()
        for i in range(args.n_test):
            c = evaluate_steered(problems[i], li)
            correct_list.append(c)
            if (i+1) % 10 == 0:
                print(f"  [{i+1}/{args.n_test}] acc={sum(correct_list)}/{i+1} ({100*sum(correct_list)/(i+1):.0f}%)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        acc = sum(correct_list) / len(correct_list)
        results[f"L{li}"] = {"correct": sum(correct_list), "total": len(correct_list),
                              "per_problem": correct_list, "accuracy": acc}
        print(f"  L{li}: {sum(correct_list)}/{len(correct_list)} ({100*acc:.1f}%) Δ={100*(acc-baseline_acc):+.1f}pp [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"400-TOKEN SWEEP (α={args.alpha})", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Layer':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'base':>6} {100*baseline_acc:7.1f}%")
    best_l, best_d = None, -999
    for li in LAYERS:
        r = results[f"L{li}"]
        acc = r["accuracy"]
        d = acc - baseline_acc
        if d > best_d: best_d, best_l = d, li
        m = " ◀ BEST" if d >= best_d else ""
        print(f"  L{li:3d} {100*acc:7.1f}% {100*d:+7.1f}pp{m}", flush=True)
    print(f"\n  Best: L{best_l} ({100*best_d:+.1f}pp)", flush=True)

    with open("sweep_400tok.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to sweep_400tok.json", flush=True)

if __name__ == "__main__":
    main()
