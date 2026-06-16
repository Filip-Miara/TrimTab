#!/usr/bin/env python3 -u
"""Sweep unexplored late layers on Qwen2.5-7B.

Tests L20, L22, L23, L25, L26 — flagged by logit-delta probe
as having highest potential trim-tab signal.
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
N_LAYERS = 28
D_MODEL = 3584
N_KV_HEADS = 4
HEAD_DIM = 128
MAX_GEN = 200
TARGET_LAYERS = [20, 22, 23, 25, 26]

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
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    print("Loading Qwen2.5-7B (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print(f"Loading TT from {TT_PATH}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=D_MODEL).to(DEVICE)
    tt.load_state_dict(torch.load(TT_PATH, map_location="cpu"), strict=False)
    tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    results = {}

    for li in TARGET_LAYERS:
        print(f"\n{'='*60}", flush=True)
        print(f"Layer {li}, α={args.alpha}", flush=True)
        correct, total = 0, 0

        for idx, prob in enumerate(problems):
            msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
            input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                return_tensors="pt").input_ids.to(DEVICE)

            past, generated_tokens, first_step = None, [], True
            for step in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)
                next_tok = fwd.logits[0, -1, :].argmax().item()
                if next_tok == tok.eos_token_id: break
                generated_tokens.append(next_tok)

                if not first_step:
                    hs = fwd.hidden_states
                    hp = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                    x = hp.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        v = tt(x)

                    h_act = hs[li + 1][0, -1, :]
                    v_li = v[0, li, :]
                    h_steered = h_act + args.alpha * v_li

                    layer = model.model.layers[li]
                    k = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))
                    vo = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
                    k = k.view(1, N_KV_HEADS, 1, HEAD_DIM)
                    vo = vo.view(1, N_KV_HEADS, 1, HEAD_DIM)
                    lc = fwd.past_key_values.layers[li]
                    lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
                    lc.values[0, :, -1:, :] = vo.to(lc.values.dtype)

                past = fwd.past_key_values
                input_ids = torch.tensor([[next_tok]], device=DEVICE)
                first_step = False

            gen = tok.decode(generated_tokens, skip_special_tokens=True)
            predicted = extract_number(gen)
            ca = re.search(r"####\s*(-?\d+)", prob["answer"])
            if predicted is not None and ca and predicted == ca.group(1):
                correct += 1
            total += 1

            if (idx + 1) % 5 == 0:
                print(f"  L{li:2d} [{idx+1}/{args.n_test}] acc={correct}/{total} "
                      f"({100*correct/total:.0f}%)", flush=True)
                gc.collect()
                torch.cuda.empty_cache()

        results[f"L{li}"] = {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}
        print(f"  L{li}: {correct}/{total} ({100*correct/max(total,1):.1f}%)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"LATE-LAYER SWEEP — Qwen2.5-7B (α={args.alpha})", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Layer':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    baseline_acc = 0.73  # known 7B baseline
    print(f"  {'base':>6} {100*baseline_acc:7.1f}%")
    best_l, best_d = None, -999
    for li in TARGET_LAYERS:
        r = results[f"L{li}"]
        acc = r["accuracy"]
        delta = acc - baseline_acc
        if delta > best_d: best_d, best_l = delta, li
        marker = " ✓ LATENT TRIM" if delta > 0.05 else (" ✗ TRIM-FAIL" if delta < -0.05 else "")
        print(f"  L{li:3d} {100*acc:7.1f}% {100*delta:+7.1f}pp{marker}", flush=True)
    print(f"\n  Best: L{best_l} ({100*best_d:+.1f}pp)", flush=True)

    with open("late_layer_sweep_7b.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to late_layer_sweep_7b.json", flush=True)

if __name__ == "__main__":
    main()
