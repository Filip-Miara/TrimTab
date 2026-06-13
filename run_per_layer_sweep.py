#!/usr/bin/env python3 -u
"""Per-layer steering sweep: find "trim tab" layers for Qwen2.5-7B.

Steers one layer at a time (all others unmodified) to measure
each layer's contribution to accuracy change.
"""
from __future__ import annotations

import gc, json, os, re, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MAX_GEN = 200
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
D_MODEL = 3584
N_LAYERS = 28
N_KV_HEADS = 4
HEAD_DIM = 128


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def steer_single_layer(model, hidden_states, velocity, past_key_values, alpha, target_layer):
    """Steer only one specific layer's KV cache."""
    h_actual = hidden_states[target_layer + 1][0, -1, :]
    v = velocity[0, target_layer, :]
    h_steered = h_actual + alpha * v

    layer = model.model.layers[target_layer]
    k = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))
    v_out = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
    k = k.view(1, N_KV_HEADS, 1, HEAD_DIM)
    v_out = v_out.view(1, N_KV_HEADS, 1, HEAD_DIM)
    lc = past_key_values.layers[target_layer]
    lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
    lc.values[0, :, -1:, :] = v_out.to(lc.values.dtype)


def evaluate_layer(problems, tok, model, tt, alpha, target_layer):
    correct, total = 0, 0
    t0 = time.time()

    for idx, prob in enumerate(problems):
        msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
        input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_tensors="pt").input_ids.to(DEVICE)

        if alpha == 0.0 or target_layer is None:
            am = input_ids.ne(tok.pad_token_id).long() if tok.pad_token_id is not None else None
            out = model.generate(input_ids, attention_mask=am, max_new_tokens=MAX_GEN,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        else:
            past, generated_tokens, first_step = None, [], True
            for step in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)
                next_tok = fwd.logits[0, -1, :].argmax().item()
                if next_tok == tok.eos_token_id: break
                generated_tokens.append(next_tok)

                if not first_step:
                    hs = fwd.hidden_states
                    h_pos = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                    x = h_pos.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        v = tt(x)
                    steer_single_layer(model, hs, v, fwd.past_key_values, alpha, target_layer)

                past = fwd.past_key_values
                input_ids = torch.tensor([[next_tok]], device=DEVICE)
                first_step = False
            gen = tok.decode(generated_tokens, skip_special_tokens=True)

        ca = re.search(r"####\s*(-?\d+)", prob["answer"])
        predicted = extract_number(gen)
        if predicted is not None and ca and predicted == ca.group(1):
            correct += 1
        total += 1

        if (idx + 1) % 5 == 0:
            label = f"L{target_layer:2d}" if target_layer is not None else "base"
            print(f"  {label} α={alpha} [{idx+1}/{len(problems)}] acc={correct}/{total} "
                  f"({100*correct/total:.0f}%)", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tt-path", type=str, default="best_gen_tt_7b.pt")
    args = parser.parse_args()

    print("Loading Qwen2.5-7B-Instruct (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print(f"Loading TT from {args.tt_path}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=D_MODEL)
    tt.load_state_dict(torch.load(args.tt_path, map_location="cpu"), strict=False)
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    results = {}

    # Baseline
    print(f"\n{'='*60}", flush=True)
    print("Baseline (no steering)", flush=True)
    r = evaluate_layer(problems, tok, model, tt, 0.0, None)
    results["baseline"] = r
    print(f"  Baseline: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    # Single-layer steering for each layer
    for li in range(N_LAYERS):
        print(f"\n{'='*60}", flush=True)
        print(f"Layer {li} only, α={args.alpha}", flush=True)
        r = evaluate_layer(problems, tok, model, tt, args.alpha, li)
        results[f"L{li}"] = r
        print(f"  L{li}: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"PER-LAYER SWEEP SUMMARY (α={args.alpha}, {args.n_test} problems)", flush=True)
    print(f"{'='*60}", flush=True)
    baseline_acc = results["baseline"]["accuracy"]
    print(f"  {'Layer':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    print(f"  {'base':>6} {100*baseline_acc:7.1f}%")
    for li in range(N_LAYERS):
        r = results[f"L{li}"]
        acc = r["accuracy"]
        delta = acc - baseline_acc
        marker = " ← BEST" if acc > baseline_acc else ""
        print(f"  L{li:3d} {100*acc:7.1f}% {100*delta:+7.1f}pp{marker}", flush=True)

    with open("per_layer_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved.", flush=True)


if __name__ == "__main__":
    main()
