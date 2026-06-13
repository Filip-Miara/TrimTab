#!/usr/bin/env python3 -u
"""KV-cache steering on Qwen2.5-7B-Instruct with gen-trained TT.

Tests whether generation-trained velocity predictions (R²=0.855)
enable effective steering on a 7B model with 73% GSM8K baseline.
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


def steer_kv_cache(model, hidden_states, velocity, past_key_values, alpha):
    """Modify K/V cache at ALL 28 layers using velocity predictions."""
    for li in range(N_LAYERS):
        h_actual = hidden_states[li + 1][0, -1, :]
        v = velocity[0, li, :]
        h_steered = h_actual + alpha * v

        layer = model.model.layers[li]
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj

        k = k_proj(h_steered.to(torch.bfloat16))    # (512,)
        v = v_proj(h_steered.to(torch.bfloat16))    # (512,)

        k = k.view(1, N_KV_HEADS, 1, HEAD_DIM)
        v = v.view(1, N_KV_HEADS, 1, HEAD_DIM)

        lc = past_key_values.layers[li]
        lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
        lc.values[0, :, -1:, :] = v.to(lc.values.dtype)


def evaluate(problems, tok, model, tt, alpha):
    correct, total = 0, 0
    t0 = time.time()

    for idx, prob in enumerate(problems):
        correct_ans = extract_number(prob["answer"])

        msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
        input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)

        if alpha == 0.0:
            # Use model.generate() for fast baseline
            am = input_ids.ne(tok.pad_token_id).long() if tok.pad_token_id is not None else None
            out = model.generate(input_ids, attention_mask=am,
                                 max_new_tokens=MAX_GEN, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        else:
            past = None
            generated_tokens = []
            first_step = True

            for step in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)
                next_tok = fwd.logits[0, -1, :].argmax().item()
                if next_tok == tok.eos_token_id:
                    break
                generated_tokens.append(next_tok)

                if not first_step:
                    hs = fwd.hidden_states
                    h_pos = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                    x = h_pos.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        v = tt(x)
                    steer_kv_cache(model, hs, v, fwd.past_key_values, alpha)

                past = fwd.past_key_values
                input_ids = torch.tensor([[next_tok]], device=DEVICE)
                first_step = False

            gen = tok.decode(generated_tokens, skip_special_tokens=True)

        predicted = extract_number(gen)
        if predicted is not None and correct_ans is not None and predicted == correct_ans:
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            print(f"  α={alpha} [{idx+1}/{len(problems)}] acc={correct}/{total} "
                  f"({100*correct/total:.0f}%) {time.time()-t0:.0f}s", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.3])
    parser.add_argument("--tt-path", type=str, default="best_gen_tt_7b.pt")
    args = parser.parse_args()

    print("Loading Qwen2.5-7B-Instruct (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    print(f"Loading gen-trained TT from {args.tt_path}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=D_MODEL)
    tt.load_state_dict(torch.load(args.tt_path, map_location="cpu"), strict=False)
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False
    print(f"  {sum(p.numel() for p in tt.parameters()):,} params (R²=0.855)", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    print(f"\nTesting {len(problems)} problems, {len(args.alphas)} alpha values", flush=True)

    results = {}
    for alpha in args.alphas:
        print(f"\n{'='*60}", flush=True)
        print(f"Alpha = {alpha}", flush=True)
        print(f"{'='*60}", flush=True)
        r = evaluate(problems, tok, model, tt, alpha)
        results[alpha] = r

    print(f"\n{'='*60}", flush=True)
    print(f"QWEN2.5-7B KV-CACHE STEERING (gen TT, R²=0.855)", flush=True)
    print(f"{'='*60}", flush=True)
    baseline = results.get(0.0, list(results.values())[0])
    print(f"  {'Alpha':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    for a in sorted(results.keys()):
        r = results[a]
        delta = r['accuracy'] - baseline['accuracy'] if a != 0.0 else 0.0
        print(f"  {a:6.2f} {100*r['accuracy']:7.1f}% {100*delta:+7.1f}pp", flush=True)

    with open("qwen25_7b_steering_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to qwen25_7b_steering_results.json", flush=True)


if __name__ == "__main__":
    main()
