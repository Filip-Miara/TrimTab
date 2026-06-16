#!/usr/bin/env python3 -u
"""Death sign flip on L15 (0.8B FA death layer).

Tests whether steering in the OPPOSITE direction (negative α) on the
death layer recovers accuracy — validating the phase inversion theory.
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
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
TT_PATH = "best_tt_08b.pt"
MAX_GEN = 256
ALPHAS = [-0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2]
N_LAYERS = 24
TARGET_LAYER = 15  # death layer on 0.8B

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
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--tt-path", type=str, default=TT_PATH)
    args = parser.parse_args()

    print("Loading Qwen3.5-0.8B (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    cfg = model.config.get_text_config()
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    print(f"Loading TT from {args.tt_path}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=cfg.hidden_size)
    tt.load_state_dict(torch.load(args.tt_path, map_location="cpu"), strict=False)
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    print(f"\nL{TARGET_LAYER} sign flip ({args.n_test} problems, {len(args.alphas)} alphas)", flush=True)

    results = {a: {"correct": 0, "total": 0} for a in args.alphas}
    t0 = time.time()

    for idx, prob in enumerate(problems):
        msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
        input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_tensors="pt").input_ids.to(DEVICE)

        for alpha in args.alphas:
            if alpha == 0.0:
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

                        h_actual = hs[TARGET_LAYER + 1][0, -1, :]
                        v_li = v[0, TARGET_LAYER, :]
                        h_steered = h_actual + alpha * v_li

                        layer = model.model.layers[TARGET_LAYER]
                        k = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))
                        v_out = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
                        k = k.view(1, n_kv_heads, 1, head_dim)
                        v_out = v_out.view(1, n_kv_heads, 1, head_dim)
                        lc = fwd.past_key_values.layers[TARGET_LAYER]
                        lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
                        lc.values[0, :, -1:, :] = v_out.to(lc.values.dtype)

                    past = fwd.past_key_values
                    input_ids = torch.tensor([[next_tok]], device=DEVICE)
                    first_step = False
                gen = tok.decode(generated_tokens, skip_special_tokens=True)

            predicted = extract_number(gen)
            ca = re.search(r"####\s*(-?\d+)", prob["answer"])
            if predicted is not None and ca and predicted == ca.group(1):
                results[alpha]["correct"] += 1
            results[alpha]["total"] += 1

        if (idx + 1) % 5 == 0:
            status = " | ".join(f"α={a}: {results[a]['correct']}/{results[a]['total']}" for a in args.alphas)
            print(f"  [{idx+1}/{args.n_test}] {status}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*60}", flush=True)
    print(f"SIGN FLIP — L{TARGET_LAYER} (death layer at α=0.1)", flush=True)
    print(f"{'='*60}", flush=True)
    baseline_acc = results[0.0]["correct"] / max(results[0.0]["total"], 1)
    print(f"  {'α':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    for a in sorted(results.keys()):
        r = results[a]
        acc = r["correct"] / max(r["total"], 1)
        d = acc - baseline_acc
        print(f"  {a:6.2f} {100*acc:7.1f}% {100*d:+7.1f}pp", flush=True)

    out = {"layer": TARGET_LAYER, "alphas": {str(a): results[a]["correct"] / max(results[a]["total"], 1)
                                               for a in results}, "baseline": baseline_acc}
    with open(f"death_sign_flip_L{TARGET_LAYER}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to death_sign_flip_L{TARGET_LAYER}.json", flush=True)
    print(f"Time: {(time.time()-t0)/60:.0f} min", flush=True)

if __name__ == "__main__":
    main()
