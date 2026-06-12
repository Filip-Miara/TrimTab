#!/usr/bin/env python3
"""Hybrid KV-cache steering for Qwen3.5 hybrid attention.

Architecture: 18× GatedDeltaNet (linear) + 6× Full Attention (MHA)
Full Attention layers: 3, 7, 11, 15, 19, 23

Steering: modify K/V cache entries at Full Attention layers using
velocity predictions from TrajectoryTransformer.

For each Full Attention layer l:
  1. h = hidden_states[l+1]  (state after layer l)
  2. v = velocity[l]  (predicted change through layer l)
  3. h' = h + α · v
  4. k' = k_proj(h'), v' = v_proj(h')
  5. Replace K/V cache entries at layer l, last position

This changes what FUTURE tokens attend to at these key global layers.
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
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
MAX_GEN = 100
# Layer 23 excluded: TT doesn't predict v[23] (only 23 transitions 0→22)
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19]
HEAD_DIM = 256
N_KV_HEADS = 2


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def steer_full_attention_layers(model, hidden_states, velocity, past_key_values, alpha):
    """Modify K/V cache entries at Full Attention layers."""
    for li in FULL_ATTN_LAYERS:
        layer = model.model.layers[li]
        # h after layer li = hidden_states[li + 1]
        h_actual = hidden_states[li + 1][0, -1, :]  # (D,)
        v = velocity[0, li, :]  # (D,) — velocity for transition through layer li
        h_steered = h_actual + alpha * v

        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj

        k_steered = k_proj(h_steered.to(k_proj.weight.dtype))  # (512,)
        v_steered = v_proj(h_steered.to(v_proj.weight.dtype))  # (512,)

        # Reshape to (1, n_kv_heads, 1, head_dim)
        k_steered = k_steered.view(1, N_KV_HEADS, 1, HEAD_DIM)
        v_steered = v_steered.view(1, N_KV_HEADS, 1, HEAD_DIM)

        # Replace last position in the cache
        layer_cache = past_key_values.layers[li]
        layer_cache.keys[0, :, -1:, :] = k_steered[0, :, :, :].to(layer_cache.keys.dtype)
        layer_cache.values[0, :, -1:, :] = v_steered[0, :, :, :].to(layer_cache.values.dtype)


def evaluate(problems, tok, model, tt, alpha):
    """Evaluate GSM8K accuracy with hybrid KV-cache steering."""
    correct, total = 0, 0
    t0 = time.time()

    for idx, prob in enumerate(problems):
        prompt = f"Q: {prob['question']}\nA:"
        correct_ans = parse_answer(prob["answer"])

        input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids
        plen = input_ids.shape[1]
        past = None
        generated_tokens = []
        first_step = True

        for step in range(MAX_GEN):
            with torch.no_grad():
                fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)

            logits = fwd.logits[0, -1, :]
            next_tok = logits.argmax().item()
            generated_tokens.append(next_tok)

            if next_tok == tok.eos_token_id:
                break

            # Steer K/V cache for Full Attention layers
            if alpha > 0 and not first_step:
                hs = fwd.hidden_states
                h_pos = torch.stack([h[0, -1, :].float() for h in hs[:24]], dim=0)
                x = h_pos[:23].unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    v = tt(x)
                steer_full_attention_layers(model, hs, v, fwd.past_key_values, alpha)

            past = fwd.past_key_values
            input_ids = torch.tensor([[next_tok]], device=DEVICE)
            first_step = False

        gen_text = tok.decode(generated_tokens, skip_special_tokens=True)
        predicted = parse_answer(gen_text)
        if predicted is not None and correct_ans is not None and predicted == correct_ans:
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(problems)}] α={alpha} acc={correct}/{total} "
                  f"({100*correct/total:.0f}%) {time.time()-t0:.0f}s")
            gc.collect(); torch.cuda.empty_cache()

    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.3, 0.5])
    parser.add_argument("--tt-path", type=str, default="best_trajectory_transformer.pt")
    args = parser.parse_args()

    print("Loading Qwen3.5-2B (hybrid: 18×GatedDeltaNet + 6×FullAttention)...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    print(f"Loading TT from {args.tt_path}...")
    tt = TrajectoryTransformer(d_model=512, n_layers=6, n_heads=8, d_ff=2048, d_input=2048, n_positions=23)
    tt.load_state_dict(torch.load(args.tt_path, map_location="cpu"), strict=False)
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False
    print(f"  {sum(p.numel() for p in tt.parameters()):,} params")

    print(f"Steering Full Attention layers: {FULL_ATTN_LAYERS}")

    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    print(f"\nTesting {len(problems)} problems, {len(args.alphas)} alpha values")

    results = {}
    for alpha in args.alphas:
        print(f"\n{'='*60}")
        print(f"Alpha = {alpha}")
        print(f"{'='*60}")
        r = evaluate(problems, tok, model, tt, alpha)
        results[alpha] = r

    print(f"\n{'='*60}")
    print(f"HYBRID KV-CACHE STEERING (6 Full Attention layers)")
    print(f"{'='*60}")
    print(f"  {'Alpha':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    baseline = results.get(0.0, list(results.values())[0])
    for a in sorted(results.keys()):
        r = results[a]
        delta = r['accuracy'] - baseline['accuracy'] if a != 0.0 else 0.0
        print(f"  {a:6.2f} {100*r['accuracy']:7.1f}% {100*delta:+7.1f}pp")

    with open("hybrid_steering_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to hybrid_steering_results.json")


if __name__ == "__main__":
    main()
