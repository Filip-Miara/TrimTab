#!/usr/bin/env python3
"""KV-cache downstream accuracy: Phase 2/3 on GSM8K.

Tests whether KV-cache steering improves generation accuracy, using
layer-to-layer velocity predictions from TrajectoryTransformer.

Hypothesis: modifying KV entries changes what FUTURE tokens attend to,
affecting the entire generation trajectory, not just the next token.

Key metric: GSM8K answer accuracy vs baseline.
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
N_KV_HEADS = 2
HEAD_DIM = 256


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def steer_kv_cache(model, hidden_states, velocity, past_key_values, alpha):
    """Steer all layers' KV cache entries using velocity predictions."""
    L = velocity.shape[1]  # 23 transitions
    new_pkv = []
    for layer_idx in range(L):
        h_actual = hidden_states[layer_idx + 1][0, -1, :]
        v = velocity[0, layer_idx, :]
        h_steered = h_actual + alpha * v

        layer = model.model.layers[layer_idx]
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj

        k_steered = k_proj(h_steered.to(k_proj.weight.dtype))
        v_steered = v_proj(h_steered.to(v_proj.weight.dtype))

        k_steered = k_steered.view(1, N_KV_HEADS, HEAD_DIM).unsqueeze(0)
        v_steered = v_steered.view(1, N_KV_HEADS, HEAD_DIM).unsqueeze(0)

        k_cache, v_cache = past_key_values[layer_idx]
        k_cache[0, :, -1:, :] = k_steered[0, :, :, :]
        v_cache[0, :, -1:, :] = v_steered[0, :, :, :]
        new_pkv.append((k_cache, v_cache))

    return tuple(new_pkv)


def evaluate(problems, tok, model, tt, alpha, label=""):
    """Evaluate GSM8K accuracy with KV-cache steering at given alpha."""
    correct, total = 0, 0
    t0 = time.time()

    for idx, prob in enumerate(problems):
        prompt = f"Q: {prob['question']}\nA:"
        correct_ans = parse_answer(prob["answer"])

        # Initial forward pass with full prompt
        input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids
        plen = input_ids.shape[1]
        past = None
        generated_tokens = []

        for step in range(MAX_GEN):
            with torch.no_grad():
                fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)

            logits = fwd.logits[0, -1, :]
            next_tok = logits.argmax().item()
            generated_tokens.append(next_tok)

            if next_tok == tok.eos_token_id:
                break

            if alpha > 0:
                hs = fwd.hidden_states
                h_pos = torch.stack([h[0, -1, :].float() for h in hs], dim=0)
                x = h_pos[:23].unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    v = tt(x)

                past = steer_kv_cache(model, hs, v, fwd.past_key_values, alpha=alpha)
            else:
                past = fwd.past_key_values

            # Next step: feed only the new token (KV cache handles the rest)
            input_ids = torch.tensor([[next_tok]], device=DEVICE)

        gen_text = tok.decode(generated_tokens, skip_special_tokens=True)
        predicted = parse_answer(gen_text)
        if predicted is not None and correct_ans is not None and predicted == correct_ans:
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(problems)}] {label} acc={correct}/{total} "
                  f"({100*correct/total:.0f}%) {time.time()-t0:.0f}s")
            gc.collect(); torch.cuda.empty_cache()

    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--tt-path", type=str, default="best_trajectory_transformer.pt")
    args = parser.parse_args()

    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    print(f"Loading TrajectoryTransformer from {args.tt_path}...")
    tt = TrajectoryTransformer(d_model=512, n_layers=6, n_heads=8, d_ff=2048, d_input=2048, n_positions=23)
    tt.load_state_dict(torch.load(args.tt_path, map_location="cpu"), strict=False)
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False
    print(f"  {sum(p.numel() for p in tt.parameters()):,} params")

    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    print(f"\nTesting {len(problems)} problems with {len(args.alphas)} alpha values")

    results = {}
    for alpha in args.alphas:
        label = f"α={alpha}"
        print(f"\n{'='*60}")
        print(f"Alpha = {alpha}")
        print(f"{'='*60}")
        r = evaluate(problems, tok, model, tt, alpha, label=label)
        results[alpha] = r

    print(f"\n{'='*60}")
    print(f"KV-CACHE STEERING: DOWNSTREAM ACCURACY")
    print(f"{'='*60}")
    print(f"  {'Alpha':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    baseline = results.get(0.0, results.get(list(results.keys())[0]))
    for a in sorted(results.keys()):
        r = results[a]
        delta = r['accuracy'] - baseline['accuracy'] if a != 0.0 else 0.0
        print(f"  {a:6.2f} {100*r['accuracy']:7.1f}% {100*delta:+7.1f}pp")

    with open("kvcache_accuracy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to kvcache_accuracy_results.json")


if __name__ == "__main__":
    main()
