#!/usr/bin/env python3 -u
"""KV-cache steering on SmolLM2-360M with generation-trained TT.

Now that we have R²=0.94 on generation trajectories, test if
this enables effective KV-cache steering (modifying K/V entries
based on velocity predictions).
"""
from __future__ import annotations

import gc, glob, json, os, re, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
MAX_GEN = 100
D_MODEL = 960
N_LAYERS = 32
N_KV_HEADS = 5
HEAD_DIM = 64


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def steer_kv_cache(model, hidden_states, velocity, past_key_values, alpha):
    """Modify K/V cache at ALL layers using velocity predictions.

    Args:
        model: SmolLM2 model
        hidden_states: tuple from output_hidden_states=True (N+1 entries)
        velocity: (1, N, D) tensor from TT — predicted per-layer velocities
        past_key_values: DynamicCache with key/value tensors
        alpha: steering strength
    """
    # hidden_states[l+1] = state after layer l
    # velocity[0, l] = predicted change through layer l
    for li in range(N_LAYERS):
        h_actual = hidden_states[li + 1][0, -1, :]  # (D,)
        v = velocity[0, li, :]  # (D,)
        h_steered = h_actual + alpha * v

        layer = model.model.layers[li]
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj

        k_steered = k_proj(h_steered.to(k_proj.weight.dtype))    # (320,)
        v_steered = v_proj(h_steered.to(v_proj.weight.dtype))    # (320,)

        k_steered = k_steered.view(1, N_KV_HEADS, 1, HEAD_DIM)
        v_steered = v_steered.view(1, N_KV_HEADS, 1, HEAD_DIM)

        layer_cache = past_key_values.layers[li]
        layer_cache.keys[0, :, -1:, :] = k_steered.to(layer_cache.keys.dtype)
        layer_cache.values[0, :, -1:, :] = v_steered.to(layer_cache.values.dtype)


def evaluate(problems, tok, model, tt, alpha):
    """Evaluate GSM8K accuracy with KV-cache steering."""
    correct, total = 0, 0
    t0 = time.time()

    examples = ("Q: Janet has 5 ducks. She buys 3 more. How many does she have?\n"
                "A: Step 1: 5 ducks. Step 2: buy 3 more. Step 3: 5+3=8.\nSo answer is 8.\n\n"
                "Q: A bakery sells 12 muffins per hour. How many in 8 hours?\n"
                "A: Step 1: 12/hour. Step 2: 8 hours. Step 3: 12x8=96.\nSo answer is 96.\n\n")

    for idx, prob in enumerate(problems):
        prompt = f"{examples}Q: {prob['question']}\nA:"
        correct_ans = parse_answer(prob["answer"])

        input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids
        past = None
        generated_tokens = []
        first_step = True

        for step in range(MAX_GEN):
            with torch.no_grad():
                fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)

            next_tok = fwd.logits[0, -1, :].argmax().item()
            generated_tokens.append(next_tok)

            if next_tok == tok.eos_token_id:
                break

            # Steer KV cache after the first step
            if alpha > 0 and not first_step:
                hs = fwd.hidden_states
                h_pos = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                x = h_pos.unsqueeze(0).to(DEVICE)  # (1, 32, 960)
                with torch.no_grad():
                    v = tt(x)  # (1, 32, 960)
                steer_kv_cache(model, hs, v, fwd.past_key_values, alpha)

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
                  f"({100*correct/total:.0f}%) {time.time()-t0:.0f}s "
                  f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.3])
    parser.add_argument("--tt-path", type=str, default="best_gen_tt.pt")
    args = parser.parse_args()

    # Find SmolLM2 snapshot
    model_dir = os.path.join(CACHE_DIR, "models--HuggingFaceTB--SmolLM2-360M", "snapshots")
    snaps = glob.glob(model_dir + "/*/")
    if not snaps: raise FileNotFoundError("No SmolLM2-360M")
    model_path = snaps[0]

    print("Loading SmolLM2-360M (standard MHA)...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)

    print(f"Loading generation-trained TT from {args.tt_path}...", flush=True)
    tt = TrajectoryTransformer(d_model=512, n_layers=6, n_heads=8, d_ff=2048,
                                n_positions=N_LAYERS, d_input=D_MODEL)
    tt.load_state_dict(torch.load(args.tt_path, map_location="cpu"), strict=False)
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False
    print(f"  {sum(p.numel() for p in tt.parameters()):,} params (R²=0.94)", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
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
    print(f"SMOLM2-360M KV-CACHE STEERING (gen-trained TT)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Alpha':>6} {'Acc':>8s} {'Δ':>8s}", flush=True)
    print(f"  {'-'*6} {'-'*8} {'-'*8}", flush=True)
    baseline = results.get(0.0, list(results.values())[0])
    for a in sorted(results.keys()):
        r = results[a]
        delta = r['accuracy'] - baseline['accuracy'] if a != 0.0 else 0.0
        print(f"  {a:6.2f} {100*r['accuracy']:7.1f}% {100*delta:+7.1f}pp", flush=True)

    with open("smolm2_steering_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to smolm2_steering_results.json", flush=True)


if __name__ == "__main__":
    main()
