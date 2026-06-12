#!/usr/bin/env python3
"""Recurrent-state steering for linear attention (Qwen3.5 Gated DeltaNet).

Architecture discovery: Qwen3.5-2B uses LINEAR attention (Gated DeltaNet),
NOT standard MHA. The KV cache stores recurrent states [1, 16, 128, 128]
per layer, not key/value pairs.

Steering approach: inject velocity-predicted perturbations into the hidden
state at each layer BEFORE the linear_attn computation. This affects:
  1. The QKV projections → changes the recurrent state update
  2. The gate computations → changes forget/input gates
  3. All FUTURE token computations through the modified recurrent state

Unlike standard KV-cache steering (modify K/V entries for one position),
recurrent-state steering modifies the attention memory itself.
"""
from __future__ import annotations

import gc, json, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
MAX_GEN = 100


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def make_velocity_hooks(model, velocity_predictions, alpha):
    """Create forward pre-hooks that inject velocity at each layer.

    Args:
        model: HuggingFace model
        velocity_predictions: list of (B, D) tensors, one per layer
        alpha: steering strength
    Returns:
        hooks: list of hook handles (call .remove() to clean up)
    """
    hooks = []
    for layer_idx in range(23):  # layers 0..22 get velocity v[0..22]
        v = velocity_predictions[layer_idx]  # (D,)

        def hook_fn(module, args, layer_v=v, layer_alpha=alpha):
            # args[0] is the hidden state (B, S, D) before input_layernorm
            h = args[0]
            if h.shape[1] > 0:
                h[:, -1:, :] = h[:, -1:, :] + layer_alpha * layer_v.to(h.dtype).to(h.device)
            return (h,) + args[1:]

        layer = model.model.layers[layer_idx]
        hook = layer.register_forward_pre_hook(hook_fn)
        hooks.append(hook)
    return hooks


def evaluate(problems, tok, model, tt, alpha):
    """Evaluate GSM8K accuracy with recurrent-state steering.

    Flow per step:
      1. Forward pass (no hooks) → get hidden states + logits
      2. Compute velocity from hidden states
      3. Register hooks for NEXT step's forward pass
      4. Get next token from logits, update past
    This means steering lags by one step: step t's velocity
    injects into step t+1's forward pass.
    """
    correct, total = 0, 0
    t0 = time.time()

    for idx, prob in enumerate(problems):
        prompt = f"Q: {prob['question']}\nA:"
        correct_ans = parse_answer(prob["answer"])

        input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids
        plen = input_ids.shape[1]
        past = None
        generated_tokens = []
        hooks = []  # velocity hooks for NEXT step
        first_step = True

        for step in range(MAX_GEN):
            # Remove previous step's hooks before this forward
            for h in hooks:
                h.remove()
            hooks = []

            with torch.no_grad():
                fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)

            logits = fwd.logits[0, -1, :]
            next_tok = logits.argmax().item()
            generated_tokens.append(next_tok)

            if next_tok == tok.eos_token_id:
                break

            # Get velocities for NEXT step's hooks
            # Use hidden states from this step's forward
            if alpha > 0 and step < MAX_GEN - 1 and not first_step:
                hs = fwd.hidden_states
                h_pos = torch.stack([h[0, -1, :].float() for h in hs[:24]], dim=0)
                x = h_pos[:23].unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    v = tt(x)
                velocities = [v[0, li, :] for li in range(23)]
                hooks = make_velocity_hooks(model, velocities, alpha)

            past = fwd.past_key_values
            input_ids = torch.tensor([[next_tok]], device=DEVICE)
            first_step = False

        # Clean up
        for h in hooks:
            h.remove()

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
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.3])
    parser.add_argument("--tt-path", type=str, default="best_trajectory_transformer.pt")
    args = parser.parse_args()

    print("Loading model (Qwen3.5-2B, Gated DeltaNet)...")
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
    print(f"RECURRENT-STATE STEERING (Linear Attention)")
    print(f"{'='*60}")
    print(f"  {'Alpha':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    baseline = results.get(0.0, list(results.values())[0])
    for a in sorted(results.keys()):
        r = results[a]
        delta = r['accuracy'] - baseline['accuracy'] if a != 0.0 else 0.0
        print(f"  {a:6.2f} {100*r['accuracy']:7.1f}% {100*delta:+7.1f}pp")

    with open("recurrent_steering_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to recurrent_steering_results.json")


if __name__ == "__main__":
    main()
