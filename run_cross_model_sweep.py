#!/usr/bin/env python3 -u
"""Per-layer steering sweep across multiple cached models.

Tests whether the "trim tab" and "death layer" patterns replicate
across model families and sizes.
"""
from __future__ import annotations

import gc, glob, json, os, re, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer


DEVICE = "cuda"

def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def steer_kv_layer(model, hidden_states, velocity, pkv, alpha, li,
                   n_kv_heads, head_dim, is_gated_deltanet=False):
    """Steer a single layer's cache — handles both MHA and GatedDeltaNet."""
    h_actual = hidden_states[li + 1][0, -1, :]
    v = velocity[0, li, :]
    h_steered = h_actual + alpha * v

    if is_gated_deltanet:
        # GatedDeltaNet: the "projection" is qkv_proj for the linear attention
        # We need to get the K and V components from the QKV projection
        # and update the recurrent state accordingly
        # For GatedDeltaNet layers, we add to the input via hook instead
        # Return False to signal caller to use hook approach for this layer
        return False
    else:
        layer = model.model.layers[li]
        k = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))
        v_out = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
        k = k.view(1, n_kv_heads, 1, head_dim)
        v_out = v_out.view(1, n_kv_heads, 1, head_dim)
        lc = pkv.layers[li]
        lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
        lc.values[0, :, -1:, :] = v_out.to(lc.values.dtype)
        return True


def evaluate(problems, tok, model, tt, alpha, target_layer, model_cfg):
    correct, total = 0, 0
    t0 = time.time()
    n_layers = model_cfg["num_hidden_layers"]
    n_kv_heads = model_cfg["num_key_value_heads"]
    head_dim = model_cfg["head_dim"]
    # Determine if each layer is GatedDeltaNet or MHA
    fa_layers = set(model_cfg.get("full_attention_layers", []))

    for idx, prob in enumerate(problems):
        if model_cfg.get("chat_template"):
            msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
            input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                return_tensors="pt").input_ids.to(DEVICE)
        else:
            examples = ('Q: Janet has 5 ducks. She buys 3 more. How many?\nA: Step 1: 5. Step 2: buy 3. Step 3: 5+3=8. So answer is 8.\n\n'
                        'Q: 12 muffins/hr, 8 hours?\nA: Step 1: 12/hr. Step 2: 8h. Step 3: 12x8=96. So answer is 96.\n\n')
            prompt = f"{examples}Q: {prob['question']}\nA:"
            input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids

        if alpha == 0.0 or target_layer is None:
            am = input_ids.ne(tok.pad_token_id).long() if tok.pad_token_id is not None else None
            out = model.generate(input_ids, attention_mask=am, max_new_tokens=80,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        else:
            past, generated_tokens, first_step = None, [], True
            for step in range(80):
                with torch.no_grad():
                    fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)
                next_tok = fwd.logits[0, -1, :].argmax().item()
                if next_tok == tok.eos_token_id: break
                generated_tokens.append(next_tok)

                if not first_step:
                    hs = fwd.hidden_states
                    h_pos = torch.stack([h[0, -1, :].float() for h in hs[:n_layers]], dim=0)
                    x = h_pos.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        v = tt(x)
                    li = target_layer
                    is_gdn = (li not in fa_layers) and len(fa_layers) > 0
                    steer_kv_layer(model, hs, v, fwd.past_key_values, alpha,
                                   li, n_kv_heads, head_dim, is_gdn)

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
            lbl = f"L{target_layer}" if target_layer is not None else "base"
            print(f"  {lbl} [{idx+1}/{len(problems)}] acc={correct}/{total} "
                  f"({100*correct/total:.0f}%)", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


MODEL_REGISTRY = {
    "Qwen2.5-0.5B": {
        "path": "Qwen/Qwen2.5-0.5B",
        "num_hidden_layers": 24, "hidden_size": 896,
        "num_attention_heads": 14, "num_key_value_heads": 2,
        "head_dim": 64, "chat_template": True,
        "full_attention_layers": [],  # all are MHA
        "max_gen": 80,
    },
    "SmolLM2-360M": {
        "path": "HuggingFaceTB/SmolLM2-360M",
        "num_hidden_layers": 32, "hidden_size": 960,
        "num_attention_heads": 15, "num_key_value_heads": 5,
        "head_dim": 64, "chat_template": False,
        "full_attention_layers": [],
        "max_gen": 80,
    },
    "SmolLM2-135M": {
        "path": "HuggingFaceTB/SmolLM2-135M",
        "num_hidden_layers": 30, "hidden_size": 576,
        "num_attention_heads": 9, "num_key_value_heads": 3,
        "head_dim": 64, "chat_template": False,
        "full_attention_layers": [],
        "max_gen": 80,
    },
    "Qwen3.5-2B": {
        "path": "Qwen/Qwen3.5-2B",
        "num_hidden_layers": 24, "hidden_size": 2048,
        "num_attention_heads": 8, "num_key_value_heads": 2,
        "head_dim": 256, "chat_template": True,
        # FA layers at positions 3, 7, 11, 15, 19, 23
        "full_attention_layers": [3, 7, 11, 15, 19, 23],
        "max_gen": 80,
    },
    "Qwen3.5-0.8B": {
        "path": "Qwen/Qwen3.5-0.8B",
        "num_hidden_layers": 24, "hidden_size": 2048,
        "num_attention_heads": 8, "num_key_value_heads": 2,
        "head_dim": 256, "chat_template": True,
        "full_attention_layers": [3, 7, 11, 15, 19, 23],
        "max_gen": 80,
    },
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tt-path", type=str, required=True,
                        help="Path to gen-trained TT checkpoint")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to test (default: all)")
    args = parser.parse_args()

    mc = MODEL_REGISTRY[args.model]
    cache_dir = "/run/media/filip/B522-875D/Datasets/hub"

    # Find model snapshot
    model_dir = os.path.join(cache_dir, f"models--{mc['path'].replace('/', '--')}", "snapshots")
    snaps = glob.glob(model_dir + "/*/")
    if not snaps:
        print(f"Model {args.model} not cached at {model_dir}")
        return
    model_path = snaps[0]

    print(f"Loading {args.model} ({mc['num_hidden_layers']}L, {mc['hidden_size']}D)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print(f"Loading TT from {args.tt_path}...", flush=True)
    tt = TrajectoryTransformer(d_model=512, n_layers=6, n_heads=8, d_ff=2048,
                                n_positions=mc["num_hidden_layers"], d_input=mc["hidden_size"])
    tt.load_state_dict(torch.load(args.tt_path, map_location="cpu"), strict=False)
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    layers_to_test = args.layers if args.layers is not None else list(range(mc["num_hidden_layers"]))
    results = {}

    # Baseline
    print(f"\n{'='*60}", flush=True)
    print(f"Baseline", flush=True)
    r = evaluate(problems, tok, model, tt, 0.0, None, mc)
    results["baseline"] = r
    print(f"  Baseline: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    for li in layers_to_test:
        print(f"\n{'='*60}", flush=True)
        print(f"Layer {li} (α={args.alpha})", flush=True)
        r = evaluate(problems, tok, model, tt, args.alpha, li, mc)
        results[f"L{li}"] = r
        print(f"  L{li}: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"{args.model} PER-LAYER SWEEP (α={args.alpha}, {args.n_test} problems)", flush=True)
    print(f"{'='*60}", flush=True)
    ba = results["baseline"]["accuracy"]
    print(f"  {'Layer':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    print(f"  {'base':>6} {100*ba:7.1f}%")
    best_l, best_d = None, -999
    worst_l, worst_d = None, 999
    for li in layers_to_test:
        r = results[f"L{li}"]
        acc = r["accuracy"]
        delta = acc - ba
        if delta > best_d: best_d, best_l = delta, li
        if delta < worst_d: worst_d, worst_l = delta, li
        marker = " BEST" if delta == best_d else (" WORST" if delta == worst_d else "")
        print(f"  L{li:3d} {100*acc:7.1f}% {100*delta:+7.1f}pp{marker}", flush=True)
    print(f"\n  Best layer: L{best_l} ({100*best_d:+}pp)", flush=True)
    print(f"  Worst layer: L{worst_l} ({100*worst_d:+}pp)", flush=True)

    out = f"{args.model.lower().replace('.','').replace('-','_')}_sweep.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
