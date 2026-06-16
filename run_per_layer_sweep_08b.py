#!/usr/bin/env python3 -u
"""Per-layer sweep for Qwen3.5-0.8B (24 layers, hybrid FA/GDN).

Steers one layer at a time:
  FA layers:  KV cache patching (proven approach)
  GDN layers: Input modification via forward hook (avoids conv1d bypass)
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

def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None

def steer_fa_layer(model, hidden_states, velocity, pkv, alpha, li, n_kv_heads, head_dim):
    h_actual = hidden_states[li + 1][0, -1, :]
    v = velocity[0, li, :]
    h_steered = h_actual + alpha * v

    layer = model.model.layers[li]
    k = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))
    v_out = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
    k = k.view(1, n_kv_heads, 1, head_dim)
    v_out = v_out.view(1, n_kv_heads, 1, head_dim)
    lc = pkv.layers[li]
    lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
    lc.values[0, :, -1:, :] = v_out.to(lc.values.dtype)

def evaluate(problems, tok, model, tt, alpha, target_layer, model_cfg):
    correct, total = 0, 0
    t0 = time.time()
    n_layers = model_cfg["num_hidden_layers"]
    n_kv_heads = model_cfg["num_key_value_heads"]
    head_dim = model_cfg["head_dim"]
    fa_layers = set(model_cfg.get("full_attention_layers", []))
    is_gdn = target_layer is not None and target_layer not in fa_layers

    # For GDN: pre-forward hook reads velocity from this mutable buffer
    _gdn_v_buffer = [None]
    if is_gdn:
        def _gdn_hook(module, args):
            h = args[0]
            if _gdn_v_buffer[0] is not None:
                h[0, -1, :] = h[0, -1, :] + alpha * _gdn_v_buffer[0].to(h.dtype)
            return (h,)
        _gdn_handle = model.model.layers[target_layer].register_forward_pre_hook(_gdn_hook)

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

                    if target_layer in fa_layers:
                        steer_fa_layer(model, hs, v, fwd.past_key_values, alpha,
                                       target_layer, n_kv_heads, head_dim)
                    else:
                        _gdn_v_buffer[0] = v[0, target_layer, :].detach().clone().float()

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
            print(f"  {lbl} α={alpha} [{idx+1}/{len(problems)}] acc={correct}/{total} "
                  f"({100*correct/total:.0f}%)", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    if is_gdn:
        _gdn_handle.remove()
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tt-path", type=str, default=TT_PATH)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to test (default: all 24)")
    args = parser.parse_args()

    print("Loading Qwen3.5-0.8B (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    cfg = model.config.get_text_config()
    n_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    fa_layers = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    gdn_layers = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
    print(f"  {n_layers} layers, {hidden_size}D, FA={fa_layers}, GDN={gdn_layers}", flush=True)

    model_cfg = {
        "num_hidden_layers": n_layers,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "full_attention_layers": fa_layers,
    }

    print(f"Loading TT from {args.tt_path}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=n_layers, d_input=hidden_size)
    tt.load_state_dict(torch.load(args.tt_path, map_location="cpu"), strict=False)
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    layers_to_test = args.layers if args.layers is not None else list(range(n_layers))
    results = {}

    print(f"\n{'='*60}", flush=True)
    print("Baseline (no steering)", flush=True)
    r = evaluate(problems, tok, model, tt, 0.0, None, model_cfg)
    results["baseline"] = r
    print(f"  Baseline: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    for li in layers_to_test:
        print(f"\n{'='*60}", flush=True)
        layer_type = "FA" if li in fa_layers else "GDN"
        print(f"Layer {li} ({layer_type}) α={args.alpha}", flush=True)
        r = evaluate(problems, tok, model, tt, args.alpha, li, model_cfg)
        results[f"L{li}"] = r
        print(f"  L{li}: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"PER-LAYER SWEEP Qwen3.5-0.8B (α={args.alpha}, {args.n_test} problems)", flush=True)
    print(f"{'='*60}", flush=True)
    baseline_acc = results["baseline"]["accuracy"]
    print(f"  {'Layer':>6} {'Type':>4} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*4} {'-'*8} {'-'*8}")
    print(f"  {'base':>6} {'':4} {100*baseline_acc:7.1f}%")
    best_l, best_d = None, -999
    worst_l, worst_d = None, 999
    for li in layers_to_test:
        r = results[f"L{li}"]
        acc = r["accuracy"]
        delta = acc - baseline_acc
        if delta > best_d: best_d, best_l = delta, li
        if delta < worst_d: worst_d, worst_l = delta, li
        lt = "FA" if li in fa_layers else "GDN"
        marker = " ← BEST" if delta >= best_d else (" ← WORST" if delta <= worst_d else "")
        print(f"  L{li:3d} {lt:>4} {100*acc:7.1f}% {100*delta:+7.1f}pp{marker}", flush=True)
    print(f"\n  Best:  L{best_l} ({100*best_d:+.1f}pp)", flush=True)
    print(f"  Worst: L{worst_l} ({100*worst_d:+.1f}pp)", flush=True)

    out = "per_layer_sweep_08b.json"
    with open(out, "w") as f:
        json.dump({"model": "Qwen3.5-0.8B", "alpha": args.alpha, "n_test": args.n_test,
                    "baseline": baseline_acc, "full_attention_layers": fa_layers,
                    "results": results}, f, indent=2)
    print(f"\nSaved to {out}", flush=True)

if __name__ == "__main__":
    main()
