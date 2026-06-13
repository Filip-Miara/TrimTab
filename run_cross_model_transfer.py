#!/usr/bin/env python3 -u
"""Cross-model steering transfer: apply TT from one model to another.

Key insight: the TT's transformer body works in d_model=512 latent space.
Only input_proj (d_input→512) and output_proj (512→d_input) are
dimension-specific. We transplant the body and adapt the projections.

This tests whether velocity prediction is learning a model-agnostic
function of the hidden state trajectory.
"""
from __future__ import annotations

import gc, glob, json, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MAX_GEN = 100


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def load_domain_transferred_tt(source_tt_path, source_d_input, target_d_input):
    """Load TT from source model, transplant body to target dimensions."""
    # Load source TT weights — figure out d_model from checkpoint keys
    chk = torch.load(source_tt_path, map_location="cpu")
    d_model = chk["input_proj.weight"].shape[0]  # d_model = output dim of input_proj
    n_positions = chk["pos_embed.weight"].shape[0]  # number of layers
    print(f"  Source TT: d_model={d_model}, n_positions={n_positions}, d_input={source_d_input}", flush=True)

    src_tt = TrajectoryTransformer(
        d_model=d_model, n_layers=6, n_heads=8, d_ff=d_model*4,
        n_positions=n_positions, d_input=source_d_input,
    )
    src_tt.load_state_dict(chk, strict=False)

    # Create target TT with same d_model, target d_input + n_positions
    tgt_tt = TrajectoryTransformer(
        d_model=d_model, n_layers=6, n_heads=8, d_ff=d_model*4,
        n_positions=n_positions, d_input=target_d_input,
    )

    # Transfer weights that match (body layers)
    src = src_tt.state_dict()
    tgt = tgt_tt.state_dict()

    for key in src:
        if key in tgt and src[key].shape == tgt[key].shape:
            tgt[key].copy_(src[key])
        elif key in tgt and 'pos_embed' in key:
            # Handle different n_positions by copying first N positions
            min_pos = min(src[key].shape[0], tgt[key].shape[0])
            tgt[key][:min_pos].copy_(src[key][:min_pos])
            print(f"  Adapted {key}: {src[key].shape} → {tgt[key].shape} (trunc/pad)", flush=True)
        elif key in tgt:
            print(f"  Skipped {key}: {src[key].shape} vs {tgt[key].shape} (shape mismatch)", flush=True)

    # For input_proj and output_proj: handle dimension change
    with torch.no_grad():
        # input_proj.weight: (d_model, source_d_input) → (d_model, target_d_input)
        min_dim_in = min(source_d_input, target_d_input)
        tgt_tt.input_proj.weight[:, :min_dim_in] = src_tt.input_proj.weight[:, :min_dim_in]
        tgt_tt.input_proj.bias.copy_(src_tt.input_proj.bias)
        print(f"  Adapted input_proj: {src_tt.input_proj.weight.shape} → {tgt_tt.input_proj.weight.shape}", flush=True)

        # output_proj.weight: (source_d_input, d_model) → (target_d_input, d_model)
        min_dim_out = min(source_d_input, target_d_input)
        tgt_tt.output_proj.weight[:min_dim_out, :] = src_tt.output_proj.weight[:min_dim_out, :]
        tgt_tt.output_proj.bias[:min_dim_out] = src_tt.output_proj.bias[:min_dim_out]
        print(f"  Adapted output_proj: {src_tt.output_proj.weight.shape} → {tgt_tt.output_proj.weight.shape}", flush=True)

    return tgt_tt


def steer_mha_layers(model, hidden_states, velocity, pkv, alpha, n_kv_heads, head_dim, full_attn_layers=None):
    """Steer KV cache for MHA layers. If full_attn_layers given, only steer those."""
    n_layers = len(hidden_states) - 1
    for li in range(n_layers):
        if full_attn_layers is not None and li not in full_attn_layers:
            continue
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


def steer_gdn_layers(model, hidden_states, velocity, pkv, alpha, gdn_layers):
    """Steer GatedDeltaNet recurrent states for GDN layers."""
    for li in gdn_layers:
        h_actual = hidden_states[li + 1][0, -1, :]
        v = velocity[0, li, :]
        h_steered = h_actual + alpha * v

        layer = model.model.layers[li]
        la = layer.linear_attn
        # GDN: in_proj_qkv projects h → combined qkv
        # We extract k and v components
        qkv = la.in_proj_qkv(h_steered.to(torch.bfloat16))
        hidden_size = h_steered.shape[-1]
        q, k, v_out = qkv.chunk(3, dim=-1)

        # GDN uses multi-head with 16 heads
        # The recurrent state is (1, 16, 128, 128) → 16 heads, 128×128 state
        num_kv_heads = 16
        head_dim = 128
        k = k.view(1, num_kv_heads, head_dim)
        v_out = v_out.view(1, num_kv_heads, head_dim)

        # Get the input gate beta from in_proj_a
        beta = torch.sigmoid(la.in_proj_a(h_steered.to(torch.bfloat16)))  # (1, 16)

        # Update recurrent state: S += β * (k ⊗ v)
        lc = pkv.layers[li]
        delta = beta.unsqueeze(-1) * (k.unsqueeze(-1) @ v_out.unsqueeze(-2))  # (1, 16, 128, 128)
        lc.recurrent_states += delta.to(lc.recurrent_states.dtype)


def evaluate(problems, tok, model, tt, alpha, model_cfg, steer_only_fa=False):
    correct, total = 0, 0
    t0 = time.time()
    n_layers = model_cfg["num_hidden_layers"]
    n_kv_heads = model_cfg.get("num_key_value_heads", 4)
    head_dim = model_cfg.get("head_dim", 128)
    fa_layers = set(model_cfg.get("full_attention_layers", []))
    gdn_layers = [i for i in range(n_layers) if i not in fa_layers] if fa_layers else []

    # Determine which layers to steer
    target_layers = model_cfg.get("steer_layers", None)
    is_hybrid = len(fa_layers) > 0
    if target_layers is None:
        if steer_only_fa and fa_layers:
            target_layers = list(fa_layers)
        elif not steer_only_fa:
            target_layers = list(range(n_layers))
        else:
            target_layers = []

    for idx, prob in enumerate(problems):
        prompt = f"Q: {prob['question']}\nA:"
        input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids

        if alpha == 0.0 or not target_layers:
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
                    h_pos = torch.stack([h[0, -1, :].float() for h in hs[:n_layers]], dim=0)
                    x = h_pos.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        v = tt(x)
                    # Steer only target layers
                    steer_mha_layers(model, hs, v, fwd.past_key_values, alpha,
                                     n_kv_heads, head_dim,
                                     full_attn_layers=target_layers)
                    # Steer GDN layers only if model has hybrid attention
                    gdn_targets = [l for l in target_layers if l not in fa_layers]
                    if gdn_targets and is_hybrid:
                        steer_gdn_layers(model, hs, v, fwd.past_key_values, alpha, gdn_targets)

                past = fwd.past_key_values
                input_ids = torch.tensor([[next_tok]], device=DEVICE)
                first_step = False
            gen = tok.decode(generated_tokens, skip_special_tokens=True)

        predicted = extract_number(gen)
        ca = re.search(r"####\s*(-?\d+)", prob["answer"])
        if predicted is not None and ca and predicted == ca.group(1):
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(problems)}] acc={correct}/{total} "
                  f"({100*correct/total:.0f}%) {time.time()-t0:.0f}s", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


MODEL_REGISTRY = {
    "Qwen2.5-7B": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "num_hidden_layers": 28, "hidden_size": 3584,
        "num_attention_heads": 28, "num_key_value_heads": 4, "head_dim": 128,
        "full_attention_layers": [],
        "quantize": True,
    },
    "Qwen3.5-2B": {
        "path": "Qwen/Qwen3.5-2B",
        "num_hidden_layers": 24, "hidden_size": 2048,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256,
        "full_attention_layers": [3, 7, 11, 15, 19, 23],
        "quantize": False,
    },
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--source-tt", type=str, required=True)
    parser.add_argument("--source-d-input", type=int, required=True)
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--mode", choices=["all", "fa_only", "gdn_only"], default="all")
    parser.add_argument("--layer", type=int, default=None,
                        help="Steer only this specific layer")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Steer these specific layers (runs each independently)")
    args = parser.parse_args()

    mc = MODEL_REGISTRY[args.target_model]
    cache_dir = "/run/media/filip/B522-875D/Datasets/hub"
    model_dir = os.path.join(cache_dir, f"models--{mc['path'].replace('/', '--')}", "snapshots")
    snaps = glob.glob(model_dir + "/*/")
    model_path = snaps[0]

    print(f"Loading {args.target_model} ({mc['num_hidden_layers']}L, {mc['hidden_size']}D)...", flush=True)
    kw = {}
    if mc.get("quantize"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map=DEVICE, **kw)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print(f"Transferring TT from d_input={args.source_d_input}→{mc['hidden_size']}...", flush=True)
    tt = load_domain_transferred_tt(args.source_tt, args.source_d_input, mc["hidden_size"])
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    # Determine which layers to test
    if args.layers:
        test_configs = [{"steer_layers": [l], "alpha": args.alpha, "label": f"L{l}"} for l in args.layers]
    elif args.layer is not None:
        test_configs = [{"steer_layers": [args.layer], "alpha": args.alpha, "label": f"L{args.layer}"}]
    else:
        test_configs = [{"steer_layers": None, "alpha": 0.0, "label": "baseline"}]
        if args.mode == "fa_only":
            test_configs.append({"steer_layers": list(mc["full_attention_layers"]), "alpha": args.alpha, "label": "all_FA"})
        elif args.mode != "gdn_only":
            test_configs.append({"steer_layers": None, "alpha": args.alpha, "label": "all_layers"})

    results = {}
    for cfg in test_configs:
        mc_copy = dict(mc)
        if cfg["steer_layers"] is not None:
            mc_copy["steer_layers"] = cfg["steer_layers"]

        is_steered = cfg["alpha"] > 0.0 and cfg["steer_layers"] is not None
        steer_only_fa = is_steered and any(l in mc["full_attention_layers"] for l in (
            cfg["steer_layers"] if cfg["steer_layers"] else []))

        print(f"\n{'='*60}", flush=True)
        print(f"  {cfg['label']} (α={cfg['alpha']})", flush=True)
        print(f"{'='*60}", flush=True)
        r = evaluate(problems, tok, model, tt, cfg["alpha"], mc_copy, steer_only_fa=steer_only_fa)
        results[cfg["label"]] = r
        print(f"  {cfg['label']}: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"CROSS-MODEL TRANSFER SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    ba = results.get("baseline", {}).get("accuracy", 0)
    print(f"  {'Config':>12} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8}")
    for label, r in sorted(results.items()):
        acc = r["accuracy"]
        delta = acc - ba if label != "baseline" else 0
        print(f"  {label:>12} {100*acc:7.1f}% {100*delta:+7.1f}pp", flush=True)

    out = f"transfer_{args.target_model.lower()}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
