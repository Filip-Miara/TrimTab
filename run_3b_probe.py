#!/usr/bin/env python3 -u
"""Test Qwen2.5-3B for trim-tabs via cross-model TT transfer (7B→3B).

Quick probe: test 3 layers on 15 problems each. If any show positive Δ,
we switch to 3B for all future experiments.
"""
from __future__ import annotations

import gc, json, re, sys, time
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
TT_PATH = "best_gen_tt_7b.pt"
N_LAYERS_7B = 28
N_LAYERS_3B = 36
D_INPUT_7B = 3584
D_INPUT_3B = 2048
D_MODEL = 768
ALPHA = 0.1
MAX_GEN = 400

MODEL_PATH = "/home/filip/.cache/huggingface/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None

def transfer_tt(source_path):
    """Transfer 7B TT to 3B dimensions (adapt projections + position embeddings)."""
    chk = torch.load(source_path, map_location="cpu")
    d_model = chk["input_proj.weight"].shape[0]
    n_pos_7b = chk["pos_embed.weight"].shape[0]

    src_tt = TrajectoryTransformer(d_model=d_model, n_layers=6, n_heads=8, d_ff=d_model*4,
                                    n_positions=n_pos_7b, d_input=D_INPUT_7B)
    src_tt.load_state_dict(chk, strict=False)

    tgt_tt = TrajectoryTransformer(d_model=d_model, n_layers=6, n_heads=8, d_ff=d_model*4,
                                    n_positions=N_LAYERS_3B, d_input=D_INPUT_3B)

    src = src_tt.state_dict()
    tgt = tgt_tt.state_dict()
    for key in src:
        if key in tgt and src[key].shape == tgt[key].shape:
            tgt[key].copy_(src[key])
        elif key in tgt and 'pos_embed' in key:
            min_pos = min(src[key].shape[0], tgt[key].shape[0])
            tgt[key][:min_pos].copy_(src[key][:min_pos])

    with torch.no_grad():
        min_dim_in = min(D_INPUT_7B, D_INPUT_3B)
        tgt_tt.input_proj.weight[:, :min_dim_in] = src_tt.input_proj.weight[:, :min_dim_in]
        tgt_tt.input_proj.bias.copy_(src_tt.input_proj.bias)
        min_dim_out = min(D_INPUT_7B, D_INPUT_3B)
        tgt_tt.output_proj.weight[:min_dim_out, :] = src_tt.output_proj.weight[:min_dim_out, :]
        tgt_tt.output_proj.bias[:min_dim_out] = src_tt.output_proj.bias[:min_dim_out]

    return tgt_tt


def main():
    print("Loading Qwen2.5-3B (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    head_dim = 128
    n_kv_heads = 2

    print("Transferring TT from 7B→3B...", flush=True)
    tt = transfer_tt(TT_PATH)
    tt.to(DEVICE)
    tt.eval()
    for p in tt.parameters(): p.requires_grad = False
    n_params = sum(p.numel() for p in tt.parameters())
    print(f"  Transferred TT: {n_params:,} params", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:15]

    # Test layers: L8 (known 7B trim-tab), L10 (proportional 8→10 mapping),
    # L2 (known 7B trim-tab), and L18 (midpoint)
    test_layers = [2, 8, 10, 18]

    # Baseline
    print(f"\nBaseline (15 problems)...", flush=True)
    correct, total = 0, 0
    for i, prob in enumerate(problems):
        msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
        iids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                        return_tensors="pt").input_ids.to(DEVICE)
        am = iids.ne(tok.pad_token_id).long()
        out = model.generate(iids, attention_mask=am, max_new_tokens=MAX_GEN,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, iids.shape[1]:], skip_special_tokens=True)
        pa = extract_number(gen)
        ca = re.search(r"####\s*(-?\d+)", prob["answer"])
        if pa and ca and pa == ca.group(1): correct += 1
        total += 1
    baseline_acc = correct / total
    print(f"  Baseline: {correct}/{total} ({100*baseline_acc:.1f}%)", flush=True)

    for li in test_layers:
        print(f"\nLayer {li}...", flush=True)
        correct, total = 0, 0
        t0 = time.time()
        for i, prob in enumerate(problems):
            msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
            input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                return_tensors="pt").input_ids.to(DEVICE)
            past, generated_tokens, first_step = None, [], True
            for step in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)
                nt = fwd.logits[0, -1, :].argmax().item()
                if nt == tok.eos_token_id: break
                generated_tokens.append(nt)
                if not first_step:
                    hs = fwd.hidden_states
                    hp = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS_3B]], dim=0)
                    x = hp.unsqueeze(0).to(DEVICE)
                    with torch.no_grad(): v = tt(x)
                    h_act = hs[li + 1][0, -1, :]
                    h_steered = h_act + ALPHA * v[0, li, :]
                    layer = model.model.layers[li]
                    k = layer.self_attn.k_proj(h_steered.to(torch.bfloat16)).view(1, n_kv_heads, 1, head_dim)
                    vo = layer.self_attn.v_proj(h_steered.to(torch.bfloat16)).view(1, n_kv_heads, 1, head_dim)
                    pkv_l = fwd.past_key_values.layers[li]
                    pkv_l.keys[0, :, -1:, :] = k.to(pkv_l.keys.dtype)
                    pkv_l.values[0, :, -1:, :] = vo.to(pkv_l.values.dtype)
                past = fwd.past_key_values
                input_ids = torch.tensor([[nt]], device=DEVICE)
                first_step = False
            gen = tok.decode(generated_tokens, skip_special_tokens=True)
            pa = extract_number(gen)
            ca = re.search(r"####\s*(-?\d+)", prob["answer"])
            if pa and ca and pa == ca.group(1): correct += 1
            total += 1
            if (i+1) % 5 == 0:
                print(f"  [{i+1}/15] acc={correct}/{total} ({100*correct/total:.0f}%)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        acc = correct / total
        d = acc - baseline_acc
        print(f"  L{li}: {correct}/{total} ({100*acc:.1f}%) Δ={100*d:+.1f}pp [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'='*60}")
    print(f"3B TRIM-TAB PROBE (via 7B→3B TT transfer)")
    print(f"{'='*60}")
    print(f"  Baseline: {100*baseline_acc:.1f}%")
    for li in test_layers:
        acc = correct/total  # inaccurate, just placeholder
    print(f"  (See above for per-layer results)")

if __name__ == "__main__":
    main()
