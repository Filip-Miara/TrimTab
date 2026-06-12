#!/usr/bin/env python3
"""Phase 1: Single-layer KV-cache perturbation test.

Tests whether modifying the KV cache using TT velocity predictions
changes token selection more than random noise.
"""
from __future__ import annotations

import gc, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
TARGET_LAYER = 23
N_KV_HEADS = 2
HEAD_DIM = 256
ALPHAS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=DEVICE)
model.eval()

tt = TrajectoryTransformer(d_model=512, n_layers=6, n_heads=8, d_ff=2048, n_positions=50)
tt.load_state_dict(torch.load("best_reasoning_transformer.pt", map_location="cpu"))
tt.to(DEVICE); tt.eval()

layer = model.model.layers[TARGET_LAYER]
k_proj, v_proj = layer.self_attn.k_proj, layer.self_attn.v_proj
def get_k(past): return past.layers[TARGET_LAYER].keys
def get_v(past): return past.layers[TARGET_LAYER].values

ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
prompts = [f"Q: {r['question']}\nA:" for r in ds if len(r["question"]) > 50][:5]

results = {a: {"diff": 0, "tot": 0} for a in ALPHAS}
noise_r = {"diff": 0, "tot": 0}
t0 = time.time()

for pidx, prompt in enumerate(prompts):
    full_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE).input_ids
    hbuf = []

    for step in range(30):
        if step == 0:
            with torch.no_grad(): out = model(full_ids, use_cache=True, output_hidden_states=True)
        else:
            with torch.no_grad(): out = model(full_ids, past_key_values=out.past_key_values, use_cache=True, output_hidden_states=True)

        hbuf.append(out.hidden_states[-1][0, -1, :].cpu().float())
        bas_tok = out.logits[0, -1, :].argmax().item()

        if len(hbuf) >= 2:
            n = min(len(hbuf), 48)
            traj = torch.stack(hbuf[-n:]).unsqueeze(0).float()
            with torch.no_grad():
                vp = tt(traj.to(DEVICE), causal=True)[0, -1]
            hc = hbuf[-1].to(DEVICE)
            vn = vp.norm().item()

            # Test each alpha
            for a in ALPHAS:
                hs = hc + a * vp
                ks = k_proj(hs.to(k_proj.weight.dtype).unsqueeze(0))[0].view(N_KV_HEADS, HEAD_DIM)
                vs = v_proj(hs.to(v_proj.weight.dtype).unsqueeze(0))[0].view(N_KV_HEADS, HEAD_DIM)

                k_t = get_k(out.past_key_values)
                v_t = get_v(out.past_key_values)
                orig_k, orig_v = k_t[0, :, -1, :].clone(), v_t[0, :, -1, :].clone()
                k_t[0, :, -1, :] = ks.to(k_t.dtype)
                v_t[0, :, -1, :] = vs.to(v_t.dtype)

                ni = torch.tensor([[bas_tok]], device=DEVICE)
                with torch.no_grad():
                    o2 = model(ni, past_key_values=out.past_key_values, use_cache=True)
                st = o2.logits[0, 0, :].argmax().item()

                if st != bas_tok: results[a]["diff"] += 1
                results[a]["tot"] += 1

                k_t[0, :, -1, :] = orig_k.to(k_t.dtype)
                v_t[0, :, -1, :] = orig_v.to(v_t.dtype)

            # Noise control
            nk = torch.randn(N_KV_HEADS, HEAD_DIM, device=DEVICE) * (vn * 0.3)
            nv = torch.randn(N_KV_HEADS, HEAD_DIM, device=DEVICE) * (vn * 0.3)
            k_noise = get_k(out.past_key_values)
            v_noise = get_v(out.past_key_values)
            k_noise[0, :, -1, :] += nk.to(k_noise.dtype)
            v_noise[0, :, -1, :] += nv.to(v_noise.dtype)
            ni = torch.tensor([[bas_tok]], device=DEVICE)
            with torch.no_grad():
                o2 = model(ni, past_key_values=out.past_key_values, use_cache=True)
            nt = o2.logits[0, 0, :].argmax().item()
            if nt != bas_tok: noise_r["diff"] += 1
            noise_r["tot"] += 1
            k_noise[0, :, -1, :] -= nk.to(k_noise.dtype)
            v_noise[0, :, -1, :] -= nv.to(v_noise.dtype)

        if bas_tok == tok.eos_token_id:
            break
        full_ids = torch.cat([full_ids, torch.tensor([[bas_tok]], device=DEVICE)], dim=1)

    print(f"  Problem {pidx+1}: {time.time()-t0:.0f}s")
    gc.collect()
    torch.cuda.empty_cache()

nr = 100 * noise_r["diff"] / max(noise_r["tot"], 1)
print(f"\nPhase 1: KV-cache steering at layer {TARGET_LAYER}")
print(f"  Tokens tested: {noise_r['tot']}")
print(f"  Noise control: {nr:.1f}% changed")
print(f"  {'Alpha':>6} {'Changed':>8} {'Rate':>8} {'Ratio vs noise':>16}")
for a in ALPHAS:
    r = results[a]
    rat = 100 * r["diff"] / max(r["tot"], 1)
    print(f"  {a:6.1f} {r['diff']:8d} {rat:7.1f}% {rat/max(nr,0.1):14.1f}x")
    if rat / max(nr, 0.1) > 2.0 and a >= 0.5:
        print(f"  ✅ Signal detected at α={a}!")
