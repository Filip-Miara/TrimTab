#!/usr/bin/env python3
"""KV-cache steering: modify last layer's KV entries using TT velocity.

After generating token t, modify its key/value cache entries using the
TT's predicted velocity. Future tokens attend to the steered representation
of token t.

Architecture:
  1. Forward → logits + hidden states + KV cache → token t
  2. TT predicts v[t] = h[t+1]_pred - h[t] (from trajectory)
  3. For the last layer (layer 23, full-attention):
     h'[t] = h[t] + α * v[t]
     k' = k_proj(h'[t]), v' = v_proj(h'[t])
  4. Replace KV cache entries for position t with (k', v')
  5. Generate token t+1 — attends to modified cache
"""
from __future__ import annotations

import gc, re, sys, time
import numpy as np
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
TT_PATH = "best_reasoning_transformer.pt"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
MAX_GEN = 100
N_LAYERS = 24
# In Qwen3.5-2B, each 4th layer is full-attention (with k_proj, v_proj)
# Layers 3, 7, 11, 15, 19, 23 are full-attention
# Layer 23 is the final full-attention layer


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--mode", choices=["baseline", "kvcache"], default="kvcache")
    parser.add_argument("--alpha", type=float, default=0.5, help="KV-cache steering strength")
    parser.add_argument("--layer", type=int, default=23, help="Layer to steer (23=final full-attn)")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()

    tt = TrajectoryTransformer(d_model=512, n_layers=6, n_heads=8, d_ff=2048, n_positions=50)
    tt.load_state_dict(torch.load(TT_PATH, map_location=DEVICE))
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    # Get k_proj/v_proj for the target layer
    target_layer = model.model.layers[args.layer]
    if hasattr(target_layer, 'self_attn') and hasattr(target_layer.self_attn, 'k_proj'):
        k_proj = target_layer.self_attn.k_proj
        v_proj = target_layer.self_attn.v_proj
        print(f"Layer {args.layer}: full-attention with k_proj/v_proj")
    else:
        print(f"Layer {args.layer}: NOT a full-attention layer! Using anyway...")
        k_proj = v_proj = None

    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    examples = ("Q: Janet has 5. She buys 3. How many?\nA: Step 1: 5. Step 2: She buys 3. Step 3: 5+3=8.\nSo answer is 8.\n\n"
                "Q: Bakery sells 12/hr. How many in 8h?\nA: Step 1: 12/hr. Step 2: 8 hours. Step 3: 12x8=96.\nSo answer is 96.\n\n")

    correct, total, steer_count = 0, 0, 0
    t0 = time.time()

    for idx, prob in enumerate(problems):
        question = prob["question"]
        correct_ans = parse_answer(prob["answer"])
        prompt = f"{examples}Q: {question}\nA:"

        if args.mode == "baseline":
            inp = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=MAX_GEN, do_sample=False, pad_token_id=tok.eos_token_id)
            gen_text = tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            full_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE).input_ids
            plen = full_ids.shape[1]
            past = None
            hidden_buffer = []  # store final-layer hidden states for TT trajectory

            for step in range(MAX_GEN):
                with torch.no_grad():
                    out = model(full_ids, past_key_values=past, use_cache=True, output_hidden_states=True)

                hs = out.hidden_states
                h_final = hs[-1][0, -1, :]  # last token's final layer hidden state
                hidden_buffer.append(h_final.cpu().float())
                past = out.past_key_values
                orig_logits = out.logits[0, -1, :]
                next_token = orig_logits.argmax().item()

                actual_step = full_ids.shape[1] - plen + 1

                if actual_step >= 2 and len(hidden_buffer) >= 2 and k_proj is not None:
                    # Compute TT velocity
                    traj = torch.stack(hidden_buffer[-min(actual_step, 48):]).unsqueeze(0).float()
                    with torch.no_grad():
                        v_pred = tt(traj.to(DEVICE), causal=True)[0, -1]

                    # Modify KV cache at the target layer
                    # h_steered = h_final + α * v
                    h_steered = h_final.float() + args.alpha * v_pred.cpu()

                    # Project through k_proj/v_proj
                    k_delta = k_proj(h_steered.to(k_proj.weight.dtype).unsqueeze(0))[0]  # (n_kv_heads * head_dim,)
                    v_delta = v_proj(h_steered.to(v_proj.weight.dtype).unsqueeze(0))[0]

                    # Reshape to match cache format: (n_heads, head_dim)
                    n_heads = target_layer.self_attn.num_key_value_heads
                    head_dim = target_layer.self_attn.head_dim
                    k_delta = k_delta.view(n_heads, head_dim)
                    v_delta = v_delta.view(n_heads, head_dim)

                    # Get current KV cache entries for the last position
                    k_cache = past.key_cache[args.layer]
                    v_cache = past.value_cache[args.layer]

                    # Compute: what would the ORIGINAL key/value be? Use h_final.
                    k_orig = k_proj(h_final.to(k_proj.weight.dtype).unsqueeze(0))[0].view(n_heads, head_dim)
                    v_orig = v_proj(h_final.to(v_proj.weight.dtype).unsqueeze(0))[0].view(n_heads, head_dim)

                    # Replace KV cache: use steered hidden state's keys/values
                    k_new = k_steered = k_proj(h_steered.to(k_proj.weight.dtype).unsqueeze(0))[0].view(n_heads, head_dim)
                    v_new = v_steered = v_proj(h_steered.to(v_proj.weight.dtype).unsqueeze(0))[0].view(n_heads, head_dim)

                    # Modify cache in-place
                    # k_cache shape: (1, n_kv_heads, seq_len, head_dim) or (n_kv_heads, seq_len, head_dim)?
                    # DynamicCache stores (batch, heads, seq, head_dim)
                    k_cache[0, :, -1, :] = k_new.to(k_cache.dtype)
                    v_cache[0, :, -1, :] = v_new.to(v_cache.dtype)
                    steer_count += 1

                if next_token == tok.eos_token_id:
                    break
                full_ids = torch.cat([full_ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)

            gen_text = tok.decode(full_ids[0, plen:], skip_special_tokens=True)

        predicted = parse_answer(gen_text)
        if predicted is not None and correct_ans is not None and predicted == correct_ans:
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{args.n_test}] acc={correct}/{total} ({100*correct/total:.0f}%) "
                  f"steer={steer_count} {time.time()-t0:.0f}s")
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}" + (f" (α={args.alpha}, layer={args.layer})" if args.mode == "kvcache" else ""))
    print(f"  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
