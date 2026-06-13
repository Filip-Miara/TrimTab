#!/usr/bin/env python3 -u
"""DoRA-inspired KV-cache steering: separate direction from magnitude.

Standard steering: h' = h + α · v_pred (TT predicts both direction + magnitude)
DoRA steering:     h' = h + α · conf · (v_pred / ||v_pred||)
                       where direction = TT prediction (what we trust)
                       where magnitude = confidence signal (separate source)

Confidence sources tested:
  [0] Uniform — standard steering (baseline)
  [1] Inv-mag — 1/(1+||v_pred||), lower confidence when TT predicts large steps
  [2] Inv-var — 1/(1+var(||v[l]||)), lower confidence when layer-to-layer varies
"""
from __future__ import annotations

import gc, json, os, re, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MAX_GEN = 200
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
D_MODEL = 3584
N_LAYERS = 28
N_KV_HEADS = 4
HEAD_DIM = 128


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def compute_confidence(v_pred, mode):
    """Compute per-token confidence from velocity predictions.
    Args:
        v_pred: (1, 28, 3584) velocity predictions
        mode: 'uniform', 'inv_mag', or 'inv_var'
    Returns:
        conf: scalar [0, 1]
    """
    v = v_pred[0]  # (28, 3584)
    v_norms = v.norm(dim=-1)  # (28,) — magnitude per layer
    if mode == "uniform":
        return 1.0
    elif mode == "inv_mag":
        return 1.0 / (1.0 + v_norms.mean().item())
    elif mode == "inv_var":
        var = v_norms.var().item()
        return 1.0 / (1.0 + var)
    return 1.0


def steer_kv_cache_dora(model, hidden_states, velocity, past_key_values, alpha, conf_mode):
    """DoRA-inspired: separate direction from magnitude in steering."""
    for li in range(N_LAYERS):
        h_actual = hidden_states[li + 1][0, -1, :]  # (3584,)
        v = velocity[0, li, :]  # (3584,)
        v_norm = v.norm().item()
        v_unit = v / (v_norm + 1e-8)  # direction only

        # Confidence from the SEPARATE magnitude source
        # v_norm is the TT-predicted step size — we modulate it
        if conf_mode == "uniform":
            effective_alpha = alpha
        elif conf_mode == "inv_mag":
            effective_alpha = alpha / (1.0 + v_norm)
        elif conf_mode == "inv_var":
            # Pre-computed per-token confidence
            layer_var = velocity[0, :, :].norm(dim=-1).var().item()
            effective_alpha = alpha / (1.0 + layer_var)
        else:
            effective_alpha = alpha

        h_steered = h_actual + effective_alpha * v_unit.to(h_actual.dtype)

        layer = model.model.layers[li]
        k = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))
        v_out = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
        k = k.view(1, N_KV_HEADS, 1, HEAD_DIM)
        v_out = v_out.view(1, N_KV_HEADS, 1, HEAD_DIM)
        lc = past_key_values.layers[li]
        lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
        lc.values[0, :, -1:, :] = v_out.to(lc.values.dtype)


def evaluate(problems, tok, model, tt, alpha, conf_mode="uniform"):
    n_correct, n_total = 0, 0
    t0 = time.time()

    for idx, prob in enumerate(problems):
        correct_ans = extract_number(prob["answer"])
        msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
        input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_tensors="pt").input_ids.to(DEVICE)

        if alpha == 0.0:
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
                    h_pos = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                    x = h_pos.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        v = tt(x)
                    steer_kv_cache_dora(model, hs, v, fwd.past_key_values, alpha, conf_mode)

                past = fwd.past_key_values
                input_ids = torch.tensor([[next_tok]], device=DEVICE)
                first_step = False
            gen = tok.decode(generated_tokens, skip_special_tokens=True)

        predicted = extract_number(gen)
        if predicted is not None and correct_ans is not None and predicted == correct_ans:
            n_correct += 1
        n_total += 1

        if (idx + 1) % 10 == 0:
            print(f"  α={alpha} {conf_mode} [{idx+1}/{len(problems)}] acc={n_correct}/{n_total} "
                  f"({100*n_correct/n_total:.0f}%) {time.time()-t0:.0f}s", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    return {"correct": n_correct, "total": n_total, "accuracy": n_correct / max(n_total, 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.3])
    parser.add_argument("--tt-path", type=str, default="best_gen_tt_7b.pt")
    parser.add_argument("--conf-modes", nargs="+", default=["uniform", "inv_mag", "inv_var"])
    args = parser.parse_args()

    print("Loading Qwen2.5-7B-Instruct (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    print(f"Loading TT from {args.tt_path}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=D_MODEL)
    tt.load_state_dict(torch.load(args.tt_path, map_location="cpu"), strict=False)
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    print(f"\n{len(problems)} problems, {len(args.alphas)} alphas, {len(args.conf_modes)} conf modes", flush=True)

    results = {}
    for conf_mode in args.conf_modes:
        for alpha in args.alphas:
            label = f"{conf_mode}_a{alpha}"
            print(f"\n{'='*60}", flush=True)
            print(f"{conf_mode} | α={alpha}", flush=True)
            print(f"{'='*60}", flush=True)
            r = evaluate(problems, tok, model, tt, alpha, conf_mode)
            results[label] = r

    print(f"\n{'='*60}", flush=True)
    print(f"DORA STEERING SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    baseline_acc = results.get("uniform_a0.0", {}).get("accuracy", 0)
    print(f"  {'Mode':20s} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}")
    for label, r in sorted(results.items()):
        acc = r["accuracy"]
        delta = acc - baseline_acc
        print(f"  {label:20s} {100*acc:7.1f}% {100*delta:+7.1f}pp", flush=True)

    with open("qwen25_7b_dora_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved.", flush=True)


if __name__ == "__main__":
    main()
