#!/usr/bin/env python3 -u
"""Test whether 7B steering transfers beyond GSM8K to SVAMP.

Baseline vs L8 (best trim tab, +20pp on GSM8K) vs L9 (death layer, -23pp).
If the layer-specific pattern replicates, steering is learning general
reasoning structure, not GSM8K-specific memorization.
"""
from __future__ import annotations

import gc, json, re, sys, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TT_PATH = "best_gen_tt_7b.pt"
N_LAYERS = 28
N_KV_HEADS = 4
HEAD_DIM = 128


def extract_number(text):
    nums = re.findall(r"-?\d+", text)
    if nums:
        return nums[-1]
    return None


def steer_layer(model, hs, v, pkv, alpha, li):
    h = hs[li + 1][0, -1, :] + alpha * v[0, li, :]
    l = model.model.layers[li]
    k = l.self_attn.k_proj(h.to(torch.bfloat16)).view(1, N_KV_HEADS, 1, HEAD_DIM)
    v_out = l.self_attn.v_proj(h.to(torch.bfloat16)).view(1, N_KV_HEADS, 1, HEAD_DIM)
    lc = pkv.layers[li]
    lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
    lc.values[0, :, -1:, :] = v_out.to(lc.values.dtype)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    print("Loading Qwen2.5-7B-Instruct (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print(f"Loading TT from {TT_PATH}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=3584)
    tt.load_state_dict(torch.load(TT_PATH, map_location="cpu"), strict=False)
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("ChilleD/SVAMP", split="test")
    problems = [r for r in ds][:args.n_test]
    print(f"SVAMP: {len(problems)} problems, answer format: number", flush=True)

    def evaluate(steer_layer_idx=None):
        correct, total = 0, 0
        t0 = time.time()
        for idx, prob in enumerate(problems):
            prompt = f"Q: {prob['question_concat']}\nA:"
            input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids

            if steer_layer_idx is None:
                am = input_ids.ne(tok.pad_token_id).long() if tok.pad_token_id is not None else None
                out = model.generate(input_ids, attention_mask=am, max_new_tokens=100,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
            else:
                past, gens, first = None, [], True
                for step in range(100):
                    with torch.no_grad():
                        fwd = model(input_ids, past_key_values=past, use_cache=True,
                                     output_hidden_states=True)
                    nt = fwd.logits[0, -1, :].argmax().item()
                    if nt == tok.eos_token_id: break
                    gens.append(nt)
                    if not first:
                        hs = fwd.hidden_states
                        hp = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                        v = tt(hp.unsqueeze(0).to(DEVICE))
                        steer_layer(model, hs, v, fwd.past_key_values, args.alpha, steer_layer_idx)
                    past = fwd.past_key_values
                    input_ids = torch.tensor([[nt]], device=DEVICE)
                    first = False
                gen = tok.decode(gens, skip_special_tokens=True)

            pa = extract_number(gen)
            ca = str(prob["Answer"])
            if pa == ca:
                correct += 1
            total += 1
            if (idx + 1) % 50 == 0:
                lbl = f"L{steer_layer_idx}" if steer_layer_idx is not None else "base"
                print(f"  {lbl} [{idx+1}/{len(problems)}] acc={correct}/{total} "
                      f"({100*correct/total:.0f}%) {time.time()-t0:.0f}s", flush=True)
                gc.collect(); torch.cuda.empty_cache()
        return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}

    results = {}
    for label, layer in [("baseline", None), ("L8", 8), ("L9", 9)]:
        print(f"\n{'='*60}", flush=True)
        print(f"  {label}", flush=True)
        print(f"{'='*60}", flush=True)
        r = evaluate(layer)
        results[label] = r
        print(f"  {label}: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"SVAMP GENERALIZATION TEST", flush=True)
    print(f"{'='*60}", flush=True)
    ba = results["baseline"]["accuracy"]
    for label, r in results.items():
        d = r["accuracy"] - ba if label != "baseline" else 0
        print(f"  {label:>10}: {100*r['accuracy']:5.1f}% ({100*d:+5.1f}pp)", flush=True)
    print(f"\nGSM8K reference: L8=+20pp, L9=-23pp vs baseline", flush=True)

    with open("svamp_generalization.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved.", flush=True)


if __name__ == "__main__":
    main()
