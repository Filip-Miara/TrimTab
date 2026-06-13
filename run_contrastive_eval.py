#!/usr/bin/env python3 -u
"""Contrastive steering evaluation on Qwen2.5-Math-1.5B.

Steers using v_contrastive = TT_correct(h) − TT_incorrect(h).
Tests baseline vs contrastive at known trim-tab layers.
"""
from __future__ import annotations

import gc, json, re, sys, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
N_LAYERS = 28
N_KV_HEADS = 2
HEAD_DIM = 64  # 1536/12 * 12/2 ? No: hidden=1536, q_heads=12, kv_heads=2, head_dim=1536/12=128
# Actually: k_proj output = kv_heads * head_dim. k_proj shape is (256, 1536).
# 256 = 2 * 128 = kv_heads * head_dim. So head_dim = 128.
HEAD_DIM = 128
MAX_GEN = 200


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"####\s*(-?\d+)", r"So answer is\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def steer_layer(model, hidden_states, velocity, pkv, alpha, li):
    h = hidden_states[li + 1][0, -1, :] + alpha * velocity[0, li, :]
    l = model.model.layers[li]
    k = l.self_attn.k_proj(h.to(torch.bfloat16)).view(1, N_KV_HEADS, 1, HEAD_DIM)
    v_out = l.self_attn.v_proj(h.to(torch.bfloat16)).view(1, N_KV_HEADS, 1, HEAD_DIM)
    lc = pkv.layers[li]
    lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
    lc.values[0, :, -1:, :] = v_out.to(lc.values.dtype)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--layers", type=int, nargs="+", default=[7, 8, 9],
                        help="Layers to test (individual + combined)")
    args = parser.parse_args()

    print("Loading Math-1.5B (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print("Loading TTs...", flush=True)
    tt_correct = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                        n_positions=N_LAYERS, d_input=1536).to(DEVICE)
    tt_correct.load_state_dict(torch.load("best_tt_correct.pt", map_location="cpu"), strict=False)
    tt_correct.eval()
    for p in tt_correct.parameters(): p.requires_grad = False

    tt_incorrect = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                          n_positions=N_LAYERS, d_input=1536).to(DEVICE)
    tt_incorrect.load_state_dict(torch.load("best_tt_incorrect.pt", map_location="cpu"), strict=False)
    tt_incorrect.eval()
    for p in tt_incorrect.parameters(): p.requires_grad = False

    examples = ('Q: Janet has 5 ducks. She buys 3 more. How many?\nA: Step 1: 5. Step 2: buy 3. Step 3: 5+3=8. So answer is 8.\n\n'
                'Q: 12 muffins/hr, 8 hours?\nA: Step 1: 12/hr. Step 2: 8h. Step 3: 12x8=96. So answer is 96.\n\n')

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    def evaluate(steer_layers=None):
        """If steer_layers is None = baseline. If list = steer those layers."""
        correct, total = 0, 0
        t0 = time.time()
        for idx, prob in enumerate(problems):
            prompt = f"{examples}Q: {prob['question']}\nA:"
            input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids

            if steer_layers is None:
                out = model.generate(input_ids, max_new_tokens=MAX_GEN, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
            else:
                past, gens, first = None, [], True
                for step in range(MAX_GEN):
                    with torch.no_grad():
                        fwd = model(input_ids, past_key_values=past, use_cache=True,
                                     output_hidden_states=True)
                    nt = fwd.logits[0, -1, :].argmax().item()
                    if nt == tok.eos_token_id: break
                    gens.append(nt)
                    if not first:
                        hs = fwd.hidden_states
                        hp = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                        x = hp.unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            v_c = tt_correct(x)
                            v_i = tt_incorrect(x)
                        v_contrastive = v_c - v_i
                        for li in steer_layers:
                            steer_layer(model, hs, v_contrastive, fwd.past_key_values, args.alpha, li)
                    past = fwd.past_key_values
                    input_ids = torch.tensor([[nt]], device=DEVICE)
                    first = False
                gen = tok.decode(gens, skip_special_tokens=True)

            ca = re.search(r"####\s*(-?\d+)", prob["answer"])
            pa = extract_number(gen)
            if pa and ca and pa == ca.group(1): correct += 1
            total += 1
            if (idx + 1) % 20 == 0:
                lbl = "base" if steer_layers is None else f"L{steer_layers}"
                print(f"  {lbl} [{idx+1}/{len(problems)}] acc={correct}/{total} "
                      f"({100*correct/total:.0f}%) {time.time()-t0:.0f}s", flush=True)
                gc.collect(); torch.cuda.empty_cache()
        return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}

    results = {}
    # Baseline
    print(f"\n{'='*60}", flush=True); print("Baseline", flush=True); print(f"{'='*60}", flush=True)
    r = evaluate(None)
    results["baseline"] = r
    print(f"  Baseline: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    # Single layers + combined
    configs = {f"L{l}": [l] for l in args.layers}
    configs[f"L{'+'.join(str(l) for l in args.layers)}"] = list(args.layers)
    for label, layers in configs.items():
        print(f"\n{'='*60}", flush=True); print(f"Contrastive {label}", flush=True); print(f"{'='*60}", flush=True)
        r = evaluate(layers)
        results[label] = r
        print(f"  {label}: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"CONTRASTIVE STEERING SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    ba = results["baseline"]["accuracy"]
    for label, r in sorted(results.items()):
        d = r["accuracy"] - ba if label != "baseline" else 0
        print(f"  {label:>12}: {100*r['accuracy']:5.1f}% ({100*d:+5.1f}pp)", flush=True)

    with open("contrastive_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved.", flush=True)


if __name__ == "__main__":
    main()
