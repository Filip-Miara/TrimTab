#!/usr/bin/env python3 -u
"""Test GDN steering via output-side hook on Qwen3.5-0.8B.

Registers a hook on layer li that adds α·v[l] to the output
DURING the forward pass. This modifies the residual stream
before it reaches the next layer.
"""
from __future__ import annotations

import gc, json, re, sys, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
TT_PATH = "best_tt_08b.pt"
MAX_GEN = 256
BASELINE = 0.30  # known 0.8B GSM8K baseline


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=15)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 4, 5, 6])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.02, 0.05, 0.1])
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
    fa_layers = set(i for i, t in enumerate(cfg.layer_types) if t == "full_attention")
    print(f"  {n_layers} layers, FA={sorted(fa_layers)}", flush=True)

    print(f"Loading TT from {TT_PATH}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=n_layers, d_input=cfg.hidden_size)
    tt.load_state_dict(torch.load(TT_PATH, map_location="cpu"), strict=False)
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    results = {}

    for li in args.layers:
        lt = "FA" if li in fa_layers else "GDN"
        print(f"\n{'='*60}", flush=True)
        print(f"Layer {li} ({lt})", flush=True)

        for alpha in args.alphas:
            correct, total = 0, 0
            _v_buffer = [None]

            # Hook: adds α·v[li] to this layer's output during forward
            def make_hook(buf, a):
                def hook(module, inp, out):
                    if buf[0] is not None:
                        return out + a * buf[0].to(out.dtype)
                    return out
                return hook

            hook_handle = model.model.layers[li].register_forward_hook(
                make_hook(_v_buffer, alpha))

            t0 = time.time()
            for idx, prob in enumerate(problems):
                msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
                input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    return_tensors="pt").input_ids.to(DEVICE)

                past, generated_tokens, first_step = None, [], True
                for step in range(MAX_GEN):
                    with torch.no_grad():
                        fwd = model(input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)
                    next_tok = fwd.logits[0, -1, :].argmax().item()
                    if next_tok == tok.eos_token_id: break
                    generated_tokens.append(next_tok)

                    if not first_step:
                        hs = fwd.hidden_states
                        hp = torch.stack([h[0, -1, :].float() for h in hs[:n_layers]], dim=0)
                        x = hp.unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            v = tt(x)
                        _v_buffer[0] = v[0, li, :].detach().clone()

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
                    print(f"  L{li} α={alpha} [{idx+1}/{args.n_test}] acc={correct}/{total} "
                          f"({100*correct/total:.0f}%)", flush=True)
                    gc.collect(); torch.cuda.empty_cache()

            hook_handle.remove()
            key = f"L{li}_a{alpha}"
            results[key] = {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}
            print(f"  L{li} α={alpha}: {correct}/{total} ({100*correct/max(total,1):.1f}%) "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"GDN OUTPUT-STEER (OUTPUT HOOK) — Qwen3.5-0.8B", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Layer':>6} {'Type':>4} {'α':>6} {'Acc':>8s} {'vs base':>8s}")
    print(f"  {'base':>6} {'':4} {'':6} {100*BASELINE:7.1f}%")
    for li in args.layers:
        for alpha in args.alphas:
            key = f"L{li}_a{alpha}"
            r = results[key]
            acc = r["accuracy"]
            d = acc - BASELINE
            lt = "FA" if li in fa_layers else "GDN"
            marker = " ✓" if d > 0.03 else (" ✗ DEATH" if d < -0.05 else "")
            print(f"  L{li:3d}  {lt:>4}  {alpha:5.2f}  {100*acc:7.1f}%  {100*d:+7.1f}pp{marker}", flush=True)

    with open("gdn_output_steer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to gdn_output_steer_results.json", flush=True)


if __name__ == "__main__":
    main()
