#!/usr/bin/env python3 -u
"""Per-layer steering sweep on AWQ Qwen2.5-7B with our trained TT.

Measures GSM8K accuracy with steering at each layer to find trim-tab/death layers.
"""
import gc, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.optimization import setup_optimizations
setup_optimizations()

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MAX_GEN = 200
MODEL_PATH = "D:\\Qwen2.5-7B-AWQ\\qwen7b_awq"
TT_PATH = "best_tt_awq_7b.pt"
D_MODEL = 3584
N_LAYERS = 28
N_KV_HEADS = 4
HEAD_DIM = 128
N_TEST = 30
ALPHA = 0.1


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def steer_single_layer(model, hidden_states, velocity, past_key_values, alpha, target_layer):
    """Steer one specific layer's KV cache using TT velocity."""
    h_actual = hidden_states[target_layer + 1][0, -1, :]
    v = velocity[0, target_layer, :]
    h_steered = h_actual + alpha * v

    layer = model.model.layers[target_layer]
    k = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))
    v_out = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
    k = k.view(1, N_KV_HEADS, 1, HEAD_DIM)
    v_out = v_out.view(1, N_KV_HEADS, 1, HEAD_DIM)
    lc = past_key_values.layers[target_layer]
    lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
    lc.values[0, :, -1:, :] = v_out.to(lc.values.dtype)


def evaluate_layer(problems, tok, model, tt, alpha, target_layer):
    correct, total = 0, 0
    t0 = time.time()

    for idx, prob in enumerate(problems):
        msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
        input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_tensors="pt").input_ids.to(DEVICE)

        if alpha == 0.0 or target_layer is None:
            # Baseline: no steering (must pass attention_mask explicitly)
            am = input_ids.ne(tok.pad_token_id).long() if tok.pad_token_id is not None else None
            out = model.generate(input_ids, attention_mask=am, max_new_tokens=MAX_GEN,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        else:
            # Steered generation
            past, generated_tokens, first_step = None, [], True
            for step in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(input_ids, past_key_values=past, use_cache=True,
                                output_hidden_states=True)
                next_tok = fwd.logits[0, -1, :].argmax().item()
                if next_tok == tok.eos_token_id:
                    break
                generated_tokens.append(next_tok)

                if not first_step:
                    hs = fwd.hidden_states
                    if hs is None:
                        continue  # skip if hidden states not available
                    h_pos = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                    x = h_pos.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        v = tt(x)
                    steer_single_layer(model, hs, v, fwd.past_key_values, alpha, target_layer)

                past = fwd.past_key_values
                input_ids = torch.tensor([[next_tok]], device=DEVICE)
                first_step = False
            gen = tok.decode(generated_tokens, skip_special_tokens=True)

        # Check answer
        ca = re.search(r"####\s*(-?\d+)", prob["answer"])
        predicted = extract_number(gen)
        if predicted is not None and ca and predicted == ca.group(1):
            correct += 1
        total += 1

        if (idx + 1) % 5 == 0:
            label = f"L{target_layer:2d}" if target_layer is not None else "base"
            print(f"  {label} [{idx+1}/{len(problems)}] acc={correct}/{total} "
                  f"({100*correct/total:.0f}%)", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--tt-path", type=str, default=TT_PATH)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to test (default: all 28)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run baseline, skip steering")
    args = parser.parse_args()

    layers_to_test = args.layers if args.layers is not None else list(range(N_LAYERS))

    # Load model
    print(f"Loading AWQ model from {args.model_path}...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    print(f"Model loaded in {time.time()-t0:.1f}s, "
          f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

    # Load TT
    print(f"Loading TT from {args.tt_path}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=D_MODEL)
    tt.load_state_dict(torch.load(args.tt_path, map_location="cpu"), strict=False)
    tt.to(DEVICE)
    tt.eval()
    for p in tt.parameters():
        p.requires_grad = False
    print(f"TT loaded ({sum(p.numel() for p in tt.parameters()):,} params)", flush=True)

    # Load GSM8K
    print("Loading GSM8K test set...", flush=True)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    print(f"Testing on {len(problems)} problems", flush=True)

    results = {}

    # Baseline
    print(f"\n{'='*60}", flush=True)
    print("Baseline (no steering)", flush=True)
    print(f"{'='*60}", flush=True)
    r = evaluate_layer(problems, tok, model, tt, 0.0, None)
    results["baseline"] = r
    print(f"  Baseline: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    if args.baseline_only:
        print("\nBaseline-only mode. Skipping steering layers.", flush=True)
    else:
        # Sweep specified layers
        for li in layers_to_test:
            print(f"\n{'='*60}", flush=True)
            print(f"Layer {li} only, alpha={args.alpha}", flush=True)
            print(f"{'='*60}", flush=True)
            r = evaluate_layer(problems, tok, model, tt, args.alpha, li)
            results[f"L{li}"] = r
            print(f"  L{li}: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"PER-LAYER SWEEP SUMMARY (alpha={args.alpha}, {args.n_test} problems)", flush=True)
    print(f"{'='*60}", flush=True)
    baseline_acc = results["baseline"]["accuracy"]
    print(f"  {'Layer':>6} {'Acc':>8s} {'Delta':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    print(f"  {'base':>6} {100*baseline_acc:7.1f}%")
    best_delta = 0
    best_layer = None
    for li in layers_to_test:
        r = results.get(f"L{li}")
        if r:
            acc = r["accuracy"]
            delta = acc - baseline_acc
            marker = "  <- BEST" if acc > baseline_acc else ""
            if delta > best_delta:
                best_delta = delta
                best_layer = li
            print(f"  L{li:3d} {100*acc:7.1f}% {100*delta:+7.1f}pp{marker}", flush=True)

    if best_layer is not None:
        print(f"\n  Best trim-tab: L{best_layer} ({100*best_delta:+.1f}pp)", flush=True)

    # Save
    out_path = f"awq_sweep_a{args.alpha}_n{args.n_test}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
