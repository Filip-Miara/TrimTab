#!/usr/bin/env python3 -u
"""Per-layer steering sweep with batched problem processing.

Instead of one problem at a time (~25s each), processes B problems
simultaneously with batched TT inference and KV-cache steering.
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
BATCH_SIZE = 8


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def steer_batch(model, hidden_states, velocity, pkv_list, alpha, target_layer):
    """Steer one layer for ALL items in a batch."""
    # hidden_states: tuple of (batch, seq, d_model) per layer
    # velocity: (batch, n_layers, d_model)
    hs_layer = hidden_states[target_layer + 1]  # (batch, seq, d_model)
    batch_vel = velocity[:, target_layer, :]    # (batch, d_model)
    
    h_end = hs_layer[:, -1, :]  # (batch, d_model) - last token for each item
    h_steered = h_end + alpha * batch_vel       # (batch, d_model)
    
    layer = model.model.layers[target_layer]
    
    # Compute K and V for all batch items at once
    # But KV cache modification is per-item due to different sequence lengths
    k_all = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))  # (batch, kv_heads * head_dim)
    v_all = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
    
    k_all = k_all.view(-1, N_KV_HEADS, 1, HEAD_DIM)    # (batch, kv_heads, 1, head_dim)
    v_all = v_all.view(-1, N_KV_HEADS, 1, HEAD_DIM)
    
    for idx, pkv in enumerate(pkv_list):
        lc = pkv.layers[target_layer]
        # Replace LAST token's K/V for this batch item
        lc.keys[idx:idx+1, :, -1:, :] = k_all[idx:idx+1, :, :, :].to(lc.keys.dtype)
        lc.values[idx:idx+1, :, -1:, :] = v_all[idx:idx+1, :, :, :].to(lc.values.dtype)


def evaluate_layer_batched(problems, tok, model, tt, alpha, target_layer, batch_size=BATCH_SIZE):
    """Evaluate with batched problem processing."""
    n_problems = len(problems)
    results = []
    t0 = time.time()
    
    for batch_start in range(0, n_problems, batch_size):
        batch_end = min(batch_start + batch_size, n_problems)
        batch = problems[batch_start:batch_end]
        batch_size = len(batch)
        
        # Prepare batch inputs
        prompts = []
        for prob in batch:
            prompts.append(f'Q: {prob["question"]}\nA:')
        
        if alpha == 0.0 or target_layer is None:
            # BASELINE: use model.generate with batched input
            msgs_list = [[{"role": "user", "content": p}] for p in prompts]
            input_ids_list = []
            for msgs in msgs_list:
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                              return_tensors="pt").input_ids.to(DEVICE)
                input_ids_list.append(ids)
            
            # Pad to same length
            max_len = max(ids.shape[1] for ids in input_ids_list)
            padded = torch.full((batch_size, max_len), tok.pad_token_id, dtype=torch.long, device=DEVICE)
            attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=DEVICE)
            for i, ids in enumerate(input_ids_list):
                padded[i, :ids.shape[1]] = ids[0]
                attn_mask[i, :ids.shape[1]] = 1
            
            out = model.generate(padded, attention_mask=attn_mask, max_new_tokens=MAX_GEN,
                                 do_sample=False, pad_token_id=tok.eos_token_id, use_cache=True)
            
            for i in range(batch_size):
                gen = tok.decode(out[i, input_ids_list[i].shape[1]:], skip_special_tokens=True)
                ca = re.search(r"####\s*(-?\d+)", batch[i]["answer"])
                predicted = extract_number(gen)
                correct = (predicted is not None and ca and predicted == ca.group(1))
                results.append(correct)
        else:
            # STEERED: manual loop with batch processing
            # Prefill all batch items
            past_key_values_list = [None] * batch_size
            generated_ids = [[] for _ in range(batch_size)]
            input_ids_list = []
            unfinished = [True] * batch_size
            
            for i, p in enumerate(prompts):
                msgs = [{"role": "user", "content": p}]
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                              return_tensors="pt").input_ids.to(DEVICE)
                input_ids_list.append(ids)
            
            first_step = True
            for step in range(MAX_GEN):
                if not any(unfinished):
                    break
                
                # Forward pass for all unfinished items
                fwd_list = []
                for i in range(batch_size):
                    if not unfinished[i]:
                        fwd_list.append(None)
                        continue
                    
                    with torch.no_grad():
                        fwd = model(
                            input_ids_list[i],
                            past_key_values=past_key_values_list[i],
                            use_cache=True,
                            output_hidden_states=True,
                        )
                    fwd_list.append(fwd)
                
                # Get next tokens and check EOS
                for i in range(batch_size):
                    if not unfinished[i]:
                        continue
                    fwd = fwd_list[i]
                    nt = fwd.logits[0, -1, :].argmax().item()
                    if nt == tok.eos_token_id:
                        unfinished[i] = False
                        continue
                    generated_ids[i].append(nt)
                
                # Apply steering (only after first step, when we have hidden states)
                if not first_step:
                    # Collect hidden states for all unfinished items
                    hs_list = []
                    batch_indices = []
                    for i in range(batch_size):
                        if unfinished[i] and fwd_list[i] is not None:
                            hs = fwd_list[i].hidden_states
                            if hs is not None:
                                h_pos = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                                hs_list.append(h_pos)
                                batch_indices.append(i)
                    
                    if hs_list:
                        # Stack and run TT on batch
                        x = torch.stack(hs_list, dim=0).to(DEVICE)  # (n_unfinished, n_layers, d_model)
                        with torch.no_grad():
                            v = tt(x)  # (n_unfinished, n_layers, d_model)
                        
                        # Steer each item
                        pkv_list = [fwd_list[i].past_key_values for i in batch_indices]
                        steer_batch(model, fwd_list[batch_indices[0]].hidden_states, v, pkv_list, alpha, target_layer)
                        
                        # Update past key values for steered items
                        for j, i in enumerate(batch_indices):
                            past_key_values_list[i] = fwd_list[i].past_key_values
                
                # Update input_ids for next step
                for i in range(batch_size):
                    if unfinished[i]:
                        input_ids_list[i] = torch.tensor([[generated_ids[i][-1]]], device=DEVICE)
                        if first_step and step == 0:
                            past_key_values_list[i] = fwd_list[i].past_key_values
                
                first_step = False
            
            # Decode and check answers
            for i in range(batch_size):
                gen = tok.decode(generated_ids[i], skip_special_tokens=True)
                ca = re.search(r"####\s*(-?\d+)", batch[i]["answer"])
                predicted = extract_number(gen)
                correct = (predicted is not None and ca and predicted == ca.group(1))
                results.append(correct)
        
        # Progress report - compute correct so far
        done = len(results)
        correct_count = sum(results)
        label = f"L{target_layer:2d}" if target_layer is not None else "base"
        print(f"  {label} a={alpha} [{done}/{n_problems}] acc={correct_count}/{done} "
              f"({100*correct_count/done:.0f}%) ({time.time()-t0:.0f}s)", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    
    correct = sum(results)
    total = len(results)
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tt-path", type=str, default=TT_PATH)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    
    layers_to_test = args.layers if args.layers is not None else list(range(N_LAYERS))
    
    # Load model
    print(f"Loading AWQ model...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    print(f"Model loaded in {time.time()-t0:.1f}s, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)
    
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
    print("Loading GSM8K...", flush=True)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    print(f"Testing on {len(problems)} problems (batch_size={BATCH_SIZE})", flush=True)
    
    results = {}
    
    # Baseline
    print(f"\n{'='*60}", flush=True)
    print(f"Baseline (no steering, batch={BATCH_SIZE})", flush=True)
    print(f"{'='*60}", flush=True)
    r = evaluate_layer_batched(problems, tok, model, tt, 0.0, None, args.batch_size)
    results["baseline"] = r
    print(f"  Baseline: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)
    
    if not args.baseline_only:
        for li in layers_to_test:
            print(f"\n{'='*60}", flush=True)
            print(f"Layer {li} only, alpha={args.alpha}, batch={args.batch_size}", flush=True)
            print(f"{'='*60}", flush=True)
            r = evaluate_layer_batched(problems, tok, model, tt, args.alpha, li, args.batch_size)
            results[f"L{li}"] = r
            print(f"  L{li}: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)", flush=True)
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SWEEP SUMMARY (alpha={args.alpha}, {args.n_test} problems, batch={BATCH_SIZE})", flush=True)
    print(f"{'='*60}", flush=True)
    baseline_acc = results["baseline"]["accuracy"]
    print(f"  {'Layer':>6} {'Acc':>8s} {'Delta':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    print(f"  {'base':>6} {100*baseline_acc:7.1f}%")
    for li in layers_to_test:
        r = results.get(f"L{li}")
        if r:
            acc = r["accuracy"]
            delta = acc - baseline_acc
            marker = "  <- BEST" if delta > 0 else ""
            print(f"  L{li:3d} {100*acc:7.1f}% {100*delta:+7.1f}pp{marker}", flush=True)
    
    out_path = f"awq_sweep_b{BATCH_SIZE}_a{args.alpha}_n{args.n_test}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
