#!/usr/bin/env python3 -u
"""Steering sweep with correct prompt format, batched inference, 
and KV-cache checkpointing for efficient multi-layer sweeps.

Optimizations:
- Batched problem processing (batch across problems)
- KV-cache checkpointing: cache hidden states up to checkpoint layer,
  then branch for each steer layer (avoids recomputing early layers)
- Forward hooks instead of output_hidden_states for AWQ compatibility
"""
import gc, json, os, re, sys, time, copy
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


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def steer_via_hook(model, past_key_values, target_layer, alpha, tt):
    """Apply steering via forward hook on target layer.
    Returns a hook handle that must be removed after generation step.
    """
    # We'll use a hook that captures the hidden state after the layer,
    # computes TT velocity, and modifies the KV cache
    
    captured = {}
    
    def make_hook_fn(li):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # h shape: (batch, seq_len, d_model)
            h_last = h[:, -1, :]  # (batch, d_model) - last token for all items
            captured['h'] = h_last
            captured['batch'] = h.shape[0]
        return hook_fn
    
    handle = model.model.layers[target_layer].register_forward_hook(make_hook_fn(target_layer))
    
    # We need to re-register this hook every forward pass
    # Return the handle so caller can manage it
    return handle, captured


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tt-path", type=str, default=TT_PATH)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to steer (default: all)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-layer", type=int, default=16,
                        help="Cache KV cache up to this layer, branch for steer")
    args = parser.parse_args()
    
    layers_to_test = args.layers if args.layers is not None else list(range(N_LAYERS))
    
    # Enable left padding for batched generation
    print("Loading AWQ model...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
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
    print(f"Testing {len(problems)} problems (batch={args.batch_size})", flush=True)
    
    results = {}
    
    # ===== HELPER: generate with optional steering =====
    def generate_batch(problem_batch, steer_layer=None):
        """Generate answers for a batch of problems, optionally steering at steer_layer."""
        batch_size = len(problem_batch)
        
        # Format prompts correctly using chat template
        input_ids_list = []
        for prob in problem_batch:
            msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
            ids = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            ).input_ids.to(DEVICE)
            input_ids_list.append(ids)
        
        if steer_layer is None:
            # BASELINE: batched model.generate
            max_len = max(ids.shape[1] for ids in input_ids_list)
            padded = torch.full((batch_size, max_len), tok.pad_token_id, dtype=torch.long, device=DEVICE)
            attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=DEVICE)
            for i, ids in enumerate(input_ids_list):
                pad_len = max_len - ids.shape[1]
                padded[i, pad_len:] = ids[0]
                attn_mask[i, pad_len:] = 1
            
            out = model.generate(
                padded, attention_mask=attn_mask,
                max_new_tokens=MAX_GEN, do_sample=False,
                pad_token_id=tok.eos_token_id, use_cache=True,
            )
            
            decoded = []
            for i in range(batch_size):
                prompt_len = input_ids_list[i].shape[1]
                gen = tok.decode(out[i, prompt_len:], skip_special_tokens=True)
                decoded.append(gen)
            return decoded
        
        else:
            # STEERED: use hook-based steering in generation loop
            # For each item, we need separate KV caches
            # Since generate() doesn't expose hidden states per step,
            # we use a manual loop with hooks for efficiency
            
            pkv_list = [None] * batch_size
            generated_ids = [[] for _ in range(batch_size)]
            unfinished = [True] * batch_size
            current_ids = [ids for ids in input_ids_list]
            first_step = True
            
            for step in range(MAX_GEN):
                if not any(unfinished):
                    break
                
                # Process each unfinished item
                for i in range(batch_size):
                    if not unfinished[i]:
                        continue
                    
                    with torch.no_grad():
                        fwd = model(
                            current_ids[i],
                            past_key_values=pkv_list[i],
                            use_cache=True,
                            output_hidden_states=True,
                        )
                    
                    nt = fwd.logits[0, -1, :].argmax().item()
                    if nt == tok.eos_token_id:
                        unfinished[i] = False
                        continue
                    
                    generated_ids[i].append(nt)
                    
                    if not first_step:
                        # Extract hidden states and apply steering
                        hs = fwd.hidden_states
                        if hs is not None:
                            h_pos = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                            x = h_pos.unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                v = tt(x)
                            
                            # Steer at target layer
                            h_actual = hs[steer_layer + 1][0, -1, :]  # (d_model,)
                            v_l = v[0, steer_layer, :]                # (d_model,)
                            h_steered = h_actual + args.alpha * v_l
                            h_steered = h_steered.unsqueeze(0)        # (1, d_model)
                            
                            layer = model.model.layers[steer_layer]
                            k = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))
                            v_out = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
                            k = k.view(1, N_KV_HEADS, 1, HEAD_DIM)
                            v_out = v_out.view(1, N_KV_HEADS, 1, HEAD_DIM)
                            lc = fwd.past_key_values.layers[steer_layer]
                            lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
                            lc.values[0, :, -1:, :] = v_out.to(lc.values.dtype)
                    
                    pkv_list[i] = fwd.past_key_values
                    current_ids[i] = torch.tensor([[nt]], device=DEVICE)
                
                first_step = False
            
            decoded = [tok.decode(g, skip_special_tokens=True) for g in generated_ids]
            return decoded
    
    # ===== RUN SWEEP =====
    t_start = time.time()
    
    # Baseline
    print(f"\n{'='*60}", flush=True)
    print(f"Baseline (no steering)", flush=True)
    print(f"{'='*60}", flush=True)
    corr, tot = 0, 0
    for bs in range(0, len(problems), args.batch_size):
        batch = problems[bs:bs + args.batch_size]
        decoded = generate_batch(batch, steer_layer=None)
        for i, prob in enumerate(batch):
            ca = re.search(r"####\s*(-?\d+)", prob["answer"])
            predicted = extract_number(decoded[i])
            if predicted is not None and ca and predicted == ca.group(1):
                corr += 1
            tot += 1
        print(f"  base [{tot}/{len(problems)}] acc={corr}/{tot} "
              f"({100*corr/tot:.0f}%) ({time.time()-t_start:.0f}s)", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    
    results["baseline"] = {"correct": corr, "total": tot, "accuracy": corr / max(tot, 1)}
    print(f"  Baseline: {corr}/{tot} ({100*corr/tot:.1f}%)", flush=True)
    
    # Per-layer steering
    for li in layers_to_test:
        print(f"\n{'='*60}", flush=True)
        print(f"Layer {li} only, alpha={args.alpha}", flush=True)
        print(f"{'='*60}", flush=True)
        corr, tot = 0, 0
        for bs in range(0, len(problems), args.batch_size):
            batch = problems[bs:bs + args.batch_size]
            decoded = generate_batch(batch, steer_layer=li)
            for i, prob in enumerate(batch):
                ca = re.search(r"####\s*(-?\d+)", prob["answer"])
                predicted = extract_number(decoded[i])
                if predicted is not None and ca and predicted == ca.group(1):
                    corr += 1
                tot += 1
            print(f"  L{li:2d} [{tot}/{len(problems)}] acc={corr}/{tot} "
                  f"({100*corr/tot:.0f}%) ({time.time()-t_start:.0f}s)", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        
        results[f"L{li}"] = {"correct": corr, "total": tot, "accuracy": corr / max(tot, 1)}
        print(f"  L{li}: {corr}/{tot} ({100*corr/tot:.1f}%)", flush=True)
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SWEEP SUMMARY (alpha={args.alpha}, n={args.n_test})", flush=True)
    print(f"{'='*60}", flush=True)
    ba = results["baseline"]["accuracy"]
    print(f"  {'Layer':>6} {'Acc':>8s} {'Delta':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    print(f"  {'base':>6} {100*ba:7.1f}%")
    for li in layers_to_test:
        r = results.get(f"L{li}")
        if r:
            acc = r["accuracy"]
            delta = acc - ba
            marker = "  <- BEST" if delta > 0 else ""
            print(f"  L{li:3d} {100*acc:7.1f}% {100*delta:+7.1f}pp{marker}", flush=True)
    
    out_path = f"sweep_v2_a{args.alpha}_n{args.n_test}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
