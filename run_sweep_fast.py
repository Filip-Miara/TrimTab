#!/usr/bin/env python3 -u
"""Fast steering sweep. Batches ALL problems at once for baseline throughput.
For steering: processes problems in batches with forwarded hooks.

Baseline: all problems in one model.generate() call.
Steering: small batches with manual loop but forward hook capture.
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
MAX_GEN = 400
MODEL_PATH = "D:\\Qwen2.5-7B-AWQ\\qwen7b_awq"
TT_PATH = "best_tt_awq_7b.pt"
D_MODEL = 3584
N_LAYERS = 28
N_KV_HEADS = 4
HEAD_DIM = 128


def extract_number(text):
    # GSM8K format
    m = re.search(r"####\s*(-?\d+)", text)
    if m: return m.group(1)
    # The answer is N / answer is N
    for p in [r"(?:answer|result|value|\bThe\s+answer)\s+is\s*(-?\d+)",
              r"(?:final answer|therefore|thus|so)[\s,]*.*?(-?\d+)\s*(?:\..*)?$"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    # Last number from last 300 chars (final reasoning usually has the answer)
    tail = text[-300:]
    nums = re.findall(r"(?:^|\s)(\d+)(?:\s|$|\.|,)", tail)
    if nums: return nums[-1]
    # Absolute fallback: last number in full text
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    
    layers_to_test = args.layers if args.layers is not None else list(range(N_LAYERS))
    
    print("Loading model...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    print(f"Loaded in {time.time()-t0:.1f}s, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)
    
    # Load TT
    print(f"Loading TT...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=D_MODEL)
    tt.load_state_dict(torch.load(TT_PATH, map_location="cpu"), strict=False)
    tt.to(DEVICE).eval()
    for p in tt.parameters():
        p.requires_grad = False
    print(f"TT: {sum(p.numel() for p in tt.parameters()):,} params", flush=True)
    
    # GSM8K
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    print(f"Testing {len(problems)} problems", flush=True)
    
    results = {}
    t_start = time.time()
    
    # ===== BASELINE: batched generate (b=batch_size for memory) =====
    print(f"\n{'='*60}", flush=True)
    print(f"BASELINE (batch={args.batch_size})", flush=True)
    print(f"{'='*60}", flush=True)
    
    corr = 0
    t_start_base = time.time()
    n = len(problems)
    
    for bs_start in range(0, n, args.batch_size):
        bs_end = min(bs_start + args.batch_size, n)
        batch = problems[bs_start:bs_end]
        bs = len(batch)
        
        msgs_batch = [[{"role": "user", "content": f'Q: {p["question"]}\nA:'}] for p in batch]
        ids_batch = [
            tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                    return_tensors="pt").input_ids.to(DEVICE)
            for msgs in msgs_batch
        ]
        max_len = max(ids.shape[1] for ids in ids_batch)
        padded = torch.full((bs, max_len), tok.pad_token_id, dtype=torch.long, device=DEVICE)
        mask = torch.zeros((bs, max_len), dtype=torch.long, device=DEVICE)
        for i, ids in enumerate(ids_batch):
            pad_amt = max_len - ids.shape[1]
            padded[i, pad_amt:] = ids[0]
            mask[i, pad_amt:] = 1
        
        out = model.generate(padded, attention_mask=mask, max_new_tokens=MAX_GEN,
                             do_sample=False, pad_token_id=tok.eos_token_id, use_cache=True)
        
        for i, prob in enumerate(batch):
            gen = tok.decode(out[i, max_len:], skip_special_tokens=True)
            ca = re.search(r"####\s*(-?\d+)", prob["answer"])
            pred = extract_number(gen)
            if pred is not None and ca and pred == ca.group(1):
                corr += 1
        
        done = min(bs_end, n)
        print(f"  base [{done}/{n}] acc={corr}/{done} "
              f"({100*corr/done:.0f}%) ({time.time()-t_start:.0f}s)", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    
    results["baseline"] = {"correct": corr, "total": n, "accuracy": corr / n}
    print(f"  Baseline: {corr}/{n} ({100*corr/n:.1f}%) ({time.time()-t_start_base:.0f}s)", flush=True)
    
    # ===== STEERED LAYERS: batched per layer =====
    # Only test key layers plus any user-specified
    key_layers = [l for l in [2, 8, 9, 15] if l in layers_to_test]
    other_layers = [l for l in layers_to_test if l not in key_layers]
    for li in key_layers + other_layers:
        print(f"\n{'='*60}", flush=True)
        print(f"LAYER {li} (alpha={args.alpha}, batch={args.batch_size})", flush=True)
        print(f"{'='*60}", flush=True)
        
        corr = 0
        layer_time = time.time()
        
        for bs_start in range(0, n, args.batch_size):
            bs_end = min(bs_start + args.batch_size, n)
            batch = problems[bs_start:bs_end]
            bs = len(batch)
            
            # Prepare batch prompts
            msgs_batch = [[{"role": "user", "content": f'Q: {p["question"]}\nA:'}] for p in batch]
            ids_batch = [
                tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                        return_tensors="pt").input_ids.to(DEVICE)
                for msgs in msgs_batch
            ]
            
            # Manual generation loop with steering
            pkv_list = [None] * bs
            gen_ids = [[] for _ in range(bs)]
            unfinished = [True] * bs
            cur_ids = [ids for ids in ids_batch]
            first_step = True
            
            for step in range(MAX_GEN):
                if not any(unfinished):
                    break
                
                for i in range(bs):
                    if not unfinished[i]:
                        continue
                    
                    with torch.no_grad():
                        fwd = model(cur_ids[i], past_key_values=pkv_list[i],
                                     use_cache=True, output_hidden_states=True)
                    
                    nt = fwd.logits[0, -1, :].argmax().item()
                    if nt == tok.eos_token_id:
                        unfinished[i] = False
                        continue
                    gen_ids[i].append(nt)
                    
                    if not first_step:
                        hs = fwd.hidden_states
                        if hs is not None:
                            h_t = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0)
                            x = h_t.unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                v = tt(x)
                            h_act = hs[li + 1][0, -1, :]
                            h_steered = (h_act + args.alpha * v[0, li, :]).unsqueeze(0)
                            
                            layer = model.model.layers[li]
                            k = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))
                            vo = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
                            k = k.view(1, N_KV_HEADS, 1, HEAD_DIM)
                            vo = vo.view(1, N_KV_HEADS, 1, HEAD_DIM)
                            lc = fwd.past_key_values.layers[li]
                            lc.keys[0, :, -1:, :] = k.to(lc.keys.dtype)
                            lc.values[0, :, -1:, :] = vo.to(lc.values.dtype)
                    
                    pkv_list[i] = fwd.past_key_values
                    cur_ids[i] = torch.tensor([[nt]], device=DEVICE)
                
                first_step = False
            
            # Evaluate batch
            for i, prob in enumerate(batch):
                gen = tok.decode(gen_ids[i], skip_special_tokens=True)
                ca = re.search(r"####\s*(-?\d+)", prob["answer"])
                pred = extract_number(gen)
                if pred is not None and ca and pred == ca.group(1):
                    corr += 1
            
            done = bs_end
            print(f"  L{li:2d} [{done}/{n}] acc={corr}/{done} "
                  f"({100*corr/done:.0f}%) ({time.time()-t_start:.0f}s)", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        
        results[f"L{li}"] = {"correct": corr, "total": n, "accuracy": corr / n}
        print(f"  L{li}: {corr}/{n} ({100*corr/n:.1f}%)", flush=True)
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SWEEP SUMMARY", flush=True)
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
            print(f"  L{li:3d} {100*acc:7.1f}% {100*delta:+7.1f}pp")
    
    out_path = f"sweep_fast_a{args.alpha}_n{args.n_test}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path} ({time.time()-t_start:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
