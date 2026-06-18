#!/usr/bin/env python3 -u
"""Steering with FULLY BATCHED forward pass.
Instead of looping over items, concatenates KV caches and runs
ONE forward pass per token for ALL items.
"""
import gc, json, os, re, sys, time, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.optimization import setup_optimizations
setup_optimizations()

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
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
    m = re.search(r"####\s*(-?\d+)", text)
    if m: return m.group(1)
    for p in [r"(?:answer|result|value|\bThe\s+answer)\s+is\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    tail = text[-200:]
    nums = re.findall(r"-?\d+", tail)
    if nums: return nums[-1]
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None

def concat_pkv(pkv_list, indices, n_layers=N_LAYERS):
    """Concatenate per-item KV caches into a single batched DynamicCache."""
    if not indices:
        return None
    if len(indices) == 1:
        return pkv_list[indices[0]]
    
    keys = []
    values = []
    for l in range(n_layers):
        k = torch.cat([pkv_list[i].key_cache[l] for i in indices], dim=0)
        v = torch.cat([pkv_list[i].value_cache[l] for i in indices], dim=0)
        keys.append(k)
        values.append(v)
    return DynamicCache.from_legacy_cache((keys, values))

def split_pkv(batched_pkv, indices, n_layers=N_LAYERS):
    """Split a batched DynamicCache back into per-item caches."""
    result = {}
    for idx_pos, orig_idx in enumerate(indices):
        keys = []
        values = []
        for l in range(n_layers):
            k = batched_pkv.key_cache[l][idx_pos:idx_pos+1]
            v = batched_pkv.value_cache[l][idx_pos:idx_pos+1]
            keys.append(k)
            values.append(v)
        result[orig_idx] = DynamicCache.from_legacy_cache((keys, values))
    return result

def steer_batch_at_layer(model, fwd, tt, alpha, target_layer, unfinished_indices, bs):
    """Apply steering to the batched forward's outputs."""
    hs = fwd.hidden_states
    if hs is None:
        return
    
    # Get hidden states at target layer for all items
    h_at_layer = hs[target_layer + 1]  # (bs, seq_len, d_model)
    h_last = h_at_layer[:, -1, :]  # (bs, d_model) - last token for each item
    
    # We need all 28 hidden states for the TT - use hs tuple
    # hs[layer_idx] has shape (batch, seq_len, d_model)
    # Stack all layer outputs for the TT
    all_h = torch.stack([h[:, -1, :].float() for h in hs[:N_LAYERS]], dim=0)  # (n_layers, bs, d_model)
    all_h = all_h.permute(1, 0, 2)  # (bs, n_layers, d_model)
    
    with torch.no_grad():
        v = tt(all_h)  # (bs, n_layers, d_model)
    
    v_target = v[:, target_layer, :]  # (bs, d_model)
    h_steered = h_last + alpha * v_target  # (bs, d_model)
    
    layer = model.model.layers[target_layer]
    k_all = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))  # (bs, kv_heads * head_dim)
    v_all = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
    k_all = k_all.view(-1, N_KV_HEADS, 1, HEAD_DIM)
    v_all = v_all.view(-1, N_KV_HEADS, 1, HEAD_DIM)
    
    pkv = fwd.past_key_values
    lc = pkv.layers[target_layer]
    lc.keys[:, :, -1:, :] = k_all.to(lc.keys.dtype)
    lc.values[:, :, -1:, :] = v_all.to(lc.values.dtype)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

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
    
    print("Loading TT...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=D_MODEL)
    tt.load_state_dict(torch.load(TT_PATH, map_location="cpu"), strict=False)
    tt.to(DEVICE).eval()
    for p in tt.parameters(): p.requires_grad = False
    print(f"TT: {sum(p.numel() for p in tt.parameters()):,} params", flush=True)
    
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    print(f"Testing {len(problems)} problems, L{args.layer} a={args.alpha} bs={args.batch_size}", flush=True)
    
    t_start = time.time()
    li = args.layer
    corr = 0
    
    for bs_start in range(0, len(problems), args.batch_size):
        bs_end = min(bs_start + args.batch_size, len(problems))
        batch = problems[bs_start:bs_end]
        bs = len(batch)
        
        # Tokenize
        msgs = [[{"role": "user", "content": f'Q: {p["question"]}\nA:'}] for p in batch]
        all_ids = [tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE) for m in msgs]
        max_len = max(ids.shape[1] for ids in all_ids)
        padded = torch.full((bs, max_len), tok.pad_token_id, dtype=torch.long, device=DEVICE)
        mask = torch.zeros((bs, max_len), dtype=torch.long, device=DEVICE)
        for i, ids in enumerate(all_ids):
            pad_amt = max_len - ids.shape[1]
            padded[i, pad_amt:] = ids[0]
            mask[i, pad_amt:] = 1
        
        # PREFILL: run full forward pass to get initial KV cache
        with torch.no_grad():
            fwd = model(padded, attention_mask=mask, use_cache=True, output_hidden_states=True)
        
        pkv = fwd.past_key_values  # batched KV cache (bs, heads, seq_len, head_dim)
        gen_ids = [[] for _ in range(bs)]
        unfinished = [True] * bs
        cur_tokens = fwd.logits[:, -1, :].argmax(dim=-1)  # (bs,) - first generated token
        first_step = True
        
        for step in range(MAX_GEN):
            if not any(unfinished):
                break
            
            # Process ALL unfinished items in ONE forward pass
            u_indices = [i for i in range(bs) if unfinished[i]]
            n_u = len(u_indices)
            
            if n_u < bs:
                # Some items finished - extract subset of KV cache
                u_pkv = concat_pkv({i: pkv for i in [0]}, u_indices if n_u < bs else list(range(bs)))
                # Actually we need to handle pkv correctly - let's just use a different approach
                # For now, if some finished, continue with all (they'll just generate EOS)
                pass
            
            with torch.no_grad():
                fwd = model(cur_tokens.unsqueeze(-1), past_key_values=pkv, use_cache=True, output_hidden_states=True)
            
            next_toks = fwd.logits[:, -1, :].argmax(dim=-1)
            
            # Update per-item state
            for i in range(bs):
                if not unfinished[i]:
                    continue
                nt = next_toks[i].item()
                if nt == tok.eos_token_id:
                    unfinished[i] = False
                else:
                    gen_ids[i].append(nt)
            
            # Apply steering (after first step, when we have hidden states from prefill+1)
            if not first_step:
                steer_batch_at_layer(model, fwd, tt, args.alpha, li, 
                                    [i for i in range(bs) if unfinished[i]], bs)
            
            pkv = fwd.past_key_values
            cur_tokens = next_toks
            first_step = False
        
        # Evaluate
        for i, prob in enumerate(batch):
            gen = tok.decode(gen_ids[i], skip_special_tokens=True)
            ca = re.search(r"####\s*(-?\d+)", prob["answer"])
            pred = extract_number(gen)
            if pred is not None and ca and pred == ca.group(1):
                corr += 1
        
        print(f"  L{li:2d} [{bs_end}/{len(problems)}] acc={corr}/{bs_end} "
              f"({100*corr/bs_end:.0f}%) ({time.time()-t_start:.0f}s)", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\n  L{li}: {corr}/{len(problems)} ({100*corr/len(problems):.1f}%) "
          f"a={args.alpha} ({time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
