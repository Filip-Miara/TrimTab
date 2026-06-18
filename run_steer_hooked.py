#!/usr/bin/env python3 -u
"""Steering with forward hooks instead of output_hidden_states.
Saves ~800MB VRAM by not storing all 28 full hidden states.
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

class HiddenStateCollector:
    """Collects last-token hidden states from all layers via forward hooks.
    Each hook stores (batch, d_model) for its layer - only last token.
    After forward pass, stack into (batch, n_layers, d_model) for TT.
    Also captures last-token hidden state at steer layer for K/V computation.
    Only clones last token, NOT full sequence - saves ~800MB.
    """
    def __init__(self, model, steer_layer):
        self.handles = []
        self.states = [None] * N_LAYERS
        self.steer_last = None  # only last token, not full sequence
        self.steer_layer = steer_layer
        
        for li in range(N_LAYERS):
            def make_hook(layer_idx):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    self.states[layer_idx] = h[:, -1:, :].clone()  # (batch, 1, d_model)
                    if layer_idx == steer_layer:
                        self.steer_last = h[:, -1:, :].clone()  # (batch, 1, d_model)
                return hook
            handle = model.model.layers[li].register_forward_hook(make_hook(li))
            self.handles.append(handle)
    
    def get_tensor(self, batch_size):
        """Stack collected states into (batch, n_layers, d_model) for TT."""
        h_list = []
        for s in self.states[:N_LAYERS]:
            if s is not None:
                h_list.append(s[:batch_size, 0, :].float())  # (batch, d_model)
        if len(h_list) == N_LAYERS:
            return torch.stack(h_list, dim=1)  # (batch, n_layers, d_model)
        return None
    
    def get_steer_hidden(self):
        """Get last-token hidden state at steer layer for K/V computation."""
        if self.steer_last is not None:
            return self.steer_last[:, 0, :]  # (batch, d_model)
        return None
    
    def remove(self):
        for h in self.handles:
            h.remove()

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
        
        # Register hooks BEFORE forward
        collector = HiddenStateCollector(model, li)
        
        # PREFILL: run without hooks for initial KV cache
        collector.remove()  # remove hooks for prefill
        with torch.no_grad():
            fwd = model(padded, attention_mask=mask, use_cache=True)
        collector = HiddenStateCollector(model, li)  # re-register hooks for generation
        
        pkv = fwd.past_key_values
        gen_ids = [[] for _ in range(bs)]
        unfinished = [True] * bs
        cur_tokens = fwd.logits[:, -1, :].argmax(dim=-1)
        first_step = True
        
        for step in range(MAX_GEN):
            if not any(unfinished):
                break
            
            # Forward pass WITHOUT output_hidden_states - hooks capture what we need
            with torch.no_grad():
                fwd = model(cur_tokens.unsqueeze(-1), past_key_values=pkv, use_cache=True)
            
            next_toks = fwd.logits[:, -1, :].argmax(dim=-1)
            
            for i in range(bs):
                if not unfinished[i]:
                    continue
                nt = next_toks[i].item()
                if nt == tok.eos_token_id:
                    unfinished[i] = False
                else:
                    gen_ids[i].append(nt)
            
            # Apply steering (after first step)
            if not first_step:
                # Get hidden states from collector
                all_h = collector.get_tensor(bs)  # (batch, n_layers, d_model)
                steer_h = collector.get_steer_hidden()  # (batch, d_model)
                
                if all_h is not None and steer_h is not None:
                    with torch.no_grad():
                        v_all = tt(all_h)  # (batch, n_layers, d_model)
                    
                    # steer_h is already (batch, d_model) - the last token's hidden state
                    h_steered = steer_h + args.alpha * v_all[:, li, :]
                    
                    layer = model.model.layers[li]
                    k_all = layer.self_attn.k_proj(h_steered.to(torch.bfloat16))
                    v_out_all = layer.self_attn.v_proj(h_steered.to(torch.bfloat16))
                    k_all = k_all.view(-1, N_KV_HEADS, 1, HEAD_DIM)
                    v_out_all = v_out_all.view(-1, N_KV_HEADS, 1, HEAD_DIM)
                    
                    lc = fwd.past_key_values.layers[li]
                    lc.keys[:, :, -1:, :] = k_all.to(lc.keys.dtype)
                    lc.values[:, :, -1:, :] = v_out_all.to(lc.values.dtype)
            
            pkv = fwd.past_key_values
            cur_tokens = next_toks
            first_step = False
        
        collector.remove()
        
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
