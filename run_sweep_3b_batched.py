#!/usr/bin/env python3 -u
"""Batched 36-layer sweep on Qwen2.5-3B via 7B→3B TT transfer.
Processes BS problems in parallel for ~10× speedup.
"""
from __future__ import annotations

import gc, json, re, sys, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_PATH = "/home/filip/.cache/huggingface/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
TT_PATH = "best_gen_tt_7b.pt"
N_LAYERS_7B = 28; N_LAYERS_3B = 36
D_INPUT_7B = 3584; D_INPUT_3B = 2048; D_MODEL = 768
ALPHA = 0.1; MAX_GEN = 400
BS = 16  # process 16 problems at once

def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None

def transfer_tt():
    chk = torch.load(TT_PATH, map_location="cpu")
    d_model = chk["input_proj.weight"].shape[0]
    src_tt = TrajectoryTransformer(d_model=d_model, n_layers=6, n_heads=8, d_ff=d_model*4,
                                    n_positions=N_LAYERS_7B, d_input=D_INPUT_7B)
    src_tt.load_state_dict(chk, strict=False)
    tgt_tt = TrajectoryTransformer(d_model=d_model, n_layers=6, n_heads=8, d_ff=d_model*4,
                                    n_positions=N_LAYERS_3B, d_input=D_INPUT_3B)
    src, tgt = src_tt.state_dict(), tgt_tt.state_dict()
    for key in src:
        if key in tgt and src[key].shape == tgt[key].shape: tgt[key].copy_(src[key])
        elif key in tgt and 'pos_embed' in key:
            n = min(src[key].shape[0], tgt[key].shape[0])
            tgt[key][:n].copy_(src[key][:n])
    with torch.no_grad():
        n = min(D_INPUT_7B, D_INPUT_3B)
        tgt_tt.input_proj.weight[:, :n].copy_(src_tt.input_proj.weight[:, :n])
        tgt_tt.input_proj.bias.copy_(src_tt.input_proj.bias)
        tgt_tt.output_proj.weight[:n, :].copy_(src_tt.output_proj.weight[:n, :])
        tgt_tt.output_proj.bias[:n].copy_(src_tt.output_proj.bias[:n])
    return tgt_tt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--bs", type=int, default=BS)
    args = parser.parse_args()

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id; tok.padding_side = "left"
    head_dim = 128; n_kv_heads = 2

    print("Transferring 7B TT→3B...", flush=True)
    tt = transfer_tt().to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad_(False)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]
    layers = args.layers if args.layers else list(range(N_LAYERS_3B))

    # Batch baseline
    print(f"\nBaseline ({args.n_test} problems, BS={args.bs})...", flush=True)
    base_correct = [0] * args.n_test
    for start in range(0, args.n_test, args.bs):
        end = min(start + args.bs, args.n_test)
        batch = problems[start:end]
        prompts = [f"Q: {p['question']}\nA:" for p in batch]
        encoded = tok(prompts, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(DEVICE)
        attn_mask = encoded["attention_mask"].to(DEVICE)
        out = model.generate(input_ids, attention_mask=attn_mask, max_new_tokens=MAX_GEN,
                              do_sample=False, pad_token_id=tok.eos_token_id)
        for b in range(len(batch)):
            gen = tok.decode(out[b, input_ids.shape[1]:], skip_special_tokens=True)
            pa = extract_number(gen)
            ca = re.search(r"####\s*(-?\d+)", problems[start+b]["answer"])
            if pa and ca and pa == ca.group(1): base_correct[start+b] = 1
        print(f"  [{end}/{args.n_test}] acc={sum(base_correct)}/{end}", flush=True)
    baseline_acc = sum(base_correct) / len(base_correct)
    results = {"baseline": {"correct": sum(base_correct), "per_problem": base_correct, "accuracy": baseline_acc}}
    print(f"  Baseline: {sum(base_correct)}/{args.n_test} ({100*baseline_acc:.1f}%)", flush=True)

    for li in layers:
        print(f"\nLayer {li}/36...", flush=True)
        correct_list = [0] * args.n_test
        t0 = time.time()
        for start in range(0, args.n_test, args.bs):
            end = min(start + args.bs, args.n_test)
            batch = problems[start:end]
            B = len(batch)
            prompts = [f"Q: {p['question']}\nA:" for p in batch]
            encoded = tok(prompts, return_tensors="pt", padding=True)
            input_ids = encoded["input_ids"].to(DEVICE)
            attn_mask = encoded["attention_mask"].to(DEVICE)

            past, gen_lists, first, done = None, [[] for _ in range(B)], True, [False] * B
            cur_ids = input_ids
            cur_mask = attn_mask

            for _ in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(cur_ids, past_key_values=past, use_cache=True, output_hidden_states=True,
                                attention_mask=cur_mask)
                nts = fwd.logits[:, -1, :].argmax(dim=-1)
                for b in range(B):
                    if not done[b]:
                        tid = nts[b].item()
                        if tid == tok.eos_token_id:
                            done[b] = True
                        else:
                            gen_lists[b].append(tid)
                if all(done): break

                if not first:
                    hs = fwd.hidden_states
                    hp = torch.stack([h[:, -1, :].float() for h in hs[:N_LAYERS_3B]], dim=1)
                    x = hp.to(DEVICE)
                    with torch.no_grad(): v = tt(x)
                    h_act = hs[li + 1][:, -1, :]
                    h_st = h_act + ALPHA * v[:, li, :]
                    ly = model.model.layers[li]
                    k = ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B, n_kv_heads, 1, head_dim)
                    vo = ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B, n_kv_heads, 1, head_dim)
                    pk = fwd.past_key_values.layers[li]
                    pk.keys[:, :, -1:, :] = k.to(pk.keys.dtype)
                    pk.values[:, :, -1:, :] = vo.to(pk.values.dtype)

                past = fwd.past_key_values
                cur_ids = nts.unsqueeze(-1)
                # Extend mask: all batch items have a new real token
                cur_mask = torch.cat([cur_mask, torch.ones(B, 1, device=DEVICE, dtype=cur_mask.dtype)], dim=1)
                first = False

            for b in range(B):
                gen = tok.decode(gen_lists[b], skip_special_tokens=True)
                pa = extract_number(gen)
                ca = re.search(r"####\s*(-?\d+)", problems[start+b]["answer"])
                if pa and ca and pa == ca.group(1): correct_list[start+b] = 1

            print(f"  [{end}/{args.n_test}] acc={sum(correct_list)}/{end}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

        acc = sum(correct_list) / len(correct_list)
        results[f"L{li}"] = {"correct": sum(correct_list), "per_problem": correct_list, "accuracy": acc}
        print(f"  L{li}: {sum(correct_list)}/{args.n_test} ({100*acc:.1f}%) Δ={100*(acc-baseline_acc):+.1f}pp [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'='*60}")
    print(f"3B BATCHED SWEEP (α={ALPHA}, BS={args.bs}, 7B→3B TT)")
    print(f"{'='*60}")
    print(f"  base: {100*baseline_acc:.1f}%")
    best_l, best_d = None, -999
    for li in layers:
        r = results[f"L{li}"]; a = r["accuracy"]; d = a - baseline_acc
        if d > best_d: best_d, best_l = d, li
        m = " ◀ BEST" if d >= best_d else ""
        print(f"  L{li:3d} {100*a:7.1f}% {100*d:+7.1f}pp{m}")
    print(f"\n  Best: L{best_l} ({100*best_d:+.1f}pp)")

    with open("sweep_3b_batched.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to sweep_3b_batched.json")

if __name__ == "__main__":
    main()
