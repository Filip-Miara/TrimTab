#!/usr/bin/env python3 -u
"""Full 36-layer sweep on Qwen2.5-3B via 7B→3B TT transfer."""
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
N_LAYERS_7B = 28
N_LAYERS_3B = 36
D_INPUT_7B = 3584; D_INPUT_3B = 2048; D_MODEL = 768
ALPHA = 0.1; MAX_GEN = 400

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
        if key in tgt and src[key].shape == tgt[key].shape:
            tgt[key].copy_(src[key])
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
    args = parser.parse_args()

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    head_dim = 128; n_kv_heads = 2

    print("Transferring 7B TT→3B...", flush=True)
    tt = transfer_tt().to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad_(False)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    layers = args.layers if args.layers else list(range(N_LAYERS_3B))

    # Baseline
    print(f"\nBaseline ({args.n_test} problems)...", flush=True)
    base_correct = []
    for i, prob in enumerate(problems):
        msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
        iids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                        return_tensors="pt").input_ids.to(DEVICE)
        am = iids.ne(tok.pad_token_id).long()
        out = model.generate(iids, attention_mask=am, max_new_tokens=MAX_GEN,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, iids.shape[1]:], skip_special_tokens=True)
        pa = extract_number(gen)
        ca = re.search(r"####\s*(-?\d+)", prob["answer"])
        base_correct.append(1 if (pa and ca and pa == ca.group(1)) else 0)
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{args.n_test}] acc={sum(base_correct)}/{i+1}", flush=True)
    baseline_acc = sum(base_correct) / len(base_correct)
    results = {"baseline": {"correct": sum(base_correct), "total": len(base_correct),
                             "per_problem": base_correct, "accuracy": baseline_acc}}
    print(f"  Baseline: {sum(base_correct)}/{len(base_correct)} ({100*baseline_acc:.1f}%)", flush=True)

    for li in layers:
        print(f"\nLayer {li}/36...", flush=True)
        correct_list = []; t0 = time.time()
        for i, prob in enumerate(problems):
            msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
            iids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt").input_ids.to(DEVICE)
            past, gen_toks, first = None, [], True
            for _ in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(iids, past_key_values=past, use_cache=True, output_hidden_states=True)
                nt = fwd.logits[0, -1, :].argmax().item()
                if nt == tok.eos_token_id: break
                gen_toks.append(nt)
                if not first:
                    hs = fwd.hidden_states
                    hp = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS_3B]], dim=0)
                    x = hp.unsqueeze(0).to(DEVICE)
                    with torch.no_grad(): v = tt(x)
                    h_act = hs[li + 1][0, -1, :]
                    h_st = h_act + ALPHA * v[0, li, :]
                    ly = model.model.layers[li]
                    k = ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(1, n_kv_heads, 1, head_dim)
                    vo = ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(1, n_kv_heads, 1, head_dim)
                    pk = fwd.past_key_values.layers[li]
                    pk.keys[0, :, -1:, :] = k.to(pk.keys.dtype)
                    pk.values[0, :, -1:, :] = vo.to(pk.values.dtype)
                past = fwd.past_key_values
                iids = torch.tensor([[nt]], device=DEVICE)
                first = False
            gen = tok.decode(gen_toks, skip_special_tokens=True)
            pa = extract_number(gen)
            ca = re.search(r"####\s*(-?\d+)", prob["answer"])
            correct_list.append(1 if (pa and ca and pa == ca.group(1)) else 0)
            if (i+1) % 10 == 0:
                print(f"  [{i+1}/{args.n_test}] acc={sum(correct_list)}/{i+1}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        acc = sum(correct_list) / len(correct_list)
        results[f"L{li}"] = {"correct": sum(correct_list), "total": len(correct_list),
                              "per_problem": correct_list, "accuracy": acc}
        print(f"  L{li}: {sum(correct_list)}/{len(correct_list)} ({100*acc:.1f}%) Δ={100*(acc-baseline_acc):+.1f}pp [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'='*60}")
    print(f"3B FULL SWEEP (α={ALPHA}, 7B→3B TT)")
    print(f"{'='*60}")
    print(f"  base: {100*baseline_acc:.1f}%")
    best_l, best_d = None, -999
    for li in layers:
        r = results[f"L{li}"]; acc = r["accuracy"]; d = acc - baseline_acc
        if d > best_d: best_d, best_l = d, li
        m = " ◀ BEST" if d >= best_d else ""
        print(f"  L{li:3d} {100*acc:7.1f}% {100*d:+7.1f}pp{m}")
    print(f"\n  Best: L{best_l} ({100*best_d:+.1f}pp)")

    with open("sweep_3b_full.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to sweep_3b_full.json")

if __name__ == "__main__":
    main()
