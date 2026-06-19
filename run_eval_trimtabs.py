#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Evaluate the d_model=1536 + MSE+cosine TT on trim-tab layers."""
import gc, json, re, sys, time, warnings
warnings.filterwarnings('ignore')
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, '.')
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/C27C20AB7C209C63/models/Qwen2.5-7B-Instruct"
TT_PATH = "best_tt_d1536_cos.pt"
N_LAYERS = 28; D_MODEL = 3584; N_KV_HEADS = 4; HEAD_DIM = 128
ALPHA = 0.1; MAX_GEN = 200; BS = 8
# Trim-tab layers of interest
TARGET_LAYERS = [2, 5, 8, 9, 10]

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
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    print("Loading BnB 4-bit model...", flush=True)
    t0 = time.time()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=quant,
                                                  device_map=DEVICE, trust_remote_code=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id; tok.padding_side = "left"
    print(f"Model loaded in {time.time()-t0:.0f}s", flush=True)

    # Load TT with d_model=1536 architecture
    print(f"Loading TT from {TT_PATH}...", flush=True)
    tt = TrajectoryTransformer(d_model=1536, n_layers=6, n_heads=8, d_ff=1536*4,
                                n_positions=N_LAYERS, d_input=D_MODEL).to(DEVICE)
    sd = torch.load(TT_PATH, map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    tt.load_state_dict(sd, strict=False)
    tt.eval()
    for p in tt.parameters(): p.requires_grad = False
    print(f"  {sum(p.numel() for p in tt.parameters()):,} params", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    results = {}

    # Baseline
    print(f"\nBaseline ({args.n_test} problems)...", flush=True)
    base_correct = [0] * args.n_test
    for start in range(0, args.n_test, BS):
        end = min(start + BS, args.n_test)
        prompts = [f"Q: {p['question']}\nA:" for p in problems[start:end]]
        enc = tok(prompts, return_tensors="pt", padding=True)
        out = model.generate(enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE),
                              max_new_tokens=MAX_GEN, do_sample=False, pad_token_id=tok.eos_token_id)
        for b in range(len(prompts)):
            gen = tok.decode(out[b, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pa = extract_number(gen)
            ca = re.search(r"####\s*(-?\d+)", problems[start+b]["answer"])
            if pa and ca and pa == ca.group(1): base_correct[start+b] = 1
        print(f"  [{end}/{args.n_test}] acc={sum(base_correct)}/{end}", flush=True)
    base_acc = sum(base_correct) / args.n_test
    results["baseline"] = base_acc
    print(f"  Baseline: {sum(base_correct)}/{args.n_test} ({100*base_acc:.1f}%)", flush=True)

    # Steered layers
    for li in TARGET_LAYERS:
        print(f"\nLayer {li}, α={args.alpha}...", flush=True)
        correct = [0] * args.n_test
        t0 = time.time()
        for start in range(0, args.n_test, BS):
            end = min(start + BS, args.n_test)
            B = end - start
            prompts = [f"Q: {p['question']}\nA:" for p in problems[start:end]]
            enc = tok(prompts, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(DEVICE); attn_mask = enc["attention_mask"].to(DEVICE)
            past, gen_lists, first, done = None, [[] for _ in range(B)], True, [False] * B
            cur_ids, cur_mask = input_ids, attn_mask
            for _ in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(cur_ids, past_key_values=past, use_cache=True, output_hidden_states=True, attention_mask=cur_mask)
                nts = fwd.logits[:, -1, :].argmax(dim=-1)
                for b in range(B):
                    if not done[b]:
                        tid = nts[b].item()
                        if tid == tok.eos_token_id: done[b] = True
                        else: gen_lists[b].append(tid)
                if all(done): break
                if not first:
                    hs = fwd.hidden_states
                    hp = torch.stack([h[:, -1, :].float() for h in hs[:N_LAYERS]], dim=1)
                    with torch.no_grad(): v = tt(hp)
                    h_st = hs[li + 1][:, -1, :] + args.alpha * v[:, li, :]
                    ly = model.model.layers[li]
                    k = ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    vo = ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    if hasattr(fwd.past_key_values, 'key_cache'):
                        fwd.past_key_values.key_cache[li][:, :, -1:, :] = k.to(fwd.past_key_values.key_cache[li].dtype)
                        fwd.past_key_values.value_cache[li][:, :, -1:, :] = vo.to(fwd.past_key_values.value_cache[li].dtype)
                    else:
                        lc = fwd.past_key_values.layers[li]
                        lc.keys[:, :, -1:, :] = k.to(lc.keys.dtype)
                        lc.values[:, :, -1:, :] = vo.to(lc.values.dtype)
                past = fwd.past_key_values
                cur_ids = nts.unsqueeze(-1)
                cur_mask = torch.cat([cur_mask, torch.ones(B, 1, device=DEVICE, dtype=cur_mask.dtype)], dim=1)
                first = False
            for b in range(B):
                gen = tok.decode(gen_lists[b], skip_special_tokens=True)
                pa = extract_number(gen)
                ca = re.search(r"####\s*(-?\d+)", problems[start+b]["answer"])
                if pa and ca and pa == ca.group(1): correct[start+b] = 1
            print(f"  [{end}/{args.n_test}] acc={sum(correct)}/{end}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        acc = sum(correct) / args.n_test
        results[f"L{li}"] = acc
        delta = acc - base_acc
        print(f"  L{li}: {sum(correct)}/{args.n_test} ({100*acc:.1f}%) Δ={100*delta:+.1f}pp [{time.time()-t0:.0f}s]", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"TRIM-TAB RESULTS (α={args.alpha}, {args.n_test} problems, TT R²=0.900)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Layer':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    print(f"  {'base':>6} {100*base_acc:7.1f}%")
    for li in TARGET_LAYERS:
        a = results[f"L{li}"]; d = a - base_acc
        m = " ← TRIM" if d > 0 else (" ← DEATH" if d < 0 else "")
        print(f"  L{li:3d} {100*a:7.1f}% {100*d:+7.1f}pp{m}", flush=True)
    best_l = max(TARGET_LAYERS, key=lambda l: results[f"L{l}"])
    print(f"\n  Best: L{best_l} ({100*(results[f'L{best_l}']-base_acc):+.1f}pp)", flush=True)

    json.dump({"baseline": base_acc, "results": results, "alpha": args.alpha, "n_test": args.n_test,
               "tt_r2": 0.900, "model": "d1536_cosine"},
              open("trimtab_eval_results.json", "w"), indent=2)
    print(f"\nSaved to trimtab_eval_results.json", flush=True)

if __name__ == "__main__":
    main()
