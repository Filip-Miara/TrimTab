#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Evaluate d_model=1536 TT on AWQ model — trim-tab layers.
Forces GEMM kernel to avoid Marlin JIT compilation issues."""
import gc, json, os, re, sys, time, warnings

# Set env before any imports
os.environ["PATH"] = f"{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}"

warnings.filterwarnings('ignore')
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

sys.path.insert(0, '.')
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
AWQ_PATH = "/run/media/filip/C27C20AB7C209C63/Qwen2.5-7B-AWQ/qwen7b_awq"
TOK_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TT_PATH = "best_tt_d1536_cos.pt"
N_LAYERS = 28; D_MODEL = 3584; N_KV_HEADS = 4; HEAD_DIM = 128
ALPHA = 0.1; MAX_GEN = 200; BS = 8
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

    print("Loading AWQ 7B (GEMM backend)...", flush=True)
    t0 = time.time()
    quant = AwqConfig(bits=4, group_size=128, zero_point=True, backend="gemm")
    model = AutoModelForCausalLM.from_pretrained(AWQ_PATH, quantization_config=quant,
                                                  device_map='cuda', trust_remote_code=True,
                                                  torch_dtype=torch.float16)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id; tok.padding_side = "left"
    print(f"Loaded in {time.time()-t0:.0f}s, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    print(f"Loading TT (d_model=1536)...", flush=True)
    tt = TrajectoryTransformer(d_model=1536, n_layers=6, n_heads=8, d_ff=1536*4,
                                n_positions=N_LAYERS, d_input=D_MODEL).to(DEVICE)
    sd = torch.load(TT_PATH, map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    tt.load_state_dict(sd, strict=False)
    tt.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    results = {}

    # Batched baseline
    print(f"\nBaseline ({args.n_test} problems, BS={BS})...", flush=True)
    base_correct = [0] * args.n_test
    for start in range(0, args.n_test, BS):
        end = min(start + BS, args.n_test)
        prompts = [f"Q: {p['question']}\nA:" for p in problems[start:end]]
        enc = tok(prompts, return_tensors="pt", padding=True)
        iids = enc["input_ids"].to(DEVICE)
        am = enc["attention_mask"].to(DEVICE) if "attention_mask" in enc else None
        out = model.generate(iids, attention_mask=am, max_new_tokens=MAX_GEN,
                              do_sample=False, pad_token_id=tok.eos_token_id)
        for b in range(end - start):
            gen = tok.decode(out[b, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pa = extract_number(gen); ca = re.search(r"####\s*(-?\d+)", problems[start+b]["answer"])
            if pa and ca and pa == ca.group(1): base_correct[start+b] = 1
        print(f"  [{end}/{args.n_test}] acc={sum(base_correct)}/{end} ({100*sum(base_correct)/end:.0f}%)", flush=True)
    base_acc = sum(base_correct) / args.n_test
    results["baseline"] = base_acc
    print(f"  Baseline: {sum(base_correct)}/{args.n_test} ({100*base_acc:.1f}%)", flush=True)

    # Steered layers
    for li in TARGET_LAYERS:
        print(f"\nLayer {li}, α={args.alpha}...", flush=True)
        correct = [0] * args.n_test; t0 = time.time()
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
                    pkv = fwd.past_key_values
                    if hasattr(pkv, "key_cache") and len(pkv.key_cache) > 0:
                        pkv.key_cache[li][:, :, -1:, :] = k.to(pkv.key_cache[li].dtype)
                        pkv.value_cache[li][:, :, -1:, :] = vo.to(pkv.value_cache[li].dtype)
                    else:
                        lc = pkv if isinstance(pkv, list) else getattr(pkv, "layers", [None])[li]
                        if lc is not None and hasattr(lc, "keys"):
                            lc.keys[:, :, -1:, :] = k.to(lc.keys.dtype)
                            lc.values[:, :, -1:, :] = vo.to(lc.values.dtype)
                past = fwd.past_key_values
                cur_ids = nts.unsqueeze(-1)
                cur_mask = torch.cat([cur_mask, torch.ones(B, 1, device=DEVICE, dtype=cur_mask.dtype)], dim=1)
                first = False
            for b in range(B):
                gen = tok.decode(gen_lists[b], skip_special_tokens=True)
                pa = extract_number(gen); ca = re.search(r"####\s*(-?\d+)", problems[start+b]["answer"])
                if pa and ca and pa == ca.group(1): correct[start+b] = 1
            print(f"  [{end}/{args.n_test}] acc={sum(correct)}/{end} ({100*sum(correct)/end:.0f}%)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        acc = sum(correct) / args.n_test
        results[f"L{li}"] = acc
        print(f"  L{li}: {sum(correct)}/{args.n_test} ({100*acc:.1f}%) Δ={100*(acc-base_acc):+.1f}pp [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"AWQ TRIM-TAB RESULTS (α={args.alpha}, {args.n_test} problems, TT R²=0.900)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Layer':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    print(f"  {'base':>6} {100*base_acc:7.1f}%")
    for li in TARGET_LAYERS:
        a = results[f"L{li}"]; d = a - base_acc
        m = " \U0001f4aa" if d > 0 else (" \u2620" if d < 0 else "")
        print(f"  L{li:3d} {100*a:7.1f}% {100*d:+7.1f}pp{m}", flush=True)
    json.dump({"baseline": base_acc, "results": results, "alpha": args.alpha, "n_test": args.n_test,
               "tt_r2": 0.900, "model": "d1536_cosine", "quant": "awq"},
              open("trimtab_awq_results.json", "w"), indent=2)
    print(f"\nSaved to trimtab_awq_results.json", flush=True)

if __name__ == "__main__":
    main()
