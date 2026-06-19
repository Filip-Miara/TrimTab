#!/mnt/windows/awq_env2/bin/python -u
"""Batched trim-tab sweep on Qwen2.5-7B via AWQ-Marlin + new TT (R²=0.8482).
Processes BS problems in parallel for ~BS× speedup.
"""
import gc, json, re, sys, time, warnings
warnings.filterwarnings('ignore')
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '.')
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_PATH = "/mnt/windows/7B-AWQ/qwen7b_awq"
TOK_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TT_PATH = "best_gen_tt_7b_new.pt"
N_LAYERS = 28
D_MODEL = 3584
D_MODEL_TT = 768
N_KV_HEADS = 4
HEAD_DIM = 128
ALPHA = 0.1
MAX_GEN = 200
BS = 16
TARGET_LAYERS = [2, 3, 5, 8, 10, 20] + list(range(11, 20))


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
    parser.add_argument("--bs", type=int, default=BS)
    args = parser.parse_args()

    print("Loading AWQ 7B...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map='cuda', trust_remote_code=True, torch_dtype=torch.float16
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id; tok.padding_side = "left"
    print(f"Model loaded in {time.time()-t0:.0f}s, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    print(f"Loading TT from {TT_PATH}...", flush=True)
    tt = TrajectoryTransformer(d_model=D_MODEL_TT, n_layers=6, n_heads=8, d_ff=D_MODEL_TT*4,
                                n_positions=N_LAYERS, d_input=D_MODEL).to(DEVICE)
    tt.load_state_dict(torch.load(TT_PATH, map_location="cpu"), strict=False)
    tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

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
        print(f"  baseline [{end}/{args.n_test}] acc={sum(base_correct)}/{end} ({100*sum(base_correct)/end:.0f}%)", flush=True)
    baseline_acc = sum(base_correct) / len(base_correct)
    results = {"baseline": {"correct": sum(base_correct), "accuracy": baseline_acc}}
    print(f"  Baseline: {sum(base_correct)}/{args.n_test} ({100*baseline_acc:.1f}%)", flush=True)

    # Batched per-layer steering
    for li in TARGET_LAYERS:
        print(f"\n{'='*60}", flush=True)
        print(f"Layer {li}, α={args.alpha}", flush=True)
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
                    hp = torch.stack([h[:, -1, :].float() for h in hs[:N_LAYERS]], dim=1)
                    x = hp.to(DEVICE)
                    with torch.no_grad(): v = tt(x)
                    h_act = hs[li + 1][:, -1, :]
                    h_st = h_act + args.alpha * v[:, li, :]
                    ly = model.model.layers[li]
                    k = ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    vo = ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    kc = fwd.past_key_values.key_cache[li]
                    vc = fwd.past_key_values.value_cache[li]
                    kc[:, :, -1:, :] = k.to(kc.dtype)
                    vc[:, :, -1:, :] = vo.to(vc.dtype)

                past = fwd.past_key_values
                cur_ids = nts.unsqueeze(-1)
                cur_mask = torch.cat([cur_mask, torch.ones(B, 1, device=DEVICE, dtype=cur_mask.dtype)], dim=1)
                first = False

            for b in range(B):
                gen = tok.decode(gen_lists[b], skip_special_tokens=True)
                pa = extract_number(gen)
                ca = re.search(r"####\s*(-?\d+)", problems[start+b]["answer"])
                if pa and ca and pa == ca.group(1): correct_list[start+b] = 1

            print(f"  L{li} [{end}/{args.n_test}] acc={sum(correct_list)}/{end} ({100*sum(correct_list)/end:.0f}%)", flush=True)
            gc.collect(); torch.cuda.empty_cache()

        acc = sum(correct_list) / len(correct_list)
        results[f"L{li}"] = {"correct": sum(correct_list), "accuracy": acc}
        print(f"  L{li}: {sum(correct_list)}/{args.n_test} ({100*acc:.1f}%) Δ={100*(acc-baseline_acc):+.1f}pp [{time.time()-t0:.0f}s]", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"TRIM-TAB SWEEP (α={args.alpha}, BS={args.bs}, {args.n_test} problems, TT R²=0.8482)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Layer':>6} {'Acc':>8s} {'Δ':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    print(f"  {'base':>6} {100*baseline_acc:7.1f}%")
    best_l, best_d, worst_l, worst_d = None, -999, None, 999
    for li in TARGET_LAYERS:
        r = results[f"L{li}"]; a = r["accuracy"]; d = a - baseline_acc
        if d > best_d: best_d, best_l = d, li
        if d < worst_d: worst_d, worst_l = d, li
        marker = " ← TRIM" if d > 0 else (" ← DEATH" if d < 0 else "")
        print(f"  L{li:3d} {100*a:7.1f}% {100*d:+7.1f}pp{marker}", flush=True)
    print(f"\n  Best trim-tab: L{best_l} ({100*best_d:+.1f}pp)", flush=True)
    print(f"  Worst death: L{worst_l} ({100*worst_d:+.1f}pp)", flush=True)

    json.dump({"baseline": baseline_acc, "results": results, "alpha": args.alpha,
               "n_test": args.n_test, "bs": args.bs, "tt_r2": 0.8482},
              open("trimtab_sweep_results.json", "w"), indent=2)
    print(f"\nSaved to trimtab_sweep_results.json", flush=True)


if __name__ == "__main__":
    main()
