#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Sweep α values for L8 and L10 with the new TT on AWQ."""
import gc, json, re, sys, time, warnings, os
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
MAX_GEN = 200; BS = 8
ALPHAS = [0.05, 0.1, 0.2, 0.5, 1.0]
TARGET_LAYERS = [8, 10]

def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None

print("Loading AWQ 7B...", flush=True)
t0 = time.time()
quant = AwqConfig(bits=4, group_size=128, zero_point=True, backend="gemm")
model = AutoModelForCausalLM.from_pretrained(AWQ_PATH, quantization_config=quant,
                                              device_map='cuda', trust_remote_code=True, torch_dtype=torch.float16)
model.eval()
tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
tok.pad_token_id = tok.eos_token_id; tok.padding_side = "left"
print(f"Loaded in {time.time()-t0:.0f}s", flush=True)

print(f"Loading TT (d_model=1536)...", flush=True)
tt = TrajectoryTransformer(d_model=1536, n_layers=6, n_heads=8, d_ff=1536*4,
                            n_positions=N_LAYERS, d_input=D_MODEL).to(DEVICE)
sd = torch.load(TT_PATH, map_location="cpu")
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
tt.load_state_dict(sd, strict=False)
tt.eval()

ds = load_dataset("openai/gsm8k", "main", split="test")
problems = [r for r in ds if len(r["question"]) > 50][:30]

# Baseline
print(f"\nBaseline (30 problems)...", flush=True)
base_correct = [0] * 30
for start in range(0, 30, BS):
    end = min(start + BS, 30)
    prompts = [f"Q: {p['question']}\nA:" for p in problems[start:end]]
    enc = tok(prompts, return_tensors="pt", padding=True)
    out = model.generate(enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE),
                          max_new_tokens=MAX_GEN, do_sample=False, pad_token_id=tok.eos_token_id)
    for b in range(end - start):
        gen = tok.decode(out[b, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pa = extract_number(gen); ca = re.search(r"####\s*(-?\d+)", problems[start+b]["answer"])
        if pa and ca and pa == ca.group(1): base_correct[start+b] = 1
    print(f"  [{end}/30] acc={sum(base_correct)}/{end}", flush=True)
base_acc = sum(base_correct) / 30
print(f"Baseline: {sum(base_correct)}/30 ({100*base_acc:.1f}%)", flush=True)

results = {"baseline": base_acc, "alpha_sweep": {}}

for li in TARGET_LAYERS:
    for alpha in ALPHAS:
        print(f"\nLayer {li}, α={alpha}...", flush=True)
        correct = [0] * 30; t0 = time.time()
        for start in range(0, 30, BS):
            end = min(start + BS, 30)
            B = end - start; prompts = [f"Q: {p['question']}\nA:" for p in problems[start:end]]
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
                    h_st = hs[li + 1][:, -1, :] + alpha * v[:, li, :]
                    ly = model.model.layers[li]
                    k = ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    vo = ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    if hasattr(fwd.past_key_values, "key_cache") and len(fwd.past_key_values.key_cache) > 0:
                        fwd.past_key_values.key_cache[li][:, :, -1:, :] = k.to(fwd.past_key_values.key_cache[li].dtype)
                        fwd.past_key_values.value_cache[li][:, :, -1:, :] = vo.to(fwd.past_key_values.value_cache[li].dtype)
                past = fwd.past_key_values
                cur_ids = nts.unsqueeze(-1)
                cur_mask = torch.cat([cur_mask, torch.ones(B, 1, device=DEVICE, dtype=cur_mask.dtype)], dim=1)
                first = False
            for b in range(B):
                gen = tok.decode(gen_lists[b], skip_special_tokens=True)
                pa = extract_number(gen); ca = re.search(r"####\s*(-?\d+)", problems[start+b]["answer"])
                if pa and ca and pa == ca.group(1): correct[start+b] = 1
            print(f"  [{end}/30] acc={sum(correct)}/{end}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        acc = sum(correct) / 30
        key = f"L{li}_α{alpha}"
        results["alpha_sweep"][key] = acc
        print(f"  L{li} α={alpha}: {sum(correct)}/30 ({100*acc:.1f}%) Δ={100*(acc-base_acc):+.1f}pp [{time.time()-t0:.0f}s]", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"ALPHA SWEEP RESULTS (TT R²=0.900)", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Baseline: {100*base_acc:.1f}%", flush=True)
print(f"  {'Layer':>6} {'α':>6} {'Acc':>7} {'Δ':>7}", flush=True)
print(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*7}", flush=True)
for li in TARGET_LAYERS:
    for alpha in ALPHAS:
        a = results["alpha_sweep"][f"L{li}_α{alpha}"]; d = a - base_acc
        print(f"  L{li:3d}  {alpha:4.2f}  {100*a:5.1f}%  {100*d:+5.1f}pp", flush=True)

json.dump(results, open("alpha_sweep_results.json", "w"), indent=2)
print(f"\nSaved to alpha_sweep_results.json", flush=True)
