#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Extreme α sweep on L10: huge, tiny, and negative values."""
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
MAX_GEN = 200; BS = 8; LI = 10
ALPHAS = [2.0, 3.0, 0.01, 0.001, -0.05, -0.1, -0.2, -0.5, -1.0]

def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"####\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None

print("Loading AWQ 7B...", flush=True)
quant = AwqConfig(bits=4, group_size=128, zero_point=True, backend="gemm")
model = AutoModelForCausalLM.from_pretrained(AWQ_PATH, quantization_config=quant,
                                              device_map='cuda', trust_remote_code=True, torch_dtype=torch.float16).eval()
tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
tok.pad_token_id = tok.eos_token_id; tok.padding_side = "left"

print("Loading TT...", flush=True)
tt = TrajectoryTransformer(d_model=1536, n_layers=6, n_heads=8, d_ff=1536*4,
                            n_positions=N_LAYERS, d_input=D_MODEL).to(DEVICE)
sd = torch.load(TT_PATH, map_location="cpu")
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
tt.load_state_dict(sd, strict=False)
tt.eval()

ds = load_dataset("openai/gsm8k", "main", split="test")
problems = [r for r in ds if len(r["question"]) > 50][:30]

# Baseline
print("Baseline...", flush=True)
base_c = [0]*30
for s in range(0, 30, BS):
    e = min(s+BS, 30)
    prompts = [f"Q: {p['question']}\nA:" for p in problems[s:e]]
    enc = tok(prompts, return_tensors="pt", padding=True)
    out = model.generate(enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE),
                          max_new_tokens=MAX_GEN, do_sample=False, pad_token_id=tok.eos_token_id)
    for b in range(e-s):
        g = tok.decode(out[b, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pa = extract_number(g); ca = re.search(r"####\s*(-?\d+)", problems[s+b]["answer"])
        if pa and ca and pa == ca.group(1): base_c[s+b] = 1
    print(f"  [{e}/30] acc={sum(base_c)}/{e}", flush=True)
base_acc = sum(base_c)/30
print(f"Baseline: {sum(base_c)}/30 ({100*base_acc:.1f}%)", flush=True)

results = {"baseline": base_acc, "L10": {}}
for alpha in ALPHAS:
    print(f"\nL10 α={alpha}...", flush=True)
    c = [0]*30; t0 = time.time()
    for s in range(0, 30, BS):
        e = min(s+BS, 30); B = e-s
        prompts = [f"Q: {p['question']}\nA:" for p in problems[s:e]]
        enc = tok(prompts, return_tensors="pt", padding=True)
        iids = enc["input_ids"].to(DEVICE); am = enc["attention_mask"].to(DEVICE)
        past, gl, first, done = None, [[] for _ in range(B)], True, [False]*B
        cur_ids, cur_mask = iids, am
        for _ in range(MAX_GEN):
            with torch.no_grad():
                fwd = model(cur_ids, past_key_values=past, use_cache=True, output_hidden_states=True, attention_mask=cur_mask)
            nts = fwd.logits[:, -1, :].argmax(dim=-1)
            for b in range(B):
                if not done[b]:
                    tid = nts[b].item()
                    if tid == tok.eos_token_id: done[b] = True
                    else: gl[b].append(tid)
            if all(done): break
            if not first:
                hs = fwd.hidden_states
                hp = torch.stack([h[:, -1, :].float() for h in hs[:N_LAYERS]], dim=1)
                with torch.no_grad(): v = tt(hp)
                h_st = hs[LI+1][:, -1, :] + alpha * v[:, LI, :]
                ly = model.model.layers[LI]
                k = ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                vo = ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                if hasattr(fwd.past_key_values, "key_cache") and len(fwd.past_key_values.key_cache) > 0:
                    fwd.past_key_values.key_cache[LI][:, :, -1:, :] = k.to(fwd.past_key_values.key_cache[LI].dtype)
                    fwd.past_key_values.value_cache[LI][:, :, -1:, :] = vo.to(fwd.past_key_values.value_cache[LI].dtype)
            past = fwd.past_key_values
            cur_ids = nts.unsqueeze(-1)
            cur_mask = torch.cat([cur_mask, torch.ones(B, 1, device=DEVICE, dtype=cur_mask.dtype)], dim=1)
            first = False
        for b in range(B):
            g = tok.decode(gl[b], skip_special_tokens=True)
            pa = extract_number(g); ca = re.search(r"####\s*(-?\d+)", problems[s+b]["answer"])
            if pa and ca and pa == ca.group(1): c[s+b] = 1
        print(f"  [{e}/30] acc={sum(c)}/{e}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    acc = sum(c)/30; d = acc - base_acc
    results["L10"][alpha] = acc
    print(f"  L10 α={alpha}: {sum(c)}/30 ({100*acc:.1f}%) Δ={100*d:+.1f}pp [{time.time()-t0:.0f}s]", flush=True)

print(f"\nBaseline: {100*base_acc:.1f}%")
for a in ALPHAS:
    d = results["L10"][a] - base_acc
    print(f"  α={a:5.3f}: {100*results['L10'][a]:5.1f}% ({100*d:+5.1f}pp)")

json.dump(results, open("alpha_extreme_results.json", "w"), indent=2)
print(f"\nSaved to alpha_extreme_results.json")
