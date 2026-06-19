#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Retroactive steering: compute velocity at later layer, inject at earlier layer, re-process."""
import torch, sys, time, gc, warnings, os, json, re
os.environ["PATH"] = f"{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}"
warnings.filterwarnings('ignore')
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
sys.path.insert(0, '.')
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"; AWQ_PATH = "/run/media/filip/C27C20AB7C209C63/Qwen2.5-7B-AWQ/qwen7b_awq"
TOK_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
N_LAYERS=28; N_KV_HEADS=4; HEAD_DIM=128; BS=8; MAX_GEN=200; ALPHA=0.1

def extr(t):
    for p in [r'answer\s+is\s*(-?\d+)', r'The answer is\s*(-?\d+)', r'####\s*(-?\d+)']:
        m = re.search(p, t, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r'-?\d+', t); return nums[-1] if nums else None

quant = AwqConfig(bits=4, group_size=128, zero_point=True, backend='gemm')
model = AutoModelForCausalLM.from_pretrained(AWQ_PATH, quantization_config=quant, device_map='cuda', trust_remote_code=True, torch_dtype=torch.float16).eval()
tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
tok.pad_token_id = tok.eos_token_id; tok.padding_side = 'left'
tt = TrajectoryTransformer(d_model=1536, n_layers=6, n_heads=8, d_ff=1536*4, n_positions=28, d_input=3584).to(DEVICE)
tt.load_state_dict({k.replace('_orig_mod.',''):v for k,v in torch.load('best_tt_d1536_cos.pt', map_location='cpu').items()}, strict=False); tt.eval()
ds = load_dataset('openai/gsm8k', 'main', split='test')
problems = [r for r in ds if len(r['question']) > 50][:30]

def run_test(name):
    correct = [0]*30; t0 = time.time()
    for s in range(0, 30, BS):
        e = min(s+BS, 30); B = e-s
        prompts = [f"Q: {p['question']}\nA:" for p in problems[s:e]]
        enc = tok(prompts, return_tensors='pt', padding=True)
        iids = enc['input_ids'].to(DEVICE); am = enc['attention_mask'].to(DEVICE)
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
                
                if name == "standard_L10":
                    # Standard: steer L10 with L10 velocity
                    steer_li, vel_li = 10, 10
                    h_st = hs[steer_li+1][:, -1, :] + ALPHA * v[:, vel_li, :]
                    k = model.model.layers[steer_li].self_attn.k_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    vo = model.model.layers[steer_li].self_attn.v_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    fwd.past_key_values.layers[steer_li].keys[:,:,-1:,:] = k.to(fwd.past_key_values.layers[steer_li].keys.dtype)
                    fwd.past_key_values.layers[steer_li].values[:,:,-1:,:] = vo.to(fwd.past_key_values.layers[steer_li].values.dtype)
                
                elif name == "retro_L10_at_L9":
                    # Idea 1: compute v at L10, inject at L9, re-process
                    steer_li, vel_li = 9, 10
                    # Cache original L9 hidden state (before modification)
                    cached_h9 = hs[10][:, -1, :].clone()  # output of L9
                    # Compute steering from L10's velocity, apply to L9's cached state
                    h_st = cached_h9 + ALPHA * v[:, vel_li, :]
                    k = model.model.layers[steer_li].self_attn.k_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    vo = model.model.layers[steer_li].self_attn.v_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    fwd.past_key_values.layers[steer_li].keys[:,:,-1:,:] = k.to(fwd.past_key_values.layers[steer_li].keys.dtype)
                    fwd.past_key_values.layers[steer_li].values[:,:,-1:,:] = vo.to(fwd.past_key_values.layers[steer_li].values.dtype)
                    # Re-process: run forward pass again with modified L9 KV
                    with torch.no_grad():
                        fwd = model(cur_ids, past_key_values=fwd.past_key_values, use_cache=True, output_hidden_states=True, attention_mask=cur_mask)
                
                elif name == "retro_L9_at_L9":
                    # Idea 2: compute v at L9 (with full context through L10), inject at L9, re-process
                    steer_li, vel_li = 9, 9
                    cached_h9 = hs[10][:, -1, :].clone()
                    h_st = cached_h9 + ALPHA * v[:, vel_li, :]
                    k = model.model.layers[steer_li].self_attn.k_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    vo = model.model.layers[steer_li].self_attn.v_proj(h_st.to(torch.bfloat16)).view(B, N_KV_HEADS, 1, HEAD_DIM)
                    fwd.past_key_values.layers[steer_li].keys[:,:,-1:,:] = k.to(fwd.past_key_values.layers[steer_li].keys.dtype)
                    fwd.past_key_values.layers[steer_li].values[:,:,-1:,:] = vo.to(fwd.past_key_values.layers[steer_li].values.dtype)
                    with torch.no_grad():
                        fwd = model(cur_ids, past_key_values=fwd.past_key_values, use_cache=True, output_hidden_states=True, attention_mask=cur_mask)
                        
            past = fwd.past_key_values
            cur_ids = nts.unsqueeze(-1)
            cur_mask = torch.cat([cur_mask, torch.ones(B, 1, device=DEVICE, dtype=cur_mask.dtype)], dim=1)
            first = False
        for b in range(B):
            g = tok.decode(gl[b], skip_special_tokens=True)
            pa = extr(g); ca = re.search(r'####\s*(-?\d+)', problems[s+b]['answer'])
            if pa and ca and pa == ca.group(1): correct[s+b] = 1
        print(f"  [{e}/30] acc={sum(correct)}/{e}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    return sum(correct)/30

# Baseline
print("Baseline...", flush=True)
base_c = [0]*30
for s in range(0, 30, BS):
    e = min(s+BS, 30)
    prompts = [f"Q: {p['question']}\nA:" for p in problems[s:e]]
    enc = tok(prompts, return_tensors='pt', padding=True)
    out = model.generate(enc['input_ids'].to(DEVICE), attention_mask=enc['attention_mask'].to(DEVICE), max_new_tokens=MAX_GEN, do_sample=False, pad_token_id=tok.eos_token_id)
    for b in range(e-s):
        g = tok.decode(out[b, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        pa = extr(g); ca = re.search(r'####\s*(-?\d+)', problems[s+b]['answer'])
        if pa and ca and pa == ca.group(1): base_c[s+b] = 1
    print(f"  [{e}/30] acc={sum(base_c)}/{e}", flush=True)
base_acc = sum(base_c)/30; print(f"Baseline: {sum(base_c)}/30 ({100*base_acc:.1f}%)", flush=True)

results = {"baseline": base_acc}
for name in ["standard_L10", "retro_L10_at_L9", "retro_L9_at_L9"]:
    print(f"\n=== {name} ===", flush=True)
    acc = run_test(name); d = acc - base_acc
    results[name] = acc
    print(f"{name}: {100*acc:.1f}% Δ={100*d:+.1f}pp", flush=True)

print(f"\n{'='*60}")
print("RETROACTIVE STEERING")
print(f"{'='*60}")
print(f"  Baseline:                  {100*base_acc:.1f}%")
for n in ["standard_L10", "retro_L10_at_L9", "retro_L9_at_L9"]:
    d = results[n] - base_acc
    print(f"  {n:<26} {100*results[n]:5.1f}% ({100*d:+5.1f}pp)")

json.dump(results, open("retroactive_results.json", "w"), indent=2)
print(f"\nSaved")
