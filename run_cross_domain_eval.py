#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Cross-domain generalization: evaluate TT steering on diverse local datasets."""
import torch, sys, time, gc, warnings, os, json, re, glob
os.environ["PATH"] = f"{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}"
warnings.filterwarnings('ignore')
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
sys.path.insert(0, '.')
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"; AWQ_PATH = "/run/media/filip/C27C20AB7C209C63/Qwen2.5-7B-AWQ/qwen7b_awq"
TOK_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
N_LAYERS=28; N_KV_HEADS=4; HEAD_DIM=128; BS=8; MAX_GEN=200; ALPHA=0.1

DATA_BASE = "/run/media/filip/B522-875D/Datasets/datasets"

# Diverse datasets across domains
DATASETS = {
    "gsm8k": {"dir": "gsm8k/main/0.0.0/*/gsm8k-test.arrow", "question": "question", "answer": "answer", "extract": "gsm8k"},
    "science_qa": {"dir": "derek-thomas___science_qa/default/0.0.0/*/science_qa-test.arrow", "question": "question", "answer": "answer", "extract": "lastnum"},
    "bbh_date": {"dir": "lukaemon___bbh/date_understanding/0.0.0/*/bbh-test.arrow", "question": "input", "answer": "target", "extract": "lastnum"},
    "bbh_geo": {"dir": "lukaemon___bbh/geometric_shapes/0.0.0/*/bbh-test.arrow", "question": "input", "answer": "target", "extract": "lastnum"},
}

def extract_answer(text, mode):
    if mode == "gsm8k":
        m = re.search(r"####\s*(-?\d+)", text)
        return m.group(1) if m else None
    elif mode == "numina":
        nums = re.findall(r"-?\d+", text)
        return nums[-1] if nums else None
    elif mode == "lastnum":
        nums = re.findall(r"-?\d+", text)
        return nums[-1] if nums else None

def extract(text):
    for p in [r"answer\s+is\s*(-?\d+)",r"The answer is\s*(-?\d+)",r"####\s*(-?\d+)",r"The answer is (\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None

def load_ds(key, cfg):
    path = os.path.join(DATA_BASE, cfg["dir"])
    files = sorted(glob.glob(path))
    if not files:
        print(f"  No data files for {key}")
        return None
    ds = load_dataset("arrow", data_files={cfg.get("split","test"): files}, split=cfg.get("split","test"))
    return ds

def run_baseline(ds, cfg):
    correct = 0; total = 0
    for p in ds:
        q = p[cfg["question"]]
        a = extract_answer(str(p[cfg["answer"]]), cfg["extract"])
        if a is None: continue
        pp = f"Q: {q}\nA:"
        enc = tok([pp], return_tensors="pt", padding=True)
        out = model.generate(enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE), max_new_tokens=MAX_GEN, do_sample=False, pad_token_id=tok.eos_token_id)
        g = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pa = extract(g)
        if pa and pa == a: correct += 1
        total += 1
        if total % 10 == 0: print(f"  [{total}] acc={correct}/{total} ({100*correct/total:.0f}%)", flush=True)
        if total >= 30: break
    return correct/max(total,1)

def run_steered(ds, cfg, li, mode="prompt"):
    correct = 0; total = 0
    for p in ds:
        q = p[cfg["question"]]
        a = extract_answer(str(p[cfg["answer"]]), cfg["extract"])
        if a is None: continue
        pp = f"Q: {q}\nA:"
        enc = tok([pp], return_tensors="pt", padding=True)
        iids = enc["input_ids"].to(DEVICE); am = enc["attention_mask"].to(DEVICE)
        
        if mode == "prompt":
            with torch.no_grad():
                fwd = model(iids, use_cache=True, output_hidden_states=True, attention_mask=am)
            next_tok = fwd.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(-1)
            hs = fwd.hidden_states
            hp = torch.stack([h[0, -1, :].float() for h in hs[:N_LAYERS]], dim=0).unsqueeze(0)
            with torch.no_grad(): v = tt(hp)
            h_st = hs[li+1][0, -1, :] + ALPHA * v[0, li, :]
            ly = model.model.layers[li]
            k = ly.self_attn.k_proj(h_st.unsqueeze(0).to(torch.bfloat16))
            vo = ly.self_attn.v_proj(h_st.unsqueeze(0).to(torch.bfloat16))
            fwd.past_key_values.layers[li].keys[:,:,-1:,:] = k.view(1,N_KV_HEADS,1,HEAD_DIM).to(fwd.past_key_values.layers[li].keys.dtype)
            fwd.past_key_values.layers[li].values[:,:,-1:,:] = vo.view(1,N_KV_HEADS,1,HEAD_DIM).to(fwd.past_key_values.layers[li].values.dtype)
            past = fwd.past_key_values
            ci = next_tok
            cm = torch.cat([am, torch.ones(1, 1, device=DEVICE, dtype=am.dtype)], dim=1)
            gl = []
            for _ in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(ci, past_key_values=past, use_cache=True, attention_mask=cm)
                nt = fwd.logits[0, -1, :].argmax().item()
                if nt == tok.eos_token_id: break
                gl.append(nt)
                ci = torch.tensor([[nt]], device=DEVICE)
                cm = torch.cat([cm, torch.ones(1, 1, device=DEVICE, dtype=cm.dtype)], dim=1)
                past = fwd.past_key_values
            g = tok.decode(gl, skip_special_tokens=True)
        else:
            with torch.no_grad():
                out = model.generate(iids, max_new_tokens=MAX_GEN, do_sample=False, pad_token_id=tok.eos_token_id)
            g = tok.decode(out[0, iids.shape[1]:], skip_special_tokens=True)
        
        pa = extract(g)
        if pa and pa == a: correct += 1
        total += 1
        if total % 10 == 0: print(f"  [{total}] acc={correct}/{total} ({100*correct/total:.0f}%)", flush=True)
        if total >= 30: break
        gc.collect(); torch.cuda.empty_cache()
    return correct/max(total,1)

print("Loading model and TT...", flush=True)
quant = AwqConfig(bits=4,group_size=128,zero_point=True,backend='gemm')
model = AutoModelForCausalLM.from_pretrained(AWQ_PATH,quantization_config=quant,device_map='cuda',trust_remote_code=True,torch_dtype=torch.float16).eval()
tok = AutoTokenizer.from_pretrained(TOK_PATH,trust_remote_code=True)
tok.pad_token_id=tok.eos_token_id
tt = TrajectoryTransformer(d_model=1536,n_layers=6,n_heads=8,d_ff=1536*4,n_positions=28,d_input=3584).to(DEVICE)
tt.load_state_dict({k.replace('_orig_mod.',''):v for k,v in torch.load('best_tt_d1536_cos.pt',map_location='cpu').items()},strict=False);tt.eval()

results = {}
for key, cfg in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"DATASET: {key}")
    print(f"{'='*60}", flush=True)
    ds = load_ds(key, cfg)
    if ds is None: continue
    
    b = run_baseline(ds, cfg)
    results[f"{key}_baseline"] = b
    print(f"Baseline: {100*b:.1f}%", flush=True)
    
    s = run_steered(ds, cfg, 2, "prompt")
    results[f"{key}_steered_L2"] = s
    print(f"Steered L2: {100*s:.1f}% Δ={100*(s-b):+.1f}pp", flush=True)

print(f"\n{'='*60}")
print("CROSS-DOMAIN GENERALIZATION")
for key in DATASETS:
    b = results.get(f"{key}_baseline", 0)
    s = results.get(f"{key}_steered_L2", 0)
    print(f"  {key:<15} base={100*b:5.1f}% steer={100*s:5.1f}% ({100*(s-b):+.1f}pp)")

json.dump(results, open("cross_domain_results.json", "w"), indent=2)
print("Saved to cross_domain_results.json")
