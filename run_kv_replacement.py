#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Full 28-layer KV replacement sweep + β interpolation at L10 heads 2&3."""
import torch, sys, time, gc, warnings, os, json, re
os.environ["PATH"] = f"{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}"
warnings.filterwarnings('ignore')
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

DEVICE = "cuda"; AWQ_PATH = "/run/media/filip/C27C20AB7C209C63/Qwen2.5-7B-AWQ/qwen7b_awq"
TOK_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
N_LAYERS=28; N_KV_HEADS=4; HEAD_DIM=128; BS=8; MAX_GEN=200; N_PROBS=15

def extr(t):
    for p in [r"answer\s+is\s*(-?\d+)",r"The answer is\s*(-?\d+)",r"####\s*(-?\d+)"]:
        m = re.search(p,t,re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+",t); return nums[-1] if nums else None

quant=AwqConfig(bits=4,group_size=128,zero_point=True,backend='gemm')
model=AutoModelForCausalLM.from_pretrained(AWQ_PATH,quantization_config=quant,device_map='cuda',trust_remote_code=True,torch_dtype=torch.float16).eval()
tok=AutoTokenizer.from_pretrained(TOK_PATH,trust_remote_code=True);tok.pad_token_id=tok.eos_token_id;tok.padding_side='left'
ds=load_dataset("openai/gsm8k","main",split="test");problems=[r for r in ds if len(r["question"])>50][:N_PROBS]

def replace_heads(fwd, li, heads, beta=1.0):
    """Replace K/V at layer li, specified heads, with interpolation β."""
    hs=fwd.hidden_states; B=hs[0].shape[0]
    h_st=hs[li+1][:,-1,:]  # fresh K/V from layer output
    ly=model.model.layers[li]
    k_new=ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
    v_new=ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
    for hi in heads:
        if beta == 1.0:
            fwd.past_key_values.layers[li].keys[:,hi:hi+1,-1:,:]=k_new[:,hi:hi+1].to(fwd.past_key_values.layers[li].keys.dtype)
            fwd.past_key_values.layers[li].values[:,hi:hi+1,-1:,:]=v_new[:,hi:hi+1].to(fwd.past_key_values.layers[li].values.dtype)
        else:
            k_old=fwd.past_key_values.layers[li].keys[:,hi:hi+1,-1:,:].clone()
            v_old=fwd.past_key_values.layers[li].values[:,hi:hi+1,-1:,:].clone()
            k_int=(1-beta)*k_old+beta*k_new[:,hi:hi+1].to(k_old.dtype)
            v_int=(1-beta)*v_old+beta*v_new[:,hi:hi+1].to(v_old.dtype)
            fwd.past_key_values.layers[li].keys[:,hi:hi+1,-1:,:]=k_int
            fwd.past_key_values.layers[li].values[:,hi:hi+1,-1:,:]=v_int

def run(li, heads, beta=1.0, steps=30):
    correct=[0]*N_PROBS;t0=time.time()
    for s in range(0,N_PROBS,BS):
        e=min(s+BS,N_PROBS);B=e-s
        pp=[f'Q: {p["question"]}\nA:' for p in problems[s:e]]
        enc=tok(pp,return_tensors='pt',padding=True)
        iids=enc['input_ids'].to(DEVICE);am=enc['attention_mask'].to(DEVICE)
        past,gl,first,done=None,[[] for _ in range(B)],True,[False]*B
        ci,cm=iids,am
        for step in range(1,MAX_GEN+1):
            with torch.no_grad():
                fwd=model(ci,past_key_values=past,use_cache=True,output_hidden_states=True,attention_mask=cm)
            nt=fwd.logits[:,-1,:].argmax(dim=-1)
            for b in range(B):
                if not done[b]:
                    tid=nt[b].item()
                    if tid==tok.eos_token_id:done[b]=True
                    else:gl[b].append(tid)
            if all(done):break
            if not first and step<=steps:
                replace_heads(fwd,li,heads,beta)
            past=fwd.past_key_values;ci=nt.unsqueeze(-1)
            cm=torch.cat([cm,torch.ones(B,1,device=DEVICE,dtype=cm.dtype)],dim=1);first=False
        for b in range(B):
            g=tok.decode(gl[b],skip_special_tokens=True)
            pa=extr(g);ca=re.search(r'####\s*(-?\d+)',problems[s+b]['answer'])
            if pa and ca and pa==ca.group(1):correct[s+b]=1
        print(f'  [{e}/{N_PROBS}] acc={sum(correct)}/{e}',flush=True);gc.collect();torch.cuda.empty_cache()
    return sum(correct)/N_PROBS

print("Baseline...",flush=True)
bc=[0]*N_PROBS
for s in range(0,N_PROBS,BS):
    e=min(s+BS,N_PROBS)
    pp=[f'Q: {p["question"]}\nA:' for p in problems[s:e]]
    enc=tok(pp,return_tensors='pt',padding=True)
    out=model.generate(enc['input_ids'].to(DEVICE),attention_mask=enc['attention_mask'].to(DEVICE),max_new_tokens=MAX_GEN,do_sample=False,pad_token_id=tok.eos_token_id)
    for b in range(e-s):
        g=tok.decode(out[b,enc['input_ids'].shape[1]:],skip_special_tokens=True)
        pa=extr(g);ca=re.search(r'####\s*(-?\d+)',problems[s+b]['answer'])
        if pa and ca and pa==ca.group(1):bc[s+b]=1
    print(f'  [{e}/{N_PROBS}] acc={sum(bc)}/{e}',flush=True)
base=sum(bc)/N_PROBS;print(f'Baseline: {sum(bc)}/{N_PROBS} ({100*base:.1f}%)',flush=True)
results={"baseline":base}

# Experiment 1: Full 28-layer sweep (heads 2&3, β=1.0, 30 steps)
print("\n=== FULL 28-LAYER SWEEP (heads 2&3) ===",flush=True)
for li in range(N_LAYERS):
    acc=run(li,{2,3},1.0,30);d=acc-base
    results[f"L{li}_h23"]=acc
    print(f"  L{li:2d} h23: {100*acc:.1f}% Δ={100*d:+.1f}pp",flush=True)
    json.dump(results,open("kv_replacement_results.json","w"),indent=2)

# Experiment 2: β interpolation at L10 heads 2&3
print("\n=== β INTERPOLATION (L10 heads 2&3) ===",flush=True)
for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
    acc=run(10,{2,3},beta,30);d=acc-base
    results[f"L10_h23_b{beta}"]=acc
    print(f"  β={beta:.2f}: {100*acc:.1f}% Δ={100*d:+.1f}pp",flush=True)
    json.dump(results,open("kv_replacement_results.json","w"),indent=2)

print(f"\n{'='*60}")
print("KV REPLACEMENT FULL RESULTS")
for k in results:
    if k=="baseline":continue
    d=results[k]-base
    print(f"  {k:<15} {100*results[k]:5.1f}% ({100*d:+5.1f}pp)")
json.dump(results,open("kv_replacement_results.json","w"),indent=2)
print("Saved to kv_replacement_results.json")
