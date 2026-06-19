#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Exp 1: Prompt injection + periodic per-token refresh. Exp 2: Prompt position sweep."""
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
    for p in [r"answer\s+is\s*(-?\d+)",r"The answer is\s*(-?\d+)",r"####\s*(-?\d+)"]:
        m = re.search(p,t,re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+",t); return nums[-1] if nums else None

quant = AwqConfig(bits=4,group_size=128,zero_point=True,backend='gemm')
model = AutoModelForCausalLM.from_pretrained(AWQ_PATH,quantization_config=quant,device_map='cuda',trust_remote_code=True,torch_dtype=torch.float16).eval()
tok = AutoTokenizer.from_pretrained(TOK_PATH,trust_remote_code=True)
tok.pad_token_id=tok.eos_token_id;tok.padding_side='left'
tt = TrajectoryTransformer(d_model=1536,n_layers=6,n_heads=8,d_ff=1536*4,n_positions=28,d_input=3584).to(DEVICE)
tt.load_state_dict({k.replace('_orig_mod.',''):v for k,v in torch.load('best_tt_d1536_cos.pt',map_location='cpu').items()},strict=False);tt.eval()
ds = load_dataset("openai/gsm8k","main",split="test")
problems = [r for r in ds if len(r["question"])>50][:30]

def steer(li, pos, h_st, pkv, B):
    ly=model.model.layers[li]
    k=ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
    vo=ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
    pkv.layers[li].keys[:,:,pos:pos+1,:]=k.to(pkv.layers[li].keys.dtype)
    pkv.layers[li].values[:,:,pos:pos+1,:]=vo.to(pkv.layers[li].values.dtype)

def steer_current(fwd, li, B):
    hs=fwd.hidden_states
    hp=torch.stack([h[:,-1,:].float() for h in hs[:N_LAYERS]],dim=1)
    with torch.no_grad(): v=tt(hp)
    h_st=hs[li+1][:,-1,:]+ALPHA*v[:,li,:]
    steer(li, -1, h_st, fwd.past_key_values, B)

def generate(li_prompt, li_refresh, refresh_every, label):
    correct=[0]*30;t0=time.time()
    for s in range(0,30,BS):
        e=min(s+BS,30);B=e-s
        pp=[f'Q: {p["question"]}\nA:' for p in problems[s:e]]
        enc=tok(pp,return_tensors='pt',padding=True)
        iids=enc['input_ids'].to(DEVICE);am=enc['attention_mask'].to(DEVICE)
        past,gl,first,done=None,[[] for _ in range(B)],True,[False]*B
        ci,cm=iids,am
        for step in range(MAX_GEN):
            with torch.no_grad():
                fwd=model(ci,past_key_values=past,use_cache=True,output_hidden_states=True,attention_mask=cm)
            nt=fwd.logits[:,-1,:].argmax(dim=-1)
            for b in range(B):
                if not done[b]:
                    tid=nt[b].item()
                    if tid==tok.eos_token_id:done[b]=True
                    else:gl[b].append(tid)
            if all(done):break
            if not first:
                if step==1 and li_prompt is not None:
                    # Prompt injection at specified layer
                    steer_current(fwd, li_prompt, B)
                elif li_refresh is not None and refresh_every>0 and step%refresh_every==0:
                    steer_current(fwd, li_refresh, B)
            past=fwd.past_key_values
            ci=nt.unsqueeze(-1)
            cm=torch.cat([cm,torch.ones(B,1,device=DEVICE,dtype=cm.dtype)],dim=1)
            first=False
        for b in range(B):
            g=tok.decode(gl[b],skip_special_tokens=True)
            pa=extr(g);ca=re.search(r'####\s*(-?\d+)',problems[s+b]['answer'])
            if pa and ca and pa==ca.group(1):correct[s+b]=1
        print(f'  [{e}/30] acc={sum(correct)}/{e}',flush=True)
        gc.collect();torch.cuda.empty_cache()
    acc=sum(correct)/30
    print(f'{label}: {100*acc:.1f}% Δ={100*(acc-baseline):+.1f}pp [{time.time()-t0:.0f}s]',flush=True)
    return acc

# Baseline
print("Baseline...",flush=True)
bc=[0]*30
for s in range(0,30,BS):
    e=min(s+BS,30)
    pp=[f'Q: {p["question"]}\nA:' for p in problems[s:e]]
    enc=tok(pp,return_tensors='pt',padding=True)
    out=model.generate(enc['input_ids'].to(DEVICE),attention_mask=enc['attention_mask'].to(DEVICE),max_new_tokens=MAX_GEN,do_sample=False,pad_token_id=tok.eos_token_id)
    for b in range(e-s):
        g=tok.decode(out[b,enc['input_ids'].shape[1]:],skip_special_tokens=True)
        pa=extr(g);ca=re.search(r'####\s*(-?\d+)',problems[s+b]['answer'])
        if pa and ca and pa==ca.group(1):bc[s+b]=1
    print(f'  [{e}/30] acc={sum(bc)}/{e}',flush=True)
baseline=sum(bc)/30;print(f'Baseline: {sum(bc)}/30 ({100*baseline:.1f}%)',flush=True)
results={"baseline":baseline}

# EXP 1: Periodic refresh
print("\n=== EXP 1: Periodic Refresh ===",flush=True)
for label,li_p,li_r,every in [
    ("prompt_L2_only",2,None,0),
    ("prompt_L2_refresh5",2,8,5),
    ("prompt_L2_refresh10",2,8,10),
    ("prompt_L2_refresh20",2,8,20),
    ("per_token_L8_only",None,8,1),
]:
    results[label]=generate(li_p,li_r,every,label)

# EXP 2: Prompt position sweep
print("\n=== EXP 2: Prompt Position Sweep ===",flush=True)
for pos_pct in [0.25, 0.50, 0.75, 1.0]:
    label=f"pos_{int(pos_pct*100)}pct"
    # Modify generate to steer at prompt position = pos_pct * prompt_len
    correct=[0]*30;t0=time.time()
    for s in range(0,30,BS):
        e=min(s+BS,30);B=e-s
        pp=[f'Q: {p["question"]}\nA:' for p in problems[s:e]]
        enc=tok(pp,return_tensors='pt',padding=True)
        iids=enc['input_ids'].to(DEVICE);am=enc['attention_mask'].to(DEVICE)
        past,gl,first,done=None,[[] for _ in range(B)],True,[False]*B
        ci,cm=iids,am
        for step in range(MAX_GEN):
            with torch.no_grad():
                fwd=model(ci,past_key_values=past,use_cache=True,output_hidden_states=True,attention_mask=cm)
            nt=fwd.logits[:,-1,:].argmax(dim=-1)
            for b in range(B):
                if not done[b]:
                    tid=nt[b].item()
                    if tid==tok.eos_token_id:done[b]=True
                    else:gl[b].append(tid)
            if all(done):break
            if not first:
                if step==1:
                    # Steer at position = pos_pct * prompt_length (first generated token)
                    prompt_len=am.shape[1]
                    target_pos=int(prompt_len*pos_pct)-1
                    hs=fwd.hidden_states
                    hp=torch.stack([h[:,-1,:].float() for h in hs[:N_LAYERS]],dim=1)
                    with torch.no_grad(): v=tt(hp)
                    h_st=hs[2+1][:,-1,:]+ALPHA*v[:,2,:]
                    steer(2, target_pos, h_st, fwd.past_key_values, B)
            past=fwd.past_key_values
            ci=nt.unsqueeze(-1)
            cm=torch.cat([cm,torch.ones(B,1,device=DEVICE,dtype=cm.dtype)],dim=1)
            first=False
        for b in range(B):
            g=tok.decode(gl[b],skip_special_tokens=True)
            pa=extr(g);ca=re.search(r'####\s*(-?\d+)',problems[s+b]['answer'])
            if pa and ca and pa==ca.group(1):correct[s+b]=1
        print(f'  [{e}/30] acc={sum(correct)}/{e}',flush=True)
        gc.collect();torch.cuda.empty_cache()
    acc=sum(correct)/30;results[label]=acc
    print(f'{label}: {100*acc:.1f}% Δ={100*(acc-baseline):+.1f}pp [{time.time()-t0:.0f}s]',flush=True)

# Summary
print(f"\n{'='*60}")
print("EXP 1: PERIODIC REFRESH")
for k in ["prompt_L2_only","prompt_L2_refresh5","prompt_L2_refresh10","prompt_L2_refresh20","per_token_L8_only"]:
    if k in results: d=results[k]-baseline;print(f'  {k:<25} {100*results[k]:5.1f}% ({100*d:+5.1f}pp)')
print("\nEXP 2: PROMPT POSITION SWEEP")
for k in ["pos_25pct","pos_50pct","pos_75pct","pos_100pct"]:
    if k in results: d=results[k]-baseline;print(f'  {k:<25} {100*results[k]:5.1f}% ({100*d:+5.1f}pp)')

json.dump(results,open("exp12_results.json","w"),indent=2)
print("\nSaved to exp12_results.json")
