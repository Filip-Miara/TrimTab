#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""6.4 Iterative Multi-Layer Steering — different layers at different generation stages."""
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

quant=AwqConfig(bits=4,group_size=128,zero_point=True,backend='gemm')
model=AutoModelForCausalLM.from_pretrained(AWQ_PATH,quantization_config=quant,device_map='cuda',trust_remote_code=True,torch_dtype=torch.float16).eval()
tok=AutoTokenizer.from_pretrained(TOK_PATH,trust_remote_code=True)
tok.pad_token_id=tok.eos_token_id;tok.padding_side='left'
tt=TrajectoryTransformer(d_model=1536,n_layers=6,n_heads=8,d_ff=1536*4,n_positions=28,d_input=3584).to(DEVICE)
tt.load_state_dict({k.replace('_orig_mod.',''):v for k,v in torch.load('best_tt_d1536_cos.pt',map_location='cpu').items()},strict=False);tt.eval()
ds=load_dataset("openai/gsm8k","main",split="test");problems=[r for r in ds if len(r["question"])>50][:30]

def steer(fwd, li):
    hs=fwd.hidden_states;B=hs[0].shape[0]
    hp=torch.stack([h[:,-1,:].float() for h in hs[:N_LAYERS]],dim=1)
    with torch.no_grad(): v=tt(hp)
    h_st=hs[li+1][:,-1,:]+ALPHA*v[:,li,:]
    ly=model.model.layers[li]
    k=ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
    vo=ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
    fwd.past_key_values.layers[li].keys[:,:,-1:,:]=k.to(fwd.past_key_values.layers[li].keys.dtype)
    fwd.past_key_values.layers[li].values[:,:,-1:,:]=vo.to(fwd.past_key_values.layers[li].values.dtype)

def run_schedule(schedule,label):
    """schedule: list of (layer, start_step, end_step) tuples. Step counts from 1."""
    correct=[0]*30;t0=time.time()
    for s in range(0,30,BS):
        e=min(s+BS,30);B=e-s
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
            if not first:
                # Check schedule: which layer to steer at this step
                for li,start,end in schedule:
                    if start<=step<=end:
                        steer(fwd,li)
                        break
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

# Single-layer baselines
for li in [2,8,10]:
    label=f"L{li}_only"
    results[label]=run_schedule([(li,1,999)],label)

# Multi-layer schedules (step ranges divide 200 steps into phases)
# Phase boundaries: early (1-20), mid (21-60), late (61+)
schedules = [
    ("L10_early_L8_mid_L2_late", [(10,1,20),(8,21,60),(2,61,999)]),
    ("L2_early_L8_mid_L10_late", [(2,1,20),(8,21,60),(10,61,999)]),
    ("L10_only_first30_then_off", [(10,1,30)]),
    ("L10_1-10_L2_11-20_L8_21-30", [(10,1,10),(2,11,20),(8,21,30)]),
    ("L10+L2_alternate", [(10,1,10),(2,11,20),(10,21,30),(2,31,40),(10,41,50),(2,51,60)]),
]
for label,sched in schedules:
    results[label]=run_schedule(sched,label)

# Summary
print(f"\n{'='*60}")
print("ITERATIVE MULTI-LAYER STEERING")
for k in results:
    if k=="baseline":continue
    d=results[k]-baseline
    print(f'  {k:<35} {100*results[k]:5.1f}% ({100*d:+5.1f}pp)')

json.dump(results,open("iterative_multilayer_results.json","w"),indent=2)
print("Saved")
