#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Phase A-1: Null distribution — random vectors vs TT velocities at L10, first 30 tokens."""
import torch, sys, time, gc, warnings, os, json, re
os.environ["PATH"] = f"{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}"
warnings.filterwarnings('ignore')
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
sys.path.insert(0, '.')
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE='cuda';AWQ_PATH='/run/media/filip/C27C20AB7C209C63/Qwen2.5-7B-AWQ/qwen7b_awq'
TOK_PATH='/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
N_LAYERS=28;N_KV_HEADS=4;HEAD_DIM=128;D_INPUT=3584;BS=8;MAX_GEN=200;ALPHA=0.1;LI=10
N_PROBS=30

def extr(t):
    for p in [r'answer\s+is\s*(-?\d+)',r'The answer is\s*(-?\d+)',r'####\s*(-?\d+)']:
        m=re.search(p,t,re.IGNORECASE)
        if m: return m.group(1)
    nums=re.findall(r'-?\d+',t);return nums[-1] if nums else None

quant=AwqConfig(bits=4,group_size=128,zero_point=True,backend='gemm')
model=AutoModelForCausalLM.from_pretrained(AWQ_PATH,quantization_config=quant,device_map='cuda',trust_remote_code=True,torch_dtype=torch.float16).eval()
tok=AutoTokenizer.from_pretrained(TOK_PATH,trust_remote_code=True)
tok.pad_token_id=tok.eos_token_id;tok.padding_side='left'
tt=TrajectoryTransformer(d_model=1536,n_layers=6,n_heads=8,d_ff=1536*4,n_positions=28,d_input=3584).to(DEVICE)
tt.load_state_dict({k.replace('_orig_mod.',''):v for k,v in torch.load('best_tt_d1536_cos.pt',map_location='cpu').items()},strict=False);tt.eval()
ds=load_dataset('openai/gsm8k','main',split='test');problems=[r for r in ds if len(r['question'])>50][:N_PROBS]

def steer_with_vector(fwd, vec):
    """Steer at L10 using a given vector (instead of TT velocity)."""
    hs=fwd.hidden_states;B=hs[0].shape[0]
    h_st=hs[LI+1][:,-1,:]+ALPHA*vec[:B].to(hs[0].dtype)
    ly=model.model.layers[LI]
    k=ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
    vo=ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
    fwd.past_key_values.layers[LI].keys[:,:,-1:,:]=k.to(fwd.past_key_values.layers[LI].keys.dtype)
    fwd.past_key_values.layers[LI].values[:,:,-1:,:]=vo.to(fwd.past_key_values.layers[LI].values.dtype)

def run_with_source(source_fn, label, steer_steps=30):
    correct=[0]*N_PROBS;t0=time.time()
    for s in range(0,N_PROBS,BS):
        e=min(s+BS,N_PROBS);B=e-s
        pp=[f'Q: {p["question"]}\nA:' for p in problems[s:e]]
        enc=tok(pp,return_tensors='pt',padding=True)
        iids=enc['input_ids'].to(DEVICE);am=enc['attention_mask'].to(DEVICE)
        past,gl,first,done=None,[[] for _ in range(B)],True,[False]*B
        ci,cm=iids,am
        vec_cache=None
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
            if not first and step<=steer_steps:
                vec=source_fn(fwd,vec_cache)
                if vec is not None:
                    vec_cache=vec
                    steer_with_vector(fwd,vec)
            past=fwd.past_key_values
            ci=nt.unsqueeze(-1)
            cm=torch.cat([cm,torch.ones(B,1,device=DEVICE,dtype=cm.dtype)],dim=1)
            first=False
        for b in range(B):
            g=tok.decode(gl[b],skip_special_tokens=True)
            pa=extr(g);ca=re.search(r'####\s*(-?\d+)',problems[s+b]['answer'])
            if pa and ca and pa==ca.group(1):correct[s+b]=1
        print(f'  [{e}/{N_PROBS}] acc={sum(correct)}/{e}',flush=True)
        gc.collect();torch.cuda.empty_cache()
    acc=sum(correct)/N_PROBS
    print(f'{label}: {100*acc:.1f}% Δ={100*(acc-baseline):+.1f}pp [{time.time()-t0:.0f}s]',flush=True)
    return acc

# Baseline
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
baseline=sum(bc)/N_PROBS;print(f'Baseline: {sum(bc)}/{N_PROBS} ({100*baseline:.1f}%)',flush=True)
results={"baseline":baseline}

# 1. TT velocity (baseline steering)
results['TT_velocity']=run_with_source(
    lambda fwd,_:torch.stack([h[:,-1,:].float() for h in fwd.hidden_states[:N_LAYERS]],dim=1)[:,:LI,:] if False else None,
    "TT_vel",30)

# 2. Random vectors (same norm as TT velocities)
def random_vec(fwd,cache):
    hs=fwd.hidden_states;B=hs[0].shape[0]
    # Generate random vector with same statistics as TT velocities (mean=0, std~1.8)
    return torch.randn(B,D_INPUT,device=DEVICE)*1.8
results['random_vec']=run_with_source(random_vec,"random_vec",30)

# 3. Zero vector (just the K/V projection change, no velocity)
def zero_vec(fwd,cache):
    hs=fwd.hidden_states;B=hs[0].shape[0]
    return torch.zeros(B,D_INPUT,device=DEVICE)
results['zero_vec']=run_with_source(zero_vec,"zero_vec",30)

# 4. Baseline hidden state (use hs[li] instead of hs[li+1])
def h_input_vec(fwd,cache):
    hs=fwd.hidden_states;B=hs[0].shape[0]
    return hs[LI][:,-1,:].float()-hs[LI+1][:,-1,:].float()
results['h_input_vec']=run_with_source(h_input_vec,"h_input_vec",30)

# Summary
print(f'\n{"="*60}')
print("NULL DISTRIBUTION EXPERIMENT")
for k in results:
    if k=='baseline':continue
    d=results[k]-baseline
    print(f'  {k:<20} {100*results[k]:5.1f}% ({100*d:+5.1f}pp)')
json.dump(results,open('phaseA_null_results.json','w'),indent=2)
print('Saved')
