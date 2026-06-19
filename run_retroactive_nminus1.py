#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Approach 1: modify K/V at N-1, re-process N. Approach 2: regenerate from L after retroactive N-1."""
import torch, sys, time, gc, warnings, os, json, re
os.environ["PATH"] = f"{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}"
warnings.filterwarnings('ignore')
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
sys.path.insert(0, '.')
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"; AWQ_PATH = "/run/media/filip/C27C20AB7C209C63/Qwen2.5-7B-AWQ/qwen7b_awq"
TOK_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
N_LAYERS=28; N_KV_HEADS=4; HEAD_DIM=128; BS=8; MAX_GEN=200; ALPHA=0.1; LI=10

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

BASE_SEQ = None  # track position of first generated token

def steer_at_pos(past_key_values, li, pos, vel, B):
    """Modify K/V at layer li, position pos."""
    h_st = None  # will be set by caller
    ly = model.model.layers[li]
    k = ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
    vo = ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
    # Modify at position pos instead of -1
    past_key_values.layers[li].keys[:,:,pos:pos+1,:] = k.to(past_key_values.layers[li].keys.dtype)
    past_key_values.layers[li].values[:,:,pos:pos+1,:] = vo.to(past_key_values.layers[li].values.dtype)

results = {}
N_PROBS = 15  # 15 for speed, 30 for full

print("Baseline (15 problems)...", flush=True)
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
results['baseline']=base

print("\n=== Per-token standard (L10) ===",flush=True)
correct=[0]*N_PROBS;t0=time.time()
for s in range(0,N_PROBS,BS):
    e=min(s+BS,N_PROBS);B=e-s
    pp=[f'Q: {p["question"]}\nA:' for p in problems[s:e]]
    enc=tok(pp,return_tensors='pt',padding=True)
    iids=enc['input_ids'].to(DEVICE);am=enc['attention_mask'].to(DEVICE)
    past,gl,first,done=None,[[] for _ in range(B)],True,[False]*B
    ci,cm=iids,am;BASE_SEQ=ci.shape[1]
    for _ in range(MAX_GEN):
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
            hs=fwd.hidden_states
            hp=torch.stack([h[:,-1,:].float() for h in hs[:N_LAYERS]],dim=1)
            with torch.no_grad():v=tt(hp)
            h_st=hs[LI+1][:,-1,:]+ALPHA*v[:,LI,:]
            ly=model.model.layers[LI]
            k=ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
            vo=ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
            fwd.past_key_values.layers[LI].keys[:,:,-1:,:]=k.to(fwd.past_key_values.layers[LI].keys.dtype)
            fwd.past_key_values.layers[LI].values[:,:,-1:,:]=vo.to(fwd.past_key_values.layers[LI].values.dtype)
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
acc=sum(correct)/N_PROBS;results['per_token']=acc
print(f'Per-token: {100*acc:.1f}% Δ={100*(acc-base):+.1f}pp [{time.time()-t0:.0f}s]',flush=True)

print("\n=== Approach 1: steer at N-1, re-process N ===",flush=True)
correct=[0]*N_PROBS;t0=time.time()
for s in range(0,N_PROBS,BS):
    e=min(s+BS,N_PROBS);B=e-s
    pp=[f'Q: {p["question"]}\nA:' for p in problems[s:e]]
    enc=tok(pp,return_tensors='pt',padding=True)
    iids=enc['input_ids'].to(DEVICE);am=enc['attention_mask'].to(DEVICE)
    past,gl,first,done=None,[[] for _ in range(B)],True,[False]*B
    ci,cm=iids,am;BASE_SEQ=ci.shape[1]
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
        if not first and step>=2:
            # Retroactive: modify position N-2 (two steps back), re-forward current token
            hs=fwd.hidden_states
            hp=torch.stack([h[:,-1,:].float() for h in hs[:N_LAYERS]],dim=1)
            with torch.no_grad():v=tt(hp)
            # Compute steering for current token, apply to N-2 position
            h_st=hs[LI+1][:,-1,:]+ALPHA*v[:,LI,:]
            ly=model.model.layers[LI]
            k=ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
            vo=ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
            # Position N-2 (relative to current seq length)
            pos=fwd.past_key_values.layers[LI].keys.shape[-2]-2
            fwd.past_key_values.layers[LI].keys[:,:,pos:pos+1,:]=k.to(fwd.past_key_values.layers[LI].keys.dtype)
            fwd.past_key_values.layers[LI].values[:,:,pos:pos+1,:]=vo.to(fwd.past_key_values.layers[LI].values.dtype)
            # Re-forward current token with modified past
            with torch.no_grad():
                fwd=model(ci,past_key_values=fwd.past_key_values,use_cache=True,attention_mask=cm)
            # Re-check logits
            nt=fwd.logits[:,-1,:].argmax(dim=-1)
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
acc=sum(correct)/N_PROBS;results['approach1']=acc
print(f'Approach 1: {100*acc:.1f}% Δ={100*(acc-base):+.1f}pp [{time.time()-t0:.0f}s]',flush=True)

print(f"\n{'='*60}")
for k in ['per_token','approach1']:
    d=results[k]-base
    print(f'  {k:<15} {100*results[k]:5.1f}% ({100*d:+5.1f}pp)')
json.dump(results,open('retro_n1_results.json','w'),indent=2)
print('Saved')
