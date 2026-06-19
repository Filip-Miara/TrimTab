#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Exp 3: Cross-dataset generalization — evaluate TT steering on GSM8k, SVAMP, MATH."""
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
    for p in [r"answer\s+is\s*(-?\d+)",r"The answer is\s*(-?\d+)",r"####\s*(-?\d+)",r"The answer is (\d+)"]:
        m = re.search(p,t,re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+",t); return nums[-1] if nums else None

quant = AwqConfig(bits=4,group_size=128,zero_point=True,backend='gemm')
model = AutoModelForCausalLM.from_pretrained(AWQ_PATH,quantization_config=quant,device_map='cuda',trust_remote_code=True,torch_dtype=torch.float16).eval()
tok = AutoTokenizer.from_pretrained(TOK_PATH,trust_remote_code=True)
tok.pad_token_id=tok.eos_token_id;tok.padding_side='left'
tt = TrajectoryTransformer(d_model=1536,n_layers=6,n_heads=8,d_ff=1536*4,n_positions=28,d_input=3584).to(DEVICE)
tt.load_state_dict({k.replace('_orig_mod.',''):v for k,v in torch.load('best_tt_d1536_cos.pt',map_location='cpu').items()},strict=False);tt.eval()

datasets = {
    "gsm8k": load_dataset("openai/gsm8k","main",split="test"),
}
results = {}

for dset_name, dset in datasets.items():
    print(f"\n{'='*60}")
    print(f"DATASET: {dset_name}")
    print(f"{'='*60}")
    
    # Filter and select problems
    if dset_name == "gsm8k":
        probs = [r for r in dset if len(r["question"])>50][:30]
        answer_key = "answer"
    elif dset_name == "svamp":
        probs = list(dset)[:30]
        answer_key = "Answer"
    
    print(f"{len(probs)} problems", flush=True)
    
    def get_answer(p):
        if dset_name == "gsm8k":
            m = re.search(r"####\s*(-?\d+)", p[answer_key])
            return m.group(1) if m else None
        elif dset_name == "svamp":
            return str(p[answer_key])
    
    # Baseline
    print("Baseline...",flush=True)
    bc=[0]*len(probs)
    for s in range(0,len(probs),BS):
        e=min(s+BS,len(probs));B=e-s
        pp=[f'Q: {p["question"]}\nA:' for p in probs[s:e]]
        enc=tok(pp,return_tensors='pt',padding=True)
        out=model.generate(enc['input_ids'].to(DEVICE),attention_mask=enc['attention_mask'].to(DEVICE),max_new_tokens=MAX_GEN,do_sample=False,pad_token_id=tok.eos_token_id)
        for b in range(B):
            g=tok.decode(out[b,enc['input_ids'].shape[1]:],skip_special_tokens=True)
            pa=extr(g);ca=get_answer(probs[s+b])
            if pa and ca and pa==ca:bc[s+b]=1
        print(f'  [{e}/{len(probs)}] acc={sum(bc)}/{e}',flush=True)
    base=sum(bc)/len(probs);print(f'Baseline: {sum(bc)}/{len(probs)} ({100*base:.1f}%)',flush=True)
    results[f"{dset_name}_baseline"]=base
    
    # Prompt injection at L2 (best prompt layer)
    print(f"Steering at L2 (prompt injection)...",flush=True)
    correct=[0]*len(probs);t0=time.time()
    for s in range(0,len(probs),BS):
        e=min(s+BS,len(probs));B=e-s
        pp=[f'Q: {p["question"]}\nA:' for p in probs[s:e]]
        enc=tok(pp,return_tensors='pt',padding=True)
        iids=enc['input_ids'].to(DEVICE);am=enc['attention_mask'].to(DEVICE)
        with torch.no_grad():
            fwd=model(iids,use_cache=True,output_hidden_states=True,attention_mask=am)
        next_tok=fwd.logits[:,-1,:].argmax(dim=-1)
        # Prompt injection
        hs=fwd.hidden_states
        hp=torch.stack([h[:,-1,:].float() for h in hs[:N_LAYERS]],dim=1)
        with torch.no_grad():v=tt(hp)
        h_st=hs[2+1][:,-1,:]+ALPHA*v[:,2,:]
        ly=model.model.layers[2]
        k=ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
        vo=ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
        fwd.past_key_values.layers[2].keys[:,:,-1:,:]=k.to(fwd.past_key_values.layers[2].keys.dtype)
        fwd.past_key_values.layers[2].values[:,:,-1:,:]=vo.to(fwd.past_key_values.layers[2].values.dtype)
        # Generate
        past,gl,done=fwd.past_key_values,[[] for _ in range(B)],[False]*B
        ci=next_tok.unsqueeze(-1)
        cm=torch.cat([am,torch.ones(B,1,device=DEVICE,dtype=am.dtype)],dim=1)
        for _ in range(MAX_GEN):
            with torch.no_grad():
                fwd=model(ci,past_key_values=past,use_cache=True,attention_mask=cm)
            nts=fwd.logits[:,-1,:].argmax(dim=-1)
            for b in range(B):
                if not done[b]:
                    tid=nts[b].item()
                    if tid==tok.eos_token_id:done[b]=True
                    else:gl[b].append(tid)
            if all(done):break
            past=fwd.past_key_values
            ci=nts.unsqueeze(-1)
            cm=torch.cat([cm,torch.ones(B,1,device=DEVICE,dtype=cm.dtype)],dim=1)
        for b in range(B):
            g=tok.decode(gl[b],skip_special_tokens=True)
            pa=extr(g);ca=get_answer(probs[s+b])
            if pa and ca and pa==ca:correct[s+b]=1
        print(f'  [{e}/{len(probs)}] acc={sum(correct)}/{e}',flush=True)
        gc.collect();torch.cuda.empty_cache()
    acc=sum(correct)/len(probs);d=acc-base
    results[f"{dset_name}_steered"]=acc
    print(f'{dset_name} steered (L2): {100*acc:.1f}% Δ={100*d:+.1f}pp [{time.time()-t0:.0f}s]',flush=True)
    
    # Per-token at L10 (best per-token layer)
    print(f"Steering at L10 (per-token)...",flush=True)
    correct=[0]*len(probs);t0=time.time()
    for s in range(0,len(probs),BS):
        e=min(s+BS,len(probs));B=e-s
        pp=[f'Q: {p["question"]}\nA:' for p in probs[s:e]]
        enc=tok(pp,return_tensors='pt',padding=True)
        iids=enc['input_ids'].to(DEVICE);am=enc['attention_mask'].to(DEVICE)
        past,gl,first,done=None,[[] for _ in range(B)],True,[False]*B
        ci,cm=iids,am
        for _ in range(MAX_GEN):
            with torch.no_grad():
                fwd=model(ci,past_key_values=past,use_cache=True,output_hidden_states=True,attention_mask=cm)
            nts=fwd.logits[:,-1,:].argmax(dim=-1)
            for b in range(B):
                if not done[b]:
                    tid=nts[b].item()
                    if tid==tok.eos_token_id:done[b]=True
                    else:gl[b].append(tid)
            if all(done):break
            if not first:
                hs=fwd.hidden_states
                hp=torch.stack([h[:,-1,:].float() for h in hs[:N_LAYERS]],dim=1)
                with torch.no_grad():v=tt(hp)
                h_st=hs[10+1][:,-1,:]+ALPHA*v[:,10,:]
                ly=model.model.layers[10]
                k=ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
                vo=ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
                fwd.past_key_values.layers[10].keys[:,:,-1:,:]=k.to(fwd.past_key_values.layers[10].keys.dtype)
                fwd.past_key_values.layers[10].values[:,:,-1:,:]=vo.to(fwd.past_key_values.layers[10].values.dtype)
            past=fwd.past_key_values
            ci=nts.unsqueeze(-1)
            cm=torch.cat([cm,torch.ones(B,1,device=DEVICE,dtype=cm.dtype)],dim=1)
            first=False
        for b in range(B):
            g=tok.decode(gl[b],skip_special_tokens=True)
            pa=extr(g);ca=get_answer(probs[s+b])
            if pa and ca and pa==ca:correct[s+b]=1
        print(f'  [{e}/{len(probs)}] acc={sum(correct)}/{e}',flush=True)
        gc.collect();torch.cuda.empty_cache()
    acc=sum(correct)/len(probs);d=acc-base
    results[f"{dset_name}_per_token"]=acc
    print(f'{dset_name} per-token (L10): {100*acc:.1f}% Δ={100*d:+.1f}pp [{time.time()-t0:.0f}s]',flush=True)

# Summary
print(f"\n{'='*60}")
print("CROSS-DATASET GENERALIZATION")
print(f"{'='*60}")
for d in datasets:
    b=results.get(f"{d}_baseline",0)
    s=results.get(f"{d}_steered",0)
    p=results.get(f"{d}_per_token",0)
    print(f"  {d}: baseline={100*b:.1f}% prompt_L2={100*s:.1f}% ({100*(s-b):+.1f}pp) per_token_L10={100*p:.1f}% ({100*(p-b):+.1f}pp)")

json.dump(results,open("exp3_results.json","w"),indent=2)
print("Saved to exp3_results.json")
