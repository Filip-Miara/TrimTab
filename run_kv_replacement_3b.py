#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Full 36-layer KV replacement sweep on Qwen2.5-3B-AWQ."""
import torch,sys,time,gc,warnings,os,json,re
os.environ["PATH"]=f"{os.path.dirname(sys.executable)}:{os.environ.get('PATH','')}"
warnings.filterwarnings('ignore')
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,AwqConfig

DEVICE='cuda';MODEL_PATH='/home/filip/Projects/qwen3b_awq'
N_LAYERS=36;N_KV_HEADS=2;HEAD_DIM=128;BS=8;MAX_GEN=200;N_PROBS=15

def extr(t):
    for p in [r"answer\s+is\s*(-?\d+)",r"The answer is\s*(-?\d+)",r"####\s*(-?\d+)"]:
        m=re.search(p,t,re.IGNORECASE)
        if m:return m.group(1)
    nums=re.findall(r"-?\d+",t);return nums[-1] if nums else None

quant=AwqConfig(bits=4,group_size=128,zero_point=True,backend='gemm')
model=AutoModelForCausalLM.from_pretrained(MODEL_PATH,quantization_config=quant,device_map='cuda',trust_remote_code=True,torch_dtype=torch.float16).eval()
tok=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True);tok.pad_token_id=tok.eos_token_id;tok.padding_side='left'
ds=load_dataset("openai/gsm8k","main",split="test");problems=[r for r in ds if len(r["question"])>50][:N_PROBS]

def run(li, heads, beta=0.75, steps=30):
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
                hs=fwd.hidden_states;h_st=hs[li+1][:,-1,:]
                ly=model.model.layers[li]
                k_new=ly.self_attn.k_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
                v_new=ly.self_attn.v_proj(h_st.to(torch.bfloat16)).view(B,N_KV_HEADS,1,HEAD_DIM)
                for hi in heads:
                    ko=fwd.past_key_values.layers[li].keys[:,hi:hi+1,-1:,:].clone()
                    vo=fwd.past_key_values.layers[li].values[:,hi:hi+1,-1:,:].clone()
                    fwd.past_key_values.layers[li].keys[:,hi:hi+1,-1:,:]=(1-beta)*ko+beta*k_new[:,hi:hi+1].to(ko.dtype)
                    fwd.past_key_values.layers[li].values[:,hi:hi+1,-1:,:]=(1-beta)*vo+beta*v_new[:,hi:hi+1].to(vo.dtype)
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
base=sum(bc)/N_PROBS;print(f"Baseline: {sum(bc)}/{N_PROBS} ({100*base:.1f}%)",flush=True)
results={"baseline":base}

# Sweep all 36 layers with β=0.75, all heads
all_heads={0,1}
for li in range(N_LAYERS):
    acc=run(li,all_heads,0.75,30);d=acc-base
    results[f"L{li}"]=acc
    print(f"L{li:2d}: {100*acc:.1f}% Δ={100*d:+.1f}pp",flush=True)
    json.dump(results,open("kv_3b_sweep_results.json","w"),indent=2)

print(f"\n{'='*60}")
print("3B KV REPLACEMENT SWEEP (β=0.75)")
top=sorted([(li,results[f"L{li}"]) for li in range(N_LAYERS)],key=lambda x:-x[1])[:5]
for li,acc in top:
    d=acc-base;print(f"  L{li:2d}: {100*acc:.1f}% ({100*d:+5.1f}pp)")
json.dump(results,open("kv_3b_sweep_results.json","w"),indent=2);print("Saved")
