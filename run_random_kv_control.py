#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Random K/V control: compare correct vs random vs no replacement at L10 heads 2&3."""
import torch,sys,time,gc,warnings,os,json,re
os.environ["PATH"]=f"{os.path.dirname(sys.executable)}:{os.environ.get('PATH','')}"
warnings.filterwarnings('ignore')
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,AwqConfig

DEVICE='cuda';AWQ_PATH='/run/media/filip/C27C20AB7C209C63/Qwen2.5-7B-AWQ/qwen7b_awq'
TOK_PATH='/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
NK=4;HD=128;BS=8;MG=200;N=30

def xt(t):
    for p in [r"answer\s+is\s*(-?\d+)",r"The answer is\s*(-?\d+)",r"####\s*(-?\d+)"]:
        m=re.search(p,t,re.IGNORECASE)
        if m:return m.group(1)
    ns=re.findall(r"-?\d+",t);return ns[-1] if ns else None

quant=AwqConfig(bits=4,group_size=128,zero_point=True,backend='gemm')
md=AutoModelForCausalLM.from_pretrained(AWQ_PATH,quantization_config=quant,device_map='cuda',trust_remote_code=True,torch_dtype=torch.float16).eval()
tk=AutoTokenizer.from_pretrained(TOK_PATH,trust_remote_code=True);tk.pad_token_id=tk.eos_token_id;tk.padding_side='left'
ds=load_dataset("openai/gsm8k","main",split="test");ps=[r for r in ds if len(r["question"])>50][:N]

results={}

def run(label,mode,steps=30):
    """mode: 'none', 'replace' (correct K/V), 'random' (random K/V matched norm)."""
    cc=[0]*N;t0=time.time()
    for s in range(0,N,BS):
        e=min(s+BS,N);B=e-s
        pp=[f'Q: {p["question"]}\nA:' for p in ps[s:e]]
        en=tk(pp,return_tensors='pt',padding=True)
        ii=en['input_ids'].to(DEVICE);am=en['attention_mask'].to(DEVICE)
        pt,gl,fr,dn=None,[[] for _ in range(B)],True,[False]*B
        ci,cm=ii,am
        for stp in range(1,MG+1):
            with torch.no_grad():
                fw=md(ci,past_key_values=pt,use_cache=True,output_hidden_states=True,attention_mask=cm)
            nt=fw.logits[:,-1,:].argmax(dim=-1)
            for b in range(B):
                if not dn[b]:
                    td=nt[b].item()
                    if td==tk.eos_token_id:dn[b]=True
                    else:gl[b].append(td)
            if all(dn):break
            if not fr and stp<=steps and mode!='none':
                hh=fw.hidden_states;hx=hh[11][:,-1,:];ly=md.model.layers[10]
                if mode=='replace':
                    kn=ly.self_attn.k_proj(hx.to(torch.bfloat16)).view(B,NK,1,HD)
                    vn=ly.self_attn.v_proj(hx.to(torch.bfloat16)).view(B,NK,1,HD)
                elif mode=='random':
                    # Generate random K/V with matched statistics
                    with torch.no_grad():
                        k_ex=ly.self_attn.k_proj(hx.to(torch.bfloat16)).view(B,NK,1,HD)
                        v_ex=ly.self_attn.v_proj(hx.to(torch.bfloat16)).view(B,NK,1,HD)
                    k_std=k_ex.std();k_mean=k_ex.mean()
                    v_std=v_ex.std();v_mean=v_ex.mean()
                    kn=torch.randn(B,NK,1,HD,device=DEVICE,dtype=torch.bfloat16)*k_std+k_mean
                    vn=torch.randn(B,NK,1,HD,device=DEVICE,dtype=torch.bfloat16)*v_std+v_mean
                for hi in [2,3]:
                    ko=fw.past_key_values.layers[10].keys[:,hi:hi+1,-1:,:].clone()
                    vo=fw.past_key_values.layers[10].values[:,hi:hi+1,-1:,:].clone()
                    fw.past_key_values.layers[10].keys[:,hi:hi+1,-1:,:]=0.25*ko+0.75*kn[:,hi:hi+1].to(ko.dtype)
                    fw.past_key_values.layers[10].values[:,hi:hi+1,-1:,:]=0.25*vo+0.75*vn[:,hi:hi+1].to(vo.dtype)
            pt=fw.past_key_values;ci=nt.unsqueeze(-1)
            cm=torch.cat([cm,torch.ones(B,1,device=DEVICE,dtype=cm.dtype)],dim=1);fr=False
        for b in range(B):
            g=tk.decode(gl[b],skip_special_tokens=True)
            pa=xt(g);ca=re.search(r'####\s*(-?\d+)',ps[s+b]['answer'])
            if pa and ca and pa==ca.group(1):cc[s+b]=1
        print(f'  [{e}/{N}] acc={sum(cc)}/{e}',flush=True);gc.collect();torch.cuda.empty_cache()
    ac=sum(cc)/N;results[label]=ac
    print(f'{label}: {100*ac:.1f}% [{time.time()-t0:.0f}s]',flush=True)

# Loop baseline (no replacement)
print("Loop baseline (no replacement)...",flush=True)
run("loop_baseline","none",0)

# Correct K/V replacement (β=0.75, 30 tokens)
print("\nCorrect K/V replacement (β=0.75, L10 h23)...",flush=True)
run("correct_kv","replace",30)

# Random K/V replacement (matched norm, β=0.75, 30 tokens)
print("\nRandom K/V replacement (matched norm, β=0.75, L10 h23)...",flush=True)
run("random_kv","random",30)

# Correct K/V at shorter window (15 tokens)
print("\nCorrect K/V (β=0.75, L10 h23, 15 tokens)...",flush=True)
run("correct_kv_15t","replace",15)

print(f"\n{'='*60}")
print("RANDOM K/V CONTROL EXPERIMENT")
lb=results.get("loop_baseline",0)
for k in results:
    d=results[k]-lb
    print(f"  {k:<20} {100*results[k]:5.1f}% ({100*d:+5.1f}pp vs loop)")
json.dump(results,open("random_kv_results.json","w"),indent=2);print("Saved")
