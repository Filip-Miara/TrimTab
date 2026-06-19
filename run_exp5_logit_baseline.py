#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Exp 5: Logit baseline — store logit hashes for model validation across sessions."""
import torch, hashlib, json, warnings, os, sys, re, time
os.environ["PATH"] = f"{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}"
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

DEVICE = "cuda"; AWQ_PATH = "/run/media/filip/C27C20AB7C209C63/Qwen2.5-7B-AWQ/qwen7b_awq"
TOK_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

def logit_hash(logits):
    """Create a compact hash of top-20 logit values for baseline verification."""
    top_vals, top_idx = torch.topk(logits[0, -1, :].softmax(dim=-1), 20)
    data = (top_idx.cpu().numpy().tobytes(), top_vals.cpu().numpy().tobytes())
    return hashlib.sha256(b"".join(data)).hexdigest()[:16]

quant = AwqConfig(bits=4, group_size=128, zero_point=True, backend="gemm")
model = AutoModelForCausalLM.from_pretrained(AWQ_PATH, quantization_config=quant, device_map='cuda', trust_remote_code=True, torch_dtype=torch.float16).eval()
tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
tok.pad_token_id = tok.eos_token_id

# Reference problem
question = "Q: Janet has 5 ducks. She buys 3 more. How many ducks does she have?\nA:"
msgs = [{"role": "user", "content": question}]
inp = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors='pt')
iids = inp["input_ids"].to(DEVICE)

with torch.no_grad():
    out = model.generate(iids, max_new_tokens=100, do_sample=False, pad_token_id=tok.eos_token_id)
    gen_ids = out[0, iids.shape[1]:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

print(f"Reference problem: {question}", flush=True)
print(f"Generated: {gen_text}", flush=True)
print(f"First 5 tokens: {[tok.decode([t]) for t in gen_ids[:5]]}", flush=True)

# Store baseline
baseline = {
    "model": "Qwen2.5-7B-Instruct AWQ",
    "quant": "AWQ 4-bit (gemm)",
    "question": question,
    "generated_text": gen_text,
    "first_5_tokens": [tok.decode([t.item()]) for t in gen_ids[:5]],
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "nvidia_driver": os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null").read().strip(),
}

json.dump(baseline, open("logit_baseline.json", "w"), indent=2)
print(f"\nSaved baseline to logit_baseline.json")

# Verification function
def verify_baseline():
    with torch.no_grad():
        out = model.generate(iids, max_new_tokens=100, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, iids.shape[1]:], skip_special_tokens=True)
    return gen == gen_text

print(f"Self-verification: {'PASS' if verify_baseline() else 'FAIL'}", flush=True)
