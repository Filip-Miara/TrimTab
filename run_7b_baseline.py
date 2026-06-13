#!/usr/bin/env python3 -u
"""Qwen2.5-7B-Instruct GSM8K baseline with proper chat template."""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, re, json
from datasets import load_dataset

MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'

print('[7B] Loading...', flush=True)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                              quantization_config=quant, device_map='cuda')
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tok.pad_token = tok.eos_token
print('[7B] Loaded', flush=True)

def extract_last_number(text):
    for p in [r'answer\s+is\s*(-?\d+)', r'The answer is\s*(-?\d+)', r'####\s*(-?\d+)']:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r'-?\d+', text)
    if nums: return nums[-1]
    return None

ds = load_dataset('openai/gsm8k', 'main', split='test')
correct, n = 0, 100
for i in range(n):
    prob = ds[i]
    msgs = [{'role': 'user', 'content': f'Q: {prob["question"]}\nA:'}]
    inp = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    inp = inp.input_ids.to('cuda')
    out = model.generate(inp, max_new_tokens=400, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, inp.shape[1]:], skip_special_tokens=True)
    pa = extract_last_number(gen)
    ca = re.search(r'####\s*(-?\d+)', prob['answer'])
    if pa and ca and pa == ca.group(1): correct += 1
    if (i+1) % 10 == 0: print(f'[7B] [{i+1}/{n}] acc={correct}/{i+1} ({100*correct/(i+1):.0f}%)', flush=True)

print(f'[7B] Qwen2.5-7B-Instruct: {correct}/{n} ({100*correct/n:.0f}%)', flush=True)
with open('qwen25_7b_baseline.json', 'w') as f:
    json.dump({'accuracy': correct/n, 'correct': correct, 'total': n}, f)
print('[7B] Done', flush=True)
