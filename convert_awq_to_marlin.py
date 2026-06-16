#!/usr/bin/env python
"""Convert AWQ model from GEMM to Marlin format for faster inference."""
import torch, time, gc
from awq import AutoAWQForCausalLM
from awq.modules.linear.gemm import WQLinear_GEMM
from awq.modules.linear.marlin import WQLinear_Marlin
from transformers import AutoTokenizer

MODEL_PATH = "/home/filip/Projects/qwen3b_awq"
SAVE_PATH = "/home/filip/Projects/qwen3b_marlin"

print("Loading AWQ model...", flush=True)
t0 = time.time()
model = AutoAWQForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda")
print(f"Loaded in {time.time()-t0:.0f}s", flush=True)

# Count GEMM layers
gemm_count = 0
for name, mod in model.named_modules():
    if isinstance(mod, WQLinear_GEMM):
        gemm_count += 1
print(f"Found {gemm_count} WQLinear_GEMM layers to convert", flush=True)

# Convert each GEMM layer to Marlin
t0 = time.time()
for name, mod in model.named_modules():
    for child_name, child in list(mod.named_children()):
        if isinstance(child, WQLinear_GEMM):
            marlin = WQLinear_Marlin.from_linear(
                child, w_bit=4, group_size=128,
                scales=getattr(child, 'scales', None),
                zeros=None,
            )
            setattr(mod, child_name, marlin)
            del child
            gc.collect()
            torch.cuda.empty_cache()

print(f"Conversion done in {time.time()-t0:.0f}s", flush=True)

# Verify
marlin_count = 0
for name, mod in model.named_modules():
    if isinstance(mod, WQLinear_Marlin):
        marlin_count += 1
print(f"Verified: {marlin_count} WQLinear_Marlin layers", flush=True)

# Save
print(f"Saving to {SAVE_PATH}...", flush=True)
model.save_pretrained(SAVE_PATH)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True,
                                           cache_dir="/run/media/filip/B522-875D/Datasets/hub")
tokenizer.save_pretrained(SAVE_PATH)
print(f"Saved!", flush=True)

# Benchmark
print("\nBenchmarking Marlin model...", flush=True)
tok = AutoTokenizer.from_pretrained(SAVE_PATH, trust_remote_code=True)
tok.pad_token = tok.eos_token
inp = tok("Q: What is 2+2?\nA:", return_tensors="pt").to("cuda")
t0 = time.time()
for _ in range(10):
    out = model.generate(**inp, max_new_tokens=100, do_sample=False, pad_token_id=tok.eos_token_id)
print(f"Marlin 3B: {10*100/(time.time()-t0):.0f} tok/s, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
