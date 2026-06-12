#!/usr/bin/env python3
"""Generate reasoning-step trajectories: token-to-token hidden states.

Each trajectory: [h_0, h_1, ..., h_{T-1}] where h_t is the final-layer
hidden state for the last token after generating t tokens.

Flow matching: given h[t], predict h[t+1] - h[t].
This captures how the model's internal representation evolves as it
reasons step by step — fundamentally different from layer-to-layer.
"""
from __future__ import annotations

import gc, json, os, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
OUTPUT_DIR = "/run/media/filip/B522-875D/Datasets/project_data/reasoning_trajs_5k/"
N_TRAJ = 1000
MAX_GEN = 50  # tokens per answer

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading GSM8K...")
ds_test = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
ds_train = load_dataset("openai/gsm8k", "main", split="train", cache_dir=CACHE_DIR)
all_problems = [r for r in ds_test if len(r["question"]) > 50] + \
              [r for r in ds_train if len(r["question"]) > 50]
np.random.shuffle(all_problems)
print(f"  {len(all_problems)} problems available")

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

examples = ("Q: Janet has 5. She buys 3. How many?\nA: Step 1: 5. Step 2: She buys 3. Step 3: 5+3=8.\nSo answer is 8.\n\n"
            "Q: Bakery sells 12/hr. How many in 8h?\nA: Step 1: 12/hr. Step 2: 8 hours. Step 3: 12x8=96.\nSo answer is 96.\n\n")

t0 = time.time()
trajs = []

for idx, prob in enumerate(all_problems):
    if len(trajs) >= N_TRAJ:
        break

    prompt = f"{examples}Q: {prob['question']}\nA:"
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    plen = inp.input_ids.shape[1]

    # Fast generation
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_GEN, do_sample=False, pad_token_id=tok.eos_token_id)

    gen_len = out.shape[1] - plen
    if gen_len < 5:
        continue

    # Single forward pass with output_hidden_states=True
    with torch.no_grad():
        fwd = model(out, output_hidden_states=True)

    hs = fwd.hidden_states  # tuple of (1, S, D) per layer, len=25

    # Extract final-layer hidden state for each generated position
    # For position pos (0-indexed within full sequence):
    # h_final[pos] = hs[-1][0, pos, :]  — last layer's output at this position
    traj = torch.stack([hs[-1][0, pos, :].cpu().float() for pos in range(plen, out.shape[1])], dim=0)
    # traj shape: (gen_len, 2048)

    trajs.append(traj)

    if (len(trajs)) % 500 == 0:
        elapsed = time.time() - t0
        rate = len(trajs) / elapsed
        eta = (N_TRAJ - len(trajs)) / max(rate, 0.01)
        avg_len = np.mean([t.shape[0] for t in trajs])
        print(f"  {len(trajs)}/{N_TRAJ} | avg_len={avg_len:.0f} | {elapsed:.0f}s | {rate:.1f}/s | ETA {eta:.0f}s")
        gc.collect()
        torch.cuda.empty_cache()

# Save
meta = {
    "n_trajectories": len(trajs),
    "d_model": 2048,
    "type": "reasoning_step",
    "max_gen": MAX_GEN,
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_time_s": round(time.time() - t0, 2),
}
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

# Pad to max length for consistent tensors
max_len = max(t.shape[0] for t in trajs)
padded = torch.zeros(len(trajs), max_len, 2048, dtype=torch.float32)
for i, t in enumerate(trajs):
    padded[i, :t.shape[0]] = t

torch.save(padded, os.path.join(OUTPUT_DIR, "all_trajs.pt"))

# Also save lengths for masking
lengths = torch.tensor([t.shape[0] for t in trajs])
torch.save(lengths, os.path.join(OUTPUT_DIR, "lengths.pt"))

elapsed = time.time() - t0
print(f"Done: {len(trajs)} trajs in {elapsed:.1f}s ({len(trajs)/elapsed:.1f}/s)")
print(f"  Avg length: {np.mean([t.shape[0] for t in trajs]):.0f} tokens")
print(f"  Max length: {max_len}")
print(f"  Saved to {OUTPUT_DIR}")
