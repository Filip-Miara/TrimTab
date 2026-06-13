#!/usr/bin/env python3 -u
"""Collect generation trajectories using forward hooks.

No extra forward pass — hooks capture hidden states during model.generate().
This avoids the expensive use_cache=False forward entirely.
"""
from __future__ import annotations

import gc, json, os, re, sys, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEVICE = "cuda"
OUTPUT_DIR = "/run/media/filip/B522-875D/Datasets/project_data"
MAX_GEN = 200
D_MODEL = 1536
N_LAYERS = 28
BATCH_SIZE = 5000
N_PROBLEMS = 500
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def extract_number(text):
    for p in [r"answer\s+is\s*(-?\d+)", r"####\s*(-?\d+)", r"So answer is\s*(-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+", text)
    if nums: return nums[-1]
    return None


def save_batch(hidden_list, vel_list, token_list, correct_list, batch_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"math15_trajs_batch_{batch_idx:04d}.pt")
    data = {
        "hidden_seqs": torch.stack(hidden_list).half(),
        "velocity_targets": torch.stack(vel_list).half(),
        "token_ids": torch.tensor(token_list),
        "is_correct": torch.tensor(correct_list),
    }
    torch.save(data, path)
    print(f"  Saved batch {batch_idx}: {len(hidden_list)} tokens -> {path} "
          f"({os.path.getsize(path)/1e6:.0f}MB)", flush=True)
    hidden_list.clear(); vel_list.clear(); token_list.clear(); correct_list.clear()
    gc.collect()
    return batch_idx + 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-problems", type=int, default=N_PROBLEMS)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M params", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    examples = ('Q: Janet has 5 ducks. She buys 3 more. How many?\nA: Step 1: 5. Step 2: buy 3. Step 3: 5+3=8. So answer is 8.\n\n'
                'Q: 12 muffins/hr, 8 hours?\nA: Step 1: 12/hr. Step 2: 8h. Step 3: 12x8=96. So answer is 96.\n\n')

    ds = load_dataset("openai/gsm8k", "main", split="train")
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_problems]
    print(f"  {len(problems)} problems", flush=True)

    hidden_list, vel_list, token_list, correct_list = [], [], [], []
    n_tokens, batch_idx, n_correct_problems = 0, 0, 0
    t0 = time.time()

    # Pre-register hooks once (they persist across generate() calls)
    # Each hook captures (layer_output, position) — we slice the last token
    hooks = []
    for li in range(N_LAYERS + 1):
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output[0, -1, :] = last token's hidden state for this layer
                hook_data[layer_idx].append(output[0, -1, :].cpu().half())
            return hook
        hook = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(hook)

    import threading
    hook_data_lock = threading.Lock()
    hook_data = [[] for _ in range(N_LAYERS + 1)]

    for idx, prob in enumerate(problems):
        prompt = f"{examples}Q: {prob['question']}\nA:"
        correct_ans = extract_number(prob["answer"])
        full_input = tok(prompt, return_tensors="pt").to(DEVICE)
        plen = full_input.input_ids.shape[1]

        # Reset hook buffers for this problem
        for hd in hook_data:
            hd.clear()

        # Generate with hooks capturing hidden states
        with torch.no_grad():
            out = model.generate(**full_input, max_new_tokens=MAX_GEN, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen_ids = out[0, plen:]
        if len(gen_ids) <= 1:
            continue

        # Each hook_data[li] contains hidden states for each generated step
        # First capture is the prompt's last position, subsequent are generated tokens
        # Trim to match gen_ids length
        n_captured = min(len(hook_data[0]), len(gen_ids))
        n_vel = n_captured - 1  # need 2 positions for one velocity

        for i in range(n_vel):
            # Stack all layers' hidden states for position i and i+1
            h_here = torch.stack([hook_data[li][i] for li in range(N_LAYERS)], dim=0)  # (28, 1536)
            h_next = torch.stack([hook_data[li][i + 1] for li in range(N_LAYERS)], dim=0)
            vel = h_next - h_here

            hidden_list.append(h_here)
            vel_list.append(vel)
            token_list.append(gen_ids[i].item())
            n_tokens += 1

        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        predicted = extract_number(gen_text)
        is_correct = (predicted is not None and correct_ans is not None and predicted == correct_ans)
        if is_correct: n_correct_problems += 1
        for _ in gen_ids: correct_list.append(is_correct)

        if len(hidden_list) >= BATCH_SIZE:
            batch_idx = save_batch(hidden_list, vel_list, token_list, correct_list, batch_idx,
                                   os.path.join(args.output_dir, "math15_gen_trajs"))

        if (idx + 1) % 50 == 0 and n_tokens > 0:
            rate = n_tokens / (time.time() - t0)
            print(f"  [{idx+1}/{len(problems)}] {n_tokens} tokens ({rate:.0f} tok/s) "
                  f"acc={n_correct_problems}/{idx+1} ({100*n_correct_problems/(idx+1):.0f}%) "
                  f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    if hidden_list:
        batch_idx = save_batch(hidden_list, vel_list, token_list, correct_list, batch_idx,
                               os.path.join(args.output_dir, "math15_gen_trajs"))

    print(f"\nCollection complete!", flush=True)
    print(f"  Tokens: {n_tokens}, Batches: {batch_idx}", flush=True)
    print(f"  Gen accuracy: {100*n_correct_problems/len(problems):.1f}%", flush=True)

    meta = {"n_tokens": n_tokens, "n_batches": batch_idx, "n_problems": len(problems),
            "d_model": D_MODEL, "n_layers": N_LAYERS, "model": MODEL_NAME,
            "accuracy": n_correct_problems / len(problems)}
    with open(os.path.join(args.output_dir, "math15_gen_trajs_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
