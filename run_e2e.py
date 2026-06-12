#!/usr/bin/env python3
"""End-to-end: combine Phase 1/3 (logit correction) + Phase 2 (layer-0 injection).

Pipeline:
  For each token:
    1. Forward → hidden states
    2. Perceiver → latents + velocities
    3. Correction head → logit offsets (+13.8pp proven)
    4. corrected_logits = logits + offset → sample next token
    5. Inject velocity at layer 0 for next forward pass
    6. Continue
"""
from __future__ import annotations

import gc, os, re, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.thought_diffusion import ThoughtDiffusion

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
PERCEIVER_PATH = "best_perceiver.pt"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
MAX_GEN = 150


class CorrectionHead(nn.Module):
    def __init__(self, d_latent=128, n_latents=32, d_model=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents * d_latent, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, d_model),
        )
    def forward(self, latents):
        return self.net(latents.reshape(latents.shape[0], -1))


def parse_answer(text: str) -> str | None:
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for pat in [
        r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
        r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--mode", choices=["baseline", "corrected", "both"], default="both")
    parser.add_argument("--alpha", type=float, default=1.0, help="Layer-0 injection strength")
    parser.add_argument("--head-prefix", type=str, default="best_head_",
                        help="Prefix for trained correction head files")
    parser.add_argument("--n-heads", type=int, default=3, help="Number of ensemble heads")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()

    per = ThoughtDiffusion(d_model=2048, n_layers=24, d_latent=128, n_latents=32, d_text_ctx=2048)
    per.load_state_dict(torch.load(PERCEIVER_PATH, map_location="cpu"), strict=False)
    per.to(DEVICE); per.eval()
    for p in per.parameters(): p.requires_grad = False

    heads = nn.ModuleList([CorrectionHead().to(DEVICE) for _ in range(args.n_heads)])
    for hi in range(args.n_heads):
        heads[hi].load_state_dict(torch.load(f"{args.head_prefix}{hi}.pt", map_location=DEVICE))
        heads[hi].eval()

    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    examples = ("Q: Janet has 5. She buys 3. How many?\nA: Step 1: 5. Step 2: She buys 3. Step 3: 5+3=8.\nSo answer is 8.\n\n"
                "Q: Bakery sells 12/hr. How many in 8h?\nA: Step 1: 12/hr. Step 2: 8 hours. Step 3: 12x8=96.\nSo answer is 96.\n\n")

    def get_velocity_and_correction(hidden_states):
        h_pos = torch.stack([h[0, -1, :].float() for h in hidden_states], dim=0)
        x = h_pos[:23].unsqueeze(0).to(DEVICE)
        ctx = h_pos[0].unsqueeze(0).to(DEVICE)
        # Get velocity from Perceiver
        v = per(x, ctx)[0]  # (1, 23, 2048)
        # Get latents (need return_latents)
        _, lat = per(x, ctx, return_latents=True)
        return v, lat

    def get_correction(latents):
        off = sum(lm_head(h(latents).to(lm_head.weight.dtype)) for h in heads) / len(heads)
        return off[0].float()

    all_results = []

    for mode in (["baseline", "corrected"] if args.mode == "both" else [args.mode]):
        print(f"\n{'='*60}")
        print(f"Mode: {mode}" + (f" (α={args.alpha})" if mode == "corrected" else ""))
        print(f"{'='*60}")

        correct = 0
        t_start = time.time()

        for idx, prob in enumerate(problems):
            question = prob["question"]
            correct_ans = parse_answer(prob["answer"])
            prompt = f"{examples}Q: {question}\nA:"

            input_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids
            prompt_len = input_ids.shape[1]
            gen_len = 0

            if mode == "baseline":
                # Standard greedy generation
                with torch.no_grad():
                    out = model.generate(
                        input_ids, max_new_tokens=MAX_GEN, do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                gen_text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)

            else:
                # Corrected generation: logit correction + layer-0 injection
                full_ids = input_ids.clone()
                vel_hook = None

                for step in range(MAX_GEN):
                    # Forward pass
                    with torch.no_grad():
                        fwd = model(full_ids, output_hidden_states=True)

                    hs = fwd.hidden_states
                    logits = fwd.logits[0, -1, :]

                    # Get velocity and latents
                    v, lat = get_velocity_and_correction(hs)

                    # Correction head → logit offset
                    offset = get_correction(lat)
                    corrected_logits = logits.float() + offset

                    next_token = corrected_logits.argmax().item()

                    if next_token == tok.eos_token_id:
                        break

                    full_ids = torch.cat([full_ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)
                    gen_len += 1

                    if step == 0 and args.alpha > 0:
                        v0 = v[0]  # (2048,) — layer 0 velocity
                        v0_for_hook = v0.clone()

                        def make_v_hook():
                            v_data = v0_for_hook
                            alpha_val = args.alpha
                            def hook(module, args, kwargs):
                                h = args[0]
                                h[:, -1, :] = h[:, -1, :] + alpha_val * v_data.to(h.dtype).to(h.device)
                                return ((h, *args[1:]) if len(args) > 1 else (h,)), kwargs
                            return hook

                        if vel_hook is not None:
                            vel_hook.remove()
                        vel_hook = model.model.layers[0].register_forward_pre_hook(
                            make_v_hook(), with_kwargs=True
                        )

                    if step >= MAX_GEN - 1:
                        break

                if vel_hook is not None:
                    vel_hook.remove()

                gen_text = tok.decode(full_ids[0, prompt_len:], skip_special_tokens=True)

            predicted = parse_answer(gen_text)
            is_correct = predicted is not None and correct_ans is not None and predicted == correct_ans
            if is_correct:
                correct += 1

            elapsed = time.time() - t_start
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(problems)}] acc={correct}/{idx+1} ({100*correct/(idx+1):.0f}%) {elapsed:.0f}s")

            gc.collect()
            torch.cuda.empty_cache()

        acc = correct / len(problems)
        print(f"\n  Final: {correct}/{len(problems)} ({100*acc:.1f}%) in {time.time()-t_start:.0f}s")
        all_results.append({"mode": mode, "correct": correct, "total": len(problems), "accuracy": acc})

        # Save
        import json
        with open("e2e_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    for r in all_results:
        print(f"  {r['mode']}: {r['accuracy']*100:.1f}% ({r['correct']}/{r['total']})")
    if len(all_results) == 2:
        delta = all_results[1]['accuracy'] - all_results[0]['accuracy']
        print(f"  Delta: {delta*100:+.1f}pp")


if __name__ == "__main__":
    main()
