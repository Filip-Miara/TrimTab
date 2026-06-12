#!/usr/bin/env python3
"""Combined: reasoning-step error gates correction head + layer-0 injection.

Architecture:
  For each token step:
    1. Forward → hidden states + logits
    2. Reasoning-step TT → prediction error (gate signal)
    3. If error > τ:
       a. Perceiver → latents → correction head → logit offset
       b. Layer-0 velocity injection for next step
       c. corrected_logits = logits + offset
    4. Sample next token
"""
from __future__ import annotations

import gc, re, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from src.adapters.thought_diffusion import ThoughtDiffusion
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
MAX_GEN = 100


class CorrectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32 * 128, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 2048),
        )
    def forward(self, latents):
        return self.net(latents.reshape(latents.shape[0], -1))


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def get_reasoning_error(tt, hidden_states_traj, t, DEVICE):
    """Compute prediction error for position t in the trajectory."""
    traj = hidden_states_traj[:t + 1].unsqueeze(0).float()
    with torch.no_grad():
        v_pred = tt(traj.to(DEVICE), causal=True)[0, t]
    v_actual = (hidden_states_traj[t + 1] - hidden_states_traj[t]).to(DEVICE)
    return F.mse_loss(v_pred, v_actual).item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--mode", choices=["baseline", "gated", "full"], default="gated")
    parser.add_argument("--tau", type=float, default=1.4, help="Error threshold for gating")
    parser.add_argument("--alpha", type=float, default=0.5, help="Layer-0 injection strength")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()

    # Load reasoning-step TT
    print("Loading reasoning-step TT...")
    tt = TrajectoryTransformer(d_model=512, n_layers=6, n_heads=8, d_ff=2048, n_positions=50)
    tt.load_state_dict(torch.load("best_reasoning_transformer.pt", map_location=DEVICE))
    tt.to(DEVICE); tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    # Load Perceiver + correction heads if needed
    per, heads = None, None
    if args.mode in ("gated", "full"):
        print("Loading Perceiver...")
        per = ThoughtDiffusion(d_model=2048, n_layers=24, d_latent=128, n_latents=32, d_text_ctx=2048)
        per.load_state_dict(torch.load("best_perceiver.pt", map_location="cpu"), strict=False)
        per.to(DEVICE); per.eval()
        for p in per.parameters(): p.requires_grad = False

        heads = nn.ModuleList([CorrectionHead().to(DEVICE) for _ in range(3)])
        for hi in range(3):
            heads[hi].load_state_dict(torch.load(f"best_head_{hi}.pt", map_location=DEVICE))
            heads[hi].eval()

    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    examples = ("Q: Janet has 5. She buys 3. How many?\nA: Step 1: 5. Step 2: She buys 3. Step 3: 5+3=8.\nSo answer is 8.\n\n"
                "Q: Bakery sells 12/hr. How many in 8h?\nA: Step 1: 12/hr. Step 2: 8 hours. Step 3: 12x8=96.\nSo answer is 96.\n\n")

    correct, total = 0, 0
    gate_count, correction_count = 0, 0
    t_start = time.time()

    for idx, prob in enumerate(problems):
        question = prob["question"]
        correct_ans = parse_answer(prob["answer"])
        prompt = f"{examples}Q: {question}\nA:"

        if args.mode == "baseline":
            inp = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=MAX_GEN, do_sample=False, pad_token_id=tok.eos_token_id)
            gen_text = tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            full_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids
            plen = full_ids.shape[1]
            vel_hook = None

            for step in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(full_ids, output_hidden_states=True)
                hs = fwd.hidden_states
                final_hs = hs[-1][0]
                actual_step = full_ids.shape[1] - plen
                orig_logits = fwd.logits[0, -1, :]
                next_token = orig_logits.argmax().item()
                should_correct = False

                if actual_step >= 2:
                    n = min(actual_step, 48)
                    traj = final_hs[plen:plen + n]
                    err = get_reasoning_error(tt, traj, n - 2, DEVICE)
                    should_correct = (args.mode == "full") or (args.mode == "gated" and err > args.tau)

                if should_correct and per is not None:
                    gate_count += 1
                    h_pos = torch.stack([h[0, -1, :].float() for h in hs], dim=0)
                    x = h_pos[:23].unsqueeze(0).to(DEVICE)
                    ctx = h_pos[0].unsqueeze(0).to(DEVICE)
                    _, latents = per(x, ctx, return_latents=True)
                    off = sum(lm_head(h(latents).to(lm_head.weight.dtype)) for h in heads) / len(heads)
                    next_token = (orig_logits.float() + off[0].float()).argmax().item()
                    correction_count += 1
                    if args.alpha > 0 and vel_hook is None:
                        v0 = traj[-1] - traj[-2]
                        def make_v_hook(v_data, alpha):
                            def hook(module, args, kwargs):
                                h = args[0]
                                h[:, -1, :] = h[:, -1, :] + alpha * v_data.to(h.dtype).to(h.device)
                                return ((h, *args[1:]) if len(args) > 1 else (h,)), kwargs
                            return hook
                        vel_hook = model.model.layers[0].register_forward_pre_hook(make_v_hook(v0, args.alpha), with_kwargs=True)

                if next_token == tok.eos_token_id: break
                full_ids = torch.cat([full_ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)

            if vel_hook is not None: vel_hook.remove()
            gen_text = tok.decode(full_ids[0, plen:], skip_special_tokens=True)

        predicted = parse_answer(gen_text)
        if predicted is not None and correct_ans is not None and predicted == correct_ans: correct += 1
        total += 1
        if (idx + 1) % 10 == 0:
            gc.collect(); torch.cuda.empty_cache()
            print(f"  [{idx+1}/{args.n_test}] acc={correct}/{total} ({100*correct/total:.0f}%) {time.time()-t_start:.0f}s")

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}" + (f" (τ={args.tau}, α={args.alpha})" if args.mode != "baseline" else ""))
    print(f"  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    if args.mode != "baseline":
        print(f"  Gate rate: {100*gate_count/max(total,1):.0f}% ({gate_count}/{total} steps gated)")
        print(f"  Correction rate: {100*correction_count/max(total,1):.0f}% ({correction_count}/{total} steps corrected)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
