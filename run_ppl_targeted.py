#!/usr/bin/env python3
"""Targeted evaluation: few modes, concurrent per-step evaluation.

Evaluates 4 modes sharing the same model forward pass per step:
  baseline, full_prompt, pplhard_prompt, pplhard_gen

Key: each mode has its own token sequence that diverges.
"""
from __future__ import annotations

import gc, json, os, re, sys, time
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


class UncertaintyProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(32 * 128, 128), nn.GELU(), nn.Linear(128, 1),
        )
    def forward(self, latents):
        return self.probe(latents.reshape(latents.shape[0], -1)).squeeze(-1)


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def train_reading_head(perceiver, model, tokenizer, n_samples=100, n_epochs=20):
    print(f"Training reading head ({n_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=CACHE_DIR)
    texts = [f"Q: {r['question']}\nA: {r['answer']}" for r in ds if len(r.get("question","")) > 50][:n_samples]
    probe = UncertaintyProbe().to(DEVICE)
    all_l, all_p = [], []
    with torch.no_grad():
        for text in texts:
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            sl = inp.input_ids.shape[1]
            if sl < 10: continue
            out = model(**inp, output_hidden_states=True)
            hs = out.hidden_states
            probs = F.softmax(out.logits[0], dim=-1)
            ids = inp.input_ids[0]
            for pos in range(1, sl - 2):
                all_p.append(-probs[pos-1, ids[pos]].log().item())
                h_pos = torch.stack([h[0, pos, :].float() for h in hs], dim=0)
                _, lat = perceiver(h_pos[:23].unsqueeze(0).to(DEVICE), h_pos[0].unsqueeze(0).to(DEVICE), return_latents=True)
                all_l.append(lat[0].cpu())
            gc.collect(); torch.cuda.empty_cache()

    all_l = torch.stack(all_l).float()
    all_p = torch.tensor(all_p, dtype=torch.float32)
    print(f"  {len(all_p)} tokens")
    perm = torch.randperm(len(all_l))
    n_tr = int(len(all_l) * 0.8)
    p_mean, p_std = all_p[perm[:n_tr]].mean(), all_p[perm[:n_tr]].std() + 1e-8
    tr_pn = (all_p[perm[:n_tr]] - p_mean) / p_std
    te_l, te_p = all_l[perm[n_tr:]], all_p[perm[n_tr:]]
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    best_corr = 0.0
    for ep in range(n_epochs):
        probe.train()
        perm2 = torch.randperm(n_tr)
        for i in range(0, n_tr, 256):
            idx = perm2[i:i+256]
            pred = probe(all_l[perm[:n_tr]][idx].to(DEVICE))
            loss = F.mse_loss(pred, tr_pn[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        all_pred = []
        with torch.no_grad():
            for i in range(0, len(te_l), 256):
                all_pred.append(probe(te_l[i:i+256].to(DEVICE)).cpu())
            corr = np.corrcoef(torch.cat(all_pred).numpy(), te_p.numpy())[0, 1]
        if abs(corr) > abs(best_corr):
            best_corr = corr
            torch.save(probe.state_dict(), "best_reading_head.pt")
    probe.load_state_dict(torch.load("best_reading_head.pt", map_location=DEVICE))
    print(f"  r={best_corr:.4f}")
    return probe


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--tau", type=float, default=1.5)
    args = parser.parse_args()

    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()

    print("Loading Perceiver...")
    per = ThoughtDiffusion(d_model=2048, n_layers=24, d_latent=128, n_latents=32, d_text_ctx=2048)
    per.load_state_dict(torch.load(PERCEIVER_PATH, map_location="cpu"), strict=False)
    per.to(DEVICE); per.eval()
    for p in per.parameters(): p.requires_grad = False

    # Load both head types
    print("Loading correction heads...")
    ph = nn.ModuleList([CorrectionHead().to(DEVICE) for _ in range(3)])
    gh = nn.ModuleList([CorrectionHead().to(DEVICE) for _ in range(3)])
    for hi in range(3):
        ph[hi].load_state_dict(torch.load(f"best_head_{hi}.pt", map_location=DEVICE)); ph[hi].eval()
        gh[hi].load_state_dict(torch.load(f"gen_head_{hi}.pt", map_location=DEVICE)); gh[hi].eval()

    probe = train_reading_head(per, model, tok)
    probe.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    # Define modes: each has (label, get_offsets_fn)
    # where get_offsets_fn(orig_logits, off_p, off_g, ppl_pred) → (corrected_logits, alpha)
    def make_modes(tau):
        return {
            "baseline": lambda o, op, og, p: (o, 0.0),
            "full_prompt": lambda o, op, og, p: (o + op, 1.0),
            "full_gen": lambda o, op, og, p: (o + og, 1.0),
            f"hard_prompt_t{tau}": lambda o, op, og, p: (o + op if p > tau else o, 1.0 if p > tau else 0.0),
            f"hard_gen_t{tau}": lambda o, op, og, p: (o + og if p > tau else o, 1.0 if p > tau else 0.0),
        }

    modes = make_modes(args.tau)
    results = {m: {"correct": 0, "total": 0, "alphas": [], "divergence": 0} for m in modes}
    t0 = time.time()

    for idx, prob in enumerate(problems):
        prompt = f"Q: {prob['question']}\nA:"
        correct_ans = parse_answer(prob["answer"])

        # Each mode runs its own generation
        for mode_name, mode_fn in modes.items():
            full_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids
            plen = full_ids.shape[1]
            baseline_tokens = []

            for step in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(full_ids, output_hidden_states=True)
                hs = fwd.hidden_states
                orig = fwd.logits[0, -1, :]
                bt = orig.argmax().item()
                baseline_tokens.append(bt)

                # Shared: latents, both correction offsets
                h_pos = torch.stack([h[0, -1, :].float() for h in hs], dim=0)
                _, lat = per(h_pos[:23].unsqueeze(0).to(DEVICE),
                             h_pos[0].unsqueeze(0).to(DEVICE), return_latents=True)
                ppl = probe(lat.to(DEVICE)).item()
                off_p = sum(lm_head(h(lat).to(lm_head.weight.dtype)) for h in ph).float() / 3
                off_g = sum(lm_head(h(lat).to(lm_head.weight.dtype)) for h in gh).float() / 3

                corr, alpha = mode_fn(orig, off_p[0].float(), off_g[0].float(), ppl)
                nt = corr.argmax().item()
                results[mode_name]["alphas"].append(alpha)
                if step > 0 and nt != baseline_tokens[step]:
                    results[mode_name]["divergence"] += 1

                if nt == tok.eos_token_id: break
                full_ids = torch.cat([full_ids, torch.tensor([[nt]], device=DEVICE)], dim=1)

            gen_text = tok.decode(full_ids[0, plen:], skip_special_tokens=True)
            predicted = parse_answer(gen_text)
            if predicted is not None and correct_ans is not None and predicted == correct_ans:
                results[mode_name]["correct"] += 1
            results[mode_name]["total"] += 1

        gc.collect(); torch.cuda.empty_cache()
        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1}/{args.n_test}] {time.time()-t0:.0f}s")

    print(f"\n{'='*60}")
    print(f"TARGETED SWEEP (τ={args.tau})")
    print(f"{'='*60}")
    ba = results["baseline"]["correct"] / max(results["baseline"]["total"], 1)
    print(f"  {'Mode':25s} {'Acc':>8s} {'Δ':>8s} {'ᾱ':>6s} {'Div%':>6s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for m in sorted(modes.keys()):
        r = results[m]
        acc = r["correct"] / max(r["total"], 1)
        delta = acc - ba
        aa = np.mean(r["alphas"]) if r["alphas"] else 0
        dv = 100 * r["divergence"] / max(len(r["alphas"]), 1)
        print(f"  {m:25s} {100*acc:7.1f}% {100*delta:+7.1f}pp {aa:5.3f} {dv:5.1f}%")

    out = {m: {"correct": r["correct"], "total": r["total"],
               "accuracy": r["correct"]/max(r["total"],1),
               "avg_alpha": float(np.mean(r["alphas"])) if r["alphas"] else 0,
               "divergence_pct": float(100*r["divergence"]/max(len(r["alphas"]),1))}
           for m, r in results.items()}
    with open("ppl_targeted_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to ppl_targeted_results.json")


if __name__ == "__main__":
    main()
