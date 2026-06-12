#!/usr/bin/env python3
"""Sweep: PPL-modulated correction with multiple β, τ combos.

Tests hypothesis: at generation, correction heads produce WRONG offsets.
PPL gating can mitigate harm but can't fix bad corrections.

Compares:
  - prompt-trained heads (best_head_0/1/2, +13.8pp on prompt)
  - gen-trained heads (gen_head_0/1/2, -3.4pp on prompt, 20% gen)
  - Multiple gating strategies: soft (ppl_modulated), hard (threshold)
  - Multiple β, τ values
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
    def __init__(self, d_latent=128, n_latents=32, d_model=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents * d_latent, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, d_model),
        )
    def forward(self, latents):
        return self.net(latents.reshape(latents.shape[0], -1))


class UncertaintyProbe(nn.Module):
    def __init__(self, n_latents: int = 32, d_latent: int = 128):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(n_latents * d_latent, 128), nn.GELU(), nn.Linear(128, 1),
        )
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.probe(latents.reshape(latents.shape[0], -1)).squeeze(-1)


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def train_reading_head(perceiver, model, tokenizer, n_train_samples=100, n_epochs=20):
    print(f"\nTraining reading head on {n_train_samples} samples...")
    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=CACHE_DIR)
    texts = [f"Q: {r['question']}\nA: {r['answer']}" for r in ds if len(r.get("question","")) > 50][:n_train_samples]

    probe = UncertaintyProbe().to(DEVICE)
    all_latents, all_ppl = [], []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            seq_len = inputs.input_ids.shape[1]
            if seq_len < 10: continue
            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states
            logits = out.logits[0]
            probs = F.softmax(logits, dim=-1)
            target_ids = inputs.input_ids[0]
            for pos in range(1, seq_len - 2):
                all_ppl.append(-probs[pos-1, target_ids[pos]].log().item())
                h_pos = torch.stack([h[0, pos, :].float() for h in hs], dim=0)
                _, lat = perceiver(h_pos[:23].unsqueeze(0).to(DEVICE), h_pos[0].unsqueeze(0).to(DEVICE), return_latents=True)
                all_latents.append(lat[0].cpu())
            gc.collect(); torch.cuda.empty_cache()

    n_tokens = len(all_latents)
    print(f"  {n_tokens} tokens collected")
    all_latents_t = torch.stack(all_latents).float()
    all_ppl_t = torch.tensor(all_ppl, dtype=torch.float32)

    perm = torch.randperm(n_tokens)
    n_train = int(n_tokens * 0.8)
    train_l = all_latents_t[perm[:n_train]]
    train_p = all_ppl_t[perm[:n_train]]
    test_l = all_latents_t[perm[n_train:]]
    test_p = all_ppl_t[perm[n_train:]]

    p_mean, p_std = train_p.mean(), train_p.std() + 1e-8
    train_p_norm = (train_p - p_mean) / p_std

    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    best_corr = 0.0
    for epoch in range(n_epochs):
        probe.train()
        perm2 = torch.randperm(n_train)
        for i in range(0, n_train, 256):
            idx = perm2[i:i+256]
            pred = probe(train_l[idx].to(DEVICE))
            loss = F.mse_loss(pred, train_p_norm[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        all_pred, all_actual = [], []
        with torch.no_grad():
            for i in range(0, len(test_l), 256):
                pred = probe(test_l[i:i+256].to(DEVICE)).cpu()
                all_pred.append(pred)
                all_actual.append(test_p[i:i+256])
            pred = torch.cat(all_pred)
            actual = torch.cat(all_actual)
            corr = np.corrcoef(pred.numpy(), actual.numpy())[0, 1]
        if abs(corr) > abs(best_corr):
            best_corr = corr
            torch.save(probe.state_dict(), "best_reading_head.pt")
    probe.load_state_dict(torch.load("best_reading_head.pt", map_location=DEVICE))
    print(f"  Best r={best_corr:.4f}")
    return probe, p_mean, p_std


def evaluate(problems, tok, model, lm_head, perceiver, heads, probe, mode, beta=None, tau=None):
    correct, total = 0, 0
    gate_values = []
    t0 = time.time()

    for idx, prob in enumerate(problems):
        prompt = f"Q: {prob['question']}\nA:"
        correct_ans = parse_answer(prob["answer"])

        if mode == "baseline":
            inp = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=MAX_GEN, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            gen_text = tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            full_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids
            plen = full_ids.shape[1]
            for step in range(MAX_GEN):
                with torch.no_grad():
                    fwd = model(full_ids, output_hidden_states=True)
                hs = fwd.hidden_states
                orig_logits = fwd.logits[0, -1, :]
                next_token = orig_logits.argmax().item()

                h_pos = torch.stack([h[0, -1, :].float() for h in hs], dim=0)
                _, latents = perceiver(h_pos[:23].unsqueeze(0).to(DEVICE),
                                       h_pos[0].unsqueeze(0).to(DEVICE), return_latents=True)

                if mode == "full":
                    off = sum(lm_head(h(latents).to(lm_head.weight.dtype)) for h in heads) / len(heads)
                    next_token = (orig_logits.float() + off[0].float()).argmax().item()
                elif mode == "ppl_soft":
                    ppl_pred = probe(latents.to(DEVICE)).item()
                    alpha = 1.0 / (1.0 + np.exp(-beta * (ppl_pred - tau)))
                    gate_values.append(alpha)
                    if alpha > 0.01:
                        off = sum(lm_head(h(latents).to(lm_head.weight.dtype)) for h in heads) / len(heads)
                        next_token = (orig_logits.float() + alpha * off[0].float()).argmax().item()
                elif mode == "ppl_hard":
                    ppl_pred = probe(latents.to(DEVICE)).item()
                    gate_values.append(1.0 if ppl_pred > tau else 0.0)
                    if ppl_pred > tau:
                        off = sum(lm_head(h(latents).to(lm_head.weight.dtype)) for h in heads) / len(heads)
                        next_token = (orig_logits.float() + off[0].float()).argmax().item()

                if next_token == tok.eos_token_id: break
                full_ids = torch.cat([full_ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)
            gen_text = tok.decode(full_ids[0, plen:], skip_special_tokens=True)

        predicted = parse_answer(gen_text)
        if predicted is not None and correct_ans is not None and predicted == correct_ans:
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            gc.collect(); torch.cuda.empty_cache()

    r = {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}
    if gate_values:
        r["avg_alpha"] = float(np.mean(gate_values))
        r["gate_rate"] = float(np.mean([1 if g > 0.5 else 0 for g in gate_values]))
    return r


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--head-type", choices=["prompt", "gen"], default="prompt")
    parser.add_argument("--n-reading-samples", type=int, default=100)
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
    perceiver = ThoughtDiffusion(d_model=2048, n_layers=24, d_latent=128, n_latents=32, d_text_ctx=2048)
    perceiver.load_state_dict(torch.load(PERCEIVER_PATH, map_location="cpu"), strict=False)
    perceiver.to(DEVICE); perceiver.eval()
    for p in perceiver.parameters(): p.requires_grad = False

    # Load heads
    if args.head_type == "prompt":
        head_files = ["best_head_0.pt", "best_head_1.pt", "best_head_2.pt"]
        head_label = "prompt-trained (+13.8pp on prompt)"
    else:
        head_files = ["gen_head_0.pt", "gen_head_1.pt", "gen_head_2.pt"]
        head_label = "gen-trained (20% gen)"
    heads = nn.ModuleList([CorrectionHead().to(DEVICE) for _ in range(3)])
    for hi, f in enumerate(head_files):
        heads[hi].load_state_dict(torch.load(f, map_location=DEVICE))
        heads[hi].eval()
    print(f"Loaded {args.head_type} heads ({head_label})")

    # Train reading head
    probe, ppl_mean, ppl_std = train_reading_head(perceiver, model, tok,
                                                   n_train_samples=args.n_reading_samples,
                                                   n_epochs=20)

    # Load problems
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    # Sweep configurations
    sweeps = [("baseline", {})]
    sweeps.append(("full", {}))

    # Soft PPL gating: β=steepness, τ=threshold (z-scored perplexity)
    for beta in [1.0, 2.0, 5.0]:
        for tau in [-0.5, 0.0, 0.5, 1.0, 1.5]:
            sweeps.append(("ppl_soft", {"beta": beta, "tau": tau}))

    # Hard PPL gating: τ=threshold
    for tau in [0.5, 1.0, 1.5, 2.0]:
        sweeps.append(("ppl_hard", {"tau": tau}))

    results = {}
    for mode, params in sweeps:
        label = mode
        if params:
            param_str = "_".join(f"{k}{v}" for k, v in params.items())
            label = f"{mode}_{param_str}"
        print(f"\n{'='*60}")
        print(f"Mode: {mode} {params}")
        print(f"{'='*60}")
        r = evaluate(problems, tok, model, lm_head, perceiver, heads, probe, mode, **params)
        results[label] = r
        print(f"  {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)" +
              (f" ᾱ={r.get('avg_alpha', 0):.3f} gate={r.get('gate_rate', 0):.3f}" if "avg_alpha" in r else ""))

    print(f"\n{'='*60}")
    print(f"SWEEP SUMMARY (heads={args.head_type}, {head_label})")
    print(f"{'='*60}")
    baseline_acc = results.get("baseline", {}).get("accuracy", 0)
    print(f"  {'Mode':30s} {'Acc':>8s} {'Δ':>8s} {'ᾱ':>6s} {'Gate':>6s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for label, r in sorted(results.items()):
        acc = r['accuracy']
        delta = acc - baseline_acc
        print(f"  {label:30s} {100*r['accuracy']:7.1f}% {100*delta:+7.1f}pp "
              f"{r.get('avg_alpha', 0):5.3f} {r.get('gate_rate', 0):5.3f}")

    outfile = f"ppl_sweep_{args.head_type}.json"
    with open(outfile, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
