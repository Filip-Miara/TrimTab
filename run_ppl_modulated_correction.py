#!/usr/bin/env python3
"""PPL-modulated correction: reading head gates correction strength.

Key insight (3-delegation consensus):
  α = σ(β · (ppl_pred - τ))
  corrected_logits = logits + α · offset

Reading head (r=0.86) predicts token perplexity from Perceiver latents.
When perplexity is high (uncertain), apply stronger correction.
When perplexity is low (confident), apply weaker correction (avoid regress).

This avoids:
  - Hard-gating failure (0% accuracy in run_combined.py)
  - Distribution shift (reading head trained on same latent distribution)
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
    """Linear probe: Perceiver latents → uncertainty score."""
    def __init__(self, n_latents: int = 32, d_latent: int = 128):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(n_latents * d_latent, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        B, K, D = latents.shape
        x = latents.reshape(B, -1)
        return self.probe(x).squeeze(-1)


def parse_answer(text):
    text = re.split(r"\nQ:|\nA:|\n---", text)[0]
    for p in [r"####\s*(-?\d+)", r"The answer is\s*(-?\d+)", r"answer is\s*(-?\d+)",
              r"=\s*(-?\d+)\s*$", r"Therefore,? .*? (-?\d+)", r"So,? .*? (-?\d+)"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None


def train_reading_head(perceiver, model, tokenizer, n_train_samples=200, n_epochs=30):
    """Train a reading head probe on Perceiver latents to predict perplexity."""
    print(f"\n{'='*60}")
    print(f"Training reading head (r=0.86 target) on {n_train_samples} GSM8K samples...")
    print(f"{'='*60}")

    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=CACHE_DIR)
    texts = [f"Q: {r['question']}\nA: {r['answer']}" for r in ds if len(r.get("question","")) > 50][:n_train_samples]
    print(f"  {len(texts)} samples")

    probe = UncertaintyProbe(n_latents=32, d_latent=128).to(DEVICE)
    print(f"  Probe: {sum(p.numel() for p in probe.parameters()):,} params")

    all_latents, all_ppl = [], []
    with torch.no_grad():
        for idx, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=256).to(DEVICE)
            seq_len = inputs.input_ids.shape[1]
            if seq_len < 10: continue

            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states
            logits = out.logits[0]
            probs = F.softmax(logits, dim=-1)
            target_ids = inputs.input_ids[0]

            for pos in range(1, seq_len - 2):
                token_ppl = -probs[pos-1, target_ids[pos]].log().item()
                h_pos = torch.stack([h[0, pos, :].float() for h in hs], dim=0)
                x = h_pos[:23].unsqueeze(0)
                ctx = h_pos[0].unsqueeze(0)

                _, latents = perceiver(x.to(DEVICE), ctx.to(DEVICE), return_latents=True)
                all_latents.append(latents[0].cpu())
                all_ppl.append(token_ppl)

            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(texts)}] {len(all_latents)} tokens")
                gc.collect(); torch.cuda.empty_cache()

    n_tokens = len(all_latents)
    print(f"  Collected {n_tokens} tokens")

    all_latents_t = torch.stack(all_latents).float()
    all_ppl_t = torch.tensor(all_ppl, dtype=torch.float32)

    perm = torch.randperm(n_tokens)
    n_train = int(n_tokens * 0.8)
    train_l = all_latents_t[perm[:n_train]]
    train_p = all_ppl_t[perm[:n_train]]
    test_l = all_latents_t[perm[n_train:]]
    test_p = all_ppl_t[perm[n_train:]]

    train_p_mean = train_p.mean()
    train_p_std = train_p.std() + 1e-8
    train_p_norm = (train_p - train_p_mean) / train_p_std

    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    best_corr = 0.0
    batch_size = 256

    for epoch in range(n_epochs):
        probe.train()
        perm2 = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm2[i:i+batch_size]
            pred = probe(train_l[idx].to(DEVICE))
            loss = F.mse_loss(pred, train_p_norm[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        all_pred, all_actual = [], []
        with torch.no_grad():
            for i in range(0, len(test_l), batch_size):
                pred = probe(test_l[i:i+batch_size].to(DEVICE)).cpu()
                all_pred.append(pred)
                all_actual.append(test_p[i:i+batch_size])
            pred = torch.cat(all_pred)
            actual = torch.cat(all_actual)
            corr = np.corrcoef(pred.numpy(), actual.numpy())[0, 1]

        if abs(corr) > abs(best_corr):
            best_corr = corr
            torch.save(probe.state_dict(), "best_reading_head.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  ep={epoch+1:2d} | corr={corr:.4f} | best={best_corr:.4f}")

    probe.load_state_dict(torch.load("best_reading_head.pt", map_location=DEVICE))
    print(f"  ✅ Reading head saved: r={best_corr:.4f}")
    return probe, train_p_mean, train_p_std


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--beta", type=float, default=2.0, help="Steepness of sigmoid gate")
    parser.add_argument("--tau", type=float, default=0.0, help="Threshold (z-scored perplexity)")
    parser.add_argument("--mode", choices=["baseline", "full", "ppl_modulated", "all"], default="all")
    parser.add_argument("--n-reading-samples", type=int, default=200)
    args = parser.parse_args()

    # --- Load model ---
    print("Loading model & tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()

    # --- Load Perceiver ---
    print("Loading Perceiver...")
    perceiver = ThoughtDiffusion(d_model=2048, n_layers=24, d_latent=128, n_latents=32, d_text_ctx=2048)
    perceiver.load_state_dict(torch.load(PERCEIVER_PATH, map_location="cpu"), strict=False)
    perceiver.to(DEVICE); perceiver.eval()
    for p in perceiver.parameters(): p.requires_grad = False

    # --- Load correction heads (Phase 3, prompt-trained, +13.8pp) ---
    print("Loading 3-head correction ensemble...")
    heads = nn.ModuleList([CorrectionHead().to(DEVICE) for _ in range(3)])
    for hi in range(3):
        ckpt = torch.load(f"best_head_{hi}.pt", map_location=DEVICE)
        heads[hi].load_state_dict(ckpt)
        heads[hi].eval()
        print(f"  head_{hi}: {sum(p.numel() for p in heads[hi].parameters()):,} params")

    # --- Train reading head ---
    probe, ppl_mean, ppl_std = train_reading_head(
        perceiver, model, tok,
        n_train_samples=args.n_reading_samples,
        n_epochs=30,
    )
    probe.eval()

    # --- Load test problems ---
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    problems = [r for r in ds if len(r["question"]) > 50][:args.n_test]

    examples = ("Q: Janet has 5. She buys 3. How many?\nA: Step 1: 5. Step 2: She buys 3. Step 3: 5+3=8.\nSo answer is 8.\n\n"
                "Q: Bakery sells 12/hr. How many in 8h?\nA: Step 1: 12/hr. Step 2: 8 hours. Step 3: 12x8=96.\nSo answer is 96.\n\n")

    def evaluate(mode, beta=None, tau=None):
        """Evaluate on test set with given correction mode."""
        correct, total = 0, 0
        gate_values = []
        t0 = time.time()

        for idx, prob in enumerate(problems):
            question = prob["question"]
            correct_ans = parse_answer(prob["answer"])
            prompt = f"{examples}Q: {question}\nA:"

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

                    # Get latents for this token
                    h_pos = torch.stack([h[0, -1, :].float() for h in hs], dim=0)
                    x = h_pos[:23].unsqueeze(0).to(DEVICE)
                    ctx = h_pos[0].unsqueeze(0).to(DEVICE)
                    _, latents = perceiver(x, ctx, return_latents=True)

                    if mode == "full":
                        off = sum(lm_head(h(latents).to(lm_head.weight.dtype)) for h in heads) / len(heads)
                        next_token = (orig_logits.float() + off[0].float()).argmax().item()
                    elif mode == "ppl_modulated":
                        ppl_pred = probe(latents.to(DEVICE)).item()
                        alpha = 1.0 / (1.0 + np.exp(-beta * (ppl_pred - tau)))
                        gate_values.append(alpha)
                        if alpha > 0.01:
                            off = sum(lm_head(h(latents).to(lm_head.weight.dtype)) for h in heads) / len(heads)
                            next_token = (orig_logits.float() + alpha * off[0].float()).argmax().item()

                    if next_token == tok.eos_token_id: break
                    full_ids = torch.cat([full_ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)

                gen_text = tok.decode(full_ids[0, plen:], skip_special_tokens=True)

            predicted = parse_answer(gen_text)
            if predicted is not None and correct_ans is not None and predicted == correct_ans:
                correct += 1
            total += 1

            if (idx + 1) % 10 == 0:
                gc.collect(); torch.cuda.empty_cache()
                avg_alpha = np.mean(gate_values) if gate_values else 0
                print(f"  [{idx+1}/{args.n_test}] acc={correct}/{total} ({100*correct/total:.0f}%) ᾱ={avg_alpha:.3f} {time.time()-t0:.0f}s")

        result = {
            "mode": mode,
            "correct": correct,
            "total": total,
            "accuracy": correct / max(total, 1),
        }
        if mode == "ppl_modulated":
            result["avg_alpha"] = float(np.mean(gate_values)) if gate_values else 0.0
            result["gate_values"] = {
                "min": float(np.min(gate_values)) if gate_values else 0,
                "max": float(np.max(gate_values)) if gate_values else 0,
                "mean": float(np.mean(gate_values)) if gate_values else 0,
                "median": float(np.median(gate_values)) if gate_values else 0,
            }
        return result

    results = {}

    if args.mode in ("baseline", "all"):
        print(f"\n{'='*60}")
        print("Mode: BASELINE (no correction)")
        print(f"{'='*60}")
        results["baseline"] = evaluate("baseline")
        print(f"  Accuracy: {results['baseline']['correct']}/{results['baseline']['total']} "
              f"({100*results['baseline']['accuracy']:.1f}%)")

    if args.mode in ("full", "all"):
        print(f"\n{'='*60}")
        print("Mode: FULL CORRECTION (α=1, always on)")
        print(f"{'='*60}")
        results["full"] = evaluate("full")
        print(f"  Accuracy: {results['full']['correct']}/{results['full']['total']} "
              f"({100*results['full']['accuracy']:.1f}%)")

    if args.mode in ("ppl_modulated", "all"):
        beta = args.beta
        tau = args.tau
        print(f"\n{'='*60}")
        print(f"Mode: PPL-MODULATED (β={beta}, τ={tau})")
        print(f"       α = σ({beta} · (ppl_pred - {tau}))")
        print(f"{'='*60}")
        results["ppl_modulated"] = evaluate("ppl_modulated", beta=beta, tau=tau)
        r = results["ppl_modulated"]
        print(f"  Accuracy: {r['correct']}/{r['total']} ({100*r['accuracy']:.1f}%)")
        print(f"  Avg α: {r['avg_alpha']:.4f}")
        print(f"  α range: [{r['gate_values']['min']:.4f}, {r['gate_values']['max']:.4f}]")
        print(f"  α median: {r['gate_values']['median']:.4f}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for mode, r in results.items():
        delta = ""
        if "baseline" in results and mode != "baseline":
            d = (r['accuracy'] - results['baseline']['accuracy'])
            delta = f" ({100*d:+.1f}pp vs baseline)"
        extra = ""
        if "avg_alpha" in r:
            extra = f" | ᾱ={r['avg_alpha']:.3f}"
        print(f"  {mode:15s}: {r['correct']:3d}/{r['total']:3d} ({100*r['accuracy']:.1f}%){delta}{extra}")

    # Save results
    import json
    with open("ppl_modulated_results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "gate_values"}
                   for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to ppl_modulated_results.json")


if __name__ == "__main__":
    main()
