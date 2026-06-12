#!/usr/bin/env python3
"""Reading heads: linear probes on Perceiver latents for token uncertainty.

Hypothesis: the Perceiver's latent representations encode information about
the model's internal state, including uncertainty. A linear probe trained
on latents should predict token-level perplexity above chance.

Architecture: freeze Perceiver → extract latents → linear probe → perplexity
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.thought_diffusion import ThoughtDiffusion

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
PERCEIVER_PATH = "best_perceiver.pt"
CACHE_DIR = "/run/media/filip/B522-875D/Datasets/hub"
N_SAMPLES = 500
MAX_LENGTH = 256





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


def main():
    print(f"Loading GSM8K for data...")
    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=CACHE_DIR)
    texts = [f"Q: {r['question']}\nA: {r['answer']}" for r in ds if len(r.get("question","")) > 50][:N_SAMPLES]
    print(f"  {len(texts)} samples")

    print(f"Loading model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    n_layers = getattr(cfg, 'num_hidden_layers', 24)
    d_model = getattr(cfg, 'hidden_size', 2048)

    print(f"Loading Perceiver...")
    perceiver = ThoughtDiffusion(d_model=d_model, n_layers=n_layers, d_latent=128, n_latents=32, d_text_ctx=d_model)
    state = torch.load(PERCEIVER_PATH, map_location=DEVICE)
    perceiver.load_state_dict(state, strict=False)
    perceiver.to(DEVICE)
    perceiver.eval()
    for p in perceiver.parameters(): p.requires_grad = False
    print(f"  Frozen ({sum(p.numel() for p in perceiver.parameters()):,} params)")

    probe = UncertaintyProbe(n_latents=32, d_latent=128).to(DEVICE)
    n_probe_params = sum(p.numel() for p in probe.parameters())
    print(f"  Probe: {n_probe_params:,} params")

    all_latents, all_ppl = [], []

    with torch.no_grad():
        for idx, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=MAX_LENGTH).to(DEVICE)
            seq_len = inputs.input_ids.shape[1]
            if seq_len < 10: continue

            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states
            logits = out.logits[0]  # (S, V)
            probs = F.softmax(logits, dim=-1)
            target_ids = inputs.input_ids[0]

            for pos in range(1, seq_len - 2):
                token_ppl = -probs[pos-1, target_ids[pos]].log().item()
                h_pos = torch.stack([h[0, pos, :].float() for h in hs], dim=0)
                x = h_pos[:n_layers-1].unsqueeze(0)
                ctx = h_pos[0].unsqueeze(0)

                _, latents = perceiver(x.to(DEVICE), ctx.to(DEVICE), return_latents=True)
                all_latents.append(latents[0].cpu())
                all_ppl.append(token_ppl)

            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(texts)}] collected {len(all_latents)} tokens")
                gc.collect()
                torch.cuda.empty_cache()

    n_tokens = len(all_latents)
    print(f"\nCollected {n_tokens} tokens with latent vectors")
    if n_tokens < 100:
        print("Too few tokens, aborting")
        return

    all_latents_t = torch.stack(all_latents).float()
    all_ppl_t = torch.tensor(all_ppl, dtype=torch.float32)

    perm = torch.randperm(n_tokens)
    n_train = int(n_tokens * 0.8)

    train_l = all_latents_t[perm[:n_train]]
    train_p = all_ppl_t[perm[:n_train]]
    test_l = all_latents_t[perm[n_train:]]
    test_p = all_ppl_t[perm[n_train:]]

    train_p_norm = (train_p - train_p.mean()) / (train_p.std() + 1e-8)

    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    best_corr = 0.0

    batch_size = 256
    for epoch in range(50):
        probe.train()
        perm2 = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm2[i:i+batch_size]
            pred = probe(train_l[idx].to(DEVICE))
            loss = F.mse_loss(pred, train_p_norm[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()

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

        if abs(corr) > abs(best_corr): best_corr = corr
        if (epoch + 1) % 10 == 0:
            print(f"  ep={epoch+1:2d} | corr={corr:.4f}")

    print(f"\n{'='*60}")
    print(f"Best test correlation: {best_corr:.4f}")
    if abs(best_corr) > 0.2:
        print("✅ Perceiver latents encode uncertainty signal!")
    elif abs(best_corr) > 0.05:
        print("⚠️  Weak uncertainty signal")
    else:
        print("❌ Perceiver latents don't encode uncertainty")
    print(f"{'='*60}")

    results = {"best_correlation": float(best_corr), "n_tokens": n_tokens}
    with open("reading_heads_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
