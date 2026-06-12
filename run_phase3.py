#!/usr/bin/env python3
"""Phase 3: DiMAE multi-head ensemble.

3 independent correction heads, averaged at inference.
No logit storage needed (avoids OOM from 151936-dim tensors).
"""
from __future__ import annotations

import gc, os, sys, time
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
MAX_LEN, N_HEADS, BS = 256, 3, 32


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


def main():
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

    heads = nn.ModuleList([CorrectionHead().to(DEVICE) for _ in range(N_HEADS)])
    total_p = sum(sum(p.numel() for p in h.parameters()) for h in heads)
    print(f"{N_HEADS} heads × {total_p // N_HEADS:,} = {total_p:,} params")

    examples = ("Q: Janet has 5. She buys 3. How many?\nA: Step 1: 5. Step 2: She buys 3. Step 3: 5+3=8.\nSo answer is 8.\n\n"
                "Q: Bakery sells 12/hr. How many in 8h?\nA: Step 1: 12/hr. Step 2: 8 hours. Step 3: 12x8=96.\nSo answer is 96.\n\n")
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    all_p = [r for r in ds if len(r["question"]) > 50]
    test_p, train_p = all_p[:50], all_p[50:250]

    # Collect latents and targets only (NO logits — they're too large)
    print(f"Collecting {len(train_p)} training problems...")
    all_la, all_tg = [], []
    for prob in train_p:
        txt = f"{examples}Q: {prob['question']}\nA:"
        inp = tok(txt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        hs, ids = out.hidden_states, inp.input_ids[0]
        for pos in range(1, hs[0].shape[1] - 2):
            h_pos = torch.stack([h[0, pos, :].float() for h in hs], dim=0)
            x, ctx = h_pos[:23].unsqueeze(0), h_pos[0].unsqueeze(0)
            _, lat = per(x.to(DEVICE), ctx.to(DEVICE), return_latents=True)
            all_la.append(lat[0].cpu().half())  # use float16 to save RAM
            all_tg.append(ids[pos + 1].item())
        if len(all_la) % 500 == 0:
            rss = torch.cuda.memory_allocated() / 1e9
            print(f"  {len(all_la)} tokens | {rss:.2f}GB VRAM")
            gc.collect(); torch.cuda.empty_cache()
    print(f"  Total: {len(all_la)} token-latent pairs")

    # Convert to tensors (float16 for latents saves 50% RAM vs float32)
    tr_l = torch.stack(all_la).float()  # convert back for training
    tr_t = torch.tensor(all_tg)
    print(f"  Shape: {tr_l.shape}, RAM: {tr_l.numel()*4/1e9:.2f}GB")
    del all_la, all_tg; gc.collect()

    nv = max(1, len(tr_l) // 10)
    perm = np.random.permutation(len(tr_l))
    va_l, va_t = tr_l[perm[:nv]], tr_t[perm[:nv]]
    tr_l, tr_t = tr_l[perm[nv:]], tr_t[perm[nv:]]
    print(f"  Train: {len(tr_t)}, Val: {len(va_t)}")

    # Training
    opts = [torch.optim.AdamW(h.parameters(), lr=3e-4, weight_decay=1e-4) for h in heads]
    best_val, t0 = 0, time.time()

    for ep in range(25):
        for hidx, (head, opt) in enumerate(zip(heads, opts)):
            head.train()
            ph = np.random.permutation(len(tr_l))
            for i in range(0, len(tr_l), BS):
                idx = ph[i:i+BS]
                h_code = head(tr_l[idx].to(DEVICE))
                offset = lm_head(h_code.to(lm_head.weight.dtype))
                loss = F.cross_entropy(offset, tr_t[idx].to(DEVICE))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()

        # Ensemble validation
        for h in heads: h.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for i in range(0, len(va_l), BS):
                lat = va_l[i:i+BS].to(DEVICE)
                tgt = va_t[i:i+BS].to(DEVICE)
                off = sum(lm_head(h(lat).to(lm_head.weight.dtype)) for h in heads) / N_HEADS
                vc += (off.argmax(dim=-1) == tgt).sum().item(); vt += len(tgt)
        vacc = vc / vt
        if vacc > best_val:
            best_val = vacc
            for hi, h in enumerate(heads): torch.save(h.state_dict(), f"best_head_{hi}.pt")
        if (ep+1) % 5 == 0: print(f"  ep={ep+1:2d} val_acc={100*vacc:.1f}% {time.time()-t0:.0f}s")

    # Test
    for hi, h in enumerate(heads): h.load_state_dict(torch.load(f"best_head_{hi}.pt", map_location=DEVICE))
    print(f"\n{'='*60}")
    print(f"Test: {N_HEADS}-head ensemble on {len(test_p)} problems")
    print(f"{'='*60}")

    bc, cc, total = 0, 0, 0
    for prob in test_p:
        txt = f"{examples}Q: {prob['question']}\nA:"
        inp = tok(txt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        hs = out.hidden_states
        for pos in range(1, hs[0].shape[1] - 2):
            tid = inp.input_ids[0, pos + 1].item()
            h_pos = torch.stack([h[0, pos, :].float() for h in hs], dim=0)
            x, ctx = h_pos[:23].unsqueeze(0), h_pos[0].unsqueeze(0)
            _, lat = per(x.to(DEVICE), ctx.to(DEVICE), return_latents=True)
            off = sum(lm_head(h(lat).to(lm_head.weight.dtype)) for h in heads) / N_HEADS
            cl = out.logits[0, pos].float() + off[0].float()
            if out.logits[0, pos].argmax().item() == tid: bc += 1
            if cl.argmax().item() == tid: cc += 1
            total += 1
        gc.collect(); torch.cuda.empty_cache()

    print(f"  Baseline:  {bc}/{total} ({100*bc/total:.1f}%)")
    print(f"  Corrected: {cc}/{total} ({100*cc/total:.1f}%)")
    print(f"  Delta:     {cc-bc}/{total} ({100*(cc-bc)/total:+.1f}pp)")
    print(f"  Phase 1: +12.6pp (1 head). Phase 3: {100*(cc-bc)/total:+.1f}pp ({N_HEADS} heads)")


if __name__ == "__main__":
    main()
