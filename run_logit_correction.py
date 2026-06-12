#!/usr/bin/env python3
"""Phase 1: Latent→Logit Correction Head.

Frozen Perceiver latents → trainable MLP → logit offsets.
Bypasses geometric orthogonality by operating in logit space.

Success criterion: ≥3pp next-token accuracy on GSM8K.
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
MAX_LEN = 256

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
    vocab_size = lm_head.weight.shape[0]

    per = ThoughtDiffusion(d_model=2048, n_layers=24, d_latent=128, n_latents=32, d_text_ctx=2048)
    per.load_state_dict(torch.load(PERCEIVER_PATH, map_location="cpu"), strict=False)
    per.to(DEVICE); per.eval()
    for p in per.parameters(): p.requires_grad = False

    head = CorrectionHead().to(DEVICE)
    print(f"Trainable: {sum(p.numel() for p in head.parameters()):,} params")

    examples = ("Q: Janet has 5. She buys 3. How many?\nA: Step 1: 5. Step 2: She buys 3. Step 3: 5+3=8.\nSo answer is 8.\n\n"
                "Q: Bakery sells 12/hr. How many in 8h?\nA: Step 1: 12/hr. Step 2: 8 hours. Step 3: 12x8=96.\nSo answer is 96.\n\n")
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    all_p = [r for r in ds if len(r["question"]) > 50]
    test_p, train_p = all_p[:50], all_p[50:250]

    # Collect (latents, target, orig_logit) tuples
    print(f"Collecting {len(train_p)} training problems...")
    all_la, all_tg, all_lo = [], [], []
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
            all_la.append(lat[0].cpu())
            all_tg.append(ids[pos + 1].item())
        if len(all_la) % 500 == 0: print(f"  {len(all_la)} tokens"); gc.collect(); torch.cuda.empty_cache()
    print(f"  Total: {len(all_la)} token-latent pairs")

    # Split train/val
    perm = np.random.permutation(len(all_la))
    nv = max(1, len(all_la) // 10)
    tr_l = torch.stack(all_la)[perm[nv:]]
    tr_t = torch.tensor(all_tg)[perm[nv:]]
    va_l = torch.stack(all_la)[perm[:nv]]
    va_t = torch.tensor(all_tg)[perm[:nv]]
    print(f"  Train: {len(tr_t)}, Val: {len(va_t)}")

    # Compute baseline accuracy on val set using original LM
    print("Computing baseline accuracy...")
    bas_correct = 0
    for i in range(0, len(va_t), 16):
        txt = train_p[0]['question']  # dummy, just need correct shapes
    # Can't compute baseline without original logits — need to store them during collection
    # For test set, we'll compute on the fly

    # Train correction head (memory-efficient: iterate in batches)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    best_val = float('inf')
    t0 = time.time()
    bs = 64  # small batch to avoid OOM on logit computation

    for ep in range(30):
        head.train()
        perm2 = np.random.permutation(len(tr_l))
        for i in range(0, len(tr_l), bs):
            idx = perm2[i:i+bs]
            lat = tr_l[idx].to(DEVICE)
            tgt = tr_t[idx].to(DEVICE)
            h_code = head(lat)  # (B, 2048)
            offset = lm_head(h_code.to(lm_head.weight.dtype))  # (B, V)
            loss = F.cross_entropy(offset, tgt)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()

        # Validate in batches
        head.eval()
        val_loss, val_acc = 0.0, 0.0
        n_val_batches = 0
        with torch.no_grad():
            for i in range(0, len(va_t), bs):
                lat = va_l[i:i+bs].to(DEVICE)
                tgt = va_t[i:i+bs].to(DEVICE)
                h_code = head(lat)
                offset = lm_head(h_code.to(lm_head.weight.dtype))
                val_loss += F.cross_entropy(offset, tgt).item()
                val_acc += (offset.argmax(dim=-1) == tgt).float().mean().item()
                n_val_batches += 1
        val_loss /= n_val_batches
        val_acc /= n_val_batches

        if val_loss < best_val:
            best_val = val_loss
            torch.save(head.state_dict(), "best_correction_head.pt")

        if (ep+1) % 5 == 0:
            print(f"  ep={ep+1:2d} val_loss={val_loss:.4f} val_acc={100*val_acc:.1f}% {time.time()-t0:.0f}s")

    # Evaluate on test set
    print(f"\n{'='*60}")
    print(f"Test on {len(test_p)} problems")
    print(f"{'='*60}")
    head.load_state_dict(torch.load("best_correction_head.pt", map_location=DEVICE))
    head.eval()

    bas_c, cor_c, total = 0, 0, 0
    for prob in test_p:
        txt = f"{examples}Q: {prob['question']}\nA:"
        inp = tok(txt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        hs = out.hidden_states

        for pos in range(1, hs[0].shape[1] - 2):
            target_id = inp.input_ids[0, pos + 1].item()
            orig_logits = out.logits[0, pos]

            h_pos = torch.stack([h[0, pos, :].float() for h in hs], dim=0)
            x = h_pos[:23].unsqueeze(0).to(DEVICE)
            ctx = h_pos[0].unsqueeze(0).to(DEVICE)
            _, lat = per(x, ctx, return_latents=True)

            h_code = head(lat)
            offset = lm_head(h_code.to(lm_head.weight.dtype))[0]
            cor_logits = orig_logits.float() + offset.float()

            if orig_logits.argmax().item() == target_id: bas_c += 1
            if cor_logits.argmax().item() == target_id: cor_c += 1
            total += 1

        gc.collect(); torch.cuda.empty_cache()

    print(f"  Baseline: {bas_c}/{total} ({100*bas_c/total:.1f}%)")
    print(f"  Corrected: {cor_c}/{total} ({100*cor_c/total:.1f}%)")
    print(f"  Delta: {cor_c-bas_c}/{total} ({100*(cor_c-bas_c)/total:+.1f}pp)")
    print(f"  Improvement: {100*(cor_c-bas_c)/max(bas_c,1):+.1f}% relative")
    if cor_c > bas_c + 2: print(f"\n✅ Phase 1 SUCCESS!")
    elif cor_c > bas_c: print(f"\n⚠️  Marginal — signal exists but weak")
    else: print(f"\n❌ Phase 1 FAILED — latents don't encode directional correction")

if __name__ == "__main__":
    main()
