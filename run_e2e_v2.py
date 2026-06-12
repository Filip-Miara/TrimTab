#!/usr/bin/env python3
"""Collect generation-time training data, train correction heads, evaluate.

Distribution shift fix: train correction heads on hidden states from
DURING autoregressive generation, not just prompt forward passes.
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

    heads = nn.ModuleList([CorrectionHead().to(DEVICE) for _ in range(3)])
    print(f"3 heads × {sum(p.numel() for p in heads[0].parameters()):,} = {sum(sum(p.numel() for p in h.parameters()) for h in heads):,} params")

    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    all_p = [r for r in ds if len(r["question"]) > 50]
    test_p, train_p = all_p[:50], all_p[50:250]

    examples = ("Q: Janet has 5. She buys 3. How many?\nA: Step 1: 5. Step 2: She buys 3. Step 3: 5+3=8.\nSo answer is 8.\n\n"
                "Q: Bakery sells 12/hr. How many in 8h?\nA: Step 1: 12/hr. Step 2: 8 hours. Step 3: 12x8=96.\nSo answer is 96.\n\n")

    # Collect generation-time training data
    print(f"Collecting generation-time data from {len(train_p)} problems...")
    all_la, all_tg = [], []
    for prob in train_p:
        prompt = f"{examples}Q: {prob['question']}\nA:"
        full_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids
        prompt_len = full_ids.shape[1]

        for step in range(MAX_GEN):
            with torch.no_grad():
                fwd = model(full_ids, output_hidden_states=True)

            # Get next token from original logits
            next_tok = fwd.logits[0, -1, :].argmax().item()
            if next_tok == tok.eos_token_id:
                break

            # Record latents for this step
            hs = fwd.hidden_states
            h_pos = torch.stack([h[0, -1, :].float() for h in hs], dim=0)
            x = h_pos[:23].unsqueeze(0).to(DEVICE)
            ctx = h_pos[0].unsqueeze(0).to(DEVICE)
            _, lat = per(x, ctx, return_latents=True)
            all_la.append(lat[0].cpu().half())
            all_tg.append(next_tok)

            full_ids = torch.cat([full_ids, torch.tensor([[next_tok]], device=DEVICE)], dim=1)

            if step == 0 and len(all_la) % 1000 < 3:
                print(f"  first token: '{tok.decode(next_tok)}' (id={next_tok})")

        if len(train_p) >= 20 and len(all_la) % 1000 < 50:
            print(f"  collected {len(all_la)} tokens ({len(all_la)//len(train_p)}/prob avg)")
            gc.collect(); torch.cuda.empty_cache()

    print(f"  Total: {len(all_la)} token-latent pairs")

    # Train
    tr_l = torch.stack(all_la).float()
    tr_t = torch.tensor(all_tg)
    del all_la, all_tg; gc.collect()

    nv = max(1, len(tr_l) // 10)
    perm = np.random.permutation(len(tr_l))
    va_l, va_t = tr_l[perm[:nv]], tr_t[perm[:nv]]
    tr_l, tr_t = tr_l[perm[nv:]], tr_t[perm[nv:]]
    print(f"  Train: {len(tr_t)}, Val: {len(va_t)}")

    opts = [torch.optim.AdamW(h.parameters(), lr=3e-4, weight_decay=1e-4) for h in heads]
    best_val, t0, bs = 0.0, time.time(), 32

    for ep in range(20):
        for hidx, (head, opt) in enumerate(zip(heads, opts)):
            head.train()
            ph = np.random.permutation(len(tr_l))
            for i in range(0, len(tr_l), bs):
                idx = ph[i:i+bs]
                off = lm_head(head(tr_l[idx].to(DEVICE)).to(lm_head.weight.dtype))
                loss = F.cross_entropy(off, tr_t[idx].to(DEVICE))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()

        # Validate ensemble
        for h in heads: h.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for i in range(0, len(va_l), bs):
                off = torch.stack([lm_head(h(va_l[i:i+bs].to(DEVICE)).to(lm_head.weight.dtype)) for h in heads]).mean(0)
                vc += (off.argmax(dim=-1) == va_t[i:i+bs].to(DEVICE)).sum().item()
                vt += min(bs, len(va_l) - i)
        vacc = vc / vt
        if vacc > best_val: best_val = vacc
        if (ep+1) % 5 == 0: print(f"  ep={ep+1:2d} val_acc={100*vacc:.1f}% best={100*best_val:.1f}% {time.time()-t0:.0f}s")

    for hi, h in enumerate(heads):
        torch.save(h.state_dict(), f"gen_head_{hi}.pt")

    # Evaluate on test
    print(f"\n{'='*60}")
    print("Test: generation-trained correction heads")
    print(f"{'='*60}")

    bc, cc, total = 0, 0, 0
    for prob in test_p:
        prompt = f"{examples}Q: {prob['question']}\nA:"
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        hs = out.hidden_states
        for pos in range(1, hs[0].shape[1] - 2):
            tid = inp.input_ids[0, pos + 1].item()
            h_pos = torch.stack([h[0, pos, :].float() for h in hs], dim=0)
            x, ctx = h_pos[:23].unsqueeze(0), h_pos[0].unsqueeze(0)
            _, lat = per(x.to(DEVICE), ctx.to(DEVICE), return_latents=True)
            off = torch.stack([lm_head(h(lat).to(lm_head.weight.dtype)) for h in heads]).mean(0)
            cl = out.logits[0, pos].float() + off[0].float()
            if out.logits[0, pos].argmax().item() == tid: bc += 1
            if cl.argmax().item() == tid: cc += 1
            total += 1
        gc.collect(); torch.cuda.empty_cache()

    print(f"  Baseline:  {bc}/{total} ({100*bc/total:.1f}%)")
    print(f"  Corrected: {cc}/{total} ({100*cc/total:.1f}%)")
    print(f"  Delta:     {cc-bc}/{total} ({100*(cc-bc)/total:+.1f}pp)")
    print(f"  Prompt-trained heads: +13.8pp. Gen-trained heads: {100*(cc-bc)/total:+.1f}pp")

    # Also generate full answers with corrected head
    print(f"\n  Full answer generation with corrections:")
    cor = 0
    for idx, prob in enumerate(test_p[:20]):
        prompt = f"{examples}Q: {prob['question']}\nA:"
        full_ids = tok(prompt, return_tensors="pt").to(DEVICE).input_ids
        correct_ans = parse_answer(prob["answer"])
        plen = full_ids.shape[1]

        for step in range(MAX_GEN):
            with torch.no_grad():
                fwd = model(full_ids, output_hidden_states=True)
            hs = fwd.hidden_states
            h_pos = torch.stack([h[0, -1, :].float() for h in hs], dim=0)
            x, ctx = h_pos[:23].unsqueeze(0), h_pos[0].unsqueeze(0)
            _, lat = per(x.to(DEVICE), ctx.to(DEVICE), return_latents=True)
            off = torch.stack([lm_head(h(lat).to(lm_head.weight.dtype)) for h in heads]).mean(0)
            cor_logits = fwd.logits[0, -1, :].float() + off[0].float()
            nt = cor_logits.argmax().item()
            if nt == tok.eos_token_id: break
            full_ids = torch.cat([full_ids, torch.tensor([[nt]], device=DEVICE)], dim=1)
            if step >= MAX_GEN - 1: break

        gen_text = tok.decode(full_ids[0, plen:], skip_special_tokens=True)
        pred = parse_answer(gen_text)
        is_cor = pred is not None and correct_ans is not None and pred == correct_ans
        if is_cor: cor += 1
        gc.collect(); torch.cuda.empty_cache()

    print(f"  Corrected generation accuracy: {cor}/20 ({100*cor/20:.0f}%)")
    print(f"  (Baseline was: 40.0% on full 50)")


if __name__ == "__main__":
    main()
