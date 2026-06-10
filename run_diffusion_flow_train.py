#!/usr/bin/env python3
"""Train WeightDiffusion on diverse datasets with composed flag conditioning."""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.stream_fusion import PlainStreamExpert, FLAG_NAMES, N_FLAGS
from src.adapters.diffusion_weight_flow import WeightDiffusion, DiffusionFlowTrainer, augment_trajectories

MODEL_PATH = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub"),
    "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17",
)
DEVICE = "cuda"
RANK = 8
N_TRAJ = 40  # 40 trajectories × 3 noise levels × 20 steps ≈ 2400 training samples
STEPS_PER_TRAJ = 20


def load_diverse_texts(n_total: int) -> list[str]:
    """Load a diverse mix of texts from multiple datasets."""
    np.random.seed(42)
    texts = []
    sources = []

    # 1. Wikipedia (general knowledge)
    try:
        wiki = load_dataset("wikipedia", "20220301.simple", split="train", cache_dir=None)
        wiki_texts = [r["text"][:2000] for r in wiki if len(r.get("text", "")) > 500]
        np.random.shuffle(wiki_texts)
        n = max(1, n_total // 6)
        texts.extend(wiki_texts[:n])
        sources.extend(["wiki"] * n)
        print(f"  Wikipedia: {n} texts")
    except Exception as e:
        print(f"  Wikipedia: FAILED ({e})")

    # 2. Arxiv (scientific)
    try:
        parq_dir = "/run/media/filip/B522-875D/Datasets/hub/datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section"
        import pyarrow.parquet as pq
        arxiv_texts = []
        for pf in sorted(os.listdir(parq_dir))[:3]:
            tbl = pq.read_table(os.path.join(parq_dir, pf), columns=["article"])
            for art in tbl.column("article"):
                t = art.as_py()
                if len(t) > 500:
                    arxiv_texts.append(t[:2000])
        n = max(1, n_total // 6)
        arxiv_texts = arxiv_texts[:n]
        texts.extend(arxiv_texts)
        sources.extend(["arxiv"] * len(arxiv_texts))
        print(f"  Arxiv: {len(arxiv_texts)} texts")
    except Exception as e:
        print(f"  Arxiv: FAILED ({e})")

    # 3. Dolly (instruction)
    try:
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train", cache_dir=None)
        dolly_texts = []
        for r in dolly:
            t = f"{r.get('instruction', '')}\\n{r.get('response', '')}"
            if len(t) > 500:
                dolly_texts.append(t[:2000])
        np.random.shuffle(dolly_texts)
        n = max(1, n_total // 6)
        texts.extend(dolly_texts[:n])
        sources.extend(["dolly"] * n)
        print(f"  Dolly: {n} texts")
    except Exception as e:
        print(f"  Dolly: FAILED ({e})")

    # 4. Tiny Shakespeare (creative)
    try:
        sh = load_dataset("Trelis/tiny-shakespeare", split="train", cache_dir=None)
        sh_texts = [r["text"][:2000] for r in sh if len(r.get("text", "")) > 500][:50]
        n = max(1, n_total // 6)
        texts.extend(sh_texts[:n])
        sources.extend(["shakespeare"] * len(sh_texts[:n]))
        print(f"  Shakespeare: {n} texts")
    except Exception as e:
        print(f"  Shakespeare: FAILED ({e})")

    # 5. AG News (news)
    try:
        ag = load_dataset("ag_news", split="train", cache_dir=None)
        ag_texts = [r["text"][:2000] for r in ag if len(r.get("text", "")) > 500]
        np.random.shuffle(ag_texts)
        n = max(1, n_total // 6)
        texts.extend(ag_texts[:n])
        sources.extend(["ag_news"] * n)
        print(f"  AG News: {n} texts")
    except Exception as e:
        print(f"  AG News: FAILED ({e})")

    # 6. GSM8K (math)
    try:
        gsm = load_dataset("gsm8k", "main", split="train", cache_dir=None)
        gsm_texts = [f"{r['question']}\\n{r['answer']}" for r in gsm if len(r.get("question", "")) > 100]
        np.random.shuffle(gsm_texts)
        n = max(1, n_total // 6)
        texts.extend(gsm_texts[:n])
        sources.extend(["gsm8k"] * n)
        print(f"  GSM8K: {n} texts")
    except Exception as e:
        print(f"  GSM8K: FAILED ({e})")

    print(f"  Total: {len(texts)} texts from {len(set(sources))} sources")
    return texts


def main():
    print("=" * 60)
    print(f"  Diffusion Weight Flow — Diverse Data Training")
    print(f"  Model: Qwen3.5-0.8B, Rank={RANK}, Trajectories={N_TRAJ}")
    print("=" * 60)

    print("\nLoading diverse texts...")
    texts = load_diverse_texts(N_TRAJ)
    n_train = int(N_TRAJ * 0.75)
    train_texts = texts[:n_train]
    test_texts = texts[n_train:]

    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)

    target_name = "model.layers.3.self_attn.q_proj"
    target_module = model.get_submodule(target_name)
    orig_forward = target_module.forward
    d_in, d_out = target_module.in_features, target_module.out_features
    n_weights = PlainStreamExpert.absorb_dim(d_in, d_out, RANK)
    d_ctx = d_in + d_out
    print(f"Target: {target_name} (in={d_in}, out={d_out})")
    print(f"Weight dim: {n_weights}, Context dim: {d_ctx}")

    # Collect trajectories
    all_traj = []
    for traj_idx in range(len(train_texts)):
        text = train_texts[traj_idx]
        print(f"\nTrajectory {traj_idx + 1}/{len(train_texts)}", end=" ", flush=True)

        adapter = PlainStreamExpert(d_in, d_out, RANK, 64, 64, 8.0 / RANK).to(DEVICE, dtype=torch.bfloat16)
        def make_fwd(ada, orig):
            def fwd(x): return orig(x) + ada(x)
            return fwd
        target_module.forward = make_fwd(adapter, orig_forward)

        params = [p for n, p in adapter.named_parameters() if 'lora_A' in n or 'lora_B' in n]
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        labels = input_ids.clone()

        # Forward hook for hidden states
        cache = {}
        def fwd_hook(m, inp, out):
            cache["x"] = inp[0].detach().float().mean(dim=1)
        fwd_handle = target_module.register_forward_hook(fwd_hook)
        model(input_ids=input_ids)
        fwd_handle.remove()
        ctx_hidden = cache.get("x", torch.zeros(1, d_in))

        opt = torch.optim.Adam(params, lr=1e-3)
        weights, ctxs = [], []
        for step in range(STEPS_PER_TRAJ):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if step == 0:
                grad_ctx = torch.zeros(1, d_out)
                for n, p in adapter.named_parameters():
                    if 'lora_B' in n and p.grad is not None:
                        grad_ctx = p.grad.float().mean(dim=1).unsqueeze(0).cpu()
                        break

            flat = torch.cat([p.data.flatten() for p in params]).float().cpu()
            weights.append(flat)
            ctxs.append(torch.cat([ctx_hidden.cpu(), grad_ctx], dim=-1))

            if step % 5 == 0:
                print(f"{loss.item():.2f}", end=" ", flush=True)

        all_traj.append((weights, ctxs))
        target_module.forward = orig_forward
        del adapter, opt; gc.collect(); torch.cuda.empty_cache()
        print()

    # Augment trajectories
    n_noise = 5
    augmented = augment_trajectories(all_traj, n_noise_levels=n_noise)
    print(f"\nAugmented: {len(augmented)} samples ({len(all_traj)} traj × {STEPS_PER_TRAJ} steps × {n_noise} noise levels)")

    # Create and train diffusion model
    flow = WeightDiffusion(n_weights, d_latent=64, n_latents=16, d_ctx=d_ctx)
    trainer = DiffusionFlowTrainer(flow, lr=1e-3, device=DEVICE)

    print(f"\nTraining diffusion flow...")
    t0 = time.time()
    for epoch in range(15):
        losses, ld, lf = [], [], []
        np.random.shuffle(augmented)
        for d in augmented:
            l, d_l, f_l = trainer.train_step(
                d['clean'].unsqueeze(0).to(DEVICE),
                d['next'].unsqueeze(0).to(DEVICE),
                d['flags'].unsqueeze(0).to(DEVICE),
                d['ctx'].to(DEVICE),
                torch.tensor([[d['t_noise']]]).to(DEVICE),
                torch.tensor([[d['t_flow']]]).to(DEVICE),
            )
            losses.append(l); ld.append(d_l); lf.append(f_l)
        dt = time.time() - t0
        print(f"  Epoch {epoch}: total={np.mean(losses):.6f} diff={np.mean(ld):.6f} flow={np.mean(lf):.6f} ({dt:.0f}s)")

    # Save model
    flow.cpu()
    torch.save(flow.state_dict(), "diffusion_weight_flow.pt")
    meta = {
        "n_weights": n_weights, "d_ctx": d_ctx,
        "n_train": len(train_texts), "n_test": len(test_texts),
        "n_augmented": len(augmented), "final_loss": float(np.mean(losses)),
    }
    with open("diffusion_weight_flow_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nModel saved: diffusion_weight_flow.pt ({os.path.getsize('diffusion_weight_flow.pt')/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
