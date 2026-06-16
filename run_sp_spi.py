#!/usr/bin/env python3 -u
"""Single-Pass Steering Potential Index (SP-SPI).

Combines curvature κ[l] with attention gain g[l] into a single
per-layer predictor: SPI[l] = g[l] / sqrt(1 + κ[l]²)

g[l] computed from softmax Jacobian norm = sqrt(1 - sum(attn²))
κ[l] from pre-computed trajectory analysis.

Validates against known FA-layer Δaccuracy on Qwen3.5-0.8B.
"""
from __future__ import annotations

import gc, glob, json, math, os, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
N_PROBLEMS = 50


def compute_attention_gain(model, tok, problems, fa_layers, n_kv_heads, head_dim):
    """Compute attention gain g[l] for FA layers from attention entropy.

    g[l] = mean_h( sqrt(1 - sum_k attn(q,k)²) ) — softmax Jacobian norm.
    """
    all_gains = {l: [] for l in fa_layers}

    for idx, prob in enumerate(problems):
        msgs = [{"role": "user", "content": f'Q: {prob["question"]}\nA:'}]
        inputs = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          return_tensors="pt").input_ids.to(DEVICE)

        with torch.no_grad():
            out = model(inputs, output_attentions=True)

        for attn_idx, li in enumerate(fa_layers):
            if attn_idx < len(out.attentions) and out.attentions[attn_idx] is not None:
                attn = out.attentions[attn_idx][0]  # (n_heads, seq_len, seq_len)
                attn_last = attn[:, -1, :]  # (n_heads, seq_len)
                jacobian_norm = torch.sqrt(1 - (attn_last ** 2).sum(dim=-1))
                gain = jacobian_norm.mean().item()
                all_gains[li].append(gain)

        if (idx + 1) % 10 == 0:
            print(f"  attention [{idx+1}/{len(problems)}]", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    return {l: float(np.mean(vals)) for l, vals in all_gains.items()}


def load_kappa():
    """Load pre-computed κ values from trajectory data."""
    DATA_DIR = "/home/filip/Projects/Personal/AI/RankAdaptation/data/qwen35_08b_trajs"
    files = sorted(glob.glob(os.path.join(DATA_DIR, "batch_*.pt")))
    all_kappa = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        hs = d["hidden_seqs"].float()
        v = hs[:, 1:] - hs[:, :-1]
        a = v[:, 1:] - v[:, :-1]
        v_norm = v[:, :-1].norm(dim=-1)
        a_norm = a.norm(dim=-1)
        all_kappa.append(a_norm / (v_norm + 1e-8))
        del d, hs
    kappa = torch.cat(all_kappa).mean(dim=0).numpy()
    return kappa  # (22,) for layers 1..22


def main():
    t0 = time.time()

    print("Loading Qwen3.5-0.8B for attention extraction...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16, device_map=DEVICE,
                                                  attn_implementation="eager")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    cfg = model.config.get_text_config()
    n_layers = cfg.num_hidden_layers
    fa_layers = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    gdn_layers = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
    print(f"  FA layers: {fa_layers}, GDN layers: {gdn_layers}", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50][:N_PROBLEMS]
    print(f"  Running {len(problems)} problems for attention stats...", flush=True)

    gains = compute_attention_gain(model, tok, problems, fa_layers, cfg.num_key_value_heads, cfg.head_dim)
    print(f"\n  Attention gains per FA layer:", flush=True)
    for l in sorted(gains.keys()):
        print(f"    L{l}: g={gains[l]:.4f}", flush=True)

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nLoading pre-computed κ values...", flush=True)
    kappa = load_kappa()  # (22,) for layers 1..22
    print(f"  κ shape: {kappa.shape}", flush=True)

    known_deltas = {3: -0.033, 7: -0.100, 11: -0.033, 15: -0.267, 19: -0.167}

    print(f"\n{'='*70}", flush=True)
    print(f"SP-SPI RESULTS — Qwen3.5-0.8B", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Layer':>6} {'Type':>4} {'κ':>8} {'g':>8} {'SPI':>10} {'SPI_norm':>10} {'Δacc':>8} {'Match':>8}", flush=True)
    print(f"  {'-'*6} {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8}", flush=True)

    results_fa = []
    for li in fa_layers:
        if li == 0:
            continue
        if not (1 <= li <= 22):
            continue  # skip layers without κ
        k = float(kappa[li - 1])
        g = gains.get(li, 0.5)
        spi = g / math.sqrt(1 + k ** 2)
        d = known_deltas.get(li)
        results_fa.append((li, k, g, spi, d))

    # Normalize SPI to [0, 1] for FA layers
    spi_vals = [r[3] for r in results_fa]
    spi_min, spi_max = min(spi_vals), max(spi_vals)
    spi_range = spi_max - spi_min if spi_max > spi_min else 1.0

    for li, k, g, spi, d in results_fa:
        spi_n = (spi - spi_min) / spi_range
        d_str = f"{d*100:+.0f}pp" if d is not None else "  N/A"
        match = "✓" if (d is not None and ((spi_n < 0.5 and d < 0) or (spi_n >= 0.5 and d >= 0))) else "?"
        print(f"  L{li:3d}  {'FA':>4}  {k:8.4f}  {g:8.4f}  {spi:10.4f}  {spi_n:10.4f}  {d_str:>8}  {match:>8}", flush=True)

    # Correlation
    valid = [(r[3], known_deltas[r[0]]) for r in results_fa if r[0] in known_deltas]
    if len(valid) >= 3:
        spis, deltas = zip(*valid)
        r = np.corrcoef(list(spis), list(deltas))[0, 1]
        print(f"\n  ρ(SPI, Δacc) = {r:.4f} (FA layers)", flush=True)

    # Also compute for GDN layers (g placeholder)
    print(f"\n  GDN layers (no attention data — using g=0.5 default):", flush=True)
    for li in range(1, n_layers - 1):
        if li in fa_layers:
            continue
        k = float(kappa[li - 1])
        g = 0.5
        spi = g / math.sqrt(1 + k ** 2)
        print(f"    L{li:2d}  κ={k:.4f}  SPI={spi:.4f}", flush=True)

    out = {
        "model": "Qwen3.5-0.8B",
        "attention_gains": gains,
        "kappa": {str(i+1): float(kappa[i]) for i in range(len(kappa))},
        "spi_fa": {str(l): g / math.sqrt(1 + float(kappa[l-1])**2)
                   for l in fa_layers if 1 <= l <= 22
                   for g in [gains.get(l, 0.5)]},
        "correlation": float(r) if len(valid) >= 3 else None,
    }
    with open("sp_spi_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to sp_spi_results.json", flush=True)
    print(f"Total time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
