#!/usr/bin/env python3 -u
"""Logit-Delta Probe on Qwen2.5-7B.

Predicts per-layer steering polarity from a SINGLE forward pass.
For each layer l, computes:
  Δlogit(correct) = log P(correct|steered) - log P(correct|original)

If Δlogit(correct) > 0 for layer l → the TT's predicted velocity at l
increases the model's confidence in the correct answer at the next token
→ layer is a trim-tab. If < 0 → death layer.

Uses existing trajectory data — 0 generation needed.
"""
from __future__ import annotations

import gc, glob, json, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
TT_PATH = "best_gen_tt_7b.pt"
DATA_DIR = "/run/media/filip/B522-875D/Datasets/project_data/qwen25_7b_gen_trajs"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
N_LAYERS = 28
D_MODEL = 3584
ALPHA = 0.1


def main():
    t0 = time.time()

    print("Loading Qwen2.5-7B (4-bit)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  quantization_config=quant, device_map=DEVICE)
    model.eval()
    lm_head = model.lm_head or model.get_output_embeddings()

    # Move LM head to CPU to avoid GPU OOM (model takes 7.4GB of 8GB)
    lm_head_cpu = lm_head.to("cpu")
    del lm_head  # free GPU reference
    lm_dtype = lm_head_cpu.weight.dtype
    print(f"  LM head: {lm_head_cpu.weight.shape} (CPU)", flush=True)

    print(f"Loading TT from {TT_PATH}...", flush=True)
    tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                                n_positions=N_LAYERS, d_input=D_MODEL).to(DEVICE)
    tt.load_state_dict(torch.load(TT_PATH, map_location="cpu"), strict=False)
    tt.eval()
    for p in tt.parameters(): p.requires_grad = False

    files = sorted(glob.glob(os.path.join(DATA_DIR, "gen_trajs_7b_batch_*.pt")))
    d = torch.load(files[0], map_location="cpu")
    hs = d["hidden_seqs"].float()
    token_ids = d["token_ids"]
    print(f"  Trajectories: {len(hs)}", flush=True)
    del d

    bs = 8
    n = len(hs)
    all_deltas = {l: [] for l in range(N_LAYERS - 1)}

    for start in range(0, n, bs):
        end = min(start + bs, n)
        h_batch = hs[start:end].to(DEVICE)
        tok_batch = token_ids[start:end]

        with torch.no_grad():
            v_pred = tt(h_batch)

        for li in range(N_LAYERS - 1):
            h_actual = h_batch[:, li + 1, :].float()
            v_l = v_pred[:, li, :].float()
            h_steered = h_actual + ALPHA * v_l

            # Logit computation on CPU to avoid GPU OOM
            lo = lm_head_cpu(h_actual.cpu().to(lm_dtype))
            ls = lm_head_cpu(h_steered.cpu().to(lm_dtype))

            b = end - start
            lp_orig = lo[range(b), tok_batch] - torch.logsumexp(lo, dim=-1)
            lp_steer = ls[range(b), tok_batch] - torch.logsumexp(ls, dim=-1)
            all_deltas[li].append(lp_steer - lp_orig)

        if (start // bs) % 4 == 0:
            print(f"  batch {start//bs + 1}/{(n + bs - 1)//bs}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    del lm_head_cpu

    layer_deltas = {}
    for li in range(N_LAYERS - 1):
        all_d = torch.cat(all_deltas[li])
        layer_deltas[li] = float(all_d.mean().item())

    del hs, token_ids, model, tt
    gc.collect()

    known = {
        0: -3, 1: 0, 2: 17, 3: 13, 4: 7, 5: 13,
        6: 0, 7: -5, 8: 20, 9: -23, 10: 17,
        11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
        16: 0, 17: 0, 18: 0, 19: 0, 20: 0,
        21: 0, 22: 0, 23: 0, 24: -10, 25: 0, 26: 0, 27: 0,
    }

    print(f"\n{'='*80}", flush=True)
    print(f"LOGIT-DELTA PROBE — Qwen2.5-7B", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'Layer':>6} {'Δlogit':>10} {'Δacc(pp)':>10} {'Pred':>10}", flush=True)

    for li in sorted(layer_deltas.keys()):
        d_l = layer_deltas[li]
        d_acc = known.get(li, 0)
        pred = "TRIM" if d_l > 0 else "DEATH"
        print(f"  L{li:3d}  {d_l:+10.6f}  {d_acc:+9d}pp  {pred:>10}", flush=True)

    valid = [(layer_deltas[l], known[l]) for l in layer_deltas if l in known]
    deltas_logit, deltas_acc = zip(*valid)
    r = np.corrcoef(list(deltas_logit), list(deltas_acc))[0, 1]
    print(f"\n  ρ(Δlogit, Δacc) = {r:.4f} (all layers)", flush=True)

    sig = [(ld, ad) for ld, ad in zip(deltas_logit, deltas_acc) if abs(ad) > 5]
    if len(sig) >= 3:
        ls_k, as_k = zip(*sig)
        r_sig = np.corrcoef(list(ls_k), list(as_k))[0, 1]
        print(f"  ρ(Δlogit, Δacc) = {r_sig:.4f} (|Δacc| > 5pp)", flush=True)

    for l_trim, l_death in [(8, 9), (2, 9), (10, 9)]:
        dt = layer_deltas[l_trim]
        dd = layer_deltas[l_death]
        correct = (dt > dd) if known[l_trim] > 0 else (dt < dd)
        print(f"  Δlogit[L{l_trim}]={dt:+.6f} vs Δlogit[L{l_death}]={dd:+.6f} → {'✓' if correct else '✗'}", flush=True)

    out = {
        "model": "Qwen2.5-7B", "alpha": ALPHA,
        "layer_deltas": layer_deltas, "correlation_all": float(r),
    }
    with open("logit_delta_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to logit_delta_probe.json", flush=True)
    print(f"Time: {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
