#!/usr/bin/env python3
"""Evaluate ThoughtDiffusion on thought trajectories.

Compares flow matching vs baselines on held-out data.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.thought_diffusion import ThoughtDiffusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="best_thought_diffusion.pt")
    parser.add_argument("--trajectories", type=str, default="./thought_trajectories_500/")
    parser.add_argument("--d-latent", type=int, default=64)
    args = parser.parse_args()

    with open(os.path.join(args.trajectories, "meta.json")) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    d_model = meta["d_model"]

    trajectories = [torch.load(os.path.join(args.trajectories, f"traj_{i:04d}.pt"), map_location="cpu")
                    for i in range(meta["n_trajectories"])]

    n_test = max(1, len(trajectories) // 5)
    test_traj = trajectories[-n_test:]

    test_x, test_y, test_l, test_ctx = [], [], [], []
    for traj in test_traj:
        for l in range(traj.shape[0] - 2):
            test_x.append(traj[l])
            test_y.append(traj[l + 1] - traj[l])
            test_l.append(l)
            test_ctx.append(traj[0])

    test_x = torch.stack(test_x).float()
    test_y = torch.stack(test_y).float()
    test_l = torch.tensor(test_l, dtype=torch.long)
    test_ctx = torch.stack(test_ctx).float()
    print(f"Held-out: {len(test_x)} pairs from {n_test} trajectories")

    model = ThoughtDiffusion(d_model=d_model, n_layers=n_layers,
                              d_latent=args.d_latent, d_text_ctx=d_model)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    with torch.no_grad():
        _, v_pred = model(test_x.to(DEVICE), test_l.to(DEVICE),
                          torch.zeros(len(test_x), 1, device=DEVICE),
                          test_l.float().unsqueeze(-1).to(DEVICE) / max(n_layers - 1, 1),
                          test_ctx.to(DEVICE))

        flow_mse = F.mse_loss(v_pred.cpu(), test_y).item()
        zero_mse = F.mse_loss(torch.zeros_like(test_y), test_y).item()
        mean_v = test_y.mean(dim=0, keepdim=True).expand_as(test_y)
        const_mse = F.mse_loss(mean_v, test_y).item()
        flow_cos = F.cosine_similarity(v_pred.cpu().flatten(1), test_y.flatten(1), dim=-1).mean().item()

    r2 = max(0, 1 - flow_mse / zero_mse)
    impr_zero = (zero_mse - flow_mse) / zero_mse * 100

    print(f"\n{'='*60}")
    print(f"{'Metric':<30} {'Value':<15}")
    print(f"{'-'*30} {'-'*15}")
    print(f"{'Flow MSE':<30} {flow_mse:.6f}")
    print(f"{'Zero MSE (h[l+1]=h[l])':<30} {zero_mse:.6f}")
    print(f"{'Constant MSE (mean v)':<30} {const_mse:.6f}")
    print(f"{'Flow cosine similarity':<30} {flow_cos:.4f}")
    print(f"{'Improvement over zero':<30} {impr_zero:.2f}%")
    print(f"{'R²':<30} {r2:.4f}")

    if r2 > 0.3:
        print(f"\n✅ Thoughts have strong learnable structure!")
    elif r2 > 0.05:
        print(f"\n✅ Thoughts have moderate structure (R²={r2:.3f})")
    elif r2 > 0.01:
        print(f"\n⚠️  Thoughts have weak structure (R²={r2:.3f})")
    else:
        print(f"\n❌ Thoughts have no detectable structure")

    results = {"flow_mse": flow_mse, "zero_mse": zero_mse, "const_mse": const_mse,
               "cosine": flow_cos, "r_squared": r2, "improvement_pct": impr_zero}
    with open("thought_flow_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
