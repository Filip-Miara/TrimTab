#!/usr/bin/env python3
"""Train Perceiver-based ThoughtDiffusion on full trajectory sequences."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.thought_diffusion import ThoughtDiffusion, ThoughtFlowTrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=str, default="./thought_trajectories_500/")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-latent", type=int, default=64)
    parser.add_argument("--n-latents", type=int, default=16)
    args = parser.parse_args()

    with open(os.path.join(args.trajectories, "meta.json")) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    d_model = meta["d_model"]

    trajectories = [torch.load(os.path.join(args.trajectories, f"traj_{i:04d}.pt"), map_location="cpu")
                    for i in range(meta["n_trajectories"])]

    split = int(len(trajectories) * 0.8)
    train_traj = trajectories[:split]
    test_traj = trajectories[split:]

    def _build_batches(trajs):
        xs, ys, ctxs = [], [], []
        for traj in trajs:
            L = traj.shape[0] - 2
            x = traj[:L]
            y = traj[1:L + 1] - traj[:L]
            ctx = traj[0]
            xs.append(x)
            ys.append(y)
            ctxs.append(ctx)
        return (
            torch.stack(xs).float(),
            torch.stack(ys).float(),
            torch.stack(ctxs).float(),
        )

    train_x, train_y, train_ctx = _build_batches(train_traj)
    test_x, test_y, test_ctx = _build_batches(test_traj)

    print(f"Train: {len(train_x)} trajectories, {train_x.shape}")
    print(f"Test:  {len(test_x)} trajectories, {test_x.shape}")

    model = ThoughtDiffusion(d_model=d_model, n_layers=n_layers,
                              d_latent=args.d_latent, n_latents=args.n_latents,
                              d_text_ctx=d_model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    trainer = ThoughtFlowTrainer(model, lr=args.lr, device=DEVICE)

    n_train = len(train_x)
    batch_size = min(16, n_train)
    steps_per_epoch = max(1, n_train // batch_size)

    best_test = float("inf")
    t_start = time.time()
    zero_mse = 0.0

    for epoch in range(args.epochs):
        perm = torch.randperm(n_train)
        epoch_losses = []
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            loss = trainer.train_step(train_x[idx], train_y[idx], train_ctx[idx])
            epoch_losses.append(loss)

        model.eval()
        with torch.no_grad():
            vp = model(test_x.to(DEVICE), test_ctx.to(DEVICE))
            test_mse = F.mse_loss(vp.cpu(), test_y).item()
            zero_mse = F.mse_loss(torch.zeros_like(test_y), test_y).item()
            cos = F.cosine_similarity(vp.cpu().flatten(1), test_y.flatten(1), dim=-1).mean().item()
        model.train()

        avg_loss = np.mean(epoch_losses)
        if test_mse < best_test:
            best_test = test_mse
            torch.save(model.state_dict(), "best_perceiver.pt")
        r2 = max(0, 1 - best_test / zero_mse)

        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d} | train={avg_loss:.6f} | test={test_mse:.6f} | "
              f"cos={cos:.4f} | zero={zero_mse:.6f} | best={best_test:.6f} | R²={r2:.4f} | {elapsed:.0f}s")

    print(f"\nDone: {elapsed:.0f}s | Best test: {best_test:.6f} | Zero: {zero_mse:.6f} | R²: {r2:.4f}")


if __name__ == "__main__":
    main()
