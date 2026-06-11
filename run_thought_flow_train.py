#!/usr/bin/env python3
"""Train ThoughtDiffusion on thought trajectories (hidden states).

Key experiment: do thought trajectories have enough structure for flow
matching to generalize?

Usage:
  python3 run_thought_flow_train.py --trajectories ./thought_trajectories_500/ --epochs 200
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.thought_diffusion import (
    ThoughtDiffusion,
    ThoughtFlowTrainer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ThoughtTrajectoryDataset:
    def __init__(self, traj_dir: str):
        with open(os.path.join(traj_dir, "meta.json")) as f:
            self.meta = json.load(f)

        self.trajectories = []
        for i in range(self.meta["n_trajectories"]):
            t = torch.load(os.path.join(traj_dir, f"traj_{i:04d}.pt"), map_location="cpu")
            self.trajectories.append(t)

        self.n_layers = self.meta["n_layers"]
        self.d_model = self.meta["d_model"]
        print(f"Loaded {len(self.trajectories)} trajectories")
        print(f"  Each: (25, {self.d_model})")

    def get_train_data(self, train_frac: float = 0.8) -> tuple:
        n = len(self.trajectories)
        n_train = int(n * train_frac)
        train_traj = self.trajectories[:n_train]
        test_traj = self.trajectories[n_train:]

        def _extract_pairs(trajs):
            xs, ys, layers, ctxs = [], [], [], []
            for traj in trajs:
                for l in range(traj.shape[0] - 2):
                    xs.append(traj[l])
                    ys.append(traj[l + 1] - traj[l])
                    layers.append(l)
                    ctxs.append(traj[0])
            return (
                torch.stack(xs).float(),
                torch.stack(ys).float(),
                torch.tensor(layers, dtype=torch.long),
                torch.stack(ctxs).float(),
            )

        train_x, train_y, train_l, train_ctx = _extract_pairs(train_traj)
        test_x, test_y, test_l, test_ctx = _extract_pairs(test_traj)

        print(f"  Train: {len(train_x)} pairs from {n_train} trajectories")
        print(f"  Test:  {len(test_x)} pairs from {n - n_train} trajectories")
        return (train_x, train_y, train_l, train_ctx), (test_x, test_y, test_l, test_ctx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=str, default="./thought_trajectories_500/",
                        help="Path to thought trajectories (layer-to-layer)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-latent", type=int, default=64)
    parser.add_argument("--n-latents", type=int, default=8)
    args = parser.parse_args()

    dataset = ThoughtTrajectoryDataset(args.trajectories)
    (train_x, train_y, train_l, train_ctx), (test_x, test_y, test_l, test_ctx) = \
        dataset.get_train_data(train_frac=0.8)

    model = ThoughtDiffusion(
        d_model=dataset.d_model,
        n_layers=dataset.n_layers,
        d_latent=args.d_latent,
        n_latents=args.n_latents,
        d_text_ctx=dataset.d_model,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    trainer = ThoughtFlowTrainer(
        model, lr=args.lr, device=DEVICE,
        lambda_diff=0.0, lambda_flow=1.0, lambda_stagnation=0.0,
    )
    # Add weight decay
    trainer.opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    n_train = len(train_x)
    n_test = len(test_x)
    steps_per_epoch = max(1, n_train // args.batch_size)

    print(f"\nTraining: {n_train} train / {n_test} test pairs")
    print(f"  Batch: {args.batch_size}, steps/epoch: {steps_per_epoch}, epochs: {args.epochs}\n")

    best_test_loss = float("inf")
    t_start = time.time()
    test_flow_mse = 0.0

    for epoch in range(args.epochs):
        trainer.set_epoch(epoch)
        epoch_losses = []
        perm = torch.randperm(n_train)

        for step in range(steps_per_epoch):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            bx, by, bl, bc = train_x[idx], train_y[idx], train_l[idx], train_ctx[idx]

            t_noise = torch.rand(len(bx), 1) * 0.0
            t_flow = bl.float().unsqueeze(-1) / max(dataset.n_layers - 1, 1)

            loss, loss_flow, _ = trainer.train_step(
                bx, by, bl, bc, t_noise, t_flow,
            )
            epoch_losses.append(loss)

        # Eval
        model.eval()
        with torch.no_grad():
            _, velocity_pred = model(
                test_x.to(DEVICE), test_l.to(DEVICE),
                torch.zeros(n_test, 1, device=DEVICE),
                test_l.float().unsqueeze(-1).to(DEVICE) / max(dataset.n_layers - 1, 1),
                test_ctx.to(DEVICE),
            )
            target_velocity = test_y.to(DEVICE)
            test_flow_mse = F.mse_loss(velocity_pred, target_velocity).item()
            cos_sim = F.cosine_similarity(
                velocity_pred.flatten(1), target_velocity.flatten(1), dim=-1
            ).mean().item()
            zero_mse = F.mse_loss(torch.zeros_like(target_velocity), target_velocity).item()
        model.train()

        avg_train = np.mean(epoch_losses)
        if test_flow_mse < best_test_loss:
            best_test_loss = test_flow_mse
            torch.save(model.state_dict(), "best_thought_diffusion.pt")

        elapsed = time.time() - t_start
        print(f"  Epoch {epoch:3d} | train={avg_train:.6f} | test={test_flow_mse:.6f} | "
              f"cos={cos_sim:.4f} | zero={zero_mse:.6f} | {elapsed:.0f}s")
        if n_train > 10000:
            break

    total_time = time.time() - t_start
    r2 = max(0, 1 - best_test_loss / (zero_mse + 1e-10))

    print(f"\n{'='*60}")
    print(f"Training: {total_time:.0f}s | Best test MSE: {best_test_loss:.6f}")
    print(f"Zero baseline: {zero_mse:.6f} | R²: {r2:.4f}")

    if best_test_loss < zero_mse * 0.9:
        print("✅ Thoughts HAVE structure — flow matching beats zero!")
    elif best_test_loss < zero_mse * 0.99:
        print("⚠️  Marginal structure")
    else:
        print("❌ Thoughts unstructured — flow matching ≈ zero")

    results = {
        "best_test_loss": best_test_loss,
        "zero_mse": zero_mse,
        "r_squared": r2,
    }
    with open("thought_flow_train_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
