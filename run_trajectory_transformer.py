#!/usr/bin/env python3
"""Train TrajectoryTransformer with memory-efficient streaming."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer, TrajectoryTrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MemmapDataset:
    """Trajectory dataset using memory-mapped consolidated tensor."""

    def __init__(self, consolidated_path: str, train_frac: float = 0.8):
        self.data = torch.load(consolidated_path, map_location="cpu", mmap=True)
        self.n_total = self.data.shape[0]
        self.n_train = int(self.n_total * train_frac)
        self.n_test = self.n_total - self.n_train
        self.n_layers = self.data.shape[1] - 1
        self.d_model = self.data.shape[2]

    def _traj_to_pairs(self, traj: torch.Tensor) -> tuple:
        L = self.n_layers - 1
        return traj[:L].clone(), (traj[1:L + 1] - traj[:L]).clone()

    def epoch_batches(self, batch_size: int, train: bool = True):
        start = 0 if train else self.n_train
        end = self.n_train if train else self.n_total
        indices = np.arange(start, end)
        if train:
            np.random.shuffle(indices)
        bx, by = [], []
        for idx in indices:
            x, y = self._traj_to_pairs(self.data[idx])
            bx.append(x.unsqueeze(0)); by.append(y.unsqueeze(0))
            if len(bx) >= batch_size:
                yield torch.cat(bx, dim=0), torch.cat(by, dim=0)
                bx, by = [], []
        if bx:
            yield torch.cat(bx, dim=0), torch.cat(by, dim=0)

    def full_test(self) -> tuple:
        xs, ys = [], []
        for idx in range(self.n_train, self.n_total):
            x, y = self._traj_to_pairs(self.data[idx])
            xs.append(x.unsqueeze(0)); ys.append(y.unsqueeze(0))
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=str,
                        default="/run/media/filip/B522-875D/Datasets/project_data/thought_trajs_25k/all_trajs.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8, help="Smaller batch to save RAM")
    args = parser.parse_args()

    dataset = MemmapDataset(args.trajectories)
    print(f"Data: {dataset.n_total} trajectories ({dataset.n_train} train, {dataset.n_test} test)")
    print(f"  Each: {dataset.data.shape}")

    model = TrajectoryTransformer(
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, d_input=dataset.d_model,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} ({n_params/1e6:.1f}M) params")

    trainer = TrajectoryTrainer(model, lr=args.lr, device=DEVICE)

    # Pre-load test set (smaller, 20%)
    test_x, test_y = dataset.full_test()
    test_zero_mse = F.mse_loss(torch.zeros_like(test_y), test_y).item()
    print(f"Test: {len(test_x)} trajectories, zero MSE = {test_zero_mse:.5f}")

    best_test = float("inf")
    r2, cos = 0.0, 0.0
    t_start = time.time()
    start_epoch = 0
    checkpoint_path = "transformer_checkpoint.pt"

    def save_checkpoint(sig=None, frame=None):
        print(f"\nCheckpoint at epoch {epoch} (best test: {best_test:.5f}, R²: {r2:.4f})")
        torch.save({"model": model.state_dict(), "best_test": best_test,
                     "epoch": epoch, "r2": r2, "cos": cos},
                    checkpoint_path)
        if sig is not None:
            print("Exiting due to signal...")
            sys.exit(0)

    signal.signal(signal.SIGINT, save_checkpoint)
    signal.signal(signal.SIGTERM, save_checkpoint)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        best_test = ckpt.get("best_test", float("inf"))
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resumed from epoch {start_epoch} (best: {best_test:.5f})")

    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []
        for bx, by in dataset.epoch_batches(args.batch_size, train=True):
            loss = trainer.train_step(bx, by)
            epoch_losses.append(loss)

        # Evaluate
        model.eval()
        with torch.no_grad():
            vp = model(test_x.to(DEVICE))
            test_mse = F.mse_loss(vp.cpu(), test_y).item()
            cos = F.cosine_similarity(vp.cpu().flatten(1), test_y.flatten(1), dim=-1).mean().item()
        model.train()

        if test_mse < best_test:
            best_test = test_mse
            torch.save(model.state_dict(), "best_transformer_25k.pt")

        r2 = max(0, 1 - best_test / test_zero_mse)
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        elapsed = time.time() - t_start
        sps = (epoch + 1) / (elapsed + 1e-6)

        print(f"Epoch {epoch:3d} | train={avg_loss:.5f} | test={test_mse:.5f} | "
              f"cos={cos:.4f} | best={best_test:.5f} | R²={r2:.4f} | {elapsed:.0f}s | {sps:.3f}ep/s")

    save_checkpoint()
    print(f"\nDone: {elapsed:.0f}s | Best test: {best_test:.5f} | Zero: {test_zero_mse:.5f} | R²: {r2:.4f}")

    results = {"best_test_mse": best_test, "zero_mse": test_zero_mse, "r_squared": r2, "cosine": cos}
    with open("transformer_25k_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
