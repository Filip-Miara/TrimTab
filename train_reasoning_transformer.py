#!/usr/bin/env python3
"""Train TrajectoryTransformer on reasoning-step trajectories.

Uses causal masking (each position only attends to past positions)
to model token-to-token hidden state evolution.

Data format: padded (N, max_len, 2048) tensor + (N,) lengths tensor.
Missing positions are masked in the loss.
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer, TrajectoryTrainer

DEVICE = "cuda"
DATA_DIR = "/run/media/filip/B522-875D/Datasets/project_data/reasoning_trajs_5k/"
N_EPOCHS = 100


def main():
    print(f"Loading data from {DATA_DIR}...")
    meta = json.load(open(os.path.join(DATA_DIR, "meta.json")))
    print(f"  Meta: {json.dumps(meta, indent=2)}")

    data = torch.load(os.path.join(DATA_DIR, "all_trajs.pt"), map_location="cpu", mmap=True)
    lengths = torch.load(os.path.join(DATA_DIR, "lengths.pt"), map_location="cpu")
    N, max_len, D = data.shape
    print(f"  Data: {N} trajectories × {max_len} max_len × {D} dims")

    # Split
    split = int(N * 0.8)
    train_d = data[:split]
    train_l = lengths[:split]
    test_d = data[split:]
    test_l = lengths[split:]
    print(f"  Train: {len(train_d)}, Test: {len(test_d)}")

    # Build model (causal TrajectoryTransformer for reasoning steps)
    model = TrajectoryTransformer(
        d_model=512, n_layers=6, n_heads=8, d_ff=2048,
        n_positions=max_len, d_input=D,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} ({n_params/1e6:.1f}M) params")

    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_test = float("inf")
    t0 = time.time()
    bs = 16

    for epoch in range(N_EPOCHS):
        model.train()
        perm = np.random.permutation(len(train_d))
        epoch_losses = []

        for i in range(0, len(train_d), bs):
            idx = perm[i:i+bs]
            batch = train_d[idx].to(DEVICE)
            lens = train_l[idx]
            actual_bs = len(idx)

            # Build input (h[0..L-2]) and target (v = h[1..L-1] - h[0..L-2])
            lens_d = lens.to(DEVICE)
            max_batch_len = lens.max().item()
            x = batch[:, :max_batch_len - 1, :].contiguous()
            y = (batch[:, 1:max_batch_len, :] - batch[:, :max_batch_len - 1, :]).contiguous()

            v_pred = model(x, causal=True)

            mask = torch.arange(max_batch_len - 1, device=DEVICE).unsqueeze(0) < (lens_d - 1).unsqueeze(1)
            mask = mask.unsqueeze(-1).float()

            loss = (F.mse_loss(v_pred, y, reduction='none') * mask).sum() / (mask.sum() + 1e-8)
            epoch_losses.append(loss.item())

            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        # Evaluate
        model.eval()
        test_losses, zero_losses = [], []
        with torch.no_grad():
            for idx in range(0, len(test_d), bs):
                batch = test_d[idx:idx+bs].to(DEVICE)
                lens_b = test_l[idx:idx+bs]
                lens_d = lens_b.to(DEVICE)
                max_bl = lens_b.max().item()
                if max_bl < 2: continue
                x = batch[:, :max_bl - 1, :].contiguous()
                y = (batch[:, 1:max_bl, :] - batch[:, :max_bl - 1, :]).contiguous()

                v_pred = model(x, causal=True)

                mask = torch.arange(max_bl - 1, device=DEVICE).unsqueeze(0) < (lens_d - 1).unsqueeze(1)
                mask = mask.unsqueeze(-1).float()
                tloss = (F.mse_loss(v_pred, y, reduction='none') * mask).sum() / (mask.sum() + 1e-8)
                test_losses.append(tloss.item())

                zl = (y ** 2 * mask).sum() / (mask.sum() + 1e-8)
                zero_losses.append(zl.item())

        avg_test = np.mean(test_losses)
        if avg_test < best_test:
            best_test = avg_test
            torch.save(model.state_dict(), "best_reasoning_transformer.pt")

        avg_zero = np.mean(zero_losses) if zero_losses else 1.0
        r2 = max(0, 1 - best_test / avg_zero)

        avg_train = np.mean(epoch_losses) if epoch_losses else 0
        elapsed = time.time() - t0
        if (epoch + 1) % 10 == 0:
            print(f"ep={epoch+1:3d} train={avg_train:.6f} test={avg_test:.6f} "
                  f"zero={avg_zero:.6f} best={best_test:.6f} r2={r2:.4f} {elapsed:.0f}s")

    print(f"\nDone: {elapsed:.0f}s | Best test MSE: {best_test:.6f} | Zero baseline: {avg_zero:.6f} | R²: {r2:.4f}")
    print(f"  Layer-to-layer R² was: 0.62")
    print(f"  Reasoning-step R² is:  {r2:.4f}")


if __name__ == "__main__":
    main()
