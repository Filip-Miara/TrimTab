#!/usr/bin/env python3
"""Batch hyperparameter sweep over WeightDiffusion on pre-computed 2B trajectories.

Loads trajectories from disk (trajectories_2B/), trains multiple configs,
evaluates on held-out data, ranks results. Then launches full training
with the best config + stagnation penalty.

Usage:
  python3 run_sweep_train.py                  # full sweep
  python3 run_sweep_train.py --quick          # small subset for fast iteration
"""
from __future__ import annotations

import gc
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.diffusion_weight_flow import (
    WeightDiffusion, DiffusionFlowTrainer, augment_trajectories,
)
from src.adapters.stream_fusion import N_FLAGS
from generate_trajectories import TrajectoryDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SweepConfig:
    """Hyperparameter grid for the sweep."""
    d_latent: list[int] = field(default_factory=lambda: [32, 64])
    n_latents: list[int] = field(default_factory=lambda: [8, 16])
    n_blocks: list[int] = field(default_factory=lambda: [2])
    lr: list[float] = field(default_factory=lambda: [1e-3, 3e-4])
    lambda_flow: list[float] = field(default_factory=lambda: [0.1, 0.5])
    lambda_optimal: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    lambda_stagnation: list[float] = field(default_factory=lambda: [0.2])
    epochs: int = 3
    n_train: int = 35
    n_test: int = 10

    @property
    def grid_size(self) -> int:
        return (len(self.d_latent) * len(self.n_latents) * len(self.n_blocks) *
                len(self.lr) * len(self.lambda_flow) *
                len(self.lambda_optimal) * len(self.lambda_stagnation))


def make_config_name(params: dict) -> str:
    return (f"dl{params['d_latent']}"
            f"_nl{params['n_latents']}"
            f"_lr{params['lr']}"
            f"_fl{params['lambda_flow']}"
            f"_ot{params['lambda_optimal']}"
            f"_st{params['lambda_stagnation']}")


def load_data(traj_dir: str = "./trajectories_2B", n_train: int = 35, n_test: int = 10):
    """Load pre-computed trajectories from disk."""
    print(f"Loading trajectories from {traj_dir}...")
    ds = TrajectoryDataset(traj_dir)
    n_weights = ds.n_weights
    d_ctx = ds.d_ctx
    all_traj = ds.load_all()
    all_traj = [(t["weights"], t["ctxs"], t.get("grads")) for t in all_traj]
    np.random.seed(0)
    idx = np.random.permutation(len(all_traj)).tolist()
    train = [all_traj[i] for i in idx[:n_train]]
    test = [all_traj[i] for i in idx[n_train:n_train + n_test]]
    print(f"  {len(train)} train + {len(test)} test trajectories")
    print(f"  weight_dim={n_weights}, ctx_dim={d_ctx}")
    return train, test, n_weights, d_ctx


def train_and_eval(
    train_traj: list,
    test_traj: list,
    params: dict,
    n_weights: int,
    d_ctx: int,
    min_alignment: float = 0.3,
) -> dict:
    """Train a single WeightDiffusion configuration and evaluate."""
    name = make_config_name(params)
    n_noise = 3
    n_epochs = params.get("epochs", 3)

    augmented = augment_trajectories(train_traj, n_noise_levels=n_noise, min_alignment=min_alignment)
    test_aug = augment_trajectories(test_traj, n_noise_levels=n_noise, min_alignment=min_alignment)

    model = WeightDiffusion(
        n_weights=n_weights,
        d_latent=params["d_latent"],
        n_latents=params["n_latents"],
        d_ctx=d_ctx,
        n_perceiver_blocks=params.get("n_blocks", 2),
    )

    trainer = DiffusionFlowTrainer(
        model, lr=params["lr"], device=DEVICE,
        lambda_flow=params["lambda_flow"],
        lambda_optimal=params["lambda_optimal"],
        lambda_stagnation=params["lambda_stagnation"],
    )

    train_losses = []
    batch_size = 20
    for epoch in range(n_epochs):
        trainer.set_epoch(epoch)
        ep_losses = []
        np.random.shuffle(augmented)
        t_ep = time.time()
        for i in range(0, len(augmented), batch_size):
            batch = augmented[i:i + batch_size]
            B = len(batch)
            cl = torch.stack([d["clean"] for d in batch]).to(DEVICE)
            nx = torch.stack([d["next"] for d in batch]).to(DEVICE)
            fl = torch.stack([d["flags"] for d in batch]).to(DEVICE)
            cx = torch.stack([d["ctx"] for d in batch]).to(DEVICE)
            tn = torch.tensor([[d["t_noise"]] for d in batch]).to(DEVICE)
            tf = torch.tensor([[d["t_flow"]] for d in batch]).to(DEVICE)

            opt_t = None
            if batch[0]["optimal_target"] is not None:
                opt_t = torch.stack([d["optimal_target"] for d in batch]).to(DEVICE)

            l, _, _, _, _ = trainer.train_step(cl, nx, fl, cx, tn, tf, optimal_target=opt_t)
            ep_losses.append(l)
        train_losses.append(float(np.mean(ep_losses)))
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"    epoch {epoch}: loss={train_losses[-1]:.4f} mem={mem:.0f}MB ({time.time()-t_ep:.0f}s)", flush=True)
    print(f"    train done: final={train_losses[-1]:.4f}", flush=True)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        eval_losses = []
        for i in range(0, len(test_aug), batch_size):
            batch = test_aug[i:i + batch_size]
            B = len(batch)
            cl = torch.stack([d["clean"] for d in batch]).to(DEVICE)
            fl = torch.stack([d["flags"] for d in batch]).to(DEVICE)
            cx = torch.stack([d["ctx"] for d in batch]).to(DEVICE)
            tf = torch.tensor([[d["t_flow"]] for d in batch]).to(DEVICE)
            nx = torch.stack([d["next"] for d in batch]).to(DEVICE)

            _, v_pred = model(
                cl, torch.zeros(B, 1, device=DEVICE), tf, fl, cx,
            )
            target = (nx - cl)
            eval_losses.append(F.mse_loss(v_pred, target).item())
    final_eval = float(np.mean(eval_losses))
    final_train = train_losses[-1] if train_losses else float("inf")

    return {
        "name": name,
        "params": params,
        "train_loss": final_train,
        "eval_loss": final_eval,
        "train_trajectory": train_losses,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for WeightDiffusion")
    parser.add_argument("--quick", action="store_true", help="Small quick sweep")
    parser.add_argument("--traj-dir", type=str, default="./trajectories_2B")
    parser.add_argument("--n-train", type=int, default=35)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--min-alignment", type=float, default=0.3,
                        help="Min cosine similarity between SGD update and gradient (default: 0.3)")
    parser.add_argument("--output", type=str, default="sweep_results.json")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    train_traj, test_traj, n_weights, d_ctx = load_data(
        args.traj_dir, args.n_train, args.n_test,
    )

    if args.quick:
        cfg = SweepConfig(
            d_latent=[32],
            n_latents=[8],
            lr=[3e-4],
            lambda_flow=[0.1, 0.5],
            lambda_optimal=[0.0, 0.5],
            lambda_stagnation=[0.2],
            epochs=args.epochs,
            n_train=args.n_train, n_test=args.n_test,
        )
    else:
        cfg = SweepConfig(epochs=args.epochs, n_train=args.n_train, n_test=args.n_test)

    grid = cfg.grid_size
    print(f"\nSweep grid: {grid} configurations")
    print(f"  d_latent={cfg.d_latent}, n_latents={cfg.n_latents}")
    print(f"  lr={cfg.lr}")
    print(f"  lambda_flow={cfg.lambda_flow}, lambda_optimal={cfg.lambda_optimal}")
    print(f"  lambda_stagnation={cfg.lambda_stagnation}")
    print(f"  epochs={cfg.epochs}, n_train={cfg.n_train}, n_test={cfg.n_test}")
    print()

    results = []
    keys = ["d_latent", "n_latents", "n_blocks", "lr", "lambda_flow", "lambda_optimal", "lambda_stagnation"]
    param_sets = [
        dict(zip(keys, vals))
        for vals in itertools.product(cfg.d_latent, cfg.n_latents, cfg.n_blocks,
                                       cfg.lr, cfg.lambda_flow, cfg.lambda_optimal,
                                       cfg.lambda_stagnation)
    ]

    t0 = time.time()
    for i, params in enumerate(param_sets):
        params["epochs"] = cfg.epochs

        # Isolate each run: purge CUDA state completely
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Clear any leftover references
        for j in range(i):
            if 'model_cleanup' in dir():
                pass

        print(f"\n  [{i + 1}/{grid}] {make_config_name(params)}...", flush=True)
        try:
            r = train_and_eval(train_traj, test_traj, params, n_weights, d_ctx, args.min_alignment)
            best_marker = " ← BEST" if (not results or r["eval_loss"] < min(x["eval_loss"] for x in results)) else ""
            # Track if eval decreases monotonically (could indicate state leakage)
            if len(results) > 0:
                prev_eval = results[-1]["eval_loss"]
                trend = "↓" if r["eval_loss"] < prev_eval else "↑" if r["eval_loss"] > prev_eval else "→"
                print(f"  train={r['train_loss']:.4f} eval={r['eval_loss']:.6f} ({trend} from {prev_eval:.6f}){best_marker}", flush=True)
            else:
                print(f"  train={r['train_loss']:.4f} eval={r['eval_loss']:.6f}{best_marker}", flush=True)
            results.append(r)
        except Exception as e:
            print(f"FAILED: {e}")

        # Save intermediate results
        with open(args.output, "w") as f:
            json.dump({
                "config": {k: getattr(cfg, k) for k in ["d_latent", "n_latents", "lr", "lambda_flow", "lambda_optimal", "lambda_stagnation"]},
                "results": sorted(results, key=lambda r: r["eval_loss"]),
                "total_time": time.time() - t0,
            }, f, indent=2)

    # Final ranking
    ranked = sorted(results, key=lambda r: r["eval_loss"])
    print(f"\n{'='*70}")
    print(f"  RANKING ({len(ranked)} configs, {time.time()-t0:.0f}s total)")
    print(f"{'='*70}")
    print(f"  {'Rank':>4s} {'Name':>40s} {'Train':>8s} {'Eval':>8s}")
    print(f"  {'-'*62}")
    for i, r in enumerate(ranked, 1):
        print(f"  {i:>4d} {r['name']:>40s} {r['train_loss']:>8.4f} {r['eval_loss']:>8.4f}")

    best = ranked[0]
    print(f"\n  Best config: {best['name']}")
    print(f"  Best eval loss: {best['eval_loss']:.6f}")
    print(f"  Params: {json.dumps(best['params'], default=str)}")

    # Launch full training with best config + stagnation
    print(f"\n{'='*70}")
    print(f"  Launching full training with best config + stagnation...")
    print(f"{'='*70}")

    # Train full model on ALL trajectories with best config
    n_noise = 5
    n_epochs_full = 15
    # Train full model on TRAIN trajectories only (NOT test — keep held out for eval)
    all_augmented = augment_trajectories(train_traj, n_noise_levels=5, min_alignment=args.min_alignment)
    print(f"  Full training on {len(all_augmented)} augmented samples from {len(train_traj)} train trajectories")
    print(f"  Test trajectories ({len(test_traj)}) held out for evaluation")

    model = WeightDiffusion(
        n_weights=n_weights,
        d_latent=best["params"]["d_latent"],
        n_latents=best["params"]["n_latents"],
        d_ctx=d_ctx,
        n_perceiver_blocks=best["params"].get("n_blocks", 2),
    )
    trainer = DiffusionFlowTrainer(
        model, lr=best["params"]["lr"], device=DEVICE,
        lambda_flow=best["params"]["lambda_flow"],
        lambda_optimal=best["params"]["lambda_optimal"],
        lambda_stagnation=best["params"]["lambda_stagnation"],
    )

    t_start = time.time()
    batch_size = 20
    for epoch in range(n_epochs_full):
        trainer.set_epoch(epoch)
        losses = []
        np.random.shuffle(all_augmented)
        for i in range(0, len(all_augmented), batch_size):
            batch = all_augmented[i:i + batch_size]
            B = len(batch)
            cl = torch.stack([d["clean"] for d in batch]).to(DEVICE)
            nx = torch.stack([d["next"] for d in batch]).to(DEVICE)
            fl = torch.stack([d["flags"] for d in batch]).to(DEVICE)
            cx = torch.stack([d["ctx"] for d in batch]).to(DEVICE)
            tn = torch.tensor([[d["t_noise"]] for d in batch]).to(DEVICE)
            tf = torch.tensor([[d["t_flow"]] for d in batch]).to(DEVICE)

            opt_t = None
            if batch[0]["optimal_target"] is not None:
                opt_t = torch.stack([d["optimal_target"] for d in batch]).to(DEVICE)

            l, dl, fl_l, ol, sl = trainer.train_step(cl, nx, fl, cx, tn, tf, optimal_target=opt_t)
            losses.append(l)
        dt = time.time() - t_start
        print(f"  Epoch {epoch}: loss={np.mean(losses):.6f} diff={float(dl):.6f} flow={float(fl_l):.6f} opt={float(ol):.6f} stag={float(sl):.6f} ({dt:.0f}s)", flush=True)

    # Save model
    model.cpu()
    torch.save(model.state_dict(), "diffusion_weight_flow.pt")
    meta = {
        "n_weights": n_weights, "d_ctx": d_ctx,
        "n_train": len(train_traj), "n_test": len(test_traj),
        "n_augmented": len(all_augmented),
        "final_loss": float(np.mean(losses)),
        **best["params"],
    }
    with open("diffusion_weight_flow_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    fsize = os.path.getsize("diffusion_weight_flow.pt")
    print(f"\nModel saved: diffusion_weight_flow.pt ({fsize/1024/1024:.1f} MB)")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
