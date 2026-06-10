#!/usr/bin/env python3
"""Multi-seed, multi-segment sweep for StreamFusion variants.
Runs each variant with configurable seeds and segments, computes mean ± SD trajectories."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time


VARIANTS = [
    # (name, variant_type, extra_flags)
    ("V", "hybrid", "--use-vectors"),
    ("BVA", "hybrid", "--bidirectional --use-vectors --use-norm"),
    ("B", "hybrid", "--bidirectional"),
    ("plain", "plain", ""),
    ("dora", "dora", ""),
    # Unorthodox variants
    ("AFA", "hybrid", "--use-activation"),
    ("AUR", "hybrid", "--use-autoencoder"),
    ("PERA", "hybrid", "--use-polynomial"),
    ("AFA+AUR", "hybrid", "--use-activation --use-autoencoder"),
    ("B+AFA", "hybrid", "--bidirectional --use-activation"),
    ("V+AFA", "hybrid", "--use-vectors --use-activation"),
]

SEEDS = [1, 42, 123]
N_SEGMENTS = 5
STEPS_PER_SEGMENT = 20
BASE_CMD = (
    f"cd {os.path.dirname(os.path.abspath(__file__))} && "
    f"PYTHONPATH=. "
    f"{sys.executable} run_stream_fusion.py "
    f"--n-segments {N_SEGMENTS} --steps-per-segment {STEPS_PER_SEGMENT} "
    f"--num-texts 50 --n-latents 16 --d-latent 32 --top-m 4 --r 8"
)

OUTPUT_DIR = "sweep_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single(name: str, variant: str, flags: str, seed: int) -> dict | None:
    outfile = f"{OUTPUT_DIR}/{name}_seed{seed}.json"
    cmd = f"{BASE_CMD} --expert-variant {variant} {flags} --seed {seed} --output {outfile}"
    print(f"  [{name} seed={seed}] Running...", end=" ", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"FAILED ({elapsed:.0f}s)")
        print(f"    {r.stderr[-300:]}")
        return None
    try:
        with open(outfile) as f:
            data = json.load(f)
        print(f"OK ({elapsed:.0f}s, {len(data.get('segments', []))} segs)")
        return data
    except Exception as e:
        print(f"PARSE ERROR ({elapsed:.0f}s): {e}")
        return None


def extract_trajectory(data: dict) -> dict:
    """Extract per-segment metrics as lists."""
    traj = {"segments": [], "train_ppl": [], "eval_ppl": [], "expert_count": []}
    for seg in data.get("segments", []):
        traj["segments"].append(seg["segment"])
        traj["train_ppl"].append(seg.get("training_ppl", 0))
        traj["eval_ppl"].append(seg.get("eval_ppl", None))
        traj["expert_count"].append(seg.get("expert_count", 0))
    return traj


def main():
    print(f"Sweep: {len(VARIANTS)} variants × {len(SEEDS)} seeds × {N_SEGMENTS} segments")
    print(f"Total runs: {len(VARIANTS) * len(SEEDS)}")
    print()

    all_results = {}

    for name, variant, flags in VARIANTS:
        trajectories = []
        for seed in SEEDS:
            data = run_single(name, variant, flags, seed)
            if data is not None:
                trajectories.append(extract_trajectory(data))

        if not trajectories:
            print(f"  [{name}] ALL FAILED, skipping")
            continue

        # Compute statistics per segment
        max_seg = max(len(t["segments"]) for t in trajectories)
        stats = {}
        for i in range(max_seg):
            train_vals = [t["train_ppl"][i] for t in trajectories if i < len(t["train_ppl"])]
            eval_vals = [t["eval_ppl"][i] for t in trajectories if i < len(t["eval_ppl"]) and t["eval_ppl"][i] is not None]

            import numpy as np
            seg_stat = {"segment": i + 1}

            if train_vals:
                seg_stat["train_ppl_mean"] = float(np.mean(train_vals))
                seg_stat["train_ppl_std"] = float(np.std(train_vals))
                seg_stat["train_ppl_raw"] = train_vals

            if eval_vals:
                seg_stat["eval_ppl_mean"] = float(np.mean(eval_vals))
                seg_stat["eval_ppl_std"] = float(np.std(eval_vals))
                seg_stat["eval_ppl_raw"] = eval_vals

            stats[i + 1] = seg_stat

        all_results[name] = {
            "config": {"variant": variant, "flags": flags, "seeds": SEEDS, "n_segments": N_SEGMENTS},
            "n_completed": len(trajectories),
            "per_segment": stats,
            "final_eval": stats.get(max_seg, {}),
        }

        # Print summary for this variant
        final = stats.get(max_seg, {})
        fe_mean = final.get("eval_ppl_mean", float("nan"))
        fe_std = final.get("eval_ppl_std", float("nan"))
        first = stats.get(1, {})
        fe_first = first.get("eval_ppl_mean", float("nan"))
        improvement = fe_first - fe_mean if (fe_first != float("nan") and fe_mean != float("nan")) else 0
        print(f"  [{name}] Final: {fe_mean:.0f} ± {fe_std:.0f} | First: {fe_first:.0f} | Impr: {improvement:+.0f}")

    # Overall ranking by final eval PPL (mean)
    print()
    print("=" * 70)
    print(f"  FINAL RANKING (mean ± SD, {N_SEGMENTS} segments × {len(SEEDS)} seeds)")
    print("=" * 70)
    ranked = sorted(
        [(name, r["per_segment"].get(max(r["per_segment"].keys()) if r["per_segment"] else 0, {}))
         for name, r in all_results.items()],
        key=lambda x: x[1].get("eval_ppl_mean", float("inf")),
    )
    print(f"  {'Rank':>4s} {'Variant':>8s} {'Eval PPL':>12s} {'Train PPL':>12s} {'Improvement':>12s}")
    print(f"  {'-'*48}")
    for i, (name, final) in enumerate(ranked, 1):
        ep_mean = final.get("eval_ppl_mean", float("nan"))
        ep_std = final.get("eval_ppl_std", float("nan"))
        tp_mean = final.get("train_ppl_mean", float("nan"))
        first = all_results[name]["per_segment"].get(1, {})
        fe_first = first.get("eval_ppl_mean", float("nan"))
        impr = fe_first - ep_mean if (fe_first != float("nan") and ep_mean != float("nan")) else 0
        print(f"  {i:>4d} {name:>8s} {ep_mean:>8.0f} ± {ep_std:<5.0f} {tp_mean:>8.0f}           {impr:>+6.0f}")

    # Save all
    outpath = f"{OUTPUT_DIR}/summary.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved to {outpath}")


if __name__ == "__main__":
    main()
