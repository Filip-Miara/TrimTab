#!/usr/bin/env python3
"""Evolution-strategy optimization over adapter lifecycles.

Maintains a population of MetaController variants. Each variant suggests
adapter configurations across segments. Fitness = cumulative eval PPL
improvement. Best variants breed via crossover + mutation.

Architecture:
  ES Population (n=8)
    ├── Individual 0: params → controller → suggest configs → run → fitness
    ├── Individual 1: ...
    └── Elites breed next generation
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters import AdapterConfig, AdapterWrappedLinear, adapt_linear_layer
from src.adapters.stream_fusion import StreamFusionLoRA
from src.adapters.adapter_evolution import (
    AdapterState, MetaController, AdapterEvolution, FLAG_NAMES,
)
from src.training.trainer import compute_perplexity

MODEL_PATH = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub"),
    "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17",
)
MODEL_FALLBACK = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = MODEL_FALLBACK
ARXIV_PARQUET_DIR = "/run/media/filip/B522-875D/Datasets/hub/datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section"


def load_arxiv_data(num_texts=50):
    parq_files = sorted(
        os.path.join(ARXIV_PARQUET_DIR, f) for f in os.listdir(ARXIV_PARQUET_DIR) if f.endswith(".parquet"))
    all_texts = []
    for pf in parq_files:
        tbl = pq.read_table(pf, columns=["article"])
        for art in tbl.column("article"):
            t = art.as_py()
            if len(t) > 500:
                all_texts.append(t)
                if len(all_texts) >= num_texts + 10:
                    return all_texts[:num_texts], all_texts[num_texts:num_texts + 10]
    return all_texts[:num_texts], all_texts[num_texts:num_texts + 10]


def tokenize_texts(tokenizer, texts, max_length=512):
    return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")


def build_extra_kwargs(flags: dict[str, bool], poly_order: int) -> dict:
    return {
        "expert_variant": "hybrid", "bidirectional": flags.get("bidirectional", False),
        "use_vectors": flags.get("use_vectors", False), "use_norm": flags.get("use_norm", False),
        "use_gate": flags.get("use_gate", False), "use_activation": flags.get("use_activation", False),
        "use_autoencoder": flags.get("use_autoencoder", False),
        "use_polynomial": flags.get("use_polynomial", False), "poly_order": poly_order,
        "n_latents": 16, "d_latent": 32, "d_key": 64, "top_m": 4, "max_experts": 100, "abs_steps": 2,
    }


class EvolutionRunner:
    """Manages one ES generation: runs all individuals, collects fitness."""

    def __init__(self, args, device: str):
        self.args = args
        self.device = device
        self.target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

        train_texts, eval_texts = load_arxiv_data(args.num_texts)
        self.train_texts = train_texts
        self.eval_enc = tokenize_texts(
            AutoTokenizer.from_pretrained(MODEL_PATH), eval_texts)

    def evaluate_individual(self, controller: MetaController, tokenizer) -> tuple[float, list[dict]]:
        """Run one individual: generate configs for N segments, train, return fitness."""
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
            device_map=self.device if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        model.requires_grad_(False)
        model.gradient_checkpointing_enable()
        model.to(self.device)

        adapter_modules = []
        history: list[AdapterState] = []

        for seg_idx in range(self.args.n_segments):
            soft_flags, poly_order, morph_rate, _ = controller(history, temperature=1.0)
            active = [f"{k}:{v:.2f}" for k, v in soft_flags.items() if v > 0.1]
            print(f"    seg {seg_idx + 1}: targets={{' '.join(active)}} morph={morph_rate:.2f}", flush=True)

            if seg_idx == 0:
                config = AdapterConfig(
                    r=self.args.r, lora_alpha=self.args.alpha, lora_dropout=0.0,
                    target_modules=self.target_modules,
                    extra_kwargs=build_extra_kwargs(soft_flags, poly_order),
                )
                for name, module in model.named_modules():
                    if any(name.endswith(t) for t in self.target_modules) and isinstance(module, nn.Linear):
                        wrapped = adapt_linear_layer(module, StreamFusionLoRA, config)
                        pn, _, cn = name.rpartition(".")
                        parent = model.get_submodule(pn) if pn else model
                        setattr(parent, cn, wrapped)
                        adapter_modules.append(wrapped.adapter)
                model.to(self.device)
            else:
                pass  # config set per-expert via set_targets below

            # Add new expert with soft target flags
            for am in adapter_modules:
                am.add_expert()
                # Cast expert to HybridStreamExpert for soft flag support
                expert = am.experts[-1]
                if hasattr(expert, 'set_targets'):
                    expert.set_targets(soft_flags, morph_rate)

            seg_text = self.train_texts[seg_idx % len(self.train_texts)]
            enc = tokenize_texts(tokenizer, [seg_text])
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            labels = enc["input_ids"].clone().to(self.device)

            model.train()
            trainable = [p for p in model.parameters() if p.requires_grad]
            if not trainable:
                for am in adapter_modules:
                    for p in am.parameters():
                        p.requires_grad_(True)
                trainable = [p for p in model.parameters() if p.requires_grad]

            opt = torch.optim.Adam(trainable, lr=self.args.lr)
            losses = []
            for step in range(self.args.steps_per_segment):
                # Morph soft flags each step
                for am in adapter_modules:
                    for expert in am.experts:
                        if hasattr(expert, 'morph_step'):
                            expert.morph_step()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                l = outputs.loss
                losses.append(l.item())
                l.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()

            # Eval
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for i in range(len(self.eval_enc["input_ids"])):
                    ids = self.eval_enc["input_ids"][i:i + 1].to(self.device)
                    mask = self.eval_enc["attention_mask"][i:i + 1].to(self.device)
                    lbls = self.eval_enc["input_ids"][i:i + 1].clone().to(self.device)
                    out = model(input_ids=ids, attention_mask=mask, labels=lbls)
                    eval_losses.append(out.loss.item())
                    del out
            torch.cuda.empty_cache()
            avg_eval = sum(eval_losses) / max(len(eval_losses), 1)
            eval_ppl = compute_perplexity(avg_eval)

            # Record final soft flags as state
            final_flags = {}
            if adapter_modules and hasattr(adapter_modules[0].experts[-1], 'get_soft_flags'):
                final_flags = adapter_modules[0].experts[-1].get_soft_flags()

            state = AdapterState(
                flags={k: v > 0.5 for k, v in final_flags.items()} if final_flags else soft_flags,
                poly_order=poly_order,
                loss_history=losses,
                age_steps=(seg_idx + 1) * self.args.steps_per_segment,
                eval_ppl_before=history[-1].eval_ppl_after if history else 0,
                eval_ppl_after=eval_ppl,
                improvement=(history[-1].eval_ppl_after - eval_ppl) if history else 0,
            )
            history.append(state)
            active_now = [f"{k}:{v:.2f}" for k, v in final_flags.items() if v > 0.1] if final_flags else []
            print(f"      ppl={eval_ppl:.0f} flags={{' '.join(active_now)}}", flush=True)

            del opt
            gc.collect()
            torch.cuda.empty_cache()

        final_ppl = history[-1].eval_ppl_after if history else float("inf")
        fitness = -final_ppl

        del model, adapter_modules
        gc.collect()
        torch.cuda.empty_cache()

        return fitness, [{
            "segment": i + 1, "flags": s.flags, "poly_order": s.poly_order,
            "eval_ppl": s.eval_ppl_after, "improvement": s.improvement,
        } for i, s in enumerate(history)]


def main():
    parser = argparse.ArgumentParser(description="ES optimization over adapter lifecycles")
    parser.add_argument("--n-segments", type=int, default=3)
    parser.add_argument("--steps-per-segment", type=int, default=10)
    parser.add_argument("--num-texts", type=int, default=30)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop-size", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--output", type=str, default="meta_evolution_results.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    controller = MetaController(d_model=32, nhead=2, num_layers=2)
    evolution = AdapterEvolution(controller, pop_size=args.pop_size, device="cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    runner = EvolutionRunner(args, args.device)

    all_generations = []

    for gen in range(args.generations):
        print(f"\n{'='*60}")
        print(f"  Generation {gen + 1}/{args.generations}")
        print(f"{'='*60}")

        gen_fitness = []
        gen_trajectories = []

        for i in range(args.pop_size):
            print(f"\n  Individual {i + 1}/{args.pop_size}")
            evolution._set_params(evolution.base + evolution.population[i])

            t0 = time.time()
            fitness, trajectory = runner.evaluate_individual(controller, tokenizer)
            elapsed = time.time() - t0

            evolution.record_fitness(i, fitness)
            gen_fitness.append(fitness)
            gen_trajectories.append(trajectory)
            print(f"    Fitness: {fitness:.0f} ({elapsed:.0f}s)")

        best_fitness, best_params = evolution.evolve()
        evolution.base = best_params.clone()

        print(f"\n  Gen {gen + 1} summary:")
        print(f"    Fitnesses: {[f'{f:.0f}' for f in gen_fitness]}")
        print(f"    Best: {best_fitness:.0f}")

        # Extract best trajectory
        best_idx = int(torch.argmax(evolution.fitness).item()) if gen == args.generations - 1 else 0
        if gen == args.generations - 1:
            evolution._set_params(best_params)
            _, best_traj = runner.evaluate_individual(controller, tokenizer)
        else:
            best_traj = gen_trajectories[np.argmax(gen_fitness)]

        gen_record = {
            "generation": gen + 1,
            "fitnesses": gen_fitness,
            "best_fitness": best_fitness,
            "best_trajectory": best_traj,
            "population_variance": float(evolution.population.var().item()),
        }
        all_generations.append(gen_record)

        with open(args.output, "w") as f:
            json.dump({
                "config": vars(args),
                "generations": all_generations,
            }, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  EVOLUTION COMPLETE")
    print(f"{'='*60}")
    best_gen = max(all_generations, key=lambda g: g["best_fitness"])
    print(f"  Best generation: {best_gen['generation']} (fitness {best_gen['best_fitness']:.0f})")
    print(f"  Best lifecycle:")
    for seg in best_gen["best_trajectory"]:
        active = [k for k, v in seg["flags"].items() if v]
        print(f"    Seg {seg['segment']}: {', '.join(active) if active else 'plain'} → eval_ppl {seg['eval_ppl']:.0f}")


if __name__ == "__main__":
    main()
