#!/usr/bin/env python3
"""Meta-evolution over adapter lifecycles.

The MetaController observes the full history of adapter configurations and
their outcomes, then decides the best config for the next segment.

If --use-metacontroller is set, the controller learns via ES across trials.
Otherwise, a simple schedule (PERA→BVA) is used as baseline.
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
from src.adapters.adapter_evolution import FLAG_NAMES
from src.adapters.gradient_decomposition import TaylorContribution
from src.adapters.adapter_evolution import (
    AdapterState, MetaController, AdapterEvolution, LifecycleConfig,
)
from src.training.trainer import compute_perplexity

MODEL_PATH = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub"),
    "models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
)
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
ARXIV_PARQUET_DIR = "/run/media/filip/B522-875D/Datasets/hub/datasets--ccdv--arxiv-summarization/snapshots/240aaf1a969b3f8cd0ade6986bfad0cd730ee288/section"


def load_arxiv_data(num_texts=50):
    parq_files = sorted(
        os.path.join(ARXIV_PARQUET_DIR, f)
        for f in os.listdir(ARXIV_PARQUET_DIR)
        if f.endswith(".parquet")
    )
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


def build_adapter_kwargs(flags: dict[str, bool], poly_order: int) -> dict:
    return {
        "expert_variant": "hybrid",
        "bidirectional": flags.get("bidirectional", False),
        "use_vectors": flags.get("use_vectors", False),
        "use_norm": flags.get("use_norm", False),
        "use_gate": flags.get("use_gate", False),
        "use_activation": flags.get("use_activation", False),
        "use_autoencoder": flags.get("use_autoencoder", False),
        "use_polynomial": flags.get("use_polynomial", False),
        "poly_order": poly_order,
        "n_latents": 16,
        "d_latent": 32,
        "d_key": 64,
        "top_m": 4,
        "max_experts": 200,
        "abs_steps": 4,
    }


def main():
    parser = argparse.ArgumentParser(description="Meta-evolution over adapter lifecycles")
    parser.add_argument("--n-segments", type=int, default=5)
    parser.add_argument("--steps-per-segment", type=int, default=20)
    parser.add_argument("--num-texts", type=int, default=50)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="lifecycle_results.json")

    # MetaController / schedule
    parser.add_argument("--use-metacontroller", action="store_true")
    parser.add_argument("--schedule-type", type=str, default="constant",
                        choices=["constant", "linear", "threshold", "pera_to_bva"])
    parser.add_argument("--switch-segment", type=int, default=3)
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    import random

    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=args.device if args.device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
    model.to(args.device)

    print(f"Loading data...")
    train_texts, eval_texts = load_arxiv_data(args.num_texts)
    print(f"  {args.n_segments} segments, {len(eval_texts)} eval texts")

    # Build config based on current segment
    def get_config_for_segment(seg_idx: int, history: list) -> dict:
        if args.use_metacontroller and history:
            return history[-1].flags  # controller decides externally
        if args.schedule_type == "pera_to_bva":
            if seg_idx <= 1:
                return {"use_polynomial": True}
            elif seg_idx <= 3:
                return {"use_vectors": True, "use_norm": True}
            else:
                return {"bidirectional": True, "use_vectors": True, "use_norm": True}
        if args.schedule_type == "linear":
            t = min(1.0, seg_idx / args.switch_segment)
            return {"bidirectional": t > 0.5, "use_vectors": t > 0.3, "use_norm": t > 0.7}
        return {}  # constant — all False

    # Initialize MetaController if needed
    lifecycle_ctrl = None
    mc = None
    if args.use_metacontroller:
        mc = MetaController(d_model=32, nhead=2, num_layers=2).to(args.device)
        lifecycle_ctrl = LifecycleConfig(use_metacontroller=True)

    # Streaming loop
    adapter_modules = []
    history: list[AdapterState] = []

    for seg_idx in range(args.n_segments):
        print(f"\n{'─'*50}\n  Segment {seg_idx + 1}/{args.n_segments}\n{'─'*50}")

        # Decide config
        if args.use_metacontroller and mc is not None:
            flags, poly_order, _ = mc(history, temperature=1.0)
        else:
            f = get_config_for_segment(seg_idx, history)
            flags = {k: f.get(k, False) for k in FLAG_NAMES}
            poly_order = 2

        active_flags = [k for k, v in flags.items() if v]
        print(f"  Config: {', '.join(active_flags) if active_flags else 'plain'}, poly={poly_order}")

        # Rebuild model adapters with new config
        for m in adapter_modules:
            m.reset()
        model.requires_grad_(False)
        # Need to re-inject since config changed
        if seg_idx == 0:
            config = AdapterConfig(
                r=args.r, lora_alpha=args.alpha, lora_dropout=0.0,
                target_modules=target_modules,
                extra_kwargs=build_adapter_kwargs(flags, poly_order),
            )
            adapter_modules = []
            for name, module in model.named_modules():
                if any(name.endswith(t) for t in target_modules) and isinstance(module, nn.Linear):
                    wrapped = adapt_linear_layer(module, StreamFusionLoRA, config)
                    parent_name, _, child_name = name.rpartition(".")
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                    setattr(parent, child_name, wrapped)
                    adapter_modules.append(wrapped.adapter)
            model.to(args.device)
            print(f"  Injected {len(adapter_modules)} adapters")
        else:
            # Update config for new experts
            for am in adapter_modules:
                if hasattr(am, 'sf'):
                    am.sf.bidirectional = flags.get("bidirectional", False)
                    am.sf.use_vectors = flags.get("use_vectors", False)
                    am.sf.use_norm = flags.get("use_norm", False)
                    am.sf.use_gate = flags.get("use_gate", False)
                    am.sf.use_activation = flags.get("use_activation", False)
                    am.sf.use_autoencoder = flags.get("use_autoencoder", False)
                    am.sf.use_polynomial = flags.get("use_polynomial", False)
                    am.sf.poly_order = poly_order

        # Add a new expert
        for am in adapter_modules:
            am.add_expert()

        # Train on current segment
        seg_text = train_texts[seg_idx % len(train_texts)]
        enc = tokenize_texts(tokenizer, [seg_text])
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc["attention_mask"].to(args.device)
        labels = enc["input_ids"].clone().to(args.device)

        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            print("  WARNING: no trainable params! Enabling adapter grads manually.")
            for am in adapter_modules:
                for p in am.parameters():
                    p.requires_grad_(True)
            trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")
        opt = torch.optim.Adam(trainable, lr=args.lr)

        losses = []
        for step in range(args.steps_per_segment):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad()

        avg_loss = sum(losses) / len(losses)
        train_ppl = compute_perplexity(avg_loss)

        # Eval
        model.eval()
        eval_losses = []
        with torch.no_grad():
            eenc = tokenize_texts(tokenizer, eval_texts)
            for i in range(len(eenc["input_ids"])):
                ids = eenc["input_ids"][i:i + 1].to(args.device)
                mask = eenc["attention_mask"][i:i + 1].to(args.device)
                lbls = eenc["input_ids"][i:i + 1].clone().to(args.device)
                out = model(input_ids=ids, attention_mask=mask, labels=lbls)
                eval_losses.append(out.loss.item())
                del out
        torch.cuda.empty_cache()
        avg_eval = sum(eval_losses) / max(len(eval_losses), 1)
        eval_ppl = compute_perplexity(avg_eval)

        # Compute Taylor contributions for the first adapter's first expert
        tc = TaylorContribution()
        first_expert = adapter_modules[0].experts[-1] if adapter_modules[0].experts.__len__() > 0 else None
        taylor_c = 0.0
        if first_expert is not None:
            try:
                # Need one more forward+backward for contributions
                loss2 = model(input_ids=input_ids[:1], attention_mask=attention_mask[:1], labels=labels[:1]).loss
                loss2.backward()
                contribs = tc.rank1_contributions(first_expert)
                taylor_c = sum(contribs.values()) if contribs else 0.0
            except Exception:
                pass

        # Record state
        state = AdapterState(
            flags=flags.copy(),
            poly_order=poly_order,
            loss_history=losses,
            grad_norm_history=[],
            taylor_contribution=taylor_c,
            age_steps=(seg_idx + 1) * args.steps_per_segment,
            eval_ppl_before=history[-1].eval_ppl_after if history else 0,
            eval_ppl_after=eval_ppl,
            improvement=(history[-1].eval_ppl_after - eval_ppl) if history else 0,
        )
        history.append(state)

        print(f"  Train PPL: {train_ppl:.1f} | Eval PPL: {eval_ppl:.1f} | Experts/adapt:{adapter_modules[0].experts.__len__()}")

        # Absorb + prune
        for am in adapter_modules:
            am.absorb()
            am.prune(threshold=0.02)

        del opt, outputs, loss
        gc.collect()
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Lifecycle Trajectory:")
    for i, s in enumerate(history):
        active = [k for k, v in s.flags.items() if v]
        print(f"  Seg {i+1}: {', '.join(active) if active else 'plain'} → eval_ppl {s.eval_ppl_after:.0f} ({s.improvement:+.0f})")

    results = {
        "config": vars(args),
        "history": [
            {"segment": i + 1, "flags": s.flags, "poly_order": s.poly_order,
             "eval_ppl": s.eval_ppl_after, "improvement": s.improvement}
            for i, s in enumerate(history)
        ],
        "final_eval_ppl": history[-1].eval_ppl_after if history else 0,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
