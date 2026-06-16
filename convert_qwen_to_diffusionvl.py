#!/usr/bin/env python3
"""Convert a pretrained Qwen2.5 checkpoint into DiffusionVL block-diffusion format.

Usage:
    python convert_qwen_to_diffusionvl.py \
        --source Qwen/Qwen2.5-0.5B-Instruct \
        --dest ./checkpoints/qwen0.5b-diffusionvl \
        [--block-size 8] [--mask-token "[MASK]"]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.diffusion_vlm import (
    DiffVLMConfig,
    BlockDiffusionQwenForCausalLM,
    add_mask_token_and_resize,
)


def convert(args: argparse.Namespace) -> None:
    print(f"[1/5] Loading source model: {args.source}")
    src = AutoModelForCausalLM.from_pretrained(
        args.source,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print(f"[2/5] Loading tokenizer: {args.source}")
    tokenizer = AutoTokenizer.from_pretrained(args.source)

    mask_token_id = add_mask_token_and_resize(tokenizer, args.mask_token)
    print(f"  MASK token ID: {mask_token_id}")

    cfg = src.config

    diff_config = DiffVLMConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=getattr(cfg, 'rms_norm_eps', 1e-6),
        rope_theta=getattr(cfg, 'rope_theta', 1000000.0),
        block_size=args.block_size,
        mask_token_id=mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    print(f"[3/5] Building diffusion model (same architecture)...")
    model = BlockDiffusionQwenForCausalLM(cfg, diff_config=diff_config)
    model.to(dtype=torch.bfloat16)

    print(f"[4/5] Copying weights from source...")
    src_state = src.state_dict()
    own_state = model.state_dict()

    copied = 0
    skipped = 0
    for k in own_state:
        if k in src_state and own_state[k].shape == src_state[k].shape:
            own_state[k].copy_(src_state[k].to(dtype=torch.bfloat16))
            copied += 1
        elif 'mask_embedding' in k:
            skipped += 1
        else:
            print(f"  Warning: shape mismatch on {k}")
            print(f"    model:  {own_state[k].shape}")
            print(f"    source: {src_state[k].shape if k in src_state else 'N/A'}")
            skipped += 1

    print(f"  Copied {copied} tensors, skipped {skipped} (mask_embedding + mismatches).")

    # Initialize mask embedding with mean token embedding
    with torch.no_grad():
        avg = model.model.embed_tokens.weight.mean(dim=0, keepdim=True)
        model.mask_embedding.weight.copy_(avg)

    print(f"[5/5] Saving to {args.dest}")
    os.makedirs(args.dest, exist_ok=True)

    model.save_pretrained(args.dest)
    tokenizer.save_pretrained(args.dest)

    cfg_dict = cfg.to_dict()
    cfg_dict["diffusion_config"] = {
        "block_size": diff_config.block_size,
        "mask_token_id": diff_config.mask_token_id,
        "noise_schedule_eps": diff_config.noise_schedule_eps,
        "antithetic_sampling": diff_config.antithetic_sampling,
    }

    with open(os.path.join(args.dest, "diffusion_config.json"), "w") as f:
        json.dump(cfg_dict["diffusion_config"], f, indent=2)

    print(f"Done. Model saved to {args.dest}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Block size: {diff_config.block_size}")
    print(f"  Mask token: '{args.mask_token}' (ID {mask_token_id})")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen2.5 → DiffusionVL (BD3-LM block diffusion)"
    )
    parser.add_argument("--source", type=str, required=True,
                        help="HF model name or local path (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--dest", type=str, required=True,
                        help="Output directory for diffusion model")
    parser.add_argument("--block-size", type=int, default=8,
                        help="BD3-LM block size (default: 8)")
    parser.add_argument("--mask-token", type=str, default="[MASK]",
                        help="Mask token string to add to tokenizer")
    args = parser.parse_args()

    convert(args)


if __name__ == "__main__":
    main()
