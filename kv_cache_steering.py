#!/usr/bin/env python3
"""KV-cache steering: modify KV cache entries using velocity predictions.

The geometric problem with last-hidden-state steering:
  - Perceiver/Transformer predicts v[l] = h[l+1] - h[l] (inter-layer dynamics)
  - LM head projects h[last] → logits (intra-token space)
  - These subspaces are geometrically orthogonal — velocity doesn't change argmax

KV-cache steering solves this:
  - Future tokens attend to the KV cache entries for all past tokens
  - Modifying KV entries changes what future tokens "see"
  - This IS in the right geometric space for affecting token selection

Method:
  1. After each token is generated, compute velocity for all layers
  2. For each layer l: h'[l] = h[l] + α * v[l]
  3. Compute new keys/values: k' = W_k(h'), v' = W_v(h')
  4. Replace KV cache entries for the last position
  5. Future tokens attend to steered representations
"""
from __future__ import annotations

import gc
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
TRANSFORMER_PATH = "best_transformer_25k.pt"
# TRY: alpha ∈ {0.0 (baseline), 0.1, 0.5, 1.0, 2.0, 5.0}


def steer_kv_cache(
    model, hidden_states: tuple, velocity: torch.Tensor,
    past_key_values, alpha: float = 1.0,
) -> tuple:
    """Modify the last position's KV cache entries using velocity predictions.

    Args:
        model: HuggingFace model
        hidden_states: tuple of (B, S, D) from output_hidden_states=True
        velocity: (B, L-1, D) from TrajectoryTransformer. v[l] = predicted h[l+1] - h[l]
        past_key_values: existing KV cache (tuple of tuples)
        alpha: steering strength
    Returns:
        modified past_key_values
    """
    B, S, D = hidden_states[0].shape
    L = velocity.shape[1]  # 23 transitions (h0→h1 through h22→h23)

    # h_actual[l] = hidden states at position S-1 for layer l
    # For layers 0..L (0..23), we have hidden_states[l] at the last position
    new_pkv = []

    for layer_idx in range(L):
        # Get the actual hidden state for this layer's output
        h_actual = hidden_states[layer_idx + 1][0, -1, :]  # (D,)
        v = velocity[0, layer_idx, :]  # (D,)

        # Steered hidden state
        h_steered = h_actual + alpha * v

        # Project through this layer's key and value projections
        layer = model.model.layers[layer_idx]
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj

        k_steered = k_proj(h_steered.to(k_proj.weight.dtype))  # (n_kv_heads * head_dim,)
        v_steered = v_proj(h_steered.to(v_proj.weight.dtype))

        # Reshape to (1, n_kv_heads, head_dim)
        n_kv_heads = layer.self_attn.num_key_value_heads
        head_dim = layer.self_attn.head_dim
        k_steered = k_steered.view(1, n_kv_heads, head_dim).unsqueeze(0)  # (1, 1, n_kv_heads, head_dim)
        v_steered = v_steered.view(1, n_kv_heads, head_dim).unsqueeze(0)

        # Get the existing KV cache entry for this layer
        k_cache, v_cache = past_key_values[layer_idx]
        # Shape: (1, n_kv_heads, seq_len, head_dim)

        # Replace the last position
        k_cache[0, :, -1:, :] = k_steered[0, :, :, :]
        v_cache[0, :, -1:, :] = v_steered[0, :, :, :]

        new_pkv.append((k_cache, v_cache))

    return tuple(new_pkv)


def demo(alpha: float = 0.5, n_steps: int = 5):
    print(f"KV-Cache Steering Demo (alpha={alpha})")
    print("=" * 60)

    print("Loading model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    print("Loading TrajectoryTransformer...")
    tt = TrajectoryTransformer(d_model=512, n_layers=6, n_heads=8, d_ff=2048)
    tt.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE))
    tt.to(DEVICE)
    tt.eval()

    prompt = "Q: What is 17 × 23?\nA: Let me solve this step by step.\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    generated = inputs.input_ids.clone()

    print(f"\nPrompt: {prompt}")
    print(f"Steering at each step with alpha={alpha}")
    print(f"\nGenerated tokens:")

    past = None
    for step in range(n_steps):
        with torch.no_grad():
            out = model(generated, past_key_values=past, use_cache=True, output_hidden_states=True)

        logits = out.logits[0, -1, :]
        next_token = logits.argmax().item()

        # Get hidden states for last position
        hs = out.hidden_states
        h_pos = torch.stack([h[0, -1, :].float() for h in hs], dim=0)

        # Get velocity from TrajectoryTransformer
        x = h_pos[:23].unsqueeze(0).to(DEVICE)
        v = tt(x)  # (1, 23, 2048)

        # Token before steering
        token_before = tokenizer.decode(next_token)

        # Steer KV cache
        past = steer_kv_cache(model, hs, v, out.past_key_values, alpha=alpha)

        # Now generate the next token with steered cache
        # The "next token" is already determined by logits above, but let's
        # see if the NEXT forward pass (with steered cache) produces different logits

        # Actually, the logits above were WITHOUT steering. Let's re-forward
        # with the steered cache to see if the logits change
        with torch.no_grad():
            # Forward just the last token with the steered KV cache
            next_input = torch.tensor([[next_token]], device=DEVICE)
            out2 = model(next_input, past_key_values=past, use_cache=True, output_hidden_states=False)

        logits_after = out2.logits[0, 0, :]  # Only one token generated
        token_after = logits_after.argmax().item()

        token_before_str = tokenizer.decode(next_token)
        token_after_str = tokenizer.decode(token_after)
        changed = next_token != token_after

        print(f"  Step {step}: '{token_before_str}' → '{token_after_str}' {'✓' if changed else ' '}")

        # Update for next iteration: use the token from steered logits
        generated = torch.cat([generated, torch.tensor([[token_after]], device=DEVICE)], dim=1)
        past = out2.past_key_values

        gc.collect()
        torch.cuda.empty_cache()

    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nResult: {result[len(prompt):]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    demo(alpha=args.alpha, n_steps=args.steps)
