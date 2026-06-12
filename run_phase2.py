#!/usr/bin/env python3
"""Phase 2: Multi-layer velocity injection via forward hooks.

Injects velocity residuals at EARLY transformer layers (0, 6, 12, 18)
instead of only the last layer. Early perturbations get amplified by the
remaining 24-layer stack, producing larger logit-space effects.

Two-pass approach:
  Pass 1: Forward → get hidden states → Perceiver/Transformer → velocities
  Pass 2: Forward with velocity injection hooks → modified logits
  Compare: logit argmax changes at various α and injection layers.
"""
from __future__ import annotations

import gc, sys, time
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer
from src.adapters.thought_diffusion import ThoughtDiffusion

DEVICE = "cuda"
MODEL_PATH = "/run/media/filip/B522-875D/Datasets/hub/models--Qwen--Qwen3.5-2B/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
TT_PATH = "best_trajectory_transformer.pt"  # R²=0.62 model

def main():
    print("Loading model & tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    print("Loading TrajectoryTransformer...")
    tt = TrajectoryTransformer(d_model=512, n_layers=6, n_heads=8, d_ff=2048)
    tt.load_state_dict(torch.load(TT_PATH, map_location=DEVICE))
    tt.to(DEVICE)
    tt.eval()

    prompts = [
        "Q: What is 17 × 24?\nA: Let me solve this step by step.\n",
        "Q: If a train travels 120 miles in 2 hours, what is its speed?\nA: Let me solve this step by step.\n",
        "Q: Janet has 15 apples. She gives 7 to John and eats 3. How many does she have left?\nA: Let me solve this step by step.\n",
        "Q: A rectangle has length 8 and width 5. What is its area?\nA: Let me solve this step by step.\n",
        "Q: There are 24 students in a class. 1/3 are girls. How many are boys?\nA: Let me solve this step by step.\n",
    ]

    test_alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt[:60]}...")

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)

        # Pass 1: Get velocities
        with torch.no_grad():
            out1 = model(**inputs, output_hidden_states=True)
        hs = out1.hidden_states

        # Extract hidden states for the last token across all layers
        h_pos = torch.stack([h[0, -1, :].float() for h in hs], dim=0)  # (25, 2048)
        x = h_pos[:23].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            velocities = tt(x)[0].cpu()  # (23, 2048)

        orig_logits = out1.logits[0, -1, :]
        orig_token = orig_logits.argmax().item()

        print(f"  Original token: '{tok.decode(orig_token)}' (id={orig_token})")

        for alpha in test_alphas:
            if alpha == 0.0:
                print(f"  α=0.0: baseline (no injection)")
                continue

            # Track how many layers were actually injected
            layers_injected = []

            def make_hook(layer_idx):
                def hook(module, args, kwargs):
                    h = args[0]  # (B, S, D)
                    if layer_idx < velocities.shape[0]:
                        v = velocities[layer_idx].to(h.dtype).to(h.device)
                        h[:, -1, :] = h[:, -1, :] + alpha * v
                        layers_injected.append(layer_idx)
                    return ((h, *args[1:]) if len(args) > 1 else (h,)), kwargs
                return hook

            hooks = []
            for li in [0, 6, 12, 18]:
                hooks.append(model.model.layers[li].register_forward_pre_hook(
                    make_hook(li), with_kwargs=True
                ))

            # Pass 2: Forward with velocity injection
            layers_injected.clear()
            with torch.no_grad():
                out2 = model(**inputs, output_hidden_states=False)

            for h in hooks:
                h.remove()

            steered_logits = out2.logits[0, -1, :]
            steered_token = steered_logits.argmax().item()
            changed = steered_token != orig_token

            print(f"  α={alpha:.1f}: '{tok.decode(steered_token)}' (id={steered_token}) "
                  f"{'✓ CHANGED' if changed else ' '} | injected {len(layers_injected)} layers")

        gc.collect()
        torch.cuda.empty_cache()

    # Ablation: which layer matters most?
    print(f"\n{'='*60}")
    print("Layer ablation study")
    print(f"{'='*60}")
    prompt = prompts[0]
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states
    h_pos = torch.stack([h[0, -1, :].float() for h in hs], dim=0)
    x = h_pos[:23].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        velocities = tt(x)[0].cpu()
    orig_logits = out.logits[0, -1, :]
    orig_token = orig_logits.argmax().item()
    print(f"  Original: '{tok.decode(orig_token)}'")

    for alpha in [2.0]:
        for li in [0, 3, 6, 9, 12, 15, 18, 21]:
            def make_hook2(lidx):
                def hook(module, args, kwargs):
                    h = args[0]
                    if lidx < velocities.shape[0]:
                        v = velocities[lidx].to(h.dtype).to(h.device)
                        h[:, -1, :] = h[:, -1, :] + alpha * v
                    return ((h, *args[1:]) if len(args) > 1 else (h,)), kwargs
                return hook
            hooks = []
            hooks.append(model.model.layers[li].register_forward_pre_hook(
                make_hook2(li), with_kwargs=True
            ))
            with torch.no_grad():
                out2 = model(**inputs, output_hidden_states=False)
            for h in hooks: h.remove()
            st = out2.logits[0, -1, :].argmax().item()
            changed = st != orig_token
            print(f"  Layer {li:2d}: '{tok.decode(st)}' (id={st}) {'✓' if changed else ' '}")


if __name__ == "__main__":
    main()
