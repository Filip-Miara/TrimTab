from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2Model, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


# ═══════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════

@dataclass
class DiffVLMConfig:
    """BD3-LM block diffusion configuration for Qwen2.5 conversion."""

    # Model dimensions (auto-filled from source Qwen)
    hidden_size: int = 896
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    intermediate_size: int = 4864
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # BD3-LM block diffusion params
    block_size: int = 8
    mask_token_id: int = 151671
    eos_token_id: int = 151645
    pad_token_id: int = 151643

    # Noise schedule
    noise_schedule_eps: float = 1e-3
    antithetic_sampling: bool = True
    sampling_eps_min: float = 1e-3
    sampling_eps_max: float = 1.0

    # Decoding
    default_gen_length: int = 256
    default_steps: int = 8
    default_temperature: float = 0.0
    remasking_strategy: Literal["low_confidence_static", "low_confidence_dynamic"] = "low_confidence_static"
    confidence_threshold: float = 0.85


# ═══════════════════════════════════════════════
# Noise Schedule
# ═══════════════════════════════════════════════

class LogLinearNoise(nn.Module):
    """Log-linear noise schedule from BD3-LM.

    σ(t) = -log(1 - (1-ε)·t)
    Instanteous rate: (1-ε)/(1-(1-ε)·t)
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-(1 - self.eps) * t)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loss_scale = self.rate_noise(t) / (1 - t)
        p = t
        return loss_scale, p


# ═══════════════════════════════════════════════
# Attention Mask Builder
# ═══════════════════════════════════════════════

def build_block_diffusion_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Build the BD3-LM attention mask for [noised | clean] concatenated input.

    The mask has shape (2L, 2L) where first L = noised, second L = clean.

    Rules (from paper Eq. 5):
    - Same block, same side (noised↔noised or clean↔clean): bidirectional ✓
    - Cross-side: noised attends to earlier clean blocks ✓
    - Clean side: attends to earlier clean blocks (causal) ✓
    - Noised attends to noised in earlier blocks: NO ✗
    - Clean attends to noised: NO ✗
    """
    total_len = seq_len * 2
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)

    q_idx = torch.arange(total_len, device=device)[:, None]
    kv_idx = torch.arange(total_len, device=device)[None, :]

    q_noised = q_idx < seq_len
    kv_noised = kv_idx < seq_len
    q_clean = ~q_noised
    kv_clean = ~kv_noised

    q_block = torch.where(q_noised, q_idx, q_idx - seq_len) // block_size
    kv_block = torch.where(kv_noised, kv_idx, kv_idx - seq_len) // block_size

    # 1) Same-block, same-side bidirectional
    same_block = q_block == kv_block
    same_side = q_noised == kv_noised
    mask |= same_block & same_side

    # 2) Noised attends to earlier clean blocks (offset block-causal)
    mask |= q_noised & kv_clean & (q_block > kv_block)

    # 3) Clean attends to earlier or same clean blocks (block-causal)
    mask |= q_clean & kv_clean & (q_block >= kv_block)

    return mask


# ═══════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════

class BlockDiffusionQwenModel(Qwen2Model):
    """Qwen2 backbone with block-diffusion non-causal attention."""

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.is_causal = False


class BlockDiffusionQwenForCausalLM(PreTrainedModel):
    """Qwen2.5 converted to BD3-LM block diffusion paradigm.

    Keeps exact same architecture (same weights). Changes:
    - Attention: causal → block-diagonal (intra-block bidirectional)
    - Training: next-token CE → masked diffusion CE on noised positions
    - Inference: AR → iterative block-level denoising with KV-cache reuse
    """

    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen2Config, diff_config: DiffVLMConfig | None = None):
        super().__init__(config)
        self.diff_config = diff_config or DiffVLMConfig()

        self.model = BlockDiffusionQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.noise_scheduler = LogLinearNoise(eps=self.diff_config.noise_schedule_eps)

        self.mask_embedding = nn.Embedding(1, config.hidden_size)
        self._mask_embed_initialized = False

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    # ─── Weight tying ──────────────────────────────────────

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    # ─── Noise application ─────────────────────────────────

    def _sample_block_noise(
        self,
        batch_size: int,
        seq_len: int,
        labels: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample block-wise noise levels and compute mask indices + loss scales.

        Returns:
            move_indices: (B, L) bool — which tokens are masked
            loss_scale: (B, L) — per-token loss reweighting
            avg_noise_level: float scalar
        """
        block_size = self.diff_config.block_size
        num_blocks = math.ceil(seq_len / block_size)
        dc = self.diff_config

        eps_b = torch.rand(batch_size, num_blocks, device=device)

        if dc.antithetic_sampling:
            num_samples = eps_b.numel()
            offset = torch.arange(num_samples, device=device).view(eps_b.shape) / num_samples
            eps_b = (eps_b / num_samples + offset) % 1

        t = eps_b.repeat_interleave(block_size, dim=-1)[:, :seq_len]
        t = t * (dc.sampling_eps_max - dc.sampling_eps_min) + dc.sampling_eps_min

        loss_scale, p = self.noise_scheduler(t)

        move_prob = torch.rand(batch_size, seq_len, device=device)
        text_mask = labels != -100
        move_indices = (move_prob <= p) & text_mask

        avg_noise = move_indices.float().mean().item()
        return move_indices, loss_scale, avg_noise

    def _apply_noise(
        self,
        inputs_embeds: torch.Tensor,
        move_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Replace masked positions with mask embedding."""
        if not self._mask_embed_initialized:
            with torch.no_grad():
                avg_token = inputs_embeds.mean(dim=(0, 1))
                self.mask_embedding.weight.data.copy_(avg_token)
            self._mask_embed_initialized = True

        mask_emb = self.mask_embedding.weight.unsqueeze(0)  # [1, 1, H]
        noised = torch.where(move_indices.unsqueeze(-1), mask_emb, inputs_embeds)
        return noised

    # ─── Forward (training) ────────────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        device = input_ids.device if input_ids is not None else self.device
        inputs_embeds = self.model.embed_tokens(input_ids)

        B, L = inputs_embeds.shape[:2]

        move_indices, loss_scale, avg_noise = self._sample_block_noise(
            B, L, labels, device
        )

        noised_embeds = self._apply_noise(inputs_embeds, move_indices)

        bd3lm_inputs = torch.cat([noised_embeds, inputs_embeds], dim=1)

        attn_mask = build_block_diffusion_mask(
            L, self.diff_config.block_size, device, dtype=torch.bool
        )
        attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0)

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        position_ids = torch.cat([pos, pos], dim=1)

        outputs = self.model(
            inputs_embeds=bd3lm_inputs,
            attention_mask=attn_mask_4d.expand(B, -1, -1, -1).to(inputs_embeds.dtype),
            position_ids=position_ids,
            return_dict=True,
        )

        hidden = outputs.last_hidden_state[:, :L]
        logits = self.lm_head(hidden).float()

        loss = None
        if labels is not None:
            loss = self._compute_bd3lm_loss(logits, labels, move_indices, loss_scale)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _compute_bd3lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        move_indices: torch.Tensor,
        loss_scale: torch.Tensor,
    ) -> torch.Tensor:
        masked = move_indices & (labels != -100)

        if not masked.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        ce = F.cross_entropy(logits[masked], labels[masked], reduction="none")
        scaled = ce * loss_scale[masked].abs()

        prompt_mask = (labels == -100).float()
        noisy_len = prompt_mask.shape[1] - prompt_mask.sum(dim=-1, keepdim=True)
        noisy_len = noisy_len.clamp(min=1)
        noisy_len_flat = noisy_len.expand_as(labels)[masked]

        loss = (scaled / noisy_len_flat).sum() / labels.shape[0]
        return loss

    # ─── Inference (block decoding) ────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        gen_length: int | None = None,
        steps: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> torch.LongTensor:
        """Block diffusion decoding (simplified, no KV-cache).

        Fills a fixed-length sequence block-by-block, denoising each block
        iteratively with the block diffusion attention mask.
        """
        dc = self.diff_config
        gen_length = gen_length or dc.default_gen_length
        steps = steps or dc.default_steps
        temperature = temperature if temperature is not None else dc.default_temperature

        device = input_ids.device
        B, prompt_len = input_ids.shape
        block_size = dc.block_size
        mask_id = dc.mask_token_id
        eos_id = dc.eos_token_id

        num_blocks = math.ceil((prompt_len + gen_length) / block_size)
        total_len = num_blocks * block_size

        x_ids = torch.full((B, total_len), mask_id, dtype=torch.long, device=device)
        x_embeds = self.model.embed_tokens(x_ids)

        prompt_embeds = self.model.embed_tokens(input_ids)
        x_embeds[:, :prompt_len] = prompt_embeds
        prompt_logits = self.lm_head(prompt_embeds)
        x_ids[:, :prompt_len] = prompt_logits.argmax(dim=-1)

        num_transfer = self._num_transfer_tokens(block_size, steps)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        def build_gen_mask(
            seq_len: int, block_start: int, block_end: int,
        ) -> torch.Tensor:
            """Block diffusion mask for generation.

            Full size: (seq_len, seq_len).
            - indices < block_start: clean context (causal)
            - indices in [block_start, block_end): current block (bidirectional)
            - indices >= block_end: future (not attended to)
            """
            m = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            for q in range(seq_len):
                if q < block_start:
                    for k in range(seq_len):
                        m[q, k] = (k <= q)
                elif q < block_end:
                    for k in range(seq_len):
                        if k < block_start:
                            m[q, k] = True  # clean context
                        elif k < block_end:
                            m[q, k] = True  # intra-block bidirectional
                        else:
                            m[q, k] = False
                else:
                    m[q, :] = False
            return m

        for block_idx in range(num_blocks):
            bs = block_idx * block_size
            be = min(bs + block_size, total_len)
            block_len = be - bs

            if be <= prompt_len:
                continue

            cur_embeds = x_embeds[:, bs:be].clone()
            cur_ids = x_ids[:, bs:be].clone()

            for step in range(steps + 1):
                is_mask = (cur_ids == mask_id)
                if not is_mask.any():
                    break

                gen_mask = build_gen_mask(be, bs, be)
                gen_mask_4d = gen_mask.unsqueeze(0).unsqueeze(0).to(x_embeds.dtype)
                gen_mask_4d = gen_mask_4d.masked_fill(~gen_mask, torch.finfo(x_embeds.dtype).min)

                full_embeds = torch.cat([x_embeds[:, :bs], cur_embeds], dim=1)
                pos_ids_full = torch.arange(be, device=device).unsqueeze(0).expand(B, -1)

                outputs = self.model(
                    inputs_embeds=full_embeds,
                    attention_mask=gen_mask_4d.expand(B, -1, -1, -1),
                    position_ids=pos_ids_full,
                    return_dict=True,
                )

                cur_logits = self.lm_head(outputs.last_hidden_state[:, -block_len:]).float()

                if step < steps:
                    x0, x0_p = self._sample_tokens(cur_logits, temperature)

                    for b in range(B):
                        if done[b]:
                            continue
                        n_keep = min(num_transfer[step].item(), is_mask[b].sum().item())
                        if n_keep == 0:
                            continue

                        conf = torch.where(is_mask[b], x0_p[b], torch.tensor(-torch.inf, device=device))
                        _, idx = torch.topk(conf, n_keep)

                        cur_ids[b, idx] = x0[b, idx]
                        x0_e = self.model.embed_tokens(x0[b:b+1, idx])
                        cur_embeds[b:b+1, idx] = x0_e

                    if (cur_ids == eos_id).any():
                        done = done | (cur_ids == eos_id).any(dim=1)

            x_ids[:, bs:be] = cur_ids
            x_embeds[:, bs:be] = cur_embeds

            if done.all():
                break

        return x_ids[:, prompt_len:prompt_len + gen_length]

    def _num_transfer_tokens(self, block_length: int, steps: int) -> torch.Tensor:
        if steps == 0:
            return torch.zeros(0, dtype=torch.int64)
        base = block_length // steps
        rem = block_length % steps
        n = torch.zeros(steps + 1, dtype=torch.int64) + base
        n[:rem] += 1
        return n

    def _sample_tokens(
        self, logits: torch.Tensor, temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)

        if temperature == 0.0:
            tokens = logits.argmax(dim=-1)
        else:
            scaled = logits / temperature
            probs_mod = F.softmax(scaled, dim=-1)
            tokens = torch.multinomial(probs_mod.view(-1, probs_mod.shape[-1]), 1).view(*probs.shape[:-1])

        conf = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        return tokens, conf


# ═══════════════════════════════════════════════
# Conversion helpers
# ═══════════════════════════════════════════════

def convert_ar_to_diffusion(
    source_path: str,
    diff_config: DiffVLMConfig | None = None,
    device: str = "cpu",
) -> BlockDiffusionQwenForCausalLM:
    """Load a pretrained Qwen2.5 checkpoint and wrap it for BD3-LM diffusion.

    The architecture is identical — only attention pattern + loss changes.
    This copies all weights from the AR model into the diffusion wrapper.
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading source AR model from {source_path} ...")
    src = AutoModelForCausalLM.from_pretrained(
        source_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    qwen_config = src.config
    dc = diff_config or DiffVLMConfig()
    dc.hidden_size = qwen_config.hidden_size
    dc.num_hidden_layers = qwen_config.num_hidden_layers
    dc.num_attention_heads = qwen_config.num_attention_heads
    dc.num_key_value_heads = qwen_config.num_key_value_heads
    dc.intermediate_size = qwen_config.intermediate_size
    dc.vocab_size = qwen_config.vocab_size
    dc.max_position_embeddings = qwen_config.max_position_embeddings
    if hasattr(qwen_config, 'rms_norm_eps'):
        dc.rms_norm_eps = qwen_config.rms_norm_eps
    if hasattr(qwen_config, 'rope_theta'):
        dc.rope_theta = qwen_config.rope_theta

    print("Creating diffusion model with same architecture...")
    model = BlockDiffusionQwenForCausalLM(qwen_config, diff_config=dc)
    model.to(dtype=torch.bfloat16, device=device)

    print("Copying weights (architecture is identical)...")
    src_state = src.state_dict()
    own_state = model.state_dict()

    missing = set(own_state.keys()) - set(src_state.keys())
    extra = set(src_state.keys()) - set(own_state.keys())

    for k in own_state:
        if k in src_state and own_state[k].shape == src_state[k].shape:
            own_state[k].copy_(src_state[k])
        elif 'mask_embedding' not in k:
            print(f"  Shape mismatch or missing: {k} — own {own_state[k].shape} vs src {src_state.get(k, 'N/A').shape if isinstance(src_state.get(k), torch.Tensor) else 'N/A'}")

    if missing:
        print(f"  Missing keys in source (new params): {sorted(missing)}")
    if extra:
        print(f"  Extra keys in source (not copied): {sorted(extra)[:10]}...")

    print("Conversion complete.")
    return model


def add_mask_token_and_resize(tokenizer, mask_token: str = "[MASK]") -> int:
    """Add a mask token to the tokenizer if not present, return its ID."""
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
    return tokenizer.convert_tokens_to_ids(mask_token)
