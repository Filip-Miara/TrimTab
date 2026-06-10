from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LowRankAdapter, AdapterConfig


ExpertVariant = Literal["plain", "dora", "bora", "vera", "hybrid"]


@dataclass
class StreamFusionConfig(AdapterConfig):
    expert_variant: ExpertVariant = "plain"
    n_latents: int = 32
    d_latent: int = 64
    d_key: int = 64
    d_expert_emb: int = 64
    n_heads: int = 4
    top_m: int = 8
    tau_routing: float = 1.0
    max_experts: int = 200
    abs_steps: int = 8
    beta_absorb: float = 0.5
    gamma_div: float = 0.1
    gamma_bal: float = 0.05
    ve_rank: int = 0
    dora_eps: float = 1e-5
    bidirectional: bool = False
    use_vectors: bool = False
    use_norm: bool = False
    use_gate: bool = False
    use_activation: bool = False
    use_autoencoder: bool = False
    use_polynomial: bool = False
    poly_order: int = 2
    anneal_rate: float = 0.05


class StreamExpert(ABC, nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, d_key: int, d_emb: int, scaling: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = scaling
        self.key = nn.Parameter(torch.randn(d_key) * 0.1)
        self.lambda_ = nn.Parameter(torch.tensor(1.0))
        self.embedding = nn.Parameter(torch.randn(d_emb) * 0.02)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def absorb_params(self) -> torch.Tensor:
        ...

    @classmethod
    @abstractmethod
    def absorb_dim(cls, in_features: int, out_features: int, rank: int, **kwargs) -> int:
        ...

    @classmethod
    @abstractmethod
    def variant_name(cls) -> ExpertVariant:
        ...


class PlainStreamExpert(StreamExpert):
    def __init__(self, in_features: int, out_features: int, rank: int, d_key: int, d_emb: int, scaling: float):
        super().__init__(in_features, out_features, rank, d_key, d_emb, scaling)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambda_ * self.scaling * (x @ self.lora_A.T @ self.lora_B.T)

    def absorb_params(self) -> torch.Tensor:
        return torch.cat([self.lora_A.detach().flatten(), self.lora_B.detach().flatten()])

    @classmethod
    def absorb_dim(cls, in_features: int, out_features: int, rank: int, **kwargs) -> int:
        return rank * (in_features + out_features)

    @classmethod
    def variant_name(cls) -> ExpertVariant:
        return "plain"


class DoRAStreamExpert(StreamExpert):
    def __init__(self, in_features: int, out_features: int, rank: int, d_key: int, d_emb: int, scaling: float, eps: float = 1e-5):
        super().__init__(in_features, out_features, rank, d_key, d_emb, scaling)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.magnitude = nn.Parameter(torch.ones(out_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BA = self.lambda_ * self.scaling * (self.lora_B @ self.lora_A)
        col_norm = BA.norm(p=2, dim=1, keepdim=True)
        normalized = BA / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        return x @ adapted.T

    def absorb_params(self) -> torch.Tensor:
        return torch.cat([
            self.lora_A.detach().flatten(),
            self.lora_B.detach().flatten(),
            self.magnitude.detach().flatten(),
        ])

    @classmethod
    def absorb_dim(cls, in_features: int, out_features: int, rank: int, **kwargs) -> int:
        return rank * (in_features + out_features) + out_features

    @classmethod
    def variant_name(cls) -> ExpertVariant:
        return "dora"


class BoRAStreamExpert(StreamExpert):
    def __init__(self, in_features: int, out_features: int, rank: int, d_key: int, d_emb: int, scaling: float, eps: float = 1e-5):
        super().__init__(in_features, out_features, rank, d_key, d_emb, scaling)
        self.lora_A_fwd = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B_fwd = nn.Parameter(torch.zeros(out_features, rank))
        self.magnitude_fwd = nn.Parameter(torch.ones(out_features))
        self.lora_A_bwd = nn.Parameter(torch.randn(rank, out_features) / math.sqrt(rank))
        self.lora_B_bwd = nn.Parameter(torch.zeros(in_features, rank))
        self.magnitude_bwd = nn.Parameter(torch.ones(in_features))
        self.eps = eps

    def _dora_delta(self, BA, magnitude):
        col_norm = BA.norm(p=2, dim=1, keepdim=True)
        normalized = BA / (col_norm + self.eps)
        return magnitude[:, None] * normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BA_fwd = self.lambda_ * self.scaling * (self.lora_B_fwd @ self.lora_A_fwd)
        BA_bwd = self.lambda_ * self.scaling * (self.lora_B_bwd @ self.lora_A_bwd)
        delta_fwd = self._dora_delta(BA_fwd, self.magnitude_fwd)
        delta_bwd = self._dora_delta(BA_bwd, self.magnitude_bwd)
        return x @ delta_fwd.T + x @ delta_bwd

    def absorb_params(self) -> torch.Tensor:
        return torch.cat([
            self.lora_A_fwd.detach().flatten(),
            self.lora_B_fwd.detach().flatten(),
            self.magnitude_fwd.detach().flatten(),
            self.lora_A_bwd.detach().flatten(),
            self.lora_B_bwd.detach().flatten(),
            self.magnitude_bwd.detach().flatten(),
        ])

    @classmethod
    def absorb_dim(cls, in_features: int, out_features: int, rank: int, **kwargs) -> int:
        fwd = rank * in_features + out_features * rank + out_features
        bwd = rank * out_features + in_features * rank + in_features
        return fwd + bwd

    @classmethod
    def variant_name(cls) -> ExpertVariant:
        return "bora"


class VeRaStreamExpert(StreamExpert):
    def __init__(self, in_features: int, out_features: int, rank: int, d_key: int, d_emb: int, scaling: float, ve_rank: int = 0, eps: float = 1e-5):
        super().__init__(in_features, out_features, rank, d_key, d_emb, scaling)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.magnitude = nn.Parameter(torch.ones(out_features))
        vr = ve_rank if ve_rank > 0 else max(1, rank // 2)
        self.ve_rank = vr
        self.register_buffer("ve_A", torch.randn(vr, in_features) / math.sqrt(in_features))
        self.register_buffer("ve_B", torch.randn(out_features, vr) / math.sqrt(vr))
        self.ve_lambda = nn.Parameter(torch.randn(vr) * 0.01)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BA = self.lambda_ * self.scaling * (self.lora_B @ self.lora_A)
        VE = self.ve_B @ (self.ve_lambda[:, None] * self.ve_A)
        merged = BA + VE
        col_norm = merged.norm(p=2, dim=1, keepdim=True)
        normalized = merged / (col_norm + self.eps)
        adapted = self.magnitude[:, None] * normalized
        return x @ adapted.T

    def absorb_params(self) -> torch.Tensor:
        return torch.cat([
            self.lora_A.detach().flatten(),
            self.lora_B.detach().flatten(),
            self.magnitude.detach().flatten(),
            self.ve_lambda.detach().flatten(),
        ])

    @classmethod
    def absorb_dim(cls, in_features: int, out_features: int, rank: int, **kwargs) -> int:
        ve_r = kwargs.get("ve_rank", 0) or max(1, rank // 2)
        return rank * (in_features + out_features) + out_features + ve_r

    @classmethod
    def variant_name(cls) -> ExpertVariant:
        return "vera"


class HybridStreamExpert(StreamExpert):
    """Configurable expert supporting any combination of features."""

    def __init__(self, in_features: int, out_features: int, rank: int, d_key: int, d_emb: int, scaling: float,
                 bidirectional: bool = False, use_vectors: bool = False, use_norm: bool = False,
                 use_gate: bool = False, use_activation: bool = False, use_autoencoder: bool = False,
                 use_polynomial: bool = False, poly_order: int = 2,
                 ve_rank: int = 0, eps: float = 1e-5, anneal_rate: float = 0.05):
        super().__init__(in_features, out_features, rank, d_key, d_emb, scaling)
        self.bidirectional = bidirectional
        self.use_vectors = use_vectors
        self.use_norm = use_norm
        self.use_gate = use_gate
        self.use_activation = use_activation
        self.use_autoencoder = use_autoencoder
        self.use_polynomial = use_polynomial
        self.poly_order = poly_order
        self.eps = eps
        self.anneal_rate = anneal_rate
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

        self.lora_A_fwd = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B_fwd = nn.Parameter(torch.zeros(out_features, rank))
        self.magnitude_fwd = nn.Parameter(torch.ones(out_features))

        if bidirectional:
            self.lora_A_bwd = nn.Parameter(torch.randn(rank, out_features) / math.sqrt(rank))
            self.lora_B_bwd = nn.Parameter(torch.zeros(in_features, rank))
            self.magnitude_bwd = nn.Parameter(torch.ones(in_features))

        if use_vectors:
            vr = ve_rank if ve_rank > 0 else max(1, rank // 2)
            self.ve_rank_actual = vr
            self.register_buffer("ve_A_fwd", torch.randn(vr, in_features) / math.sqrt(in_features))
            self.register_buffer("ve_B_fwd", torch.randn(out_features, vr) / math.sqrt(vr))
            self.ve_lambda_fwd = nn.Parameter(torch.randn(vr) * 0.01)
            if bidirectional:
                self.register_buffer("ve_A_bwd", torch.randn(vr, out_features) / math.sqrt(out_features))
                self.register_buffer("ve_B_bwd", torch.randn(in_features, vr) / math.sqrt(vr))
                self.ve_lambda_bwd = nn.Parameter(torch.randn(vr) * 0.01)

        if use_autoencoder:
            hidden_r = max(1, rank * 2)
            self.ANL_fwd = nn.Sequential(
                nn.Linear(rank, hidden_r), nn.GELU(), nn.Linear(hidden_r, rank),
            )
            if bidirectional:
                self.ANL_bwd = nn.Sequential(
                    nn.Linear(rank, hidden_r), nn.GELU(), nn.Linear(hidden_r, rank),
                )
            # init near-identity
            for m in [self.ANL_fwd] + ([self.ANL_bwd] if bidirectional else []):
                nn.init.xavier_uniform_(m[0].weight, gain=0.1)
                nn.init.zeros_(m[0].bias)
                nn.init.xavier_uniform_(m[2].weight, gain=0.1)
                nn.init.zeros_(m[2].bias)

        if use_polynomial:
            self.poly_coeff_fwd = nn.Parameter(torch.randn(poly_order) * 0.01)
            if bidirectional:
                self.poly_coeff_bwd = nn.Parameter(torch.randn(poly_order) * 0.01)

        if use_norm:
            self.norm = nn.LayerNorm(out_features)

        if use_gate:
            self.gate = nn.Parameter(torch.tensor(1.0))

    def _get_alpha(self) -> float:
        t = self._step.item()
        return math.exp(-self.anneal_rate * max(0, t))

    def _apply_activation(self, A: torch.Tensor) -> torch.Tensor:
        if not self.use_activation:
            return A
        alpha = self._get_alpha()
        return (1 - alpha) * A + alpha * torch.tanh(A)

    def _compute_BA(self, A, B, ve_A=None, ve_B=None, ve_lambda=None, poly_coeffs=None, anl=None):
        A_act = self._apply_activation(A)

        if anl is not None:
            A_act = anl(A_act.T).T

        BA = B @ A_act

        if ve_A is not None and ve_lambda is not None:
            BA = BA + ve_B @ (ve_lambda[:, None] * ve_A)

        if poly_coeffs is not None:
            result = poly_coeffs[0] * BA
            for order in range(1, len(poly_coeffs)):
                result = result + poly_coeffs[order] * (BA ** (order + 1))
            BA = result

        return BA

    def _compute_delta_dir(self, A, B, magnitude, **extras):
        BA = self.lambda_ * self.scaling * self._compute_BA(A, B, **extras)
        col_norm = BA.norm(p=2, dim=1, keepdim=True)
        normalized = BA / (col_norm + self.eps)
        return magnitude[:, None] * normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extras_fwd = dict(
            ve_A=getattr(self, 've_A_fwd', None), ve_B=getattr(self, 've_B_fwd', None),
            ve_lambda=getattr(self, 've_lambda_fwd', None),
            poly_coeffs=getattr(self, 'poly_coeff_fwd', None),
            anl=getattr(self, 'ANL_fwd', None),
        )
        delta_fwd = self._compute_delta_dir(
            self.lora_A_fwd, self.lora_B_fwd, self.magnitude_fwd, **extras_fwd,
        )
        out = x @ delta_fwd.T

        if self.bidirectional:
            extras_bwd = dict(
                ve_A=getattr(self, 've_A_bwd', None), ve_B=getattr(self, 've_B_bwd', None),
                ve_lambda=getattr(self, 've_lambda_bwd', None),
                poly_coeffs=getattr(self, 'poly_coeff_bwd', None),
                anl=getattr(self, 'ANL_bwd', None),
            )
            delta_bwd = self._compute_delta_dir(
                self.lora_A_bwd, self.lora_B_bwd, self.magnitude_bwd, **extras_bwd,
            )
            out = out + x @ delta_bwd

        if self.use_norm:
            out = self.norm(out)

        if self.use_gate:
            out = out * torch.sigmoid(self.gate)

        self._step += 1
        return out

    def absorb_params(self) -> torch.Tensor:
        parts = [
            self.lora_A_fwd.detach().flatten(), self.lora_B_fwd.detach().flatten(),
            self.magnitude_fwd.detach().flatten(),
        ]
        if self.bidirectional:
            parts += [
                self.lora_A_bwd.detach().flatten(), self.lora_B_bwd.detach().flatten(),
                self.magnitude_bwd.detach().flatten(),
            ]
        if self.use_vectors:
            parts.append(self.ve_lambda_fwd.detach().flatten())
            if self.bidirectional:
                parts.append(self.ve_lambda_bwd.detach().flatten())
        if self.use_polynomial:
            parts.append(self.poly_coeff_fwd.detach().flatten())
            if self.bidirectional:
                parts.append(self.poly_coeff_bwd.detach().flatten())
        if self.use_autoencoder:
            for p in self.ANL_fwd.parameters():
                parts.append(p.detach().flatten())
            if self.bidirectional:
                for p in self.ANL_bwd.parameters():
                    parts.append(p.detach().flatten())
        if self.use_gate:
            parts.append(self.gate.detach().flatten())
        return torch.cat(parts)

    @classmethod
    def absorb_dim(cls, in_features: int, out_features: int, rank: int, **kwargs) -> int:
        total = rank * (in_features + out_features) + out_features
        if kwargs.get("bidirectional", False):
            total += rank * (out_features + in_features) + in_features
        if kwargs.get("use_vectors", False):
            vr = kwargs.get("ve_rank", 0) or max(1, rank // 2)
            total += vr
            if kwargs.get("bidirectional", False):
                total += vr
        if kwargs.get("use_polynomial", False):
            po = kwargs.get("poly_order", 2)
            total += po
            if kwargs.get("bidirectional", False):
                total += po
        if kwargs.get("use_autoencoder", False):
            hidden_r = max(1, rank * 2)
            ae = rank * hidden_r + hidden_r + hidden_r * rank + rank
            total += ae
            if kwargs.get("bidirectional", False):
                total += ae
        if kwargs.get("use_gate", False):
            total += 1
        return total

    @classmethod
    def variant_name(cls) -> ExpertVariant:
        return "hybrid"


def _hybrid_name(flags: dict) -> str:
    parts = []
    if flags.get("bidirectional"): parts.append("B")
    if flags.get("use_vectors"): parts.append("V")
    if flags.get("use_activation"): parts.append("AFA")
    if flags.get("use_autoencoder"): parts.append("AUR")
    if flags.get("use_polynomial"): parts.append("PERA")
    if flags.get("use_norm"): parts.append("A")
    if flags.get("use_gate"): parts.append("GA")
    return "".join(parts) if parts else "plain"


EXPERT_REGISTRY: dict[ExpertVariant, type[StreamExpert]] = {
    "plain": PlainStreamExpert,
    "dora": DoRAStreamExpert,
    "bora": BoRAStreamExpert,
    "vera": VeRaStreamExpert,
    "hybrid": HybridStreamExpert,
}


def create_expert(variant: ExpertVariant, in_features: int, out_features: int, rank: int, d_key: int, d_emb: int, scaling: float, **kwargs) -> StreamExpert:
    cls = EXPERT_REGISTRY[variant]
    extra = {}
    if variant in ("dora", "bora", "vera", "hybrid"):
        extra["eps"] = kwargs.get("dora_eps", 1e-5)
    if variant == "vera":
        extra["ve_rank"] = kwargs.get("ve_rank", 0)
    if variant == "hybrid":
        for k in ("bidirectional", "use_vectors", "use_norm", "use_gate",
                   "use_activation", "use_autoencoder", "use_polynomial",
                   "poly_order", "ve_rank", "anneal_rate"):
            extra[k] = kwargs.get(k, False)
    return cls(in_features, out_features, rank, d_key, d_emb, scaling, **extra)


class PerceiverFusion(nn.Module):
    def __init__(self, d_model: int, d_latent: int, n_latents: int, n_heads: int, d_expert_emb: int, d_key: int, d_out: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.n_latents = n_latents
        self.d_emb = d_expert_emb
        self.d_key = d_key
        self.d_out = d_out if d_out is not None else d_model

        self.latent_init = nn.Sequential(
            nn.Linear(d_model + d_key, d_latent),
            nn.LayerNorm(d_latent),
        )

        need_proj = d_expert_emb != d_latent
        self.proj_emb = nn.Linear(d_expert_emb, d_latent) if need_proj else nn.Identity()

        self.cross_in = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_in = nn.LayerNorm(d_latent)

        self.self_attn = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_sa = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, 4 * d_latent), nn.GELU(), nn.Linear(4 * d_latent, d_latent),
        )
        self.norm_ff = nn.LayerNorm(d_latent)

        self.cross_out_q = nn.Linear(d_model, d_latent, bias=False)
        self.cross_out = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(d_latent)

        self.out_proj = nn.Linear(d_latent, self.d_out, bias=False)

    def init_latents(self, h: torch.Tensor, keys: torch.Tensor | None) -> torch.Tensor:
        B = h.shape[0]
        if keys is not None and keys.shape[1] > 0:
            k_pool = keys.mean(dim=1, keepdim=True).expand(-1, self.n_latents, -1)
        else:
            k_pool = torch.zeros(B, self.n_latents, self.d_key, device=h.device, dtype=h.dtype)
        h_exp = h.unsqueeze(1).expand(-1, self.n_latents, -1)
        inp = torch.cat([h_exp, k_pool.to(h.dtype)], dim=-1)
        z = self.latent_init(inp.to(h.dtype))
        return z

    def forward(self, h: torch.Tensor, embeddings: torch.Tensor, keys: torch.Tensor | None = None):
        B = h.shape[0]
        Z = self.init_latents(h, keys)

        if embeddings.shape[1] > 0:
            emb_proj = self.proj_emb(embeddings)
            Z_in, _ = self.cross_in(Z, emb_proj, emb_proj, need_weights=False)
            Z = self.norm_in(Z + Z_in)

        Z_sa, _ = self.self_attn(Z, Z, Z, need_weights=False)
        Z = self.norm_sa(Z + Z_sa)
        Z = self.norm_ff(Z + self.ffn(Z))

        h_q = self.cross_out_q(h).unsqueeze(1)
        Z_out, _ = self.cross_out(h_q, Z, Z, need_weights=False)
        fused = self.norm_out(Z_out)

        delta = self.out_proj(fused)
        return delta


class StreamFusionLoRA(LowRankAdapter):
    def __init__(self, in_features: int, out_features: int, config: AdapterConfig):
        super().__init__(in_features, out_features, config)
        sf_config = StreamFusionConfig(**config.extra_kwargs) if config.extra_kwargs else StreamFusionConfig()
        sf_config.r = config.r
        sf_config.lora_alpha = config.lora_alpha
        sf_config.lora_dropout = config.lora_dropout

        self.sf = sf_config
        self.expert_variant: ExpertVariant = sf_config.expert_variant
        self.d_key = sf_config.d_key
        self.n_latents = sf_config.n_latents
        self.d_latent = sf_config.d_latent
        self.d_emb = sf_config.d_expert_emb
        self.top_m = sf_config.top_m
        self.tau = sf_config.tau_routing
        self.max_experts = sf_config.max_experts
        self.abs_steps = sf_config.abs_steps
        self.beta_absorb = sf_config.beta_absorb
        self._absorb_dim_cached: int | None = None

        self.hidden_dim = in_features
        self.out_dim = out_features
        self.fusion = PerceiverFusion(
            d_model=in_features,
            d_latent=self.d_latent,
            n_latents=self.n_latents,
            n_heads=sf_config.n_heads,
            d_expert_emb=self.d_emb,
            d_key=self.d_key,
            d_out=out_features,
        )

        self.query_proj = nn.Linear(in_features, self.d_key, bias=False)
        self.experts: nn.ModuleList = nn.ModuleList()

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_absorb_dim(self) -> int:
        if self._absorb_dim_cached is None:
            cls = EXPERT_REGISTRY[self.expert_variant]
            self._absorb_dim_cached = cls.absorb_dim(
                self.in_features, self.out_features, self.r,
                ve_rank=self.sf.ve_rank,
                bidirectional=self.sf.bidirectional,
                use_vectors=self.sf.use_vectors,
                use_norm=self.sf.use_norm,
                use_gate=self.sf.use_gate,
                use_activation=self.sf.use_activation,
                use_autoencoder=self.sf.use_autoencoder,
                use_polynomial=self.sf.use_polynomial,
                poly_order=self.sf.poly_order,
            )
        return self._absorb_dim_cached

    def add_expert(self) -> int:
        idx = len(self.experts)
        ref_param = next(self.fusion.parameters())
        expert = create_expert(
            self.expert_variant, self.in_features, self.out_features, self.r,
            self.d_key, self.d_emb, self.scaling,
            dora_eps=self.sf.dora_eps, ve_rank=self.sf.ve_rank,
            bidirectional=self.sf.bidirectional, use_vectors=self.sf.use_vectors,
            use_norm=self.sf.use_norm, use_gate=self.sf.use_gate,
            use_activation=self.sf.use_activation, use_autoencoder=self.sf.use_autoencoder,
            use_polynomial=self.sf.use_polynomial, poly_order=self.sf.poly_order,
            anneal_rate=self.sf.anneal_rate,
        )
        expert.to(device=ref_param.device, dtype=ref_param.dtype)
        self.experts.append(expert)
        self._absorb_dim_cached = None
        return idx

    def _routing_weights(self, h: torch.Tensor) -> torch.Tensor:
        n = self.experts.__len__()
        if n == 0:
            return torch.empty(h.shape[0], 0, device=h.device)
        q = F.normalize(self.query_proj(h), dim=-1)
        keys = F.normalize(torch.stack([e.key for e in self.experts]), dim=-1)
        keys = keys.to(q.dtype)
        logits = q @ keys.T / self.tau
        return F.softmax(logits, dim=-1)

    def _top_m_embeddings(self, alpha: torch.Tensor) -> torch.Tensor:
        n = self.experts.__len__()
        if n == 0:
            return torch.zeros(1, 0, self.d_emb, device=alpha.device, dtype=alpha.dtype)
        m = min(self.top_m, n)
        vals, idx = alpha.topk(m, dim=-1)
        emb_list = []
        for b in range(alpha.shape[0]):
            embs = torch.stack([self.experts[i.item()].embedding.to(alpha.device, alpha.dtype) for i in idx[b]])
            emb_list.append(embs)
        return torch.stack(emb_list)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        n = self.experts.__len__()
        if n == 0:
            return torch.zeros(x.shape[0], self.out_dim, dtype=x.dtype, device=x.device)

        h = x
        if x.dim() == 3:
            h = x.mean(dim=1)

        alpha = self._routing_weights(h)
        embeddings = self._top_m_embeddings(alpha)

        if n > 0:
            all_keys = torch.stack([e.key.to(h.device, h.dtype) for e in self.experts]).unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            all_keys = torch.zeros(x.shape[0], 0, self.d_key, device=h.device, dtype=h.dtype)

        fused_delta = self.fusion(h, embeddings, all_keys)

        delta = fused_delta.squeeze(1)
        if x.dim() == 2:
            return delta
        return delta.unsqueeze(1).expand(-1, x.shape[1], -1).contiguous()

    def absorb(self):
        n = self.experts.__len__()
        if n == 0:
            return
        ref_p = next(self.fusion.parameters())
        dev = ref_p.device
        dt = ref_p.dtype
        if not hasattr(self, 'absorb_decoder'):
            total_ab = self._get_absorb_dim()
            self.absorb_decoder = nn.Linear(self.d_latent, total_ab, device=dev, dtype=dt)
        opt = torch.optim.Adam(self.absorb_decoder.parameters(), lr=1e-4)
        dummy_h = torch.zeros(1, self.hidden_dim, device=dev, dtype=dt)
        with torch.no_grad():
            Z_avg = self.fusion.init_latents(dummy_h, None).mean(dim=1, keepdim=True)
        for _ in range(self.abs_steps):
            total = torch.zeros(1, device=dev, dtype=dt)
            for expert in self.experts:
                target = expert.absorb_params()
                pred = self.absorb_decoder(Z_avg.squeeze(1))
                total = total + F.mse_loss(pred.squeeze(), target)
            total = total / n
            total.backward()
            opt.step()
            opt.zero_grad()

    def prune(self, threshold: float = 0.01):
        n = self.experts.__len__()
        if n < 2 or not hasattr(self, 'absorb_decoder'):
            return
        ref_p = next(self.fusion.parameters())
        dev = ref_p.device
        dt = ref_p.dtype
        dummy_h = torch.zeros(1, self.hidden_dim, device=dev, dtype=dt)
        with torch.no_grad():
            Z_avg = self.fusion.init_latents(dummy_h, None).mean(dim=1, keepdim=True)
        keep: list[StreamExpert] = []
        for expert in self.experts:
            target = expert.absorb_params()
            pred = self.absorb_decoder(Z_avg.squeeze(1))
            err = F.mse_loss(pred.squeeze(), target).item()
            if err > threshold:
                keep.append(expert)
        self.experts = nn.ModuleList(keep)

    def reset(self):
        self.experts = nn.ModuleList()
        self._absorb_dim_cached = None
