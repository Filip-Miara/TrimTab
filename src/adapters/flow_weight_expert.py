"""FlowWeightExpert: generates adapter weights via learned velocity field.

Flow matching over the adapter weight manifold, conditioned on task data.
The velocity field learns the optimal update dynamics — at inference,
integrate from W_0 = 0 to generate trained weights without SGD.

Theoretical connection: for a linear layer with MSE loss, the optimal
rank-r LoRA update has a closed form via SVD of R·X⁺ where R = Y - WX
is the residual and X⁺ is the pseudo-inverse of inputs. Gradient flow
on BA traces a path through the low-rank matrix manifold. The velocity
field v_θ learns to approximate this optimal transport map.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_closed_form_lora(
    X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimal rank-r LoRA update via SVD of R·X⁺.

    Solves: min_{A,B} ||Y - (W + BA)X||²_F
    Solution: BA* = U_r Σ_r V_r^T where UΣV = SVD((Y-WX)X⁺)

    Args:
        X: input activations (B, d_in) or (B, S, d_in)
        Y: target outputs (B, d_out)
        W: frozen weight matrix (d_out, d_in)
        rank: target LoRA rank

    Returns:
        A_opt: (rank, d_in)
        B_opt: (d_out, rank)
    """
    x_flat = X.reshape(-1, X.shape[-1])
    y_flat = Y.reshape(-1, Y.shape[-1])

    R = y_flat - x_flat @ W.T

    # Pseudo-inverse: X⁺ = (X^T X)⁻¹ X^T  (d_in, B)
    XtX = x_flat.T @ x_flat
    reg = 1e-6 * torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
    X_pinv = torch.linalg.solve(XtX + reg, x_flat.T)

    # Optimal full-rank update: R · X⁺
    delta_full = R.T @ X_pinv.T  # (d_out, d_in)

    # Rank-r truncation via SVD
    U, S, Vh = torch.linalg.svd(delta_full.float(), full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    B_opt = U_r @ torch.diag(S_r.sqrt())
    A_opt = torch.diag(S_r.sqrt()) @ Vh_r

    return A_opt.to(X.dtype), B_opt.to(X.dtype)


class DataEncoder(nn.Module):
    """Encodes (X, Y, W) into a rich context vector for conditioning.

    Computes: mean(X), mean(R), gradient direction X^T R, loss scalar.
    """
    def __init__(self, d_in: int, d_out: int, d_latent: int = 64):
        super().__init__()
        ctx_dim = d_in + d_out + d_in + 1
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(self, X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        x_mean = X.mean(dim=(0, 1)) if X.dim() == 3 else X.mean(dim=0)
        y_mean = Y.mean(dim=(0, 1)) if Y.dim() == 3 else Y.mean(dim=0)
        x_flat = X.reshape(-1, X.shape[-1])
        y_flat = Y.reshape(-1, Y.shape[-1])
        R = y_flat - x_flat @ W.T
        grad_dir = (x_flat.T @ R).mean(dim=1)
        loss = F.mse_loss(y_flat, x_flat @ W.T).detach()
        ctx = torch.cat([x_mean, y_mean, grad_dir, loss.unsqueeze(0)])
        return self.net(ctx).unsqueeze(0)


class PerceiverWeightFlow(nn.Module):
    """Perceiver-bottleneck velocity field over adapter weights.

    v_θ: (W_t, t, data_ctx) → δW (predicted weight change)

    Uses K latents for O(NK+K²) scaling with weight space dimension N.
    """
    def __init__(self, n_weights: int, d_latent: int = 64, n_latents: int = 32,
                 n_heads: int = 4, d_context: int = 64):
        super().__init__()
        self.n_weights = n_weights
        self.d_latent = d_latent
        self.n_latents = n_latents

        self.pos_embed = nn.Embedding(n_weights, d_latent)
        self.context_proj = nn.Linear(d_context, d_latent)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_latent), nn.GELU(), nn.Linear(d_latent, d_latent),
        )
        self.weight_proj = nn.Linear(1, d_latent)

        self.latents = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)
        self.cross_in = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_in = nn.LayerNorm(d_latent)
        self.self_attn = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_sa = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, 4 * d_latent), nn.GELU(), nn.Linear(4 * d_latent, d_latent),
        )
        self.norm_ff = nn.LayerNorm(d_latent)
        self.cross_out = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(d_latent)

        self.decode = nn.Linear(d_latent, 1)

    def forward(self, weights: torch.Tensor, t: torch.Tensor,
                data_ctx: torch.Tensor | None = None) -> torch.Tensor:
        B = weights.shape[0]
        N = self.n_weights

        w_emb = self.weight_proj(weights.unsqueeze(-1))
        w_emb = w_emb + self.pos_embed.weight.unsqueeze(0)
        w_emb = w_emb + self.time_embed(t.unsqueeze(-1)).unsqueeze(1)

        ctx = torch.zeros(1, 1, self.d_latent, device=weights.device)
        if data_ctx is not None:
            ctx = self.context_proj(data_ctx).unsqueeze(1)

        Z = self.latents.unsqueeze(0).expand(B, -1, -1)
        Z_in, _ = self.cross_in(Z + ctx, w_emb, w_emb, need_weights=False)
        Z = self.norm_in(Z + Z_in)
        Z_sa, _ = self.self_attn(Z, Z, Z, need_weights=False)
        Z = self.norm_sa(Z + Z_sa)
        Z = self.norm_ff(Z + self.ffn(Z))
        Z_out, _ = self.cross_out(w_emb, Z, Z, need_weights=False)
        w_out = self.norm_out(Z_out)

        return self.decode(w_out).squeeze(-1) * 0.1


class FlowWeightExpert(nn.Module):
    """Expert that generates its weights via flow matching.

    Instead of learnable A, B parameters, contains a velocity field
    that generates the optimal weights conditioned on input data.
    At inference: integrate from zero, no SGD needed.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 d_key: int, d_emb: int, scaling: float,
                 d_latent: int = 32, n_latents: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = scaling

        n_weights = rank * (in_features + out_features)
        self.n_weights = n_weights
        self.flow = PerceiverWeightFlow(n_weights, d_latent, n_latents, d_context=d_latent)
        self.data_encoder = DataEncoder(in_features, out_features, d_latent)

        self.key = nn.Parameter(torch.randn(d_key) * 0.1)
        self.lambda_ = nn.Parameter(torch.tensor(1.0))
        self.embedding = nn.Parameter(torch.randn(d_emb) * 0.02)
        self.register_buffer("_cache_w", torch.zeros(1, n_weights))
        self.register_buffer("_cache_ctx", torch.zeros(d_latent))
        self._cache_valid = False

    def encode(self, X: torch.Tensor, W_base: torch.Tensor) -> torch.Tensor:
        """Encode input data into context vector for flow conditioning."""
        return self.data_encoder(X, X @ W_base.T, W_base)

    @torch.no_grad()
    def generate_weights(self, ctx: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """Generate adapter weights by integrating velocity field from zero."""
        self.flow.eval()
        w = torch.zeros(1, self.n_weights, device=ctx.device)
        for step in range(n_steps):
            t = torch.tensor([step / max(n_steps - 1, 1)], device=ctx.device)
            v = self.flow(w, t, ctx)
            w = w + v / n_steps
        return w

    def forward(self, X: torch.Tensor, W_base: torch.Tensor) -> torch.Tensor:
        ctx = self.encode(X, W_base)

        if not self._cache_valid or not (ctx == self._cache_ctx).all():
            flat_w = self.generate_weights(ctx)
            self._cache_w.copy_(flat_w)
            self._cache_ctx.copy_(ctx)
            self._cache_valid = True

        A = self._cache_w[0, :self.rank * self.in_features].reshape(self.rank, self.in_features)
        B = self._cache_w[0, self.rank * self.in_features:].reshape(self.out_features, self.rank)

        return self.lambda_ * self.scaling * (X @ A.T @ B.T)

    def reset_cache(self):
        self._cache_valid = False


def analyze_flow_vs_sgd_vs_closed_form():
    """Compare flow-generated weights against SGD and closed-form optimal.

    For a linear layer W with MSE loss on (X, Y), the optimal rank-r
    update is the SVD of R·X⁺. This is the 'ground truth' target.
    SGD traces a trajectory toward this optimum.
    Flow matching learns to approximate the vector field of this trajectory.
    """
    import numpy as np
    from .stream_fusion import PlainStreamExpert

    print("=" * 60)
    print("  Theoretical Analysis: Flow Matching vs SGD vs Closed-Form")
    print("=" * 60)

    d_in, d_out, r = 64, 32, 4
    W_true = torch.randn(d_out, d_in)
    X = torch.randn(16, d_in)
    Y = X @ W_true.T + 0.01 * torch.randn(16, d_out)

    # 1. Closed-form optimal rank-r update (factorized)
    A_opt, B_opt = compute_closed_form_lora(X, Y, W_true, r)
    delta_opt = torch.cat([A_opt.flatten(), B_opt.flatten()])  # factorized: (r*d_in + d_out*r) = 384

    # 2. SGD trajectory
    expert = PlainStreamExpert(d_in, d_out, r, 16, 16, 1.0)
    params = [p for n, p in expert.named_parameters()
              if p.requires_grad and ('lora_A' in n or 'lora_B' in n)]
    sgd_trajectory = []
    opt = torch.optim.SGD(params, lr=1e-2)
    for step in range(50):
        pred = X @ W_true.T + expert(X)
        loss = F.mse_loss(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == 49:
            flat = torch.cat([p.data.flatten() for p in params])
            sgd_trajectory.append(flat.clone())

    sgd_final = torch.cat([p.data.flatten() for p in params])

    # 3. Train flow model on SGD trajectory
    from .weight_flow import WeightFlowField, WeightFlowTrainer
    n_weights = sgd_final.shape[0]
    flow = PerceiverWeightFlow(n_weights, d_latent=16, n_latents=8, d_context=16)
    fopt = torch.optim.AdamW(flow.parameters(), lr=1e-3)
    with torch.no_grad():
        flow_ctx = DataEncoder(d_in, d_out, 16)(X, Y, W_true)

    for epoch in range(50):
        losses = []
        for t_idx in range(len(sgd_trajectory) - 1):
            w_t = sgd_trajectory[t_idx].unsqueeze(0)
            w_tp1 = sgd_trajectory[t_idx + 1].unsqueeze(0)
            t = torch.tensor([t_idx / (len(sgd_trajectory) - 1)])
            v_pred = flow(w_t, t, flow_ctx)
            loss = F.mse_loss(v_pred, w_tp1 - w_t)
            fopt.zero_grad(); loss.backward(); fopt.step()
            losses.append(loss.item())

    # 4. Generate via flow integration
    with torch.no_grad():
        flow.eval()
        w_f = torch.zeros(1, n_weights)
        for s in range(50):
            t = torch.tensor([s / 49]); v = flow(w_f, t, flow_ctx); w_f = w_f + v / 50
    flow_final = w_f[0]

    # 5. Compare
    def angle(a, b):
        return (a @ b / (a.norm() * b.norm() + 1e-8)).item()

    print(f"\n  Weight space dimension: {n_weights}")
    print(f"\n  Cosine similarity to closed-form optimal:")
    print(f"    SGD final:  {angle(sgd_final, delta_opt):.4f}")
    print(f"    Flow final: {angle(flow_final, delta_opt):.4f}")
    print(f"    SGD vs Flow: {angle(sgd_final, flow_final):.4f}")

    print(f"\n  MSE against closed-form optimal:")
    print(f"    SGD:  {F.mse_loss(sgd_final, delta_opt).item():.6f}")
    print(f"    Flow: {F.mse_loss(flow_final, delta_opt).item():.6f}")

    # 6. Check if velocity field predicts the closed-form update direction
    with torch.no_grad():
        w_zero = torch.zeros(1, n_weights)
        t_zero = torch.tensor([0.0])
        v_at_zero = flow(w_zero, t_zero, flow_ctx)
        print(f"\n  Velocity field at W=0, t=0:")
        print(f"    Direction vs closed-form: {angle(v_at_zero[0], delta_opt):.4f}")
        print(f"    Direction vs SGD 1st step: {angle(v_at_zero[0], sgd_trajectory[1] - sgd_trajectory[0]):.4f}")

    return flow, flow_ctx


if __name__ == "__main__":
    analyze_flow_vs_sgd_vs_closed_form()
