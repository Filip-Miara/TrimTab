"""Per-sub-matrix gradient decomposition and independent training signals.

Provides three complementary methods for decoupling sub-matrix training:
1. TaylorContribution — first-order per-component contribution to loss reduction
2. AlternatingTrainer — gradient isolation via round-robin component freezing
3. OverlapConsistency — consistency loss for overlapping sub-matrices
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rank1_components(
    expert: nn.Module,
) -> list[tuple[str, nn.Parameter | None, nn.Parameter | None, int | None]]:
    """Yield all rank-1 component pairs with leaf param + index.

    Handles both directional (lora_A_fwd/lora_B_fwd) and non-directional
    (lora_A/lora_B) naming conventions.
    """
    comps: list[tuple[str, nn.Parameter | None, nn.Parameter | None, int | None]] = []

    # Try non-directional first (lora_A, lora_B)
    A = getattr(expert, 'lora_A', None)
    B = getattr(expert, 'lora_B', None)
    if isinstance(A, nn.Parameter) and isinstance(B, nn.Parameter):
        for i in range(A.shape[0]):
            comps.append((f"lora_A[{i}]", A, B, i))
        return comps  # non-directional is authoritative if present

    # Try directional (lora_A_fwd, lora_B_fwd)
    A_fwd = getattr(expert, 'lora_A_fwd', None)
    B_fwd = getattr(expert, 'lora_B_fwd', None)
    if isinstance(A_fwd, nn.Parameter) and isinstance(B_fwd, nn.Parameter):
        for i in range(A_fwd.shape[0]):
            comps.append((f"lora_A_fwd[{i}]", A_fwd, B_fwd, i))

    # Try backward direction
    A_bwd = getattr(expert, 'lora_A_bwd', None)
    B_bwd = getattr(expert, 'lora_B_bwd', None)
    if isinstance(A_bwd, nn.Parameter) and isinstance(B_bwd, nn.Parameter):
        for i in range(A_bwd.shape[0]):
            comps.append((f"lora_A_bwd[{i}]", A_bwd, B_bwd, i))

    # Scalar params
    for name in ('ve_lambda_fwd', 've_lambda_bwd', 've_lambda'):
        p = getattr(expert, name, None)
        if isinstance(p, nn.Parameter):
            comps.append((name, p, None, None))

    for name in ('poly_coeff_fwd', 'poly_coeff_bwd', 'poly_coeff'):
        p = getattr(expert, name, None)
        if isinstance(p, nn.Parameter):
            for i in range(p.shape[0]):
                comps.append((f"{name}[{i}]", p, None, i))

    return comps


class TaylorContribution:
    """First-order per-component contribution to loss reduction.

    After one backward pass, computes ΔL_i ≈ 〈∂L/∂θ_i, Δθ_i〉 for each component.
    Requires only 1 forward + 1 backward.
    Uses leaf tensor .grad with row indexing to get per-rank contributions.
    """

    def __init__(self):
        self.contributions: dict[str, float] = {}

    def rank1_contributions(self, expert: nn.Module) -> dict[str, float]:
        """Per rank-1 component contribution via Taylor approximation.

        ΔL_i ≈ 〈∂L/∂A[i], ΔA[i]〉 + 〈∂L/∂B[:,i], ΔB[:,i]〉
        """
        contribs = {}
        for cname, A_leaf, B_leaf, rank_idx in rank1_components(expert):
            contribs[cname] = 0.0
            if isinstance(A_leaf, nn.Parameter) and A_leaf.grad is not None and rank_idx is not None:
                a_grad = A_leaf.grad[rank_idx:rank_idx + 1]
                a_delta = A_leaf[rank_idx:rank_idx + 1] - A_leaf[rank_idx:rank_idx + 1].detach()
                contribs[cname] += float((a_grad * a_delta).sum().item())
            if isinstance(B_leaf, nn.Parameter) and B_leaf.grad is not None and rank_idx is not None:
                b_grad = B_leaf.grad[:, rank_idx:rank_idx + 1]
                b_delta = B_leaf[:, rank_idx:rank_idx + 1] - B_leaf[:, rank_idx:rank_idx + 1].detach()
                contribs[cname] += float((b_grad * b_delta).sum().item())
        return contribs

    @staticmethod
    def prune_low_contributors(expert: nn.Module, contribs: dict[str, float], threshold_ratio: float = 0.1):
        """Zero out rank-1 components whose contribution is below ratio of max."""
        max_c = max(abs(c) for c in contribs.values()) if contribs else 0
        if max_c == 0:
            return
        for (cname, A_leaf, B_leaf, rank_idx) in rank1_components(expert):
            if cname in contribs and abs(contribs[cname]) < threshold_ratio * max_c:
                if isinstance(A_leaf, nn.Parameter) and rank_idx is not None:
                    A_leaf.data[rank_idx:rank_idx + 1].zero_()
                if isinstance(B_leaf, nn.Parameter) and rank_idx is not None:
                    B_leaf.data[:, rank_idx:rank_idx + 1].zero_()

    @staticmethod
    def prune_by_isolated_ablation(
        expert: nn.Module, x: torch.Tensor, loss_fn, threshold_ratio: float = 0.1,
    ) -> dict[str, float]:
        """Prune components that hurt loss (positive ablation = bad)."""
        with torch.no_grad():
            loss_all = loss_fn(expert(x)).item()
        contribs = {}
        for cname, A_leaf, B_leaf, rank_idx in rank1_components(expert):
            if rank_idx is None:
                continue
            with torch.no_grad():
                A_leaf[rank_idx:rank_idx + 1].data.zero_()
                if isinstance(B_leaf, nn.Parameter):
                    B_leaf[:, rank_idx:rank_idx + 1].data.zero_()
            loss_abl = loss_fn(expert(x)).item()
            with torch.no_grad():
                A_leaf.grad = None
            contribs[cname] = loss_abl - loss_all
        return contribs


class AlternatingTrainer:
    """Trains sub-matrices in alternation, freezing all but one per step.

    Each step, only one rank-1 component's A and B receive gradients.
    This gives truly isolated gradient signals.
    """

    def __init__(self, expert: nn.Module, lr: float = 1e-4):
        self.expert = expert
        self.components = rank1_components(expert)
        self.n_comps = len(self.components)
        self.lr = lr
        self._step = 0

    def step_isolated(self, x: torch.Tensor, target_loss_fn) -> dict:
        """Train one component in isolation. Rotates through components.

        Uses gradient masking: forward + backward with all components active,
        then zero out gradients of all non-target components before stepping.
        This avoids view requires_grad_ issues.
        """
        idx = self._step % self.n_comps
        cname, A_leaf, B_leaf, rank_idx = self.components[idx]

        if rank_idx is None:
            self._step += 1
            return {"component": cname, "loss": 0.0, "grad_norm": 0.0}

        loss = target_loss_fn(self.expert(x))
        loss.backward()

        gnorm = 0.0
        if isinstance(A_leaf, nn.Parameter) and A_leaf.grad is not None:
            mask = torch.zeros_like(A_leaf.grad)
            mask[rank_idx] = 1.0
            A_leaf.grad.data *= mask
            gnorm += A_leaf.grad[rank_idx].norm().item()
        if isinstance(B_leaf, nn.Parameter) and B_leaf.grad is not None:
            mask = torch.zeros_like(B_leaf.grad)
            mask[:, rank_idx] = 1.0
            B_leaf.grad.data *= mask
            gnorm += B_leaf.grad[:, rank_idx].norm().item()

        with torch.no_grad():
            if isinstance(A_leaf, nn.Parameter):
                A_leaf[rank_idx] -= self.lr * A_leaf.grad[rank_idx]
            if isinstance(B_leaf, nn.Parameter):
                B_leaf[:, rank_idx] -= self.lr * B_leaf.grad[:, rank_idx]

        self._step += 1
        return {"component": cname, "loss": loss.item(), "grad_norm": gnorm}

    def contributions_by_isolation(self, x: torch.Tensor, loss_fn) -> dict[str, float]:
        """Measure each component's contribution via isolated ablation (N forward passes)."""
        with torch.no_grad():
            loss_all = loss_fn(self.expert(x)).item()
        contribs = {}
        for cname, A_leaf, B_leaf, rank_idx in self.components:
            if rank_idx is None:
                continue
            with torch.no_grad():
                saved_A = A_leaf[rank_idx].data.clone()
                A_leaf[rank_idx].data.zero_()
                if isinstance(B_leaf, nn.Parameter):
                    saved_B = B_leaf[:, rank_idx].data.clone()
                    B_leaf[:, rank_idx].data.zero_()
            loss_abl = loss_fn(self.expert(x)).item()
            with torch.no_grad():
                A_leaf[rank_idx].data.copy_(saved_A)
                if isinstance(B_leaf, nn.Parameter):
                    B_leaf[:, rank_idx].data.copy_(saved_B)
            contribs[cname] = loss_abl - loss_all
        return contribs


class OverlapConsistency:
    """Consistency loss for overlapping sub-matrices.

    For overlapping rank-1 components: L_cons = Σ_{i,j} MSE(ΔW_i(x), ΔW_j(x))
    For overlapping window sub-matrices: L_cons = Σ_{windows k} MSE(ΔW_k(x), ΔW_{k+1}(x))
    """

    def __init__(self, experts: list[nn.Module] | None = None):
        self.experts = experts or []

    def pairwise_consistency(self, x: torch.Tensor, experts: list[nn.Module]) -> torch.Tensor:
        if len(experts) < 2:
            return torch.tensor(0.0, device=x.device)
        deltas = [e(x) for e in experts]
        loss = torch.tensor(0.0, device=x.device)
        n = 0
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                loss = loss + F.mse_loss(deltas[i], deltas[j])
                n += 1
        return loss / max(n, 1)

    def rank1_pairwise_consistency(self, expert: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """MSE between all pairs of rank-1 component outputs."""
        comps = rank1_components(expert)
        deltas = []
        for cname, A_leaf, B_leaf, rank_idx in comps:
            if not isinstance(B_leaf, nn.Parameter) or rank_idx is None:
                continue
            a = A_leaf[rank_idx:rank_idx + 1]
            b = B_leaf[:, rank_idx:rank_idx + 1]
            if "bwd" in cname:
                delta = x @ b @ a
            else:
                delta = (x @ a.T) @ b.T
            deltas.append(delta)
        if len(deltas) < 2:
            return torch.tensor(0.0, device=x.device)
        loss = torch.tensor(0.0, device=x.device)
        n = 0
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                loss = loss + F.mse_loss(deltas[i], deltas[j])
                n += 1
        return loss / max(n, 1)

    def overlap_window_consistency(self, model: nn.Module, x: torch.Tensor, window_size: int = 16, stride: int = 8) -> torch.Tensor:
        """Consistency loss for overlapping windows along the rank dimension."""
        has_fwd = hasattr(model, 'lora_A_fwd') and hasattr(model, 'lora_B_fwd')
        if not has_fwd:
            return torch.tensor(0.0, device=x.device)
        r = model.lora_A_fwd.shape[0]
        w, s = window_size, stride
        if w >= r:
            return torch.tensor(0.0, device=x.device)
        window_deltas = []
        start = 0
        while start + w <= r:
            A_w = model.lora_A_fwd[start:start + w]
            B_w = model.lora_B_fwd[:, start:start + w]
            delta = x @ A_w.T @ B_w.T
            window_deltas.append(delta)
            start += s
        if len(window_deltas) < 2:
            return torch.tensor(0.0, device=x.device)
        loss = torch.tensor(0.0, device=x.device)
        for i in range(len(window_deltas)):
            for j in range(i + 1, len(window_deltas)):
                loss = loss + F.mse_loss(window_deltas[i], window_deltas[j])
        return loss / max(len(window_deltas) * (len(window_deltas) - 1) / 2, 1)
