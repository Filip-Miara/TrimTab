import math

import torch
import torch.nn as nn


def sparse_to_dense(sparse_vec: torch.Tensor, omega: torch.Tensor, shape: tuple) -> torch.Tensor:
    if omega.numel() == 0 or sparse_vec.shape[0] == 0:
        return torch.zeros(shape, device=sparse_vec.device, dtype=sparse_vec.dtype)
    flat = torch.zeros(shape[0] * shape[1], device=sparse_vec.device, dtype=sparse_vec.dtype)
    idx = omega[:sparse_vec.shape[0]]
    flat = flat.scatter(0, idx, sparse_vec)
    return flat.view(shape)


def inverse_haar_2d(F: torch.Tensor) -> torch.Tensor:
    r, d = F.shape
    assert r % 2 == 0 and d % 2 == 0
    Fa = F[:r//2, :d//2]
    Fh = F[:r//2, d//2:]
    Fv = F[r//2:, :d//2]
    Fd = F[r//2:, d//2:]
    out = torch.zeros_like(F)
    out[0::2, 0::2] = 0.5 * (Fa + Fh + Fv + Fd)
    out[1::2, 0::2] = 0.5 * (Fa - Fh + Fv - Fd)
    out[0::2, 1::2] = 0.5 * (Fa + Fh - Fv - Fd)
    out[1::2, 1::2] = 0.5 * (Fa - Fh - Fv + Fd)
    return out


def spectral_transform(F: torch.Tensor, spectral_type: str = "wavelet") -> torch.Tensor:
    if spectral_type == "fourier":
        return torch.fft.ifft2(F).real
    elif spectral_type == "wavelet":
        return inverse_haar_2d(F)
    else:
        raise ValueError(f"Unknown spectral type: {spectral_type}")


class SeLoRALoRA(nn.Module):
    def __init__(self, r: int, dim: int, sparse_ratio: float = 0.4, spectral_type: str = "wavelet"):
        super().__init__()
        self.r, self.dim = r, dim
        self.spectral_type = spectral_type
        total = r * dim
        n_learnable = max(1, int((1 - sparse_ratio) * total))
        omega = torch.randperm(total)[:n_learnable]
        self.register_buffer("omega", omega)
        self.sparse_F = nn.Parameter(torch.zeros(n_learnable))

    def get_matrix(self) -> torch.Tensor:
        F = sparse_to_dense(self.sparse_F, self.omega, (self.r, self.dim))
        return spectral_transform(F, self.spectral_type)

    def reset_parameters(self):
        nn.init.normal_(self.sparse_F, mean=0.0, std=0.02)
