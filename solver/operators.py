"""Spectral operators. All take pre-built SpectralGrid caches; pure functions
except for the explicitly in-place variants (suffix _).
"""
from __future__ import annotations

import torch

from .grids import SpectralGrid


def curl_hat(u_hat: torch.Tensor, grid: SpectralGrid, out: torch.Tensor | None = None) -> torch.Tensor:
    """omega_hat = i k x u_hat.  u_hat: [3, N, N, Nh] (channels x,y,z)."""
    kx, ky, kz = grid.kx, grid.ky, grid.kz
    if out is None:
        out = torch.empty_like(u_hat)
    # omega_x = i (ky*uz_hat - kz*uy_hat), etc.
    torch.mul(u_hat[2], ky, out=out[0]);  out[0] -= kz * u_hat[1]
    torch.mul(u_hat[0], kz, out=out[1]);  out[1] -= kx * u_hat[2]
    torch.mul(u_hat[1], kx, out=out[2]);  out[2] -= ky * u_hat[0]
    out *= 1j
    return out


def cross_product(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Pointwise physical-space cross product a x b. a, b: [3, N, N, N]."""
    if out is None:
        out = torch.empty_like(a)
    torch.mul(a[1], b[2], out=out[0]);  out[0] -= a[2] * b[1]
    torch.mul(a[2], b[0], out=out[1]);  out[1] -= a[0] * b[2]
    torch.mul(a[0], b[1], out=out[2]);  out[2] -= a[1] * b[0]
    return out


def leray_project_(n_hat: torch.Tensor, grid: SpectralGrid,
                   work: torch.Tensor | None = None) -> torch.Tensor:
    """In place: n_hat <- n_hat - k (k . n_hat) / k^2  (removes the
    compressible part; equivalently eliminates pressure). `work` is an
    optional [N, N, Nh] complex scratch buffer (avoids an allocation)."""
    kx, ky, kz = grid.kx, grid.ky, grid.kz
    if work is None:
        div = kx * n_hat[0]
    else:
        div = torch.mul(n_hat[0], kx, out=work)
    div += ky * n_hat[1]
    div += kz * n_hat[2]
    div *= grid.inv_k2
    n_hat[0] -= kx * div
    n_hat[1] -= ky * div
    n_hat[2] -= kz * div
    return n_hat


def dealias_(x_hat: torch.Tensor, grid: SpectralGrid) -> torch.Tensor:
    """In place 2/3-rule truncation."""
    x_hat *= grid.dealias_mask
    return x_hat


def helical_project(u_hat: torch.Tensor, grid: SpectralGrid,
                    sign: int = +1) -> torch.Tensor:
    """Positive- (sign=+1) or negative- (sign=-1) helicity part of a field.

    For a solenoidal field the curl operator i k x has eigenvalues +/-|k| with
    eigenvectors the helical modes h_+/h_-. The projector onto helicity `sign` is
        P_s u = 1/2 ( u_sol + s (i k x u) / |k| ),
    where u_sol is the Leray (solenoidal) projection. We Leray-project first so
    the identity holds for any input; the result satisfies i k x (P_s u)=s|k|P_s u,
    is solenoidal, idempotent, and P_+ + P_- = u_sol. Returns a NEW tensor (the
    forcing uses it to inject net helicity, breaking mirror symmetry).
    """
    out = u_hat.clone()
    leray_project_(out, grid)                     # ensure solenoidal: u_sol
    curl = curl_hat(out, grid)                    # i k x u_sol  (new tensor)
    out += sign * (curl * grid.inv_kmag)          # 1/|k| cached on grid (=0 at k=0)
    out *= 0.5
    return out


def pressure_hat(u_hat: torch.Tensor, grid: SpectralGrid) -> torch.Tensor:
    """Kinematic pressure p (rho=1) from an incompressible velocity field, via
    the pressure Poisson equation that follows from taking the divergence of NS:

        lap(p) = - d_i d_j (u_i u_j)      (incompressible, div u = 0)

    In spectral space p_hat = - (k_i k_j / k^2) (u_i u_j)_hat. Computed pseudo-
    spectrally: form u_i u_j in physical space, FFT, contract with k_i k_j / k^2.
    Returns p_hat [N, N, Nh]; the k=0 (mean) mode is set to 0 (gauge).
    Mirrors the rotational-form identity grad(p + |u|^2/2) = -(I-P)(u x omega),
    used by the velocity-pressure consistency check D4.
    """
    N = grid.N
    u = torch.fft.irfftn(u_hat, s=(N, N, N), dim=(-3, -2, -1))
    p_hat = torch.zeros(N, N, grid.Nh, dtype=grid.cdtype, device=grid.device)
    ks = (grid.kx, grid.ky, grid.kz)
    for i in range(3):
        for j in range(3):
            tij_hat = torch.fft.rfftn(u[i] * u[j], dim=(-3, -2, -1))
            p_hat += (ks[i] * ks[j]) * tij_hat
    # lap(p) = -d_i d_j(u_i u_j): -k^2 p_hat = +k_i k_j T_hat -> p_hat = -k_i k_j T / k^2
    p_hat *= -grid.inv_k2          # inv_k2 is 0 at k=0, so the mean is gauged out
    return p_hat
