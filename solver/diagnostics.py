"""Offline physics diagnostics (QA on sampled frames).

Online per-step scalars (K, eps, eps_inj, umax) live on SpectralGrid /
PseudoSpectralSolver; this module hosts the heavier statistics adapted from
NeurIPS/ablations/vae_32_patch/code/physics_metrics.py (same conventions),
with viscosity and fit windows parameterized instead of hard-coded.
"""
from __future__ import annotations

import numpy as np
import torch

from .grids import SpectralGrid

_FFT_DIMS = (-3, -2, -1)


@torch.no_grad()
def derivative_skewness(u_phys: torch.Tensor, grid: SpectralGrid) -> float:
    """S3 = <(du_i/dx_i)^3> / <(du_i/dx_i)^2>^1.5, averaged over i=x,y,z.

    Classical universal value for HIT: -0.5 +/- 0.05 (acceptance 4).
    """
    ks = (grid.kx, grid.ky, grid.kz)          # derivative directions x,y,z
    s3 = []
    for i in range(3):                        # channel i differentiated along x_i
        d_hat = (1j * ks[i]) * torch.fft.rfftn(u_phys[i], dim=_FFT_DIMS)
        d = torch.fft.irfftn(d_hat, s=u_phys[i].shape, dim=_FFT_DIMS).double()
        m2 = (d * d).mean()
        m3 = (d * d * d).mean()
        s3.append(float(m3 / m2.clamp(min=1e-30) ** 1.5))
    return float(np.mean(s3))


def fit_spectrum_slope(k: np.ndarray, E: np.ndarray,
                       k_min: float, k_max: float) -> float:
    """Least-squares slope of log E vs log k in [k_min, k_max].
    256^3 convention: k in [4, 20] (knowledge base, fourth part)."""
    k = np.asarray(k, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    m = (k >= k_min) & (k <= k_max) & (E > 0)
    if m.sum() < 3:
        return float("nan")
    beta, _ = np.polyfit(np.log(k[m]), np.log(E[m]), 1)
    return float(beta)


def spectral_summary(k: np.ndarray, E: np.ndarray, nu: float) -> dict:
    """K, u', eps, L, eta, lambda_T, Re_lambda from a radial spectrum.
    Adapted from NeurIPS/scripts/compute_spectral_metrics.py (nu now an arg)."""
    k = np.asarray(k, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    m = (k > 0) & (E > 0)
    k, E = k[m], E[m]
    K = float(np.trapezoid(E, k))
    u_rms = float(np.sqrt(2.0 * K / 3.0))
    eps = float(2.0 * nu * np.trapezoid(k**2 * E, k))
    L = float((np.pi / (2.0 * u_rms**2)) * np.trapezoid(E / k, k)) if u_rms > 0 else float("nan")
    eta = float((nu**3 / eps) ** 0.25) if eps > 0 else float("nan")
    lam = float(np.sqrt(15.0 * nu * u_rms**2 / eps)) if eps > 0 else float("nan")
    re_lam = float(u_rms * lam / nu) if eps > 0 else float("nan")
    return {"K": K, "u_rms": u_rms, "eps": eps, "L": L, "eta": eta,
            "lambda_T": lam, "Re_lambda": re_lam}


def kolmogorov_normalize(k: np.ndarray, E: np.ndarray, eps: float, nu: float):
    """(k*eta, E / (eps nu^5)^(1/4)) — universal coordinates for overlaying
    spectra from different Re / unit conventions (acceptance 4, user-approved
    protocol for the JHTDB comparison)."""
    eta = (nu**3 / eps) ** 0.25
    return np.asarray(k) * eta, np.asarray(E) / (eps * nu**5) ** 0.25


def k_max_eta(grid_k_max: int, eps: float, nu: float) -> float:
    """Resolution criterion k_max*eta (acceptance 3: >= 1.5 with 2/3 rule)."""
    return float(grid_k_max * (nu**3 / eps) ** 0.25)
