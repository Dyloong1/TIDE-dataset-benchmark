"""Diagnostics: skewness of a Gaussian field ~ 0, forcing injects eps_w
exactly, IC hits its target spectrum shape and rms."""
import numpy as np
import torch

from solver.config import ForcingConfig
from solver.diagnostics import derivative_skewness, spectral_summary
from solver.forcing import NegativeDampingBandForcing
from solver.grids import SpectralGrid
from solver.initial_conditions import random_solenoidal


def test_gaussian_skewness_near_zero(device):
    N = 64
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=7)
    u = torch.fft.irfftn(u_hat, s=(N, N, N), dim=(-3, -2, -1))
    s3 = derivative_skewness(u, grid)
    # a Gaussian random field has zero derivative skewness (finite-sample noise)
    assert abs(s3) < 0.05


def test_forcing_injection_rate_exact(device):
    N, eps_w = 32, 0.123
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=3)
    f = NegativeDampingBandForcing(grid, k_f=2.0, eps_w=eps_w)
    f_hat = torch.empty_like(u_hat)
    f(u_hat, f_hat)
    inj = grid.injection_rate(u_hat, f_hat)
    assert abs(inj - eps_w) / eps_w < 1e-12


def test_random_ic_targets(device):
    N, u_rms, k_p = 64, 0.7, 3.0
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=11, k_p=k_p, u_rms=u_rms)
    # energy target
    K = grid.kinetic_energy(u_hat)
    assert abs(K - 1.5 * u_rms**2) / (1.5 * u_rms**2) < 1e-10
    # divergence-free
    div = (grid.kx * u_hat[0] + grid.ky * u_hat[1] + grid.kz * u_hat[2])
    assert float(div.abs().max()) / float(u_hat.abs().max()) < 1e-12
    # spectrum peaks near k_p
    k, E = grid.shell_spectrum(u_hat)
    k_peak = float(k[int(torch.argmax(E))])
    assert abs(k_peak - k_p) <= 1.0


def test_spectral_summary_roundtrip(device):
    """spectral_summary on a measured spectrum reproduces grid reductions."""
    N, nu = 64, 2.7e-3
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=5)
    k, E = grid.shell_spectrum(u_hat)
    summ = spectral_summary(k.numpy(), E.numpy(), nu)
    K_direct = grid.kinetic_energy(u_hat)
    # trapezoid vs discrete sum: integer-k shells, agreement to ~1%
    assert abs(summ["K"] - K_direct) / K_direct < 0.02
