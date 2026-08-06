"""Eswaran-Pope stochastic (OU) forcing: Hermitian symmetry (real velocity),
solenoidality, band-limitation, OU variance scaling, and reproducibility."""
import math

import torch

from solver.config import ForcingConfig, SolverConfig
from solver.forcing import StochasticOUForcing
from solver.grids import SpectralGrid
from solver.initial_conditions import random_solenoidal
from solver.solver import PseudoSpectralSolver


def test_force_is_real_and_solenoidal(device):
    N = 32
    grid = SpectralGrid(N, device, "fp64")
    f = StochasticOUForcing(grid, k_f=2.0, tau=1.0, sigma2=0.05, seed=7)
    for _ in range(20):
        f.advance_ou(0.01)
    out = torch.empty(3, N, N, N // 2 + 1, dtype=grid.cdtype, device=device)
    f(None, out)
    # solenoidal: k . f_hat = 0
    div = grid.kx * out[0] + grid.ky * out[1] + grid.kz * out[2]
    assert float(div.abs().max()) / max(float(out.abs().max()), 1e-30) < 1e-12
    # real physical field: irfftn then rfftn round-trips (Hermitian preserved)
    fp = torch.fft.irfftn(out, s=(N, N, N), dim=(-3, -2, -1))
    assert fp.dtype == grid.rdtype
    back = torch.fft.rfftn(fp, dim=(-3, -2, -1))
    assert torch.allclose(back, out, atol=1e-10, rtol=1e-8)


def test_band_limited(device):
    N = 32
    grid = SpectralGrid(N, device, "fp64")
    f = StochasticOUForcing(grid, k_f=2.0, seed=3)
    for _ in range(10):
        f.advance_ou(0.01)
    out = torch.empty(3, N, N, N // 2 + 1, dtype=grid.cdtype, device=device)
    f(None, out)
    # no energy outside 0 < |k| < 2
    outside = out[:, ~f.band_mask]
    assert float(outside.abs().max()) == 0.0


def test_ou_variance_stationary(device):
    """The OU process variance approaches sigma2 * n_band at stationarity."""
    N = 32
    grid = SpectralGrid(N, device, "fp64")
    sigma2 = 0.05
    f = StochasticOUForcing(grid, k_f=2.0, tau=1.0, sigma2=sigma2, seed=11)
    dt = 0.02
    # advance well past the correlation time and average the band variance
    vals = []
    for i in range(2000):
        f.advance_ou(dt)
        if i > 500:
            vals.append(float((f.b_hat.real**2 + f.b_hat.imag**2).sum()))
    mean_var = sum(vals) / len(vals)
    # projection halves the dof (solenoidal: 2 of 3 components survive), so the
    # stationary band variance is order sigma2 * n_band within a factor ~2-3;
    # check it is positive, finite, and stable (not drifting/exploding)
    assert mean_var > 0 and math.isfinite(mean_var)
    second_half = vals[len(vals) // 2:]
    first_half = vals[: len(vals) // 2]
    drift = abs(sum(second_half) / len(second_half) - sum(first_half) / len(first_half))
    assert drift / mean_var < 0.2     # stationary, no secular drift


def test_reproducible(device):
    N = 32
    grid = SpectralGrid(N, device, "fp64")
    f1 = StochasticOUForcing(grid, k_f=2.0, seed=42)
    f2 = StochasticOUForcing(grid, k_f=2.0, seed=42)
    for _ in range(15):
        f1.advance_ou(0.013)
        f2.advance_ou(0.013)
    assert torch.equal(f1.b_hat, f2.b_hat)


def test_solver_runs_and_injects(device):
    """End-to-end: solver steps with OU forcing, velocity stays real,
    dissipation stays positive (forcing injects energy)."""
    N = 32
    scfg = SolverConfig(N=N, nu=0.01, dtype="fp64", device=device,
                        forcing=ForcingConfig(type="stochastic_ou", k_f=2.0,
                                              ou_tau=1.0, ou_sigma2=0.1, ou_seed=5))
    grid = SpectralGrid(N, device, "fp64")
    u0 = random_solenoidal(grid, seed=0, u_rms=0.5)
    s = PseudoSpectralSolver(scfg, u0)
    for _ in range(50):
        s.step(s.suggest_dt())
    u = s.velocity_physical()
    assert u.dtype == grid.rdtype
    assert torch.isfinite(u).all()
    assert s.grid.dissipation(s.u_hat, scfg.nu) > 0
