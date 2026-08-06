"""Regression test for the A12 incompressibility bug: the energy-preserving
forcing's systematic gain > 1 exponentially amplified fp32 projection
round-off in the forced shells (no viscous decay at k <= 2, state never
re-projected). Fixed by re-projecting in post_step.
"""
import torch

from solver.config import ForcingConfig, SolverConfig
from solver.grids import SpectralGrid
from solver.initial_conditions import random_solenoidal
from solver.solver import PseudoSpectralSolver


def _div_ratio(s) -> float:
    uh, g = s.u_hat, s.grid
    divh = g.kx * uh[0] + g.ky * uh[1] + g.kz * uh[2]
    w = g.rfft_weight
    d2 = float((w * divh.abs() ** 2).sum())
    g2 = float((w * (g.k2 * (uh.abs() ** 2).sum(0))).sum())
    return d2 / g2


def test_fp32_forced_run_stays_solenoidal(device):
    """2000 fp32 steps with energy-preserving forcing: <(div u)^2>/<|grad u|^2>
    must stay at the round-off floor (A12 threshold is 1e-6; without the
    post_step re-projection this reaches ~1e-12 at 2000 steps and grows
    exponentially thereafter)."""
    N = 64
    grid = SpectralGrid(N, device, "fp32")
    u0 = random_solenoidal(grid, seed=3)
    s = PseudoSpectralSolver(
        SolverConfig(N=N, nu=0.005, dtype="fp32",
                     forcing=ForcingConfig(type="energy_preserving",
                                           k_f=2.0, E_f0=0.3)), u0)
    for _ in range(2000):
        s.step(s.suggest_dt())
    assert _div_ratio(s) < 1e-13
