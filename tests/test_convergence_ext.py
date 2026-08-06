"""Convergence-order tests for the extended-physics time integrators (deep-review
round 3, 2026-06-23). These are the HARD evidence that the new time-stepping is
correct to the claimed order — the kind of test that catches the bugs code-reading
misses (e.g. the scalar Lie-split that was silently 1st-order).

Order p is measured by dt-halving against a fine-dt reference: p = log2(e(dt)/e(dt/2)).
Run on CUDA fp64.
"""
import math

import solver._env  # noqa: F401
import pytest
import torch

from solver.config import SolverConfig
from solver.grids import SpectralGrid
from solver.initial_conditions import random_solenoidal
from solver.solver import PseudoSpectralSolver
from physics_ext.config_ext import RotationConfig, ScalarConfig, StratificationConfig
from physics_ext.boussinesq import BoussinesqSolver
from physics_ext.rotating import RotatingSolver
from physics_ext.scalar import ScalarTransport

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="convergence tests need CUDA")
N = 24


def _orders(run_to, dts, T):
    ref = run_to(min(dts) / 4, T)
    errs = [float((run_to(dt, T) - ref).abs().pow(2).sum().sqrt()) for dt in dts]
    return [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)], errs


def _cfg(dt):
    c = SolverConfig(N=N, nu=0.01, dtype="fp64", device="cuda", scheme="rk3",
                     cfl=1.0, dt_max=dt)
    c.forcing.type = "none"
    return c


def test_scalar_strang_is_second_order():
    """The Strang-split coupled (u, theta) advance used by run_scalar_hit must be
    2nd order (the plain frozen-u Lie split was only 1st — this pins the fix)."""
    def run_to(dt, T):
        g = SpectralGrid(N, "cuda", "fp64")
        s = PseudoSpectralSolver(_cfg(dt), random_solenoidal(g, seed=0, k_p=3, u_rms=0.7))
        torch.manual_seed(1)
        th = torch.fft.rfftn(torch.randn(N, N, N, dtype=g.rdtype, device="cuda"),
                             dim=(-3, -2, -1)).unsqueeze(0)
        scal = ScalarTransport(g, ScalarConfig(enabled=True, kappa=0.0), th)
        uph = torch.empty((3, N, N, N), dtype=g.rdtype, device="cuda")
        for _ in range(int(round(T / dt))):
            torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=(-3, -2, -1), out=uph)
            scal.step(uph, dt * 0.5)
            s.step(dt)
            torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=(-3, -2, -1), out=uph)
            scal.step(uph, dt * 0.5)
        return scal.theta_hat.clone()
    orders, _ = _orders(run_to, [0.02, 0.01, 0.005], T=0.4)
    assert min(orders) > 1.7, f"Strang split must be ~2nd order, got {orders}"


def test_scalar_frozen_u_is_first_order():
    """REGRESSION: document that the plain frozen-u split (ScalarTransport.step alone,
    no Strang) is only 1st order — so nobody 'simplifies' the entry back to it
    thinking it stays accurate."""
    def run_to(dt, T):
        g = SpectralGrid(N, "cuda", "fp64")
        s = PseudoSpectralSolver(_cfg(dt), random_solenoidal(g, seed=0, k_p=3, u_rms=0.7))
        torch.manual_seed(1)
        th = torch.fft.rfftn(torch.randn(N, N, N, dtype=g.rdtype, device="cuda"),
                             dim=(-3, -2, -1)).unsqueeze(0)
        scal = ScalarTransport(g, ScalarConfig(enabled=True, kappa=0.0), th)
        uph = torch.empty((3, N, N, N), dtype=g.rdtype, device="cuda")
        for _ in range(int(round(T / dt))):
            s.step(dt)
            torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=(-3, -2, -1), out=uph)
            scal.step(uph, dt)
        return scal.theta_hat.clone()
    orders, _ = _orders(run_to, [0.02, 0.01, 0.005], T=0.4)
    assert max(orders) < 1.6, f"frozen-u split should be ~1st order, got {orders}"


def test_boussinesq_coupled_rk3_high_order():
    """The coupled (u,b) Williamson RK3 should be high order (>=2.5; RK3 is 3rd but
    finite-dt + nonlinearity usually measures ~2.7-3.0). Pins that the two-field
    coupling did not silently drop the integrator order."""
    def run_to(dt, T):
        g = SpectralGrid(N, "cuda", "fp64")
        u0 = random_solenoidal(g, seed=0, k_p=3, u_rms=0.7)
        torch.manual_seed(2)
        b0 = torch.fft.rfftn(torch.randn(N, N, N, dtype=g.rdtype, device="cuda") * 0.1,
                             dim=(-3, -2, -1)).unsqueeze(0)
        s = BoussinesqSolver(_cfg(dt), StratificationConfig(enabled=True, N_sq=1.0, kappa=0.01),
                             u0, b0)
        for _ in range(int(round(T / dt))):
            s.step(dt)
        return s.u_hat.clone()
    orders, _ = _orders(run_to, [0.01, 0.005, 0.0025], T=0.2)
    assert min(orders) > 2.5, f"coupled RK3 should be >=3rd order, got {orders}"


def test_coriolis_rk3_high_order():
    """RotatingSolver (Coriolis in the explicit RHS, viscosity via IF) must keep the
    base RK3 order (>=2.5)."""
    def run_to(dt, T):
        g = SpectralGrid(N, "cuda", "fp64")
        u0 = random_solenoidal(g, seed=0, k_p=3, u_rms=0.7)
        s = RotatingSolver(_cfg(dt), u0, RotationConfig(enabled=True, omega_z=2.0))
        for _ in range(int(round(T / dt))):
            s.step(dt)
        return s.u_hat.clone()
    orders, _ = _orders(run_to, [0.01, 0.005, 0.0025], T=0.2)
    assert min(orders) > 2.5, f"rotating RK3 should be >=3rd order, got {orders}"
