"""Time integration: exact viscous decay (integrating factor), RK3 convergence
order ~3 against an RK4 fine-step reference, energy conservation of the
inviscid nonlinear term."""
import math

import torch

from solver.config import ForcingConfig, SolverConfig
from solver.grids import SpectralGrid
from solver.initial_conditions import random_solenoidal, taylor_green
from solver.solver import PseudoSpectralSolver


def _make(N, nu, dtype, device, scheme="rk3", ic="tg", seed=0):
    scfg = SolverConfig(N=N, nu=nu, dtype=dtype, device=device, scheme=scheme,
                        forcing=ForcingConfig(type="none"))
    grid = SpectralGrid(N, device, dtype)
    u0 = taylor_green(grid) if ic == "tg" else random_solenoidal(grid, seed=seed)
    return PseudoSpectralSolver(scfg, u0)


def test_pure_viscous_decay_exact(device):
    """With the nonlinear term zeroed, each mode must decay by exactly
    exp(-nu k^2 t) regardless of dt (integrating factor is exact)."""
    N, nu = 32, 0.05
    s = _make(N, nu, "fp64", device)
    # monkeypatch: disable nonlinear term, keep the integrator path intact
    real_rhs = s.rhs_
    s.rhs_ = lambda u_hat, out, record_stage0=False: out.zero_()
    u0 = s.u_hat.clone()
    dt, n = 0.05, 8
    for _ in range(n):
        s.step(dt)
    decay = torch.exp(-nu * s.grid.k2 * (dt * n)).to(u0.real.dtype)
    expected = u0 * decay
    err = (s.u_hat - expected).abs().max() / u0.abs().max()
    assert float(err) < 1e-13
    s.rhs_ = real_rhs


def test_inviscid_energy_conservation(device):
    """nu=0: the rotational-form nonlinear term conserves energy; over a short
    TG run K should drift only at the time-discretization level."""
    N = 32
    s = _make(N, 0.0, "fp64", device)
    K0 = s.grid.kinetic_energy(s.u_hat)
    dt = 0.005
    for _ in range(100):
        s.step(dt)
    K1 = s.grid.kinetic_energy(s.u_hat)
    assert abs(K1 - K0) / K0 < 1e-9


def test_rk3_convergence_order(device):
    """RK3 self-convergence vs an RK4 reference at dt/20: halving dt must
    shrink the error by ~2^3."""
    N, nu, T = 32, 1.0 / 100.0, 0.5

    def final_state(scheme, dt):
        s = _make(N, nu, "fp64", device, scheme=scheme)
        n = round(T / dt)
        for _ in range(n):
            s.step(T / n)
        return s.u_hat

    ref = final_state("rk4", T / 400)
    err1 = float((final_state("rk3", T / 20) - ref).abs().max())
    err2 = float((final_state("rk3", T / 40) - ref).abs().max())
    order = math.log2(err1 / err2)
    assert 2.6 < order < 3.4, f"observed order {order:.2f}"


def test_fp32_fp64_same_code_path(device):
    """Both dtypes run and stay close over a short horizon."""
    N, nu = 32, 0.02
    s32 = _make(N, nu, "fp32", device)
    s64 = _make(N, nu, "fp64", device)
    for _ in range(20):
        s32.step(0.01)
        s64.step(0.01)
    K32 = s32.grid.kinetic_energy(s32.u_hat)
    K64 = s64.grid.kinetic_energy(s64.u_hat)
    assert abs(K32 - K64) / K64 < 1e-5
