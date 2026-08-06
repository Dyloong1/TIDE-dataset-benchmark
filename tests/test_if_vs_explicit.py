"""Adversarial check of the integrating-factor (Lawson-form) integrators
against a plainly-explicit reference: classical RK4 with the viscous term
simply added to the RHS (provably correct, no variable transformation),
implemented independently in this test.

Rationale (spec review): naively multiplying exp(-nu k^2 dt) into RK stage
combinations introduces a low-order splitting error. Our implementation is
the exact Lawson transformation instead (per-stage intervals
(1/3, 5/12, 1/4) dt and the low-storage register transformed alongside the
state); these tests pin that down by direct comparison.
"""
import torch

from solver.config import ForcingConfig, SolverConfig
from solver.grids import SpectralGrid
from solver.initial_conditions import taylor_green
from solver.solver import PseudoSpectralSolver


def _solver(N, nu, device, scheme="rk3"):
    scfg = SolverConfig(N=N, nu=nu, dtype="fp64", device=device, scheme=scheme,
                        forcing=ForcingConfig(type="none"))
    grid = SpectralGrid(N, device, "fp64")
    return PseudoSpectralSolver(scfg, taylor_green(grid))


def _explicit_rk4(s, T, n_steps):
    """Independent reference: du/dt = N(u) - nu k^2 u, classical RK4,
    no integrating factor anywhere."""
    nu_k2 = (s.cfg.nu * s.grid.k2)
    def F(u_hat):
        out = torch.empty_like(u_hat)
        s.rhs_(u_hat, out)
        out -= nu_k2 * u_hat
        return out
    u = s.u_hat.clone()
    dt = T / n_steps
    for _ in range(n_steps):
        k1 = F(u)
        k2 = F(u + (0.5 * dt) * k1)
        k3 = F(u + (0.5 * dt) * k2)
        k4 = F(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return u


def test_if_rk4_matches_explicit_rk4(device):
    """Same order, same dt: IF-RK4 and explicit RK4 differ only at the
    truncation level (no splitting error)."""
    N, nu, T, n = 32, 0.01, 0.5, 500
    s_exp = _solver(N, nu, device, "rk4")
    u_explicit = _explicit_rk4(s_exp, T, n)

    s_if = _solver(N, nu, device, "rk4")
    for _ in range(n):
        s_if.step(T / n)
    rel = float((s_if.u_hat - u_explicit).abs().max() / u_explicit.abs().max())
    assert rel < 1e-9, f"IF-RK4 vs explicit RK4 rel diff {rel:.3e}"


def test_if_rk3_converges_to_explicit_reference(device):
    """IF-RK3 must converge (3rd order) to the explicit-RK4 fine-step
    trajectory — a splitting error would leave a dt-independent or low-order
    residual against this independently-coded reference."""
    N, nu, T = 32, 0.01, 0.5
    ref = _explicit_rk4(_solver(N, nu, device, "rk4"), T, 4000)

    errs = []
    for n in (50, 100):
        s = _solver(N, nu, device, "rk3")
        for _ in range(n):
            s.step(T / n)
        errs.append(float((s.u_hat - ref).abs().max()))
    order = (errs[0] / errs[1]) ** 0.5  # halving dt -> 2^3, sqrt -> ~2.83
    import math
    p = math.log2(errs[0] / errs[1])
    assert 2.6 < p < 3.4, f"order vs explicit reference: {p:.2f} (errs {errs})"
