"""Vetted acceptance tests for physics_ext (rotation / passive scalar / Boussinesq).

Judge-first: these analytic-solution targets must pass BEFORE the extended-physics
solvers are used to produce any corpus data. They also pin the ISOLATION invariant
(physics_ext must reduce to the validated solver when the new physics is off).

Run on CUDA fp64 (production precision). Mirrors the inline validations:
  - rotation: inertial oscillation period T=2pi/(2 Omega) + Coriolis does no work
  - passive scalar: variance conserved at kappa=0 under div-free advection
  - Boussinesq: N^2=0,b=0 -> bit-identical to pure NS; gravity-wave dispersion
  - resolution guards: Batchelor (Sc) + Ozmidov/Re_b flags behave
  - anti-tests: a wrong-sign / mislabeled case must NOT spuriously pass
"""
import math

import solver._env  # noqa: F401
import numpy as np
import pytest
import torch

from solver.config import SolverConfig
from solver.grids import SpectralGrid
from solver.initial_conditions import random_solenoidal
from solver.solver import PseudoSpectralSolver
from physics_ext import diagnostics_ext as dx
from physics_ext.boussinesq import BoussinesqSolver
from physics_ext.config_ext import (RotationConfig, ScalarConfig,
                                     StratificationConfig)
from physics_ext.rotating import RotatingSolver
from physics_ext.scalar import ScalarTransport

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="physics_ext DNS needs CUDA")


def _cfg(N, nu, **kw):
    c = SolverConfig(N=N, nu=nu, dtype="fp64", device="cuda", scheme="rk3",
                     cfl=kw.get("cfl", 0.3), dt_max=kw.get("dt_max", 0.008))
    c.forcing.type = "none"
    return c


# ============================================================= rotation
def test_coriolis_inertial_oscillation():
    """Uniform flow under -2 Omega_z x u rotates at the inertial frequency 2 Omega;
    after one period it returns, and |u| is unchanged (Coriolis does no work)."""
    N = 16
    cfg = _cfg(N, nu=0.0, dt_max=0.01)
    grid = SpectralGrid(N, "cuda", "fp64")
    u_hat = torch.zeros((3, N, N, grid.Nh), dtype=grid.cdtype, device="cuda")
    u_hat[0, 0, 0, 0] = 1.0 * N**3                      # uniform u=(1,0,0)
    Omega = 0.5
    s = RotatingSolver(cfg, u_hat, RotationConfig(enabled=True, omega_z=Omega))
    T = 2 * math.pi / (2 * Omega)
    dt = 0.005
    for _ in range(int(round(T / dt))):
        s._stepper(s, dt)
    ux = float(s.u_hat[0, 0, 0, 0].real) / N**3
    uy = float(s.u_hat[1, 0, 0, 0].real) / N**3
    assert math.hypot(ux, uy) == pytest.approx(1.0, abs=1e-4), "Coriolis must do no work"
    assert math.hypot(ux - 1.0, uy) < 2e-2, "should return after one inertial period"


def test_coriolis_preserves_solenoidality():
    """REGRESSION (deep-review 2026-06-23): Omega x u is NOT divergence-free
    (div = Omega.omega), so the Coriolis term must be Leray-projected. With a real
    turbulent (non-uniform) field, div(u) must stay at machine precision over a run.
    The original inertial-oscillation test missed this because a uniform field has
    omega=0."""
    N = 32
    cfg = _cfg(N, nu=0.005)
    grid = SpectralGrid(N, "cuda", "fp64")
    u0 = random_solenoidal(grid, seed=0, k_p=3.0, u_rms=0.7)
    s = RotatingSolver(cfg, u0, RotationConfig(enabled=True, omega_z=2.0))

    def divmax(u_hat):
        d = 1j * (grid.kx * u_hat[0] + grid.ky * u_hat[1] + grid.kz * u_hat[2])
        return float(torch.fft.irfftn(d, s=(N, N, N), dim=(-3, -2, -1)).abs().max())

    for _ in range(15):
        s._stepper(s, 0.004)
    assert divmax(s.u_hat) < 1e-10, "Coriolis must keep u divergence-free (Leray-projected)"


def test_coriolis_off_is_pure_ns():
    """RotatingSolver with rotation disabled == base solver bit-for-bit (isolation)."""
    N = 16
    cfg = _cfg(N, nu=0.01)
    grid = SpectralGrid(N, "cuda", "fp64")
    u0 = random_solenoidal(grid, seed=0, k_p=3.0, u_rms=0.7)
    ref = PseudoSpectralSolver(cfg, u0)
    rot = RotatingSolver(cfg, u0, RotationConfig(enabled=False))
    for _ in range(20):
        ref._stepper(ref, 0.005)
        rot._stepper(rot, 0.005)
    assert float((ref.u_hat - rot.u_hat).abs().max()) == 0.0


# ============================================================= passive scalar
def test_passive_scalar_variance_conserved():
    """kappa=0, no mean gradient: <theta^2> conserved under div-free advection
    (only RK3 truncation drift)."""
    N = 32
    cfg = _cfg(N, nu=0.005)
    grid = SpectralGrid(N, "cuda", "fp64")
    u0 = random_solenoidal(grid, seed=0, k_p=3.0, u_rms=0.7)
    s = PseudoSpectralSolver(cfg, u0)
    torch.manual_seed(1)
    th0 = torch.fft.rfftn(torch.randn(N, N, N, dtype=grid.rdtype, device="cuda"),
                          dim=(-3, -2, -1)).unsqueeze(0)
    scal = ScalarTransport(grid, ScalarConfig(enabled=True, kappa=0.0), th0)
    v0 = dx.scalar_variance(scal.theta_hat, grid)
    dt = 0.004
    for _ in range(50):
        s._stepper(s, dt)
        torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=(-3, -2, -1), out=s.u_phys)
        scal.step(s.u_phys, dt)
    v1 = dx.scalar_variance(scal.theta_hat, grid)
    assert abs(v1 - v0) / v0 < 1e-3, "passive scalar variance must be conserved at kappa=0"


def test_scalar_diffusion_decays_variance():
    """ANTI-TEST: with kappa>0 and no source, scalar variance must STRICTLY decay
    (a sign error in diffusion would grow it -> this must catch that)."""
    N = 32
    grid = SpectralGrid(N, "cuda", "fp64")
    cfg = _cfg(N, nu=0.005)
    s = PseudoSpectralSolver(cfg, random_solenoidal(grid, seed=2, k_p=3.0, u_rms=0.5))
    torch.manual_seed(3)
    th0 = torch.fft.rfftn(torch.randn(N, N, N, dtype=grid.rdtype, device="cuda"),
                          dim=(-3, -2, -1)).unsqueeze(0)
    scal = ScalarTransport(grid, ScalarConfig(enabled=True, kappa=0.02), th0)
    v0 = dx.scalar_variance(scal.theta_hat, grid)
    dt = 0.004
    for _ in range(40):
        s._stepper(s, dt)
        torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=(-3, -2, -1), out=s.u_phys)
        scal.step(s.u_phys, dt)
    v1 = dx.scalar_variance(scal.theta_hat, grid)
    assert v1 < v0, "kappa>0 must dissipate scalar variance (sign check)"


# ============================================================= Boussinesq
def test_boussinesq_zero_N_is_pure_ns():
    """N^2=0 AND b=0 -> BoussinesqSolver reproduces pure NS to machine precision
    (the strongest isolation-correctness proof)."""
    N = 32
    cfg = _cfg(N, nu=0.005)
    grid = SpectralGrid(N, "cuda", "fp64")
    u0 = random_solenoidal(grid, seed=0, k_p=3.0, u_rms=0.7)
    ref = PseudoSpectralSolver(cfg, u0)
    b0 = torch.zeros((1, N, N, grid.Nh), dtype=grid.cdtype, device="cuda")
    bsol = BoussinesqSolver(cfg, StratificationConfig(enabled=True, N_sq=0.0, kappa=0.005),
                            u0, b0)
    for _ in range(30):
        ref._stepper(ref, 0.004)
        bsol.step(0.004)
    assert float((ref.u_hat - bsol.u_hat).abs().max()) == 0.0


def test_boussinesq_advances_ou_forcing():
    """REGRESSION (deep-review round 4, 2026-06-23): BoussinesqSolver.step reimplements
    the RK3 and MUST replicate PseudoSpectralSolver.step's post-step forcing update.
    Omitting forcing.advance_ou left a stochastic-OU forced stratified run with a FROZEN
    random field (OU b_hat changed 0.0 over 20 steps) -> violates Eswaran-Pope. After
    the fix the OU state must evolve."""
    N = 24
    cfg = _cfg(N, nu=0.01)
    cfg.forcing.type = "stochastic_ou"
    cfg.forcing.k_f = 2.0; cfg.forcing.ou_tau = 1.0
    cfg.forcing.ou_sigma2 = 0.05; cfg.forcing.ou_seed = 1234
    grid = SpectralGrid(N, "cuda", "fp64")
    u0 = random_solenoidal(grid, seed=0, k_p=3.0, u_rms=0.7)
    b0 = torch.zeros((1, N, N, grid.Nh), dtype=grid.cdtype, device="cuda")
    s = BoussinesqSolver(cfg, StratificationConfig(enabled=True, N_sq=0.5, kappa=0.01), u0, b0)
    assert s.base.stateful_forcing, "test needs the OU forcing to be stateful"
    b_before = s.base.forcing.b_hat.clone()
    for _ in range(20):
        s.step(0.005)
    change = float((s.base.forcing.b_hat - b_before).abs().max())
    assert change > 1.0, f"OU forcing must advance in BoussinesqSolver, got change={change}"


def test_internal_gravity_wave_dispersion():
    """Linear internal wave on mode (kx,kz) oscillates at omega = N k_perp/|k|."""
    N = 32
    cfg = _cfg(N, nu=0.0, cfl=0.2, dt_max=0.004)
    grid = SpectralGrid(N, "cuda", "fp64")
    Nbv = 2.0
    kx_m, kz_m = 1, 2
    kperp, kmag = 1.0, math.sqrt(kx_m**2 + kz_m**2)
    omega = Nbv * kperp / kmag
    T = 2 * math.pi / omega
    eps = 1e-4
    u_hat = torch.zeros((3, N, N, grid.Nh), dtype=grid.cdtype, device="cuda")
    b_hat = torch.zeros((1, N, N, grid.Nh), dtype=grid.cdtype, device="cuda")
    u_hat[0, kz_m, 0, kx_m] = eps * kz_m * N**3           # solenoidal: u.k=0
    u_hat[2, kz_m, 0, kx_m] = -eps * kx_m * N**3
    bsol = BoussinesqSolver(cfg, StratificationConfig(enabled=True, N_sq=Nbv**2, kappa=0.0),
                            u_hat, b_hat)
    dt = 0.003
    ws, ts = [], []
    for i in range(int(round(2.5 * T / dt))):
        bsol.step(dt)
        ws.append(float(bsol.u_hat[2, kz_m, 0, kx_m].real))
        ts.append((i + 1) * dt)
    w, t = np.array(ws), np.array(ts)
    cross = np.where(np.diff(np.sign(w)) != 0)[0]
    T_meas = 2 * np.mean(np.diff(t[cross]))
    assert abs(T_meas - T) / T < 0.05, f"gravity-wave period {T_meas} vs theory {T}"


def test_buoyancy_flux_sign_in_unstable():
    """ANTI-TEST: with N^2<0 (unstable stratification) the flow releases PE ->
    buoyancy flux <wb> should be POSITIVE (convective). A buoyancy-sign error
    would flip this."""
    N = 32
    cfg = _cfg(N, nu=0.003, cfl=0.2, dt_max=0.004)
    grid = SpectralGrid(N, "cuda", "fp64")
    u0 = random_solenoidal(grid, seed=5, k_p=3.0, u_rms=0.05)   # weak seed
    # b anti-correlated initial: small noise; unstable N^2<0 amplifies overturning
    torch.manual_seed(7)
    b0 = torch.fft.rfftn(torch.randn(N, N, N, dtype=grid.rdtype, device="cuda") * 0.1,
                         dim=(-3, -2, -1)).unsqueeze(0)
    bsol = BoussinesqSolver(cfg, StratificationConfig(enabled=True, N_sq=-1.0, kappa=0.003),
                            u0, b0)
    for _ in range(60):
        bsol.step(0.003)
    flux = dx.buoyancy_flux(bsol.u_hat, bsol.b_hat, grid)
    assert flux > 0.0, "unstable stratification (N^2<0) must give positive buoyancy flux"


# ============================================================= resolution guards
def test_batchelor_resolution_gate():
    assert dx.batchelor_resolution(1.6, 1.0)["class_I"] is True      # Sc=1 -> same as eta
    assert dx.batchelor_resolution(1.6, 4.0)["class_I"] is False     # Sc=4 -> eta_B too small
    assert dx.batchelor_resolution(3.2, 4.0)["class_I"] is True      # enough margin


def test_stratified_resolution_flags():
    ok = dx.stratified_resolution(eta=0.01, l_O=0.05, L_int=1.5, Re_b=50)
    assert ok["ozmidov_resolved"] and ok["turbulent"]
    bad = dx.stratified_resolution(eta=0.01, l_O=0.005, L_int=1.5, Re_b=2)  # l_O<eta, Re_b small
    assert (not bad["ozmidov_resolved"]) and (not bad["turbulent"])


# ============================================================= anisotropy
def test_isotropic_field_small_anisotropy():
    """Isotropic IC -> b_ij eigenvalues ~ 0 (finite-N fluctuation only)."""
    N = 48
    grid = SpectralGrid(N, "cuda", "fp64")
    u = random_solenoidal(grid, seed=3, k_p=3.0, u_rms=0.7)
    _, lam = dx.anisotropy_tensor(u, grid)
    assert float(np.abs(lam).max()) < 0.08, "isotropic field must have near-zero anisotropy"
    assert abs(float(np.sum(lam))) < 1e-9, "b_ij is traceless by construction"


# ---------------------------------------------------------------------------
# D4 pressure certification for the extended-physics pressure operators (KDD
# pressure-channel decision 2026-06-24): a p-channel may only be RELEASED for a
# config whose pressure operator passes D4. These are the judge-first targets that
# gate exporting a pressure frame for rotating / stratified. CPU, N=32, fp64.
# ---------------------------------------------------------------------------
_FFTD = (-3, -2, -1)


def _poisson_and_gradp_res(p_hat, u_hat, grid, extra_source_hat=None, extra_force_hat=None):
    """Returns (poisson_res, gradp_res) relative residuals.
    Poisson: lap(p) = -d_i d_j(u_i u_j) [+ extra_source_hat].
    grad(p) balance: solver/operators.pressure_hat uses the convention
    grad(p) == (I-P)(+adv) for the pure-advection case (verified numerically:
    the isotropic pressure satisfies grad(p)==(I-P)(+adv) to machine precision,
    NOT (I-P)(-adv)). So the extra body force must be passed with the SAME sign
    convention as its Poisson source: a force whose div contributes +S to lap(p)
    enters the balance as +(that force). extra_force_hat must already carry that
    matching sign (the source S = -div(force)/... ; callers pass +(-2 Om x u) and
    +b e_z, whose divs give +2 Om.omega and +d_z b — consistent with the sources)."""
    N = grid.N
    ks = (grid.kx, grid.ky, grid.kz)
    u = torch.fft.irfftn(u_hat, s=(N, N, N), dim=_FFTD)
    lap_p = (-grid.k2) * p_hat
    poisson_rhs = torch.zeros_like(p_hat)
    adv_hat = torch.zeros_like(u_hat)
    for i in range(3):
        for j in range(3):
            tij = torch.fft.rfftn(u[i] * u[j], dim=_FFTD)
            poisson_rhs += (ks[i] * ks[j]) * tij        # = -d_i d_j(u_i u_j)
            adv_hat[i] += (1j * ks[j]) * tij             # (u.grad)u_i
    if extra_source_hat is not None:
        poisson_rhs = poisson_rhs + extra_source_hat     # lap(p) extra term (+d_z b, +2 Om.om)
    poisson_res = float((lap_p - poisson_rhs).abs().max() / max(float(poisson_rhs.abs().max()), 1e-300))
    # grad(p) == (I-P)(+adv - extra_force): adv enters +; a body force F with
    # div(F)=+S (S the Poisson source) enters as -F so its (I-P) part matches the
    # +S source sign (grad(p)=(I-P)(adv) for advection alone; adding source S means
    # subtracting F from the balanced field).
    f = adv_hat.clone()
    if extra_force_hat is not None:
        f = f - extra_force_hat
    div_f = (ks[0] * f[0] + ks[1] * f[1] + ks[2] * f[2]) * grid.inv_k2
    irrot = torch.stack([-ks[i] * div_f for i in range(3)])   # (I-P)(f) grad part
    grad_p = torch.stack([(1j * ks[i]) * p_hat for i in range(3)])
    gradp_res = float((irrot - grad_p).abs().max() / max(float(grad_p.abs().max()), 1e-300))
    return poisson_res, gradp_res


def test_D4_pressure_boussinesq():
    """Stratified pressure_hat_boussinesq satisfies lap(p) = -d_i d_j(u_i u_j) + d_z b
    and grad(p) balances the irrotational part of [-(u.grad)u + b e_z] to machine
    precision. Gates releasing the stratified p-channel."""
    from physics_ext.operators_ext import pressure_hat_boussinesq
    N = 32
    grid = SpectralGrid(N, "cpu", "fp64")
    u_hat = random_solenoidal(grid, seed=11, k_p=3.0, u_rms=0.7)
    torch.manual_seed(11)
    b = torch.randn(N, N, N, dtype=grid.rdtype) * 0.3
    b_hat = torch.fft.rfftn(b, dim=_FFTD).unsqueeze(0)
    p_hat = pressure_hat_boussinesq(u_hat, b_hat, grid)
    # buoyancy: lap(p) source +d_z b  -> (i k_z) b_hat ; body force +b e_z (z-comp only)
    src = (1j * grid.kz) * b_hat[0]
    force = torch.zeros_like(u_hat); force[2] = b_hat[0]
    pres, gradp = _poisson_and_gradp_res(p_hat, u_hat, grid, extra_source_hat=src, extra_force_hat=force)
    assert pres < 1e-10, f"Boussinesq Poisson residual {pres:.2e}"
    assert gradp < 1e-10, f"Boussinesq grad(p) balance residual {gradp:.2e}"


def test_D4_pressure_boussinesq_reduces_to_isotropic_at_b0():
    """b=0 -> pressure_hat_boussinesq == solver.operators.pressure_hat (isolation)."""
    from physics_ext.operators_ext import pressure_hat_boussinesq
    from solver.operators import pressure_hat
    N = 32
    grid = SpectralGrid(N, "cpu", "fp64")
    u_hat = random_solenoidal(grid, seed=12, u_rms=0.6)
    b0 = torch.zeros((1, N, N, grid.Nh), dtype=grid.cdtype)
    assert torch.allclose(pressure_hat_boussinesq(u_hat, b0, grid), pressure_hat(u_hat, grid),
                          atol=1e-14), "b=0 must reduce to the isotropic pressure"


def test_D4_pressure_rotating():
    """Rotating pressure_hat_rotating satisfies lap(p) = -d_i d_j(u_i u_j) + 2 Omega.omega
    and grad(p) balances the irrotational part of [-(u.grad)u - 2 Omega x u]. Gates
    releasing the rotating p-channel."""
    from physics_ext.operators_ext import pressure_hat_rotating, cross_product_spectral
    N = 32
    grid = SpectralGrid(N, "cpu", "fp64")
    u_hat = random_solenoidal(grid, seed=13, k_p=3.0, u_rms=0.7)
    omega_vec = (0.0, 0.0, 2.5)
    p_hat = pressure_hat_rotating(u_hat, omega_vec, grid)
    ks = (grid.kx, grid.ky, grid.kz)
    # Coriolis: lap(p) source +2 Omega.omega, omega_hat = i(k x u_hat)
    cx = 1j * (ks[1] * u_hat[2] - ks[2] * u_hat[1])
    cy = 1j * (ks[2] * u_hat[0] - ks[0] * u_hat[2])
    cz = 1j * (ks[0] * u_hat[1] - ks[1] * u_hat[0])
    src = 2.0 * (omega_vec[0] * cx + omega_vec[1] * cy + omega_vec[2] * cz)
    force = cross_product_spectral(omega_vec, u_hat).clone().mul_(-2.0)   # -2 Omega x u
    pres, gradp = _poisson_and_gradp_res(p_hat, u_hat, grid, extra_source_hat=src, extra_force_hat=force)
    assert pres < 1e-10, f"rotating Poisson residual {pres:.2e}"
    assert gradp < 1e-10, f"rotating grad(p) balance residual {gradp:.2e}"


def test_D4_pressure_rotating_reduces_to_isotropic_at_Omega0():
    """Omega=0 -> pressure_hat_rotating == solver.operators.pressure_hat (isolation)."""
    from physics_ext.operators_ext import pressure_hat_rotating
    from solver.operators import pressure_hat
    N = 32
    grid = SpectralGrid(N, "cpu", "fp64")
    u_hat = random_solenoidal(grid, seed=14, u_rms=0.6)
    assert torch.allclose(pressure_hat_rotating(u_hat, (0.0, 0.0, 0.0), grid),
                          pressure_hat(u_hat, grid), atol=1e-14), "Omega=0 must reduce to isotropic pressure"
