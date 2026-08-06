"""Extended-physics D-group (equation-residual) check.

Verifies the NEW physics terms satisfy their equations, via a centered-difference
NS-momentum residual computed by stepping a saved checkpoint forward/back with
FROZEN forcing (same protocol as phase0 run_trajectory_showcase, extended with the
Coriolis / buoyancy terms):

  D1  divergence residual <(div u)^2>/<|grad u|^2>           <= 1e-6
  D2  NS-momentum residual ||(C-A)/2h - RHS^n|| / ||dudt||   <= 1e-2
      RHS includes Coriolis (-2 Omega x u) for rotation and the buoyancy force
      (+b e_z) for stratification. (D2 checks the MOMENTUM residual only; the buoyancy
      transport eqn is covered by analytic tests, not a per-config residual here.)
  D3  halve h -> residual ratio in [2.5,6] (O(h^2) truncation, not an eqn error)

The analytic-solution tests in tests/test_physics_ext.py already pin the dynamics
exactly (inertial-oscillation period, N^2=0 bit-identity, gravity-wave dispersion);
this gives the per-config equation-residual numbers for the appendix.

    python eval_dynamics_ext.py <case_name>      # reads results/<case>/ckpt_t*.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:  # Windows cp1252 console: printed appendix/status may contain Chinese -> guard
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[1] / "phase0"))

import solver._env  # noqa: F401,E402
import torch  # noqa: E402

from physics_ext.config_ext import load_ext_config  # noqa: E402
from physics_ext.operators_ext import cross_product_spectral  # noqa: E402
from solver.operators import leray_project_  # noqa: E402

RES = _HERE.parents[0] / "results"
_FFT = (-3, -2, -1)


def _d4_residual(u_hat, grid, rot, scal, strat, b_hat=None):
    """D4 velocity-pressure consistency for the extended-physics pressure operator.
    Picks the certified operator by physics (rotating -> pressure_hat_rotating with
    the +2 Omega.omega source; stratified -> pressure_hat_boussinesq with +d_z b;
    isotropic/scalar -> solver.operators.pressure_hat, since a passive scalar does
    not react on momentum). Returns (poisson_res, gradp_res), both <=1e-8 = certified.
    Same math as tests/test_physics_ext.py::_poisson_and_gradp_res (which is the
    judge-tested target); here on the real 256^3 checkpoint field for the appendix."""
    from solver.operators import pressure_hat
    from physics_ext.operators_ext import (pressure_hat_rotating,
                                           pressure_hat_boussinesq, cross_product_spectral)
    N = grid.N
    ks = (grid.kx, grid.ky, grid.kz)
    u = torch.fft.irfftn(u_hat, s=(N, N, N), dim=_FFT)
    adv_hat = torch.zeros_like(u_hat)
    poisson_rhs = torch.zeros(N, N, grid.Nh, dtype=grid.cdtype, device=grid.device)
    for i in range(3):
        for j in range(3):
            tij = torch.fft.rfftn(u[i] * u[j], dim=_FFT)
            poisson_rhs += (ks[i] * ks[j]) * tij
            adv_hat[i] += (1j * ks[j]) * tij
    extra_force = None
    if rot.enabled:
        p_hat = pressure_hat_rotating(u_hat, rot.omega_vec, grid)
        cx = 1j * (ks[1] * u_hat[2] - ks[2] * u_hat[1])
        cy = 1j * (ks[2] * u_hat[0] - ks[0] * u_hat[2])
        cz = 1j * (ks[0] * u_hat[1] - ks[1] * u_hat[0])
        ov = rot.omega_vec
        poisson_rhs = poisson_rhs + 2.0 * (ov[0] * cx + ov[1] * cy + ov[2] * cz)
        extra_force = cross_product_spectral(ov, u_hat).clone().mul_(-2.0)   # -2 Omega x u
    elif strat.enabled and b_hat is not None:
        p_hat = pressure_hat_boussinesq(u_hat, b_hat, grid)
        poisson_rhs = poisson_rhs + (1j * grid.kz) * b_hat[0]
        extra_force = torch.zeros_like(u_hat); extra_force[2] = b_hat[0]       # +b e_z
    else:
        p_hat = pressure_hat(u_hat, grid)
    lap_p = (-grid.k2) * p_hat
    poisson_res = float((lap_p - poisson_rhs).abs().max() / max(float(poisson_rhs.abs().max()), 1e-300))
    f = adv_hat.clone()
    if extra_force is not None:
        f = f - extra_force
    div_f = (ks[0] * f[0] + ks[1] * f[1] + ks[2] * f[2]) * grid.inv_k2
    irrot = torch.stack([-ks[i] * div_f for i in range(3)])
    grad_p = torch.stack([(1j * ks[i]) * p_hat for i in range(3)])
    gradp_res = float((irrot - grad_p).abs().max() / max(float(grad_p.abs().max()), 1e-300))
    return poisson_res, gradp_res


def _div_residual(u_hat, grid):
    N = grid.N
    div = 1j * (grid.kx * u_hat[0] + grid.ky * u_hat[1] + grid.kz * u_hat[2])
    d = torch.fft.irfftn(div, s=(N, N, N), dim=_FFT)
    gradsq = 0.0
    for c in range(3):
        for k in (grid.kx, grid.ky, grid.kz):
            g = torch.fft.irfftn(1j * k * u_hat[c], s=(N, N, N), dim=_FFT)
            gradsq += float((g * g).mean())
    return float((d * d).mean()) / max(gradsq, 1e-30)


def _full_rhs(s):
    """Full continuous-time RHS d u_hat/dt at the current state, including the
    viscous term (which the time stepper handles via the integrating factor, so
    rhs_ omits it). rhs_ already includes nonlinear + forcing + Coriolis (for
    RotatingSolver), all Leray-projected. For BoussinesqSolver we use its
    velocity-RHS hook (includes the buoyancy force)."""
    u0 = s.u_hat.clone()
    rhs = torch.empty_like(u0)
    if hasattr(s, "_rhs_u"):                       # BoussinesqSolver
        s._rhs_u(u0, rhs, record_stage0=False)
    else:
        s.rhs_(u0, rhs)
    rhs = rhs + (-s.nu_k2) * u0                    # add viscosity (IF term)
    return rhs


def _step_frozen(s, h):
    """Advance the underlying solver by h with the OU forcing increment FROZEN
    (so a centered difference measures the true NS residual, not OU noise)."""
    frozen = None
    base = s.base if hasattr(s, "base") else s
    if hasattr(base, "forcing") and hasattr(base.forcing, "b_hat"):
        frozen = base.forcing.b_hat.clone()
    s.step(h)
    if frozen is not None:
        base.forcing.b_hat.copy_(frozen)


def _momentum_residual(s, h):
    """Centered-difference NS-momentum residual with frozen forcing. From state A,
    step to B (h) then C (2h); residual at B = (C-A)/2h - RHS(B). Returns
    ||r||/||du/dt|| (space-spectral exact, so the residual is pure time truncation
    -> shrinks O(h^2), checked by D3)."""
    A = s.u_hat.clone()
    _step_frozen(s, h)
    B = s.u_hat.clone()
    rhsB = _full_rhs(s)                            # RHS evaluated AT B
    dudt_norm = float(rhsB.abs().pow(2).sum().sqrt())
    _step_frozen(s, h)
    C = s.u_hat.clone()
    s.u_hat.copy_(A)                               # restore
    r = (C - A) / (2.0 * h) - rhsB
    return float(r.abs().pow(2).sum().sqrt()) / max(dudt_norm, 1e-30)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else None
    if not case:
        raise SystemExit("usage: eval_dynamics_ext.py <case>")
    rec = RES / case
    cfgpath = rec / "config_snapshot.yaml"
    cfg, rot, scal, strat = load_ext_config(cfgpath)
    cks = sorted(rec.glob("ckpt_t*.pt"))
    if not cks:
        raise SystemExit(f"no checkpoints in {rec} (run with --checkpoint-every)")
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False)

    # rebuild the matching solver
    from solver.grids import SpectralGrid
    grid = SpectralGrid(cfg.solver.N, cfg.solver.device, cfg.solver.dtype)
    u_hat = ck["u_hat"].to(grid.device, grid.cdtype)
    if rot.enabled:
        from physics_ext.rotating import RotatingSolver
        s = RotatingSolver(cfg.solver, u_hat, rot)
    elif strat.enabled:
        from physics_ext.boussinesq import BoussinesqSolver
        b_hat = ck.get("b_hat", torch.zeros((1, grid.N, grid.N, grid.Nh),
                                            dtype=grid.cdtype, device=grid.device))
        s = BoussinesqSolver(cfg.solver, strat, u_hat, b_hat.to(grid.device, grid.cdtype))
    else:
        from solver.solver import PseudoSpectralSolver
        s = PseudoSpectralSolver(cfg.solver, u_hat)

    d1 = _div_residual(s.u_hat, grid)
    # D2 at two step sizes for the D3 O(h^2) ratio
    saved = s.u_hat.clone()
    h1 = 1e-3
    s.u_hat.copy_(saved); r1 = _momentum_residual(s, h1)
    s.u_hat.copy_(saved); r2 = _momentum_residual(s, h1 / 2)
    s.u_hat.copy_(saved)
    ratio = r1 / max(r2, 1e-30)

    # D4 velocity-pressure consistency (certifies the pressure CHANNEL released in the
    # corpus): the ext pressure operator must satisfy the pressure-Poisson eqn + grad(p)
    # balance for THIS physics. b_hat for stratified comes from the checkpoint.
    _bh = getattr(s, "b_hat", None)
    d4_pois, d4_gradp = _d4_residual(s.u_hat, grid, rot, scal, strat, b_hat=_bh)

    out = {
        "case": case, "physics": cfg.name,
        "D1_div_residual": d1, "D1_pass": d1 <= 1e-6,
        "D2_residual_h": r1, "D2_residual_h2": r2,
        "D2_pass": r1 <= 1e-2,
        "D3_ratio": ratio, "D3_pass": 2.5 <= ratio <= 6.0,   # CLAUDE.md D-group band
        "D4_poisson_res": d4_pois, "D4_gradp_res": d4_gradp,
        "D4_pass": d4_pois <= 1e-8 and d4_gradp <= 1e-8,
    }
    print(json.dumps(out, indent=2, default=float))
    (rec / "dynamics_ext.json").write_text(json.dumps(out, indent=2, default=float),
                                           encoding="utf-8")
    print(f"\n[written] {rec / 'dynamics_ext.json'}")


if __name__ == "__main__":
    main()
