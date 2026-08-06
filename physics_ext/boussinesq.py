"""BoussinesqSolver — stratified turbulence (active scalar = buoyancy, two-way coupled).

Milestone 3 of the extended-physics phase. Now the scalar (buoyancy b) reacts back
on the momentum equation:

    du/dt = u x omega - grad(p) + nu lap u + f  + b e_z          (buoyancy force)
    db/dt = -u . grad(b) - N^2 w + kappa lap b                   (buoyancy transport)

  * +b e_z: linear, z-only body force. For a solenoidal u it stays solenoidal,
    BUT it does change the pressure (see pressure_hat_boussinesq) and hence D4.
  * -N^2 w: the stratification restoring term (N^2 = squared Brunt-Vaisala freq,
    w = u_z). Drives internal gravity waves with dispersion omega = N k_perp/|k|.
  * Both viscous nu lap u and diffusive kappa lap b are exact via SEPARATE
    integrating factors (different coefficients per mode).
  * u and b are advanced TOGETHER in one Williamson RK3 (coupled state), because
    the momentum RHS now depends on b and the buoyancy RHS on u — a per-substage
    operator split would lose the wave coupling. We re-implement the RK3 here
    (solver/timestepping.py untouched).

Reuses solver's velocity RHS via a base PseudoSpectralSolver instance for the NS
part (nonlinear + forcing + Leray + dealias), then adds buoyancy. solver/ untouched.

Resolution (256^3 DNS): besides k_max*eta>=1.5, stratified DNS must resolve the
Ozmidov scale l_O = sqrt(eps/N^3) and have buoyancy Reynolds Re_b = eps/(nu N^2)
large enough for a turbulent inertial range (Re_b too small = wave-dominated, not
turbulence). Those are acceptance gates (diagnostics_ext / eval), not solver knobs.
"""
from __future__ import annotations

import torch

from solver.grids import SpectralGrid
from solver.operators import dealias_
from solver.solver import PseudoSpectralSolver
from solver.timestepping import RK3_A, RK3_B, RK3_DTAU

from .operators_ext import scalar_gradient_phys

_FFT_DIMS = (-3, -2, -1)


class BoussinesqSolver:
    """Coupled (u, b) stratified solver. Holds a base solver for the NS velocity
    RHS and adds the buoyancy force + buoyancy transport."""

    def __init__(self, scfg, strat_cfg, u_hat0, b_hat0):
        self.base = PseudoSpectralSolver(scfg, u_hat0)
        g = self.grid = self.base.grid
        N, Nh = g.N, g.Nh
        self.N_sq = float(strat_cfg.N_sq)
        self.kappa = float(strat_cfg.kappa)
        self.b_hat = b_hat0.to(g.device, g.cdtype).clone()
        dealias_(self.b_hat, g)
        # buoyancy buffers + its own integrating factor for kappa k^2
        self.qb_buf = torch.zeros((1, N, N, Nh), dtype=g.cdtype, device=g.device)
        self.rb_buf = torch.empty((1, N, N, Nh), dtype=g.cdtype, device=g.device)
        self.b_force_buf = torch.empty((3, N, N, Nh), dtype=g.cdtype, device=g.device)
        self.kappa_k2 = (self.kappa * g.k2).to(g.rdtype)
        self.t = 0.0
        self.n_steps = 0
        self._if_dt = None

    # convenience passthroughs so the shared run loop (_ext_runner) can drive this
    # like a PseudoSpectralSolver.
    @property
    def u_hat(self):
        return self.base.u_hat

    def scalars(self) -> dict:
        return self.base.scalars()

    def velocity_physical(self):
        return self.base.velocity_physical()

    def suggest_dt(self, t_target=None):
        return self.base.suggest_dt(t_target=t_target)

    @property
    def nu_k2(self):
        return self.base.nu_k2

    def _ifs(self, dt: float):
        if self._if_dt != dt:
            self._if_dt = dt
            self._nu_E = [torch.exp(self.base.nu_k2 * (-tau * dt)) for tau in RK3_DTAU]
            self._kp_E = [torch.exp(self.kappa_k2 * (-tau * dt)) for tau in RK3_DTAU]
        return self._nu_E, self._kp_E

    # ---- momentum RHS incl. buoyancy force; buoyancy RHS incl. -N^2 w ---------
    def _rhs_u(self, u_hat, out, record_stage0):
        # base NS velocity RHS (nonlinear+forcing+ls_damp, Leray-projected/dealiased)
        self.base.rhs_(u_hat, out, record_stage0=record_stage0)
        # + buoyancy force b e_z : add b_hat to the z-component. b e_z is not
        # generally solenoidal, but the pressure (solved separately for diagnostics
        # / output) absorbs its compressible part; for the time advance we Leray-
        # project the buoyancy force so the evolved u stays divergence-free.
        self.b_force_buf.zero_()
        self.b_force_buf[2].copy_(self.b_hat[0])
        from solver.operators import leray_project_
        leray_project_(self.b_force_buf, self.grid)
        out.add_(self.b_force_buf)
        return out

    def _rhs_b(self, u_phys, b_hat, out):
        grid = self.grid
        g = scalar_gradient_phys(b_hat[0], grid)                 # grad b
        adv = u_phys[0] * g[0] + u_phys[1] * g[1] + u_phys[2] * g[2]   # u.grad b
        adv = adv + self.N_sq * u_phys[2]                        # + N^2 w -> RHS gets -(...)
        torch.fft.rfftn(adv, dim=_FFT_DIMS, out=out[0])
        out.mul_(-1.0)                                           # -u.grad b - N^2 w
        dealias_(out, grid)
        return out

    def step(self, dt: float):
        """Advance (u, b) by dt with a single coupled Williamson RK3."""
        base = self.base
        u_hat, qu, rhsu = base.u_hat, base.q_buf, base.rhs_buf
        b_hat, qb, rhsb = self.b_hat, self.qb_buf, self.rb_buf
        nu_E, kp_E = self._ifs(dt)
        qu.zero_(); qb.zero_()
        for s in range(3):
            self._rhs_u(u_hat, rhsu, record_stage0=(s == 0))
            # buoyancy RHS needs u in physical space at this stage
            torch.fft.irfftn(u_hat, s=(grid_N := self.grid.N, grid_N, grid_N),
                             dim=_FFT_DIMS, out=base.u_phys)
            self._rhs_b(base.u_phys, b_hat, rhsb)
            if s == 0:
                qu.copy_(rhsu).mul_(dt); qb.copy_(rhsb).mul_(dt)
            else:
                qu.mul_(RK3_A[s]).add_(rhsu, alpha=dt)
                qb.mul_(RK3_A[s]).add_(rhsb, alpha=dt)
            Eu, Eb = nu_E[s], kp_E[s]
            u_hat.add_(qu, alpha=RK3_B[s]).mul_(Eu); qu.mul_(Eu)
            b_hat.add_(qb, alpha=RK3_B[s]).mul_(Eb); qb.mul_(Eb)
        # advance the forcing state EXACTLY as PseudoSpectralSolver.step does — we
        # reimplement the RK3 here so we must replicate the post-step forcing update
        # (deep-review round 4, 2026-06-23: omitting this left a stochastic-OU forced
        # stratified run with a FROZEN random field -> violates Eswaran-Pope. Verified
        # OU b_hat changed 0.0 over 20 steps before this fix, ~311 after).
        if base.post_forcing:
            base.last_eps_inj = base.forcing.post_step(u_hat, dt)
        if base.stateful_forcing:
            base.forcing.advance_ou(dt)
        self.t += dt
        self.n_steps += 1
        base.t = self.t
        base.n_steps = self.n_steps
        return u_hat, b_hat
