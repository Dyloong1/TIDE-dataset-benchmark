"""ScalarTransport — a passive scalar theta advected by an UNCHANGED NS velocity.

Milestone 2 of the extended-physics phase. The momentum equation is NOT touched
(the flow is still pure incompressible NS — any validated config's u field is
reused verbatim). We only add a transported scalar:

    d theta/dt = -u . grad(theta) - w * G + kappa lap(theta)

  * -u.grad(theta): pseudo-spectral advection (spectral grad -> physical product ->
    FFT back -> dealias). Scalars have NO solenoidal constraint, so NO Leray.
  * -w*G: optional uniform mean-gradient source (G = dTheta_bg/dz). With G>0 the
    vertical velocity stirring the background gradient sustains a STATIONARY scalar
    variance (so the scalar reaches a forced-steady state alongside forced-steady u).
    G=0 gives a freely-decaying scalar.
  * kappa lap(theta): scalar diffusion, done EXACTLY via an integrating factor
    exp(-kappa k^2 dtau) per RK substage (same Williamson RK3 structure as the
    viscous IF in solver/timestepping.py, re-implemented here so that file is
    untouched).

Resolution note (256^3 DNS): the smallest scalar scale is the Batchelor scale
eta_B = eta / sqrt(Sc), Sc = nu/kappa. Choosing kappa = nu (Sc=1) makes eta_B = eta,
so the existing k_max*eta >= 1.5 gate also resolves the scalar. Sc >> 1 needs finer
grids than 256^3 and is honestly out of scope.

Stepped TOGETHER with the velocity: the driver advances u by one RK3 step, then
this advances theta over the same dt using the velocity's physical field. (For a
fully coupled single-RK variant see boussinesq.py; for a passive scalar the
operator-split per-step is standard and second-order in the scalar.)
"""
from __future__ import annotations

import torch

from solver.grids import SpectralGrid
from solver.operators import dealias_
from solver.timestepping import RK3_A, RK3_B, RK3_DTAU

from .operators_ext import scalar_gradient_phys

_FFT_DIMS = (-3, -2, -1)


class ScalarTransport:
    """Carries theta_hat [1,N,N,Nh] and advances it given the flow's u_phys."""

    def __init__(self, grid: SpectralGrid, scalar_cfg, theta_hat0: torch.Tensor):
        self.grid = grid
        self.cfg = scalar_cfg
        self.kappa = float(scalar_cfg.kappa)
        self.mean_grad = float(scalar_cfg.mean_grad)
        N, Nh = grid.N, grid.Nh
        self.theta_hat = theta_hat0.to(grid.device, grid.cdtype).clone()
        dealias_(self.theta_hat, grid)
        # buffers
        self.q_buf = torch.zeros((1, N, N, Nh), dtype=grid.cdtype, device=grid.device)
        self.rhs_buf = torch.empty((1, N, N, Nh), dtype=grid.cdtype, device=grid.device)
        self.kappa_k2 = (self.kappa * grid.k2).to(grid.rdtype)
        self._if_dt = None
        self._if_E = None

    # ---- integrating factors for kappa k^2 (mirror solver.timestepping) -------
    def _ifs(self, dt: float):
        if self._if_dt != dt:
            self._if_dt = dt
            self._if_E = [torch.exp(self.kappa_k2 * (-tau * dt)) for tau in RK3_DTAU]
        return self._if_E

    # ---- scalar RHS: -u.grad(theta) - w*G  (diffusion handled by the IF) ------
    def rhs_(self, theta_hat, u_phys, out):
        grid = self.grid
        N = grid.N
        g = scalar_gradient_phys(theta_hat[0], grid)          # [3,N,N,N] = grad theta
        adv = u_phys[0] * g[0] + u_phys[1] * g[1] + u_phys[2] * g[2]   # u.grad theta
        if self.mean_grad != 0.0:
            adv = adv + self.mean_grad * u_phys[2]            # + w*G  -> RHS gets -(u.grad+wG)
        torch.fft.rfftn(adv, dim=_FFT_DIMS, out=out[0])
        out.mul_(-1.0)
        dealias_(out, grid)
        return out

    def step(self, u_phys, dt: float):
        """Advance theta by dt with the scalar's own Williamson RK3 (3rd order in
        theta), holding the velocity field u_phys FIXED across the three substages.

        IMPORTANT (deep-review 2026-06-23, convergence-tested): with a time-frozen
        u this is a Lie operator split, so the COUPLED (u, theta) advance is only
        1st-order globally (measured p~1.1). For a 2nd-order coupled advance use the
        Strang scheme via the driver: theta.step(u_t, dt/2); solver.step(dt);
        theta.step(u_{t+dt}, dt/2)  -> measured p~2.1. The run entries use
        step_strang() which packages this; call step() directly only if 1st-order
        coupling is acceptable.
        """
        theta, q, rhs = self.theta_hat, self.q_buf, self.rhs_buf
        E3 = self._ifs(dt)
        q.zero_()
        for s in range(3):
            self.rhs_(theta, u_phys, rhs)
            if s == 0:
                q.copy_(rhs).mul_(dt)
            else:
                q.mul_(RK3_A[s]).add_(rhs, alpha=dt)
            E = E3[s]
            theta.add_(q, alpha=RK3_B[s]).mul_(E)
            q.mul_(E)
        return theta
