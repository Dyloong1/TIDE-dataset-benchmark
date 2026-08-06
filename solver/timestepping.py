"""Time integrators: Williamson low-storage RK3 and classical RK4, both with
the viscous term handled exactly by an integrating factor (knowledge base 2.1).

Derivation note (the easy-to-get-wrong part): with v = exp(-L t) u,
dv/dt = exp(-L t) N(exp(L t) v). We keep u_hat and the low-storage register q
expressed in the variable system of the *current stage time*; advancing the
reference time by dtau multiplies both by E = exp(-nu k^2 dtau). With
L = -nu k^2 only decaying exponentials ever appear. Stage times of Williamson
RK3 are c = (0, 1/3, 3/4, 1), hence dtau/dt = (1/3, 5/12, 1/4).

Verified by tests/test_timestepping.py: exact viscous decay when N(u)=0, and
third-order convergence against an RK4 fine-step reference.
"""
from __future__ import annotations

import torch

RK3_A = (0.0, -5.0 / 9.0, -153.0 / 128.0)
RK3_B = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)
RK3_DTAU = (1.0 / 3.0, 5.0 / 12.0, 1.0 / 4.0)   # c_{s+1} - c_s


def _integrating_factors(solver, dt: float):
    """exp(-nu k^2 dtau_s) for the three substages, cached on the solver.
    The CFL controller quantizes dt (snap-down grid), so consecutive steps
    usually share the same dt and the three large exp() evaluations are
    skipped."""
    if getattr(solver, "_if_dt", None) != dt:
        solver._if_dt = dt
        solver._if_E = [torch.exp(solver.nu_k2 * (-tau * dt)) for tau in RK3_DTAU]
    return solver._if_E


def rk3_williamson_step(solver, dt: float) -> None:
    """Advance solver.u_hat by dt in place."""
    u_hat, q, rhs = solver.u_hat, solver.q_buf, solver.rhs_buf
    E3 = _integrating_factors(solver, dt)
    q.zero_()
    for s in range(3):
        solver.rhs_(u_hat, rhs, record_stage0=(s == 0))
        if s == 0:
            q.copy_(rhs).mul_(dt)
        else:
            q.mul_(RK3_A[s]).add_(rhs, alpha=dt)
        E = E3[s]
        u_hat.add_(q, alpha=RK3_B[s]).mul_(E)
        q.mul_(E)


def rk4_step(solver, dt: float) -> None:
    """Classical RK4 with integrating factor (used as the convergence-order
    reference; allocates freely, not for production runs)."""
    u0 = solver.u_hat.clone()
    nu_k2 = solver.nu_k2
    Eh = torch.exp(nu_k2 * (-0.5 * dt))     # half-step decay
    rhs = solver.rhs_buf

    solver.rhs_(u0, rhs, record_stage0=True)
    N1 = rhs.clone()
    ua = Eh * (u0 + (0.5 * dt) * N1)
    solver.rhs_(ua, rhs)
    N2 = rhs.clone()
    ub = Eh * u0 + (0.5 * dt) * N2
    solver.rhs_(ub, rhs)
    N3 = rhs.clone()
    uc = Eh * (Eh * u0) + dt * (Eh * N3)
    solver.rhs_(uc, rhs)
    N4 = rhs
    solver.u_hat.copy_(
        Eh * (Eh * (u0 + (dt / 6.0) * N1) + (dt / 3.0) * (N2 + N3))
        + (dt / 6.0) * N4
    )


class CFLController:
    """dt = cfl * dx / max(|u|+|v|+|w|), capped at dt_max, with exact landing
    on requested sample times. dt is snapped DOWN onto a geometric grid
    (ratio 0.99) so consecutive steps reuse the cached integrating factors;
    snapping down can only make the step more conservative."""

    def __init__(self, cfl: float, dx: float, dt_max: float):
        self.cfl = cfl
        self.dx = dx
        self.dt_max = dt_max
        import math
        self._log_r = math.log(0.99)

    def __call__(self, umax: float, t: float, t_target: float | None = None) -> float:
        import math
        dt = self.dt_max if umax <= 0 else min(self.cfl * self.dx / umax, self.dt_max)
        if dt < self.dt_max:
            n = math.ceil(math.log(dt / self.dt_max) / self._log_r - 1e-12)
            dt = self.dt_max * (0.99 ** n)
        if t_target is not None and t + dt > t_target - 1e-12:
            dt = t_target - t
        return dt
