"""RotatingSolver — incompressible NS in a rotating frame (adds the Coriolis force).

Milestone 1 of the extended-physics phase. Subclasses the validated
PseudoSpectralSolver and adds ONE term to the momentum RHS:

    du/dt = u x omega - grad(p) + nu lap u + f  - 2 Omega x u
                                                  ^^^^^^^^^^^^ Coriolis (this class)

Physics facts:
  * Coriolis is LINEAR in u. It is NOT divergence-free in general: for constant
    Omega, div(Omega x u) = Omega . (curl u) = Omega . omega != 0. So the Coriolis
    term MUST be Leray-projected (the pressure adjusts to keep u solenoidal — the
    rotating-frame momentum eqn stays incompressible only after the pressure
    gradient absorbs the compressible part of -2 Omega x u). [Deep-review 2026-06-23
    caught the original "no projection needed" claim as a bug: div(u) grew 1e-15 ->
    ~1.5 in 10 steps without the projection. The inertial-oscillation test missed it
    because a uniform field has omega=0 so Omega.omega=0.] D4 (velocity-pressure
    consistency) is therefore unaffected only in the sense that the projected term is
    solenoidal; the pressure itself does gain a Coriolis contribution.
  * Coriolis couples velocity components (a full 3x3 per mode), so it CANNOT fold
    into the scalar viscous integrating factor — it stays explicit in the RHS, and
    because timestepping calls rhs_() every RK substage it enters all three stages.
  * Sign: the Coriolis acceleration is -2 Omega x u (rotating-frame momentum eqn).

Resolution criterion is UNCHANGED: rotation leaves the small scales ~isotropic, so
the Kolmogorov scale eta and the k_max*eta >= 1.5 (Class I) gate still apply.
"""
from __future__ import annotations

import torch

from solver.operators import leray_project_
from solver.solver import PseudoSpectralSolver

from .operators_ext import cross_product_spectral


class RotatingSolver(PseudoSpectralSolver):
    def __init__(self, scfg, u_hat0, rotation):
        super().__init__(scfg, u_hat0)
        self.rotation = rotation
        self.has_coriolis = bool(rotation.enabled) and rotation.omega_mag > 0.0
        # store as a small device tensor for clarity (scalars read out in the op)
        self.omega_vec = rotation.omega_vec
        # scratch for the Coriolis term (same shape/dtype as the RHS)
        if self.has_coriolis:
            self._coriolis_buf = torch.empty_like(self.rhs_buf)

    def rhs_(self, u_hat, out, record_stage0: bool = False):
        # base incompressible-NS RHS (nonlinear + forcing + ls_damp), already
        # Leray-projected and dealiased — solver/solver.py untouched.
        super().rhs_(u_hat, out, record_stage0=record_stage0)
        if self.has_coriolis:
            # -2 Omega x u, computed exactly in spectral space (Omega constant),
            # then Leray-projected (Omega x u is NOT divergence-free; the pressure
            # absorbs its compressible part to keep u solenoidal).
            cross_product_spectral(self.omega_vec, u_hat, out=self._coriolis_buf)
            self._coriolis_buf.mul_(-2.0)
            leray_project_(self._coriolis_buf, self.grid)
            out.add_(self._coriolis_buf)
        return out
