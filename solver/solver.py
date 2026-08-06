"""PseudoSpectralSolver: state, pre-allocated buffers, RHS, stepping.

Rotational-form incompressible NS, fully spectral (knowledge base 2.1):
    du/dt = u x omega - grad(P~) + nu lap(u) + f,   div u = 0
Nonlinear term is pseudo-spectral (9 FFTs per RK substage); Leray projection
removes the pressure; viscosity is exact via integrating factor (in
timestepping.py); 2/3-rule dealiasing.
"""
from __future__ import annotations

import torch

from .config import ExperimentConfig, SolverConfig
from .forcing import NoForcing, build_forcing
from .grids import SpectralGrid
from .initial_conditions import (abc_flow, random_solenoidal, taylor_green,
                                 vortex_tubes)
from .operators import cross_product, curl_hat, dealias_, leray_project_
from .timestepping import CFLController, rk3_williamson_step, rk4_step

_FFT_DIMS = (-3, -2, -1)


class PseudoSpectralSolver:
    def __init__(self, scfg: SolverConfig, u_hat0: torch.Tensor):
        self.cfg = scfg
        self.grid = grid = SpectralGrid(scfg.N, scfg.device, scfg.dtype,
                                        dealias_shape=scfg.dealias_shape)
        N, Nh = grid.N, grid.Nh

        self.u_hat = u_hat0.to(grid.device, grid.cdtype).clone()
        dealias_(self.u_hat, grid)

        cshape, rshape = (3, N, N, Nh), (3, N, N, N)
        self.q_buf = torch.zeros(cshape, dtype=grid.cdtype, device=grid.device)
        self.rhs_buf = torch.empty(cshape, dtype=grid.cdtype, device=grid.device)
        self.work_hat = torch.empty(cshape, dtype=grid.cdtype, device=grid.device)
        self.div_buf = torch.empty((N, N, Nh), dtype=grid.cdtype, device=grid.device)
        self.u_phys = torch.empty(rshape, dtype=grid.rdtype, device=grid.device)
        self.om_phys = torch.empty(rshape, dtype=grid.rdtype, device=grid.device)
        self.w_phys = torch.empty(rshape, dtype=grid.rdtype, device=grid.device)

        self.nu_k2 = (scfg.nu * grid.k2).to(grid.rdtype)
        self.forcing = build_forcing(grid, scfg.forcing)
        self.has_forcing = not isinstance(self.forcing, NoForcing)
        self.rhs_forcing = self.has_forcing and self.forcing.mode == "rhs"
        self.post_forcing = self.has_forcing and self.forcing.mode == "post"
        # stochastic forcing carries an internal state advanced once per step
        self.stateful_forcing = hasattr(self.forcing, "advance_ou")
        if self.rhs_forcing:
            self.f_buf = torch.empty(cshape, dtype=grid.cdtype, device=grid.device)

        # optional large-scale linear damping -alpha*u on |k| < ls_damp_kcut
        self.ls_damp_alpha = float(getattr(scfg, "ls_damp_alpha", 0.0))
        self.has_ls_damp = self.ls_damp_alpha > 0.0
        if self.has_ls_damp:
            kmag = grid.k2.sqrt()
            self._ls_damp_mask = ((kmag > 0) & (kmag < scfg.ls_damp_kcut))

        self.t = 0.0
        self.n_steps = 0
        self.last_eps_inj = 0.0
        self._stepper = {"rk3": rk3_williamson_step, "rk4": rk4_step}[scfg.scheme]
        self.cfl = CFLController(scfg.cfl, grid.dx, scfg.dt_max)

        # umax of the IC, for the first step's dt
        u0 = torch.fft.irfftn(self.u_hat, s=(N, N, N), dim=_FFT_DIMS)
        self.last_umax = float(u0.abs().sum(dim=0).max())

    # ------------------------------------------------------------------

    def rhs_(self, u_hat: torch.Tensor, out: torch.Tensor,
             record_stage0: bool = False) -> torch.Tensor:
        """out <- P(dealias(FFT(u x omega))) + f_hat.  9 FFTs."""
        grid = self.grid
        N = grid.N
        curl_hat(u_hat, grid, out=self.work_hat)
        torch.fft.irfftn(u_hat, s=(N, N, N), dim=_FFT_DIMS, out=self.u_phys)
        torch.fft.irfftn(self.work_hat, s=(N, N, N), dim=_FFT_DIMS, out=self.om_phys)
        if record_stage0:
            self.last_umax = float(self.u_phys.abs().sum(dim=0).max())
        cross_product(self.u_phys, self.om_phys, out=self.w_phys)
        torch.fft.rfftn(self.w_phys, dim=_FFT_DIMS, out=out)
        dealias_(out, grid)
        leray_project_(out, grid, work=self.div_buf)
        if self.rhs_forcing:
            self.forcing(u_hat, self.f_buf)
            if record_stage0:
                self.last_eps_inj = grid.injection_rate(u_hat, self.f_buf)
            out += self.f_buf
        if self.has_ls_damp:
            # -alpha*u on the lowest shells; u_hat is solenoidal so the term is
            # too (no re-projection needed). Drains energy from the largest scales.
            out[:, self._ls_damp_mask] -= self.ls_damp_alpha * u_hat[:, self._ls_damp_mask]
        return out

    def suggest_dt(self, t_target: float | None = None) -> float:
        return self.cfl(self.last_umax, self.t, t_target)

    def step(self, dt: float) -> None:
        self._stepper(self, dt)
        if self.post_forcing:
            self.last_eps_inj = self.forcing.post_step(self.u_hat, dt)
        if self.stateful_forcing:
            self.forcing.advance_ou(dt)
        self.t += dt
        self.n_steps += 1

    # ------------------------------------------------------------------

    def velocity_physical(self) -> torch.Tensor:
        """Fresh physical-space velocity [3, N, N, N] (allocates)."""
        N = self.grid.N
        return torch.fft.irfftn(self.u_hat, s=(N, N, N), dim=_FFT_DIMS)

    def scalars(self) -> dict:
        g = self.grid
        return {
            "t": self.t,
            "K": g.kinetic_energy(self.u_hat),
            "eps": g.dissipation(self.u_hat, self.cfg.nu),
            "eps_inj": self.last_eps_inj if self.has_forcing else 0.0,
            "umax": self.last_umax,
        }


def build_solver(cfg: ExperimentConfig) -> PseudoSpectralSolver:
    grid = SpectralGrid(cfg.solver.N, cfg.solver.device, cfg.solver.dtype)
    if cfg.ic == "taylor_green":
        u_hat0 = taylor_green(grid)
    elif cfg.ic == "random_solenoidal":
        u_hat0 = random_solenoidal(grid, seed=cfg.seed, k_p=cfg.k_p, u_rms=cfg.u_rms,
                                   spectrum_power=cfg.ic_spectrum_power)
    elif cfg.ic == "abc_flow":
        u_hat0 = abc_flow(grid, A=cfg.abc_A, B=cfg.abc_B, C=cfg.abc_C,
                          perturb=cfg.abc_perturb, seed=cfg.seed)
    elif cfg.ic == "vortex_tubes":
        u_hat0 = vortex_tubes(grid, sep=cfg.vt_sep, core=cfg.vt_core,
                              circ=cfg.vt_circ, perturb=cfg.vt_perturb, seed=cfg.seed)
    else:
        raise ValueError(f"unknown ic: {cfg.ic}")
    return PseudoSpectralSolver(cfg.solver, u_hat0)
