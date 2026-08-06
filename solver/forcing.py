"""Forcing schemes (knowledge base 2.2). Phase 0 only needs the
negative-damping band forcing; the other two arrive with the v0 corpus.
"""
from __future__ import annotations

import torch

from .grids import SpectralGrid


class NoForcing:
    mode = "rhs"

    def __call__(self, u_hat: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        out.zero_()
        return out


class NegativeDampingBandForcing:
    """f_hat = (eps_w / 2 E_f) * u_hat restricted to 0 < |k| < k_f.

    E_f is the kinetic energy inside the band, so the instantaneous power
    input sum_k w Re(f.u*)/N^6 equals eps_w exactly; at statistical
    stationarity the dissipation rate equals eps_w (knowledge base 2.2.1).
    Caveat observed in phase 0: with k_f=2 the box-scale modes wander on
    O(10 T_L) timescales with large amplitude (K excursions of tens of %).
    """

    mode = "rhs"

    def __init__(self, grid: SpectralGrid, k_f: float = 2.0, eps_w: float = 0.1):
        self.grid = grid
        self.eps_w = eps_w
        k_mag = grid.k2.sqrt()
        self.band_mask = ((k_mag > 0) & (k_mag < k_f))
        # fp64 weights restricted to the band, for the E_f reduction
        self._wband64 = (grid._w64_flat.view_as(grid.k2)
                         * self.band_mask.to(torch.float64))

    def band_energy(self, u_hat: torch.Tensor) -> float:
        e = (u_hat.real.double() ** 2 + u_hat.imag.double() ** 2).sum(dim=0)
        e = 0.5 * self._wband64 * e / float(self.grid.n_total) ** 2
        return float(e.sum())

    def __call__(self, u_hat: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        E_f = self.band_energy(u_hat)
        coeff = self.eps_w / (2.0 * max(E_f, 1e-30))
        torch.mul(u_hat, self.band_mask, out=out)
        out *= coeff
        return out


class EnergyPreservingForcing:
    """Deterministic band-energy rescaling (knowledge base 2.2.2; the JHTDB
    isotropic1024 forcing): after each full time step, rescale the modes with
    0 < |k| <= k_f so the band kinetic energy returns to the fixed value E_f0.
    Pins the large scales, eliminating the slow box-scale wandering of the
    negative-damping scheme; the injection rate emerges as a diagnostic
    eps_inj = (E_f0 - E_f_before)/dt.
    """

    mode = "post"

    def __init__(self, grid: SpectralGrid, k_f: float = 2.0, E_f0: float = 0.4):
        self.grid = grid
        self.E_f0 = E_f0
        k_mag = grid.k2.sqrt()
        self.band_mask = ((k_mag > 0) & (k_mag <= k_f))
        self._wband64 = (grid._w64_flat.view_as(grid.k2)
                         * self.band_mask.to(torch.float64))

    def band_energy(self, u_hat: torch.Tensor) -> float:
        e = (u_hat.real.double() ** 2 + u_hat.imag.double() ** 2).sum(dim=0)
        e = 0.5 * self._wband64 * e / float(self.grid.n_total) ** 2
        return float(e.sum())

    def post_step(self, u_hat: torch.Tensor, dt: float) -> float:
        # Re-project before rescaling. The rescale gain is systematically > 1
        # (it restores the energy the cascade drains from the band), and the
        # band has no viscous decay to speak of, so any non-solenoidal
        # round-off residual in the forced shells would otherwise be amplified
        # exponentially (~e^{eps_inj t / 2 E_f0}); in fp32 this grows to O(10%)
        # of the band amplitude within ~30k steps. Caught by acceptance check
        # A12 (incompressibility) of the DNS standard table.
        from .operators import leray_project_
        leray_project_(u_hat, self.grid)
        E_f = self.band_energy(u_hat)
        gain = (self.E_f0 / max(E_f, 1e-30)) ** 0.5
        u_hat[:, self.band_mask] *= gain
        return (self.E_f0 - E_f) / dt    # eps_inj


class FixedPowerBandForcing:
    """Production forcing (phase-1): exact post-step injection of eps_w * dt
    into the band 0 < |k| < k_f. Discrete-exact realization of the
    negative-damping fixed-power scheme f = (eps_w / 2 E_f) u (KB 2.2.1):
    stationary dissipation == eps_w by construction, so eta and k_max*eta are
    locked before the run starts.

    Robustness by construction against both phase-0 incidents:
      - re-projects the state first (the band gain > 1 would otherwise
        exponentially amplify round-off divergence; cf. A12 incident);
      - cannot sustain a laminar absorbing state: if the cascade dies, band
        energy grows until nonlinearity re-ignites turbulence (unlike the
        energy-preserving rescale, which maintains the dead state forever).
    """

    mode = "post"

    def __init__(self, grid: SpectralGrid, k_f: float = 2.0, eps_w: float = 0.096,
                 k_f_low: float = 0.0):
        self.grid = grid
        self.eps_w = eps_w
        k_mag = grid.k2.sqrt()
        # annular band k_f_low <= |k| < k_f. k_f_low=0 -> the usual 0 < |k| < k_f.
        # Excluding the lowest shells (k_f_low>0) means the largest box scales are
        # never pumped, removing the source of the k=1 runaway at its origin.
        self.band_mask = ((k_mag >= max(k_f_low, 1e-9)) & (k_mag < k_f) & (k_mag > 0))
        self._wband64 = (grid._w64_flat.view_as(grid.k2)
                         * self.band_mask.to(torch.float64))

    def band_energy(self, u_hat: torch.Tensor) -> float:
        e = (u_hat.real.double() ** 2 + u_hat.imag.double() ** 2).sum(dim=0)
        e = 0.5 * self._wband64 * e / float(self.grid.n_total) ** 2
        return float(e.sum())

    def post_step(self, u_hat: torch.Tensor, dt: float) -> float:
        from .operators import leray_project_
        leray_project_(u_hat, self.grid)
        E_f = self.band_energy(u_hat)
        gain = ((E_f + self.eps_w * dt) / max(E_f, 1e-30)) ** 0.5
        u_hat[:, self.band_mask] *= gain
        return self.eps_w          # exact by construction


class FixedPowerBandForcingDamped(FixedPowerBandForcing):
    """fixed_power injection + a soft cap on the lowest shells (|k| < k_damp).

    Diagnosis (phase-0, k_f=4 long run): deterministic band forcing destabilizes
    not by laminar collapse but by RUNAWAY accumulation in the k=1 shell. k=1 is
    the largest scale in the box, so the cascade cannot drain it; the fixed-power
    gain (>1 every step) then compounds its energy until ~100% of the energy
    piles into k<4 and the cascade dies (K blows up, eps drops). See
    turbgen-kf4-acceptance-risk-audit.

    Fix: after the normal fixed_power injection, clamp the energy of the lowest
    shells to a ceiling E_cap = e_cap_factor * E_low0, where E_low0 is the energy
    of those shells in the (healthy) initial field. The clamp scales only the
    EXCESS back, so:
      - in the normal stationary state the low shells sit below the ceiling and
        the clamp is a no-op -> steady eps == eps_w is preserved (k_max*eta lock
        intact, still Class I);
      - if k=1 starts to run away, the ceiling bleeds the excess and breaks the
        positive-feedback loop.
    The bled energy is reported so the energy budget stays auditable.
    """

    mode = "post"

    def __init__(self, grid: SpectralGrid, k_f: float = 4.0, eps_w: float = 0.096,
                 k_damp: float = 2.0, e_cap_factor: float = 1.5):
        super().__init__(grid, k_f=k_f, eps_w=eps_w)
        k_mag = grid.k2.sqrt()
        self.damp_mask = ((k_mag > 0) & (k_mag < k_damp))
        self._wdamp64 = (grid._w64_flat.view_as(grid.k2)
                         * self.damp_mask.to(torch.float64))
        self.e_cap_factor = e_cap_factor
        self._E_cap = None          # set lazily from the first state seen
        self.last_bled = 0.0

    def _damp_energy(self, u_hat: torch.Tensor) -> float:
        e = (u_hat.real.double() ** 2 + u_hat.imag.double() ** 2).sum(dim=0)
        e = 0.5 * self._wdamp64 * e / float(self.grid.n_total) ** 2
        return float(e.sum())

    def set_cap_from_state(self, u_hat: torch.Tensor) -> None:
        """Pin the ceiling to e_cap_factor x the current low-shell energy.
        Call once on the healthy initial/warm-start field."""
        self._E_cap = self.e_cap_factor * max(self._damp_energy(u_hat), 1e-30)

    def post_step(self, u_hat: torch.Tensor, dt: float) -> float:
        # normal fixed_power injection (also re-projects); steady eps == eps_w
        eps = super().post_step(u_hat, dt)
        if self._E_cap is None:
            self.set_cap_from_state(u_hat)
        E_low = self._damp_energy(u_hat)
        if E_low > self._E_cap:
            gain = (self._E_cap / E_low) ** 0.5
            u_hat[:, self.damp_mask] *= gain
            self.last_bled = (E_low - self._E_cap) / dt
        else:
            self.last_bled = 0.0
        return eps


class StochasticOUForcing:
    """Eswaran & Pope (1988) stochastic forcing of the low-wavenumber band.

    Each forced Fourier mode is driven by an independent complex
    Ornstein-Uhlenbeck process b(k, t):
        db = -b/T_F dt + sqrt(2 sigma^2 / T_F) dW   (per real/imag part)
    integrated exactly over a step:
        b <- b e^{-dt/T_F} + sqrt(sigma^2 (1 - e^{-2 dt/T_F})) xi,  xi ~ N(0,1).
    The force is f_hat = P(b) restricted to 0 < |k| < k_f, Leray-projected so it
    injects only solenoidal energy. The random phases destroy the deterministic
    attractor that lets band-pinned forcing relaminarize (phase-0 finding): the
    turbulent state is statistically stationary with no absorbing laminar fixed
    point.

    Hermitian symmetry: the OU increment is generated as rfftn of a real
    Gaussian field (same trick as initial_conditions.random_solenoidal), so the
    forced velocity stays real. The OU state b_hat inherits Hermitian symmetry
    from its (Hermitian) increments and (real) decay factor, and is preserved
    exactly. f_hat is frozen across the RK substages of one step and the OU
    state is advanced once per completed step (mode='rhs' + advance hook).

    The mean injected power is a diagnostic (eps_inj = <Re(f . u*)>), not pinned;
    sigma^2 and T_F set its scale. Re_lambda/eta therefore float and are
    measured, then nu is recalibrated to hit the target band (same loop as the
    other schemes).
    """

    mode = "rhs"

    def __init__(self, grid: SpectralGrid, k_f: float = 2.0,
                 tau: float = 1.0, sigma2: float = 0.05, seed: int = 1234):
        self.grid = grid
        self.tau = tau
        self.sigma2 = sigma2
        k_mag = grid.k2.sqrt()
        self.band_mask = ((k_mag > 0) & (k_mag < k_f))     # bool [N,N,Nh]
        self.n_band = int(self.band_mask.sum())
        # OU state: complex spectral field, Hermitian, band-limited, solenoidal.
        # Generator lives on the compute device: generating + transferring a
        # 3 x N^3 fp64 random field from the CPU every step costs ~600 ms at
        # N=256 (8x the whole solver step); on-GPU generation is ~1 ms. The
        # forcing process needs run-to-run reproducibility (same seed -> same
        # sequence on the same device), not cross-device byte-identity (unlike
        # the IC, which fp32/fp64 pairs must share).
        self._gen = torch.Generator(device=grid.device).manual_seed(seed)
        self.b_hat = torch.zeros(3, grid.N, grid.N, grid.Nh,
                                 dtype=grid.cdtype, device=grid.device)
        self._project_band(self.b_hat)

    def _project_band(self, f_hat: torch.Tensor) -> None:
        from .operators import dealias_, leray_project_
        f_hat *= self.band_mask
        leray_project_(f_hat, self.grid)
        dealias_(f_hat, self.grid)
        f_hat *= self.band_mask

    def _phys_var(self, f_hat: torch.Tensor) -> float:
        """Physical-space variance sum_c <f_c^2> via Parseval (same convention
        as grid.kinetic_energy: divide by N^6, rfft weights for the half plane)."""
        e = (f_hat.real.double() ** 2 + f_hat.imag.double() ** 2).sum(dim=0).flatten()
        return float((self.grid._w64_flat * e).sum() / float(self.grid.n_total) ** 2)

    def _hermitian_increment(self) -> torch.Tensor:
        """A fresh band-limited solenoidal Gaussian increment with Hermitian
        symmetry (real physical field -> rfftn), normalized to UNIT physical-
        space variance (sum_c <inc_c^2> = 1). The caller then scales by
        sqrt(sigma2 (1 - e^{-2 dt/tau})), so sigma2 is the stationary physical
        variance of the forcing acceleration (units of (du/dt)^2), N-independent.
        """
        N = self.grid.N
        noise = torch.randn(3, N, N, N, generator=self._gen, dtype=self.grid.rdtype,
                            device=self.grid.device)
        inc = torch.fft.rfftn(noise, dim=(-3, -2, -1)).to(self.grid.cdtype)
        self._project_band(inc)
        scale = (1.0 / max(self._phys_var(inc), 1e-30)) ** 0.5
        inc *= scale
        return inc

    def __call__(self, u_hat: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        out.copy_(self.b_hat)
        return out

    # Frozen-force mode: when set, advance_ou is a no-op so b_hat stays constant.
    # Used by the D3 half-dt convergence probe (run_trajectory_showcase): OU
    # reseeds a fresh random increment every step, so an h vs h/2 probe would
    # otherwise drive the two trajectories with *different* noise realizations,
    # and the residual difference would be dominated by that O(1) noise mismatch
    # rather than the O(h^2) time-truncation D3 is meant to measure. Freezing the
    # force to a fixed spectral field makes the force identical at both step
    # sizes, so the probe cleanly isolates the RK3+central-difference truncation
    # of the velocity integration (which IS O(h^2)). The OU process's own
    # evolution is validated separately by the run's stationarity (A7) and by D1
    # (the frozen force is still band-limited, solenoidal, Hermitian).
    _freeze = False

    def advance_ou(self, dt: float) -> None:
        """Advance the OU state one step (called once per completed time step)."""
        if self._freeze:
            return
        import math
        decay = math.exp(-dt / self.tau)
        diff = (self.sigma2 * (1.0 - decay * decay)) ** 0.5
        self.b_hat *= decay
        self.b_hat += diff * self._hermitian_increment()
        self._project_band(self.b_hat)


class HelicalOUForcing(StochasticOUForcing):
    """Stochastic OU forcing projected onto ONE helicity sign (default +).

    Identical to StochasticOUForcing except the forcing field is additionally
    projected onto the positive- (or negative-) helicity helical modes. This
    injects net helicity <f . curl f> != 0, so the forced turbulence carries net
    helicity <u . omega> != 0 and breaks mirror (reflection) symmetry — a flow
    statistically DISTINCT from the mirror-symmetric OU/HIT case (different
    energy transfer, helicity cascade, coherent structures). It is the third
    forcing axis in the roadmap.

    Implementation: helical-project ONLY the OU increment (in _hermitian_increment).
    Since the OU state b_hat is a linear combination of single-sign helical
    increments, it stays single-sign helical without re-projecting — so the
    per-step _project_band (band+Leray+dealias cleanup) needs NO extra helical
    projection. This avoids a redundant curl per step (~2x speedup of advance_ou
    vs projecting in _project_band). Everything else (unit-variance normalization,
    exact OU step, _freeze for D3, mode='rhs') is inherited unchanged. NOTE:
    helical projection keeps ~half the energy, so for a given sigma2 the injected
    power differs from plain OU — sigma2 is recalibrated per the k_max*eta loop.
    """

    def __init__(self, grid: SpectralGrid, k_f: float = 2.0, tau: float = 1.0,
                 sigma2: float = 0.05, seed: int = 1234, helicity_sign: int = +1):
        self.helicity_sign = int(helicity_sign)
        super().__init__(grid, k_f=k_f, tau=tau, sigma2=sigma2, seed=seed)

    def _hermitian_increment(self) -> torch.Tensor:
        from .operators import helical_project
        # base increment: band-limited, solenoidal, Hermitian (NOT yet normalized)
        N = self.grid.N
        noise = torch.randn(3, N, N, N, generator=self._gen, dtype=self.grid.rdtype,
                            device=self.grid.device)
        inc = torch.fft.rfftn(noise, dim=(-3, -2, -1)).to(self.grid.cdtype)
        self._project_band(inc)                                   # band+Leray+dealias
        inc = helical_project(inc, self.grid, sign=self.helicity_sign)
        inc *= self.band_mask                                     # keep strictly band-limited
        # normalize to unit physical variance AFTER helical projection, so sigma2
        # is the true stationary variance of the (helical) forcing acceleration.
        scale = (1.0 / max(self._phys_var(inc), 1e-30)) ** 0.5
        inc *= scale
        return inc


def build_forcing(grid: SpectralGrid, cfg):
    if cfg.type == "none":
        return NoForcing()
    if cfg.type == "negative_damping":
        return NegativeDampingBandForcing(grid, k_f=cfg.k_f, eps_w=cfg.eps_w)
    if cfg.type == "energy_preserving":
        return EnergyPreservingForcing(grid, k_f=cfg.k_f, E_f0=cfg.E_f0)
    if cfg.type == "fixed_power":
        return FixedPowerBandForcing(grid, k_f=cfg.k_f, eps_w=cfg.eps_w,
                                     k_f_low=cfg.k_f_low)
    if cfg.type == "fixed_power_damped":
        return FixedPowerBandForcingDamped(grid, k_f=cfg.k_f, eps_w=cfg.eps_w,
                                           k_damp=cfg.k_damp,
                                           e_cap_factor=cfg.e_cap_factor)
    if cfg.type == "stochastic_ou":
        return StochasticOUForcing(grid, k_f=cfg.k_f, tau=cfg.ou_tau,
                                   sigma2=cfg.ou_sigma2, seed=cfg.ou_seed)
    if cfg.type == "helical_ou":
        return HelicalOUForcing(grid, k_f=cfg.k_f, tau=cfg.ou_tau,
                                sigma2=cfg.ou_sigma2, seed=cfg.ou_seed,
                                helicity_sign=getattr(cfg, "helicity_sign", 1))
    raise ValueError(f"unknown forcing type: {cfg.type}")
