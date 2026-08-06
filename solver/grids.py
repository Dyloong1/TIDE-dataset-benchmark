"""SpectralGrid: every wavenumber-space cache the solver needs, built once.

Conventions (identical to NeurIPS/ablations/*/code/physics_metrics.py):
  - Physical fields are [3, N, N, N] with spatial axes (z, y, x) and velocity
    channels (0=u_x, 1=u_y, 2=u_z); periodic domain (2pi)^3.
  - Spectral fields are rfftn over the last 3 dims: [3, N, N, N//2 + 1] with
    the half axis along x. Wavenumbers are integer modes.
  - Shell-averaged spectrum: shell kappa collects |k| in [kappa-0.5, kappa+0.5),
    reported for kappa = 1..N/2; sum_k E(k) == 0.5 <|u|^2> (Parseval).
"""
from __future__ import annotations

import torch

DTYPES = {"fp32": torch.float32, "fp64": torch.float64}
CDTYPES = {"fp32": torch.complex64, "fp64": torch.complex128}


class SpectralGrid:
    def __init__(self, N: int, device: str = "cuda", dtype: str = "fp32",
                 dealias_shape: str = "cubic"):
        self.N = N
        self.Nh = N // 2 + 1
        self.device = torch.device(device)
        self.rdtype = DTYPES[dtype]
        self.cdtype = CDTYPES[dtype]
        self.n_total = N ** 3
        self.dx = 2.0 * torch.pi / N

        # Integer wavenumbers, built in fp64 then cast (identical fp32/fp64 grids).
        kz = torch.fft.fftfreq(N, d=1.0 / N, dtype=torch.float64)
        ky = torch.fft.fftfreq(N, d=1.0 / N, dtype=torch.float64)
        kx = torch.fft.rfftfreq(N, d=1.0 / N, dtype=torch.float64)
        self.kz = kz.to(self.device, self.rdtype).view(N, 1, 1)
        self.ky = ky.to(self.device, self.rdtype).view(1, N, 1)
        self.kx = kx.to(self.device, self.rdtype).view(1, 1, self.Nh)

        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
        k2 = (KX**2 + KY**2 + KZ**2)
        self.k2 = k2.to(self.device, self.rdtype)                      # [N,N,Nh]
        inv_k2 = torch.where(k2 > 0, 1.0 / k2, torch.zeros_like(k2))
        self.inv_k2 = inv_k2.to(self.device, self.rdtype)
        # |k| and 1/|k| (=0 at k=0), cached for the hot path (helical projection).
        kmag = k2.sqrt()
        self.k_mag = kmag.to(self.device, self.rdtype)
        self.inv_kmag = torch.where(k2 > 0, 1.0 / kmag,
                                    torch.zeros_like(kmag)).to(self.device, self.rdtype)

        # 2/3-rule dealias mask. "cubic": keep |k_i| <= N/3 per direction
        # (alias-free, but retains weakly-damped anisotropic corner modes with
        # |k| up to sqrt(3) N/3). "spherical": additionally truncate |k| > N/3
        # (isotropic; removes the corner reservoir implicated in the fp32
        # near-cut pile-up instability).
        k_cut = N / 3.0
        mask = (KX.abs() <= k_cut) & (KY.abs() <= k_cut) & (KZ.abs() <= k_cut)
        if dealias_shape == "spherical":
            mask &= (k2 <= k_cut**2)
        elif dealias_shape != "cubic":
            raise ValueError(f"unknown dealias_shape: {dealias_shape}")
        self.dealias_mask = mask.to(self.device)                       # bool [N,N,Nh]
        self.k_max_resolved = int(k_cut)                               # N=256 -> 85

        # rfft compensation weights: interior kx modes count twice
        # (kx = 0 and kx = Nyquist appear once in the half-spectrum).
        w = torch.full((N, N, self.Nh), 2.0, dtype=torch.float64)
        w[:, :, 0] = 1.0
        if N % 2 == 0:
            w[:, :, -1] = 1.0
        self.rfft_weight = w.to(self.device, self.rdtype)

        # Shell index for bincount spectra: shell = floor(|k| + 0.5).
        k_mag = k2.sqrt()
        shell = torch.floor(k_mag + 0.5).to(torch.int64)
        self.n_shells = int(shell.max().item()) + 1
        self.shell_index = shell.to(self.device).flatten()
        # weights folded in fp64 for accurate reductions regardless of state dtype
        self._w64_flat = w.to(self.device, torch.float64).flatten()
        self._k2_64_flat = k2.to(self.device, torch.float64).flatten()

    # ------------------------------------------------------------------
    # Reductions (all accumulate in fp64; inputs may be fp32 spectral states)
    # ------------------------------------------------------------------

    def _energy_density_flat(self, u_hat: torch.Tensor) -> torch.Tensor:
        """0.5 * w * sum_c |u_hat_c|^2 / N^6, flattened, fp64. Sums to K."""
        e = (u_hat.real.double() ** 2 + u_hat.imag.double() ** 2).sum(dim=0).flatten()
        return 0.5 * self._w64_flat * e / float(self.n_total) ** 2

    def kinetic_energy(self, u_hat: torch.Tensor) -> float:
        """K = 0.5 <|u|^2>  (per unit volume)."""
        return float(self._energy_density_flat(u_hat).sum())

    def dissipation(self, u_hat: torch.Tensor, nu: float) -> float:
        """eps = 2 nu sum_k k^2 E_density(k), exact k^2 (no shell binning)."""
        e = self._energy_density_flat(u_hat)
        return float(2.0 * nu * (self._k2_64_flat * e).sum())

    def shell_spectrum(self, u_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shell-summed E(k) for k = 1..N/2. Returns (k, E_k) on CPU, fp64."""
        e = self._energy_density_flat(u_hat)
        E = torch.bincount(self.shell_index, weights=e, minlength=self.n_shells)
        k_max = self.N // 2
        k = torch.arange(1, k_max + 1, dtype=torch.float64)
        return k, E[1 : k_max + 1].cpu()

    def component_moments(self, u_hat: torch.Tensor) -> tuple[list, list]:
        """Time-series ingredients for the isotropy checks (DNS standard A10):
        per-component <u_i^2> and cross <u_i u_j>, via Parseval."""
        w = self._w64_flat.view_as(self.k2)
        n2 = float(self.n_total) ** 2
        re, im = u_hat.real.double(), u_hat.imag.double()
        ui2 = [float((w * (re[i] ** 2 + im[i] ** 2)).sum() / n2) for i in range(3)]
        uij = [float((w * (re[i] * re[j] + im[i] * im[j])).sum() / n2)
               for i, j in ((0, 1), (0, 2), (1, 2))]
        return ui2, uij

    def injection_rate(self, u_hat: torch.Tensor, f_hat: torch.Tensor) -> float:
        """eps_inj = sum_k w * Re(f_hat . conj(u_hat)) / N^6."""
        prod = (f_hat.real.double() * u_hat.real.double()
                + f_hat.imag.double() * u_hat.imag.double()).sum(dim=0).flatten()
        return float((self._w64_flat * prod).sum() / float(self.n_total) ** 2)
