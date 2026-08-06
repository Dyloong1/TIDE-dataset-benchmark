"""Initial conditions.

Both ICs are generated in fp64 (random fields on CPU with a seeded Generator)
and cast afterwards, so fp32 and fp64 runs start from the byte-identical field
(acceptance criterion 5 requires the precision pair to share its IC exactly).
"""
from __future__ import annotations

import torch

from .grids import SpectralGrid
from .operators import dealias_, leray_project_


def taylor_green(grid: SpectralGrid) -> torch.Tensor:
    """u = (sin x cos y cos z, -cos x sin y cos z, 0). Returns u_hat [3,N,N,Nh].

    Standard TG vortex initial value (knowledge base 2.6); Re = 1/nu.
    """
    N = grid.N
    coord = torch.arange(N, dtype=torch.float64) * (2.0 * torch.pi / N)
    z = coord.view(N, 1, 1)
    y = coord.view(1, N, 1)
    x = coord.view(1, 1, N)
    u = torch.empty(3, N, N, N, dtype=torch.float64)
    u[0] = torch.sin(x) * torch.cos(y) * torch.cos(z)
    u[1] = -torch.cos(x) * torch.sin(y) * torch.cos(z)
    u[2] = 0.0
    u_hat = torch.fft.rfftn(u.to(grid.device), dim=(-3, -2, -1)).to(grid.cdtype)
    return dealias_(u_hat, grid)


def abc_flow(grid: SpectralGrid, A: float = 1.0, B: float = 1.0,
             C: float = 1.0, perturb: float = 0.0, seed: int = 0) -> torch.Tensor:
    """Arnold-Beltrami-Childress flow. Returns u_hat [3,N,N,Nh].

    u = (A sin z + C cos y,  B sin x + A cos z,  C sin y + B cos x)

    A maximally helical Beltrami field (u parallel to omega=curl u), an exact
    steady Euler solution (Dombre et al. JFM 1986). Single Fourier modes at |k|=1,
    analytically divergence-free.

    CRITICAL — perturb: pure ABC is an exact (unstable) NS eigenmode; with no
    perturbation it sits ON the fixed point and just decays laminarly (the first
    abc_re600 run barely moved, K dropped 6.5%, no cascade). To turbulize it you
    MUST seed its unstable manifold: perturb>0 adds a broadband solenoidal random
    field of amplitude `perturb`*u_rms (Podvigina-Pouquet 1994: 1:1:1 ABC is
    hydrodynamically unstable for Re=1/nu >~ 13; with Re~130 + a 5-10% seed it
    breaks down into maximally-helical decaying turbulence). Unequal A:B:C also
    helps break the special symmetries. The component layout matches the rfftn
    axis order dim=(-3,-2,-1)=(z,y,x).
    """
    N = grid.N
    coord = torch.arange(N, dtype=torch.float64) * (2.0 * torch.pi / N)
    z = coord.view(N, 1, 1)
    y = coord.view(1, N, 1)
    x = coord.view(1, 1, N)
    u = torch.empty(3, N, N, N, dtype=torch.float64)
    u[0] = A * torch.sin(z) + C * torch.cos(y)
    u[1] = B * torch.sin(x) + A * torch.cos(z)
    u[2] = C * torch.sin(y) + B * torch.cos(x)
    u_hat = torch.fft.rfftn(u.to(grid.device), dim=(-3, -2, -1)).to(grid.cdtype)
    dealias_(u_hat, grid)
    if perturb > 0.0:
        # broadband solenoidal seed to trigger the ABC instability (mandatory for
        # turbulization). Scale to perturb * u_rms of the ABC field.
        u_rms = (2.0 * grid.kinetic_energy(u_hat) / 3.0) ** 0.5
        pert = random_solenoidal(grid, seed=seed, k_p=4.0, u_rms=1.0, spectrum_power=2.0)
        p_rms = (2.0 * grid.kinetic_energy(pert) / 3.0) ** 0.5
        u_hat += pert * (perturb * u_rms / max(p_rms, 1e-30))
        dealias_(u_hat, grid)
        leray_project_(u_hat, grid)
    return u_hat


def vortex_tubes(grid: SpectralGrid, sep: float = torch.pi / 2,
                 core: float = 0.4, circ: float = 1.0,
                 perturb: float = 0.02, seed: int = 0) -> torch.Tensor:
    """Antiparallel Lamb-Oseen vortex tube pair (reconnection IC). Returns
    u_hat [3,N,N,Nh].

    Two straight tubes along z with opposite circulation, separated by `sep` in
    y, each a Gaussian-cored vortex (omega_z ~ exp(-r^2/core^2)) centred in x.
    A small sinusoidal-in-z displacement (`perturb`) seeds the long-wavelength
    instability that drives reconnection (Melander & Hussain 1989; Kerr 1993).

    Velocity from vorticity by spectral Biot-Savart: with omega = curl u and
    div u = 0, u_hat = i k x omega_hat / k^2. We build omega_z in physical space
    (the dominant component), FFT, then invert. dealias_ + leray_project_ are
    applied because the field is constructed numerically (not an exact mode).
    The reconnection event is anisotropic and transient -> evaluated on the
    decay/non-stationary path; its peak strain sets the smallest eta (watch the
    instantaneous-minimum k_max*eta for the resolution verdict).
    """
    N, Nh = grid.N, grid.Nh
    coord = torch.arange(N, dtype=torch.float64) * (2.0 * torch.pi / N)
    z = coord.view(N, 1, 1)
    y = coord.view(1, N, 1)
    x = coord.view(1, 1, N)
    cx = torch.pi                       # tubes centred at x = pi
    cy1, cy2 = torch.pi - sep / 2, torch.pi + sep / 2
    # small z-dependent wiggle in x to seed the reconnection instability
    gen = torch.Generator(device="cpu").manual_seed(seed)
    phase = float(torch.rand(1, generator=gen)) * 2.0 * torch.pi
    dx = perturb * torch.sin(z + phase)
    # periodic Gaussian distance uses the nearest-image via sin of half-angle
    def gauss(xc, yc):
        rx = 2.0 * torch.sin(0.5 * (x - xc - dx))
        ry = 2.0 * torch.sin(0.5 * (y - yc))
        return torch.exp(-(rx**2 + ry**2) / core**2)
    omega_z = circ * (gauss(cx, cy1) - gauss(cx, cy2))   # antiparallel
    omega = torch.zeros(3, N, N, N, dtype=torch.float64)
    omega[2] = omega_z
    omega_hat = torch.fft.rfftn(omega.to(grid.device), dim=(-3, -2, -1)).to(grid.cdtype)
    # Biot-Savart: u_hat = i k x omega_hat / k^2
    kx, ky, kz, inv_k2 = grid.kx, grid.ky, grid.kz, grid.inv_k2
    u_hat = torch.empty_like(omega_hat)
    u_hat[0] = 1j * (ky * omega_hat[2] - kz * omega_hat[1]) * inv_k2
    u_hat[1] = 1j * (kz * omega_hat[0] - kx * omega_hat[2]) * inv_k2
    u_hat[2] = 1j * (kx * omega_hat[1] - ky * omega_hat[0]) * inv_k2
    dealias_(u_hat, grid)
    leray_project_(u_hat, grid)
    return u_hat.to(grid.cdtype)


def random_solenoidal(grid: SpectralGrid, seed: int, k_p: float = 3.0,
                      u_rms: float = 0.7, spectrum_power: float = 4.0) -> torch.Tensor:
    """Random divergence-free field with shell spectrum E(k) ~ k^p exp(-2(k/k_p)^2),
    scaled to the target per-component rms (K = 3/2 u_rms^2).

    Shaping: start from white Gaussian noise (correct Hermitian symmetry comes
    free from rfftn of a real field), measure its shell energies, multiply each
    mode by sqrt(E_target/E_white) of its shell, project solenoidal, rescale.

    spectrum_power p sets the low-k spectral slope of the initial condition.
    p=4 (default, Batchelor turbulence) and p=2 (Saffman) give measurably
    different free-decay exponents K(t)~t^-alpha (the classic Saffman/Batchelor
    distinction); the slope is irrelevant once forced (the forcing overwrites the
    large scales), so this knob only matters for decaying runs.
    """
    N, Nh = grid.N, grid.Nh
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(3, N, N, N, generator=gen, dtype=torch.float64)
    u_hat = torch.fft.rfftn(noise.to(grid.device), dim=(-3, -2, -1))

    dealias_(u_hat, grid)
    leray_project_(u_hat, grid)

    # Measured shell energies of the noise field (fp64).
    e_flat = (u_hat.real**2 + u_hat.imag**2).sum(dim=0).flatten()
    e_flat = 0.5 * grid._w64_flat * e_flat / float(grid.n_total) ** 2
    E_white = torch.bincount(grid.shell_index, weights=e_flat, minlength=grid.n_shells)

    shells = torch.arange(grid.n_shells, dtype=torch.float64, device=grid.device)
    E_target = shells**spectrum_power * torch.exp(-2.0 * (shells / k_p) ** 2)
    E_target[0] = 0.0
    gain = torch.where(E_white > 0, (E_target / E_white.clamp(min=1e-300)).sqrt(),
                       torch.zeros_like(E_white))
    gain[0] = 0.0
    u_hat *= gain[grid.shell_index].view(N, N, Nh)

    # Rescale total energy to K = 3/2 u_rms^2 (projection already applied,
    # so this scaling is exact).
    K_now = grid.kinetic_energy(u_hat)
    K_target = 1.5 * u_rms**2
    u_hat *= (K_target / K_now) ** 0.5
    return u_hat.to(grid.cdtype)
