"""SpectralGrid reductions: Parseval, spectrum cross-check vs the published
physics_metrics reference implementation, dissipation consistency."""
import pytest
import torch

from solver.grids import SpectralGrid


def _random_field(N, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(3, N, N, N, generator=gen, dtype=torch.float64)


def test_parseval_kinetic_energy(device):
    N = 32
    grid = SpectralGrid(N, device, "fp64")
    u = _random_field(N).to(device)
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1))
    K_spec = grid.kinetic_energy(u_hat)
    K_phys = float(0.5 * (u**2).sum(dim=0).mean())
    assert abs(K_spec - K_phys) / K_phys < 1e-12


def test_shell_spectrum_matches_reference(device):
    """bincount spectrum == physics_metrics.energy_spectrum (loop reference).

    physics_metrics is the external NeurIPS reference impl; its path is provided
    per-machine via PHYSICS_METRICS_DIR (see conftest.py). Skip when unavailable
    so the suite stays green on machines that don't have the NeurIPS repo checked
    out, rather than hard-failing on an environment-specific dependency."""
    pm = pytest.importorskip(
        "physics_metrics",
        reason="set PHYSICS_METRICS_DIR to the NeurIPS code dir to enable this cross-check",
    )

    N = 32
    grid = SpectralGrid(N, device, "fp64")
    u = _random_field(N, seed=1).to(device)
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1))

    k_ours, E_ours = grid.shell_spectrum(u_hat)
    k_ref, E_ref = pm.energy_spectrum(u[None])  # [B=1, ...]
    E_ref = E_ref[0].cpu().double()

    assert k_ours.shape[0] == k_ref.shape[0] == N // 2
    # reference accumulates in fp32 (torch.zeros default), so compare at fp32 level
    assert torch.allclose(E_ours, E_ref, rtol=1e-6, atol=1e-9)


def test_spectrum_sums_to_energy(device):
    """Shell spectrum is reported up to k = N/2 (sphere), while dealiasing is
    per-direction (cube |k_i| <= N/3, corners reach |k| ~ 0.58 N). Use an IC
    whose energy is concentrated at low k so the corner tail is negligible —
    which is also the physically relevant regime."""
    from solver.initial_conditions import random_solenoidal

    N = 32
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=2, k_p=3.0)
    _, E_k = grid.shell_spectrum(u_hat)
    K = grid.kinetic_energy(u_hat)
    assert abs(float(E_k.sum()) - K) / K < 1e-10


def test_dissipation_exact_vs_definition(device):
    """eps = nu <|grad u|^2> for solenoidal dealiased fields == 2 nu sum k^2 E.
    (Dealias first: Nyquist modes break Hermitian symmetry under i*k
    differentiation, and the solver state is always dealiased anyway.)"""
    from solver.operators import dealias_, leray_project_

    N = 32
    grid = SpectralGrid(N, device, "fp64")
    u = _random_field(N, seed=3).to(device)
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1))
    dealias_(u_hat, grid)
    leray_project_(u_hat, grid)
    nu = 1.7e-3
    eps_spec = grid.dissipation(u_hat, nu)

    # physical-space |grad u|^2 via spectral derivatives
    ks = (grid.kx, grid.ky, grid.kz)
    g2 = 0.0
    for i in range(3):
        for j in range(3):
            d_hat = (1j * ks[j]) * u_hat[i]
            d = torch.fft.irfftn(d_hat, s=(N, N, N), dim=(-3, -2, -1))
            g2 += float((d**2).mean())
    assert abs(eps_spec - nu * g2) / (nu * g2) < 1e-10
