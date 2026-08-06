"""Spectral operators: projection annihilates divergence, curl matches
analytic fields, dealias truncates the right modes."""
import torch

from solver.grids import SpectralGrid
from solver.operators import curl_hat, dealias_, leray_project_


def _coords(N, device):
    c = torch.arange(N, dtype=torch.float64, device=device) * (2 * torch.pi / N)
    return c.view(N, 1, 1), c.view(1, N, 1), c.view(1, 1, N)  # z, y, x


def test_projection_kills_divergence(device):
    N = 32
    grid = SpectralGrid(N, device, "fp64")
    gen = torch.Generator().manual_seed(0)
    u = torch.randn(3, N, N, N, generator=gen, dtype=torch.float64).to(device)
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1))
    leray_project_(u_hat, grid)
    div_hat = (grid.kx * u_hat[0] + grid.ky * u_hat[1] + grid.kz * u_hat[2])
    scale = u_hat.abs().max()
    assert float(div_hat.abs().max()) / float(scale) < 1e-13


def test_projection_idempotent(device):
    N = 32
    grid = SpectralGrid(N, device, "fp64")
    gen = torch.Generator().manual_seed(1)
    u = torch.randn(3, N, N, N, generator=gen, dtype=torch.float64).to(device)
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1))
    leray_project_(u_hat, grid)
    once = u_hat.clone()
    leray_project_(u_hat, grid)
    assert torch.allclose(once, u_hat, rtol=1e-13, atol=1e-13)


def test_curl_analytic(device):
    """u = (sin y, 0, 0) -> omega = (0, 0, -cos y)."""
    N = 32
    grid = SpectralGrid(N, device, "fp64")
    z, y, x = _coords(N, device)
    u = torch.zeros(3, N, N, N, dtype=torch.float64, device=device)
    u[0] = torch.sin(y).expand(N, N, N)
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1))
    om_hat = curl_hat(u_hat, grid)
    om = torch.fft.irfftn(om_hat, s=(N, N, N), dim=(-3, -2, -1))
    expected_z = -torch.cos(y).expand(N, N, N)
    assert float((om[0]).abs().max()) < 1e-12
    assert float((om[1]).abs().max()) < 1e-12
    assert float((om[2] - expected_z).abs().max()) < 1e-12


def test_curl_taylor_green(device):
    """TG field: omega_z = +2 sin x sin y cos z (analytic), check spectrally."""
    from solver.initial_conditions import taylor_green

    N = 32
    grid = SpectralGrid(N, device, "fp64")
    u_hat = taylor_green(grid)
    om = torch.fft.irfftn(curl_hat(u_hat, grid), s=(N, N, N), dim=(-3, -2, -1))
    z, y, x = _coords(N, device)
    om_z_exact = 2.0 * torch.sin(x) * torch.sin(y) * torch.cos(z)
    assert float((om[2] - om_z_exact).abs().max()) < 1e-12


def test_dealias_mask_cut(device):
    N = 30  # non-power-of-two also fine
    grid = SpectralGrid(N, device, "fp64")
    assert grid.k_max_resolved == N // 3
    # a mode just above the cut must vanish
    u_hat = torch.zeros(3, N, N, N // 2 + 1, dtype=torch.complex128, device=device)
    k_hi = N // 3 + 1
    u_hat[0, 0, 0, k_hi] = 1.0
    dealias_(u_hat, grid)
    assert float(u_hat.abs().max()) == 0.0
