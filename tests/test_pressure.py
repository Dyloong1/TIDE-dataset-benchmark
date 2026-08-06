"""Velocity-pressure consistency (acceptance D4): the kinematic pressure
recovered from u must satisfy the pressure Poisson equation and the
rotational-form identity grad(p + |u|^2/2) = -(I - P)(u x omega)."""
import torch

from solver.grids import SpectralGrid
from solver.initial_conditions import random_solenoidal
from solver.operators import curl_hat, pressure_hat

_FFT = (-3, -2, -1)


def test_pressure_satisfies_poisson(device):
    """lap(p) = - d_i d_j (u_i u_j): residual to machine precision."""
    N = 32
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=1, u_rms=0.6)
    u = torch.fft.irfftn(u_hat, s=(N, N, N), dim=_FFT)
    p_hat = pressure_hat(u_hat, grid)

    # pressure Poisson equation: lap(p) = - d_i d_j (u_i u_j)
    # LHS: lap(p) = -k^2 p_hat
    lap_p = (-grid.k2) * p_hat
    # RHS: - d_i d_j (u_i u_j); d_i d_j -> (i k_i)(i k_j) = -k_i k_j, so
    #      - d_i d_j (u_i u_j) -> -(-k_i k_j) T = + k_i k_j T
    ks = (grid.kx, grid.ky, grid.kz)
    rhs = torch.zeros_like(p_hat)
    for i in range(3):
        for j in range(3):
            tij = torch.fft.rfftn(u[i] * u[j], dim=_FFT)
            rhs += (ks[i] * ks[j]) * tij
    # both sides have zero mean (k=0); compare on k != 0
    scale = float(rhs.abs().max())
    assert float((lap_p - rhs).abs().max()) / scale < 1e-10


def test_pressure_balances_advection_irrotational_part(device):
    """The convective term (u.grad)u splits into an irrotational part balanced
    exactly by grad(p) and a solenoidal remainder. Concretely, for steady or
    instantaneous incompressible flow, the irrotational part of -(u.grad)u
    equals grad(p): (I - P)[(u.grad)u]_hat = (i k) p_hat.
    This is the physical velocity-pressure relation D4 certifies."""
    N = 32
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=2, u_rms=0.6)
    u = torch.fft.irfftn(u_hat, s=(N, N, N), dim=_FFT)
    ks = (grid.kx, grid.ky, grid.kz)

    # convective term (u.grad)u_i = d_j(u_i u_j)  (since div u = 0)
    adv_hat = torch.empty(3, N, N, grid.Nh, dtype=grid.cdtype, device=device)
    for i in range(3):
        acc = torch.zeros(N, N, grid.Nh, dtype=grid.cdtype, device=device)
        for j in range(3):
            acc += (1j * ks[j]) * torch.fft.rfftn(u[i] * u[j], dim=_FFT)
        adv_hat[i] = acc
    # irrotational part of -adv: (I - P)(-adv) = grad part
    div = ks[0] * adv_hat[0] + ks[1] * adv_hat[1] + ks[2] * adv_hat[2]
    div = div * grid.inv_k2
    irrot_minus_adv = torch.stack([-ks[i] * div for i in range(3)])  # = grad part of -adv

    # this must equal grad(p) = i k p_hat
    p_hat = pressure_hat(u_hat, grid)
    grad_p = torch.stack([(1j * ks[i]) * p_hat for i in range(3)])
    scale = float(grad_p.abs().max())
    assert float((irrot_minus_adv - grad_p).abs().max()) / scale < 1e-10


def test_pressure_mean_is_gauged(device):
    N = 32
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=3)
    p_hat = pressure_hat(u_hat, grid)
    assert float(p_hat[0, 0, 0].abs()) == 0.0
