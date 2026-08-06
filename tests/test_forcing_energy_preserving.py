"""Energy-preserving forcing: band energy pinned to E_f0 after post_step,
modes outside the band untouched, eps_inj diagnostic consistent."""
import torch

from solver.forcing import EnergyPreservingForcing
from solver.grids import SpectralGrid
from solver.initial_conditions import random_solenoidal


def test_band_energy_pinned(device):
    N, E_f0 = 32, 0.37
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=4)
    f = EnergyPreservingForcing(grid, k_f=2.0, E_f0=E_f0)
    E_before = f.band_energy(u_hat)
    K_before = grid.kinetic_energy(u_hat)
    outside_before = u_hat[:, ~f.band_mask].clone()

    dt = 0.01
    eps_inj = f.post_step(u_hat, dt)

    assert abs(f.band_energy(u_hat) - E_f0) / E_f0 < 1e-12
    # post_step re-projects first (A12 regression fix); on an already
    # solenoidal input that is the identity up to round-off relative to the
    # field scale, not bitwise
    scale = float(outside_before.abs().max())
    diff = float((u_hat[:, ~f.band_mask] - outside_before).abs().max())
    assert diff < 1e-12 * scale
    assert abs(eps_inj - (E_f0 - E_before) / dt) < 1e-9
    # total energy changed by the band adjustment (up to projection round-off)
    K_after = grid.kinetic_energy(u_hat)
    assert abs((K_after - K_before) - (E_f0 - E_before)) < 1e-9


def test_idempotent_at_target(device):
    N, E_f0 = 32, 0.4
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=5)
    f = EnergyPreservingForcing(grid, k_f=2.0, E_f0=E_f0)
    f.post_step(u_hat, 0.01)
    before = u_hat.clone()
    eps2 = f.post_step(u_hat, 0.01)
    assert abs(eps2) < 1e-10
    assert torch.allclose(u_hat, before, rtol=1e-12, atol=1e-15)
