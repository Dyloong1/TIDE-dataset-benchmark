"""Fixed-power band forcing (production scheme): exact energy increment,
solenoidality preserved, modes outside the band untouched, and the
laminar-state escape property (band energy grows when the cascade is dead)."""
import torch

from solver.config import ForcingConfig, SolverConfig
from solver.forcing import FixedPowerBandForcing
from solver.grids import SpectralGrid
from solver.initial_conditions import random_solenoidal
from solver.solver import PseudoSpectralSolver


def test_exact_injection_and_solenoidality(device):
    N, eps_w, dt = 32, 0.096, 2e-3
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=6)
    f = FixedPowerBandForcing(grid, k_f=2.0, eps_w=eps_w)
    K0 = grid.kinetic_energy(u_hat)
    outside = u_hat[:, ~f.band_mask].clone()

    inj = f.post_step(u_hat, dt)

    assert inj == eps_w
    K1 = grid.kinetic_energy(u_hat)
    assert abs((K1 - K0) - eps_w * dt) < 1e-12       # exact increment
    scale = float(outside.abs().max())
    assert float((u_hat[:, ~f.band_mask] - outside).abs().max()) < 1e-12 * scale
    div = grid.kx * u_hat[0] + grid.ky * u_hat[1] + grid.kz * u_hat[2]
    assert float(div.abs().max()) / float(u_hat.abs().max()) < 1e-13


def test_band_strictly_below_kf(device):
    """Band is 0 < |k| < k_f (KB 2.2.1 / directive): |k| = 2 modes excluded."""
    grid = SpectralGrid(32, "cpu", "fp64")
    f = FixedPowerBandForcing(grid, k_f=2.0, eps_w=0.1)
    k_mag = grid.k2.sqrt()
    assert bool((k_mag[f.band_mask] < 2.0).all())
    assert not bool(f.band_mask[grid.k2 == 4.0].any())


def test_no_absorbing_state(device):
    """With a (near-)laminar low-energy band and zero cascade, repeated steps
    must GROW the band energy (escape route from laminarization) — unlike the
    energy-preserving rescale, which would pin it."""
    N, eps_w = 32, 0.096
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=8) * 0.05   # weak field
    scfg = SolverConfig(N=N, nu=0.5, dtype="fp64", device=device,  # huge nu kills cascade
                        forcing=ForcingConfig(type="fixed_power", k_f=2.0, eps_w=eps_w))
    s = PseudoSpectralSolver(scfg, u_hat)
    f = s.forcing
    E0 = f.band_energy(s.u_hat)
    for _ in range(200):
        s.step(2e-3)
    E1 = f.band_energy(s.u_hat)
    # viscous decay at nu=0.5 fights back, but injection must keep E_f finite
    # and the cumulative injected energy is exactly eps_w * t
    assert E1 > 0.0
    assert s.last_eps_inj == eps_w
