"""fixed_power_damped: the large-scale soft cap that fixes the k=1 runaway.

Two properties that matter for acceptance:
  1. below the ceiling the clamp is a NO-OP -> exact eps_w injection preserved
     (so steady eps == eps_w, k_max*eta lock, Class I are all untouched);
  2. above the ceiling the clamp bleeds the excess low-shell energy back,
     breaking the runaway feedback that destabilized the plain k_f=4 run.
"""
import torch

from solver.forcing import FixedPowerBandForcingDamped
from solver.grids import SpectralGrid
from solver.initial_conditions import random_solenoidal


def test_clamp_is_noop_below_ceiling(device):
    """Normal field, ceiling set generously: clamp does nothing, injection is
    exactly eps_w (same guarantee as plain fixed_power)."""
    N, eps_w, dt = 32, 0.096, 2e-3
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=6)
    f = FixedPowerBandForcingDamped(grid, k_f=4.0, eps_w=eps_w,
                                    k_damp=2.0, e_cap_factor=10.0)  # high ceiling
    f.set_cap_from_state(u_hat)
    K0 = grid.kinetic_energy(u_hat)
    inj = f.post_step(u_hat, dt)
    K1 = grid.kinetic_energy(u_hat)
    assert inj == eps_w
    assert f.last_bled == 0.0                          # clamp inactive
    assert abs((K1 - K0) - eps_w * dt) < 1e-12         # exact increment intact


def test_clamp_bleeds_runaway_low_shell(device):
    """Artificially inflate the low shells above the ceiling; the clamp must
    bleed the excess so post-step low-shell energy returns to the cap."""
    N, eps_w, dt = 32, 0.096, 2e-3
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=6)
    f = FixedPowerBandForcingDamped(grid, k_f=4.0, eps_w=eps_w,
                                    k_damp=2.0, e_cap_factor=1.5)
    f.set_cap_from_state(u_hat)
    # blow up the damped shells by 5x amplitude (25x energy) -> way over ceiling
    u_hat[:, f.damp_mask] *= 5.0
    f.post_step(u_hat, dt)
    E_low_after = f._damp_energy(u_hat)
    # after clamping, low-shell energy is pinned at the ceiling (within injection)
    assert E_low_after <= f._E_cap * 1.001
    assert f.last_bled > 0.0                            # excess was bled


def test_outside_damp_band_untouched_by_clamp(device):
    """The clamp only scales |k| < k_damp; mid/high shells must be untouched by
    the clamp step (they still get the normal fixed_power treatment)."""
    N, eps_w, dt = 32, 0.096, 2e-3
    grid = SpectralGrid(N, device, "fp64")
    u_hat = random_solenoidal(grid, seed=7)
    f = FixedPowerBandForcingDamped(grid, k_f=4.0, eps_w=eps_w,
                                    k_damp=2.0, e_cap_factor=1.5)
    f.set_cap_from_state(u_hat)
    u_hat[:, f.damp_mask] *= 5.0
    # modes outside BOTH the forcing band and damp band must be invariant
    untouched_mask = ~f.band_mask & ~f.damp_mask
    before = u_hat[:, untouched_mask].clone()
    f.post_step(u_hat, dt)
    after = u_hat[:, untouched_mask]
    scale = float(before.abs().max()) + 1e-30
    assert float((after - before).abs().max()) / scale < 1e-12
