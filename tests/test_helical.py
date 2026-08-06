"""Helical projection (for helical forcing): extract the positive-helicity part
of a solenoidal field. For a divergence-free field, the positive-helicity
projection u+ satisfies the eigenvalue relation of the curl operator:
    i k x u+ = +|k| u+      (curl eigenvalue +|k|)
and the projector is P+ = 1/2 (I_sol + (i k x)/|k|) acting on solenoidal fields.
Verifies: (a) idempotent on its output, (b) curl eigenvalue, (c) solenoidal,
(d) the complementary P- has eigenvalue -|k| and P+ + P- = identity on sol fields.
"""
import torch

from solver.grids import SpectralGrid
from solver.operators import helical_project, curl_hat, leray_project_
from solver.initial_conditions import random_solenoidal


def _curl(u_hat, grid):
    out = torch.empty_like(u_hat)
    return curl_hat(u_hat, grid, out=out)


def test_positive_helicity_curl_eigenvalue(device):
    """i k x u+ = +|k| u+  for the positive-helicity projection."""
    grid = SpectralGrid(32, device, "fp64")
    u = random_solenoidal(grid, seed=3)
    up = helical_project(u, grid, sign=+1)
    curl_up = _curl(up, grid)                       # = i k x up
    kmag = grid.k2.sqrt()
    target = kmag * up                              # +|k| up
    # compare on the resolved, nonzero-k modes
    m = (grid.k2 > 0)
    num = (curl_up - target).abs()[:, m].max()
    scale = up.abs()[:, m].max() * float(kmag[m].max())
    assert float(num) / float(scale) < 1e-10


def test_negative_helicity_curl_eigenvalue(device):
    grid = SpectralGrid(32, device, "fp64")
    u = random_solenoidal(grid, seed=4)
    um = helical_project(u, grid, sign=-1)
    curl_um = _curl(um, grid)
    kmag = grid.k2.sqrt()
    target = -kmag * um
    m = (grid.k2 > 0)
    num = (curl_um - target).abs()[:, m].max()
    scale = um.abs()[:, m].max() * float(kmag[m].max()) + 1e-30
    assert float(num) / float(scale) < 1e-10


def test_helical_decomposition_completeness(device):
    """u+ + u- = u (Leray-projected), on a solenoidal field."""
    grid = SpectralGrid(32, device, "fp64")
    u = random_solenoidal(grid, seed=5)            # already solenoidal
    up = helical_project(u, grid, sign=+1)
    um = helical_project(u, grid, sign=-1)
    recon = up + um
    scale = float(u.abs().max())
    assert float((recon - u).abs().max()) / scale < 1e-10


def test_helical_idempotent(device):
    grid = SpectralGrid(32, device, "fp64")
    u = random_solenoidal(grid, seed=6)
    up = helical_project(u, grid, sign=+1)
    up2 = helical_project(up, grid, sign=+1)
    scale = float(up.abs().max()) + 1e-30
    assert float((up2 - up).abs().max()) / scale < 1e-10


def test_helical_output_solenoidal(device):
    grid = SpectralGrid(32, device, "fp64")
    u = random_solenoidal(grid, seed=7)
    up = helical_project(u, grid, sign=+1)
    div = grid.kx * up[0] + grid.ky * up[1] + grid.kz * up[2]
    assert float(div.abs().max()) / float(up.abs().max()) < 1e-12


def test_positive_helicity_has_positive_helicity(device):
    """Net helicity <u . omega> of u+ must be positive (omega = curl u)."""
    grid = SpectralGrid(32, device, "fp64")
    u = random_solenoidal(grid, seed=8)
    up = helical_project(u, grid, sign=+1)
    om = _curl(up, grid)
    # helicity spectral inner product Re<up, om*> with rfft weights
    w = grid._w64_flat.view_as(grid.k2)
    H = float((w * (up.real.double()*om.real.double()
                    + up.imag.double()*om.imag.double()).sum(0)).sum())
    assert H > 0
