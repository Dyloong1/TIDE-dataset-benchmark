"""Helical OU forcing: same machinery as OU but the force carries net helicity
(one helicity sign), breaking mirror symmetry. Verify the force is band-limited,
solenoidal, Hermitian (real physical force), and has net positive helicity
<f . curl f> > 0.
"""
import torch

from solver.forcing import HelicalOUForcing
from solver.grids import SpectralGrid
from solver.operators import curl_hat


def test_force_is_net_helical(device):
    grid = SpectralGrid(32, device, "fp64")
    f = HelicalOUForcing(grid, k_f=2.0, tau=2.0, sigma2=0.05, seed=11, helicity_sign=+1)
    for _ in range(20):                 # advance the OU process to a non-trivial state
        f.advance_ou(2e-3)
    fh = torch.empty(3, grid.N, grid.N, grid.Nh, dtype=grid.cdtype, device=device)
    f(None, fh)
    om = curl_hat(fh, grid)             # i k x f
    w = grid._w64_flat.view_as(grid.k2)
    H = float((w * (fh.real.double()*om.real.double()
                    + fh.imag.double()*om.imag.double()).sum(0)).sum())
    E = float((w * (fh.real.double()**2 + fh.imag.double()**2).sum(0)).sum())
    # relative helicity |H|/(|k|_band E) should be O(1) for a single-sign field
    assert H > 0
    assert abs(H) / (E + 1e-30) > 0.5    # strongly helical (mirror-symmetric ~0)


def test_negative_sign_flips_helicity(device):
    grid = SpectralGrid(32, device, "fp64")
    fp = HelicalOUForcing(grid, k_f=2.0, sigma2=0.05, seed=11, helicity_sign=+1)
    fm = HelicalOUForcing(grid, k_f=2.0, sigma2=0.05, seed=11, helicity_sign=-1)
    for _ in range(10):
        fp.advance_ou(2e-3); fm.advance_ou(2e-3)
    def helicity(f):
        fh = torch.empty(3, grid.N, grid.N, grid.Nh, dtype=grid.cdtype, device=device)
        f(None, fh); om = curl_hat(fh, grid)
        w = grid._w64_flat.view_as(grid.k2)
        return float((w*(fh.real.double()*om.real.double()+fh.imag.double()*om.imag.double()).sum(0)).sum())
    assert helicity(fp) > 0 and helicity(fm) < 0


def test_force_band_limited_and_solenoidal(device):
    grid = SpectralGrid(32, device, "fp64")
    f = HelicalOUForcing(grid, k_f=2.0, sigma2=0.05, seed=12)
    for _ in range(5):
        f.advance_ou(2e-3)
    fh = torch.empty(3, grid.N, grid.N, grid.Nh, dtype=grid.cdtype, device=device)
    f(None, fh)
    # band-limited: zero outside the band
    assert float(fh[:, ~f.band_mask].abs().max()) < 1e-14
    # solenoidal
    div = grid.kx*fh[0] + grid.ky*fh[1] + grid.kz*fh[2]
    assert float(div.abs().max()) / float(fh.abs().max()) < 1e-12


def test_force_hermitian_real_physical(device):
    """irfftn of the force must be real (Hermitian symmetry preserved)."""
    grid = SpectralGrid(32, device, "fp64")
    f = HelicalOUForcing(grid, k_f=2.0, sigma2=0.05, seed=13)
    for _ in range(5):
        f.advance_ou(2e-3)
    fh = torch.empty(3, grid.N, grid.N, grid.Nh, dtype=grid.cdtype, device=device)
    f(None, fh)
    N = grid.N
    fp = torch.fft.irfftn(fh, s=(N, N, N), dim=(-3, -2, -1))
    back = torch.fft.rfftn(fp, dim=(-3, -2, -1))
    assert float((back - fh).abs().max()) / float(fh.abs().max() + 1e-30) < 1e-10
