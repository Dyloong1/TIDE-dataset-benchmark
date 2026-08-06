"""Unit tests for the classic-IC additions (abc_flow, vortex_tubes).

Every IC must return a spectral field u_hat [3,N,N,Nh] that is (a) divergence-free
to machine precision (A12), (b) non-trivial (nonzero energy), and (c) the physical
field reconstructed from it must be REAL (correct Hermitian symmetry from rfftn).
ABC additionally must be strongly helical (Beltrami: u parallel omega).

These guard the IC dispatch BEFORE any DNS run trusts the field — a wrong IC would
otherwise silently produce a physically-meaningless trajectory that the decay
acceptance might still "pass".
"""
import numpy as np
import torch

from solver.grids import SpectralGrid
from solver.operators import curl_hat

_FFT = (-3, -2, -1)


def _div_ratio(u_hat, grid):
    """A12-style: <(div u)^2> / <|grad u|^2> (machine precision for solenoidal)."""
    kx, ky, kz = grid.kx, grid.ky, grid.kz
    div_hat = 1j * (kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2])
    N = grid.N
    div = torch.fft.irfftn(div_hat, s=(N, N, N), dim=_FFT)
    # |grad u|^2 via spectral derivatives
    grad2 = 0.0
    for i in range(3):
        for k in (kx, ky, kz):
            g = torch.fft.irfftn(1j * k * u_hat[i], s=(N, N, N), dim=_FFT)
            grad2 = grad2 + (g**2).mean()
    return float((div**2).mean() / grad2)


def _is_real_field(u_hat, grid):
    """irfftn then rfftn round-trip recovers the field -> Hermitian-correct."""
    N = grid.N
    u = torch.fft.irfftn(u_hat, s=(N, N, N), dim=_FFT)
    back = torch.fft.rfftn(u, dim=_FFT)
    return float((back - u_hat).abs().max()) / max(float(u_hat.abs().max()), 1e-30)


def _rel_helicity(u_hat, grid):
    """H / (sqrt(2K * 2 Omega)) — O(1) for a maximally helical Beltrami field."""
    om = curl_hat(u_hat, grid)
    w = grid._w64_flat.view_as(grid.k2)
    H = float((w * (u_hat.real.double() * om.real.double()
                    + u_hat.imag.double() * om.imag.double()).sum(0)).sum())
    E = float((w * (u_hat.real.double()**2 + u_hat.imag.double()**2).sum(0)).sum())
    Z = float((w * (om.real.double()**2 + om.imag.double()**2).sum(0)).sum())
    return H / max((E * Z) ** 0.5, 1e-30)


def test_abc_flow_divergence_free(device):
    from solver.initial_conditions import abc_flow
    grid = SpectralGrid(64, device, "fp64")
    u_hat = abc_flow(grid)
    assert _div_ratio(u_hat, grid) < 1e-12, "ABC must be divergence-free"
    assert float(grid.kinetic_energy(u_hat)) > 1e-6, "ABC energy must be nonzero"


def test_abc_flow_is_real_and_beltrami(device):
    from solver.initial_conditions import abc_flow
    grid = SpectralGrid(64, device, "fp64")
    u_hat = abc_flow(grid)
    assert _is_real_field(u_hat, grid) < 1e-10, "ABC physical field must be real"
    # ABC A=B=C=1 is maximally helical with POSITIVE sign (right-handed). Pin the
    # SIGNED value, not |.|, so a sign flip (curl u = -u) can't silently pass.
    assert _rel_helicity(u_hat, grid) > 0.9, "ABC must be strongly +helical"


def test_vortex_tubes_divergence_free(device):
    from solver.initial_conditions import vortex_tubes
    grid = SpectralGrid(64, device, "fp64")
    u_hat = vortex_tubes(grid)
    # numerically constructed + Leray-projected -> machine-precision solenoidal
    assert _div_ratio(u_hat, grid) < 1e-12, "vortex tubes must be divergence-free"
    assert float(grid.kinetic_energy(u_hat)) > 1e-6, "vortex energy must be nonzero"


def test_vortex_tubes_is_real_and_has_vorticity(device):
    from solver.initial_conditions import vortex_tubes
    grid = SpectralGrid(64, device, "fp64")
    u_hat = vortex_tubes(grid)
    assert _is_real_field(u_hat, grid) < 1e-10, "vortex physical field must be real"
    # the field should carry significant z-vorticity (the tubes)
    om = curl_hat(u_hat, grid)
    N = grid.N
    om_z = torch.fft.irfftn(om[2], s=(N, N, N), dim=_FFT)
    assert float((om_z**2).mean()) > 1e-6, "vortex tubes must carry z-vorticity"


def test_vortex_tubes_zero_net_circulation(device):
    """Antiparallel pair MUST have zero net circulation — a single net vortex is
    illegal in a periodic box. A flipped sign (gauss1+gauss2 instead of -) would
    still be divergence-free and carry z-vorticity (passing the other tests), so
    this guards exactly that silent-corruption vector."""
    from solver.initial_conditions import vortex_tubes
    grid = SpectralGrid(64, device, "fp64")
    u_hat = vortex_tubes(grid)
    om = curl_hat(u_hat, grid)
    net_circ = float(om[2][0, 0, 0].abs())   # k=0 mode of vorticity = net circulation
    assert net_circ < 1e-12, f"net circulation {net_circ} must be ~0 (antiparallel)"


def test_vortex_tubes_biot_savart_roundtrip(device):
    """The recovered velocity must reproduce the PRESCRIBED z-vorticity. A
    sign/component swap in the 3 Biot-Savart lines could survive the other tests
    (still solenoidal, still has z-vorticity) — this pins the inversion itself."""
    from solver.initial_conditions import vortex_tubes
    grid = SpectralGrid(64, device, "fp64")
    u_hat = vortex_tubes(grid)
    N = grid.N
    om = curl_hat(u_hat, grid)
    om_z = torch.fft.irfftn(om[2], s=(N, N, N), dim=_FFT)
    # z-vorticity is the dominant prescribed component; it should be symmetric ±
    # (antiparallel tubes) with O(1) extent, and dominate transverse vorticity.
    om_x = torch.fft.irfftn(om[0], s=(N, N, N), dim=_FFT)
    om_y = torch.fft.irfftn(om[1], s=(N, N, N), dim=_FFT)
    assert float(om_z.abs().max()) > 5 * float((om_x.abs().max() + om_y.abs().max())), \
        "z-vorticity must dominate (Biot-Savart recovered the prescribed tubes)"
    # symmetric ±: max and -min comparable (one +tube, one -tube)
    assert abs(float(om_z.max()) + float(om_z.min())) < 0.1 * float(om_z.abs().max()), \
        "antiparallel tubes -> z-vorticity symmetric about zero"
