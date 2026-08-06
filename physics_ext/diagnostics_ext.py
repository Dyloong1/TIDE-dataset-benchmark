"""Extended-physics diagnostics: anisotropy, rotation (R-group), stratification
(S-group), passive scalar, and the new 256^3-DNS resolution criteria. All reuse
solver.grids primitives (rfft weights, energy density, shell spectrum) and
accumulate in fp64. solver/ is never touched.

Conventions match solver.diagnostics / grids.component_moments. Vertical = z
(component index 2); horizontal = x,y. k_perp^2 = kx^2 + ky^2; vertical wavenumber
= kz. The rotating/stratified setups use the z axis (Omega = Omega_z e_z, gravity
along -z), so "2D / columnar" modes are kz = 0.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from solver.grids import SpectralGrid

_FFT_DIMS = (-3, -2, -1)


# ===================================================================== anisotropy
def anisotropy_tensor(u_hat: torch.Tensor, grid: SpectralGrid):
    """Reynolds-stress anisotropy b_ij = <u_i u_j>/(2K) - delta_ij/3.

    Isotropic turbulence -> b_ij = 0 (all eigenvalues 0). Returns the 3x3 b
    matrix and its sorted eigenvalues (lam1>=lam2>=lam3). For rotation/strat the
    largest |eigenvalue| quantifies the anisotropy that A10 would otherwise (mis)
    flag as a failure — here it is the physical observable, not a gate.
    """
    ui2, uij = grid.component_moments(u_hat)        # <u_i^2>, <u_i u_j> (i<j)
    K = 0.5 * sum(ui2)
    R = np.array([[ui2[0], uij[0], uij[1]],
                  [uij[0], ui2[1], uij[2]],
                  [uij[1], uij[2], ui2[2]]], dtype=np.float64)
    b = R / (2.0 * K) - np.eye(3) / 3.0
    lam = np.sort(np.linalg.eigvalsh(b))[::-1]
    return b, lam


def energy_2d3d(u_hat: torch.Tensor, grid: SpectralGrid):
    """Partition kinetic energy into columnar (kz=0, '2D/slow manifold') vs the
    rest ('3D/wave'). For rotating turbulence E_2D/E_total grows toward 1 as
    Ro->0 (energy condenses into vertically-invariant columns). Returns
    (E_2D, E_3D, frac_2D)."""
    e = grid._energy_density_flat(u_hat).view_as(grid.k2)     # [N,N,Nh] fp64
    kz0 = (grid.kz.abs() < 0.5)                               # kz == 0 plane, broadcast [N,1,1]
    mask2d = kz0.expand_as(e)
    E_2d = float(e[mask2d].sum())
    E_tot = float(e.sum())
    E_3d = E_tot - E_2d
    frac = E_2d / E_tot if E_tot > 0 else 0.0
    return E_2d, E_3d, frac


def helicity(u_hat: torch.Tensor, grid: SpectralGrid) -> float:
    """Mean helicity H = <u . omega>, omega = curl u. Conserved (inviscid) and a
    rotation-relevant invariant. Computed spectrally: H = sum_k w Re(u_hat . conj(omega_hat))."""
    from solver.operators import curl_hat
    om = curl_hat(u_hat, grid)
    w = grid._w64_flat.view_as(grid.k2)
    n2 = float(grid.n_total) ** 2
    prod = (u_hat.real.double() * om.real.double()
            + u_hat.imag.double() * om.imag.double()).sum(dim=0)
    return float((w * prod).sum() / n2)


def rel_helicity(u_hat: torch.Tensor, grid: SpectralGrid) -> float:
    """rho = <u.omega> / (u_rms * omega_rms), in [-1,1]."""
    from solver.operators import curl_hat
    om = curl_hat(u_hat, grid)
    K = grid.kinetic_energy(u_hat)
    # enstrophy = 0.5<|omega|^2> via same density machinery
    en = float(grid._energy_density_flat(om).sum())
    u_rms = math.sqrt(2.0 * K / 3.0) * math.sqrt(3.0)        # = sqrt(<|u|^2>)
    o_rms = math.sqrt(2.0 * en)
    H = helicity(u_hat, grid)
    return H / (u_rms * o_rms) if u_rms * o_rms > 0 else 0.0


# ===================================================================== rotation (R)
def rossby_number(u_rms: float, L_int: float, omega_mag: float) -> float:
    """Ro = u' / (2 Omega L). Small Ro = strong rotation."""
    return u_rms / (2.0 * omega_mag * L_int) if omega_mag > 0 and L_int > 0 else float("inf")


# ============================================================== stratification (S)
def froude_number(u_rms: float, L_int: float, N_bv: float) -> float:
    """Fr = u' / (N L). Small Fr = strong stratification."""
    return u_rms / (N_bv * L_int) if N_bv > 0 and L_int > 0 else float("inf")


def buoyancy_reynolds(eps: float, nu: float, N_bv: float) -> float:
    """Re_b = eps / (nu N^2). The 'is it turbulent' gate for stratified DNS:
    Re_b >> 1 needed for an inertial range; Re_b ~ O(1) = wave/layer-dominated."""
    return eps / (nu * N_bv**2) if N_bv > 0 else float("inf")


def ozmidov_scale(eps: float, N_bv: float) -> float:
    """l_O = sqrt(eps / N^3). Scales below l_O are ~isotropic turbulence, above
    are buoyancy-controlled. Must satisfy eta < l_O < L for a stratified DNS."""
    return math.sqrt(eps / N_bv**3) if N_bv > 0 else float("inf")


def potential_energy(b_hat: torch.Tensor, grid: SpectralGrid, N_sq: float) -> float:
    """PE = (1/2) <b^2> / N^2  (available potential energy density, Boussinesq)."""
    bb = float(grid._energy_density_flat(b_hat).sum()) * 2.0      # <b^2> = 2 * (0.5<b^2>)
    return 0.5 * bb / N_sq if N_sq > 0 else 0.0


def buoyancy_flux(u_hat: torch.Tensor, b_hat: torch.Tensor, grid: SpectralGrid) -> float:
    """<w b> vertical buoyancy flux (w = u_z). Spectral inner product with rfft
    weights. Drives the KE<->PE exchange in stratified turbulence."""
    w = grid._w64_flat.view_as(grid.k2)
    n2 = float(grid.n_total) ** 2
    prod = (u_hat[2].real.double() * b_hat[0].real.double()
            + u_hat[2].imag.double() * b_hat[0].imag.double())
    return float((w * prod).sum() / n2)


# ===================================================================== scalar
def scalar_variance(theta_hat: torch.Tensor, grid: SpectralGrid) -> float:
    """<theta^2> (per unit volume)."""
    return float(grid._energy_density_flat(theta_hat).sum()) * 2.0


def scalar_dissipation(theta_hat: torch.Tensor, grid: SpectralGrid, kappa: float) -> float:
    """chi = 2 kappa <|grad theta|^2> = 2 kappa sum_k k^2 |theta_hat|^2 (density)."""
    e = grid._energy_density_flat(theta_hat)
    return float(2.0 * kappa * (grid._k2_64_flat * e).sum()) * 2.0


def scalar_spectrum(theta_hat: torch.Tensor, grid: SpectralGrid):
    """Shell-summed scalar variance spectrum E_theta(k) (checks Obukhov-Corrsin k^-5/3)."""
    e = grid._energy_density_flat(theta_hat) * 2.0
    E = torch.bincount(grid.shell_index, weights=e, minlength=grid.n_shells)
    k_max = grid.N // 2
    k = torch.arange(1, k_max + 1, dtype=torch.float64)
    return k, E[1:k_max + 1].cpu()


# ============================================================ resolution (256^3 DNS)
def batchelor_resolution(k_max_eta: float, Sc: float) -> dict:
    """Scalar DNS resolution. The smallest scalar scale depends on the Schmidt
    number Sc = nu/kappa:
      * Sc >= 1: Batchelor scale eta_B = eta / sqrt(Sc)  (scalar smaller than eta)
      * Sc <  1: Obukhov-Corrsin scale eta_OC = eta * Sc^(-3/4)  (scalar LARGER)
    The resolution gate is k_max * (smallest scalar scale) >= 1.5. With Sc=1 both
    reduce to the velocity gate k_max*eta. Returns the value + pass flag.
    [Deep-review 2026-06-23: the Sc<1 branch previously returned k_max_eta
    unchanged (wrong scale); fixed to Obukhov-Corrsin.]"""
    if Sc >= 1.0:
        scale_ratio = 1.0 / math.sqrt(Sc)          # eta_B/eta
        name = "k_max_eta_B"
    else:
        scale_ratio = Sc ** (-0.75)                # eta_OC/eta (>1 for Sc<1)
        name = "k_max_eta_OC"
    kmax_scalar = k_max_eta * scale_ratio
    return {"Sc": Sc, name: kmax_scalar, "k_max_scalar_scale": kmax_scalar,
            "class_I": kmax_scalar >= 1.5}


def stratified_resolution(eta: float, l_O: float, L_int: float, Re_b: float) -> dict:
    """Stratified DNS sanity: need eta < l_O < L (Ozmidov in the inertial range)
    AND Re_b large enough for turbulence. Returns the chain + a turbulent flag."""
    return {"eta": eta, "l_O": l_O, "L": L_int, "Re_b": Re_b,
            "ozmidov_resolved": eta < l_O < L_int,
            "turbulent": Re_b > 20.0}      # Re_b>~20 conventional turbulence threshold
