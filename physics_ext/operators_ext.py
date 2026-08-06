"""Extended spectral operators for the new-physics axes. Mirrors the conventions
of solver/operators.py (pure functions + explicit in-place `_` variants), but
lives here so solver/operators.py is never touched.
"""
from __future__ import annotations

import torch

from solver.grids import SpectralGrid

_FFT_DIMS = (-3, -2, -1)


def cross_product_spectral(omega_vec, u_hat: torch.Tensor,
                           out: torch.Tensor | None = None) -> torch.Tensor:
    """Spectral cross product (Omega x u_hat) for a CONSTANT vector Omega.

    Because Omega is constant (its transform is a delta at k=0), the convolution
    theorem makes (Omega x u)_hat = Omega x u_hat exactly, mode by mode — no FFT.
    NOTE: Omega x u is NOT divergence-free in general — div(Omega x u) = -Omega.omega
    (the vector identity div(a x b)=b.curl(a)-a.curl(b) with constant Omega). The
    CALLER (rotating.py) therefore Leray-projects the Coriolis term so u stays
    solenoidal; this op only does the cross product.

    omega_vec: length-3 (Omega_x, Omega_y, Omega_z), real scalars.
    u_hat:     [3, N, N, Nh] complex velocity spectrum.
    """
    ox, oy, oz = float(omega_vec[0]), float(omega_vec[1]), float(omega_vec[2])
    if out is None:
        out = torch.empty_like(u_hat)
    # (Omega x u)_x = oy*uz - oz*uy, etc.  (same index pattern as cross_product)
    torch.mul(u_hat[2], oy, out=out[0]); out[0] -= oz * u_hat[1]
    torch.mul(u_hat[0], oz, out=out[1]); out[1] -= ox * u_hat[2]
    torch.mul(u_hat[1], ox, out=out[2]); out[2] -= oy * u_hat[0]
    return out


def pressure_hat_rotating(u_hat: torch.Tensor, omega_vec, grid: SpectralGrid) -> torch.Tensor:
    """Kinematic (reduced) pressure for the ROTATING-frame momentum equation. Taking
    div of  du/dt = u x omega - grad(p) + nu lap u - 2 Omega x u  (div u = 0) gives

        lap(p) = -d_i d_j (u_i u_j) - div(2 Omega x u)
               = -d_i d_j (u_i u_j) + 2 Omega . curl(u)        [div(Omega x u) = -Omega.omega]

    so p_hat = -(k_i k_j/k^2)(u_i u_j)_hat - (1/k^2)(2 Omega.omega)_hat. The vorticity
    spectrum is omega_hat = i k x u_hat, so (Omega.omega)_hat = i Omega . (k x u_hat).
    Extends solver.operators.pressure_hat with the Coriolis source (the only new term);
    reduces to it exactly when Omega=0. k=0 mean gauged out (inv_k2=0 at k=0).

    NOT YET acceptance-blessed until the D4 test (test_physics_ext) + eval_dynamics_ext
    D4 confirm it on a real field — produced for that certification (KDD pressure-channel
    decision, 2026-06-24).
    """
    N = grid.N
    u = torch.fft.irfftn(u_hat, s=(N, N, N), dim=_FFT_DIMS)
    p_hat = torch.zeros(N, N, grid.Nh, dtype=grid.cdtype, device=grid.device)
    ks = (grid.kx, grid.ky, grid.kz)
    for i in range(3):
        for j in range(3):
            tij_hat = torch.fft.rfftn(u[i] * u[j], dim=_FFT_DIMS)
            p_hat += (ks[i] * ks[j]) * tij_hat        # = +k_i k_j (u_i u_j)_hat
    p_hat *= -grid.inv_k2                              # -> -(k_i k_j/k^2) T_hat
    # + Coriolis source: lap(p) gets +2 Omega.omega, omega_hat = i (k x u_hat)
    ox, oy, oz = float(omega_vec[0]), float(omega_vec[1]), float(omega_vec[2])
    curl_x = 1j * (ks[1] * u_hat[2] - ks[2] * u_hat[1])
    curl_y = 1j * (ks[2] * u_hat[0] - ks[0] * u_hat[2])
    curl_z = 1j * (ks[0] * u_hat[1] - ks[1] * u_hat[0])
    omega_dot = ox * curl_x + oy * curl_y + oz * curl_z     # (Omega.omega)_hat
    p_hat += -grid.inv_k2 * (2.0 * omega_dot)              # p_hat += -(1/k^2)(2 Omega.omega)_hat
    return p_hat


def scalar_gradient_phys(s_hat: torch.Tensor, grid: SpectralGrid):
    """Physical-space gradient (ds/dx, ds/dy, ds/dz) of a scalar spectrum s_hat.

    s_hat: [N, N, Nh] complex. Returns a [3, N, N, N] real tensor. Used by the
    passive-scalar / Boussinesq advection u.grad(s) (pseudo-spectral).
    """
    N = grid.N
    ks = (grid.kx, grid.ky, grid.kz)
    g = torch.empty((3, N, N, N), dtype=grid.rdtype, device=grid.device)
    for d in range(3):
        g[d] = torch.fft.irfftn(1j * ks[d] * s_hat, s=(N, N, N), dim=_FFT_DIMS)
    return g


def pressure_hat_boussinesq(u_hat: torch.Tensor, b_hat: torch.Tensor,
                            grid: SpectralGrid) -> torch.Tensor:
    """Kinematic pressure for Boussinesq flow. Taking div of the momentum eqn
    du/dt = u x omega - grad(p) + b e_z + nu lap u  (div u = 0) gives

        lap(p) = -d_i d_j (u_i u_j) + d_z b

    so p_hat = -(k_i k_j/k^2)(u_i u_j)_hat - i k_z/k^2 b_hat. Extends
    solver.operators.pressure_hat with the buoyancy source (the only new term);
    reduces to it exactly when b=0. k=0 mean gauged out (inv_k2=0 at k=0).
    """
    N = grid.N
    u = torch.fft.irfftn(u_hat, s=(N, N, N), dim=_FFT_DIMS)
    p_hat = torch.zeros(N, N, grid.Nh, dtype=grid.cdtype, device=grid.device)
    ks = (grid.kx, grid.ky, grid.kz)
    for i in range(3):
        for j in range(3):
            tij_hat = torch.fft.rfftn(u[i] * u[j], dim=_FFT_DIMS)
            p_hat += (ks[i] * ks[j]) * tij_hat        # = +k_i k_j (u_i u_j)_hat
    p_hat *= -grid.inv_k2                              # -> -(k_i k_j/k^2) T_hat
    # + buoyancy source: lap(p) gets +d_z b  ->  p_hat += -(i k_z / k^2) b_hat.
    # b_hat is canonically stored [1,N,N,Nh] (a 1-channel scalar field, as in
    # BoussinesqSolver.b_hat); squeeze the channel so it broadcasts against the
    # [N,N,Nh] p_hat.
    b = b_hat[0] if b_hat.dim() == 4 else b_hat
    p_hat += (-1j * grid.kz * grid.inv_k2) * b
    return p_hat
