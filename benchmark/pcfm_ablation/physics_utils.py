"""
Physics utilities for Navier-Stokes constraints on periodic 3D domains.

All functions operate on PHYSICAL (denormalized) velocity fields unless noted.
Domain assumed [0, 2π]³ with periodic boundary conditions.

Velocity shape convention: [B, 3, Z, H, W] where channels are (u, v, w).
Full field shape: [B, T, C, Z, H, W] where C=4 (u, v, w, p).
"""

import torch
import torch.fft
import math


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def load_norm_stats_tensors(norm_stats_dict, device='cpu'):
    """Convert norm_stats dict to tensors for fast vectorized ops.

    Args:
        norm_stats_dict: {'u': {'mean': ..., 'std': ...}, 'v': ..., ...}

    Returns:
        mean: [4] tensor (u, v, w, p order)
        std:  [4] tensor
    """
    indicators = ['u', 'v', 'w', 'p']
    mean = torch.tensor([norm_stats_dict[ind]['mean'] for ind in indicators],
                        dtype=torch.float32, device=device)
    std = torch.tensor([norm_stats_dict[ind]['std'] for ind in indicators],
                       dtype=torch.float32, device=device)
    return mean, std


def denormalize_field(field_norm, mean, std):
    """Denormalize [B, T, C, Z, H, W] or [B, C, Z, H, W] to physical scale.

    mean, std: [C] tensors.
    """
    if field_norm.ndim == 6:
        # [B, T, C, Z, H, W]
        m = mean[None, None, :, None, None, None]
        s = std[None, None, :, None, None, None]
    elif field_norm.ndim == 5:
        # [B, C, Z, H, W]
        m = mean[None, :, None, None, None]
        s = std[None, :, None, None, None]
    else:
        raise ValueError(f"Expected 5D or 6D tensor, got {field_norm.ndim}D")
    return field_norm * s + m


def renormalize_field(field_phys, mean, std):
    """Renormalize physical field back to normalized space."""
    if field_phys.ndim == 6:
        m = mean[None, None, :, None, None, None]
        s = std[None, None, :, None, None, None]
    elif field_phys.ndim == 5:
        m = mean[None, :, None, None, None]
        s = std[None, :, None, None, None]
    else:
        raise ValueError(f"Expected 5D or 6D tensor, got {field_phys.ndim}D")
    return (field_phys - m) / s


def extract_velocity(field, ndim=6):
    """Extract velocity channels (first 3) from full field.

    field: [B, T, C=4, Z, H, W] or [B, C=4, Z, H, W]
    Returns: [B, T, 3, Z, H, W] or [B, 3, Z, H, W]
    """
    if ndim == 6 or field.ndim == 6:
        return field[:, :, :3]
    return field[:, :3]


def set_velocity(field, velocity, ndim=6):
    """Set velocity channels in full field, preserving pressure.

    field: [B, T, C=4, ...] or [B, C=4, ...]
    velocity: [B, T, 3, ...] or [B, 3, ...]
    Returns: modified field (clone).
    """
    out = field.clone()
    if ndim == 6 or field.ndim == 6:
        out[:, :, :3] = velocity
    else:
        out[:, :3] = velocity
    return out


# ---------------------------------------------------------------------------
# Wavenumber grid
# ---------------------------------------------------------------------------

def wavenumber_grid_3d(Z, H, W, device='cpu', dealias_nyquist=True):
    """Build 3D wavenumber grid for periodic domain [0, 2π]³.

    Returns kz, ky, kx arrays each of shape [Z, H, W//2+1] (rfft convention).
    Wavenumbers are integers: k_i ∈ {0, 1, ..., N/2, -(N/2-1), ..., -1}.
    For rfft the last axis only has [0, ..., W//2].

    If dealias_nyquist=True (default), zeroes out Nyquist frequencies.
    This is standard in pseudospectral methods: the Nyquist mode has
    ambiguous sign (k = ±N/2) which breaks Hermitian symmetry in
    operations like Leray projection. Zeroing it ensures correctness.
    """
    # fftfreq gives fractions; multiply by N to get integer wavenumbers
    freq_z = torch.fft.fftfreq(Z, d=1.0/Z, device=device)  # [Z]
    freq_y = torch.fft.fftfreq(H, d=1.0/H, device=device)  # [H]
    freq_x = torch.fft.rfftfreq(W, d=1.0/W, device=device) # [W//2+1]

    if dealias_nyquist:
        if Z % 2 == 0:
            freq_z[Z // 2] = 0
        if H % 2 == 0:
            freq_y[H // 2] = 0
        if W % 2 == 0:
            freq_x[W // 2] = 0

    kz, ky, kx = torch.meshgrid(freq_z, freq_y, freq_x, indexing='ij')
    return kz, ky, kx


# ---------------------------------------------------------------------------
# Leray projection (divergence-free)
# ---------------------------------------------------------------------------

def leray_projection_3d(velocity_phys, grid_shape=None):
    """Project velocity field to divergence-free subspace via FFT.

    Uses Helmholtz decomposition: u = u_sol + ∇φ
    Projection removes the gradient (irrotational) part.

    In Fourier space: û_proj(k) = û(k) - k (k·û(k)) / |k|²

    Args:
        velocity_phys: [B, 3, Z, H, W] velocity in physical space
        grid_shape: (Z, H, W) tuple, inferred from velocity if None

    Returns:
        velocity_proj: [B, 3, Z, H, W] divergence-free velocity
    """
    B, C, Z, H, W = velocity_phys.shape
    assert C == 3, f"Expected 3 velocity channels, got {C}"
    device = velocity_phys.device

    # FFT each component (real-to-complex, last dim halved)
    u_hat = torch.fft.rfftn(velocity_phys[:, 0], dim=(-3, -2, -1))  # [B, Z, H, W//2+1]
    v_hat = torch.fft.rfftn(velocity_phys[:, 1], dim=(-3, -2, -1))
    w_hat = torch.fft.rfftn(velocity_phys[:, 2], dim=(-3, -2, -1))

    # Wavenumber grid
    kz, ky, kx = wavenumber_grid_3d(Z, H, W, device=device)

    # |k|² (avoid division by zero at k=0 and dealiased Nyquist modes)
    k_sq = kx**2 + ky**2 + kz**2
    k_sq_safe = k_sq.clone()
    k_sq_safe[k_sq < 1e-10] = 1.0  # Protects DC and dealiased Nyquist modes

    # k · û = kx*ux_hat + ky*uy_hat + kz*uz_hat
    k_dot_u = kx * u_hat + ky * v_hat + kz * w_hat

    # Remove irrotational part: û -= k * (k·û) / |k|²
    correction = k_dot_u / k_sq_safe
    u_hat = u_hat - kx * correction
    v_hat = v_hat - ky * correction
    w_hat = w_hat - kz * correction

    # Zero mode (k=0): correction is 0*(...)/1 = 0, so mean is preserved.
    # No explicit restoration needed — k_sq_safe[0,0,0]=1 and k=0 components
    # are all zero, so correction[0,0,0] = 0.

    # IFFT back
    u_proj = torch.fft.irfftn(u_hat, s=(Z, H, W), dim=(-3, -2, -1))
    v_proj = torch.fft.irfftn(v_hat, s=(Z, H, W), dim=(-3, -2, -1))
    w_proj = torch.fft.irfftn(w_hat, s=(Z, H, W), dim=(-3, -2, -1))

    return torch.stack([u_proj, v_proj, w_proj], dim=1)


# ---------------------------------------------------------------------------
# Divergence computation
# ---------------------------------------------------------------------------

def compute_divergence_3d(velocity_phys, grid_shape=None):
    """Compute divergence ∇·u via spectral differentiation.

    Args:
        velocity_phys: [B, 3, Z, H, W]

    Returns:
        divergence: [B, Z, H, W] real-valued divergence field
    """
    B, C, Z, H, W = velocity_phys.shape
    device = velocity_phys.device

    u_hat = torch.fft.rfftn(velocity_phys[:, 0], dim=(-3, -2, -1))
    v_hat = torch.fft.rfftn(velocity_phys[:, 1], dim=(-3, -2, -1))
    w_hat = torch.fft.rfftn(velocity_phys[:, 2], dim=(-3, -2, -1))

    kz, ky, kx = wavenumber_grid_3d(Z, H, W, device=device)

    # ∇·u in Fourier space: i*kx*û + i*ky*v̂ + i*kz*ŵ
    div_hat = 1j * (kx * u_hat + ky * v_hat + kz * w_hat)

    divergence = torch.fft.irfftn(div_hat, s=(Z, H, W), dim=(-3, -2, -1))
    return divergence


def divergence_stats(velocity_phys):
    """Compute max and mean absolute divergence.

    Args:
        velocity_phys: [B, 3, Z, H, W]

    Returns:
        div_max: [B] max |∇·u| per sample
        div_mean: [B] mean |∇·u| per sample
    """
    div = compute_divergence_3d(velocity_phys)
    div_abs = div.abs()
    div_max = div_abs.flatten(1).max(dim=1).values
    div_mean = div_abs.flatten(1).mean(dim=1)
    return div_max, div_mean


# ---------------------------------------------------------------------------
# Kinetic energy
# ---------------------------------------------------------------------------

def compute_kinetic_energy(velocity_phys, grid_shape=None):
    """Compute total kinetic energy E_k = 0.5 * ∫|u|² dx.

    For periodic domain [0, 2π]³ with N^3 grid points:
    E_k = 0.5 * mean(|u|²) * (2π)³

    Args:
        velocity_phys: [B, 3, Z, H, W]

    Returns:
        energy: [B] scalar per sample
    """
    # |u|² = u² + v² + w²
    u_sq = (velocity_phys ** 2).sum(dim=1)  # [B, Z, H, W]
    # Volume-averaged, then multiply by domain volume (2π)³
    Z, H, W = velocity_phys.shape[2:]
    domain_vol = (2 * math.pi) ** 3
    energy = 0.5 * u_sq.mean(dim=(-3, -2, -1)) * domain_vol
    return energy


def compute_kinetic_energy_per_timestep(field_phys):
    """Compute kinetic energy per timestep.

    Args:
        field_phys: [B, T, C=4, Z, H, W] full field (u, v, w, p)

    Returns:
        energy: [B, T] kinetic energy per timestep
    """
    velocity = field_phys[:, :, :3]  # [B, T, 3, Z, H, W]
    B, T = velocity.shape[:2]
    vel_flat = velocity.reshape(B * T, 3, *velocity.shape[3:])
    ek = compute_kinetic_energy(vel_flat)
    return ek.reshape(B, T)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def compute_momentum(velocity_phys, grid_shape=None):
    """Compute total momentum p = ∫u dx.

    For periodic domain: p_i = mean(u_i) * (2π)³

    Args:
        velocity_phys: [B, 3, Z, H, W]

    Returns:
        momentum: [B, 3] vector per sample
    """
    domain_vol = (2 * math.pi) ** 3
    mom = velocity_phys.mean(dim=(-3, -2, -1)) * domain_vol  # [B, 3]
    return mom


# ---------------------------------------------------------------------------
# Energy spectrum
# ---------------------------------------------------------------------------

def compute_energy_spectrum(velocity_phys, grid_shape=None):
    """Compute shell-averaged energy spectrum E(|k|).

    E(κ) = 0.5 * Σ_{|k|∈[κ-0.5, κ+0.5)} |û(k)|²

    Args:
        velocity_phys: [B, 3, Z, H, W]

    Returns:
        wavenumbers: [K] 1D array of shell wavenumber magnitudes
        spectrum: [B, K] energy per shell per sample
    """
    B, C, Z, H, W = velocity_phys.shape
    device = velocity_phys.device

    # Full complex FFT (not rfft) for proper shell averaging
    u_hat = torch.fft.fftn(velocity_phys[:, 0], dim=(-3, -2, -1))
    v_hat = torch.fft.fftn(velocity_phys[:, 1], dim=(-3, -2, -1))
    w_hat = torch.fft.fftn(velocity_phys[:, 2], dim=(-3, -2, -1))

    # Energy density per mode: 0.5 * (|û|² + |v̂|² + |ŵ|²) / N³²
    # FFT normalization: fftn has no normalization factor, so |û|² = N³² * |true coeff|²
    N_total = Z * H * W
    energy_density = 0.5 * (u_hat.abs()**2 + v_hat.abs()**2 + w_hat.abs()**2) / (N_total ** 2)

    # Wavenumber magnitudes
    kz, ky, kx = torch.meshgrid(
        torch.fft.fftfreq(Z, d=1.0/Z, device=device),
        torch.fft.fftfreq(H, d=1.0/H, device=device),
        torch.fft.fftfreq(W, d=1.0/W, device=device),
        indexing='ij'
    )
    k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)

    # Shell averaging
    k_max = int(max(Z, H, W) // 2)
    wavenumbers = torch.arange(1, k_max + 1, dtype=torch.float32, device=device)
    spectrum = torch.zeros(B, k_max, device=device)

    for i, kappa in enumerate(wavenumbers):
        shell_mask = (k_mag >= kappa - 0.5) & (k_mag < kappa + 0.5)
        for b in range(B):
            spectrum[b, i] = energy_density[b][shell_mask].sum()

    return wavenumbers.cpu(), spectrum.cpu()


# ---------------------------------------------------------------------------
# Momentum correction (zero-frequency adjustment)
# ---------------------------------------------------------------------------

def correct_momentum(velocity_phys, target_momentum):
    """Correct momentum by adjusting zero-frequency (mean velocity).

    Args:
        velocity_phys: [B, 3, Z, H, W]
        target_momentum: [B, 3] target momentum values

    Returns:
        velocity_corrected: [B, 3, Z, H, W]
    """
    domain_vol = (2 * math.pi) ** 3
    current_mom = compute_momentum(velocity_phys)
    # Momentum difference → mean velocity correction
    delta_mean = (target_momentum - current_mom) / domain_vol  # [B, 3]
    correction = delta_mean[:, :, None, None, None]  # [B, 3, 1, 1, 1]
    return velocity_phys + correction


# ---------------------------------------------------------------------------
# Energy scaling
# ---------------------------------------------------------------------------

def correct_energy(velocity_phys, target_energy):
    """Scale velocity field to match target kinetic energy.

    Uniform scaling preserves divergence-free property.

    Args:
        velocity_phys: [B, 3, Z, H, W]
        target_energy: [B] target kinetic energy

    Returns:
        velocity_scaled: [B, 3, Z, H, W]
    """
    current_energy = compute_kinetic_energy(velocity_phys)
    scale = torch.sqrt(target_energy / (current_energy + 1e-12))  # [B]
    return velocity_phys * scale[:, None, None, None, None]
