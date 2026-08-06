"""
Noise generators for FM-Hybrid flow matching prior ablation.

All generators produce tensors of shape (B, C, Z, H, W) with zero mean and
unit variance per-channel, matching the contract of torch.randn_like.

Spectral noise is generated in Fourier space (rfft convention) by multiplying
complex Gaussian random variables by a wavenumber-dependent weight w(|k|),
then transforming back and normalizing.

Divergence-free variants apply Leray projection to velocity channels (u,v,w)
and use independent white noise for pressure (p).
"""

import torch
import torch.fft
import math
import sys
from pathlib import Path

# Add project root so we can import pcfm_ablation utilities
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pcfm_ablation.physics_utils import wavenumber_grid_3d, leray_projection_3d


# ---------------------------------------------------------------------------
# Spectrum data cache (loaded once per path, reused across calls)
# ---------------------------------------------------------------------------

_spectrum_cache = {}


def _load_spectrum(path):
    """Load and cache precomputed spectrum data."""
    path = str(path)
    if path not in _spectrum_cache:
        _spectrum_cache[path] = torch.load(path, map_location='cpu', weights_only=True)
    return _spectrum_cache[path]


# ---------------------------------------------------------------------------
# Core spectral noise generator
# ---------------------------------------------------------------------------

def _spectral_noise_3d(Z, H, W, weight_func, device):
    """Generate a single 3D field with prescribed power spectrum.

    Args:
        Z, H, W: spatial dimensions
        weight_func: callable(k_mag) -> amplitude weight tensor [Z, H, W//2+1]
        device: torch device

    Returns:
        field: (Z, H, W) real tensor, zero-mean, unit-variance
    """
    # Complex Gaussian in Fourier space (rfft convention)
    real_part = torch.randn(Z, H, W // 2 + 1, device=device)
    imag_part = torch.randn(Z, H, W // 2 + 1, device=device)
    xi = torch.complex(real_part, imag_part)

    # Wavenumber magnitudes
    kz, ky, kx = wavenumber_grid_3d(Z, H, W, device=device, dealias_nyquist=True)
    k_mag = torch.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    # Apply spectral weight
    w = weight_func(k_mag)
    w[k_mag < 0.5] = 0.0  # zero DC mode (k=0)
    noise_hat = xi * w

    # IFFT to physical space
    field = torch.fft.irfftn(noise_hat, s=(Z, H, W), dim=(-3, -2, -1))

    # Normalize to zero mean, unit variance
    field = field - field.mean()
    std = field.std()
    if std > 1e-12:
        field = field / std

    return field


# ---------------------------------------------------------------------------
# Weight functions for different noise types
# ---------------------------------------------------------------------------

def _weight_kolmogorov(k_mag):
    """Per-mode weight for E(k) ~ k^{-5/3} (3D shell-averaged spectrum).

    In 3D, shell-averaged E(k) = C * k^2 * |w(k)|^2 where k^2 counts
    modes per shell. For E(k) ~ k^{-5/3}:
        k^2 * |w(k)|^2 ~ k^{-5/3}
        |w(k)|^2 ~ k^{-11/3}
        w(k) = k^{-11/6}
    """
    w = torch.zeros_like(k_mag)
    mask = k_mag > 0.5
    w[mask] = k_mag[mask] ** (-11.0 / 6.0)
    return w


def _weight_powerlaw(k_mag, beta):
    """Per-mode weight for E(k) ~ k^{-beta} (3D shell-averaged spectrum).

    w(k) = k^{-(beta+2)/2} so that k^2 * w^2 ~ k^{-beta}.
    """
    w = torch.zeros_like(k_mag)
    mask = k_mag > 0.5
    w[mask] = k_mag[mask] ** (-(beta + 2.0) / 2.0)
    return w


def _weight_spectrum_matched(k_mag, spectrum_data):
    """Per-mode weight to match data's shell-averaged energy spectrum.

    E_data(k) is already shell-averaged (includes k^2 mode counting).
    To generate noise with matching spectrum:
        E_gen(k) = C * k^2 * |w(k)|^2 = E_data(k)
        w(k) = sqrt(E_data(k) / k^2)

    spectrum_data must contain:
        'wavenumbers': 1D tensor of integer shell wavenumbers [1, 2, ..., K]
        'E_mean': 1D tensor of mean energy per shell
    """
    wavenumbers = spectrum_data['wavenumbers'].to(k_mag.device)
    E_mean = spectrum_data['E_mean'].to(k_mag.device)

    # Map continuous k_mag to nearest shell index
    k_int = torch.round(k_mag).long()
    k_max = int(wavenumbers.max().item())

    # Build lookup: index 0 unused, indices 1..k_max map to E_mean
    lookup = torch.zeros(k_max + 1, device=k_mag.device)
    n_shells = min(len(wavenumbers), k_max)
    lookup[1:n_shells + 1] = E_mean[:n_shells].clamp(min=0)

    # Per-mode weight: sqrt(E_data / k^2)
    k_safe = k_int.clamp(0, k_max)
    E_at_k = lookup[k_safe]
    k_sq = k_mag ** 2
    k_sq_safe = k_sq.clamp(min=1.0)  # avoid division by zero
    w = torch.sqrt(E_at_k / k_sq_safe)
    w[k_mag < 0.5] = 0.0
    return w


def _weight_von_karman(k_mag, k_peak):
    """Per-mode weight for von Karman spectrum: E(k) = A * k^4 / (1 + (k/k0)^2)^{17/6}.

    E(k) is shell-averaged, so per-mode weight accounts for k^2:
        k^2 * |w(k)|^2 = E(k) = A * k^4 / (1 + (k/k0)^2)^{17/6}
        |w(k)|^2 = A * k^2 / (1 + (k/k0)^2)^{17/6}
        w(k) = k / (1 + (k/k0)^2)^{17/12}
    """
    k0 = max(float(k_peak), 1.0)
    w = torch.zeros_like(k_mag)
    mask = k_mag > 0.5
    km = k_mag[mask]
    w[mask] = km / (1 + (km / k0) ** 2) ** (17.0 / 12.0)
    return w


def _weight_partial_matched(k_mag, spectrum_data, alpha=0.5):
    """Per-mode weight for partial spectrum match.

    Interpolates between flat (white, alpha=0) and full match (alpha=1):
        E_target(k) = E_data(k)^alpha * C^(1-alpha)
    Per-mode: w(k) = sqrt(E_target(k) / k^2) = (E_data(k)/k^2)^{alpha/2}

    alpha=1.0 is full match, alpha=0 is flat (white noise).
    """
    wavenumbers = spectrum_data['wavenumbers'].to(k_mag.device)
    E_mean = spectrum_data['E_mean'].to(k_mag.device)

    k_int = torch.round(k_mag).long()
    k_max = int(wavenumbers.max().item())

    lookup = torch.zeros(k_max + 1, device=k_mag.device)
    n_shells = min(len(wavenumbers), k_max)
    lookup[1:n_shells + 1] = E_mean[:n_shells].clamp(min=0)

    k_safe = k_int.clamp(0, k_max)
    E_at_k = lookup[k_safe]
    k_sq = k_mag ** 2
    k_sq_safe = k_sq.clamp(min=1.0)
    # Per-mode weight with partial matching
    w = (E_at_k / k_sq_safe).clamp(min=0) ** (alpha / 2.0)
    w[k_mag < 0.5] = 0.0
    return w


# ---------------------------------------------------------------------------
# HIGH-FREQUENCY NOISE weight functions
# ---------------------------------------------------------------------------

def _weight_blue(k_mag, alpha=0.5):
    """Per-mode weight for blue noise: E(k) ~ k^{2+2*alpha}.

    w(k) = k^alpha so that E(k) = k^2 * w^2 = k^{2+2*alpha}.
    Higher alpha => more energy at high frequencies.
    alpha=0.5: mild high-freq boost (E ~ k^3)
    alpha=1.0: strong high-freq boost (E ~ k^4)
    """
    w = torch.zeros_like(k_mag)
    mask = k_mag > 0.5
    w[mask] = k_mag[mask] ** alpha
    return w


def _weight_inverse_spectrum(k_mag, spectrum_data):
    """Per-mode weight INVERSELY proportional to data energy spectrum.

    Where data has LOW energy (high frequencies), noise has HIGH amplitude.
    This forces the flow matching refinement to focus on under-represented scales.

    w(k) = sqrt(k^2 / E_data(k)) = k / sqrt(E_data(k))
    So E_noise(k) = k^2 * w^2 = k^4 / E_data(k) ~ 1/E_data(k) * k^4
    """
    wavenumbers = spectrum_data['wavenumbers'].to(k_mag.device)
    E_mean = spectrum_data['E_mean'].to(k_mag.device)

    k_int = torch.round(k_mag).long()
    k_max = int(wavenumbers.max().item())

    lookup = torch.zeros(k_max + 1, device=k_mag.device)
    n_shells = min(len(wavenumbers), k_max)
    lookup[1:n_shells + 1] = E_mean[:n_shells].clamp(min=1e-15)

    k_safe = k_int.clamp(0, k_max)
    E_at_k = lookup[k_safe].clamp(min=1e-15)

    # w(k) = k / sqrt(E_data(k))
    w = k_mag / torch.sqrt(E_at_k)
    # Cap extreme values to prevent numerical issues
    w_valid = w[k_mag > 0.5]
    if w_valid.numel() > 0:
        cap = w_valid.median() * 20.0
        w = w.clamp(max=cap.item())
    w[k_mag < 0.5] = 0.0
    return w


def _weight_inverse_spectrum_v2(k_mag, spectrum_data):
    """Per-mode weight for TRUE inverse spectrum: E_noise(k) = 1/E_data(k).

    Corrected version: accounts for k^2 shell counting properly.
        k^2 |w|^2 = 1/E_data(k)
        w(k) = 1 / (k * sqrt(E_data(k)))

    Compare with v1 (_weight_inverse_spectrum) which has E_noise = k^4/E_data.
    This v2 gives a much milder high-frequency emphasis (~1000x from k=1 to k=64
    for Kolmogorov turbulence, vs ~4 billion x for v1).
    """
    wavenumbers = spectrum_data['wavenumbers'].to(k_mag.device)
    E_mean = spectrum_data['E_mean'].to(k_mag.device)

    k_int = torch.round(k_mag).long()
    k_max = int(wavenumbers.max().item())

    lookup = torch.zeros(k_max + 1, device=k_mag.device)
    n_shells = min(len(wavenumbers), k_max)
    lookup[1:n_shells + 1] = E_mean[:n_shells].clamp(min=1e-15)

    k_safe = k_int.clamp(0, k_max)
    E_at_k = lookup[k_safe].clamp(min=1e-15)

    # w(k) = 1 / (k * sqrt(E_data(k)))
    k_safe_val = k_mag.clamp(min=1.0)
    w = 1.0 / (k_safe_val * torch.sqrt(E_at_k))
    # Cap extreme values
    w_valid = w[k_mag > 0.5]
    if w_valid.numel() > 0:
        cap = w_valid.median() * 20.0
        w = w.clamp(max=cap.item())
    w[k_mag < 0.5] = 0.0
    return w


def _weight_error_weighted(k_mag, error_spectrum_data, data_spectrum_data):
    """Per-mode weight proportional to RELATIVE error: E_err(k) / E_data(k).

    Uses precomputed error spectrum from vanilla FM-Hybrid (white noise)
    and data energy spectrum. The ratio tells us where the model struggles
    most RELATIVE to signal strength — high frequencies have ratio ~60%.

    E_noise(k) = ratio(k) = E_err(k) / E_data(k)
    w(k) = sqrt(ratio(k) / k^2)
    """
    wavenumbers = error_spectrum_data['wavenumbers'].to(k_mag.device)
    E_err = error_spectrum_data['E_err_mean'].to(k_mag.device)
    E_data = data_spectrum_data['E_mean'].to(k_mag.device)

    k_int = torch.round(k_mag).long()
    k_max = int(wavenumbers.max().item())

    # Build ratio lookup: E_err(k) / E_data(k)
    n_shells = min(len(wavenumbers), min(len(E_err), len(E_data)))
    ratio_lookup = torch.zeros(k_max + 1, device=k_mag.device)
    for ki in range(1, n_shells + 1):
        ed = E_data[ki - 1].clamp(min=1e-15)
        ee = E_err[ki - 1].clamp(min=0)
        ratio_lookup[ki] = ee / ed

    k_safe = k_int.clamp(0, k_max)
    ratio_at_k = ratio_lookup[k_safe]
    k_sq = k_mag ** 2
    k_sq_safe = k_sq.clamp(min=1.0)
    w = torch.sqrt(ratio_at_k / k_sq_safe)
    w[k_mag < 0.5] = 0.0
    return w


# ---------------------------------------------------------------------------
# Per-channel spectral noise (non-divfree)
# ---------------------------------------------------------------------------

def _generate_spectral_channels(shape, weight_func, device):
    """Generate (B, C, Z, H, W) noise with each channel independent.

    Each channel gets the same spectral weighting but independent random phases.
    """
    B, C, Z, H, W = shape
    noise = torch.empty(B, C, Z, H, W, device=device)
    for b in range(B):
        for c in range(C):
            noise[b, c] = _spectral_noise_3d(Z, H, W, weight_func, device)
    return noise


# ---------------------------------------------------------------------------
# Divergence-free noise
# ---------------------------------------------------------------------------

def _generate_divfree_noise(shape, weight_func, device):
    """Generate (B, C, Z, H, W) noise with div-free velocity + white pressure.

    The model uses shape (B, T*C, Z, H, W) where C=4 (u,v,w,p) per timestep.
    Channel layout: [u_t0, v_t0, w_t0, p_t0, u_t1, v_t1, w_t1, p_t1, ...]

    For each group of 4 channels (one timestep):
    1. Generate spectral noise for u, v, w
    2. Apply Leray projection to enforce div(u,v,w) = 0
    3. Normalize all 3 velocity channels by a SINGLE factor (preserves div-free)
    4. Generate independent white noise for pressure

    Using a single normalization factor for all 3 velocity channels is critical:
    per-channel normalization would break the divergence-free property.
    """
    B, total_C, Z, H, W = shape

    # Determine number of timestep groups (4 channels each: u,v,w,p)
    assert total_C % 4 == 0, f"Total channels must be multiple of 4, got {total_C}"
    n_groups = total_C // 4

    noise = torch.empty(B, total_C, Z, H, W, device=device)

    for b in range(B):
        for g in range(n_groups):
            base = g * 4  # channel offset for this timestep

            # Generate 3 velocity channels with spectral structure
            vel = torch.stack([
                _spectral_noise_3d(Z, H, W, weight_func, device)
                for _ in range(3)
            ], dim=0).unsqueeze(0)  # (1, 3, Z, H, W)

            # Leray projection: make divergence-free
            vel_proj = leray_projection_3d(vel)  # (1, 3, Z, H, W)

            # Normalize all 3 velocity channels by a single factor
            # This preserves the divergence-free property
            vel_proj = vel_proj[0]  # (3, Z, H, W)
            vel_proj = vel_proj - vel_proj.mean(dim=(-3, -2, -1), keepdim=True)
            joint_std = vel_proj.std()
            if joint_std > 1e-12:
                vel_proj = vel_proj / joint_std

            noise[b, base:base+3] = vel_proj

            # Pressure: independent white noise (unit variance)
            noise[b, base+3] = torch.randn(Z, H, W, device=device)

    return noise


# ---------------------------------------------------------------------------
# Main dispatch function
# ---------------------------------------------------------------------------

# Valid noise types
NOISE_TYPES = [
    'white',
    'kolmogorov',
    'spectrum_matched',
    'divfree_white',
    'divfree_kolmogorov',
    'divfree_spectrum',
    'von_karman',
    'datadriven_beta',
    'partial_matched',
    # High-frequency noise types
    'blue_025',          # Blue noise alpha=0.25: gentle high-freq boost
    'blue_05',           # Blue noise alpha=0.5: mild high-freq boost
    'blue_075',          # Blue noise alpha=0.75: moderate high-freq boost
    'blue_10',           # Blue noise alpha=1.0: strong high-freq boost
    'inverse_spectrum',     # Inverse of data energy spectrum (aggressive: k^4/E_data)
    'inverse_spectrum_v2',  # True inverse spectrum (mild: 1/E_data)
    'error_weighted',       # Weighted by FM-Hybrid prediction error spectrum
]


def generate_noise(noise_type, shape, device, spectrum_data_path=None,
                   _spectrum_data=None):
    """Generate structured noise for flow matching prior.

    Args:
        noise_type: str, one of NOISE_TYPES
        shape: (B, C, Z, H, W) tensor shape
        device: torch device
        spectrum_data_path: path to precomputed spectrum .pt file
            (required for spectrum_matched, divfree_spectrum, von_karman,
             datadriven_beta, partial_matched)
        _spectrum_data: pre-loaded spectrum dict (overrides path loading)

    Returns:
        noise: (B, C, Z, H, W) tensor, zero-mean, unit-variance per channel
    """
    if noise_type == 'white':
        return torch.randn(*shape, device=device)

    # Load spectrum data if needed
    spectrum_data = _spectrum_data
    needs_spectrum = noise_type in (
        'spectrum_matched', 'divfree_spectrum', 'von_karman',
        'datadriven_beta', 'partial_matched', 'inverse_spectrum',
        'inverse_spectrum_v2',
    )
    if needs_spectrum and spectrum_data is None:
        if spectrum_data_path is None:
            raise ValueError(
                f"Noise type '{noise_type}' requires spectrum_data_path "
                f"(precomputed via analyze_spectrum.py)"
            )
        spectrum_data = _load_spectrum(spectrum_data_path)

    # Load error spectrum if needed (for error_weighted noise)
    error_spectrum_data = None
    if noise_type == 'error_weighted':
        if spectrum_data_path is None:
            raise ValueError(
                "Noise type 'error_weighted' requires spectrum_data_path "
                "(used to locate error_spectrum_*.pt)"
            )
        # Also load data spectrum for the ratio
        if spectrum_data is None:
            spectrum_data = _load_spectrum(spectrum_data_path)
        # Derive error spectrum path from spectrum data path
        # e.g., noise_ablation/spectrum_dns.pt -> noise_ablation/error_spectrum_dns.pt
        p = Path(spectrum_data_path)
        err_path = str(p.parent / p.name.replace('spectrum_', 'error_spectrum_'))
        if err_path not in _spectrum_cache:
            _spectrum_cache[err_path] = torch.load(err_path, map_location='cpu',
                                                    weights_only=True)
        error_spectrum_data = _spectrum_cache[err_path]

    # Non-divfree types
    if noise_type == 'kolmogorov':
        return _generate_spectral_channels(shape, _weight_kolmogorov, device)

    elif noise_type == 'spectrum_matched':
        def wf(k_mag):
            return _weight_spectrum_matched(k_mag, spectrum_data)
        return _generate_spectral_channels(shape, wf, device)

    elif noise_type == 'von_karman':
        k_peak = float(spectrum_data['k_peak'])
        def wf(k_mag):
            return _weight_von_karman(k_mag, k_peak)
        return _generate_spectral_channels(shape, wf, device)

    elif noise_type == 'datadriven_beta':
        beta_fit = float(spectrum_data['beta_fit'])
        def wf(k_mag):
            return _weight_powerlaw(k_mag, beta_fit)
        return _generate_spectral_channels(shape, wf, device)

    elif noise_type == 'partial_matched':
        def wf(k_mag):
            return _weight_partial_matched(k_mag, spectrum_data, alpha=0.5)
        return _generate_spectral_channels(shape, wf, device)

    # Divfree types
    elif noise_type == 'divfree_white':
        def wf(k_mag):
            return torch.ones_like(k_mag)
        return _generate_divfree_noise(shape, wf, device)

    elif noise_type == 'divfree_kolmogorov':
        return _generate_divfree_noise(shape, _weight_kolmogorov, device)

    elif noise_type == 'divfree_spectrum':
        def wf(k_mag):
            return _weight_spectrum_matched(k_mag, spectrum_data)
        return _generate_divfree_noise(shape, wf, device)

    # High-frequency noise types
    elif noise_type == 'blue_025':
        def wf(k_mag):
            return _weight_blue(k_mag, alpha=0.25)
        return _generate_spectral_channels(shape, wf, device)

    elif noise_type == 'blue_05':
        def wf(k_mag):
            return _weight_blue(k_mag, alpha=0.5)
        return _generate_spectral_channels(shape, wf, device)

    elif noise_type == 'blue_075':
        def wf(k_mag):
            return _weight_blue(k_mag, alpha=0.75)
        return _generate_spectral_channels(shape, wf, device)

    elif noise_type == 'blue_10':
        def wf(k_mag):
            return _weight_blue(k_mag, alpha=1.0)
        return _generate_spectral_channels(shape, wf, device)

    elif noise_type == 'inverse_spectrum':
        def wf(k_mag):
            return _weight_inverse_spectrum(k_mag, spectrum_data)
        return _generate_spectral_channels(shape, wf, device)

    elif noise_type == 'inverse_spectrum_v2':
        def wf(k_mag):
            return _weight_inverse_spectrum_v2(k_mag, spectrum_data)
        return _generate_spectral_channels(shape, wf, device)

    elif noise_type == 'error_weighted':
        def wf(k_mag):
            return _weight_error_weighted(k_mag, error_spectrum_data, spectrum_data)
        return _generate_spectral_channels(shape, wf, device)

    else:
        raise ValueError(
            f"Unknown noise_type: '{noise_type}'. "
            f"Available: {NOISE_TYPES}"
        )
