#!/usr/bin/env python3
"""
Step 0: Spectral analysis of training data for noise ablation.

Computes and saves the mean energy spectrum E(k) for DNS and JHU datasets,
along with fitted parameters (beta_fit, k_peak) needed by noise generators.

Usage:
    python noise_ablation/analyze_spectrum.py --data_source dns --data_dir /path/to/TGV_data
    python noise_ablation/analyze_spectrum.py --data_source jhu --data_dir /path/to/JHU_data

Output:
    noise_ablation/spectrum_{dns,jhu}.pt
    noise_ablation/spectrum_{dns,jhu}.png  (log-log E(k) plot)
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
import sys

CODE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_DIR))

from pcfm_ablation.physics_utils import compute_energy_spectrum


def load_dns_training_velocities(data_dir, split_config_path, norm_stats=None):
    """Load all DNS training velocity fields (denormalized).

    Returns list of (3, Z, H, W) numpy arrays.
    """
    with open(split_config_path) as f:
        config = json.load(f)

    data_dir = Path(data_dir)
    DNS_TRUNC_Z = 64
    indicators = ['u', 'v', 'w']

    # Load norm stats for denormalization
    if norm_stats is None:
        stats_path = CODE_DIR / 'Data' / 'normalization_stats.json'
        if stats_path.exists():
            with open(stats_path) as f:
                raw = json.load(f)
            inds = raw.get('indicators', ['u', 'v', 'w', 'p'])
            norm_stats = {}
            for i, ind in enumerate(inds):
                norm_stats[ind] = {'mean': raw['mean'][i], 'std': raw['std'][i]}

    # Collect unique frame indices from train split
    # Config format: splits.train is a flat list of frame indices
    train_frames = set()
    splits = config.get('splits', {})
    train_data = splits.get('train', config.get('train_sequences', config.get('train', [])))
    if train_data and isinstance(train_data[0], (int, float)):
        # Flat list of frame indices
        for f in train_data:
            train_frames.add(int(f))
    else:
        # List of sequences with frame_range or frames
        for seq in train_data:
            if isinstance(seq, dict):
                if 'frame_range' in seq:
                    start, end = seq['frame_range']
                    for f in range(start, end + 1):
                        train_frames.add(f)
                elif 'frames' in seq:
                    for f in seq['frames']:
                        train_frames.add(f)

    # Subsample if too many (keep <=50 for speed)
    frame_list = sorted(train_frames)
    if len(frame_list) > 50:
        indices = np.linspace(0, len(frame_list) - 1, 50, dtype=int)
        frame_list = [frame_list[i] for i in indices]

    print(f"Loading {len(frame_list)} DNS frames for spectral analysis...")

    velocities = []
    for fidx in frame_list:
        vel = np.zeros((3, DNS_TRUNC_Z, 128, 128), dtype=np.float32)
        for c, ind in enumerate(indicators):
            fpath = data_dir / f"img_{ind}_dns{fidx}.npy"
            if not fpath.exists():
                continue
            data = np.load(fpath).astype(np.float32)[:DNS_TRUNC_Z, :, :]
            # Denormalize to physical scale
            if norm_stats and ind in norm_stats:
                # Raw data is already physical; if dataset normalizes, we want physical
                pass  # DNS .npy files are raw physical values
            vel[c] = data
        velocities.append(vel)

    return velocities


def load_jhu_training_velocities(data_dir, split_config_path, norm_stats=None):
    """Load JHU training velocity fields (denormalized).

    Returns list of (3, Z, H, W) numpy arrays.
    """
    with open(split_config_path) as f:
        config = json.load(f)

    data_dir = Path(data_dir)
    indicators = ['u', 'v', 'w']

    # Collect unique timestamps from train split
    # Config format: splits.train is a list of sequences, each a list of timestamp strings
    train_timestamps = set()
    splits = config.get('splits', {})
    train_data = splits.get('train', config.get('train_sequences', config.get('train', [])))
    if train_data:
        if isinstance(train_data[0], list):
            # List of sequences, each containing timestamp strings
            for seq in train_data:
                for ts_str in seq:
                    train_timestamps.add(float(ts_str))
        elif isinstance(train_data[0], dict):
            # Dict-based sequences with frame_range
            for seq in train_data:
                if 'frame_range' in seq:
                    start, end = seq['frame_range']
                    for f in range(start, end + 1):
                        ts = round(5.02 + f * 0.01, 3)
                        train_timestamps.add(ts)
        else:
            # Flat list of timestamps
            for ts in train_data:
                train_timestamps.add(float(ts))

    ts_list = sorted(train_timestamps)
    if len(ts_list) > 50:
        indices = np.linspace(0, len(ts_list) - 1, 50, dtype=int)
        ts_list = [ts_list[i] for i in indices]

    print(f"Loading {len(ts_list)} JHU frames for spectral analysis...")

    velocities = []
    for ts in ts_list:
        ts_str = f"{ts:.3f}"
        vel = np.zeros((3, 128, 128, 128), dtype=np.float32)
        valid = True
        for c, ind in enumerate(indicators):
            fpath = data_dir / f"{ind}_{ts_str}.npy"
            if not fpath.exists():
                valid = False
                break
            data = np.load(fpath).astype(np.float32)
            # JHU data may be big-endian
            if data.dtype.byteorder == '>':
                data = data.byteswap().newbyteorder()
            vel[c] = data
        if valid:
            velocities.append(vel)

    return velocities


def compute_mean_spectrum(velocities, device='cpu'):
    """Compute mean energy spectrum over a list of velocity fields.

    Args:
        velocities: list of (3, Z, H, W) numpy arrays (physical scale)

    Returns:
        wavenumbers: 1D tensor [K]
        E_mean: 1D tensor [K] mean energy per shell
        E_std: 1D tensor [K] std of energy per shell
    """
    all_spectra = []
    for vel in velocities:
        vel_t = torch.from_numpy(vel).unsqueeze(0).to(device)  # (1, 3, Z, H, W)
        wn, spec = compute_energy_spectrum(vel_t)  # wn: [K], spec: [1, K]
        all_spectra.append(spec[0])  # [K]

    spectra = torch.stack(all_spectra, dim=0)  # (N, K)
    E_mean = spectra.mean(dim=0)
    E_std = spectra.std(dim=0)

    return wn, E_mean, E_std


def fit_powerlaw(wavenumbers, E_mean, fit_range=None):
    """Fit E(k) = A * k^{-beta} in log-log space.

    Args:
        wavenumbers: 1D tensor
        E_mean: 1D tensor
        fit_range: (k_min, k_max) range for fitting (inertial subrange)

    Returns:
        beta_fit: float, fitted spectral exponent
        A_fit: float, fitted amplitude
        k_min_fit, k_max_fit: actual fit range used
    """
    k = wavenumbers.numpy()
    E = E_mean.numpy()

    # Auto-detect inertial range if not specified
    if fit_range is None:
        # Skip lowest wavenumbers (energy-containing range) and highest (dissipation)
        k_max = len(k)
        k_min_idx = max(2, int(k_max * 0.1))  # skip first 10%
        k_max_idx = min(k_max - 1, int(k_max * 0.5))  # use up to 50%
    else:
        k_min_idx = np.searchsorted(k, fit_range[0])
        k_max_idx = np.searchsorted(k, fit_range[1])

    # Filter valid points (E > 0)
    mask = E[k_min_idx:k_max_idx] > 0
    if mask.sum() < 3:
        print("WARNING: Too few valid points for power-law fit, using beta=5/3")
        return 5.0 / 3.0, 1.0, float(k[k_min_idx]), float(k[k_max_idx])

    log_k = np.log(k[k_min_idx:k_max_idx][mask])
    log_E = np.log(E[k_min_idx:k_max_idx][mask])

    # Linear regression in log-log space
    coeffs = np.polyfit(log_k, log_E, 1)
    beta_fit = -coeffs[0]  # E ~ k^{-beta}, so slope = -beta
    A_fit = np.exp(coeffs[1])

    return float(beta_fit), float(A_fit), float(k[k_min_idx]), float(k[k_max_idx - 1])


def find_k_peak(wavenumbers, E_mean):
    """Find wavenumber of peak energy."""
    idx = torch.argmax(E_mean)
    return float(wavenumbers[idx])


def plot_spectrum(wavenumbers, E_mean, E_std, beta_fit, k_peak, save_path):
    """Generate log-log energy spectrum plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    k = wavenumbers.numpy()
    E = E_mean.numpy()
    E_s = E_std.numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Mean spectrum with std band
    ax.loglog(k, E, 'b-', linewidth=2, label='$E(k)$ mean')
    ax.fill_between(k, np.maximum(E - E_s, 1e-20), E + E_s, alpha=0.2, color='blue')

    # k^{-5/3} reference line
    k_ref = k[k > 2]
    E_ref_start = E[np.searchsorted(k, 3)]
    E_ref = E_ref_start * (k_ref / 3.0) ** (-5.0 / 3.0)
    ax.loglog(k_ref, E_ref, 'r--', linewidth=1.5, label='$k^{-5/3}$')

    # Fitted power law
    E_fit = E[np.searchsorted(k, 3)] * (k_ref / 3.0) ** (-beta_fit)
    ax.loglog(k_ref, E_fit, 'g:', linewidth=1.5,
              label=f'$k^{{-{beta_fit:.2f}}}$ (fit)')

    # k_peak marker
    ax.axvline(k_peak, color='orange', linestyle='-.', alpha=0.7,
               label=f'$k_{{peak}}={k_peak:.0f}$')

    ax.set_xlabel('Wavenumber $k$')
    ax.set_ylabel('Energy spectrum $E(k)$')
    ax.set_title(f'Energy Spectrum ($\\beta_{{fit}}={beta_fit:.3f}$, $k_{{peak}}={k_peak:.0f}$)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Spectrum plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Spectral analysis for noise ablation')
    parser.add_argument('--data_source', type=str, required=True,
                        choices=['dns', 'jhu'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--split_config', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for FFT computation')
    args = parser.parse_args()

    # Default split configs
    if args.split_config is None:
        if args.data_source == 'dns':
            args.split_config = str(CODE_DIR / 'Data' / 'split_config_ar.json')
        else:
            args.split_config = str(CODE_DIR / 'Dataset' / 'jhu_data_splits_ar.json')

    output_dir = Path(__file__).parent
    device = args.device

    # Load velocity fields
    print(f"\n{'='*60}")
    print(f"Spectral Analysis: {args.data_source.upper()}")
    print(f"{'='*60}")

    if args.data_source == 'dns':
        velocities = load_dns_training_velocities(args.data_dir, args.split_config)
    else:
        velocities = load_jhu_training_velocities(args.data_dir, args.split_config)

    if len(velocities) == 0:
        print("ERROR: No velocity fields loaded!")
        sys.exit(1)

    print(f"Loaded {len(velocities)} velocity fields, shape: {velocities[0].shape}")

    # Compute mean spectrum
    print("\nComputing energy spectra...")
    wavenumbers, E_mean, E_std = compute_mean_spectrum(velocities, device=device)
    print(f"Spectrum computed: {len(wavenumbers)} shells, k=[{wavenumbers[0]:.0f}, {wavenumbers[-1]:.0f}]")

    # Fit power law
    beta_fit, A_fit, k_min_fit, k_max_fit = fit_powerlaw(wavenumbers, E_mean)
    print(f"\nPower-law fit: beta = {beta_fit:.4f} (fit range: k=[{k_min_fit:.0f}, {k_max_fit:.0f}])")
    print(f"  Theory: Kolmogorov beta = {5/3:.4f}")
    print(f"  Difference from -5/3: {abs(beta_fit - 5/3):.4f}")

    # Find peak
    k_peak = find_k_peak(wavenumbers, E_mean)
    print(f"  Peak wavenumber: k_peak = {k_peak:.0f}")

    # Save spectrum data
    save_path = output_dir / f'spectrum_{args.data_source}.pt'
    torch.save({
        'wavenumbers': wavenumbers,
        'E_mean': E_mean,
        'E_std': E_std,
        'beta_fit': beta_fit,
        'A_fit': A_fit,
        'k_peak': k_peak,
        'k_fit_range': (k_min_fit, k_max_fit),
        'data_source': args.data_source,
        'num_samples': len(velocities),
        'sample_shape': list(velocities[0].shape),
    }, save_path)
    print(f"\nSpectrum data saved to: {save_path}")

    # Plot
    plot_path = output_dir / f'spectrum_{args.data_source}.png'
    plot_spectrum(wavenumbers, E_mean, E_std, beta_fit, k_peak, plot_path)

    print(f"\nDone!")


if __name__ == '__main__':
    main()
