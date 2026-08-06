"""Judge-first tests for the 5th-channel (theta/b) normalizer, _ScalarChannelNorm.

Locks the θ/b normalization consumer-side contract (reads zarr attrs norm_theta/b, applies
the same minmax/standardize convention as solver.normalizer). Uses the REAL frozen stats
from the produced scalar corpus (norm_theta), so the claim "bounded θ -> [-1,1], inverse
round-trips exact" is reproducible from a delivered test, not a throwaway script.

Run: KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_scalar_channel_norm.py -q
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Dataset.turbgen_zarr_dataset import _ScalarChannelNorm  # noqa: E402

# REAL frozen stats from scalar_sc1_256_fp64.zarr attrs['norm_theta'] (produced corpus).
REAL_THETA = {"channel": "theta", "mean": -6.43861743709367e-06,
              "std": 1.801494618992754, "vmax": 10.034212112426758, "vmin": -9.808435440063477}


def test_absmax_from_vmin_vmax():
    n = _ScalarChannelNorm(REAL_THETA, "minmax")
    # absmax = max(|vmin|, |vmax|) = 10.034 (no explicit minmax_scale key in this stats dict)
    assert abs(n.absmax - 10.034212112426758) < 1e-9


def test_bounded_theta_maps_to_unit_interval():
    n = _ScalarChannelNorm(REAL_THETA, "minmax")
    # theta is bounded in [vmin, vmax] by construction -> normalized into [-1, 1]
    theta = torch.tensor([REAL_THETA["vmin"], 0.0, REAL_THETA["vmax"], 5.0])
    z = n.apply(theta)
    assert float(z.max()) <= 1.0 + 1e-6 and float(z.min()) >= -1.0 - 1e-6
    assert abs(float(z[2]) - 1.0) < 1e-6           # vmax -> +1
    assert abs(float(z[0]) + 0.9775) < 1e-3        # vmin/absmax


def test_inverse_round_trip_exact():
    for mode in ("minmax", "standardize"):
        n = _ScalarChannelNorm(REAL_THETA, mode)
        x = torch.randn(8, 8, 8) * REAL_THETA["std"]
        assert (n.inverse(n.apply(x)) - x).abs().max() < 1e-5


def test_standardize_zero_mean_unit_scale():
    n = _ScalarChannelNorm(REAL_THETA, "standardize")
    # a value at the mean maps to ~0; one std above maps to ~1
    assert abs(float(n.apply(torch.tensor(REAL_THETA["mean"])))) < 1e-4
    assert abs(float(n.apply(torch.tensor(REAL_THETA["mean"] + REAL_THETA["std"]))) - 1.0) < 1e-4


def test_zero_scale_guard():
    # degenerate stats (a constant channel) must not divide-by-zero
    n = _ScalarChannelNorm({"mean": 0.0, "std": 0.0, "vmin": 0.0, "vmax": 0.0}, "minmax")
    out = n.apply(torch.zeros(4))
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} scalar-channel-norm tests passed")
