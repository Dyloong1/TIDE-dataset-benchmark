# -*- coding: utf-8 -*-
"""Local raw-frame cache vs direct zarr reads: byte-identity pin (correctness judge for the IO-layer optimization, 2026-07-17)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _corpus_path import corpus_dir, skip_reason  # noqa: E402
from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset  # noqa: E402

pytestmark = pytest.mark.skipif(skip_reason() is not None, reason=skip_reason() or "")


def _mk(tmp_path, flag_on):
    flag = Path(__file__).resolve().parents[2] / "checkpoints" / ".local_frame_cache"
    old = flag.read_text() if flag.exists() else None
    if flag_on:
        flag.parent.mkdir(exist_ok=True)
        flag.write_text(str(tmp_path))
    elif flag.exists():
        flag.unlink()
    ds = TurbgenZarrDataset(
        case="ou_robust_kf3_256_fp64", split="train",
        slice_manifest=str(corpus_dir() / "benchmark_slice.json"),
        data_root=str(corpus_dir().parent), task="rollout",
        t_in=1, t_out=1, frame_stride=8, patch_size=128, normalize=True)
    return ds, flag, old


def test_cache_bitident_and_restores_flag(tmp_path):
    ds_off, flag, old = _mk(tmp_path, flag_on=False)
    a = ds_off._load_frame(3, origin=[0, 128, 0])
    try:
        ds_on, _, _ = _mk(tmp_path, flag_on=True)
        assert ds_on._raw_cache is not None, "flag set, cache should be enabled"
        b1 = ds_on._load_frame(3, origin=[0, 128, 0])   # first touch: populates the cache

        b2 = ds_on._load_frame(3, origin=[0, 128, 0])   # second touch: served from memmap

        assert torch.equal(a, b1), "cache path first touch differs bitwise from the direct read"
        assert torch.equal(a, b2), "cache path memmap re-read differs bitwise from the direct read"
        assert (tmp_path / "ou_robust_kf3_256_fp64" / "u_3.npy").exists()
    finally:
        if old is not None:
            flag.write_text(old)
        elif flag.exists():
            flag.unlink()


def test_cache_bitident_native_fullframe(tmp_path):
    """The native full-frame path (origin=None) also goes through the disk cache (wired
    2026-07-17; root cause of a 700 s/epoch regression): direct read vs first-touch populate
    vs memmap re-read must agree bitwise three ways."""
    ds_off, flag, old = _mk(tmp_path, flag_on=False)
    a = ds_off._load_frame(3, origin=None)
    try:
        ds_on, _, _ = _mk(tmp_path, flag_on=True)
        b1 = ds_on._load_frame(3, origin=None)   # first touch: populates the cache (u_3/p_3.npy)
        b2 = ds_on._load_frame(3, origin=None)   # second touch: served from memmap

        assert torch.equal(a, b1), "native path first touch differs bitwise from the direct read"
        assert torch.equal(a, b2), "native path memmap re-read differs bitwise from the direct read"
        # shares the same cache file with the patch path (same key, same frame): no duplicate disk use
        c = ds_on._load_frame(3, origin=[0, 128, 0])
        assert torch.equal(a[:, 0:128, 128:256, 0:128], c), "native cached frame disagrees with the patch slice"
    finally:
        if old is not None:
            flag.write_text(old)
        elif flag.exists():
            flag.unlink()
