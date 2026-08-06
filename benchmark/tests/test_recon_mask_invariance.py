"""The recon mask must depend on PHYSICAL sample identity, not on storage layout.

Why: the benchmark's comparability contract says every model sees the SAME observed-pixel set at
index k. The mask RNG was keyed on frames[0] -- a zarr ROW INDEX -- which shifts when a store
holds a different set of seeds (seed 4 of kf4 is row 600 in the full corpus but row 150 in the
train-only NVMe stage). That made masks statistically INDEPENDENT across roots (measured overlap
~0.05 = chance) while the truth data stayed bit-identical: a silent leaderboard corruption,
invisible in logs. It became reachable in a single run once train read the NVMe stage and val read
the full corpus.

These tests use a synthetic 2-store fixture (tiny grid) rather than the real corpus so they run
anywhere and don't depend on the NVMe stage existing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

_BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH.parent))

from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset  # noqa: E402

N, FRAMES = 8, 4
SEEDS_FULL = [0, 1, 2]
SEEDS_STAGED = [2]          # a "stage" holding ONLY seed 2 -> its rows renumber to start at 0


def _make_store(root, seeds, case="tcase"):
    """Write a corpus-shaped zarr holding `seeds`, each with FRAMES frames of deterministic data."""
    p = Path(root) / "corpus" / f"{case}.zarr"
    p.parent.mkdir(parents=True, exist_ok=True)
    z = zarr.open(str(p), mode="w")
    n = len(seeds) * FRAMES
    u = np.zeros((n, 3, N, N, N), dtype=np.float32)
    pr = np.zeros((n, N, N, N), dtype=np.float32)
    sd = np.zeros(n, dtype=np.int64)
    t = np.zeros(n, dtype=np.float64)
    r = 0
    for s in seeds:
        for k in range(FRAMES):
            # content keyed on PHYSICAL identity (seed, ordinal) so both stores hold identical data
            u[r] = float(s) + 0.01 * k
            pr[r] = float(s) - 0.01 * k
            sd[r] = s
            t[r] = 0.1 * k
            r += 1
    z.create_dataset("u", data=u); z.create_dataset("p", data=pr)
    z.create_dataset("seed", data=sd); z.create_dataset("t", data=t)
    z.attrs["complete"] = True          # the dataset refuses partial stores
    return p


@pytest.fixture(scope="module")
def two_roots(tmp_path_factory):
    full = tmp_path_factory.mktemp("full"); staged = tmp_path_factory.mktemp("staged")
    _make_store(full, SEEDS_FULL); _make_store(staged, SEEDS_STAGED)
    sl = {"per_config": {"tcase": {"train": {"seeds": [2]}, "val": {"seeds": [0]},
                                   "test": {"seeds": [1]}}}}
    import json
    sp = Path(tmp_path_factory.mktemp("slice")) / "s.json"
    sp.write_text(json.dumps(sl), encoding="utf-8")
    return str(full), str(staged), str(sp)


def _recon_sample(root, slice_path):
    ds = TurbgenZarrDataset(case="tcase", split="train", slice_manifest=slice_path,
                            data_root=root, task="recon", t_in=1, t_out=1,
                            per_frame_n=FRAMES, normalize=False)
    i = 0
    s, frames, _pid = ds.samples[i]
    smp = ds[i]
    return ds, s, frames[0], smp


def test_row_index_really_differs_between_roots(two_roots):
    """Guard the guard: if both roots gave the same row, the test below would pass vacuously."""
    full, staged, sp = two_roots
    _dsf, _sf, row_full, _f = _recon_sample(full, sp)
    _dss, _ss, row_staged, _s = _recon_sample(staged, sp)
    assert row_full != row_staged, (
        f"fixture broken: seed 2 must sit at a different zarr row in the two stores "
        f"(got {row_full} both times), otherwise this suite cannot detect the bug")


def test_recon_mask_is_root_invariant(two_roots):
    """Same logical sample, two stores: truth identical AND mask identical."""
    full, staged, sp = two_roots
    _dsf, _sf, _rf, sf = _recon_sample(full, sp)
    _dss, _ss, _rs, ss = _recon_sample(staged, sp)
    assert torch.allclose(sf["output_dense"], ss["output_dense"]), \
        "fixture broken: the two stores must hold the same physical data"
    mf = (sf["input_dense"] != 0)
    ms = (ss["input_dense"] != 0)
    assert torch.equal(mf, ms), (
        "recon mask differs across data roots for the SAME logical sample -- the mask RNG is keyed "
        "on storage layout (a zarr row) instead of physical identity. Models trained against "
        "different roots would be scored on different observed pixels.")


def test_frame_ordinal_is_physical_identity(two_roots):
    """The ordinal of a seed's k-th frame is k in both stores, regardless of its row."""
    full, staged, sp = two_roots
    dsf, _s, _r, _x = _recon_sample(full, sp)
    dss, _s2, _r2, _x2 = _recon_sample(staged, sp)
    for k in range(FRAMES):
        row_f = dsf.by_seed[2][k]
        row_s = dss.by_seed[2][k]
        assert dsf._frame_ordinal(2, row_f) == k
        assert dss._frame_ordinal(2, row_s) == k
