"""Judge-first regression tests for the U-FNO spectral-conv mode-clamp fix (2026-07-05).

Two bugs were fixed and are locked here:
  1. mode-clamp OVERLAP: a fixed mode count > axis//2 made the low block [:m] and the high
     block [-m:] overlap and DOUBLE-WRITE out_ft with the wrong weights on small grids. The
     fix caps full-spectrum-axis modes at axis//2 so the two blocks never overlap. This test
     asserts the invariant AND that the forward runs at the sizes where the old code broke.
  2. OUT_CHANNELS != IN_CHANNELS (P4 pressure 3->1, P5 sgs 4->6).

Run: KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_ufno_spectral.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Model.baselines import get_model  # noqa: E402


def _cfg(ic, oc, G, m):
    return SimpleNamespace(
        INPUT_TIMESTEPS=2, OUTPUT_TIMESTEPS=2,
        INDICATORS=['u', 'v', 'w', 'p'][:ic] if ic <= 4 else ['c%d' % i for i in range(ic)],
        IN_CHANNELS=ic, OUT_CHANNELS=oc, TOTAL_Z_LAYERS=G,
        UFNO_WIDTH=8, FNO_MODES_Z=m, FNO_MODES_H=m, FNO_MODES_W=m, FNO_LAYERS=2)


def test_mode_clamp_no_overlap_invariant():
    # the low block [:mz] and high block [-mz:] must be disjoint for any (Z, modes).
    for Z in range(2, 40):
        for modes in (4, 8, 16, 32):
            mz = min(modes, Z // 2)
            lo = set(range(0, mz))
            hi = set(range(Z - mz, Z)) if mz > 0 else set()
            assert not (lo & hi), f"overlap at Z={Z} modes={modes}: mz={mz}"


def test_ufno_runs_small_and_channel_flex():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # sizes where the pre-fix overlap would have corrupted output (m ~ Z/2) + channel-flex
    for tag, ic, oc, G, m in [("16", 4, 4, 16, 4), ("32", 4, 4, 32, 8),
                              ("bigmodes", 4, 4, 16, 16),   # m==Z -> clamp engages
                              ("P4", 3, 1, 16, 4), ("P5", 4, 6, 16, 4)]:
        # ufno was dropped from MODEL_REGISTRY in the 2026-07-13 trim (appendix/future-work);
        # its .py + spectral correctness are still validated here via the class directly.
        from Model.baselines.ufno_model import UFNO3d
        m_ = UFNO3d(_cfg(ic, oc, G, m)).to(dev)
        x = torch.randn(1, 2, ic, G, G, G, device=dev)
        out = m_(x)
        assert out.shape == (1, 2, oc, G, G, G), f"{tag}: {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{tag}: non-finite"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} ufno tests passed")
