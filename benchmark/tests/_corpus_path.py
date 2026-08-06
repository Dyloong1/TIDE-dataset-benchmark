"""Locate the corpus in a machine-independent way, and REFUSE to skip silently.

Why this exists: three corpus tests hardcoded `/media/ydai17/T7 Shield/turbgen_data` -- CYB-q's
POSIX path. On CYB-t (`D:\\turbgen_data`) and CYB-3 (`C:\\Yilong\\turbgen_data`) those tests would
skip with the reason "corpus not mounted" and the run would report PASSED. Those machines DO have a
corpus; only the path differs. So the diagnostic they are being asked to run as a second pair of
eyes would have verified nothing and told them it was fine -- the same silent-degradation class as
the val=0 bug, in the one place we least want it.

It also violated a rule the repo already had (CLAUDE.md): machine-specific paths go through an env
var, never hardcoded. The var already exists and is used by the dataset itself: TURBGEN_DATA_DIR.

Resolution order:
  1. $TURBGEN_DATA_DIR                      (set this on t / CYB-3)
  2. q's T7 mount                           (fallback: the var is unset on q today, and without
                                             this fallback q's own suite would go all-skip)
"""
import os
from pathlib import Path

import pytest

_Q_FALLBACK = Path("/media/ydai17/T7 Shield/turbgen_data")


def data_root() -> Path:
    """Corpus ROOT (the dir CONTAINING `corpus/`). May not exist; call require_corpus() to gate."""
    env = os.environ.get("TURBGEN_DATA_DIR", "").strip()
    return Path(env) if env else _Q_FALLBACK


def corpus_dir() -> Path:
    """The `corpus/` dir holding <CASE>.zarr and benchmark_slice.json."""
    return data_root() / "corpus"


def skip_reason() -> str | None:
    """None if the corpus is usable here; otherwise a reason that says how to FIX it.

    The old reason was "corpus not mounted", which on t/CYB-3 was actively false and made a
    verified-nothing run look green. This one names the path actually searched and the env var to
    set, so a skip is self-diagnosing instead of self-congratulating.
    """
    root, cdir = data_root(), corpus_dir()
    src = "$TURBGEN_DATA_DIR" if os.environ.get("TURBGEN_DATA_DIR", "").strip() else "CYB-q fallback"
    if not cdir.is_dir():
        return (f"corpus not found at {cdir} (from {src}). This test verifies REAL corpus data and "
                f"cannot run without it. On CYB-t/CYB-3 set TURBGEN_DATA_DIR to your corpus root "
                f"(the dir containing corpus/), e.g. D:\\turbgen_data or C:\\Yilong\\turbgen_data. "
                f"A skip here means your data was NOT checked -- do not read it as a pass.")
    if not any(cdir.glob("*.zarr")):
        return (f"{cdir} exists but holds no *.zarr (from {src}). Your data was NOT checked.")
    return None


def slice_path():
    """The benchmark slice, wherever it actually is on THIS machine.

    q generates the merged 15-config slice and distributes it via git at docs/slice/, but every
    consumer (dataset, suite, these tests) reads it from the corpus root. Those two are different
    places, and CYB-3 hit it immediately: `test_at_least_one_case_checked` failed on their box even
    though they HAD both a local sidecar and a relam70 train split -- the test just looked in
    corpus/ and q had published to docs/slice/. Distribution point != use point.

    So resolve both, corpus-root first (that is what training actually reads, so a stale repo copy
    must never shadow a fresh local one). Returns None if neither exists.
    """
    c = corpus_dir() / "benchmark_slice.json"
    if c.exists():
        return c
    repo = Path(__file__).resolve().parents[2] / "docs" / "slice" / "benchmark_slice.json"
    return repo if repo.exists() else None


def requires_corpus(fn):
    """Decorator: skip with a self-diagnosing reason when the corpus is genuinely absent."""
    return pytest.mark.skipif(skip_reason() is not None, reason=skip_reason() or "")(fn)
