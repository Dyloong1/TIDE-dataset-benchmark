"""Seeds of a config must be INDEPENDENT trajectories, not copies of one.

Why this exists: decay_hotstart_re86 shipped 8 "seeds" that are the SAME trajectory bit-for-bit
(verified: every frame max|diff| = 0.0; the pools' own checkpoints share an md5). Its config warm-
starts from a single checkpoint and says "ic/seed fields are ignored", so the seed LABEL varied
while the physics did not, and corpus_to_zarr wrote n_seeds=8 because it counts DIRECTORIES rather
than checking content. Nothing caught it, so every downstream number treated n=1 as n=8: error bars
over 8 copies of one trajectory are meaningless, and the slice's quality ranking scored all 8 at
q=0.0 (indistinguishable) without anyone asking why.

This test is skipped when the corpus is not mounted, so it is safe in CI.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH.parent))

zarr = pytest.importorskip("zarr")

from _corpus_path import corpus_dir, skip_reason  # noqa: E402

_CORPUS = corpus_dir()

# decay_hotstart is KNOWN broken (8 copies of one trajectory) and is queued for re-production from
# 8 distinct ou_relam90 checkpoints. Listing it here keeps the suite green while the defect is
# tracked in the internal benchmark design doc -- REMOVE this entry once it is re-produced, and the
# test will then hold it to the same standard as everything else.
# EMPTY as of 2026-07-15: decay_hotstart_re86 was re-produced with 8 genuinely independent
# parent trajectories (one distinct relam90 pool checkpoint per seed, 8/8 distinct resume_ckpt_md5)
# and now passes the real check below -- closest seed pair differs by 2.55, where the old corpus
# measured exactly 0.0 on all 28 pairs. The waiver had done its job: it xfailed while the data was
# known-bad, then FAILED the moment the data got fixed ("listed in KNOWN_DUPLICATE but its seeds now
# look distinct"), which is what told us to remove it.
#
# Do not add a case here to make this suite green. A case in this set is NOT being checked; it is an
# admission that its "seeds" are copies of one trajectory, which makes every mean±std over them
# understate the true variance and makes attrs['n_seeds'] a lie.
KNOWN_DUPLICATE = set()


def _configs():
    if not _CORPUS.is_dir():
        return []
    return sorted(p.stem for p in _CORPUS.glob("*.zarr"))


@pytest.mark.parametrize("case", _configs() or ["_no_corpus"])
def test_seeds_are_distinct_trajectories(case):
    if case == "_no_corpus":
        pytest.skip(skip_reason() or "corpus not mounted")
    store = zarr.open(str(_CORPUS / f"{case}.zarr"), mode="r")
    try:
        seed = np.asarray(store["seed"][:])
        t = np.asarray(store["t"][:])
    except Exception:
        pytest.skip(f"{case}: no seed/t arrays")
    seeds = sorted(set(seed.tolist()))
    if len(seeds) < 2:
        pytest.skip(f"{case}: single seed")

    def first_row(s):
        idx = np.where(seed == s)[0]
        return idx[np.argsort(t[idx])][0]

    ref = np.asarray(store["u"][first_row(seeds[0])])
    dupes = [s for s in seeds[1:]
             if float(np.abs(ref - np.asarray(store["u"][first_row(s)])).max()) == 0.0]

    if case in KNOWN_DUPLICATE:
        assert dupes, (
            f"{case} is listed in KNOWN_DUPLICATE but its seeds now look distinct -- if it was "
            "re-produced, drop it from that set so this test guards it for real.")
        pytest.xfail(f"{case}: known duplicate seeds {dupes}, re-production queued (design doc §11)")

    assert not dupes, (
        f"{case}: seed(s) {dupes} are BIT-IDENTICAL to seed {seeds[0]} at their first frame -- "
        "these are copies of one trajectory, not independent samples. Every mean±std over them "
        "understates the true variance, and the count in attrs['n_seeds'] is a lie.")
