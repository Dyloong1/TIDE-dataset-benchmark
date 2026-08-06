"""benchmark_slice.json must be REPRODUCIBLE from make_benchmark_slice.py, not hand-edited.

Why: the slice was hand-edited at one point, and the JSON grew a `_split_rule` string describing a
rule that existed nowhere in the code -- while the generator actually sorted by a different rule
entirely. Neither rule reproduced the shipped file. For a D&B paper that is a hard defect: the
selection rule is a published claim, and a reviewer must be able to recompute it. This test pins
the shipped slice to its generator.

Skipped when the corpus is not mounted, so it is safe in CI.
"""
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "experiments" / "phase0"))

from _corpus_path import corpus_dir, skip_reason  # noqa: E402

_CORPUS = corpus_dir()
_SLICE = _CORPUS / "benchmark_slice.json"

pytestmark = pytest.mark.skipif(
    not _SLICE.exists(),
    reason=(skip_reason() or f"slice not found at {_SLICE}; your slice was NOT checked -- "
            "set TURBGEN_DATA_DIR to your corpus root. Do not read a skip as a pass."))


def _shipped():
    import json
    return json.loads(_SLICE.read_text(encoding="utf-8"))


def _regenerated():
    import make_benchmark_slice as MBS
    return MBS.build_slice(str(_CORPUS.parent), train_seeds=3, target_frames=150)


def test_shipped_slice_matches_its_generator():
    """Regenerating from the script must reproduce the shipped splits exactly."""
    shipped, regen = _shipped()["per_config"], _regenerated()["per_config"]
    diffs = []
    for case, exp in regen.items():
        got = shipped.get(case, {})
        for split in ("train", "val", "test"):
            a = (got.get(split) or {}).get("seeds")
            b = (exp.get(split) or {}).get("seeds")
            if a != b:
                diffs.append(f"{case}/{split}: shipped={a} generator={b}")
    assert not diffs, (
        "benchmark_slice.json does not match make_benchmark_slice.py — it was hand-edited, so a "
        "reviewer cannot reproduce the published selection rule. Change the RULE in the script and "
        "regenerate; never edit the JSON directly.\n  " + "\n  ".join(diffs))


def test_every_in_dist_config_has_a_full_split():
    """3 train / 1 val / 3 test, disjoint, for every in-dist config -- and NO empty split.

    An empty split is the failure mode this guards: scalar_sc1's test set came back EMPTY (its 4
    full-frame seeds could not fill 3+1+3 and the prefilter shut out the shorter usable ones), and
    nothing in the slice builder complained.
    """
    per = _shipped()["per_config"]
    bad = []
    for case, e in per.items():
        if e.get("role") != "in_dist":
            continue
        tr = (e.get("train") or {}).get("seeds") or []
        va = (e.get("val") or {}).get("seeds") or []
        te = (e.get("test") or {}).get("seeds") or []
        if not (len(tr) == 3 and len(va) == 1 and len(te) == 3):
            bad.append(f"{case}: sizes {len(tr)}/{len(va)}/{len(te)} != 3/1/3")
        if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
            bad.append(f"{case}: splits overlap (leakage) tr={tr} va={va} te={te}")
    assert not bad, "\n  ".join(bad)


def test_selection_is_model_independent():
    """The published rule must contain only physical terms -- never a model loss (iron rule)."""
    s = _shipped()
    weights = s.get("weights", {})
    assert weights, "slice does not publish its quality weights"
    allowed = {"w_res", "w_drift", "w_iso", "w_splice", "w_frames"}
    unknown = set(weights) - allowed
    assert not unknown, f"unrecognized quality weight(s) {unknown} -- selection must stay physical"
