"""The frozen normalizer must be fit on EXACTLY the seeds the slice calls train.

Why this exists (2026-07-15): the leak fix and the seed-selection fix are a DEPENDENCY CHAIN --
    quality score  ->  which seeds are train  ->  which frames the normalizer is fit on
and I ran them out of order. The train-only refit landed first, then fixing the (silently dead)
quality score re-ranked the seeds: rotating_ro0p2's train split moved [1,2,3] -> [6,3,8] and
scalar_sc1's [0,1,8] -> [1,0,2]. Both normalizers were now fit on frames that are no longer train
-- i.e. still fit on test seeds. The leak I had just "fixed" was quietly back on 2 of 6 configs,
and nothing anywhere would have said so: the sidecars were present, well-formed, and wrong.

That is the same shape as every other bug in this benchmark's history -- a stale artifact that
reads as green. So the invariant gets a test instead of my memory.

The check is cheap and exact because `freeze_norm_from_zarr.py` records its fit scope in the
sidecar note ("fit on 450/1200 frames from seeds [0, 4, 6] (--seed-filter train; <stamp>)"). This
compares that recorded scope against the slice's current train seeds.

Re-run `freeze_norm_from_zarr.py --case <CASE> --seed-filter train --slice <slice>` on the machine
holding the corpus whenever the slice's train seeds change.
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _corpus_path import corpus_dir, skip_reason, slice_path  # noqa: E402

pytestmark = pytest.mark.skipif(skip_reason() is not None, reason=skip_reason() or "")

_SEEDS_RE = re.compile(r"from seeds \[([\d,\s]*)\]")
_ALL_RE = re.compile(r"from all seeds")


def _slice():
    p = slice_path()
    if p is None:
        pytest.skip(f"no benchmark_slice.json in {corpus_dir()} nor docs/slice/ -- your data was "
                    f"NOT checked. A skip is not a pass.")
    return json.loads(p.read_text(encoding="utf-8")).get("per_config", {})


def _cases_with_sidecar():
    if skip_reason() is not None:
        return []
    p = slice_path()
    if p is None:
        return []
    try:
        per = json.loads(p.read_text(encoding="utf-8")).get("per_config", {})
    except Exception:
        return []
    out = []
    for f in sorted(corpus_dir().glob("*_norm_minmax.json")):
        case = f.name.replace("_norm_minmax.json", "")
        # only cases that BOTH have a local sidecar and appear in the slice with a train split
        tr = (per.get(case, {}) or {}).get("train")
        if isinstance(tr, dict) and tr.get("seeds"):
            out.append(case)
    return out


CASES = _cases_with_sidecar()


@pytest.mark.parametrize("case", CASES)
def test_normalizer_fit_on_current_train_seeds(case):
    """The sidecar's recorded fit scope must equal the slice's train seeds -- no more, no less."""
    per = _slice()
    want = sorted(int(s) for s in per[case]["train"]["seeds"])
    note = json.loads((corpus_dir() / f"{case}_norm_minmax.json").read_text(encoding="utf-8")).get("note", "")

    if _ALL_RE.search(note):
        pytest.fail(
            f"{case}: normalizer was fit on ALL seeds but the slice gives it a train split "
            f"{want}. Fitting on every frame includes the test seeds -- that is the transductive "
            f"leak. Re-run: freeze_norm_from_zarr.py --case {case} --seed-filter train --slice ...")

    m = _SEEDS_RE.search(note)
    assert m, (
        f"{case}: sidecar note does not record which seeds it was fit on (note={note!r}). Without "
        f"that, the fit scope is unverifiable -- re-freeze with the current "
        f"freeze_norm_from_zarr.py, which records it.")

    used = sorted(int(x) for x in m.group(1).split(",") if x.strip())
    assert used == want, (
        f"{case}: normalizer was fit on seeds {used} but the slice's train seeds are now {want}. "
        f"The normalizer therefore saw {sorted(set(used) - set(want))} which are no longer train "
        f"(leak), and never saw {sorted(set(want) - set(used))} which are. This happens when the "
        f"slice is regenerated after the refit -- they are a dependency chain: quality score -> "
        f"train seeds -> normalizer. Re-run: freeze_norm_from_zarr.py --case {case} "
        f"--seed-filter train --slice <slice>")


def test_at_least_one_case_checked():
    """Guard the guard: if CASES is empty this file is decorative and would pass silently."""
    assert CASES, (
        f"no case had BOTH a local *_norm_minmax.json and a train split in the slice, so this file "
        f"verified nothing -- the 'a skip reads as a pass' failure. Looked for the slice at "
        f"{corpus_dir() / 'benchmark_slice.json'} then docs/slice/benchmark_slice.json; found "
        f"{slice_path()}. If you hold corpus data, either the sidecars are missing (run "
        f"freeze_norm_from_zarr.py) or the slice has not reached this machine yet.")
