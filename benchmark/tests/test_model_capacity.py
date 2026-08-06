"""Pin every baseline's parameter count to its DOCUMENTED _params_M.

Why this exists: the benchmark's headline claim is a fair capacity-matched comparison, so a
model silently running at the wrong size invalidates the results, not just the table. This has
already happened three times: ufno ran 59.8M vs 33.6M documented (a pre-seeded key beat the JSON's
setdefault); transolver was documented at a retired t_in=20 (42.9M vs the true 28.0M); and deeponet
was documented at 8.8M -- its rollout 4->4 count -- when the only shape it is EVER built in is
pressure 3->1 (6.99M), so the published number described a run that will not exist. All three were
found by hand, late. This test makes the capacity contract machine-checked.

The production path is: suite._hp_args(model) -> CLI flags -> build_config -> SimpleNamespace
-> get_model. Anything that breaks that chain (a pre-seeded key, a dropped flag, a getattr on a
dict) shows up here as a param-count mismatch.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH.parent))

import run_benchmark_suite as RS  # noqa: E402
import train_benchmark as TB  # noqa: E402
from Model.baselines import get_model  # noqa: E402

_PROD = _BENCH / "Model" / "configs" / "production_configs_256.json"
MODELS = json.loads(_PROD.read_text(encoding="utf-8"))["models"]

# Each model is measured at the shape IT ACTUALLY RUNS -- not at one shape for everybody.
#
# This bit us: the table used to measure every model at C=4 (the velocity-task shape) and recorded
# deeponet at 8.77M. But REQUIRED puts deeponet in `pressure` and NOWHERE else, and pressure is
# 3->1, where it measures 6.99M. 8.77M is its rollout 4->4 count -- a configuration deeponet is
# never instantiated in. The published number described a run that will not exist, and it happened
# to understate the very capacity gap we are being transparent about (28-30M vs 7.0M = 4.3x, not
# 3.4x). Models that size from len(INDICATORS) (deeponet, transolver) have NO single param count:
# the number is only meaningful once you name the task.
#
# t_in=1 because NS is Markovian; transolver also sizes from t_in, so a stale T_IN here would
# silently mismeasure it (at t_in=20 it reads 42.9M vs the true 28.0M).
T_IN = 1

# model -> (in_ch, out_ch) at its production task. Velocity tasks are C=4 (u,v,w,p) -- confirmed
# from the live train.log banner "in=(1,4) out=(1,4)" and from the corpus zarr (3-comp u + p).
MAINLINE_SHAPE = {
    "fno3d": (4, 4),        # rollout / superres / pressure / sgs -- 4->4 is its band-defining shape
    "tfno": (4, 4),
    "transolver": (4, 4),   # rollout / recon only (it hardcodes out==in)
    "ufno": (4, 4),
    "deeponet": (3, 1),     # pressure ONLY
}
MAINLINE = list(MAINLINE_SHAPE)

# The four architecturally-unconstrained baselines are capacity-matched so the leaderboard measures
# method rather than size. deeponet is deliberately excluded: its dense 128^3 trunk OOMs a 32GB GPU
# above ~9M (measured), so it cannot join the band, and the alternative -- shrinking the other four
# to its ceiling -- was rejected because it would degrade every baseline to match the weakest.
CAPACITY_BAND = ["fno3d", "tfno", "transolver", "ufno"]
BAND_MAX_SPREAD = 1.25


def _build(model):
    """Rebuild `model` exactly the way the suite does: its own _hp_args through the real argparser."""
    hp = RS._hp_args(model, MODELS)
    argv = ["--model", model, "--task", "rollout", "--case", "c",
            "--data-root", "d", "--slice", "s"] + hp
    old = sys.argv
    try:
        sys.argv = ["train_benchmark.py"] + argv
        ap = argparse.ArgumentParser()
        # mirror train_benchmark's arch flags (the only ones build_config reads)
        ap.add_argument("--model"); ap.add_argument("--task"); ap.add_argument("--case")
        ap.add_argument("--data-root"); ap.add_argument("--slice")
        ap.add_argument("--width", type=int, default=32)
        ap.add_argument("--layers", type=int, default=4)
        ap.add_argument("--modes", type=int, default=16)
        ap.add_argument("--tfno-rank", type=int, default=8)
        ap.add_argument("--deeponet-p", type=int, default=128)
        args = ap.parse_args(argv)
    finally:
        sys.argv = old
    in_ch, out_ch = MAINLINE_SHAPE[model]
    cfg = TB.build_config(args, in_ch, out_ch, T_IN, 1, model=model, grid_n=128)
    with contextlib.redirect_stdout(io.StringIO()):
        net = get_model(model, cfg)
    return sum(p.numel() for p in net.parameters()) / 1e6


@pytest.mark.parametrize("model", MAINLINE)
def test_param_count_matches_documented(model):
    """Params must match _params_M within 3% at the shape the model actually runs."""
    doc = MODELS[model].get("_params_M")
    assert doc is not None, f"{model} has no _params_M in production_configs_256.json"
    got = _build(model)
    rel = abs(got - doc) / doc
    assert rel < 0.03, (
        f"{model}: documented {doc}M but production path builds {got:.2f}M ({100*(got-doc)/doc:+.1f}%). "
        "Either the model is not running at its documented capacity (a real bug that invalidates "
        "the capacity-matched comparison) or _params_M is stale. Do not 'fix' by loosening this "
        "test: re-measure at the shape in MAINLINE_SHAPE and update _params_M, or repair the "
        "config plumbing.")


def test_capacity_band_is_tight():
    """The four unconstrained baselines must be capacity-MATCHED, not merely same-order.

    A leaderboard where one model has 2x the parameters of another partly measures size. These
    four have no architectural ceiling, so there is no excuse for them to differ: they are pinned
    to ~28-30M (matched upward -- see production_configs_256.json _capacity_note).
    """
    sizes = {m: _build(m) for m in CAPACITY_BAND}
    lo, hi = min(sizes.values()), max(sizes.values())
    assert hi / lo < BAND_MAX_SPREAD, (
        f"capacity spread {hi/lo:.2f}x across the matched band: "
        f"{ {k: round(v, 2) for k, v in sizes.items()} }. These four have no architectural limit, "
        "so retune the outlier back into the band rather than widening this test.")


def test_deeponet_is_the_only_out_of_band_baseline():
    """deeponet sits below the band by architectural necessity, and that must stay DELIBERATE.

    If deeponet ever grows into the band, the exclusion should be removed; if another model drifts
    down to deeponet's size, that is a capacity regression, not a new exemption.
    """
    band_lo = min(_build(m) for m in CAPACITY_BAND)
    dn = _build("deeponet")
    assert dn < band_lo, (
        f"deeponet ({dn:.2f}M) is no longer below the band ({band_lo:.2f}M) -- if its trunk memory "
        "was fixed, fold it into CAPACITY_BAND instead of keeping the exemption.")
