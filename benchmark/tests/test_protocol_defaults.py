"""The finalized protocol must be the DEFAULT, not something the caller has to remember.

Why this exists: the suite passes every protocol value explicitly, so suite runs were correct while
the argparse/constructor defaults silently rotted back to the discarded protocol -- eval_benchmark
defaulted to t_out=30 / per_frame_n=4 / frame_stride=64 (the "sparse eval" idea that was measured to
be a pure loss), train_benchmark to epochs=50 (double the budget, and CosineAnnealingLR(T_max=epochs)
follows it, so the model is not even comparable), and the dataset to t_in=20 / frame_stride=5. Nobody
noticed because nobody runs the scripts bare -- until someone debugs one directly, which is exactly
how the poisson bound was developed.

A default that disagrees with the protocol is a loaded trap. Pin them.
"""
import argparse
import inspect
import re
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH.parent))

# The finalized protocol (internal benchmark design doc, section 0). Keep this table and the doc in lockstep.
TRAIN_DEFAULTS = {"--epochs": 25, "--t-in": 1, "--frame-stride": 4, "--per-frame-n": 20, "--seed": 0}
EVAL_DEFAULTS = {"--t-in": 1, "--t-out": 20, "--frame-stride": 8, "--per-frame-n": 20}
DATASET_DEFAULTS = {"t_in": 1, "t_out": 1, "frame_stride": 4, "per_frame_n": 20}


def _argparse_default(filename, flag):
    src = (_BENCH / filename).read_text(encoding="utf-8")
    m = re.search(r'add_argument\("' + re.escape(flag) + r'", type=int, default=(\d+)', src)
    assert m, f"{filename}: no int default found for {flag}"
    return int(m.group(1))


@pytest.mark.parametrize("flag,want", sorted(TRAIN_DEFAULTS.items()))
def test_train_benchmark_defaults_are_the_protocol(flag, want):
    got = _argparse_default("train_benchmark.py", flag)
    assert got == want, (
        f"train_benchmark.py {flag} defaults to {got}, protocol is {want}. The suite passes this "
        "explicitly so suite runs are fine -- but a bare invocation silently runs a different "
        "experiment. Fix the default; do not loosen this test.")


@pytest.mark.parametrize("flag,want", sorted(EVAL_DEFAULTS.items()))
def test_eval_benchmark_defaults_are_the_protocol(flag, want):
    got = _argparse_default("eval_benchmark.py", flag)
    assert got == want, (
        f"eval_benchmark.py {flag} defaults to {got}, protocol is {want}. Same trap as above: "
        "eval_benchmark once defaulted to frame_stride=64, the sparse-eval idea that was MEASURED "
        "to throw away 186 of 195 samples for the lowest N_eff.")


@pytest.mark.parametrize("param,want", sorted(DATASET_DEFAULTS.items()))
def test_dataset_constructor_defaults_are_the_protocol(param, want):
    from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset
    got = inspect.signature(TurbgenZarrDataset.__init__).parameters[param].default
    assert got == want, (
        f"TurbgenZarrDataset {param} defaults to {got}, protocol is {want}. A caller that omits "
        "this kwarg builds the wrong pairs -- t_in=20 silently made every input tensor 20x too big.")


def test_suite_passes_the_same_values_it_documents():
    """The suite's explicit values must agree with the defaults -- one protocol, not two."""
    src = (_BENCH / "run_benchmark_suite.py").read_text(encoding="utf-8")
    assert '"--t-in", "1"' in src, "suite must pass t_in=1 (NS is Markovian)"
    # v2 wiring; default 4 since 2026-07-17 (deadline protocol: stride-1 retired for new training).
    # the protective intent is unchanged -- the suite must WIRE the documented value through, and
    # the documented value lives in the argparse default. The old literal '"--frame-stride", "4"'
    # assertion silently went stale when q's 9d0ba20 parameterized the flag (caught by CYB-3 when
    # freezing the transolver v2 config; the failure was pre-existing, not JSON-related).
    assert '"--frame-stride", str(args.train_frame_stride)' in src, \
        "suite must wire the train stride through --train-frame-stride"
    assert re.search(r'add_argument\("--train-frame-stride", type=int, default=4\b', src), \
        ("deadline protocol 2026-07-17 (user decree): train stride defaults to 4; stride-1 is "
         "permanently retired for NEW training (existing stride-1 rows kept; {1,4,8} bridge "
         "ablation on kf3/relam70 quantifies the delta). Do NOT flip back to 1 without a decree.")
    assert '"--frame-stride", "8"' in src, "suite must pass eval stride=8"
    assert src.count('"--per-frame-n", "20"') >= 2, "suite must pass per_frame_n=20 for train AND eval"
    assert "t_out_eval = 20" in src, "rollout eval must be AR 20 steps (1.0 T_L)"
    assert re.search(r'add_argument\("--epochs", type=int, default=25\)', src), \
        "suite --epochs must default to 25"
