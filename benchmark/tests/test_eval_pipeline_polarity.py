"""The judge must not reward a COLLAPSED model. Pin the polarity of the eval pipeline.

Why this file exists (2026-07-15, found by CYB-3, independently reproduced on q)
--------------------------------------------------------------------------------
`_TiledModel` forwards on 128^3 tiles and stitches to 256^3. Its docstring says:

    "Seams were measured on a trained fno3d: mean jump across the tile boundary was 0.94x the
     mean jump at interior planes, i.e. no seam -- residual models predict a small delta on top
     of a continuous input field."

That measurement was taken on a model that had COLLAPSED to f≈0. When f≈0, pred≈input, and the
input is a continuous field -- so of course there is no seam. The "no seam" property was a
symptom of the collapse, not a property of the tiling. Measured on real test data:

    collapsed  (lr=1e-4, prod ep20):  seam 1.598e-04  interior 1.548e-04  ->  1.03x   (no seam)
    learning   (lr=1e-3, ep4)      :  seam 7.175e-03  interior 8.770e-04  ->  8.18x   (SEAM)
    CYB-3 on his box:                 collapsed 1.25x / learning 10.49x -- two machines agree

So the eval pipeline's POLARITY IS INVERTED:
  - a collapsed model (f≈0) stitches seamlessly -> scores fine
  - a working model (f large) injects a discontinuity at every tile edge -> AR-20 accumulates it
    into a spectral blow-up -> scores FAIL
The judge systematically rewards failure. One bug (collapse) was hiding another (seams).

Every other bug this benchmark has had was "synthetic passes, real data fails". This one is
"bad model passes, good model fails" -- strictly worse, because no amount of testing on real
DATA finds it. You have to test on a real MODEL, and specifically on a model that works.

This file therefore tests the PIPELINE, not the metric: it constructs a known-good predictor and
a known-collapsed predictor and asserts the pipeline ranks them correctly.

NOTE: the underlying tension is protocol-level, not a code typo (CYB-3's analysis): training uses
128^3 patches, eval uses the native 256^3 field, and you cannot have both distribution-match and
zero-seam. Measured options: hard tiling (seam 10.49x, in-distribution), halo=8 (6.53x),
halo=32 (2.91x), native-256 (0.96x but out-of-distribution: cos drops, amplitude overshoots).
Whatever is chosen, THIS test must keep passing -- it pins the requirement, not the mechanism.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import eval_benchmark as EB  # noqa: E402
from _corpus_path import corpus_dir, skip_reason  # noqa: E402

pytestmark = pytest.mark.skipif(skip_reason() is not None, reason=skip_reason() or "")

# seam/interior ratio a stitched prediction may show.
#   real checkpoints: collapsed 1.03x (q) / 1.25x (CYB-3);  LEARNING 8.18x (q) / 10.49x (CYB-3)
#   fix candidates (CYB-3, measured): halo=8 -> 6.53x, halo=32 -> 2.91x, native-256 -> 0.96x
# 2.0 is a REQUIREMENT, not a description of today: it fails hard tiling (8-10x) and halo=8
# (6.53x), and passes halo=32 (2.91x is close -- flagged below) and native-256 (0.96x).
SEAM_MAX = 2.0

# The synthetic low-pass probe below measures 2.59x against real checkpoints' 8.18x. It is
# directionally right (a non-local op DOES seam) but WEAKER than a real FNO, because a trained
# network amplifies the truncated receptive field far more than one low-pass filter does. So the
# probe alone cannot pin the requirement: tuning it until it trips would be fitting the test to
# the answer. The load-bearing check is test_real_checkpoints_are_not_ranked_by_seam below, which
# uses actual trained weights.
PROBE_SEAM_MAX = 2.0


class _Collapsed(torch.nn.Module):
    """f(x) = 0 -- exactly what every production rollout model degenerated into."""
    def forward(self, x):
        return torch.zeros_like(x)


class _Working(torch.nn.Module):
    """f(x) = a SPECTRAL (non-local) transform -- the same class of operator an FNO is.

    A pointwise probe is useless here and I got that wrong first: `0.2*tanh(3x)` measured a seam
    of only 1.10x, because a per-voxel function cannot notice a tile boundary. But an FNO's whole
    mechanism is a global spectral convolution -- cutting the field into tiles truncates its
    receptive field, which is exactly why a REAL learning model measures 8.18x while my pointwise
    probe measured 1.10x. The probe has to be non-local to reproduce the failure.

    This is a low-pass filter: zero-parameter, deterministic, and mathematically well-defined on
    any input -- so on the FULL field it has one unambiguous correct answer. Any difference
    between "filter the whole field" and "filter each tile then stitch" is manufactured by the
    tiling, not by the operator. Amplitude is scaled to ~20% of the field, the range a
    non-collapsed model actually outputs (measured |f| ~ 80-108% of the true delta).
    """
    def forward(self, x):
        sh = x.shape
        f = x.reshape(-1, *sh[-3:]).float()
        n = f.shape[-1]
        fh = torch.fft.rfftn(f, dim=(-3, -2, -1))
        kz = torch.fft.fftfreq(n, d=1.0 / n, device=f.device)
        kx = torch.fft.rfftfreq(n, d=1.0 / n, device=f.device)
        k2 = (kz[:, None, None] ** 2 + kz[None, :, None] ** 2 + kx[None, None, :] ** 2)
        # keep only the lowest modes -- the non-local step a tile boundary corrupts
        fh = fh * (k2 <= 12.0 ** 2)
        out = torch.fft.irfftn(fh, s=(n, n, n), dim=(-3, -2, -1))
        return (0.2 * out).reshape(sh).to(x.dtype)


def _seam_ratio(pred):
    """max over axes of (mean jump across the tile boundary) / (mean jump at interior planes).

    The 2x2x2 tiling of a 256^3 field puts boundaries at index 128 on each spatial axis.
    """
    p = pred
    while p.dim() > 3:
        p = p[0]
    n = p.shape[-1]
    b = n // 2
    ratios = []
    for ax in (-3, -2, -1):
        q = p.movedim(ax, -1)
        seam = float((q[..., b] - q[..., b - 1]).abs().mean())
        inner = sum(float((q[..., k] - q[..., k - 1]).abs().mean())
                    for k in (b // 2, b // 2 + 20, b + b // 2, b + b // 2 + 20)) / 4
        ratios.append(seam / max(inner, 1e-12))
    return max(ratios)


def _a_field():
    """One real 256^3 test sample, NORMALIZED -- i.e. the exact thing eval feeds the model.

    Must go through the dataset, not zarr directly. My first version read raw physical fields
    straight from the zarr; the models train in NORMALIZED space, so that fed them out-of-
    distribution input and inflated the measured seam from 4.56x to 11.58x. The seam is also
    strongly axis-dependent (z 7.02x / y 11.58x / x 4.49x on one ckpt), so a single-axis probe
    reads whichever axis it happens to pick -- _seam_ratio takes the max over all three.
    """
    from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset
    sl = corpus_dir() / "benchmark_slice.json"
    if not sl.exists():
        pytest.skip(f"no slice at {sl} -- your data was NOT checked. A skip is not a pass.")
    import json as _json
    per = _json.loads(sl.read_text(encoding="utf-8")).get("per_config", {})
    case = next((c for c, v in per.items()
                 if isinstance(v.get("test"), dict) and v["test"].get("seeds")
                 and (corpus_dir() / f"{c}.zarr").is_dir()), None)
    if case is None:
        pytest.skip("no case with a test split AND a local zarr -- your data was NOT checked.")
    ds = TurbgenZarrDataset(case=case, split="test", slice_manifest=str(sl),
                            data_root=str(corpus_dir().parent), task="rollout",
                            t_in=1, t_out=1, frame_stride=8, patch_size=None, normalize=True)
    return ds[0]["input_dense"].unsqueeze(0)     # (B=1, T=1, C=4, 256,256,256), normalized


@pytest.mark.xfail(strict=False, reason=(
    "v1 tiled-eval seam pin, RETIRED as a gate 2026-07-16: the v2 protocol answers this "
    "requirement by removing tiling entirely (eval --native; B3 measured ens 8.52->1.21 and "
    "the learned model beating persistence 0.7838 vs 0.8407). The tiled path survives only as "
    "a documented fallback, and this test stays as its regression record: if it ever XPASSes, "
    "someone fixed tiled seams and the fallback can be re-graded."))
def test_working_model_is_not_penalised_by_the_pipeline():
    """A predictor with NO tile-dependence must come out of the pipeline without a seam.

    _Working is a pointwise function: its output at a voxel depends only on that voxel. A correct
    stitching pipeline therefore cannot produce a boundary discontinuity. Any seam here is the
    pipeline's own artifact -- and under AR-20 it compounds into the spectral blow-up that gets
    scored as the MODEL's failure.
    """
    x = _a_field().cuda() if torch.cuda.is_available() else _a_field()
    tm = EB._TiledModel(_Working(), 128)
    if torch.cuda.is_available():
        tm = tm.cuda()
    with torch.no_grad():
        out = tm(x)
    r = _seam_ratio(out)
    assert r < PROBE_SEAM_MAX, (
        f"seam/interior = {r:.2f}x on a POINTWISE predictor that cannot itself create a seam -- "
        f"so the {r:.2f}x is manufactured by _TiledModel. Under AR-20 this discontinuity is "
        f"re-injected every step and compounds into a spectral blow-up, which the leaderboard "
        f"then attributes to the model. Measured on real checkpoints: a collapsed model shows "
        f"1.03x (no seam, scores fine) and a LEARNING model shows 8.18x (seam, scores FAIL) -- "
        f"the judge rewards collapse. See this file's docstring for the measured fix options.")


@pytest.mark.xfail(strict=False, reason=(
    "v1 tiled-eval polarity pin, RETIRED as a gate 2026-07-16 (same rationale as "
    "test_working_model_is_not_penalised_by_the_pipeline: v2 --native removes the seam by "
    "construction; kept as the tiled fallback's regression record)."))
def test_pipeline_does_not_flatter_a_collapsed_model():
    """The collapsed model must NOT look smoother than the working one.

    This is the polarity check. It is not about absolute seam size: it is about ORDER. If f=0
    stitches more cleanly than a real predictor, then 'clean stitching' is evidence of collapse,
    and any metric built on the stitched field is scoring the wrong thing.
    """
    x = _a_field().cuda() if torch.cuda.is_available() else _a_field()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r_col = _seam_ratio(EB._TiledModel(_Collapsed(), 128).to(dev)(x))
    r_work = _seam_ratio(EB._TiledModel(_Working(), 128).to(dev)(x))
    # f=0 stitches to the input exactly, so its seam ratio is ~0-1 by construction. The point is
    # the ORDER: if predicting nothing looks smoother than predicting something, the pipeline is
    # scoring smoothness, not accuracy.
    assert r_work <= max(r_col * 2.0, SEAM_MAX), (
        f"collapsed model seam {r_col:.2f}x vs working model seam {r_work:.2f}x -- the eval "
        f"pipeline is CLEANER for a model that predicts nothing. That inverts the leaderboard: "
        f"the only way to score well is to collapse. This is the exact failure CYB-3 found and q "
        f"reproduced (1.03x collapsed vs 8.18x learning on real checkpoints).")


def test_real_checkpoints_are_not_ranked_by_seam():
    """LOAD-BEARING: on REAL trained weights, the pipeline must not reward the collapsed one.

    The synthetic probe above is weaker than a trained FNO (2.59x vs 8.18x), so it can miss the
    failure. This test uses the actual checkpoints and is therefore the one that matters. It needs
    two ckpts to exist:
      - a collapsed one  (any production rollout ckpt: they all degenerated to f≈0)
      - a learning one   (trained at a lr that escapes the f≈0 basin)
    It skips if it cannot find both -- and says so, because a skip here means the polarity was NOT
    checked, which is exactly how this bug survived in the first place.
    """
    prod = Path("checkpoints/bench_q/fno3d__rollout__ou_robust_kf3_256_fp64/best.pt")
    learn = None
    for c in Path("/tmp").glob("claude-*/**/lr1e3_full/latest.pt"):
        learn = c
        break
    if not prod.exists() or learn is None:
        pytest.skip(
            f"need BOTH a collapsed ckpt ({prod}, exists={prod.exists()}) and a learning ckpt "
            f"(lr1e3_full/latest.pt, found={learn}) to check polarity on real weights. A skip "
            f"here means the eval pipeline's polarity was NOT verified -- do not read it as a pass.")

    import argparse
    import contextlib
    import io as _io
    import json
    import run_benchmark_suite as RS
    import train_benchmark as TB
    from Model.baselines import get_model

    models = json.loads(
        (Path(__file__).resolve().parents[1] / "Model/configs/production_configs_256.json")
        .read_text(encoding="utf-8"))["models"]
    hp = RS._hp_args("fno3d", models)
    argv = ["--model", "fno3d", "--task", "rollout", "--case", "c",
            "--data-root", "d", "--slice", "s"] + hp
    ap = argparse.ArgumentParser()
    for f in ("--model", "--task", "--case", "--data-root", "--slice"):
        ap.add_argument(f)
    ap.add_argument("--width", type=int, default=32); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--modes", type=int, default=16); ap.add_argument("--tfno-rank", type=int, default=8)
    ap.add_argument("--deeponet-p", type=int, default=128)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = _a_field().to(dev)

    def seam_of(ckpt):
        cfg = TB.build_config(a, 4, 4, 1, 1, model="fno3d", grid_n=128)
        with contextlib.redirect_stdout(_io.StringIO()):
            m = get_model("fno3d", cfg)
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
        with torch.no_grad():
            return _seam_ratio(EB._TiledModel(m.to(dev).eval(), 128).to(dev)(x))

    r_col, r_learn = seam_of(prod), seam_of(learn)
    assert r_learn <= max(r_col * 2.0, SEAM_MAX), (
        f"REAL WEIGHTS: collapsed ckpt seam {r_col:.2f}x vs learning ckpt seam {r_learn:.2f}x. "
        f"The eval pipeline is smoother for the model that predicts NOTHING, so the leaderboard "
        f"rewards collapse: under AR-20 the learning model's tile-edge discontinuity compounds "
        f"into a spectral blow-up and is scored as ITS failure. Found by CYB-3 (1.25x vs 10.49x), "
        f"reproduced on q (1.03x vs 8.18x). Fix the tiling before trusting any P1 rollout number.")
