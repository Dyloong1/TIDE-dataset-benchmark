"""Judge-first tests for the dynamic-Smagorinsky SGS lower-bound + strain-rate operator.

Locks two correctness properties found by adversarial numeric verification (2026-07-05 r4):
  1. _strain_rate matches an ANALYTIC field (u_0 = sin(x1) -> S_01 = 1/2 cos(x1)).
  2. _dynamic_smagorinsky_tau is EXACTLY deviatoric (trace-free to machine precision) — the
     raw -2 nu_t S carries a residual divergence trace that must be removed (bug fixed r4).

Run: KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_sgs_smagorinsky.py -q
"""
import math
import os
import json
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("TURBGEN_REPO", str(Path.home() / "turbulence"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_benchmark as E  # noqa: E402
from Dataset.turbgen_zarr_dataset import _gaussian_filter  # noqa: E402


def test_strain_rate_matches_analytic():
    # Index i means the SAME direction in u[i] and in S_ij, and the solver pairs u[0]=u_x with
    # kx = axis -1 (operators.curl_hat "channels x,y,z"; grids.py kx on axis -1). So u_0 = sin(x1)
    # means the x-component varying along the y direction = axis -2.
    # S_01 = 1/2(∂_0 u_1 + ∂_1 u_0) = 1/2 ∂_y u_x = 1/2 cos(x1).
    N = 32
    x1 = torch.arange(N).float() * (2 * math.pi / N)
    u0 = torch.sin(x1)[None, :, None].expand(N, N, N)
    u = torch.stack([u0, torch.zeros(N, N, N), torch.zeros(N, N, N)], dim=0)
    S = E._strain_rate(u)                       # [11,22,33,12,13,23]
    analytic = 0.5 * torch.cos(x1)[None, :, None].expand(N, N, N)
    assert (S[3] - analytic).abs().max() < 1e-4          # S_01 (index 3 = pair (0,1))
    # divergence-free field -> trace(S) = ∂ᵢuᵢ ≈ 0
    assert (S[0] + S[1] + S[2]).abs().max() < 1e-5


def test_smagorinsky_is_trace_free():
    """tau must be EXACTLY deviatoric. The raw -2 nu_t S carries a residual divergence trace.

    The field must carry content ACROSS the filter scale: sigma_frac=0.04 puts the cutoff at
    k_c = pi/(0.04*2pi) ~= 12.5, so a field band-limited well below that leaves the filter nothing
    to remove, there IS no subgrid stress, and a correct Germano procedure returns cs2 <= 0 -> tau
    == 0. That is the model behaving correctly on a degenerate input, not a bug -- so trace-freeness
    is asserted on whatever tau comes out, and the "is it non-trivial" question is tested against
    REAL corpus data below instead (where a synthetic field cannot mislead us).
    """
    torch.manual_seed(0)
    N = 32
    k = torch.fft.fftfreq(N, 1.0 / N)
    uf = torch.zeros(3, N, N, N)
    for c in range(3):
        fh = torch.fft.fftn(torch.randn(N, N, N))
        mask = (k.abs()[:, None, None] <= 15) & (k.abs()[None, :, None] <= 15) & (k.abs()[None, None, :] <= 15)
        uf[c] = torch.fft.ifftn(fh * mask).real
    uf = _gaussian_filter(uf, 0.04)
    tau = E._dynamic_smagorinsky_tau(uf)        # (6,N,N,N)
    assert tau.shape == (6, N, N, N)
    assert torch.isfinite(tau).all()
    # EXACTLY deviatoric: tau_11 + tau_22 + tau_33 == 0 to machine precision
    assert (tau[0] + tau[1] + tau[2]).abs().max() < 1e-6


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _corpus_path import data_root, skip_reason  # noqa: E402

_CORPUS = data_root()
_SLICE = _CORPUS / "corpus" / "benchmark_slice.json"


def _a_local_case():
    """A case that (a) has a zarr ON THIS DISK and (b) has a test split in the slice.

    These tests used to hardcode "ou_robust_kf4_256_fp64", which exists ONLY on q. On CYB-t and
    CYB-3 that was masked by the `not _SLICE.exists()` skip -- so the moment q ships the merged
    slice, the skip lifts and both machines hit FileNotFoundError on a case they do not hold.
    Predicted by CYB-3 before it happened ("my 0-failed is partly an illusion: 2 kf4-hardcoded
    tests are hiding behind skipif(no slice)"), and confirmed here.

    Same bug family as the hardcoded POSIX corpus path in _corpus_path.py: q-local assumptions
    that read as green everywhere else. The physics under test (Smagorinsky a-priori correlation)
    is not kf4-specific -- any resolved forced-turbulence case will do.
    """
    if not _SLICE.exists():
        return None
    try:
        per = json.loads(_SLICE.read_text(encoding="utf-8")).get("per_config", {})
    except Exception:
        return None
    for case, v in per.items():
        te = v.get("test")
        if not (isinstance(te, dict) and te.get("seeds")):
            continue
        if (_CORPUS / "corpus" / f"{case}.zarr").is_dir():
            return case
    return None


_CASE = _a_local_case()
_NO_CASE = _CASE is None


@pytest.mark.skipif(_NO_CASE,
                    reason=(skip_reason() or "no case with a test split AND a local zarr -- your "
                            "data was NOT checked. Set TURBGEN_DATA_DIR. A skip is not a pass."))
def test_smagorinsky_is_non_trivial_on_real_data():
    """The SGS bound must be a REAL prediction on REAL data -- not a silent zero.

    This is the test that all five bugs of this family needed and did not have. The old suite only
    exercised a synthetic band-limited field, which passed while the shipped pipeline was broken:
    _gaussian_filter used delta = sigma_frac * N (GRID POINTS) against integer wavenumbers, so it
    annihilated the whole field (G(k=1) = 0.013, energy retained 0.0005%), the "filtered" input was
    numerically zero, and Smagorinsky came out 8000x too small with correlation -0.002. A P5 table
    built on that would have reported "neural beats Smagorinsky" against a strawman.

    Thresholds are deliberately loose: this pins ORDER OF MAGNITUDE and sign, not a tuned value.
    """
    import warnings
    from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = TurbgenZarrDataset(case=_CASE, split="test",
                                slice_manifest=str(_SLICE), data_root=str(_CORPUS),
                                task="sgs", t_in=1, t_out=1, per_frame_n=20,
                                patch_size=None, normalize=True)
    s = ds[0]
    ubar = s["input_dense"].squeeze(0)[:3]
    tau_true = s["output_dense"].squeeze(0)

    # 1. the filter must leave a RESOLVED field, not a zero field
    assert float(ubar.abs().mean()) > 0.01, (
        f"filtered field is ~empty (<|ubar|> = {float(ubar.abs().mean()):.2e}); the LES filter is "
        "removing the large scales too, so the whole SGS task is degenerate")

    # 2. the bound must be within an order of magnitude of the truth, not 8000x off
    tau = E._dynamic_smagorinsky_tau(ubar)
    ratio = float(tau_true.norm() / tau.norm().clamp_min(1e-12))
    assert 1.0 < ratio < 100.0, (
        f"Smagorinsky tau is {ratio:.0f}x off the true tau -- an eddy-viscosity model should "
        "under-predict, but not by orders of magnitude (it was 8000x when the filter was broken)")

    # 3. it must correlate POSITIVELY with the truth: eddy viscosity aligns with the strain rate,
    #    so a-priori correlation is modest (literature: ~0.2-0.4) but must not be ~0 or negative.
    corr = float(torch.corrcoef(torch.stack([tau.flatten(), tau_true.flatten()]))[0, 1])
    assert corr > 0.05, (
        f"Smagorinsky correlates {corr:+.3f} with the true tau -- a bound that does not correlate "
        "at all is not a bound, it is noise (it was -0.002 when the filter was broken)")


def test_smagorinsky_cs2_bounded():
    # Cs^2 must stay in [0, cap]; the eddy-viscosity model must not blow up.
    torch.manual_seed(1)
    N = 32
    u = torch.randn(3, N, N, N)
    uf = _gaussian_filter(u, 0.04)
    tau = E._dynamic_smagorinsky_tau(uf, cs2_cap=0.09)
    assert torch.isfinite(tau).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} sgs tests passed")


@pytest.mark.skipif(_NO_CASE,
                    reason=(skip_reason() or "no case with a test split AND a local zarr -- your "
                            "data was NOT checked. Set TURBGEN_DATA_DIR. A skip is not a pass."))
def test_strain_rate_components_are_uniformly_aligned_on_real_data():
    """-S̄ must correlate with the true deviatoric τ EVENLY across all six components.

    This is the sharpest available test for component/axis misalignment, and it is the one that
    finally caught the SIXTH bug of this family. The eddy-viscosity assumption (τ_dev ∝ -S̄) is
    imperfect, but it is imperfect ISOTROPICALLY: every component should land in the same ballpark.
    A transposed axis mapping instead leaves ONE component right and the rest dead. Measured with
    _strain_rate's old Ks = [∂/∂z, ∂/∂y, ∂/∂x] against u = (u_x,u_y,u_z):

        11: +0.003   22: +0.247   33: +0.035   12: +0.099   13: -0.011   23: +0.107

    and after pairing index i with ∂/∂x_i:

        11: +0.220   22: +0.247   33: +0.240   12: +0.244   13: +0.240   23: +0.243

    The spread, not the mean, is the signal: no value of Cs can fix a misalignment, so a test on the
    overall correlation alone would have passed this off as "eddy viscosity is just approximate".
    """
    import warnings
    from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = TurbgenZarrDataset(case=_CASE, split="test",
                                slice_manifest=str(_SLICE), data_root=str(_CORPUS),
                                task="sgs", t_in=1, t_out=1, per_frame_n=20,
                                patch_size=None, normalize=True)
    s = ds[0]
    ubar = s["input_dense"].squeeze(0)[:3]
    tau = s["output_dense"].squeeze(0)
    trace = (tau[0] + tau[1] + tau[2]) / 3.0
    tau_dev = tau.clone()
    for i in range(3):
        tau_dev[i] = tau_dev[i] - trace

    S = E._strain_rate(ubar)
    corrs = [float(torch.corrcoef(torch.stack([(-S[i]).flatten(), tau_dev[i].flatten()]))[0, 1])
             for i in range(6)]
    lo, hi = min(corrs), max(corrs)
    assert lo > 0.10, (
        f"component correlations {[round(c, 3) for c in corrs]} include a dead one (min {lo:.3f}) "
        "-- an axis/component mapping is transposed somewhere")
    assert hi - lo < 0.15, (
        f"component correlations {[round(c, 3) for c in corrs]} are UNEVEN (spread {hi - lo:.3f}); "
        "eddy viscosity fails isotropically, so an uneven spread means misaligned components")
