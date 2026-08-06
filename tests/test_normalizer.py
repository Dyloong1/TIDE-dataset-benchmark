"""normalizer must be vetted before touching corpus frames — it produces the
PUBLISHED normalization constants, and the eval judges run in physical units, so a
wrong inverse silently corrupts every downstream metric. These tests pin: exact
round-trip (apply then inverse == identity), velocity normalized JOINTLY (one
scale for u/v/w, preserving the isotropy the dataset certifies), streaming stats
== single-batch stats, minmax really lands in [-1,1], standardize gives zero-mean
unit-var, freeze/JSON round-trip is idempotent, and degenerate channels don't
divide-by-zero."""
import solver._env  # noqa: F401
import torch
import pytest

from solver.normalizer import NormalizerFitter, FrozenNormalizer

torch.manual_seed(0)


def _frames(n=5, N=8):
    """Synthetic 4-channel frames; velocity zero-mean-ish, distinct pressure scale."""
    out = []
    for _ in range(n):
        u = torch.randn(3, N, N, N, dtype=torch.float64)
        p = torch.randn(N, N, N, dtype=torch.float64) * 0.3 + 1.0  # different scale+mean
        out.append((u, p))
    return out


@pytest.mark.parametrize("mode", ["minmax", "standardize"])
def test_round_trip_exact(mode):
    fit = NormalizerFitter()
    fr = _frames()
    for u, p in fr:
        fit.update(u, p)
    norm = fit.freeze(mode=mode)
    u, p = fr[0]
    un, pn = norm.apply_frame(u, p)
    u2, p2 = norm.inverse_frame(un, pn)
    assert torch.allclose(u, u2, atol=1e-12), "velocity round-trip not exact"
    assert torch.allclose(p, p2, atol=1e-12), "pressure round-trip not exact"


def test_velocity_is_joint_not_per_component():
    """One scale for all of u,v,w — feeding an anisotropic field must NOT rescale
    each component to the same range (that would manufacture isotropy)."""
    fit = NormalizerFitter()
    u = torch.zeros(3, 4, 4, 4, dtype=torch.float64)
    u[0] = 10.0    # u-component large
    u[1] = 1.0     # v-component small
    u[2] = -1.0
    fit.update(u)
    norm = fit.freeze(mode="minmax")
    un = norm.apply_frame(u)
    # joint absmax = 10 -> u-comp hits 1.0, v-comp stays 0.1 (NOT rescaled to 1.0)
    assert abs(float(un[0].max()) - 1.0) < 1e-12
    assert abs(float(un[1].max()) - 0.1) < 1e-12, "components were scaled independently"


def test_streaming_equals_batch():
    """Streaming accumulation must match stats over the concatenated batch."""
    fr = _frames(n=6, N=6)
    fit = NormalizerFitter()
    for u, p in fr:
        fit.update(u, p)
    norm = fit.freeze(mode="standardize")
    # reference: concat all velocity, all pressure
    allu = torch.cat([u.reshape(-1) for u, _ in fr])
    allp = torch.cat([p.reshape(-1) for _, p in fr])
    assert abs(norm.stats["velocity"]["mean"] - float(allu.mean())) < 1e-9
    assert abs(norm.stats["velocity"]["std"] - float(allu.std(unbiased=False))) < 1e-9
    assert abs(norm.stats["pressure"]["mean"] - float(allp.mean())) < 1e-9
    assert abs(norm.stats["pressure"]["std"] - float(allp.std(unbiased=False))) < 1e-9


def test_minmax_lands_in_unit_range():
    fit = NormalizerFitter()
    fr = _frames()
    for u, p in fr:
        fit.update(u, p)
    norm = fit.freeze(mode="minmax")
    for u, p in fr:
        un, pn = norm.apply_frame(u, p)
        assert float(un.abs().max()) <= 1.0 + 1e-12
        assert float(pn.abs().max()) <= 1.0 + 1e-12


def test_standardize_zero_mean_unit_var():
    fit = NormalizerFitter()
    fr = _frames(n=8, N=10)
    for u, p in fr:
        fit.update(u, p)
    norm = fit.freeze(mode="standardize")
    allu = torch.cat([norm.apply(u, "velocity").reshape(-1) for u, _ in fr])
    assert abs(float(allu.mean())) < 1e-9
    assert abs(float(allu.std(unbiased=False)) - 1.0) < 1e-6


def test_std_stable_large_mean_small_variance():
    """REGRESSION (deep review 2026-06-21): naive sumsq/n - mean^2 catastrophically
    cancels when mean^2 >> var and returns std=0 -> divide-by-eps garbage. Pressure
    has a large nonzero mean with small fluctuation, so this is a real corpus case.
    Welford must recover the true std."""
    fit = NormalizerFitter()
    true_std = 1e-3
    allp = []
    for _ in range(20):
        p = torch.randn(16, 16, 16, dtype=torch.float64) * true_std + 1e6
        allp.append(p.reshape(-1))
        fit.update(torch.randn(3, 16, 16, 16, dtype=torch.float64), p)
    norm = fit.freeze(mode="standardize")
    ref = float(torch.cat(allp).std(unbiased=False))
    got = norm.stats["pressure"]["std"]
    assert got > 0.5 * true_std, f"std collapsed to {got} (catastrophic cancellation)"
    assert abs(got - ref) / ref < 1e-3, f"std={got} vs batch ref={ref}"


def test_json_round_trip_idempotent(tmp_path):
    fit = NormalizerFitter()
    for u, p in _frames():
        fit.update(u, p)
    norm = fit.freeze(mode="minmax", note="unit-test")
    path = norm.to_json(tmp_path / "norm.json")
    norm2 = FrozenNormalizer.from_json(path)
    assert norm2.mode == norm.mode
    assert norm2.stats == norm.stats
    assert norm2.n_frames == norm.n_frames
    # and it still normalizes identically
    u, p = _frames(n=1)[0]
    a1, b1 = norm.apply_frame(u, p)
    a2, b2 = norm2.apply_frame(u, p)
    assert torch.allclose(a1, a2) and torch.allclose(b1, b2)


def test_three_channel_corpus_no_pressure():
    """A 3-channel corpus (no pressure) must freeze velocity-only, no crash."""
    fit = NormalizerFitter()
    for _ in range(3):
        fit.update(torch.randn(3, 6, 6, 6, dtype=torch.float64))
    norm = fit.freeze(mode="minmax")
    assert "velocity" in norm.stats and "pressure" not in norm.stats
    u = torch.randn(3, 6, 6, 6, dtype=torch.float64)
    assert torch.allclose(u, norm.inverse_frame(norm.apply_frame(u)), atol=1e-12)


def test_degenerate_channel_no_div_by_zero():
    """A constant channel (std=0, absmax=0) must not produce inf/nan."""
    fit = NormalizerFitter()
    u = torch.zeros(3, 4, 4, 4, dtype=torch.float64)   # all zero -> absmax=std=0
    fit.update(u)
    for mode in ("minmax", "standardize"):
        norm = fit.freeze(mode=mode)
        un = norm.apply_frame(u)
        assert torch.isfinite(un).all()


def test_empty_fit_raises():
    with pytest.raises(RuntimeError):
        NormalizerFitter().freeze()


def test_bad_velocity_shape_raises():
    with pytest.raises(ValueError):
        NormalizerFitter().update(torch.randn(4, 4, 4, 4))  # 4 != 3 components


def test_nan_inf_frame_rejected():
    """REGRESSION (deep review 2026-06-22): a single NaN/Inf frame in the 3000+
    frame corpus pass would silently set absmax=inf (-> all fields normalize to 0)
    or mean/std=NaN, corrupting the published constants with NO error. update() must
    fail loud at the offending frame."""
    fit = NormalizerFitter()
    u_nan = torch.randn(3, 4, 4, 4, dtype=torch.float64)
    u_nan[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        fit.update(u_nan)
    u_inf = torch.randn(3, 4, 4, 4, dtype=torch.float64)
    u_inf[1, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        fit.update(u_inf)
    # pressure path too
    with pytest.raises(ValueError):
        fit.update(torch.randn(3, 4, 4, 4, dtype=torch.float64),
                   torch.full((4, 4, 4), float("nan"), dtype=torch.float64))


def test_minmax_out_of_distribution_exceeds_unit_range():
    """DOCUMENTED behavior (not a bug): minmax is a FROZEN normalizer; a field with
    a larger extremum than any fit frame normalizes OUTSIDE [-1,1]. Pins the scope
    of the [-1,1] claim so downstream code never assumes a hard clamp on unseen
    data. The corpus is fit+applied on the same frames so this never bites in
    practice, but the guarantee must be honest."""
    fit = NormalizerFitter()
    fit.update(torch.ones(3, 4, 4, 4, dtype=torch.float64))   # absmax = 1.0
    norm = fit.freeze(mode="minmax")
    ood = torch.ones(3, 4, 4, 4, dtype=torch.float64) * 3.0    # 3x the fit extremum
    assert float(norm.apply_frame(ood).max()) > 1.0           # exceeds [-1,1] by design
    # inverse still exact regardless of range
    assert torch.allclose(ood, norm.inverse_frame(norm.apply_frame(ood)), atol=1e-12)


def test_fp32_round_trip_realistic_precision():
    """The corpus stores fp32 fields (corpus_to_zarr dtype f4). Stats accumulate in
    fp64 (good), but apply/inverse on fp32 input round-trips to ~fp32 eps (~1e-6),
    NOT 1e-12. Pin the realistic precision so no downstream code over-claims."""
    fit = NormalizerFitter()
    for _ in range(4):
        fit.update(torch.randn(3, 8, 8, 8, dtype=torch.float32))
    norm = fit.freeze(mode="minmax")
    u = torch.randn(3, 8, 8, 8, dtype=torch.float32)
    u2 = norm.inverse_frame(norm.apply_frame(u))
    err = float((u - u2).abs().max())
    assert err < 1e-5, f"fp32 round-trip {err} worse than expected"
    assert err > 0, "fp32 round-trip suspiciously exact (expected ~fp32 eps)"
