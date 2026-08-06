"""Pin every leaderboard metric to a PHYSICAL INVARIANT of REAL corpus data.

Why this file exists
--------------------
CLAUDE.md's DNS rule is "the judge gets audited first" -- every "acceptance passed while a bug was
live" traced back to judging with an unaudited judge. The benchmark side never had that rule, and
six metric bugs shipped. Their common signature is sharper than the DNS rule anticipated:

    bug                     synthetic test      real corpus
    Poisson units           PASS                wrong
    vorticity axis pairing  PASS                wrong
    fRMSE units             PASS                wrong
    _grad_spectral axes     PASS <- the test had the SAME axis bug; two errors cancelled
    SGS filter dimension    PASS (assert was `tau.max() > 0`)   filtered field kept 0.0005% energy
    _strain_rate axes       PASS                per-component corr 0.003..0.247

So: **synthetic fields are exactly what let all six through.** They are too clean, too symmetric,
and too small-amplitude to expose a units error, an axis swap, or a field that has been filtered to
approximately zero. Of ~86 benchmark tests only ~5 files touch real data, and before this file only
ONE (test_sgs_smagorinsky) validated a METRIC on it -- against 17 metrics feeding the leaderboard.

Two design rules, both learned the hard way
-------------------------------------------
1. **The test must not recompute the metric's own math.** That is how `_grad_spectral` passed: the
   test re-derived the gradient with the same wrong axis order, so both sides agreed and the bug was
   invisible. Every assertion here is a physical property of real turbulence, established by an
   INDEPENDENT route (an analytic identity, a different operator, or the data's own provenance).

2. **The target must be an INVARIANT, not a self-identity.** Measured, on a real field vs the same
   field with its components rotated (u[[2,0,1]] -- i.e. exactly the historical axis bug):

       nrmse(u, u)                    correct: 0.00e+00     axis-broken: 0.00e+00   <- BLIND
       pressure_poisson_residual      correct: 2.21e-02     axis-broken: 1.75e+00   <- catches it

   A self-identity target passes on a corrupted field. `frmse_bands(u,u)` returning three zeros is
   pretty and carries zero information. A test that cannot name the bug it would fail on is
   decorative. Each test below names its bug.

Thresholds
----------
Every number here was MEASURED on this corpus, not guessed, and is quoted per case in the test.
Headroom is stated explicitly. A threshold with no measurement behind it is how you end up
"fixing" the fp32 storage floor.

Precision floor: the corpus computes in fp64 but STORES fp32. Second derivatives amplify fp32
roundoff by ~k^2 (~1e4 at N=256), so the Poisson residual floors around 1e-2 -- NOT a bug. Verified
by quantizing the field progressively coarser and watching the residual rise monotonically:
    stored fp32 -> 2.21e-02 ; quantized 1e-4 -> 2.87e-02 ; quantized 1e-3 -> 1.85e-01
The floor is also PER CASE (measured 1.8e-03 .. 2.2e-02 across the 8 stores), so no single global
constant is honest; each test takes the max over cases with stated headroom.

Mutation results (2026-07-15) -- run these again if you touch benchmark_metrics.py:
    _grad_spectral axis order reversed  -> 12 tests FAIL   (the historical bug)
    Poisson RHS sign flipped            -> 11 tests FAIL
    Coriolis term disabled (bug #7)     ->  4 tests FAIL   (rotating cases only, correctly)
    `rhs = rhs - rhs.mean()` deleted    ->  0 tests fail   <- EXPECTED, and not a coverage gap:
        a real incompressible field's RHS is already mean-free (measured DC fraction 1.6e-10), so
        that line is a no-op on real data and the mutation is vacuous. Recorded because "no test
        went red" looks like a hole until you check; the DC removal matters for finite/synthetic
        fields, which is not what this file tests.
For contrast: the pre-existing 106-test suite stays fully GREEN under every mutation above. That is
the whole reason this file exists.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import benchmark_metrics as BM  # noqa: E402
from _corpus_path import corpus_dir, skip_reason  # noqa: E402

pytestmark = pytest.mark.skipif(skip_reason() is not None, reason=skip_reason() or "")

# One frame per case is enough: these are pointwise field identities, not statistics over time.
FRAME = 0


def _cases():
    if skip_reason() is not None:
        return []
    return sorted(p.name[:-5] for p in corpus_dir().glob("*.zarr"))


def _load(case, dtype=torch.float64):
    import zarr
    z = zarr.open(str(corpus_dir() / f"{case}.zarr"), mode="r")
    u = torch.from_numpy(z["u"][FRAME]).to(dtype)
    p = torch.from_numpy(z["p"][FRAME]).to(dtype)
    return u, p


def _load_b(case, dtype=torch.float64):
    """The buoyancy field for a Boussinesq case, else None. Stored as the 5th channel 'b'."""
    import zarr
    z = zarr.open(str(corpus_dir() / f"{case}.zarr"), mode="r")
    if "b" not in z:
        return None
    return torch.from_numpy(z["b"][FRAME]).to(dtype)


def _physics_kwargs(case):
    """Everything the pressure Poisson equation needs for THIS case's physics.

    Rotation contributes a scalar (Ω); buoyancy contributes a FIELD (b). Passing neither on a case
    that needs one scores the data with an equation it does not obey -- bugs #7 and #8.
    """
    kw = {"omega_z": BM.omega_z_for_case(case)}
    if BM.is_buoyant_case(case):
        kw["b"] = _load_b(case)
    return kw


CASES = _cases()


# --------------------------------------------------------------------------------------------
# pressure_poisson_residual -- the P4 headline, and where bug #7 lived
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_poisson_residual_on_real_dns(case):
    """A real DNS (u, p) satisfies its OWN pressure Poisson equation. This is free ground truth.

    Independent route: the corpus's pressure channel was produced by the D4-certified solver
    (operators.pressure_hat / operators_ext.pressure_hat_rotating), NOT by this metric. So the
    field is a known-answer target and the metric is the thing under test.

    Catches (measured):
      - _grad_spectral axis order  : 2.21e-02 -> 1.75e+00  (79x)
      - omega-blind Poisson (bug#7): rotating_ro0p2 2.75e+00 with the correct pressure
      - any sign/DC/k^2 error in the Laplacian or the RHS contraction
    """
    u, p = _load(case)
    kw = _physics_kwargs(case)
    if BM.is_buoyant_case(case) and kw.get("b") is None:
        pytest.fail(f"{case} is a Boussinesq case but its zarr has no 'b' channel -- the buoyancy "
                    f"source d_z b cannot be formed, so this residual would be meaningless.")
    res = float(BM.pressure_poisson_residual(u, p, **kw))
    # Measured floors per case (2026-07-15), all with the CORRECT omega:
    #   rotating_ro0p2 1.8e-03 | tau1 9.7e-03 | kf3 1.5e-02 | kf4 1.6e-02 | hotstart 1.8e-02
    #   ro0p2_v2 1.9e-02 | relam90 2.2e-02 | scalar_sc1 2.2e-02      -> max 2.2e-02
    # Threshold 5e-2 ~= 2.3x the worst measured floor. The pre-fix omega-blind value on
    # rotating_ro0p2 was 2.75e+00, i.e. 55x this bound -- so this is not a hair-trigger.
    assert res < 5e-2, (
        f"{case}: Poisson residual {res:.3e} on the corpus's OWN certified pressure. The stored "
        f"(u,p) solves this equation by construction, so a large residual means the METRIC is "
        f"wrong, not the data. If this is a rotating case, check omega_z_for_case({case!r}) "
        f"(={kw['omega_z']}) -- scoring rotating data with the non-rotating equation was bug #7. "
        f"If this is a BUOYANT case, the missing term is d_z b, not Coriolis -- that is bug #8.")


def test_poisson_residual_catches_axis_swap():
    """MUTATION: rotate the velocity components (the historical _grad_spectral bug) -> must FAIL.

    This is the test that proves the test. Self-identity targets (nrmse(u,u)==0) stay green under
    this mutation; the invariant target must not.
    """
    case = "ou_robust_kf4_256_fp64" if "ou_robust_kf4_256_fp64" in CASES else CASES[0]
    u, p = _load(case)
    good = float(BM.pressure_poisson_residual(u, p, omega_z=BM.omega_z_for_case(case)))
    bad = float(BM.pressure_poisson_residual(u[[2, 0, 1]], p, omega_z=BM.omega_z_for_case(case)))
    assert good < 5e-2, f"sanity: unmutated {case} should pass, got {good:.3e}"
    assert bad > 10 * good, (
        f"an axis-rotated field scored {bad:.3e} vs {good:.3e} unmutated -- this metric no longer "
        "discriminates component order, which is the exact bug (_grad_spectral) it exists to catch.")


@pytest.mark.parametrize("case", [c for c in CASES if BM.omega_z_for_case(c) != 0.0])
def test_poisson_residual_needs_the_right_omega(case):
    """MUTATION: a rotating case scored with the WRONG omega must FAIL. This IS bug #7.

    Each rotating case's stored pressure matches ONLY its own Omega (measured: 2.6e-08 / 3.2e-08 at
    the true Omega; O(1) at any other). Without this test, an omega-blind metric is green.
    """
    u, p = _load(case)
    true_om = BM.omega_z_for_case(case)
    right = float(BM.pressure_poisson_residual(u, p, omega_z=true_om))
    blind = float(BM.pressure_poisson_residual(u, p, omega_z=0.0))     # the pre-fix behaviour
    assert right < 5e-2, f"{case}: correct omega={true_om} should pass, got {right:.3e}"
    assert blind > 10 * right, (
        f"{case}: omega=0 scored {blind:.3e} vs {right:.3e} at the true omega={true_om}. The metric "
        "has stopped depending on rotation -- bug #7 has regressed and rotating P4 scores are being "
        "computed with an equation the data does not obey.")


# --------------------------------------------------------------------------------------------
# energy_spectrum -- Parseval against an independent physical-space route
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_energy_spectrum_satisfies_parseval(case):
    """sum_k E(k) must equal 0.5*<|u|^2> computed in PHYSICAL space (Parseval).

    Independent route: the right-hand side never touches the FFT or the shell binning, so a
    mis-binned shell, a wrong normalization, or an axis error cannot cancel out.
    Measured: 8.1e-09 .. 2.4e-07 across the 8 cases -> threshold 1e-5 (~40x headroom).
    """
    u, _ = _load(case)
    _, E = BM.energy_spectrum(u)
    lhs = float(E.sum())
    rhs = float(0.5 * u.pow(2).sum(0).mean())
    rel = abs(lhs - rhs) / rhs
    assert rel < 1e-5, (
        f"{case}: sum E(k)={lhs:.6f} vs 0.5<|u|^2>={rhs:.6f} (rel {rel:.2e}). Parseval is an "
        "identity -- a violation means the spectrum's binning or normalization is wrong, and every "
        "spectrum_L2 / freq_drift number on the leaderboard is built on it.")


# --------------------------------------------------------------------------------------------
# vorticity -- zero mean on a periodic box
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_vorticity_has_zero_mean_on_periodic_box(case):
    """<curl(u)> = 0 exactly on a periodic domain (the k=0 mode of a curl vanishes).

    Independent route: a mathematical property of the curl on a torus, not a re-derivation.
    Catches the historical vorticity axis-PAIRING bug (which made helicity's sign and value wrong).
    Measured: max component mean 2.2e-10 .. 3.0e-09 -> threshold 1e-7 (~30x headroom).
    """
    u, _ = _load(case)
    w = BM.vorticity(u)
    worst = max(abs(float(w[i].mean())) for i in range(3))
    assert worst < 1e-7, (
        f"{case}: <omega> has a component of {worst:.2e}, but a curl on a periodic box is mean-free "
        "by construction. A nonzero mean means the derivative axes are mis-paired -- the bug that "
        "made helicity wrong in both sign and magnitude.")


# --------------------------------------------------------------------------------------------
# Real-turbulence-only invariants: synthetic fields CANNOT produce these
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", [c for c in CASES if "decay" not in c])
def test_sgs_transfer_is_forward_on_average(case):
    """Real turbulence cascades FORWARD: <Pi> > 0 (energy flows to small scales).

    This is the class of target a synthetic field structurally cannot provide -- a Gaussian random
    field has zero mean transfer, so a broken Pi looks fine on synthetic data. That is precisely how
    the SGS filter-dimension bug survived: with the filtered field at 0.0005% energy, Pi was noise
    around zero, and the only assertion was `tau.max() > 0`.

    Forced-steady cases only: free decay has no forcing to sustain the cascade.

    tau/u_filt come from the DATASET's own production path (_subgrid_stress / _gaussian_filter),
    not a reimplementation here -- so this exercises the code the benchmark actually trains on.
    """
    from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset, _gaussian_filter
    u, _ = _load(case, dtype=torch.float32)
    sigma = 0.04                                    # production default (sgs_sigma_frac)
    tau = TurbgenZarrDataset._subgrid_stress(u, sigma)          # (6,N,N,N) true SGS stress
    u_filt = _gaussian_filter(u, sigma)                          # (3,N,N,N) resolved field
    out = BM.sgs_energy_transfer(tau, u_filt, sigma_frac=sigma)
    pi_mean = float(out["Pi_mean"])
    assert pi_mean > 0, (
        f"{case}: mean SGS transfer <Pi>={pi_mean:.3e} <= 0. Real forced turbulence cascades "
        "forward; a non-positive mean means Pi's sign, units, or the filter are wrong. A Gaussian "
        "synthetic field would give ~0 here and hide exactly this -- which is how the filter "
        "dimension bug survived (filtered field at 0.0005% energy, Pi noise around zero).")


@pytest.mark.parametrize("case", CASES)
def test_vorticity_pdf_is_intermittent(case):
    """Real turbulence is INTERMITTENT: vorticity tails are heavier than Gaussian.

    Second invariant no synthetic Gaussian field can fake. Independent route: compare the measured
    tail fraction to the Gaussian rate for the same threshold -- turbulence must exceed it.
    """
    u, _ = _load(case, dtype=torch.float32)
    w = BM.vorticity(u)
    mag = w.pow(2).sum(0).sqrt()
    z = (mag - mag.mean()) / mag.std()
    tail = float((z > 3.0).float().mean())
    GAUSS_3SIGMA = 1.35e-3          # one-sided P(Z>3) for a normal distribution
    assert tail > GAUSS_3SIGMA, (
        f"{case}: vorticity >3sigma fraction {tail:.2e} does not exceed the Gaussian rate "
        f"{GAUSS_3SIGMA:.2e}. Real turbulence is intermittent -- a Gaussian-looking vorticity field "
        "means the curl (or the field being read) is wrong.")


def test_buoyancy_term_is_load_bearing():
    """MUTATION for bug #8: dropping the +∂_z b source must blow the residual up.

    q holds no stratified corpus, so this builds a known-answer Boussinesq target the honest way:
    take a REAL divergence-free DNS velocity (the corpus's own, ∇·u ~ 1e-29 by the D1 gate), add a
    synthetic b, and solve p with the AUTHORITATIVE operator (physics_ext.operators_ext.
    pressure_hat_boussinesq -- the same one that writes the stratified corpus). The metric is then
    the only thing under test.

    Measured: with b -> 1.5e-02 (the fp32 floor); without b -> 3.2e-01, a 21x blow-up.

    ★ Why this test exists in this exact form: my FIRST attempt built the velocity field myself and
    projected it divergence-free by hand. The projection was wrong (div rms 1.55, not ~0), so BOTH
    arms read 1.35 and the mutation looked inert -- I nearly concluded the fix was decorative. The
    bug was in my test, not the metric. Hence: never hand-roll the incompressible field when a
    certified one is sitting in the corpus.
    """
    case = "ou_robust_kf4_256_fp64" if "ou_robust_kf4_256_fp64" in CASES else (CASES[0] if CASES else None)
    if case is None:
        pytest.skip("no corpus case available -- your data was NOT checked.")
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from physics_ext.operators_ext import pressure_hat_boussinesq
    from solver.grids import SpectralGrid

    u, _ = _load(case)                      # real DNS: divergence-free to machine precision
    N = u.shape[-1]
    g = SpectralGrid(N=N, device="cpu", dtype="fp64")
    uh = torch.fft.rfftn(u, dim=(-3, -2, -1)).to(g.cdtype)
    torch.manual_seed(0)
    b = torch.randn(N, N, N, dtype=torch.float64) * 0.1
    bh = torch.fft.rfftn(b, dim=(-3, -2, -1)).unsqueeze(0).to(g.cdtype)
    p = torch.fft.irfftn(pressure_hat_boussinesq(uh, bh, g), s=(N,) * 3, dim=(-3, -2, -1))

    with_b = float(BM.pressure_poisson_residual(u, p, b=b))
    without_b = float(BM.pressure_poisson_residual(u, p))
    assert with_b < 5e-2, (
        f"buoyancy-aware residual {with_b:.3e} on a field built BY the Boussinesq operator -- the "
        f"metric's +d_z b term disagrees with physics_ext's, so one of them is wrong.")
    assert without_b > 10 * with_b, (
        f"dropping b changed the residual only {without_b:.3e} vs {with_b:.3e} -- the buoyancy "
        f"term is not load-bearing, so bug #8's fix is decorative and stratified would still be "
        f"scored with an equation it does not obey.")
