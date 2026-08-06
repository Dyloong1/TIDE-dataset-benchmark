"""Judge-first tests for benchmark_metrics: known-answer synthetic fields.

Run: KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_benchmark_metrics.py -q
(or plain `python tests/test_benchmark_metrics.py` for a no-pytest smoke).
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_metrics as M  # noqa: E402


def test_nrmse_identity_zero():
    x = torch.randn(4, 8, 8, 8)
    assert M.nrmse(x, x).item() < 1e-6


def test_nrmse_known_ratio():
    true = torch.ones(1, 4, 4, 4)
    pred = torch.ones(1, 4, 4, 4) * 2  # diff norm == true norm -> nrmse == 1
    assert abs(M.nrmse(pred, true).item() - 1.0) < 1e-5


def test_nrmse_batched_shape():
    p = torch.randn(3, 4, 8, 8, 8)
    t = torch.randn(3, 4, 8, 8, 8)
    out = M.nrmse(p, t)
    assert out.shape == (3,)


def test_frmse_identity_zero():
    x = torch.randn(4, 16, 16, 16)
    b = M.frmse_bands(x, x)
    assert set(b) == {"low", "mid", "high"}
    assert all(v < 1e-5 for v in b.values())


def test_frmse_bands_isolate_freq():
    # a low-|k| perturbation lands in 'low'; a true high-|k| (diagonal Nyquist) lands in 'high'.
    N = 16
    z = torch.arange(N).float() * (2 * math.pi / N)
    true = torch.zeros(1, N, N, N)
    # low: single mode k=1 along one axis -> |k|=1 (low band)
    low_pert = torch.cos(z)[None, None, :].expand(1, N, N, N).clone()
    b_low = M.frmse_bands(low_pert, true)
    assert b_low["low"] > b_low["high"], f"low-freq pert: low {b_low['low']} > high {b_low['high']}"
    # high: Nyquist along ALL three axes -> |k|=sqrt(3)*N/2 (high band)
    gz, gy, gx = torch.meshgrid(z, z, z, indexing="ij")
    hi_pert = (torch.cos((N / 2) * gz) * torch.cos((N / 2) * gy) * torch.cos((N / 2) * gx))[None]
    b_hi = M.frmse_bands(hi_pert, true)
    assert b_hi["high"] > b_hi["low"], f"high-freq pert: high {b_hi['high']} > low {b_hi['low']}"


def test_crmse_energy_value():
    u = torch.ones(3, 4, 4, 4)
    # energy = 0.5 * sum_c(1) mean = 0.5*3 = 1.5
    assert abs(M.crmse(u, "energy").item() - 1.5) < 1e-5


def test_vorticity_of_uniform_is_zero():
    u = torch.ones(3, 8, 8, 8)  # constant field -> curl 0
    w = M.vorticity(u)
    assert w.abs().max().item() < 1e-4


def test_vorticity_known_single_mode():
    # CANONICAL convention (solver.operators.curl_hat): channels are (u_x,u_y,u_z)=(0,1,2) mapped
    # to spatial axes (z,y,x)=(-3,-2,-1). Put u_y = sin(x) (channel 1 varying along axis -1 = x).
    # Then ω = ∇×u:  ω_x = ∂u_z/∂y − ∂u_y/∂z = 0,  ω_y = ∂u_x/∂z − ∂u_z/∂x = 0,
    #   ω_z = ∂u_y/∂x − ∂u_x/∂y = ∂_x sin(x) = cos(x).
    N = 16
    x = torch.arange(N).float() * (2 * math.pi / N)
    uy = torch.sin(x)[None, None, :].expand(N, N, N)   # u_y = sin(x), x is axis -1
    u = torch.stack([torch.zeros(N, N, N), uy, torch.zeros(N, N, N)], dim=0)
    w = M.vorticity(u)
    expected_wz = torch.cos(x)[None, None, :].expand(N, N, N)
    assert (w[2] - expected_wz).abs().max().item() < 1e-3, "ω_z should be cos(x)"
    assert w[0].abs().max().item() < 1e-3 and w[1].abs().max().item() < 1e-3, "ω_x, ω_y should be 0"


def test_enstrophy_positive():
    u = torch.randn(3, 8, 8, 8)
    assert M.enstrophy(u).item() > 0


def test_rollout_nrmse_shape_and_zero():
    seq = torch.randn(5, 4, 8, 8, 8)
    out = M.rollout_nrmse(seq, seq)
    assert out.shape == (5,) and out.max().item() < 1e-5


def test_spectrum_l2_identity_zero():
    u = torch.randn(3, 16, 16, 16)
    assert M.spectrum_l2(u, u) < 1e-5


def test_decay_safe_skips_low_k():
    u = torch.randn(3, 8, 8, 8) * 0.01  # tiny KE
    # K0 large -> this frame below floor -> None
    assert M.nrmse_decay_safe(u, u, k0=100.0) is None
    # no k0 -> computes normally
    assert M.nrmse_decay_safe(u, u) is not None


def test_decay_safe_k0_from_initial_frame_actually_skips():
    """Consumer-side target test: the eval caliber computes k0 from the trajectory's FIRST frame
    (K0 = 0.5*mean(u0^2)) and threads it in. Verify a frame decayed to <5%*K0 IS skipped,
    and a frame near K0 is NOT — i.e. the --decay path is not a no-op (regression guard for the
    bug where eval called nrmse_decay_safe WITHOUT k0, making the skip unreachable)."""
    torch.manual_seed(0)
    u0 = torch.randn(3, 8, 8, 8)                      # initial frame
    k0 = float(0.5 * u0[:3].pow(2).mean())            # exactly the eval-side k0 formula
    decayed = u0 * 0.1                                # KE = 1% of K0 (< 5% floor) -> must skip
    healthy = u0 * 0.9                                # KE = 81% of K0 (> floor) -> must keep
    assert M.nrmse_decay_safe(decayed, decayed, k0=k0) is None
    assert M.nrmse_decay_safe(healthy, healthy, k0=k0) is not None


# --- P1 rollout: spectral drift + effective horizon (judge-first target tests) ---

def test_frequency_drift_identity_zero():
    u = torch.randn(3, 16, 16, 16)
    assert abs(float(M.frequency_drift(u, u))) < 1e-6


def test_spectral_centroid_shifts_down_when_highk_removed():
    torch.manual_seed(0)
    N = 16
    u = torch.randn(3, N, N, N)
    uh = torch.fft.rfftn(u, dim=(-3, -2, -1))
    kx = torch.fft.rfftfreq(N, d=1.0 / N).abs()
    kz = torch.fft.fftfreq(N, d=1.0 / N).abs()
    kmag = torch.sqrt(kz[:, None, None] ** 2 + kz[None, :, None] ** 2 + kx[None, None, :] ** 2)
    uh_lp = uh * (kmag <= N // 4)  # keep only low modes
    u_lp = torch.fft.irfftn(uh_lp, s=(N, N, N), dim=(-3, -2, -1))
    # centroid of low-pass field must be LOWER than full field
    assert float(M.spectral_centroid(u_lp)) < float(M.spectral_centroid(u))
    # drift(low-pass vs full) is negative (over-smoothed direction)
    assert float(M.frequency_drift(u_lp, u, signed=True)) < 0.0


def test_effective_prediction_time_interp():
    # error crosses 0.3 between step 2 (0.2) and step 3 (0.4) -> tau = 2 + (0.3-0.2)/0.2 = 2.5
    err = torch.tensor([0.05, 0.1, 0.2, 0.4, 0.6])
    tau = M.effective_prediction_time(err, threshold=0.3, dt=1.0)
    assert abs(tau - 2.5) < 1e-4
    # never crosses -> full horizon (T-1)*dt
    err2 = torch.tensor([0.01, 0.02, 0.03])
    assert abs(M.effective_prediction_time(err2, threshold=0.3) - 2.0) < 1e-9
    # first step already over -> 0
    err3 = torch.tensor([0.5, 0.6])
    assert M.effective_prediction_time(err3, threshold=0.3) == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} benchmark-metric tests passed")

def _solve_pressure_poisson(u):
    """Reference TRUE pressure of a velocity field u (3,N,N,N), via the SOLVER's own operator.

    This used to be a test-local reimplementation, and it silently encoded the SAME wrong
    component/axis pairing as the bug it was supposed to catch (it mapped u[0] to d/dz; the solver
    pairs u[0]=u_x with kx on axis -1). Two matching errors cancelled, so the test passed while
    pressure_poisson_residual was broken -- it scored 1.61 on the corpus's own D4-certified
    pressure. A reference that reimplements the thing under test is not an independent check:
    delegate to solver.operators.pressure_hat, which the D4 gate certifies.
    """
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from solver.grids import SpectralGrid
    from solver.operators import pressure_hat

    N = u.shape[-1]
    grid = SpectralGrid(N, device="cpu", dtype="fp32")
    u_hat = torch.fft.rfftn(u.float(), dim=(-3, -2, -1)).to(grid.cdtype)
    p_hat = pressure_hat(u_hat, grid)
    return torch.fft.irfftn(p_hat, s=(N, N, N), dim=(-3, -2, -1))


def test_pressure_poisson_residual_exact_is_zero():
    """The pressure that solves the Poisson equation gives a ~machine-zero relative residual;
    a WRONG-SCALE pressure (as if left normalized, absmax != velocity's) gives a large residual.
    This guards the P4 moat metric against the unit-mismatch bug (physical velocity vs normalized p)."""
    torch.manual_seed(0)
    N = 16
    # The field must be EXACTLY solenoidal, not "solenoidal-ish": pressure_hat derives p from
    # lap(p) = -d_i d_j(u_i u_j), an identity that only holds when div u = 0. The old version built
    # a few random modes with NO projection, so its "exact" p did not actually solve the equation
    # the residual checks -- the test only passed because its hand-rolled reference reproduced the
    # metric's own axis bug and the two errors cancelled.
    from solver.grids import SpectralGrid
    from solver.operators import leray_project_

    grid = SpectralGrid(N, device="cpu", dtype="fp32")
    uh = torch.fft.rfftn(torch.randn(3, N, N, N), dim=(-3, -2, -1)).to(grid.cdtype)
    kk = (grid.kx ** 2 + grid.ky ** 2 + grid.kz ** 2).sqrt()
    uh *= torch.exp(-(kk / 2.0) ** 2)          # keep it smooth: fp32 FFTs lose digits at high k
    leray_project_(uh, grid)
    u = torch.fft.irfftn(uh, s=(N, N, N), dim=(-3, -2, -1))
    p_true = _solve_pressure_poisson(u)
    r_true = M.pressure_poisson_residual(u, p_true)
    # Threshold is fp32 round-off, not machine zero: the residual re-derives second derivatives
    # through forward+inverse fp32 FFTs, which costs several digits. Measured here ~2.4e-3; real
    # corpus frames (energy at low k, fp32 storage) measure 0.0159. The old 1e-4 was calibrated on
    # a setup where the reference reproduced the metric's own axis bug, so both were wrong in the
    # same direction and the difference looked like zero. What matters is the SEPARATION from a
    # wrong p (asserted next): correct ~1e-2 vs wrong ~1e0 is two orders of magnitude.
    assert r_true < 0.05, f"exact p should satisfy Poisson (residual ~0), got {r_true}"
    # a mis-scaled p (e.g. a normalized field scored against physical velocity) must NOT pass
    r_bad = M.pressure_poisson_residual(u, p_true * 0.3)
    assert r_bad > 0.3, f"wrong-scale p must give a large residual, got {r_bad}"

def test_vorticity_analytic_single_mode():
    """ω = ∇×u on an analytic single-mode field, known answer. For u_x = sin(y) (channel 0 varying
    along axis -2), the only nonzero vorticity component is ω_z = ∂u_y/∂x − ∂u_x/∂y = −cos(y).
    This pins the curl's axis/channel pairing — the metric previously had a scrambled pairing that
    silently produced a wrong omega (flipping helicity's sign), which no test caught. Judge-first."""
    N = 16
    y = torch.arange(N).float() * 2 * math.pi / N
    u = torch.zeros(3, N, N, N)
    u[0] = torch.sin(y)[None, :, None]          # u_x = sin(y)
    w = M.vorticity(u)
    assert w[0].abs().max() < 1e-4, "omega_x should be ~0"
    assert w[1].abs().max() < 1e-4, "omega_y should be ~0"
    # omega_z = -cos(y): amplitude 1, and matches -cos(y) pointwise
    wz_expected = -torch.cos(y)[None, :, None].expand(N, N, N)
    assert (w[2] - wz_expected).abs().max() < 1e-4, "omega_z should equal -cos(y)"


def test_vorticity_matches_solver_curl():
    """Cross-check M.vorticity against solver.operators.curl_hat on a random smooth field, same
    channel/axis layout — they must agree to fp32 precision (the metric IS the solver's curl)."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    try:
        from solver.operators import curl_hat
        from solver.grids import SpectralGrid
    except Exception:
        import pytest
        pytest.skip("solver not importable")
    N = 16
    torch.manual_seed(3)
    uh = [torch.fft.rfftn(torch.randn(N, N, N), dim=(-3, -2, -1)) for _ in range(3)]
    kz = torch.fft.fftfreq(N, 1. / N); kx = torch.fft.rfftfreq(N, 1. / N)
    k2 = (kz[:, None, None]**2 + kz[None, :, None]**2 + kx[None, None, :]**2)
    u = torch.stack([torch.fft.irfftn(uh[i] * torch.exp(-k2), s=(N,)*3, dim=(-3, -2, -1))
                     for i in range(3)], dim=0)
    grid = SpectralGrid(N, device="cpu", dtype="fp32")
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1)).to(torch.complex64)
    w_solver = torch.fft.irfftn(curl_hat(u_hat, grid), s=(N,)*3, dim=(-3, -2, -1))
    w_metric = M.vorticity(u)
    assert (w_solver - w_metric).abs().max() < 1e-4, "vorticity must match solver curl_hat"


# --- energy_spectrum: the only metric that had NO test, and where the clamp bug hid -------

def test_energy_spectrum_parseval():
    """Sum of the shell spectrum must equal total KE = <|u|^2>/2 (Parseval).
    Modes outside the resolved sphere (|k| > N/2) must be DISCARDED, not folded into the
    last bin, so the sum is over the resolved sphere only and cannot exceed total KE."""
    torch.manual_seed(0)
    N = 16
    u = torch.randn(3, N, N, N)
    _ks, E = M.energy_spectrum(u)
    ke = 0.5 * u.pow(2).sum(dim=0).mean()
    # E covers k=1..N/2 (the resolved sphere); it must never EXCEED total KE.
    assert E.sum().item() <= ke.item() * (1 + 1e-5), (
        f"spectrum sum {E.sum():.6f} exceeds total KE {ke:.6f} — "
        "modes outside |k|<=N/2 are being folded in rather than discarded")


def test_energy_spectrum_single_mode_lands_in_one_bin():
    """A single Fourier mode at |k|=4 must put all its energy in bin 4 and none elsewhere."""
    N = 16
    u = torch.zeros(3, N, N, N)
    z = torch.arange(N).float()
    u[0] = torch.sin(2 * math.pi * 4 * z / N)[:, None, None].expand(N, N, N)
    ks, E = M.energy_spectrum(u)
    peak = int(torch.argmax(E).item())
    assert ks[peak].item() == 4, f"single k=4 mode landed in bin {ks[peak].item()}"
    others = E.clone(); others[peak] = 0
    assert others.max().item() < E[peak].item() * 1e-6, "energy leaked outside the k=4 shell"


def test_energy_spectrum_last_bin_not_inflated():
    """White noise has equal energy per MODE, so E(k) divided by the mode count of shell k
    must be flat across all shells. Raw E(k) is NOT flat and need not decrease at k=N/2 (that
    shell still holds more modes than its neighbour), so per-mode energy is the invariant with
    no free assumptions. Clamping |k|>N/2 into the last bin dumps ~44% of the grid there and
    spikes its per-mode energy ~6x; this test pins that bin to its neighbour."""
    torch.manual_seed(1)
    N = 16
    u = torch.randn(3, N, N, N)
    _ks, E = M.energy_spectrum(u)
    kz = torch.fft.fftfreq(N, d=1.0 / N)
    kmag = torch.sqrt(kz[:, None, None]**2 + kz[None, :, None]**2 + kz[None, None, :]**2)
    kmag = kmag.round().long()
    counts = torch.tensor([float((kmag == k).sum()) for k in range(1, N // 2 + 1)])
    per_mode = E / counts
    ratio = (per_mode[-1] / per_mode[-2]).item()
    assert 0.5 < ratio < 2.0, (
        f"last-shell energy per mode is {ratio:.2f}x its neighbour — out-of-sphere modes are "
        "being clamped into the last bin instead of discarded")
    spread = (per_mode.std() / per_mode.mean()).item()
    assert spread < 0.15, f"per-mode energy not flat for white noise (CoV={spread:.3f})"


def test_sgs_energy_transfer_is_scale_sensitive():
    """Guard mirroring the Poisson wrong-scale test: Pi = -tau_ij S_ij is LINEAR in the
    velocity scale, so feeding a NORMALIZED velocity (per-config absmax) instead of the
    physical one silently rescales Pi and makes it non-comparable across configs. This pins
    the scaling law so eval_benchmark must de-normalize before calling (as the P4 path does).
    Pearson correlation and the sign-based backscatter fraction are scale-invariant, so they
    are NOT affected -- this test documents which of the three outputs carries units."""
    torch.manual_seed(3)
    N = 8
    tau = torch.randn(6, N, N, N)
    u = torch.randn(3, N, N, N)
    base = M.sgs_energy_transfer(tau, u)
    scaled = M.sgs_energy_transfer(tau, u * 0.25)
    assert abs(scaled["Pi_mean"] - 0.25 * base["Pi_mean"]) < 1e-5 * max(abs(base["Pi_mean"]), 1e-8), (
        "Pi_mean must scale linearly with velocity -- it carries units and the caller MUST "
        "pass physical-space velocity")
    # sign-based fraction is scale-invariant (positive scaling cannot flip the sign of Pi)
    assert abs(scaled["backscatter_frac"] - base["backscatter_frac"]) < 1e-6


def test_frmse_bands_is_scale_sensitive():
    """fRMSE carries units, so eval MUST de-normalize before scoring it.

    frmse_bands returns an ABSOLUTE RMS of FFT differences (no division by the truth), unlike
    nRMSE. Scored on normalized tensors it reports physical/absmax, and absmax is per-config
    (2.50 on kf4 vs 4.35 on rotating_v2), so the same physical error prints ~1.74x differently
    across configs and a Tier-1 headline metric stops being comparable. This pins the scaling law
    that forces the de-normalization, mirroring the Poisson and SGS-Pi guards.
    """
    torch.manual_seed(5)
    p = torch.randn(3, 16, 16, 16)
    t = torch.randn(3, 16, 16, 16)
    base = M.frmse_bands(p, t)
    scaled = M.frmse_bands(p * 3.0, t * 3.0)
    for band in ("low", "mid", "high"):
        assert abs(scaled[band] - 3.0 * base[band]) < 1e-3 * max(base[band], 1e-8), (
            f"frmse_bands['{band}'] must scale linearly with the field -- it carries units, so "
            "the caller must pass PHYSICAL-space tensors")


def test_grad_spectral_axis_order_matches_solver():
    """_grad_spectral must return [d/dx, d/dy, d/dz] to match the velocity CHANNEL order.

    The solver is authoritative: operators.curl_hat takes "channels x,y,z" and grids.py puts kx on
    axis -1, kz on axis -3. So u[0]=u_x pairs with d/dx = axis -1. Callers index grads[i][j] as
    "d u_i / d x_j" and contract i with j, so a transposed order silently corrupts every
    contraction. This is checked via the divergence of a KNOWN solenoidal field: div(u) must be ~0
    only when the pairing is right (measured 0.27 vs 2.8e-12 on a corpus frame under the two
    orders). This is the 4th bug of this family (Poisson units, vorticity curl, fRMSE units, this),
    hence the test.
    """
    N = 16
    x = torch.arange(N).float() * (2 * math.pi / N)
    # A solenoidal field: u = (sin(y), sin(z), sin(x)) -> du_x/dx = du_y/dy = du_z/dz = 0.
    ux = torch.sin(x)[None, :, None].expand(N, N, N)   # varies along y (axis -2)
    uy = torch.sin(x)[:, None, None].expand(N, N, N)   # varies along z (axis -3)
    uz = torch.sin(x)[None, None, :].expand(N, N, N)   # varies along x (axis -1)
    u = torch.stack([ux, uy, uz], dim=0)
    grads = torch.stack([M._grad_spectral(u[i], N, u.device) for i in range(3)], dim=0)
    # div u = sum_i d u_i / d x_i  = grads[0][0] + grads[1][1] + grads[2][2]
    div = grads[0, 0] + grads[1, 1] + grads[2, 2]
    scale = grads.abs().max().clamp_min(1e-12)
    assert float(div.abs().max() / scale) < 1e-4, (
        "div(u) != 0 for a solenoidal field -- _grad_spectral's axis order does not match the "
        "velocity channel order (u[0]=u_x must pair with d/dx = axis -1, per solver curl_hat/grids)")


def test_poisson_residual_is_zero_for_the_exact_solution():
    """A p that SOLVES the Poisson equation must score ~0; a wrong p must score ~1.

    Guards the metric end-to-end: before the _grad_spectral axis fix this returned 1.61 for an
    exactly-correct pressure (and for the corpus's own D4-certified pressure), i.e. it failed a
    perfect answer.
    """
    # Build the field and its pressure with the SOLVER's own operators rather than by hand: a
    # hand-rolled rfft Leray projection is easy to get subtly wrong (Hermitian symmetry on the
    # half-spectrum), and a test whose "exact" answer is not exact tests nothing.
    solver = pytest.importorskip("solver.operators", reason="solver package not importable")
    from solver.grids import SpectralGrid  # noqa: E402

    torch.manual_seed(0)
    N = 32
    grid = SpectralGrid(N, device="cpu", dtype="fp32")
    uh = torch.fft.rfftn(torch.randn(3, N, N, N), dim=(-3, -2, -1)).to(grid.cdtype)
    # Damp the high modes before projecting. White noise puts most of its energy at the grid's
    # highest wavenumbers, where an fp32 forward+inverse FFT round-trip loses several digits, so a
    # white-noise "exact" solution scores a residual dominated by round-off (measured 7.8e-3) and
    # would force a meaninglessly loose threshold. Real corpus fields have their energy at low k
    # and score 0.0159. A smooth field reproduces that regime honestly.
    kk = (grid.kx ** 2 + grid.ky ** 2 + grid.kz ** 2).sqrt()
    uh *= torch.exp(-(kk / 3.0) ** 2)
    solver.leray_project_(uh, grid)                    # make it exactly solenoidal
    u = torch.fft.irfftn(uh, s=(N, N, N), dim=(-3, -2, -1))
    p = torch.fft.irfftn(solver.pressure_hat(uh, grid), s=(N, N, N), dim=(-3, -2, -1))
    r_exact = M.pressure_poisson_residual(u, p)
    r_zero = M.pressure_poisson_residual(u, torch.zeros_like(p))
    assert r_exact < 0.05, f"exact Poisson solution scored residual {r_exact:.4f}, must be ~0"
    assert r_zero > 0.5, f"a zero pressure scored {r_zero:.4f}, must be ~1 (metric not discriminating)"
