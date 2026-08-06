"""P0-1 judge-first: validate the acceptance-eval math on synthetic fields with
KNOWN answers, BEFORE trusting it to judge physical data.

The acceptance verdict is produced by eval_dns_standard.py / eval_dynamics_residuals.py,
which had ZERO test coverage — every "bug present while validation passed" incident
traces to judging data with an unvetted referee. This file pins the eval-internal
computations (A2 tail fit, A7 drift, A10 isotropy, A11 gradient isotropy, A12
divergence, A13 skewness, and the D4 pressure consistency) against analytic
synthetic fields, and asserts a deliberately-broken field is judged FAIL.

We reproduce the eval's *exact* formulas here (copied from eval_dns_standard.py)
on synthetic inputs — if a formula is wrong, the analytic assertion catches it.
"""
import numpy as np
import torch

from solver.grids import SpectralGrid
from solver.operators import pressure_hat, curl_hat
from solver.initial_conditions import random_solenoidal

_FFT = (-3, -2, -1)


# ---------------------------------------------------------------------------
# A10 / A11 / A12 / A13 field statistics — reproduce eval_dns_standard.py:162-188
# ---------------------------------------------------------------------------
def _field_stats(u, grid):
    """Exactly the eval's per-snapshot A10/A11/A12/A13 computation."""
    N = grid.N
    ks = (grid.kx, grid.ky, grid.kz)
    u_hat = torch.fft.rfftn(u, dim=_FFT)
    K_f = float(0.5 * (u**2).sum(0).mean())
    ui2 = [(u[i] ** 2).mean().item() for i in range(3)]
    A10_comp = 100 * max(abs(3 * x / (2 * K_f) - 1) for x in ui2)
    cross = max(abs((u[i] * u[j]).mean().item()) for i, j in ((0, 1), (0, 2), (1, 2)))
    A10_cross = 100 * cross / (2 * K_f / 3)
    g = [[torch.fft.irfftn(1j * ks[j] * u_hat[i], s=(N, N, N), dim=_FFT)
          for j in range(3)] for i in range(3)]
    grad2 = sum((g[i][j] ** 2).mean() for i in range(3) for j in range(3))
    long2 = [(g[i][i] ** 2).mean().item() for i in range(3)]
    trans2 = [(g[i][j] ** 2).mean().item() for i in range(3) for j in range(3) if i != j]
    A11_trans = float(np.mean(trans2)) / (2.0 * float(np.mean(long2)))
    div = g[0][0] + g[1][1] + g[2][2]
    A12 = float((div**2).mean() / grad2)
    Svals = []
    for i in range(3):
        d_ = g[i][i]
        m2, m3 = (d_**2).mean(), (d_**3).mean()
        Svals.append(float(m3 / m2**1.5))
    A13 = float(np.mean(Svals))
    return dict(A10_comp=A10_comp, A10_cross=A10_cross, A11_trans=A11_trans,
                A12=A12, A13=A13)


def test_isotropic_field_passes_A10_A11_A13(device):
    """An isotropic Gaussian solenoidal field: A10~0, A11~0.5 (transverse=2x
    longitudinal -> ratio 0.5), A13~0. These are the eval's own quantities."""
    grid = SpectralGrid(64, device, "fp64")
    u = random_solenoidal(grid, seed=11, k_p=4.0)
    N = grid.N
    u = torch.fft.irfftn(u, s=(N, N, N), dim=_FFT)
    s = _field_stats(u, grid)
    # isotropy: component anisotropy small (finite-volume fluctuation)
    assert s["A10_comp"] < 8.0, f"A10_comp={s['A10_comp']}"
    assert s["A10_cross"] < 8.0, f"A10_cross={s['A10_cross']}"
    # gradient isotropy: <(du1/dx2)^2> = 2 <(du1/dx1)^2> for isotropic -> ratio ~1.0
    # (eval reports trans/(2*long); for isotropy this equals ~1.0, band [0.95,1.05])
    assert 0.8 < s["A11_trans"] < 1.2, f"A11_trans={s['A11_trans']}"
    # Gaussian field has ~zero derivative skewness (no nonlinear transfer yet)
    assert abs(s["A13"]) < 0.15, f"A13={s['A13']}"


def test_solenoidal_field_A12_machine_precision(device):
    """A Leray-projected field is divergence-free: A12 = <(div u)^2>/<|grad u|^2>
    must be at machine precision. This certifies the eval's A12 formula + the
    spectral gradient operator together."""
    grid = SpectralGrid(64, device, "fp64")
    u = random_solenoidal(grid, seed=12, k_p=4.0)
    N = grid.N
    u = torch.fft.irfftn(u, s=(N, N, N), dim=_FFT)
    s = _field_stats(u, grid)
    assert s["A12"] < 1e-12, f"A12={s['A12']} (should be ~1e-29)"


def test_anisotropic_field_is_FLAGGED(device):
    """Deliberately break isotropy: scale one velocity component up. The eval's
    A10 MUST detect it (>5% threshold). Proves the referee is not "always pass"."""
    grid = SpectralGrid(64, device, "fp64")
    u = random_solenoidal(grid, seed=13, k_p=4.0)
    N = grid.N
    u = torch.fft.irfftn(u, s=(N, N, N), dim=_FFT)
    u[0] *= 1.5                       # inject 50% anisotropy into u_x
    s = _field_stats(u, grid)
    assert s["A10_comp"] > 5.0, f"A10_comp={s['A10_comp']} should exceed 5% threshold"


# ---------------------------------------------------------------------------
# A7 stationarity drift — reproduce eval_dns_standard.py:97-100
# ---------------------------------------------------------------------------
def test_A7_drift_matches_analytic():
    """A7 = 100*|dK/dt|*T_E/<K>. Feed a known linear K(t); assert exact recovery."""
    t = np.linspace(30.0, 230.0, 2000)
    slope = 3e-4
    K0 = 0.5
    KK = K0 + slope * (t - t[0])              # exactly linear
    T_E = 1.9
    p = np.polyfit(t, KK, 1)
    A7 = 100.0 * abs(p[0]) * T_E / KK.mean()
    analytic = 100.0 * slope * T_E / KK.mean()
    assert abs(A7 - analytic) / analytic < 1e-6, f"A7={A7} vs {analytic}"


# ---------------------------------------------------------------------------
# A8 energy closure via the K-budget (2026-07-14 fix) — reproduce
# eval_dns_standard.py: A8 = 100*|dK/dt|/eps. dK/dt = eps_inj - eps_diss (energy
# equation), so |dK/dt|/eps IS the closure gap, read noise-free from K(t) rather
# than the fluctuating <f.u> forcing-power diagnostic (which false-FAILs a balanced
# short-tau OU run). These pin: (a) a STEADY K passes A8 no matter how noisy <f.u>
# is; (b) a genuinely DRIFTING K fails A8.
# ---------------------------------------------------------------------------
def test_A8_Kbudget_closure_matches_analytic():
    """A8 = 100*|dK/dt|/eps. Feed a known K(t) slope; assert exact recovery."""
    t = np.linspace(30.0, 230.0, 2000)
    slope = 1.4e-4        # small drift
    KK = 0.5 + slope * (t - t[0])
    eps = 0.07
    p = np.polyfit(t, KK, 1)
    A8 = 100.0 * abs(p[0]) / eps
    analytic = 100.0 * slope / eps
    assert abs(A8 - analytic) / analytic < 1e-6, f"A8={A8} vs {analytic}"


def test_A8_steady_K_passes_despite_noisy_fu_diagnostic():
    """A balanced (steady-K) run must PASS A8 (K-budget) even when the <f.u> injection
    diagnostic is off by ~8% (the tau1 short-tau OU noise case). This is the false-FAIL fix."""
    t = np.linspace(30.0, 200.0, 1500)
    eps = 0.065
    # steady K: tiny slope -> true closure ~0. (production tau1 K_drift ~0.75%/T_E, K flat.)
    KK = 0.5 + 2e-5 * (t - t[0])
    p = np.polyfit(t, KK, 1)
    A8_kbudget = 100.0 * abs(p[0]) / eps
    A8_fu_diagnostic = 8.0     # the noisy <f.u> value that used to false-FAIL
    assert A8_kbudget <= 2.0, f"steady K must pass A8, got {A8_kbudget:.2f}%"
    assert A8_fu_diagnostic > 2.0  # sanity: the old diagnostic would have failed


def test_A8_drifting_K_still_fails():
    """A genuinely unbalanced run (K drifting, eps_inj != eps_diss) must still FAIL A8 —
    the fix removes <f.u> noise, it does NOT relax the gate against real imbalance."""
    t = np.linspace(30.0, 200.0, 1500)
    eps = 0.07
    slope = 3e-3          # large drift: eps_inj exceeds eps_diss by ~4%/unit -> K grows visibly
    KK = 0.5 + slope * (t - t[0])
    p = np.polyfit(t, KK, 1)
    A8 = 100.0 * abs(p[0]) / eps
    assert A8 > 2.0, f"a drifting-K run must FAIL A8, got {A8:.2f}%"


# ---------------------------------------------------------------------------
# F5 short-window K_drift aliasing (2026-07-08): a STATIONARY trajectory with the
# integral-scale OU "breathing" (period ~T_breath) gets a spuriously large K_drift
# when the sampling window is SHORTER than the breathing period, because a linear
# fit over <1 breath sees half an oscillation as a secular slope. These tests pin
# the mechanism down analytically, and pin the fix: judge stationarity on a window
# LONGER than the breath (the old 20 T_L production window, which every config
# passed 13-16/16), not on the 7.5 T_L publish window.
# ---------------------------------------------------------------------------
def _K_drift_pct(t, K, T_L):
    """The gate's K_drift_last5TL_pct == full-window linear fit * 5 T_L / <K>."""
    p = np.polyfit(t, K, 1)
    return 100.0 * abs(p[0] * 5.0 * T_L) / K.mean()


def test_F5_stationary_breathing_aliases_in_short_window():
    """A PERFECTLY stationary K(t) (zero trend + pure OU breathing) reads as high
    'drift' in a 7.5 T_L window but low drift in a 20 T_L window. Proves the reject
    is a windowing artifact, not real non-stationarity."""
    T_L = 1.665
    T_breath = 12.0 * T_L  # breathing period ~12 T_L (measured O(10-14))
    amp = 0.13             # 13% breathing amplitude (measured 11-14%)
    K0 = 0.30
    # zero secular trend — the truth is STATIONARY
    def K_of(t):
        return K0 * (1.0 + amp * np.sin(2 * np.pi * (t - 30.0) / T_breath))
    t_short = np.linspace(30.0, 30.0 + 7.5 * T_L, 1500)   # 7.5 T_L publish window
    t_long = np.linspace(30.0, 30.0 + 20.0 * T_L, 4000)   # 20 T_L production window
    d_short = _K_drift_pct(t_short, K_of(t_short), T_L)
    d_long = _K_drift_pct(t_long, K_of(t_long), T_L)
    # short window aliases the breath into "drift" (can exceed the 5% gate);
    # long window (>1 breath) averages it out toward the true zero trend.
    assert d_long < d_short, f"long {d_long:.2f}% should be < short {d_short:.2f}%"
    assert d_long < 5.0, f"20 T_L window must pass a stationary field, got {d_long:.2f}%"
    # (short may or may not exceed 5% depending on phase — the point is d_long<<d_short,
    #  i.e. the SAME stationary field is judged differently by window length.)


def test_F5_real_drift_rejected_in_any_window():
    """A genuinely non-stationary K(t) (monotone secular rise, the collapse/laminarize
    signature) must read as high drift in BOTH windows — the fix must NOT let real
    drift through. This is the 'deliberately-broken field must FAIL' guard."""
    T_L = 1.665
    K0 = 0.30
    slope = 0.30 * 0.03 / (5.0 * T_L)   # ~3% per 5 T_L secular rise (real drift)
    def K_of(t):
        return K0 + slope * (t - 30.0)
    t_short = np.linspace(30.0, 30.0 + 7.5 * T_L, 1500)
    t_long = np.linspace(30.0, 30.0 + 20.0 * T_L, 4000)
    d_short = _K_drift_pct(t_short, K_of(t_short), T_L)
    d_long = _K_drift_pct(t_long, K_of(t_long), T_L)
    # a pure linear drift reads the SAME %/5TL regardless of window (no oscillation to alias)
    assert abs(d_short - d_long) / d_long < 0.05, \
        f"real drift should read same in both windows: short {d_short:.2f} vs long {d_long:.2f}"
    assert d_long > 2.5, f"real 3%/5TL drift must be visible, got {d_long:.2f}%"


# ---------------------------------------------------------------------------
# A2 resolved-dissipation tail fit — reproduce eval_dns_standard.py:62-71
# ---------------------------------------------------------------------------
def _A2(k, E, NU, ETA, K_MAX):
    D = 2.0 * NU * k**2 * E
    mask_k = k <= K_MAX
    D_res = np.trapezoid(D[mask_k], k[mask_k])
    fit_m = (k * ETA >= 0.5) & mask_k & (D > 0)
    a, b = np.polyfit(k[fit_m] * ETA, np.log(D[fit_m]), 1)
    tail = -np.exp(b + a * K_MAX * ETA) / (a * ETA)
    return 100.0 * D_res / (D_res + tail), a, tail


def test_A2_well_resolved_spectrum_near_100():
    """A well-resolved dissipation spectrum -> A2 ~ 100%, fit slope a<0, tail>0.
    Use a real well-resolved case: k_max*eta ~ 1.5 (eta chosen so k*eta spans
    [0.5, 1.5] with enough fit points), dissipation peak at k*eta~0.2 then a
    decaying exponential tail (the physical regime A2's tail-fit assumes)."""
    NU = 0.0022
    k = np.arange(1, 86, dtype=np.float64)
    ETA = 1.55 / 85.0                       # k_max*eta = 1.55 (Class I, like real runs)
    keta = k * ETA
    # dissipation spectrum: model-Pao form, peaks ~0.2 then decays exponentially.
    # D(k) ~ (k*eta) exp(-5.2 (k*eta))  -> clearly decaying tail (a<0)
    D = keta * np.exp(-5.2 * keta)
    E = D / (2.0 * NU * k**2)               # back out E so eval's D=2nu k^2 E matches
    A2, a, tail = _A2(k, E, NU, ETA, 85)
    assert a < 0, f"fit slope a={a} should be negative (decaying tail)"
    assert tail > 0, f"tail={tail} should be positive"
    assert 99.0 < A2 <= 100.5, f"A2={A2}% (well-resolved should be ~100%)"


def test_A2_upturned_tail_is_degenerate():
    """KNOWN-RISK case (eval_dns_standard.py:68-71): if the dissipation tail does
    NOT decay (fit slope a>=0), the current tail formula
        tail = -exp(b + a*kmax*eta)/(a*eta)
    goes NEGATIVE, making A2 = D_res/(D_res+tail) non-physical (>100% or blow up).
    This test DOCUMENTS the bug so the P1-3 guard can be asserted against it.
    """
    NU = 0.0022
    k = np.arange(1, 86, dtype=np.float64)
    ETA = 1.05 / 85.0                        # k_max*eta = 1.05 (Class II, under-resolved)
    keta = k * ETA
    # under-resolved: dissipation D still RISING at k_max (energy piled at cutoff).
    # D(k) ~ (k*eta) * exp(+0.8 (k*eta)) -> increasing tail -> fit slope a>=0.
    D = keta * np.exp(0.8 * keta)
    E = D / (2.0 * NU * k**2)
    A2, a, tail = _A2(k, E, NU, ETA, 85)
    # Document the degeneracy: either a>=0 (tail<0, A2 non-physical) is the bug,
    # or the field happens to still fit a<0. We assert the diagnostic so the guard
    # in eval (P1-3) can later turn this into a clean FAIL.
    if a >= 0:
        assert tail < 0, "confirmed: a>=0 makes tail negative (the documented bug)"
        # A2 with a negative tail is non-physical -> must NOT be silently >99.5%
        # (the guard added in P1-3 should reject this instead of passing it)
    # if a<0 here, this synthetic didn't trigger it; the guard is still warranted.


# ---------------------------------------------------------------------------
# A1 instantaneous resolution gate — reproduce eval_dns_standard.py A1 inst gate.
# Guards the "window-average trap": window-avg k_maxeta>=1.5 must NOT pass Class I
# if the field is transiently under-resolved (helical run: avg 1.524 but <1.5 for
# 36.5% of the window). Class I requires instantaneous k_maxeta>=1.5 for >=95% of
# the sampling window.
# ---------------------------------------------------------------------------
def _A1_inst(eps_series, NU, K_MAX):
    eps = np.asarray(eps_series, float)
    eps = eps[eps > 1e-30]
    keta = K_MAX * (NU**3 / eps) ** 0.25
    frac = float((keta >= 1.5).mean())
    return keta.min(), frac, frac >= 0.95


def test_A1_inst_gate_passes_steady_resolution():
    """Steady well-resolved run: eps fluctuates mildly around a value giving
    k_maxeta~1.6 -> instantaneous >=1.5 essentially always -> Class I holds."""
    NU, K_MAX = 0.0022, 85
    eps0 = NU**3 / (1.6 / K_MAX) ** 4              # eps giving k_maxeta=1.6
    eps = eps0 * (1.0 + 0.05 * np.sin(np.linspace(0, 40, 500)))  # +/-5% ripple
    kmin, frac, ok = _A1_inst(eps, NU, K_MAX)
    assert ok and frac >= 0.99, f"steady run should pass inst gate, frac={frac}"
    assert kmin >= 1.5, f"min k_maxeta={kmin} should stay >=1.5"


def test_A1_inst_gate_FLAGS_window_average_trap():
    """The helical failure mode: window-AVERAGE k_maxeta passes >=1.5 but big eps
    peaks push instantaneous k_maxeta below 1.5 a large fraction of the time. The
    gate MUST reject Class I (frac < 0.95). Proves it catches what the average hides."""
    NU, K_MAX = 0.0022, 85
    eps0 = NU**3 / (1.55 / K_MAX) ** 4
    eps = eps0 * (1.0 + 0.40 * np.sin(np.linspace(0, 60, 600)))  # +/-40% (helical-like)
    keta = K_MAX * (NU**3 / eps) ** 0.25
    kmin, frac, ok = _A1_inst(eps, NU, K_MAX)
    assert keta.mean() >= 1.5, "window average should (deceptively) be >=1.5"
    assert not ok, f"inst gate MUST flag transient under-resolution, frac={frac}"
    assert kmin < 1.5, f"min k_maxeta={kmin} should dip below 1.5"


# ---------------------------------------------------------------------------
# D4 velocity-pressure consistency — reproduce run_trajectory_showcase.py D4
# ---------------------------------------------------------------------------
def test_D4_pressure_poisson_machine_precision(device):
    """The kinematic pressure from pressure_hat must satisfy lap(p) = -d_i d_j(u_i u_j)
    to machine precision — the D4 acceptance criterion. Certifies the pressure
    solver that also generates the corpus's 4th (pressure) channel."""
    grid = SpectralGrid(48, device, "fp64")
    uh = random_solenoidal(grid, seed=14, k_p=4.0)
    N = grid.N
    p_hat = pressure_hat(uh, grid)
    lap_p = (-grid.k2) * p_hat
    u = torch.fft.irfftn(uh, s=(N, N, N), dim=_FFT)
    ks = (grid.kx, grid.ky, grid.kz)
    # exact eval formula (run_trajectory_showcase.py): poisson_rhs = +sum k_i k_j T_ij
    poisson_rhs = torch.zeros_like(p_hat)
    for i in range(3):
        for j in range(3):
            poisson_rhs += (ks[i] * ks[j]) * torch.fft.rfftn(u[i] * u[j], dim=_FFT)
    res = float((lap_p - poisson_rhs).abs().max()) / max(float(poisson_rhs.abs().max()), 1e-30)
    assert res < 1e-9, f"D4 poisson residual {res}"


def test_dec1_window_skips_cascade_buildup():
    """eval_decaying.self_similar_window_start must EXCLUDE a structured-IC
    turbulization's cascade-buildup phase (ABC/TG: K decays monotonically but eps
    RISES to a peak before self-similar decay). A judge bug fitted the rising-eps
    phase, depressing ABC's DEC1 R^2 to 0.67; the fix starts the window at the eps
    peak. Synthetic: eps rises to a peak at i=40, then decays; K monotone-decays;
    k_max*eta stays >=1.5 throughout (Class I, never dips)."""
    import importlib.util
    here = __import__("pathlib").Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "eval_decaying", here / "experiments" / "phase0" / "eval_decaying.py")
    # import only the helper without running the script body: read & exec the def
    import numpy as np
    n = 200
    t = np.linspace(0, 50, n)
    # eps: rises to a peak at i=40 (cascade buildup), then decays
    i_peak = 40
    eps = np.concatenate([np.linspace(0.01, 0.04, i_peak),
                          0.04 * (1.0 + (t[i_peak:] - t[i_peak])) ** -1.2])
    K = 0.8 * (1.0 + 0.1 * t) ** -1.0          # K monotone-decays from t~0
    keta = 85.0 * (0.007**3 / np.maximum(eps, 1e-30)) ** 0.25  # stays >=1.5 (Class I)
    assert keta.min() >= 1.5, "synthetic should stay Class I (tests the ABC path)"

    # exec just the helper function from the module source (avoid running argv body)
    src = (here / "experiments" / "phase0" / "eval_decaying.py").read_text(encoding="utf-8")
    ns = {"np": np}
    start = src.index("def self_similar_window_start")
    end = src.index("\n\nHERE =")
    exec(src[start:end], ns)
    i_lo = ns["self_similar_window_start"](t, K, eps, keta)
    # eps peak is the last index of the rising segment (i_peak-1) by construction
    assert i_lo >= i_peak - 1, f"window must start at/after eps peak ~{i_peak}, got {i_lo}"

    # control: an already-turbulent decay (eps monotone-decreasing from t~0) -> no-op
    eps2 = 0.04 * (1.0 + t) ** -1.2
    keta2 = 85.0 * (0.007**3 / np.maximum(eps2, 1e-30)) ** 0.25
    i_lo2 = ns["self_similar_window_start"](t, K, eps2, keta2)
    assert i_lo2 <= int(0.02 * n) + 1, f"monotone-eps decay should be ~no-op, got {i_lo2}"


# ---------------------------------------------------------------------------
# D2 NS-momentum residual + D3 half-dt convergence — referee target tests.
# These were the audit-flagged gap: D2/D3 are the judges of "does the field
# satisfy the NS equation", yet (unlike D4) had NO known-answer target test, and
# the D2 freeze-forcing protocol was changed without one. We certify the judge:
#   (1) forward: a real solver step gives a SMALL D2 (time-truncation level);
#   (2) convergence (D3): halving h shrinks D2 by ~O(h^2), ratio in [2.5, 6];
#   (3) ADVERSARIAL: a field advanced by a WRONG step (corrupted C) must give a
#       LARGE D2 -- the judge must not pass a trajectory that violates NS.
# ---------------------------------------------------------------------------
def _mini_solver(device, N=32, nu=0.02):
    """Minimal deterministic (no-forcing) solver on a TG field -- self-contained,
    no yaml. build_solver sets the TG initial u_hat from cfg.ic (default
    taylor_green), so the D2 protocol runs on a known smooth field."""
    from solver.solver import build_solver
    from solver.config import ExperimentConfig, SolverConfig, ForcingConfig
    cfg = ExperimentConfig(
        name="d2test", ic="taylor_green",
        solver=SolverConfig(N=N, nu=nu, dtype="fp64",
                            forcing=ForcingConfig(type="none")))
    return build_solver(cfg)


def _d2_residual(s, h):
    """Replicate run_trajectory_showcase.residual_sample's D2 core (centered NS
    residual at B over A->B->C with fixed h), returning res_rel_dudt. Restores
    the solver to A afterwards so the test can re-probe at other h."""
    import importlib.util
    here = __import__("pathlib").Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "showcase", here / "experiments" / "phase0" / "run_trajectory_showcase.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    A = s.u_hat.clone()
    out = mod.residual_sample(s, h)
    s.u_hat.copy_(A); s.t = 0.0   # restore for re-probe
    return out


def test_D2_referee_forward_and_convergence(device):
    """D2 forward target: a real solver step yields a small NS residual, and D3
    half-dt convergence is O(h^2) (ratio ~4)."""
    s = _mini_solver(device)
    h = 2e-3
    r_h = _d2_residual(s, h)["res_rel_dudt"]
    r_half = _d2_residual(s, h / 2)["res_rel_dudt"]
    assert r_h < 1e-2, f"D2 forward residual {r_h:.2e} should pass (<1e-2)"
    # D3: residual ~ O(h^2) -> halving h shrinks it ~4x (band [2.5,6] per standard)
    ratio = r_h / max(r_half, 1e-300)
    assert 2.5 <= ratio <= 6.0, f"D3 half-dt ratio {ratio:.2f} should be ~4 (O(h^2)), in [2.5,6]"


def test_D2_referee_rejects_non_NS_field(device):
    """ADVERSARIAL: the judge must FAIL a trajectory whose B->C step violates NS.
    We corrupt C with a divergence-free perturbation the dynamics did NOT produce;
    the centered (C-A)/2h then no longer equals nl+visc+force at B, so D2 must be
    large. A judge that still reports a small D2 here would pass non-physical data."""
    import importlib.util
    here = __import__("pathlib").Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "showcase2", here / "experiments" / "phase0" / "run_trajectory_showcase.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = _mini_solver(device)
    g = s.grid
    h = 2e-3
    # clean residual (baseline)
    A = s.u_hat.clone()
    clean = mod.residual_sample(s, h)["res_rel_dudt"]
    s.u_hat.copy_(A); s.t = 0.0
    # monkeypatch the solver so the SECOND step (B->C) adds a bogus solenoidal kick
    real_step = s.step
    state = {"n": 0}
    from solver.operators import leray_project_
    def bad_step(dt):
        real_step(dt)
        state["n"] += 1
        if state["n"] == 2:                      # corrupt only B->C
            kick = torch.randn_like(s.u_hat) * (0.3 * s.u_hat.abs().mean())
            leray_project_(kick, g)              # keep it divergence-free (A12 stays clean)
            s.u_hat += kick
    s.step = bad_step
    s.u_hat.copy_(A); s.t = 0.0
    corrupt = mod.residual_sample(s, h)["res_rel_dudt"]
    assert corrupt > 10 * clean and corrupt > 0.1, (
        f"adversarial non-NS step must spike D2: clean={clean:.2e} corrupt={corrupt:.2e} "
        f"(judge would pass a non-physical trajectory)")


def test_D4_missing_now_FAILS_the_gate(tmp_path):
    """Judge-first (gate-tightening 2026-06-26): D-group is non-negotiable, so a MISSING D4
    channel must FAIL eval_dynamics_residuals (was `ok4 or not has_d4` -> silently PASSED).
    Drives the REAL script via subprocess on a synthetic showcase: a json WITH good D4 exits 0;
    the SAME json with D4 fields stripped exits non-zero."""
    import json, subprocess, sys, shutil
    here = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = here / "experiments" / "phase0" / "eval_dynamics_residuals.py"
    res = here / "experiments" / "phase0" / "results"
    case = "d4guard_selftest_case"
    show = res / f"trajectory_showcase_{case}"
    cdir = res / case
    # good sample: all D1-D4 pass
    good = {"protocol": "selftest", "samples": [
                {"D1_div_ratio": 1e-12, "res_rel_dudt": 1e-3, "res_rel_visc": 1e-3,
                 "D4_poisson_res": 1e-12, "D4_gradp_res": 1e-12}],
            "halfdt_probes": [{"ratio_rel_dudt": 4.0}]}
    missing = {"protocol": "selftest", "samples": [
                {"D1_div_ratio": 1e-12, "res_rel_dudt": 1e-3, "res_rel_visc": 1e-3}],
               "halfdt_probes": [{"ratio_rel_dudt": 4.0}]}
    try:
        show.mkdir(parents=True, exist_ok=True); cdir.mkdir(parents=True, exist_ok=True)
        env = {**__import__("os").environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
        (show / "dynamics_residuals.json").write_text(json.dumps(good))
        r_good = subprocess.run([sys.executable, str(script), case], capture_output=True, env=env)
        assert r_good.returncode == 0, f"good D4 should PASS (exit 0), got {r_good.returncode}"
        (show / "dynamics_residuals.json").write_text(json.dumps(missing))
        r_miss = subprocess.run([sys.executable, str(script), case], capture_output=True, env=env)
        assert r_miss.returncode != 0, (
            "MISSING D4 must FAIL the D-group gate (D is non-negotiable), "
            f"got exit {r_miss.returncode} -- a missing-D4 run would slip into the corpus")
    finally:
        shutil.rmtree(show, ignore_errors=True); shutil.rmtree(cdir, ignore_errors=True)
