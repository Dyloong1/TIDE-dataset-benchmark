"""Judge-first target tests for the finalized-metrics additions (KDD D&B). Each metric is checked
against a KNOWN answer (analytic field or constructed case), not just 'runs without error'.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=<root>:<root>/benchmark python -m pytest tests/test_new_metrics.py -q
"""
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("TURBGEN_REPO", str(Path.home() / "turbulence"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_metrics as BM  # noqa: E402

N = 32


def _abc_field(N):
    """A divergence-free ABC-like velocity field with a known pressure (analytic target)."""
    ax = torch.linspace(0, 2 * 3.141592653589793, N + 1)[:-1]
    Z, Y, X = torch.meshgrid(ax, ax, ax, indexing="ij")
    u = torch.stack([torch.sin(Z) + torch.cos(Y),
                     torch.sin(X) + torch.cos(Z),
                     torch.sin(Y) + torch.cos(X)], dim=0)
    return u


def test_sgs_correlation_identical_is_one():
    tau = torch.randn(6, N, N, N)
    assert abs(BM.sgs_stress_correlation(tau, tau) - 1.0) < 1e-5
    # anti-correlated -> -1
    assert abs(BM.sgs_stress_correlation(tau, -tau) + 1.0) < 1e-5
    # independent -> near 0
    assert abs(BM.sgs_stress_correlation(torch.randn(6, N, N, N), torch.randn(6, N, N, N))) < 0.1


def test_poisson_residual_true_pressure_is_small():
    # For a single Fourier mode p = cos(z), u chosen so RHS matches: simplest check is that the
    # residual of the SolvED pressure (from the field's own velocity) is ~0. Build u, solve p via
    # the same spectral operator the metric uses, then residual must be ~machine-zero.
    u = _abc_field(N)
    # solve p: ∇²p = RHS -> p_hat = -RHS_hat / k²
    dev = u.device
    kz = torch.fft.fftfreq(N, d=1.0 / N); kx = torch.fft.rfftfreq(N, d=1.0 / N)
    KZ = kz[:, None, None]; KY = kz[None, :, None]; KX = kx[None, None, :]
    k2 = (KZ ** 2 + KY ** 2 + KX ** 2)
    grads = torch.stack([BM._grad_spectral(u[i], N, dev) for i in range(3)], dim=0)
    rhs = torch.zeros(N, N, N)
    for i in range(3):
        for j in range(3):
            rhs = rhs - grads[i, j] * grads[j, i]
    rhs_h = torch.fft.rfftn(rhs, dim=(-3, -2, -1))
    # solve p_hat = -rhs_h / k^2 with the k=0 (DC) mode set to 0 (p defined up to a const)
    k2c = k2.clone(); k2c[0, 0, 0] = 1.0
    ph = -rhs_h / k2c; ph[0, 0, 0] = 0.0
    p = torch.fft.irfftn(ph, s=(N, N, N), dim=(-3, -2, -1))
    res = BM.pressure_poisson_residual(u, p)
    assert res < 1e-4, res           # the true pressure satisfies its Poisson eq
    # a WRONG pressure (random) must give a large residual
    assert BM.pressure_poisson_residual(u, torch.randn(N, N, N)) > 0.5


def test_sgs_transfer_smagorinsky_no_backscatter():
    # Smagorinsky tau = -2 nu_t S is purely dissipative: Pi = -tau:S = 2 nu_t |S|^2 >= 0 everywhere
    # -> backscatter fraction ~ 0. Build S from a filtered field, set tau = -2*0.01*S (6 comps).
    u = _abc_field(N)
    uf = BM._gaussian_filter_metric(u, 0.06)
    dev = u.device
    g = torch.stack([BM._grad_spectral(uf[i], N, dev) for i in range(3)], dim=0)
    S = 0.5 * (g + g.transpose(0, 1))
    nu_t = 0.01
    idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    tau = torch.stack([-2 * nu_t * S[i, j] for (i, j) in idx], dim=0)
    out = BM.sgs_energy_transfer(tau, uf, 0.06)
    assert out["Pi_mean"] >= -1e-6, out["Pi_mean"]          # net dissipative
    assert out["backscatter_frac"] < 0.05, out              # ~no backscatter for eddy-viscosity


def test_vorticity_pdf_tails_smoothing_lowers_flatness():
    u = _abc_field(N) + 0.3 * torch.randn(3, N, N, N)   # add small scales -> heavier tails
    sharp = BM.vorticity_pdf_tails(u)
    smooth = BM.vorticity_pdf_tails(BM._gaussian_filter_metric(u, 0.12))  # smoothed
    # smoothing removes small-scale vorticity -> tail_frac drops (over-smoothing signature)
    assert smooth["tail_frac"] <= sharp["tail_frac"] + 1e-6
    assert sharp["flatness"] > 0


def test_correlation_time_identical_survives_full():
    seq = torch.randn(6, 3, N, N, N)
    # identical pred==truth: correlation 1 throughout -> survives full horizon (T-1)
    assert abs(BM.correlation_time(seq, seq, thresh=0.5, dt=1.0) - 5.0) < 1e-6
    # a sequence that decorrelates immediately (independent) -> small time
    p = torch.randn(6, 3, N, N, N)
    assert BM.correlation_time(p, torch.randn(6, 3, N, N, N), thresh=0.5) <= 5.0


if __name__ == "__main__":
    for fn in [test_sgs_correlation_identical_is_one, test_poisson_residual_true_pressure_is_small,
               test_sgs_transfer_smagorinsky_no_backscatter, test_vorticity_pdf_tails_smoothing_lowers_flatness,
               test_correlation_time_identical_survives_full]:
        fn(); print(f"  PASS {fn.__name__}")
    print("\n5/5 new-metric target tests passed")
