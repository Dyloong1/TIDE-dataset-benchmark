"""D2 (NS-momentum-residual) judge must FREEZE stochastic OU forcing across the
A->B->C centered-difference frames. With LIVE forcing each step reseeds a fresh
OU increment, so (C-A)/2h carries the noise mismatch and D2 is inflated ~30x
(observed: 1.1e-2 vs the true 3e-4 on helical #7; all OU runs were biased, #1/#2
passed only by luck). Freezing b_hat isolates the genuine NS residual. This pins
that residual_sample self-freezes and that it materially lowers D2 on an OU run.
The judge change is vetted before use (CLAUDE.md 'referee first' rule)."""
import importlib.util
from pathlib import Path

import solver._env  # noqa: F401
import numpy as np
import torch

from solver.config import SolverConfig, ForcingConfig, ExperimentConfig
from solver.solver import build_solver
from solver.initial_conditions import random_solenoidal
from solver.grids import SpectralGrid

_HERE = Path(__file__).resolve().parents[1]


def _load_showcase():
    spec = importlib.util.spec_from_file_location(
        "_showcase", _HERE / "experiments" / "phase0" / "run_trajectory_showcase.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_ou_solver(N=32, seed=3):
    fc = ForcingConfig(type="stochastic_ou", k_f=2.0, ou_tau=1.0, ou_sigma2=0.05)
    sc = SolverConfig(N=N, nu=0.01, dtype="fp64", device="cpu", forcing=fc)
    cfg = ExperimentConfig(name="t_ou", solver=sc, seed=seed)
    grid = SpectralGrid(N, "cpu", "fp64")
    u0 = random_solenoidal(grid, seed=seed, k_p=2.0)
    s = build_solver(cfg)
    s.u_hat.copy_(u0.to(grid.cdtype))
    for _ in range(20):           # spin up so b_hat is a developed OU state
        s.step(0.01)
    return s


def test_residual_sample_freezes_ou_forcing():
    sh = _load_showcase()
    s = _build_ou_solver()
    assert hasattr(s.forcing, "_freeze"), "OU forcing must expose _freeze for D2"
    h = 0.01

    # frozen (the fix): residual_sample self-freezes b_hat across A->B->C
    A = s.u_hat.clone(); t_A = s.t
    r_frozen = sh.residual_sample(s, h)["res_rel_dudt"]

    # live (the bug): same A->B->C but with OU advancing each step
    s.u_hat.copy_(A); s.t = t_A
    f = s.forcing
    A2 = s.u_hat.clone()
    s.step(h); B2 = s.u_hat.clone()
    s.step(h); C2 = s.u_hat.clone()
    dudt = (C2 - A2) / (2.0 * h)
    nl, visc, force = sh.ns_terms(s, B2)
    r = dudt - (nl + visc + force)
    num = float((s.grid._w64_flat * (r.real.double()**2 + r.imag.double()**2)
                 .sum(0).flatten()).sum()) ** 0.5
    den = float((s.grid._w64_flat * (dudt.real.double()**2 + dudt.imag.double()**2)
                 .sum(0).flatten()).sum()) ** 0.5
    r_live = num / max(den, 1e-300)

    # frozen must isolate the NS residual: far smaller than the noise-contaminated
    # live measurement (the whole point of the fix)
    assert r_frozen < r_live, f"frozen D2 {r_frozen:.2e} should be < live {r_live:.2e}"
    assert r_frozen < 0.5 * r_live, (
        f"freezing should materially cut D2: frozen={r_frozen:.2e} live={r_live:.2e}")


def test_freeze_restores_forcing_state():
    """residual_sample must leave b_hat and _freeze unchanged (trajectory continues
    with live forcing afterward)."""
    sh = _load_showcase()
    s = _build_ou_solver()
    b_before = s.forcing.b_hat.clone()
    freeze_before = s.forcing._freeze
    _ = sh.residual_sample(s, 0.01)
    assert s.forcing._freeze == freeze_before, "_freeze leaked"
    assert torch.allclose(s.forcing.b_hat, b_before), "b_hat not restored after D2 sample"
