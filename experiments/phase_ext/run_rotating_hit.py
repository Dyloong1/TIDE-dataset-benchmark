"""Rotating HIT entry (extended-physics axis 1: Coriolis).

Forced isotropic turbulence in a rotating frame. Reuses the phase0 run loop
(_ext_runner) and the OU forcing (rotation needs forcing to stay statistically
steady; Coriolis injects no energy). Adds R-group metrics (Rossby, 2D/3D energy
partition, anisotropy tensor, helicity).

    python run_rotating_hit.py --config configs/rotating_ro0p1_256_fp64.yaml \
        --t-spinup 30 --t-sample 200 --checkpoint-every 10 --eps-sentinel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[1] / "phase0"))

import solver._env  # noqa: F401,E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from solver.grids import SpectralGrid  # noqa: E402
from solver.initial_conditions import random_solenoidal  # noqa: E402
from physics_ext import diagnostics_ext as dx  # noqa: E402
from physics_ext.config_ext import dump_ext_config, load_ext_config  # noqa: E402
from physics_ext.rotating import RotatingSolver  # noqa: E402

from _ext_runner import run_ext  # noqa: E402


def build_rotating(cfg, rot):
    grid = SpectralGrid(cfg.solver.N, cfg.solver.device, cfg.solver.dtype)
    if cfg.ic != "random_solenoidal":
        raise ValueError("rotating HIT expects ic: random_solenoidal")
    u0 = random_solenoidal(grid, seed=cfg.seed, k_p=cfg.k_p, u_rms=cfg.u_rms,
                           spectrum_power=cfg.ic_spectrum_power)
    return RotatingSolver(cfg.solver, u0, rot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--t-spinup", type=float, default=30.0)
    ap.add_argument("--t-sample", type=float, default=200.0)
    ap.add_argument("--checkpoint-every", type=float, default=0.0)
    ap.add_argument("--eps-sentinel", action="store_true")
    ap.add_argument("--seed", type=int, default=-1,
                    help="override the IC seed (for multi-seed ensembles)")
    ap.add_argument("--ou-seed-per-ic", action="store_true",
                    help="ou_seed = config ou_seed + IC seed -> INDEPENDENT forcing per "
                         "seed. A shared ou_seed imprints a common <u_i u_j> bias signed "
                         "A10 pooling can't cancel (proven on #1); needed for a true ensemble.")
    ap.add_argument("--frame-every-TL", type=float, default=0.0,
                    help="export a corpus frame every N*T_L in the sampling window (needs --T-L)")
    ap.add_argument("--T-L", type=float, default=0.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg, rot, _scal, _strat = load_ext_config(a.config)
    if a.seed >= 0:
        cfg.seed = a.seed   # IC seed
    if a.ou_seed_per_ic and getattr(cfg.solver.forcing, "type", "none") != "none":
        cfg.solver.forcing.ou_seed = int(cfg.solver.forcing.ou_seed) + int(cfg.seed)
        print(f"[forcing] ou_seed = {cfg.solver.forcing.ou_seed} (per-IC, independent)", flush=True)
    if not rot.enabled:
        raise SystemExit("config has no enabled `rotation:` block")
    out_dir = Path(a.out) if a.out else _HERE.parents[0] / "results" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_ext_config(cfg, rot, None, None, out_dir)  # full snapshot incl. rotation block

    s = build_rotating(cfg, rot)
    print(f"[rotating] {cfg.name} Omega={rot.omega_vec} nu={cfg.solver.nu} "
          f"forcing={cfg.solver.forcing.type}", flush=True)

    def step_fn(dt):
        s.step(dt)

    # corpus frame: 4ch (u,v,w,p), pressure from the D4-CERTIFIED rotating operator
    # (pressure_hat_rotating includes the +2 Omega.omega Coriolis source; plain
    # pressure_hat would be the wrong non-rotating pressure).
    import torch as _torch
    from physics_ext.operators_ext import pressure_hat_rotating
    _Nf = s.grid.N

    def frame_fn(solver):
        p_hat = pressure_hat_rotating(solver.u_hat, rot.omega_vec, solver.grid)
        p = _torch.fft.irfftn(p_hat, s=(_Nf,) * 3, dim=(-3, -2, -1))
        return {"u": solver.velocity_physical().to(_torch.float32).cpu(),
                "p": p.to(_torch.float32).cpu()}

    def extra(summ):
        # R-group on the final (stationary) field. Use the AUTHORITATIVE spectral
        # u_rms / L from the runner (summ) for Ro — NOT a rough u^3/eps (deep-review
        # 2026-06-23: the rough u_rms=sqrt(2K) was sqrt3 off the metrics u_rms).
        b, lam = dx.anisotropy_tensor(s.u_hat, s.grid)
        E2d, E3d, frac = dx.energy_2d3d(s.u_hat, s.grid)
        u_rms = summ["u_rms"]
        L = summ["L"]
        return {
            "physics": "rotating",
            "omega_vec": list(rot.omega_vec), "omega_mag": rot.omega_mag,
            "Ro": dx.rossby_number(u_rms, L, rot.omega_mag),
            "E_2D": E2d, "E_3D": E3d, "frac_2D": frac,
            "b_ij_eigs": lam.tolist(), "b_ij_max_abs": float(np.abs(lam).max()),
            "helicity": dx.helicity(s.u_hat, s.grid),
            "rel_helicity": dx.rel_helicity(s.u_hat, s.grid),
        }

    metrics = run_ext(cfg, s, out_dir, a.t_spinup, a.t_sample, step_fn, extra,
                      experiment="rotating_hit", checkpoint_every=a.checkpoint_every,
                      eps_sentinel=a.eps_sentinel,
                      frame_every_tl=a.frame_every_TL, frame_T_L=a.T_L, frame_fn=frame_fn)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"[rotating] done: Re_lambda={metrics['Re_lambda']:.1f} "
          f"k_max*eta={metrics['k_max_eta']:.2f} Ro={metrics['Ro']:.3f} "
          f"frac_2D={metrics['frac_2D']:.3f} b_max={metrics['b_ij_max_abs']:.3f}", flush=True)


if __name__ == "__main__":
    main()
