"""Passive-scalar HIT entry (extended-physics axis 2).

A passive scalar theta is advected by an UNCHANGED NS velocity (the flow is still
pure incompressible NS — reuses any validated forced-HIT config). Adds scalar
metrics (Schmidt number, Batchelor resolution gate, scalar variance, scalar
dissipation chi, scalar spectrum). The velocity side reuses the phase0 loop.

    python run_scalar_hit.py --config configs/scalar_sc1_256_fp64.yaml \
        --t-spinup 30 --t-sample 100 --checkpoint-every 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[1] / "phase0"))

import solver._env  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from solver.grids import SpectralGrid  # noqa: E402
from solver.initial_conditions import random_solenoidal  # noqa: E402
from solver.solver import PseudoSpectralSolver  # noqa: E402
from physics_ext import diagnostics_ext as dx  # noqa: E402
from physics_ext.config_ext import dump_ext_config, load_ext_config  # noqa: E402
from physics_ext.scalar import ScalarTransport  # noqa: E402

from _ext_runner import run_ext  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--t-spinup", type=float, default=30.0)
    ap.add_argument("--t-sample", type=float, default=100.0)
    ap.add_argument("--checkpoint-every", type=float, default=0.0)
    ap.add_argument("--eps-sentinel", action="store_true")
    ap.add_argument("--resume-u-ckpt", default="", help="warm-start the velocity from a ckpt")
    ap.add_argument("--seed", type=int, default=-1,
                    help="override the IC seed (multi-seed ensemble)")
    ap.add_argument("--ou-seed-per-ic", action="store_true",
                    help="ou_seed = config ou_seed + IC seed -> INDEPENDENT forcing per "
                         "seed (shared ou_seed imprints a common <u_i u_j> bias signed A10 "
                         "pooling can't cancel; needed for a true ensemble).")
    ap.add_argument("--frame-every-TL", type=float, default=0.0,
                    help="export a corpus frame every N*T_L in the sampling window (needs --T-L)")
    ap.add_argument("--T-L", type=float, default=0.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg, _rot, scal_cfg, _strat = load_ext_config(a.config)
    if a.seed >= 0:
        cfg.seed = a.seed   # velocity IC seed only (the scalar field IC keeps scal_cfg.seed)
    if a.ou_seed_per_ic and getattr(cfg.solver.forcing, "type", "none") != "none":
        cfg.solver.forcing.ou_seed = int(cfg.solver.forcing.ou_seed) + int(cfg.seed)
        print(f"[forcing] ou_seed = {cfg.solver.forcing.ou_seed} (per-IC, independent)", flush=True)
    if not scal_cfg.enabled:
        raise SystemExit("config has no enabled `scalar:` block")
    out_dir = Path(a.out) if a.out else _HERE.parents[0] / "results" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_ext_config(cfg, None, scal_cfg, None, out_dir)  # full snapshot incl. scalar block

    grid = SpectralGrid(cfg.solver.N, cfg.solver.device, cfg.solver.dtype)
    u0 = random_solenoidal(grid, seed=cfg.seed, k_p=cfg.k_p, u_rms=cfg.u_rms,
                           spectrum_power=cfg.ic_spectrum_power)
    s = PseudoSpectralSolver(cfg.solver, u0)
    if a.resume_u_ckpt:
        ck = torch.load(a.resume_u_ckpt, map_location="cpu", weights_only=False)
        s.u_hat.copy_(ck["u_hat"].to(grid.device, grid.cdtype))
        s.t = float(ck["t"]); s.n_steps = int(ck.get("n_steps", 0))
        print(f"[scalar] warm-start u from {Path(a.resume_u_ckpt).name} t={s.t:.1f}", flush=True)

    # scalar IC: random field with the configured low-k spectrum (variance grows /
    # equilibrates under advection + mean-gradient production)
    torch.manual_seed(scal_cfg.seed)
    th0 = torch.fft.rfftn(torch.randn(cfg.solver.N, cfg.solver.N, cfg.solver.N,
                                      dtype=grid.rdtype, device=grid.device),
                          dim=(-3, -2, -1)).unsqueeze(0)
    th0 *= scal_cfg.b_rms
    scal = ScalarTransport(grid, scal_cfg, th0)
    Sc = cfg.solver.nu / scal_cfg.kappa if scal_cfg.kappa > 0 else float("inf")
    print(f"[scalar] {cfg.name} Sc={Sc:.2f} kappa={scal_cfg.kappa} "
          f"mean_grad={scal_cfg.mean_grad} forcing={cfg.solver.forcing.type}", flush=True)

    N = cfg.solver.N
    _uph = torch.empty((3, N, N, N), dtype=grid.rdtype, device=grid.device)

    def step_fn(dt):
        # Strang split for a 2nd-order coupled (u, theta) advance (deep-review
        # 2026-06-23: the old "advance u then theta with frozen end velocity" was a
        # Lie split, only 1st order; convergence-tested Strang -> p~2.1):
        #   theta half-step with u(t)  ->  u full-step  ->  theta half-step with u(t+dt)
        torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=(-3, -2, -1), out=_uph)
        scal.step(_uph, dt * 0.5)
        s.step(dt)
        torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=(-3, -2, -1), out=_uph)
        scal.step(_uph, dt * 0.5)

    def extra(summ):
        k, Eth = dx.scalar_spectrum(scal.theta_hat, grid)
        np.save(out_dir / "E_theta_mean.npy", Eth.numpy())
        eps = s.grid.dissipation(s.u_hat, cfg.solver.nu)
        from solver.diagnostics import k_max_eta
        keta = k_max_eta(grid.k_max_resolved, eps, cfg.solver.nu)
        bat = dx.batchelor_resolution(keta, Sc if Sc != float("inf") else 1e9)
        return {
            "physics": "passive_scalar", "Sc": Sc, "kappa": scal_cfg.kappa,
            "mean_grad": scal_cfg.mean_grad,
            "scalar_variance": dx.scalar_variance(scal.theta_hat, grid),
            "scalar_dissipation_chi": dx.scalar_dissipation(scal.theta_hat, grid, scal_cfg.kappa),
            **{k2: v for k2, v in bat.items()},
        }

    # corpus frame: 5ch (u,v,w,p,theta). Pressure = plain D4-certified pressure_hat
    # (a PASSIVE scalar does not react on momentum, so the pressure is unchanged).
    from solver.operators import pressure_hat as _phat

    def frame_fn(solver):
        p = torch.fft.irfftn(_phat(solver.u_hat, solver.grid), s=(N, N, N), dim=(-3, -2, -1))
        th = torch.fft.irfftn(scal.theta_hat[0], s=(N, N, N), dim=(-3, -2, -1))
        return {"u": solver.velocity_physical().to(torch.float32).cpu(),
                "p": p.to(torch.float32).cpu(),
                "theta": th.to(torch.float32).cpu()}

    metrics = run_ext(cfg, s, out_dir, a.t_spinup, a.t_sample, step_fn, extra,
                      experiment="passive_scalar_hit", checkpoint_every=a.checkpoint_every,
                      eps_sentinel=a.eps_sentinel,
                      frame_every_tl=a.frame_every_TL, frame_T_L=a.T_L, frame_fn=frame_fn)
    # also stash the scalar field in the final checkpoint convenience
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"[scalar] done: Re_lambda={metrics['Re_lambda']:.1f} Sc={metrics['Sc']:.2f} "
          f"<theta^2>={metrics['scalar_variance']:.4f} chi={metrics['scalar_dissipation_chi']:.4e} "
          f"class_I={metrics.get('class_I')}", flush=True)


if __name__ == "__main__":
    main()
