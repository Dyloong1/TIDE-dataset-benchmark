"""Stratified turbulence entry (extended-physics axis 3: Boussinesq).

Active scalar (buoyancy) two-way coupled to the momentum equation. Adds S-group
metrics (Froude, buoyancy Reynolds Re_b, Ozmidov scale + resolvability, PE/KE,
buoyancy flux <wb>, vertical anisotropy). Forcing optional — a decaying stratified
run (forcing=none) is the simplest first config.

    python run_stratified_hit.py --config configs/stratified_fr_256_fp64.yaml \
        --t-spinup 10 --t-sample 60 --checkpoint-every 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:  # Windows cp1252 console guard (status prints may contain Chinese)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[1] / "phase0"))

import solver._env  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from solver.grids import SpectralGrid  # noqa: E402
from solver.initial_conditions import random_solenoidal  # noqa: E402
from physics_ext import diagnostics_ext as dx  # noqa: E402
from physics_ext.boussinesq import BoussinesqSolver  # noqa: E402
from physics_ext.config_ext import dump_ext_config, load_ext_config  # noqa: E402

from _ext_runner import run_ext  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--t-spinup", type=float, default=10.0)
    ap.add_argument("--t-sample", type=float, default=60.0)
    ap.add_argument("--checkpoint-every", type=float, default=0.0)
    ap.add_argument("--eps-sentinel", action="store_true")
    ap.add_argument("--frame-every-TL", type=float, default=0.0,
                    help="export a corpus frame every N*T_L in the sampling window (needs --T-L)")
    ap.add_argument("--T-L", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=-1,
                    help="override the IC seed (multi-seed ensemble)")
    ap.add_argument("--ou-seed-per-ic", action="store_true",
                    help="ou_seed = config ou_seed + IC seed -> INDEPENDENT forcing per "
                         "seed (shared ou_seed imprints a common <u_i u_j> bias signed A10 "
                         "pooling can't cancel; needed for a true ensemble).")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg, _rot, _scal, strat = load_ext_config(a.config)
    if a.seed >= 0:
        cfg.seed = a.seed   # velocity IC seed only (buoyancy IC keeps strat.seed)
    if a.ou_seed_per_ic and getattr(cfg.solver.forcing, "type", "none") != "none":
        cfg.solver.forcing.ou_seed = int(cfg.solver.forcing.ou_seed) + int(cfg.seed)
        print(f"[forcing] ou_seed = {cfg.solver.forcing.ou_seed} (per-IC, independent)", flush=True)
    if not strat.enabled:
        raise SystemExit("config has no enabled `stratification:` block")
    out_dir = Path(a.out) if a.out else _HERE.parents[0] / "results" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # dump_ext_config (config_ext) writes the base snapshot AND merges the enabled ext
    # block back, so load_ext_config(snapshot) in run_ext_showcase/eval_dynamics_ext sees
    # strat.enabled=True. Supersedes CYB-t's inline append (route-ou 510cc7b) — same fix,
    # but the shared helper is DRY, covers all 3 axes, preserves --seed, and is unit-tested.
    # (Both agents found this snapshot bug independently — strong confirmation it was real.)
    dump_ext_config(cfg, None, None, strat, out_dir)

    grid = SpectralGrid(cfg.solver.N, cfg.solver.device, cfg.solver.dtype)
    u0 = random_solenoidal(grid, seed=cfg.seed, k_p=cfg.k_p, u_rms=cfg.u_rms,
                           spectrum_power=cfg.ic_spectrum_power)
    # buoyancy IC: start unstratified perturbation (small noise) — the run develops
    # the buoyancy field under -N^2 w forcing and advection.
    torch.manual_seed(strat.seed)
    b0 = torch.fft.rfftn(torch.randn(cfg.solver.N, cfg.solver.N, cfg.solver.N,
                                     dtype=grid.rdtype, device=grid.device) * 0.01,
                         dim=(-3, -2, -1)).unsqueeze(0)
    s = BoussinesqSolver(cfg.solver, strat, u0, b0)
    N_bv = strat.N_sq ** 0.5 if strat.N_sq > 0 else 0.0
    print(f"[stratified] {cfg.name} N^2={strat.N_sq} (N={N_bv:.3f}) kappa={strat.kappa} "
          f"forcing={cfg.solver.forcing.type}", flush=True)

    def step_fn(dt):
        s.step(dt)

    def extra(summ):
        K = grid.kinetic_energy(s.u_hat)
        eps = grid.dissipation(s.u_hat, cfg.solver.nu)
        # authoritative spectral u_rms / L from the runner (Fr uses the proper
        # integral scale, not a rough u^3/eps — deep-review 2026-06-23).
        u_rms = summ["u_rms"]
        L = summ["L"]
        l_O = dx.ozmidov_scale(eps, N_bv) if N_bv > 0 else float("inf")
        eta = (cfg.solver.nu**3 / eps) ** 0.25 if eps > 0 else float("nan")
        Re_b = dx.buoyancy_reynolds(eps, cfg.solver.nu, N_bv) if N_bv > 0 else float("inf")
        PE = dx.potential_energy(s.b_hat, grid, strat.N_sq) if strat.N_sq > 0 else 0.0
        b, lam = dx.anisotropy_tensor(s.u_hat, grid)
        return {
            "physics": "stratified", "N_sq": strat.N_sq, "N_bv": N_bv,
            "kappa": strat.kappa, "Fr": dx.froude_number(u_rms, L, N_bv),
            "Re_b": Re_b, "ozmidov_l_O": l_O,
            **dx.stratified_resolution(eta, l_O, L, Re_b),
            "PE": PE, "KE": K, "PE_over_KE": (PE / K if K > 0 else float("nan")),
            "buoyancy_flux_wb": dx.buoyancy_flux(s.u_hat, s.b_hat, grid),
            "b_ij_eigs": lam.tolist(), "b_ij_max_abs": float(np.abs(lam).max()),
        }

    # corpus frame: 5ch (u,v,w,p,b). Pressure from the D4-CERTIFIED Boussinesq operator
    # (pressure_hat_boussinesq includes the +d_z b buoyancy source).
    from physics_ext.operators_ext import pressure_hat_boussinesq
    _Ns = grid.N

    def frame_fn(solver):
        p_hat = pressure_hat_boussinesq(solver.u_hat, solver.b_hat, solver.grid)
        p = torch.fft.irfftn(p_hat, s=(_Ns,) * 3, dim=(-3, -2, -1))
        b = torch.fft.irfftn(solver.b_hat[0], s=(_Ns,) * 3, dim=(-3, -2, -1))
        return {"u": solver.velocity_physical().to(torch.float32).cpu(),
                "p": p.to(torch.float32).cpu(),
                "b": b.to(torch.float32).cpu()}

    metrics = run_ext(cfg, s, out_dir, a.t_spinup, a.t_sample, step_fn, extra,
                      experiment="stratified_hit", checkpoint_every=a.checkpoint_every,
                      eps_sentinel=a.eps_sentinel,
                      frame_every_tl=a.frame_every_TL, frame_T_L=a.T_L, frame_fn=frame_fn)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"[stratified] done: Re_lambda={metrics['Re_lambda']:.1f} Fr={metrics['Fr']:.3f} "
          f"Re_b={metrics['Re_b']:.1f} l_O={metrics['ozmidov_l_O']:.4f} "
          f"turbulent={metrics.get('turbulent')} <wb>={metrics['buoyancy_flux_wb']:.4e}", flush=True)


if __name__ == "__main__":
    main()
