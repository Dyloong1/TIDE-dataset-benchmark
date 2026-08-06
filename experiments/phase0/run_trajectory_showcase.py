"""Trajectory showcase run: warm-start from the newest cert checkpoint and
integrate one corpus-length trajectory (20 T_L), producing
  1. mid-plane slices (|omega|, u_x) every 0.1 t.u.  -> slices npz (GIF source)
  2. inline NS-residual samples (D2) + half-dt convergence probes (D3)
     -> dynamics_residuals.json
  3. per-sample divergence residual (D1)

NS residual protocol (frozen):
  at sample time, with state A at t0: B = step(A, h), C = step(B, h)
  r_hat = (C - A)/(2h) - [ P(u x omega)|_B + nu lap u|_B + f_fp|_B ]
  where f_fp = (eps_w / 2 E_f) * u_hat * band_mask  (continuous form of the
  fixed-power injection). All terms spectral (spatially exact); the residual
  therefore measures TIME discretization truncation: O(h^2) central-difference
  estimator + O(h^3) RK3. D3: repeat from A with h/2; expect ~4x smaller.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import solver._env  # noqa: F401
import numpy as np
import torch

from solver.config import load_config
from solver.memguard import check_ram
from solver.operators import curl_hat, pressure_hat
from solver.solver import PseudoSpectralSolver

HERE = Path(__file__).parent
_FFT = (-3, -2, -1)


def grid_norm(grid, x_hat) -> float:
    """L2 norm over the box via Parseval (sqrt of mean square)."""
    e = (x_hat.real.double() ** 2 + x_hat.imag.double() ** 2).sum(dim=0).flatten()
    return float((grid._w64_flat * e).sum().sqrt() / grid.n_total)


def ns_terms(s, u_hat):
    """Return (advection-only, viscous, forcing) spectral terms at u_hat, so the
    caller's residual r = dudt - (nl + visc + force) holds for ANY forcing mode.

    The showcase originally assumed fixed-power (post-mode) forcing. solver.rhs_
    behaves differently per mode:
      - none (free decay): no forcing -> force = 0, nl is pure advection.
      - post-mode (fixed_power / energy_preserving): rhs_ returns advection only
        (force applied post-step) -> nl is already advection-only; rebuild the
        continuous fixed-power force (eps_w/2E_f) u on the band.
      - rhs-mode (stochastic_ou / negative_damping): rhs_ ADDS f_hat into its
        output -> subtract it back out so nl is advection-only, and report the
        force the forcing returns at this state (for OU, the frozen b_hat).
    """
    nl = torch.empty_like(u_hat)
    s.rhs_(u_hat, nl)
    visc = (-s.nu_k2) * u_hat
    f = s.forcing
    if not getattr(s, "has_forcing", True):
        force = torch.zeros_like(u_hat)   # free decay: NS residual has no force term
    elif getattr(s, "rhs_forcing", False):
        force = torch.empty_like(u_hat)
        f(u_hat, force)
        nl = nl - force          # strip the force rhs_ folded in -> advection only
    else:
        E_f = f.band_energy(u_hat)
        force = (f.eps_w / (2.0 * max(E_f, 1e-30))) * (u_hat * f.band_mask)
    return nl, visc, force


def residual_sample(s, h: float) -> dict:
    """Run A->B->C with fixed step h, compute the centered NS residual at B.
    Restores the solver to state C afterwards (the trajectory continues).

    D2 measures the NS-EQUATION residual: how well (C-A)/2h == nl+visc+force at B.
    For STOCHASTIC (OU) forcing this requires freezing b_hat across A->B->C. The
    centered difference (C-A)/2h captures the ACTUAL force applied over the two
    steps; if each step reseeds a fresh OU increment, the A->B and B->C force
    realizations differ, so (C-A)/2h contains the noise mismatch while `force`
    below is only the single B-instant value -> the residual is dominated by the
    stochastic increment, NOT the time-truncation it is meant to measure (live
    noise inflated D2 ~30x: 1.1e-2 vs the true 3e-4 here; all OU runs were biased,
    #1/#2 only passed by luck). Freezing b_hat makes the 3 frames see ONE force
    realization, isolating the genuine NS residual. No-op for deterministic
    forcings (no _freeze attribute). This is the same isolation the D3 half-dt
    probe already documents and uses; D2 must use it too, by definition of D2."""
    grid = s.grid
    f = s.forcing
    froze = hasattr(f, "_freeze")
    b_saved = f.b_hat.clone() if froze else None
    if froze:
        f._freeze = True
    A = s.u_hat.clone()
    s.step(h)
    B = s.u_hat.clone()
    t_B = s.t
    s.step(h)
    C = s.u_hat.clone()

    dudt = (C - A) / (2.0 * h)
    nl, visc, force = ns_terms(s, B)
    r = dudt - (nl + visc + force)
    if froze:
        f._freeze = False
        f.b_hat.copy_(b_saved)

    div_B = grid.kx * B[0] + grid.ky * B[1] + grid.kz * B[2]
    grad2 = float((grid._w64_flat
                   * (grid._k2_64_flat
                      * (B.real.double()**2 + B.imag.double()**2).sum(0).flatten())).sum())
    div2 = float((grid._w64_flat
                  * (div_B.real.double()**2 + div_B.imag.double()**2).flatten()).sum())

    # D4 velocity-pressure consistency at state B:
    #   Poisson residual lap(p) + d_i d_j(u_i u_j), relative to the RHS;
    #   grad(p) vs irrotational part of the convective term.
    p_hat = pressure_hat(B, grid)
    lap_p = (-grid.k2) * p_hat
    ks = (grid.kx, grid.ky, grid.kz)
    u_B = torch.fft.irfftn(B, s=(grid.N,) * 3, dim=_FFT)
    poisson_rhs = torch.zeros_like(p_hat)
    adv_hat = torch.zeros_like(B)
    for i in range(3):
        for j in range(3):
            tij = torch.fft.rfftn(u_B[i] * u_B[j], dim=_FFT)
            poisson_rhs += (ks[i] * ks[j]) * tij     # = -d_i d_j(u_i u_j)
            adv_hat[i] += (1j * ks[j]) * tij          # (u.grad)u_i = d_j(u_i u_j)
    poisson_res = float((lap_p - poisson_rhs).abs().max() / max(float(poisson_rhs.abs().max()), 1e-300))
    div_adv = (ks[0] * adv_hat[0] + ks[1] * adv_hat[1] + ks[2] * adv_hat[2]) * grid.inv_k2
    irrot = torch.stack([-ks[i] * div_adv for i in range(3)])   # grad part of -adv
    grad_p = torch.stack([(1j * ks[i]) * p_hat for i in range(3)])
    gradp_res = float((irrot - grad_p).abs().max() / max(float(grad_p.abs().max()), 1e-300))

    out = {
        "t": t_B, "h": h,
        "res_rel_dudt": grid_norm(grid, r) / grid_norm(grid, dudt),
        "res_rel_visc": grid_norm(grid, r) / grid_norm(grid, visc),
        "res_rel_nl": grid_norm(grid, r) / grid_norm(grid, nl),
        "norm_dudt": grid_norm(grid, dudt),
        "norm_nl": grid_norm(grid, nl),
        "norm_visc": grid_norm(grid, visc),
        "norm_force": grid_norm(grid, force),
        "D1_div_ratio": div2 / max(grad2, 1e-300),
        "D4_poisson_res": poisson_res,
        "D4_gradp_res": gradp_res,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cert_fixedpower_fp64.yaml")
    ap.add_argument("--ckpt-dir", default="results/cert_fixedpower_fp64")
    # Per-case isolation: pass --out results/trajectory_showcase_<case> so two
    # routes don't overwrite each other; eval_dynamics_residuals.py <case> reads
    # results/trajectory_showcase_<case>/ automatically.
    ap.add_argument("--out", default="results/trajectory_showcase")
    ap.add_argument("--n-TL", type=float, default=20.0)
    ap.add_argument("--T-L", type=float, default=1.97)
    ap.add_argument("--slice-every", type=float, default=0.1)
    ap.add_argument("--residual-every", type=float, default=4.0)
    ap.add_argument("--n-halfdt", type=int, default=3,
                    help="number of residual samples that also run the h/2 probe")
    args = ap.parse_args()

    cfg = load_config(HERE / args.config)
    out_dir = HERE / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    cks = sorted((HERE / args.ckpt_dir).glob("ckpt_t*.pt"))
    if not cks:
        # No checkpoint in the run window (e.g. a decay window shorter than the
        # checkpoint interval). Exit non-zero with a clear message instead of an
        # IndexError so the caller (run_kdd_config.ps1) quarantines the seed rather
        # than admitting frames with no D-group source.
        sys.exit(f"[showcase] no ckpt_t*.pt in {args.ckpt_dir} -- cannot warm-start "
                 f"D-group probe (window shorter than --checkpoint-every?)")
    ck = cks[-1]
    d = torch.load(ck, map_location="cpu", weights_only=False)
    s = PseudoSpectralSolver(cfg.solver, d["u_hat"])
    s.t = float(d["t"])
    print(f"[showcase] warm start from {ck.name} (t={s.t:.1f})", flush=True)

    N = s.grid.N
    t_end = s.t + args.n_TL * args.T_L
    slices_om, slices_ux, slice_t = [], [], []
    next_slice = s.t
    next_res = s.t + 2.0
    residuals, halfdt_pairs = [], []

    while s.t < t_end - 1e-9:
        if s.t >= next_slice - 1e-9:
            om = torch.fft.irfftn(curl_hat(s.u_hat, s.grid, out=s.work_hat),
                                  s=(N, N, N), dim=_FFT)
            om_sl = (om[:, N // 2] ** 2).sum(0).sqrt().float().cpu().numpy()
            ux_full = torch.fft.irfftn(s.u_hat[0], s=(N, N, N), dim=_FFT)
            ux_sl = ux_full[N // 2].float().cpu().numpy()
            slices_om.append(om_sl)
            slices_ux.append(ux_sl)
            slice_t.append(s.t)
            del om, ux_full         # full N^3 temporaries: drop before stepping
            next_slice += args.slice_every
            check_ram("showcase slices", abort_frac=0.92)
        if s.t >= next_res - 1e-9:
            h = s.suggest_dt()
            A = s.u_hat.clone()
            t_A = s.t
            r_full = residual_sample(s, h)          # advances 2 steps (state=C); self-freezes forcing
            residuals.append(r_full)
            if len(halfdt_pairs) < args.n_halfdt:
                state_C = s.u_hat.clone()           # save trajectory position
                t_C = s.t
                # D3 convergence probe: re-measure the SAME trajectory point at h
                # and h/2 and check the residual falls ~4x (O(h^2) time truncation).
                # residual_sample now self-freezes the OU forcing for BOTH probes
                # (see its docstring), so the pair sees one force realization and
                # the ratio reflects integrator truncation, not noise mismatch.
                # The full-h probe is recomputed from A under the (frozen) force so
                # it is consistent with the h/2 probe. (r_full above already used
                # the same frozen-force protocol, so rf_full ~= r_full.)
                s.u_hat.copy_(A); s.t = t_A
                rf_full = residual_sample(s, h)
                s.u_hat.copy_(A); s.t = t_A
                r_half = residual_sample(s, h / 2.0)
                halfdt_pairs.append({"full": rf_full, "half": r_half,
                                     "ratio_rel_dudt": rf_full["res_rel_dudt"]
                                                       / max(r_half["res_rel_dudt"], 1e-300)})
                s.u_hat.copy_(state_C); s.t = t_C   # restore trajectory
            next_res += args.residual_every
            print(f"  residual@t={r_full['t']:.2f}: rel(dudt)={r_full['res_rel_dudt']:.3e} "
                  f"D1={r_full['D1_div_ratio']:.2e}", flush=True)
        s.step(s.suggest_dt(t_target=t_end))

    np.savez_compressed(out_dir / "slices.npz",
                        t=np.asarray(slice_t),
                        omega=np.stack(slices_om),
                        ux=np.stack(slices_ux),
                        T_L=args.T_L, t0=slice_t[0] if slice_t else 0.0)
    _ftype = cfg.solver.forcing.type
    _fdesc = ("f = frozen OU field b_hat on band (stochastic_ou)"
              if _ftype in ("stochastic_ou", "negative_damping")
              else "f = (eps_w/2E_f) u_hat on band (fixed-power continuous form)")
    with open(out_dir / "dynamics_residuals.json", "w", encoding="utf-8") as fh:
        json.dump({"protocol": f"central-diff over 2h at B; terms spectral; {_fdesc}",
                   "forcing_type": _ftype,
                   "nu": cfg.solver.nu, "eps_w": cfg.solver.forcing.eps_w,
                   "samples": residuals, "halfdt_probes": halfdt_pairs},
                  fh, indent=2)
    print(f"[showcase] {len(slices_om)} slices, {len(residuals)} residual samples, "
          f"{len(halfdt_pairs)} half-dt probes -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
