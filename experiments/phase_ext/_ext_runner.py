"""Shared run loop for the extended-physics entries (rotating / scalar / stratified).

Mirrors experiments/phase0/run_forced_hit.py::run() exactly for the velocity side
(same timeseries.csv / spectra.npz / E_mean.npy / metrics.json structure, same A10
component moments, same A14 sentinel, same diskguard) so eval_anisotropic can reuse
eval_dns_standard's A-group logic unchanged. The new physics enters via two hooks:

  step_fn(dt)        : advance ONE dt of the coupled system (velocity + scalar/buoyancy).
  extra_metrics_fn() : return a dict of new-physics metrics merged into metrics.json
                       (Ro / 2D-3D / b_ij for rotation; chi / Sc for scalar; Fr / Re_b
                       / Ozmidov / <wb> for stratification).

This file lives under experiments/phase_ext (NOT phase0) and imports phase0/solver
modules read-only — solver/ and experiments/phase0/ are never modified.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                     # repo root (solver, ...)
sys.path.insert(0, str(_HERE.parents[1] / "phase0"))         # reuse phase0 helpers

import solver._env  # noqa: F401,E402
from solver.diagnostics import (derivative_skewness, fit_spectrum_slope,  # noqa: E402
                                k_max_eta, spectral_summary)
from solver.diskguard import check_disk, estimate_run_bytes  # noqa: E402
from solver.io_utils import CSVLogger, SpectrumLogger, save_snapshot  # noqa: E402
from solver.memguard import check_ram  # noqa: E402
from solver.operators import pressure_hat  # noqa: E402

SLOPE_WINDOW = (4.0, 20.0)


def run_ext(cfg, solver, out_dir: Path, t_spinup: float, t_sample: float,
            step_fn, extra_metrics_fn, experiment: str,
            sample_skewness_every: float = 2.0, checkpoint_every: float = 0.0,
            eps_sentinel: bool = False, save_triplet: bool = False,
            frame_every_tl: float = 0.0, frame_T_L: float = 0.0,
            frame_fn=None) -> dict:
    """Generic extended-physics run. `solver` exposes the phase0 solver API
    (grid, u_hat, t, n_steps, scalars(), velocity_physical(), suggest_dt()).
    `step_fn(dt)` advances the whole coupled system; `extra_metrics_fn()` adds the
    new-physics metrics at the end. Returns the metrics dict (also writes it).

    Corpus frame export (KDD production): frame_every_tl>0 exports a fp32 corpus
    frame every frame_every_tl*frame_T_L in the sampling window, behind the SAME
    per-frame k_maxη>=1.5 gate as phase0/run_forced_hit. `frame_fn(solver)` returns
    the channel dict for one frame (each entry supplies its CERTIFIED pressure +
    any extra scalar/buoyancy channel — keeps this loop physics-agnostic). This
    mirrors run_forced_hit.py:152-176 exactly; frames come from the SAME accepted
    fp64 trajectory (no re-run, no mutation)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = solver.grid
    csv = CSVLogger(out_dir / "timeseries.csv",
                    ["step", "t", "dt", "K", "eps", "eps_inj", "umax", "wall_s"])
    spec = SpectrumLogger(out_dir / "spectra.npz")

    t_spin_end = t_spinup
    t_end = t_spinup + t_sample
    t0 = time.perf_counter()
    samp = {"t": [], "K": [], "eps": [], "eps_inj": []}
    comp_sums = []
    triplet = []                       # (t, u_hat copies) for D-group, optional
    ckpt_next = checkpoint_every if checkpoint_every > 0 else float("inf")

    n_ckpt_est = int(t_sample / checkpoint_every) + 2 if checkpoint_every > 0 else 0
    need_gb = estimate_run_bytes(cfg.solver.N, n_ckpt_est, cfg.solver.dtype, 0) / 1e9
    check_disk(out_dir, need_gb=need_gb, where=f"{cfg.name} run start")

    sentinel_ref = cfg.solver.forcing.eps_w
    sentinel_low_since = None
    a14_aborted = False
    skew_vals, skew_next = [], t_spin_end
    E_accum, n_spec = None, 0

    # corpus frame export (mirrors run_forced_hit.py:91-130,152-176): per-frame
    # k_maxη>=1.5 gate, 4/5-channel fp32 via frame_fn, written to out_dir/frames.
    frame_dt = frame_every_tl * frame_T_L if (frame_every_tl > 0 and frame_T_L > 0) else float("inf")
    frame_next = t_spin_end if frame_dt != float("inf") else float("inf")
    frame_idx = frame_seen = frame_skipped = 0
    frame_dir = out_dir / "frames"
    if frame_dt != float("inf"):
        if frame_fn is None:
            raise ValueError("frame_every_tl>0 requires frame_fn (per-physics channels)")
        frame_dir.mkdir(exist_ok=True)
        n_frame_est = int(t_sample / frame_dt) + 2
        check_disk(frame_dir, need_gb=estimate_run_bytes(cfg.solver.N, 0, cfg.solver.dtype,
                   n_frame_est) / 1e9, where=f"{cfg.name} frames")

    while solver.t < t_end - 1e-12:
        dt = solver.suggest_dt(t_target=t_end)
        step_fn(dt)
        sc = solver.scalars()
        if solver.n_steps % cfg.log_every_steps == 0:
            csv.log({"step": solver.n_steps, "dt": dt,
                     "wall_s": time.perf_counter() - t0, **sc})
        in_sampling = solver.t >= t_spin_end
        if in_sampling:
            samp["t"].append(solver.t); samp["K"].append(sc["K"])
            samp["eps"].append(sc["eps"]); samp["eps_inj"].append(sc["eps_inj"])
        if solver.n_steps % cfg.spectrum_every_steps == 0 and in_sampling:
            k, E = grid.shell_spectrum(solver.u_hat)
            spec.log(solver.t, k.numpy(), E.numpy())
            E_accum = E.numpy() if E_accum is None else E_accum + E.numpy()
            n_spec += 1
            ui2, uij = grid.component_moments(solver.u_hat)
            comp_sums.append((ui2, uij))
        if in_sampling and solver.t >= skew_next:
            u = solver.velocity_physical()
            skew_vals.append(derivative_skewness(u, grid))
            del u
            skew_next += sample_skewness_every
        if in_sampling and solver.t >= frame_next - 1e-9:
            frame_next += frame_dt
            # per-frame resolution gate (same helper + threshold as phase0): only
            # instantaneously-Class-I frames enter the corpus.
            keta_now = k_max_eta(grid.k_max_resolved, sc["eps"], cfg.solver.nu)
            frame_seen += 1
            if keta_now < 1.5:
                frame_skipped += 1
            else:
                rec_f = frame_fn(solver)        # {"u":.., "p":.., ["theta"/"b":..]} fp32 cpu
                rec_f.update({"t": solver.t, "frame": frame_idx, "k_max_eta": float(keta_now)})
                torch.save(rec_f, frame_dir / f"frame{frame_idx:03d}.pt")
                frame_idx += 1
        if checkpoint_every > 0 and solver.t >= ckpt_next:
            check_disk(out_dir, need_gb=2.0, where=f"{cfg.name} ckpt t={solver.t:.0f}")
            ck = {"t": solver.t, "u_hat": solver.u_hat.cpu(), "n_steps": solver.n_steps}
            if hasattr(solver, "b_hat"):
                ck["b_hat"] = solver.b_hat.cpu()
            if hasattr(solver, "theta_hat"):
                ck["theta_hat"] = solver.theta_hat.cpu()
            torch.save(ck, out_dir / f"ckpt_t{solver.t:08.2f}.pt")
            spec.save(); csv.flush()
            ckpt_next += checkpoint_every
        if eps_sentinel and solver.t > 5.0 and sentinel_ref > 0:
            if sc["eps"] < 0.2 * sentinel_ref:
                if sentinel_low_since is None:
                    sentinel_low_since = solver.t
                elif solver.t - sentinel_low_since > 4.0:
                    a14_aborted = True
                    print(f"[A14] eps<0.2 eps_W >2 T_E -> ABORT t={solver.t:.2f}", flush=True)
                    break
            else:
                sentinel_low_since = None
        if solver.n_steps % 2000 == 0:
            csv.flush()
            check_ram(f"ext run step {solver.n_steps}", abort_frac=0.92)
            print(f"  step {solver.n_steps:6d} t={solver.t:7.3f} K={sc['K']:.4f} "
                  f"eps={sc['eps']:.4f} "
                  f"({(time.perf_counter()-t0)/max(solver.n_steps,1)*1e3:.1f} ms/step)",
                  flush=True)

    csv.close(); spec.save()

    # Guard: if the run aborted (A14) or ended BEFORE any spectrum sample was taken
    # (sampling starts at t_spin_end; spectra only logged in the sampling window),
    # E_accum is None and there is no sampling-window data. Write a minimal metrics
    # marking the abort instead of crashing on E_accum/None (deep-review 2026-06-23:
    # a spin-up-phase A14 abort previously hit `None / int` TypeError).
    if E_accum is None or len(samp["t"]) < 2:
        metrics = {"experiment": experiment, "name": cfg.name, "N": cfg.solver.N,
                   "nu": cfg.solver.nu, "dtype": cfg.solver.dtype, "seed": cfg.seed,
                   "t_spinup": t_spinup, "t_sample": t_sample, "n_steps": solver.n_steps,
                   "a14_aborted": a14_aborted, "aborted_before_sampling": True,
                   "abort_t": solver.t, "wall_s": time.perf_counter() - t0}
        return metrics

    _N = grid.N
    _p = torch.fft.irfftn(pressure_hat(solver.u_hat, grid), s=(_N, _N, _N), dim=(-3, -2, -1))
    save_snapshot(out_dir, solver.t, solver.velocity_physical(), p_phys=_p)

    t_arr = np.asarray(samp["t"]); K_arr = np.asarray(samp["K"]); eps_arr = np.asarray(samp["eps"])
    E_mean = E_accum / max(n_spec, 1)
    k_arr = np.arange(1, len(E_mean) + 1, dtype=np.float64)
    summ = spectral_summary(k_arr, E_mean, cfg.solver.nu)
    T_L = summ["L"] / summ["u_rms"]
    pf = np.polyfit(t_arr, K_arr, 1)
    drift_pct = 100.0 * abs(pf[0] * 5.0 * T_L) / K_arr.mean()
    eps_mean = float(eps_arr.mean())
    inj_mean = float(np.asarray(samp["eps_inj"]).mean())
    closure_pct = (100.0 * abs(eps_mean - inj_mean) / inj_mean
                   if abs(inj_mean) > 1e-30 else float("nan"))
    kmax_eta = k_max_eta(grid.k_max_resolved, eps_mean, cfg.solver.nu)
    S3_mean = float(np.mean(skew_vals)) if skew_vals else float("nan")
    S3_std = float(np.std(skew_vals)) if skew_vals else float("nan")
    slope = fit_spectrum_slope(k_arr, E_mean, *SLOPE_WINDOW)

    iso = {}
    if comp_sums:
        ui2 = np.mean([c[0] for c in comp_sums], axis=0)
        uij = np.mean([c[1] for c in comp_sums], axis=0)
        K2 = 0.5 * ui2.sum()
        iso = {"iso_comp_pct": float(100.0 * max(abs(3.0*x/(2.0*K2)-1.0) for x in ui2)),
               "iso_cross_pct": float(100.0 * max(abs(x) for x in uij) / (2.0*K2/3.0)),
               "iso_n_samples": len(comp_sums),
               "ui2_mean": ui2.tolist(), "uij_mean": uij.tolist()}

    metrics = {
        "experiment": experiment, "name": cfg.name, "N": cfg.solver.N,
        "nu": cfg.solver.nu, "dtype": cfg.solver.dtype, "seed": cfg.seed,
        "eps_w": cfg.solver.forcing.eps_w, "t_spinup": t_spinup, "t_sample": t_sample,
        "T_L": T_L, "n_sample_T_L": t_sample / T_L, "K_mean": float(K_arr.mean()),
        "eps_mean": eps_mean, "eps_inj_mean": inj_mean, "u_rms": summ["u_rms"],
        "Re_lambda": summ["Re_lambda"], "eta": summ["eta"], "L_int": summ["L"],
        "K_drift_last5TL_pct": float(drift_pct),
        "K_fluctuation_pct": float(100.0 * K_arr.std() / K_arr.mean()),
        "energy_closure_pct": float(closure_pct), "k_max_eta": float(kmax_eta),
        "S3_mean": S3_mean, "S3_std": S3_std, "n_skew_samples": len(skew_vals),
        "slope_k4_20": slope, "n_steps": solver.n_steps,
        "wall_s": time.perf_counter() - t0,
        "ms_per_step": (time.perf_counter() - t0) / max(solver.n_steps, 1) * 1e3,
        "a14_aborted": a14_aborted, **iso,
        # corpus frame retention (same fields as run_forced_hit, for the manifest)
        "frames_exported": frame_idx, "frames_seen": frame_seen,
        "frames_skipped_underresolved": frame_skipped,
        "frame_keep_frac": float(frame_idx / frame_seen) if frame_seen else float("nan"),
    }
    # new-physics metrics; pass the AUTHORITATIVE spectral u_rms / L_int so Ro/Fr/Re_b
    # use the same integral scale as the rest of metrics.json (not a rough u^3/eps).
    metrics.update(extra_metrics_fn(summ))
    np.save(out_dir / "E_mean.npy", E_mean)
    return metrics
