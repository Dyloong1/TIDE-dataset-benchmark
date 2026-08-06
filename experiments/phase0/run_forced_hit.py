"""Phase-0 experiment 2: forced homogeneous isotropic turbulence at Re_lambda ~ 100.

Spin-up to statistical stationarity, then sample ~20 large-eddy turnover times
and check acceptance criteria 2-4:
  2. kinetic-energy drift over the last 5 T_L < 1%; |<eps> - eps_W|/eps_W < 2%
  3. k_max * eta >= 1.5
  4. derivative skewness in [-0.55, -0.45]; Kolmogorov-normalized spectrum
     overlays JHTDB (slope difference < 0.05 in the shared k*eta window)

Also supports nu calibration (knowledge base 2.4):
    python run_forced_hit.py --config configs/hit_relam100_256.yaml --calibrate 100
runs a shortened simulation, measures Re_lambda, and proposes
nu_new = nu_old * (Re_lambda_measured / target)^2   (Re_lambda ~ nu^-1/2 at
fixed injection rate and forcing scale).
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Windows console defaults to cp1252 and UnicodeEncodeErrors on any non-ASCII
# print (eta, em-dash, etc.), which crashes the run AFTER the science finished and
# makes run_corpus.ps1 quarantine the seed as "FAILED". Force UTF-8 so a console
# print can never abort a long run. (Same guard the eval scripts already use.)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import solver._env  # noqa: F401
import numpy as np
import torch

from solver.memguard import check_ram
from solver.config import dump_config, load_config
from solver.diagnostics import (derivative_skewness, fit_spectrum_slope,
                                k_max_eta, kolmogorov_normalize, spectral_summary)
from solver.io_utils import (CSVLogger, SpectrumLogger, save_snapshot,
                             write_meta, write_metrics)
from solver.solver import build_solver

JHTDB_TRUTH = (Path(__file__).resolve().parent
               / "reference" / "jhtdb_isotropic1024_truth_3frames.json")
JHTDB_NU = 1.85e-4
SLOPE_WINDOW_256 = (4.0, 20.0)   # knowledge base: fixed fit window for 256^3


def run(cfg, out_dir: Path, t_spinup: float, t_sample: float,
        sample_skewness_every: float = 2.0, checkpoint_every: float = 0.0,
        eps_sentinel: bool = False, resume_ckpt: str = "",
        frame_every_tl: float = 0.0, frame_T_L: float = 0.0) -> dict:
    """frame_every_tl>0: also export 4-channel (u,v,w,p) fp32 corpus frames every
    frame_every_tl*frame_T_L in the sampling window (the run still produces the
    full acceptance structure, so the same trajectory is BOTH accepted and the
    frame source -- no unvetted frames). Pressure uses the D4-certified
    pressure_hat. fp32 store is safe (A12 ~1e-13 after round trip, verified)."""
    s = build_solver(cfg)
    # Optional warm start: replace the IC field with a saved checkpoint. Used for
    # free-decay runs (a matured turbulent state + forcing.type=none) and as
    # power-loss / OOM resilience for long runs. All time bookkeeping below is
    # offset by t_start so the run goes FORWARD from the checkpoint time.
    t_start = 0.0
    if resume_ckpt:
        ck = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        s.u_hat.copy_(ck["u_hat"].to(s.grid.device, s.grid.cdtype))
        s.t = t_start = float(ck["t"])
        s.n_steps = int(ck.get("n_steps", 0))
        # refresh the first-step dt (umax) from the loaded field
        u0 = s.velocity_physical()
        s.last_umax = float(u0.abs().sum(dim=0).max())
        del u0
        print(f"[resume] warm start from {Path(resume_ckpt).name} at t={t_start:.2f}",
              flush=True)
        # PROVENANCE: record WHICH checkpoint this run started from. dump_config() only
        # serializes the ExperimentConfig dataclass, and resume_ckpt is a CLI arg, not a config
        # field -- so a warm-started run used to leave NO trace of its own starting state. That
        # is how decay_hotstart_re86 shipped 8 "seeds" that are 8 bit-identical copies of one
        # trajectory: its yaml says "ic/seed fields are ignored" (it resumes a checkpoint), the
        # 8 pools varied only the seed LABEL while resuming the SAME state, and nothing in the
        # output recorded that they shared a source. Write it next to the config snapshot, with
        # a hash, so two runs claiming to be different seeds can always be told apart.
        import hashlib as _hashlib
        _src = Path(resume_ckpt)
        try:
            _h = _hashlib.md5(_src.read_bytes()).hexdigest()
        except Exception:
            _h = "unreadable"
        (out_dir / "resume_provenance.json").write_text(json.dumps({
            "resume_ckpt": str(_src.resolve()),
            "resume_ckpt_md5": _h,
            "t_start": t_start,
            "note": ("This run was WARM-STARTED. Its initial state came from the checkpoint above, "
                     "NOT from cfg.ic/cfg.seed (those are ignored on a resume). Two runs with "
                     "different seed labels but the same resume_ckpt_md5 are the SAME trajectory."),
        }, indent=2), encoding="utf-8")
    csv = CSVLogger(out_dir / "timeseries.csv",
                    ["step", "t", "dt", "K", "eps", "eps_inj", "umax", "wall_s"])
    spec = SpectrumLogger(out_dir / "spectra.npz")

    # all phase boundaries are relative to t_start (0 for a fresh run)
    t_spin_end = t_start + t_spinup
    t_end = t_start + t_spinup + t_sample
    t0 = time.perf_counter()
    # accumulators over the sampling window
    samp = {"t": [], "K": [], "eps": [], "eps_inj": []}
    comp_sums = []   # time series of (<u_i^2>, <u_i u_j>) for isotropy (A10)
    ckpt_next = (t_start + checkpoint_every) if checkpoint_every > 0 else float("inf")
    # corpus frame export (4-channel fp32) at frame_every_tl * T_L cadence
    frame_dt = frame_every_tl * frame_T_L if (frame_every_tl > 0 and frame_T_L > 0) else float("inf")
    frame_next = t_spin_end if frame_dt != float("inf") else float("inf")
    frame_idx = 0
    frame_seen = 0      # frame slots reached in the sampling window
    frame_skipped = 0   # slots skipped because instantaneous k_maxη < 1.5
    frame_dir = out_dir / "frames"
    from solver.diskguard import check_disk, estimate_run_bytes
    n_frame_est = int(t_sample / frame_dt) + 2 if frame_dt != float("inf") else 0
    if frame_dt != float("inf"):
        frame_dir.mkdir(exist_ok=True)
        from solver.operators import pressure_hat   # D4-certified pressure
        # guard the frame volume specifically (may be a separate D: drive from
        # the checkpoint/out_dir volume — CYB-t's corpus pipeline writes frames
        # to D:). This is the frame-export half of the shared guard.
        check_disk(frame_dir, need_gb=estimate_run_bytes(
            cfg.solver.N, 0, cfg.solver.dtype, n_frame_est) / 1e9,
            where=f"{cfg.name} frames")
    # diskguard: refuse to start if there isn't room for the full output (the
    # 2026-06-19 incident filled the root disk mid-run and corrupted metrics).
    n_ckpt_est = int(t_sample / checkpoint_every) + 2 if checkpoint_every > 0 else 0
    need_gb = estimate_run_bytes(cfg.solver.N, n_ckpt_est, cfg.solver.dtype,
                                 n_frame_est) / 1e9
    check_disk(out_dir, need_gb=need_gb, where=f"{cfg.name} run start")
    # A14 sentinel: eps/eps_W continuously < 0.2 for 2 T_E (~4 t.u.) -> abort
    sentinel_ref = cfg.solver.forcing.eps_w
    sentinel_low_since = None
    a14_aborted = False
    skew_vals, skew_next = [], t_spin_end
    E_accum, n_spec = None, 0

    while s.t < t_end - 1e-12:
        dt = s.suggest_dt(t_target=t_end)
        s.step(dt)
        sc = s.scalars()
        if s.n_steps % cfg.log_every_steps == 0:
            csv.log({"step": s.n_steps, "dt": dt,
                     "wall_s": time.perf_counter() - t0, **sc})
        in_sampling = s.t >= t_spin_end
        if in_sampling:
            samp["t"].append(s.t); samp["K"].append(sc["K"])
            samp["eps"].append(sc["eps"]); samp["eps_inj"].append(sc["eps_inj"])
        if s.n_steps % cfg.spectrum_every_steps == 0 and in_sampling:
            k, E = s.grid.shell_spectrum(s.u_hat)
            spec.log(s.t, k.numpy(), E.numpy())
            E_accum = E.numpy() if E_accum is None else E_accum + E.numpy()
            n_spec += 1
            ui2, uij = s.grid.component_moments(s.u_hat)
            comp_sums.append((ui2, uij))
        if in_sampling and s.t >= skew_next:
            u = s.velocity_physical()
            skew_vals.append(derivative_skewness(u, s.grid))
            del u
            skew_next += sample_skewness_every
        if checkpoint_every > 0 and s.t >= ckpt_next:
            check_disk(out_dir, need_gb=2.0, where=f"{cfg.name} checkpoint t={s.t:.0f}")
            torch.save({"t": s.t, "u_hat": s.u_hat.cpu(), "n_steps": s.n_steps},
                       out_dir / f"ckpt_t{s.t:08.2f}.pt")
            spec.save()
            csv.flush()
            ckpt_next += checkpoint_every
        if in_sampling and s.t >= frame_next - 1e-9:
            frame_next += frame_dt
            # Per-FRAME resolution gate: only export frames that are INSTANTANEOUSLY
            # Class I (k_maxη>=1.5). A window-averaged k_maxη~1.57 can still dip
            # below 1.5 for ~12% of the window (eps fluctuation), and those frames
            # would teach a foundation model under-resolved small scales. This is
            # the per-frame extension of the per-trajectory gate (the trajectory is
            # still ACCEPTed by window-average class; this just curates which frames
            # enter the corpus). k_max_eta uses the same helper as the metrics.
            keta_now = k_max_eta(s.grid.k_max_resolved, sc["eps"], cfg.solver.nu)
            frame_seen += 1
            if keta_now < 1.5:
                frame_skipped += 1
            else:
                # 4-channel (u,v,w,p) fp32 corpus frame. Pressure from D4-certified
                # pressure_hat. The trajectory also produces the full acceptance
                # structure (spectra/scalars/checkpoints), so frames are never unvetted.
                _Nf = s.grid.N
                _pf = torch.fft.irfftn(pressure_hat(s.u_hat, s.grid), s=(_Nf,) * 3, dim=(-3, -2, -1))
                torch.save({"t": s.t, "frame": frame_idx, "k_max_eta": float(keta_now),
                            "u": s.velocity_physical().to(torch.float32).cpu(),
                            "p": _pf.to(torch.float32).cpu()},
                           frame_dir / f"frame{frame_idx:03d}.pt")
                del _pf
                frame_idx += 1
        if eps_sentinel and s.t > t_start + 5.0:
            if sc["eps"] < 0.2 * sentinel_ref:
                if sentinel_low_since is None:
                    sentinel_low_since = s.t
                elif s.t - sentinel_low_since > 4.0:
                    a14_aborted = True
                    print(f"[A14 SENTINEL] eps/eps_W < 0.2 for > 2 T_E "
                          f"(since t={sentinel_low_since:.2f}) — ABORT at t={s.t:.2f}",
                          flush=True)
                    break
            else:
                sentinel_low_since = None
        if s.n_steps % 2000 == 0:
            csv.flush()
            # Host-RAM guard on the long run: a 470 t.u. trajectory is ~90k
            # steps; if anything leaks host memory we abort cleanly (flushing
            # the CSV/spectra first) instead of letting the OS OOM-reboot the
            # box (the failure that lost the previous working directory).
            check_ram(f"long run step {s.n_steps}", abort_frac=0.92)
            print(f"  step {s.n_steps:6d}  t={s.t:7.3f}  K={sc['K']:.4f} "
                  f"eps={sc['eps']:.4f}  "
                  f"({(time.perf_counter() - t0) / s.n_steps * 1e3:.1f} ms/step)",
                  flush=True)

    csv.close()
    spec.save()
    # 4-channel snapshot: velocity + kinematic pressure (corpus channel 4).
    # pressure_hat is the same solver certified by D4 / tests/test_pressure.
    from solver.operators import pressure_hat
    _N = s.grid.N
    _p = torch.fft.irfftn(pressure_hat(s.u_hat, s.grid), s=(_N, _N, _N), dim=(-3, -2, -1))
    save_snapshot(out_dir, s.t, s.velocity_physical(), p_phys=_p)

    t_arr = np.asarray(samp["t"]); K_arr = np.asarray(samp["K"]); eps_arr = np.asarray(samp["eps"])
    E_mean = E_accum / max(n_spec, 1)
    k_arr = np.arange(1, len(E_mean) + 1, dtype=np.float64)
    summ = spectral_summary(k_arr, E_mean, cfg.solver.nu)

    # T_L from the measured integral scale and u'
    T_L = summ["L"] / summ["u_rms"]
    # criterion 2a: secular drift of K, expressed over 5 T_L.
    # Band-forced HIT breathes at the integral scale with O(10 T_L)
    # correlation time, so a linear fit over only the last 5 T_L aliases that
    # natural oscillation into "drift". Primary metric: linear trend fitted
    # over the FULL sampling window, quoted as percent change per 5 T_L.
    # The raw last-5TL fit and half-window means are reported for transparency.
    pf = np.polyfit(t_arr, K_arr, 1)
    drift_pct = 100.0 * abs(pf[0] * 5.0 * T_L) / K_arr.mean()
    m5 = t_arr >= (t_arr[-1] - 5.0 * T_L)
    p5 = np.polyfit(t_arr[m5], K_arr[m5], 1)
    drift_last5_pct = 100.0 * abs(p5[0] * 5.0 * T_L) / K_arr[m5].mean()
    h = len(K_arr) // 2
    drift_halves_pct = 100.0 * abs(K_arr[h:].mean() - K_arr[:h].mean()) / K_arr.mean()
    # criterion 2b: energy budget closure — <eps> against the realized mean
    # injection rate (== eps_w for negative damping; measured from the
    # band-rescaling increments for energy_preserving forcing)
    eps_mean = float(eps_arr.mean())
    inj_mean = float(np.asarray(samp["eps_inj"]).mean())
    # Energy closure is meaningless without injection (free decay, forcing=none):
    # inj_mean == 0 -> report NaN instead of dividing by zero. Decaying runs are
    # judged by the decay law (eval_decaying.py), not A8.
    closure_pct = (100.0 * abs(eps_mean - inj_mean) / inj_mean
                   if abs(inj_mean) > 1e-30 else float("nan"))
    # criterion 3
    kmax_eta = k_max_eta(s.grid.k_max_resolved, eps_mean, cfg.solver.nu)
    # criterion 4a
    S3_mean = float(np.mean(skew_vals)) if skew_vals else float("nan")
    S3_std = float(np.std(skew_vals)) if skew_vals else float("nan")
    # criterion 4b: spectrum slope in fixed window + JHTDB overlay
    slope = fit_spectrum_slope(k_arr, E_mean, *SLOPE_WINDOW_256)
    jhtdb = compare_jhtdb(k_arr, E_mean, eps_mean, cfg.solver.nu)

    # component isotropy from time-averaged moments (DNS standard A10)
    iso = {}
    if comp_sums:
        ui2 = np.mean([c[0] for c in comp_sums], axis=0)
        uij = np.mean([c[1] for c in comp_sums], axis=0)
        K2 = 0.5 * ui2.sum()
        iso = {
            "iso_comp_pct": float(100.0 * max(abs(3.0 * x / (2.0 * K2) - 1.0) for x in ui2)),
            "iso_cross_pct": float(100.0 * max(abs(x) for x in uij) / (2.0 * K2 / 3.0)),
            "iso_n_samples": len(comp_sums),
            "ui2_mean": ui2.tolist(),
            "uij_mean": uij.tolist(),
        }

    metrics = {
        "experiment": "forced_hit",
        "name": cfg.name,
        "N": cfg.solver.N,
        "nu": cfg.solver.nu,
        "dtype": cfg.solver.dtype,
        "seed": cfg.seed,
        "ou_seed": getattr(cfg.solver.forcing, "ou_seed", None),  # effective (per-IC) forcing seed
        "eps_w": cfg.solver.forcing.eps_w,
        "t_spinup": t_spinup,
        "t_sample": t_sample,
        "T_L": T_L,
        "n_sample_T_L": t_sample / T_L,
        "K_mean": float(K_arr.mean()),
        "eps_mean": eps_mean,
        "eps_inj_mean": inj_mean,
        "u_rms": summ["u_rms"],
        "Re_lambda": summ["Re_lambda"],
        "eta": summ["eta"],
        "L_int": summ["L"],
        # NAMING DEBT (2026-07-08 review): despite the name "last5TL", this field is the
        # FULL-window linear-fit drift (drift_pct, line ~224), NOT a last-5-T_L fit — the full
        # window is intentional (see :217-222: a last-5TL fit aliases the OU breathing). The
        # genuine last-5TL fit is the separate field below (K_drift_raw_last5TL_fit_pct).
        "K_drift_last5TL_pct": float(drift_pct),
        "K_drift_raw_last5TL_fit_pct": float(drift_last5_pct),
        "K_drift_half_means_pct": float(drift_halves_pct),
        "K_fluctuation_pct": float(100.0 * K_arr.std() / K_arr.mean()),
        "energy_closure_pct": float(closure_pct),
        "k_max_eta": float(kmax_eta),
        "S3_mean": S3_mean,
        "S3_std": S3_std,
        "n_skew_samples": len(skew_vals),
        "slope_k4_20": slope,
        "jhtdb": jhtdb,
        "n_steps": s.n_steps,
        "wall_s": time.perf_counter() - t0,
        "ms_per_step": (time.perf_counter() - t0) / max(s.n_steps, 1) * 1e3,
        "a14_aborted": a14_aborted,
        "frames_exported": frame_idx,
        "frames_seen": frame_seen,
        "frames_skipped_underresolved": frame_skipped,
        "frame_keep_frac": float(frame_idx / frame_seen) if frame_seen else float("nan"),
        **iso,
    }
    if frame_dt != float("inf"):
        # ASCII only: Windows console is cp1252 and would UnicodeEncodeError on
        # a Greek eta here, crashing the whole run AFTER the trajectory completed.
        print(f"[frames] exported {frame_idx}/{frame_seen} "
              f"(skipped {frame_skipped} with instantaneous k_max*eta<1.5)", flush=True)
    np.save(out_dir / "E_mean.npy", E_mean)
    return metrics


def compare_jhtdb(k: np.ndarray, E: np.ndarray, eps: float, nu: float) -> dict:
    """Overlay our Kolmogorov-normalized spectrum with JHTDB's and compare
    log-log slopes in the shared k*eta window (user-approved protocol)."""
    if not JHTDB_TRUTH.exists():
        return {"available": False}
    with open(JHTDB_TRUTH, "r", encoding="utf-8") as fh:
        truth = json.load(fh)
    frames = [truth[f] for f in ("900", "940", "980") if f in truth]
    kj = np.asarray(frames[0]["wavenumbers"], dtype=np.float64)
    Ej = np.mean([np.asarray(f["E_k"], dtype=np.float64) for f in frames], axis=0)
    epsj = float(np.mean([f["epsilon"] for f in frames]))

    x_ours, y_ours = kolmogorov_normalize(k, E, eps, nu)
    x_j, y_j = kolmogorov_normalize(kj, Ej, epsj, JHTDB_NU)

    # Shared *universal-range* window: a common local slope can only be
    # expected where both flows are beyond their energy-containing scales
    # (Re_lambda 87 vs 417 differ everywhere else, and our |k|<=2 shells are
    # pinned by the forcing). Lower edge = 2x the spectral-peak wavenumber of
    # either flow in Kolmogorov units; upper edge 0.25 (dissipation knee).
    eta_ours = (nu**3 / eps) ** 0.25
    k_peak = float(k[np.argmax(E)])
    eta_j = (JHTDB_NU**3 / epsj) ** 0.25
    k_peak_j = float(kj[np.argmax(Ej)])
    lo = max(2.0 * k_peak * eta_ours, 2.0 * k_peak_j * eta_j,
             float(x_ours.min()), float(x_j.min()))
    hi = min(0.25, float(x_ours.max()), float(x_j.max()))

    def slope_of(x, y, lo_, hi_):
        m = (x >= lo_) & (x <= hi_) & (y > 0)
        if m.sum() < 3:
            return float("nan")
        b, _ = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
        return float(b)

    s_ours = slope_of(x_ours, y_ours, lo, hi)
    s_j = slope_of(x_j, y_j, lo, hi)
    # window-sensitivity record (transparency: the overlay plot shows shape)
    sensitivity = {}
    for lo_alt, hi_alt in ((lo, hi), (0.10, 0.25), (0.10, 0.30), (0.075, 0.25)):
        so = slope_of(x_ours, y_ours, lo_alt, hi_alt)
        sj = slope_of(x_j, y_j, lo_alt, hi_alt)
        sensitivity[f"[{lo_alt:.3f},{hi_alt:.3f}]"] = round(abs(so - sj), 4)
    return {
        "available": True,
        "keta_window": [float(lo), float(hi)],
        "window_rule": "lo = 2*k_peak*eta of either flow (beyond energy-containing scales); hi = 0.25",
        "slope_ours": s_ours,
        "slope_jhtdb": s_j,
        "slope_diff": abs(s_ours - s_j),
        "slope_diff_window_sensitivity": sensitivity,
        "eps_jhtdb": epsj,
        "Re_lambda_jhtdb": float(np.mean([f["derived"]["Re_lambda"] for f in frames])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--t-spinup", type=float, default=20.0)
    ap.add_argument("--t-sample", type=float, default=40.0)
    ap.add_argument("--calibrate", type=float, default=0.0, metavar="RE_TARGET",
                    help="short run; report measured Re_lambda and proposed nu")
    ap.add_argument("--checkpoint-every", type=float, default=0.0,
                    help="save spectral state + flush spectra every T time units")
    ap.add_argument("--seed", type=int, default=-1,
                    help="override the config's initial-condition seed")
    ap.add_argument("--ou-seed", type=int, default=-1,
                    help="override the OU forcing seed (the random forcing sequence). "
                         "MUST differ per pool seed for a true IC-ensemble: a shared "
                         "ou_seed gives every seed the SAME forcing realization, which "
                         "imprints a common <u_i u_j> direction bias that signed A10 "
                         "pooling cannot cancel (proven: #1's 16 seeds all had <vw> > 0). "
                         "Use --ou-seed-per-ic to derive it from --seed automatically.")
    ap.add_argument("--ou-seed-per-ic", action="store_true",
                    help="set ou_seed = (config ou_seed) + (IC seed). Gives each pool "
                         "seed an INDEPENDENT forcing sequence so cross-correlation signs "
                         "randomize and signed A10 pooling cancels to the isotropic value. "
                         "no-op for forcing.type=none (free decay).")
    ap.add_argument("--eps-sentinel", action="store_true",
                    help="A14: abort if eps/eps_W stays < 0.2 for > 2 T_E")
    ap.add_argument("--resume-from-ckpt", default="",
                    help="warm-start from a saved checkpoint (.pt) instead of the "
                         "IC; run goes forward from the checkpoint time. Used for "
                         "free-decay (forcing.type=none) and power-loss resilience.")
    ap.add_argument("--frame-every-TL", type=float, default=0.0,
                    help="export 4-channel (u,v,w,p) fp32 corpus frames every this "
                         "many T_L in the sampling window (needs --T-L)")
    ap.add_argument("--T-L", type=float, default=0.0,
                    help="large-eddy turnover time, for the frame cadence")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed >= 0:
        cfg.seed = args.seed
    # OU forcing seed: for a true IC-ensemble each pool seed needs an INDEPENDENT forcing
    # realization, else all seeds share one <u_i u_j> direction bias (signed A10 pooling
    # cannot cancel a COMMON bias -> A10 stuck ~2%; proven by #1's <vw> all-positive over
    # 16 seeds). --ou-seed-per-ic derives a distinct ou_seed per IC seed; --ou-seed sets it
    # explicitly. Both are no-ops physically for forcing.type=none (free decay ignores it).
    if getattr(cfg.solver.forcing, "type", "none") != "none":
        if args.ou_seed >= 0:
            cfg.solver.forcing.ou_seed = args.ou_seed
        elif args.ou_seed_per_ic:
            cfg.solver.forcing.ou_seed = int(cfg.solver.forcing.ou_seed) + int(cfg.seed)
        print(f"[forcing] ou_seed = {cfg.solver.forcing.ou_seed} "
              f"(IC seed = {cfg.seed}) — independent forcing per seed"
              if (args.ou_seed >= 0 or args.ou_seed_per_ic) else
              f"[forcing] ou_seed = {cfg.solver.forcing.ou_seed} (shared — NOT per-seed)",
              flush=True)
    suffix = "_cal" if args.calibrate else ""
    out_dir = Path(args.out or cfg.out_dir or f"results/{cfg.name}{suffix}")
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir)
    write_meta(out_dir, dataclasses.asdict(cfg))

    if args.calibrate:
        t_spin, t_samp = 15.0, 10.0   # short: enough for a Re_lambda estimate
    else:
        t_spin, t_samp = args.t_spinup, args.t_sample

    print(f"[{cfg.name}{suffix}] N={cfg.solver.N} nu={cfg.solver.nu} "
          f"eps_w={cfg.solver.forcing.eps_w} spinup={t_spin} sample={t_samp}",
          flush=True)
    metrics = run(cfg, out_dir, t_spin, t_samp,
                  checkpoint_every=args.checkpoint_every,
                  eps_sentinel=args.eps_sentinel,
                  resume_ckpt=args.resume_from_ckpt,
                  frame_every_tl=args.frame_every_TL, frame_T_L=args.T_L)

    if args.calibrate:
        re_meas = metrics["Re_lambda"]
        nu_new = cfg.solver.nu * (re_meas / args.calibrate) ** 2
        metrics["calibration"] = {
            "Re_lambda_target": args.calibrate,
            "Re_lambda_measured": re_meas,
            "nu_current": cfg.solver.nu,
            "nu_proposed": nu_new,
        }
        print(f"  calibration: Re_lambda={re_meas:.1f} (target {args.calibrate}) "
              f"-> nu {cfg.solver.nu:.4e} -> {nu_new:.4e}")

    write_metrics(out_dir, metrics)
    print(f"[{cfg.name}{suffix}] done: Re_lambda={metrics['Re_lambda']:.1f} "
          f"k_max*eta={metrics['k_max_eta']:.2f} S3={metrics['S3_mean']:.3f} "
          f"drift={metrics['K_drift_last5TL_pct']:.2f}% "
          f"closure={metrics['energy_closure_pct']:.2f}% "
          f"({metrics['ms_per_step']:.1f} ms/step)")


if __name__ == "__main__":
    main()
