"""Power-loss-resumable OU cross-validation production run.

Goal: a self-consistent time-averaged A10 (<=2%) from a full >=240 T_E OU run,
which needs the DENSE per-100-step component-moment series (checkpoints alone are
too sparse — they bottom out ~2.6-2.9%).

Power-loss hardening: every checkpoint also dumps the running accumulators
(comp_sums, spectra sum, skew) to accum_t*.npz. On restart, the script loads the
LATEST checkpoint + its accumulators and continues — so a power loss costs only
the work since the last checkpoint (<=30 t.u.), not the whole run. Re-running the
same command resumes automatically.

OU note: the OU forcing state (b_hat, RNG) is NOT in the checkpoint. On resume we
re-seed the OU process; statistically the forcing is stationary so a fresh OU
realization from the resumed field is valid (the run is one long stationary
sample, not a deterministic trajectory). The accumulators carry the A10/spectrum
statistics across the seam.
"""
import sys, time, json, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import solver._env  # noqa
import numpy as np
import torch
from solver.config import ForcingConfig, SolverConfig
from solver.solver import PseudoSpectralSolver
from solver.initial_conditions import random_solenoidal
from solver.grids import SpectralGrid
from solver.diagnostics import derivative_skewness, spectral_summary, k_max_eta
from solver.io_utils import CSVLogger

HERE = Path(__file__).parent
NU = 0.0022
CASE = "ou_xval_cybq_v2_256_fp64"
IC_SEED = 7
SPEC_EVERY = 100
CKPT_EVERY = 30.0


def latest_checkpoint(out):
    cks = sorted(glob.glob(str(out / "ckpt_t*.pt")))
    return cks[-1] if cks else None


def main(t_spinup=30.0, t_sample=470.0, case=None, ic_seed=None):
    case = case or CASE
    ic_seed = IC_SEED if ic_seed is None else ic_seed
    out = HERE / "results" / case
    out.mkdir(parents=True, exist_ok=True)
    # completion guard: if final metrics already exist, the run is DONE.
    # Exit 0 so a systemd auto-restart service does not loop-rerun a finished run.
    if (out / "metrics.json").exists():
        print(f"[{case}] already complete (metrics.json present); nothing to do.", flush=True)
        return
    dev = "cuda"
    grid = SpectralGrid(256, dev, "fp64")
    fcfg = ForcingConfig(type="stochastic_ou", k_f=2.0, ou_tau=2.0,
                         ou_sigma2=0.0725, ou_seed=1234)
    scfg = SolverConfig(N=256, nu=NU, dtype="fp64", device=dev, scheme="rk3",
                        cfl=0.4, dt_max=0.005, forcing=fcfg)
    t_end = t_spinup + t_sample

    ck = latest_checkpoint(out)
    if ck:
        d = torch.load(ck, map_location=dev, weights_only=False)
        s = PseudoSpectralSolver(scfg, d["u_hat"].to(dev))
        s.t = float(d["t"]); s.n_steps = int(d.get("n_steps", 0))
        acc = np.load(ck.replace("ckpt_t", "accum_t").replace(".pt", ".npz"), allow_pickle=True)
        comp_sums = list(acc["comp_sums"]); E_accum = acc["E_accum"]
        n_spec = int(acc["n_spec"]); skew_vals = list(acc["skew_vals"])
        print(f"[{case}] RESUME from {Path(ck).name} at t={s.t:.1f} "
              f"({n_spec} spec samples carried)", flush=True)
        mode = "a"
    else:
        u0 = random_solenoidal(grid, seed=ic_seed, k_p=3.0, u_rms=0.7)
        s = PseudoSpectralSolver(scfg, u0)
        comp_sums, E_accum, n_spec, skew_vals = [], None, 0, []
        print(f"[{case}] FRESH start; spin-up {t_spinup} + sample {t_sample}", flush=True)
        mode = "w"

    csv = CSVLogger(out / "timeseries.csv",
                    ["step", "t", "dt", "K", "eps", "eps_inj", "umax", "wall_s"]) \
        if mode == "w" else _append_csv(out / "timeseries.csv")
    skew_next = max(s.t, t_spinup)
    ckpt_next = (int(s.t / CKPT_EVERY) + 1) * CKPT_EVERY
    sent_low_since, a14 = None, False
    t0 = time.perf_counter()

    while s.t < t_end - 1e-9:
        dt = s.suggest_dt(t_target=t_end)
        s.step(dt)
        K = s.grid.kinetic_energy(s.u_hat); eps = s.grid.dissipation(s.u_hat, NU)
        csv.log({"step": s.n_steps, "t": s.t, "dt": dt, "K": K, "eps": eps,
                 "eps_inj": s.last_eps_inj, "umax": s.last_umax,
                 "wall_s": time.perf_counter()-t0})
        if s.t >= t_spinup and s.n_steps % SPEC_EVERY == 0:
            k, E = s.grid.shell_spectrum(s.u_hat)
            E_accum = E.numpy() if E_accum is None else E_accum + E.numpy()
            n_spec += 1
            comp_sums.append(s.grid.component_moments(s.u_hat))
        if s.t >= t_spinup and s.t >= skew_next:
            u = torch.fft.irfftn(s.u_hat, s=(256,)*3, dim=(-3,-2,-1))
            skew_vals.append(derivative_skewness(u, s.grid)); del u; skew_next += 2.0
        if s.t >= ckpt_next:
            torch.save({"t": s.t, "u_hat": s.u_hat.cpu(), "n_steps": s.n_steps},
                       out / f"ckpt_t{s.t:08.2f}.pt")
            np.savez(out / f"accum_t{s.t:08.2f}.npz",
                     comp_sums=np.array(comp_sums, dtype=object),
                     E_accum=E_accum if E_accum is not None else np.zeros(1),
                     n_spec=n_spec, skew_vals=np.array(skew_vals))
            csv.flush(); ckpt_next += CKPT_EVERY
            print(f"  ckpt t={s.t:.1f} K={K:.3f} eps={eps:.4f} n_spec={n_spec}", flush=True)
        if eps < 0.2 * 0.096:
            sent_low_since = sent_low_since or s.t
            if s.t - sent_low_since > 4.0:
                a14 = True; print(f"[A14] ABORT t={s.t:.1f}", flush=True); break
        else:
            sent_low_since = None
    csv.close()

    # final metrics with time-averaged A10
    E_mean = E_accum / max(n_spec, 1); np.save(out / "E_mean.npy", E_mean)
    k_arr = np.arange(1, len(E_mean)+1, dtype=np.float64)
    summ = spectral_summary(k_arr, E_mean, NU); T_L = summ["L"]/summ["u_rms"]
    ui2 = np.mean([c[0] for c in comp_sums], axis=0)
    uij = np.mean([c[1] for c in comp_sums], axis=0); K2 = 0.5*ui2.sum()
    iso_comp = 100.0*max(abs(3.0*x/(2.0*K2)-1.0) for x in ui2)
    iso_cross = 100.0*max(abs(x) for x in uij)/(2.0*K2/3.0)
    ts = np.genfromtxt(out / "timeseries.csv", delimiter=",", names=True)
    msk = ts["t"] >= t_spinup
    eps_mean = float(ts["eps"][msk].mean())
    kme = k_max_eta(85, eps_mean, NU)
    metrics = {"name": case, "N": 256, "nu": NU, "dtype": "fp64",
               "forcing": "stochastic_ou", "k_f": 2.0, "ou_sigma2": 0.0725,
               "eps_w": 0.1, "t_spinup": t_spinup, "t_sample": t_sample,
               "T_L": T_L, "n_sample_T_L": t_sample/T_L,
               "K_mean": float(ts["K"][msk].mean()), "eps_mean": eps_mean,
               "eps_inj_mean": float(ts["eps_inj"][msk].mean()),
               "u_rms": summ["u_rms"], "Re_lambda": summ["Re_lambda"],
               "eta": summ["eta"], "L_int": summ["L"], "k_max_eta": float(kme),
               "energy_closure_pct": 100.0*abs(eps_mean-float(ts["eps_inj"][msk].mean()))/abs(float(ts["eps_inj"][msk].mean())),
               "iso_comp_pct": float(iso_comp), "iso_cross_pct": float(iso_cross),
               "iso_n_samples": len(comp_sums),
               # raw time-averaged moments so eval_dns_standard.py uses the POOLED
               # (time-averaged) A10 path, not its per-snapshot fallback (which
               # over-reads ~8% finite-volume fluctuation as anisotropy).
               "ui2_mean": ui2.tolist(), "uij_mean": uij.tolist(),
               "S3_mean": float(np.mean(skew_vals)), "a14_aborted": a14}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(f"\n=== [{case}] done: Re_l={summ['Re_lambda']:.1f} k_max_eta={kme:.3f} "
          f"A10={iso_comp:.2f}%/{iso_cross:.2f}% ({len(comp_sums)} samples) a14={a14} ===", flush=True)


def _append_csv(path):
    import csv as _csv
    class _A:
        def __init__(s,p): s.fh=open(p,"a",newline=""); s.w=_csv.DictWriter(s.fh,
            fieldnames=["step","t","dt","K","eps","eps_inj","umax","wall_s"])
        def log(s,row): s.w.writerow(row)
        def flush(s): s.fh.flush()
        def close(s): s.fh.close()
    return _A(path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="results subdir / metrics name")
    ap.add_argument("--ic-seed", type=int, default=None, help="initial-condition seed")
    a = ap.parse_args()
    main(case=a.case, ic_seed=a.ic_seed)
