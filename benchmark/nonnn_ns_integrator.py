"""ns_integrator — the physics-ceiling non-NN rollout baseline (G2, 2026-07-16).

Runs the PRODUCTION pseudo-spectral solver (rk4 + integrating-factor viscosity, the same code
that generated the corpus) forward from each eval input frame WITH NO FORCING, and scores the
resulting 20-frame trajectory in the official rollout protocol. It is "perfect physics minus the
hidden OU forcing state": the gap between this baseline and the truth measures exactly
(unobservable forcing) + (chaotic divergence), which is the floor any learned model should be
compared against — persistence is a data-autocorrelation lower bound, not a physical one.

Why not a single Euler step per frame: one frame interval = O(250) solver steps; explicit Euler
at that dt violates CFL by ~2 orders and explodes within a few AR steps. The honest integrator
is the solver itself.

Scope v1: isotropic forced family + decay (4-channel u,v,w,p). Rotating needs the Coriolis term
and scalar/stratified need a 5th-channel equation — the production solver runs those through
config machinery this script does not replicate; it REFUSES those cases rather than integrating
the wrong equation set (same rule as eval's buoyant-pressure refusal).

Precision: fp32 by default. The known fp32 truncation-scale instability sets in at ~100 T_L;
these rollouts are 1 T_L. Pass --precision fp64 to double-check (≈2x cost).

Cost: ~250 solver steps/frame x 20 frames ~ minutes per sample at 256^3 — hence --samples
(default 8; the baseline is deterministic physics, sample variance is small and n is recorded
in the row). Validate speed on ONE sample before sweeping (--samples 1).

Output: <out>/ns_integrator__rollout__<case>.json in the official row schema, so
aggregate_leaderboard.py picks it up unchanged.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import solver  # noqa: F401  (sets KMP env before torch import — repo iron rule)
import torch
import yaml

import benchmark_metrics as BM
from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset
from solver.config import SolverConfig, ForcingConfig
from solver.solver import PseudoSpectralSolver
from solver.operators import pressure_hat

_FFT_DIMS = (-3, -2, -1)

# cases whose dynamics this script's equation set (incompressible NS, no body force beyond
# viscosity) actually matches. Refuse everything else — wrong-equation numbers are worse
# than no numbers.
_SUPPORTED_PREFIXES = ("ou_robust_", "ou_relam", "decay_", "helical_")


def _nu_for_case(case: str) -> float:
    cfg_dir = HERE.parent / "experiments" / "phase0" / "configs"
    y = cfg_dir / f"{case.replace('_256_fp64', '')}_256_fp64.yaml"
    if not y.exists():
        y = cfg_dir / f"{case}.yaml"
    if not y.exists():
        # decay_hotstart_re86 has no _256_fp64 suffix in its yaml name
        y = cfg_dir / (case.split("_256_fp64")[0] + ".yaml")
    if not y.exists():
        raise SystemExit(f"[ns_integrator] no config yaml found for case {case} under {cfg_dir}")
    cfg = yaml.safe_load(y.read_text())
    nu = cfg.get("solver", {}).get("nu", cfg.get("nu"))
    if nu is None:
        raise SystemExit(f"[ns_integrator] yaml {y} has no solver.nu")
    return float(nu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-out", type=int, default=20)
    ap.add_argument("--frame-stride", type=int, default=8)   # official eval sampling
    ap.add_argument("--samples", type=int, default=8,
                    help="eval samples (deterministic baseline -> small variance; n recorded)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp64"])
    ap.add_argument("--scheme", default="rk4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--decay", action="store_true", help="use low-K-safe nRMSE (decay iron rule)")
    args = ap.parse_args()

    if not args.case.startswith(_SUPPORTED_PREFIXES):
        raise SystemExit(
            f"[ns_integrator] case {args.case} needs equation terms (Coriolis / scalar / buoyancy) "
            f"this baseline does not integrate. Refusing (wrong-equation numbers are worse than "
            f"none). Supported prefixes: {_SUPPORTED_PREFIXES}")

    nu = _nu_for_case(args.case)
    dev = torch.device(args.device)
    ds = TurbgenZarrDataset(case=args.case, split="test",
                            slice_manifest=args.slice, data_root=args.data_root,
                            task="rollout", t_in=1, t_out=args.t_out,
                            frame_stride=args.frame_stride, patch_size=None, normalize=True)
    if ds.norm is None:
        raise SystemExit("[ns_integrator] normalizer sidecar missing — refusing unnormalized scoring")
    vel_scale, vel_shift = ds.norm._scale_shift("velocity")
    p_scale, p_shift = ds.norm._scale_shift("pressure")
    assert abs(vel_shift) < 1e-12 and abs(p_shift) < 1e-12, "minmax normalizer must be zero-shift"

    n_total = len(ds)
    idxs = list(range(0, n_total, max(1, n_total // args.samples)))[: args.samples]
    print(f"[ns_integrator] {args.case}: nu={nu}  samples={len(idxs)}/{n_total}  "
          f"precision={args.precision} scheme={args.scheme}")

    per_step_all, ens_ratios, frmse_his, spec_l2s = [], [], [], []
    nrmses = []
    for si, i in enumerate(idxs):
        t0 = time.time()
        s = ds[i]
        x = s["input_dense"]                          # (1, 4, 256^3) normalized, CPU
        Y = s["output_dense"]                         # (20, 4, 256^3) normalized, CPU
        ts = s["global_timesteps"]                    # (21,) physical frame times
        u0_phys = (x[0, :3].to(dev, torch.float64 if args.precision == "fp64" else torch.float32)
                   * vel_scale)
        scfg = SolverConfig(N=u0_phys.shape[-1], nu=nu, dtype=args.precision,
                            device=args.device, scheme=args.scheme,
                            forcing=ForcingConfig(type="none"))
        u_hat0 = torch.fft.rfftn(u0_phys, dim=_FFT_DIMS)
        sol = PseudoSpectralSolver(scfg, u_hat0)

        per_step = []
        t_base = float(ts[0])
        for k in range(args.t_out):
            t_target = float(ts[k + 1]) - t_base
            while sol.t < t_target - 1e-10:
                sol.step(sol.suggest_dt(t_target=t_target))
            u_hat = sol.u_hat
            u_phys = torch.fft.irfftn(u_hat, s=u0_phys.shape[-3:], dim=_FFT_DIMS)
            p_phys = torch.fft.irfftn(pressure_hat(u_hat, sol.grid), s=u0_phys.shape[-3:],
                                      dim=_FFT_DIMS)
            pred_n = torch.cat([(u_phys / vel_scale).float(),
                                (p_phys / p_scale).float().unsqueeze(0)], dim=0)
            tg = Y[k].to(dev)
            e = (BM.nrmse_decay_safe(pred_n, tg) if args.decay else BM.nrmse(pred_n, tg))
            val = float(e) if e is not None else float("nan")
            per_step.append(val)
            if val == val:
                nrmses.append(val)
            if k == args.t_out - 1:                   # last-frame physics metrics, official style
                tg_u_phys = tg[:3].to(u_phys.dtype) * vel_scale
                ens_ratios.append(float(BM.enstrophy(u_phys.float().unsqueeze(0))
                                        / max(float(BM.enstrophy(tg_u_phys.float().unsqueeze(0))), 1e-30)))
                frmse_his.append(BM.frmse_bands(u_phys.float(), tg_u_phys.float())["high"])
                spec_l2s.append(BM.spectrum_l2(u_phys.float(), tg_u_phys.float()))
        per_step_all.append(per_step)
        print(f"  sample {si+1}/{len(idxs)} (ds idx {i}): s0={per_step[0]:.4f} "
              f"s{args.t_out-1}={per_step[-1]:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        del sol
        torch.cuda.empty_cache()

    import statistics
    def _m(v):
        v = [x for x in v if x == x]
        return statistics.mean(v) if v else None
    def _sd(v):
        v = [x for x in v if x == x]
        return statistics.pstdev(v) if len(v) > 1 else 0.0

    row = {
        "model": "ns_integrator", "task": "rollout", "case": args.case,
        "n_test_samples": len(idxs), "ckpt": None,
        "nRMSE_mean": _m(nrmses), "nRMSE_std": _sd(nrmses),
        "fRMSE_high_mean": _m(frmse_his),
        "enstrophy_ratio_mean": _m(ens_ratios),
        "spectrum_L2_mean": _m(spec_l2s),
        "rollout_nRMSE_by_step": [
            _m([ps[k] for ps in per_step_all]) for k in range(args.t_out)],
        "physical_space_metrics": True,
        "note": (f"production solver ({args.scheme}, {args.precision}), forcing=none; measures "
                 f"(hidden forcing state)+(chaos); nu={nu}; deterministic -> n={len(idxs)} "
                 f"samples suffice"),
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"ns_integrator__rollout__{args.case}.json"
    f.write_text(json.dumps(row, indent=2))
    print(f"[ns_integrator] wrote {f}")


if __name__ == "__main__":
    main()
