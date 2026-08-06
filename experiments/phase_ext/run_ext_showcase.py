"""Extended-physics GIF slice export. Warm-starts from a case's final checkpoint,
builds the CORRECT ext-physics solver (rotation / scalar / stratification — NOT a
plain PseudoSpectralSolver, which would render the wrong dynamics), integrates a
short window stepping the right physics, and writes slices.npz (|omega| + u_x mid-
plane, + a buoyancy slice for stratified). Feeds the unchanged
phase0/make_corpus_gif_unified.py (point --out at phase0/results/trajectory_showcase_<case>).

GIF-only: no D-group residual sampling (that's eval_dynamics_ext.py's job).

    python run_ext_showcase.py --case rotating_ro0p2_256_fp64 \
        --out ../phase0/results/trajectory_showcase_rotating_ro0p2_256_fp64 --n-TL 10
"""
from __future__ import annotations

import argparse
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
from solver.operators import curl_hat  # noqa: E402
from solver.memguard import check_ram  # noqa: E402
from physics_ext.config_ext import load_ext_config  # noqa: E402

_FFT = (-3, -2, -1)
RES = _HERE.parents[0] / "results"


def _resolve_rec(case):
    """Find the results dir that actually holds this case's config + ckpt.

    The KDD corpus pipeline writes per-seed dirs "<case>_corpus" (seed 0) and
    "<case>_corpus_pool<K>" (seed K), NOT a bare "<case>" dir — and after the zarr write the
    per-seed FRAME .pt are deleted but the config_snapshot.yaml + ckpt_t*.pt survive. So prefer,
    in order: bare <case> (legacy single-run layout) -> <case>_corpus (seed 0) -> first
    <case>_corpus_pool* that still has a config+ckpt. (Fixes ext-gif skip: showcase used to look
    only at RES/<case> which the corpus pipeline never creates, so every ext gif silently skipped.)"""
    cand = [RES / case, RES / f"{case}_corpus"] + sorted(RES.glob(f"{case}_corpus_pool*"))
    for d in cand:
        if (d / "config_snapshot.yaml").exists() and sorted(d.glob("ckpt_t*.pt")):
            return d
    # nothing with both config+ckpt; return the base for a clear downstream error
    return RES / case


def _build(case):
    """Rebuild the correct ext-physics solver from a case's config + last ckpt.
    Mirrors eval_dynamics_ext.py's dispatch. Returns (solver, kind, step_fn, T_L_guess)."""
    rec = _resolve_rec(case)
    cfg, rot, scal, strat = load_ext_config(rec / "config_snapshot.yaml")
    cks = sorted(rec.glob("ckpt_t*.pt"))
    if not cks:
        raise SystemExit(f"no checkpoints in {rec} (showcase needs a final ckpt)")
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
    grid = SpectralGrid(cfg.solver.N, cfg.solver.device, cfg.solver.dtype)
    u_hat = ck["u_hat"].to(grid.device, grid.cdtype)

    if rot.enabled:
        from physics_ext.rotating import RotatingSolver
        s = RotatingSolver(cfg.solver, u_hat, rot)
        s.t = float(ck["t"])
        return s, "rotating", (lambda dt: s.step(dt)), None
    if strat.enabled:
        from physics_ext.boussinesq import BoussinesqSolver
        b_hat = ck.get("b_hat", torch.zeros((1, grid.N, grid.N, grid.Nh),
                                            dtype=grid.cdtype, device=grid.device))
        s = BoussinesqSolver(cfg.solver, strat, u_hat, b_hat.to(grid.device, grid.cdtype))
        s.t = float(ck["t"]); s.base.t = s.t
        return s, "stratified", (lambda dt: s.step(dt)), None
    if scal.enabled:
        from solver.solver import PseudoSpectralSolver
        from physics_ext.scalar import ScalarTransport
        s = PseudoSpectralSolver(cfg.solver, u_hat)
        s.t = float(ck["t"])
        th = ck.get("theta_hat", torch.zeros((1, grid.N, grid.N, grid.Nh),
                                             dtype=grid.cdtype, device=grid.device))
        scal_t = ScalarTransport(grid, scal, th.to(grid.device, grid.cdtype))
        N = grid.N
        _uph = torch.empty((3, N, N, N), dtype=grid.rdtype, device=grid.device)

        def step_fn(dt):   # Strang split (same as run_scalar_hit)
            torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=_FFT, out=_uph)
            scal_t.step(_uph, dt * 0.5)
            s.step(dt)
            torch.fft.irfftn(s.u_hat, s=(N, N, N), dim=_FFT, out=_uph)
            scal_t.step(_uph, dt * 0.5)
        s._scalar = scal_t   # stash for slice export
        return s, "scalar", step_fn, None
    raise SystemExit(f"{case}: no enabled physics block")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True, help="dir for slices.npz (point at phase0 trajectory_showcase_<case>)")
    ap.add_argument("--n-TL", type=float, default=10.0)
    ap.add_argument("--T-L", type=float, default=0.0, help="0 -> read from the case metrics.json")
    ap.add_argument("--slice-every", type=float, default=0.0, help="0 -> 0.2*T_L")
    a = ap.parse_args()

    import json
    T_L = a.T_L
    if T_L <= 0:
        m = json.load(open(_resolve_rec(a.case) / "metrics.json", encoding="utf-8"))
        T_L = float(m.get("T_L", 1.7))
    slice_every = a.slice_every if a.slice_every > 0 else 0.2 * T_L

    s, kind, step_fn, _ = _build(a.case)
    grid = s.grid if hasattr(s, "grid") else s.base.grid
    N = grid.N
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ext-showcase] {a.case} ({kind}) warm-start t={s.t:.1f} T_L={T_L:.2f} "
          f"slice_every={slice_every:.3f}", flush=True)

    t_end = s.t + a.n_TL * T_L
    # work_hat scratch for curl: RotatingSolver inherits .work_hat; BoussinesqSolver
    # has it on .base; PseudoSpectralSolver (scalar) has .work_hat. (avoid `or` on
    # tensors — ambiguous truth value.)
    work = s.work_hat if hasattr(s, "work_hat") else s.base.work_hat
    slices_om, slices_ux, slices_b, slice_t = [], [], [], []
    next_slice = s.t
    while s.t < t_end - 1e-9:
        if s.t >= next_slice - 1e-9:
            om = torch.fft.irfftn(curl_hat(s.u_hat, grid, out=work), s=(N, N, N), dim=_FFT)
            slices_om.append((om[:, N // 2] ** 2).sum(0).sqrt().float().cpu().numpy())
            ux = torch.fft.irfftn(s.u_hat[0], s=(N, N, N), dim=_FFT)
            slices_ux.append(ux[N // 2].float().cpu().numpy())
            if kind == "stratified":
                b = torch.fft.irfftn(s.b_hat[0], s=(N, N, N), dim=_FFT)
                slices_b.append(b[N // 2].float().cpu().numpy()); del b
            del om, ux
            slice_t.append(s.t)
            next_slice += slice_every
            check_ram("ext-showcase", abort_frac=0.92)
        step_fn(s.suggest_dt(t_target=t_end))

    npz = {"t": np.asarray(slice_t), "omega": np.stack(slices_om),
           "ux": np.stack(slices_ux), "T_L": T_L, "t0": slice_t[0] if slice_t else 0.0}
    if slices_b:
        npz["b"] = np.stack(slices_b)
    np.savez_compressed(out_dir / "slices.npz", **npz)
    print(f"[ext-showcase] {len(slice_t)} slices -> {out_dir / 'slices.npz'}", flush=True)


if __name__ == "__main__":
    main()
