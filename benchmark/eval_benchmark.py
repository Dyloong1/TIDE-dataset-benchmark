"""eval_benchmark.py — evaluate a trained baseline (or a non-NN lower-bound) on a task's
test split, compute benchmark metrics, write a leaderboard JSON row.

  python eval_benchmark.py --model fno3d --task rollout --case ou_relam70_256_fp64 \
      --slice .../benchmark_slice.json --ckpt checkpoints/.../best.pt --out results/bench/

Non-NN lower bounds (no --ckpt): --model persistence (rollout) / spectral_interp (superres)
/ identity (recon/denoise). These anchor the leaderboard.

Metrics (benchmark_metrics.py, judge-tested): nRMSE, fRMSE(low/mid/high), cRMSE(energy/helicity — implemented in benchmark_metrics but NOT wired into this eval; do not report as output) /
helicity drift), rollout-nRMSE(τ), energy-spectrum L2, enstrophy(ζ) ratio. Reports
mean±std over the test set; decay tasks use the low-K-safe nRMSE.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Repo rule: import solver._env BEFORE torch. Windows ships two OpenMP runtimes (torch + MKL) and
# aborts with "OMP: Error #15" without KMP_DUPLICATE_LIB_OK, which _env sets at import time. This
# runs on CYB-t (Windows), so an import ordering that only works because a launch script happens to
# export the env var is not good enough. Must precede `import torch` below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solver._env  # noqa: E402,F401

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_metrics as BM  # noqa: E402


def _denorm_out(ds, pred, truth, task, out_ch, dev):
    """Map a task's OUTPUT tensors back to physical units for unit-carrying metrics.

    Metrics that return an absolute magnitude (fRMSE, the Poisson residual, the SGS Pi) must be
    scored in physical space: on normalized tensors they report physical/absmax, and absmax is
    per-config (velocity 2.50 on kf4 vs 4.35 on rotating_v2 = 1.74x), so the SAME physical error
    would print different numbers per config and the metric stops being comparable across the
    dataset. Scale-invariant metrics (nRMSE, Pearson correlation, ratios, sign fractions) are
    unaffected either way.

    Each task's output lives in a different normalizer group:
      rollout/superres/recon -> velocity (+ pressure when out_ch==4; the 5th scalar/b channel has
                                its own group in the zarr attrs and is left alone here)
      pressure               -> pressure
      sgs                    -> tau has NO group of its own: it is built from the already-normalized
                                frame, so it is returned unchanged and its caller (the Pi path)
                                does the velocity^2 rescale explicitly.
    Returns (pred, truth) unchanged when the dataset ran un-normalized (no sidecar).
    """
    norm = getattr(ds, "norm", None)
    if norm is None or task == "sgs":
        return pred, truth

    def _inv(x):
        x = x.float()
        if task == "pressure":
            return norm.inverse(x.cpu(), "pressure").to(dev)
        # velocity-output tasks: channels 0..2 are velocity, an optional channel 3 is pressure
        out = x.clone()
        out[:3] = norm.inverse(x[:3].cpu(), "velocity").to(dev)
        if x.shape[0] >= 4:
            out[3:4] = norm.inverse(x[3:4].cpu(), "pressure").to(dev)
        return out

    return _inv(pred), _inv(truth)
from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset  # noqa: E402
from Model.baselines import get_model, MODEL_REGISTRY  # noqa: E402
from train_benchmark import build_config  # noqa: E402  (shared config wiring: production_configs -> model)

NONNN = {"persistence", "spectral_interp", "identity", "smagorinsky", "poisson"}


def _strain_rate(u):
    """Symmetric strain-rate S_ij = ½(∂_i u_j + ∂_j u_i) via spectral derivatives.
    u (3,N,N,N) -> S (6,N,N,N) ordered [11,22,33,12,13,23]. Integer-mode wavenumbers.

    AXIS ORDER: index i in S_ij and in u[i] means the SAME direction, so Ks[i] must be ∂/∂x_i for
    the same i. The solver is authoritative (operators.curl_hat takes "channels x,y,z"; grids.py
    puts kx on axis -1, kz on axis -3), i.e. u[0]=u_x pairs with kx = axis -1.

    This listed Ks as [∂/∂z, ∂/∂y, ∂/∂x] while u is (u_x,u_y,u_z), so d(comp,axis) silently
    transposed every derivative -- the same bug already fixed in _grad_spectral, living in a second
    copy. Its fingerprint was the per-component correlation of -S̄ against the true deviatoric τ:
    +0.247 on component 22 (the axis that maps to itself under the transposition) but +0.003 and
    -0.011 on the others. A merely-imperfect eddy-viscosity assumption degrades all six components
    ALIKE; one good component and five dead ones means the components are misaligned.
    """
    N = u.shape[-1]
    dev = u.device
    kz = torch.fft.fftfreq(N, d=1.0 / N, device=dev)
    kx = torch.fft.rfftfreq(N, d=1.0 / N, device=dev)
    # Ks[i] = ∂/∂x_i : x -> axis -1 (rfft axis), y -> axis -2, z -> axis -3
    Ks = [(1j * kx)[None, None, :], (1j * kz)[None, :, None], (1j * kz)[:, None, None]]
    uh = torch.fft.rfftn(u.float(), dim=(-3, -2, -1))
    def d(comp, axis):  # ∂_axis u_comp
        return torch.fft.irfftn(Ks[axis] * uh[comp], s=(N, N, N), dim=(-3, -2, -1))
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    return torch.stack([0.5 * (d(j, i) + d(i, j)) for (i, j) in pairs], dim=0)


def _dynamic_smagorinsky_tau(u_filt, delta_frac=0.04, cs2_cap=0.09):
    """Dynamic-Smagorinsky SGS-stress prediction from the FILTERED velocity — the classic
    physics lower-bound for the LES-closure task (P5). Deviatoric part:
        tau_ij - (1/3)δ_ij tau_kk = -2 (Cs Δ)² |S̄| S̄_ij
    Cs² is estimated globally via a plane-averaged Germano-Lilly identity (single scalar,
    the robust variant), clamped to [0, cs2_cap] (Cs≈0.17 => Cs²≈0.029; cap 0.09 is generous).
    u_filt (3,N,N,N) -> tau (6,N,N,N) ordered [11,22,33,12,13,23], deviatoric (trace-free)."""
    from Dataset.turbgen_zarr_dataset import _gaussian_filter
    N = u_filt.shape[-1]
    # Same unit rule as _gaussian_filter: delta_frac is a fraction OF THE BOX, and the strain rate
    # below is built from integer wavenumbers (box = 2π), so Δ must be in those units too.
    # `delta_frac * N` (grid points) made (Cs·Δ)² ~1660x too large while the filter simultaneously
    # zeroed the field -- the two errors did not cancel, they compounded.
    delta = delta_frac * 2.0 * math.pi
    S = _strain_rate(u_filt)                                   # (6,N,N,N)
    # |S̄| = sqrt(2 S_ij S_ij); off-diagonals count twice
    Smag = torch.sqrt(2.0 * (S[:3].pow(2).sum(0) + 2.0 * S[3:].pow(2).sum(0)) + 1e-20)
    # test-filter (2Δ) fields for the Germano identity
    tf = 2.0 * delta_frac
    uf2 = _gaussian_filter(u_filt, tf)
    S2 = _strain_rate(uf2)
    S2mag = torch.sqrt(2.0 * (S2[:3].pow(2).sum(0) + 2.0 * S2[3:].pow(2).sum(0)) + 1e-20)
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    # Leonard stress L_ij = filter2(u_i u_j) - filter2(u_i) filter2(u_j)
    L = []
    for (i, j) in pairs:
        L.append(_gaussian_filter((u_filt[i] * u_filt[j]).unsqueeze(0), tf)[0] - uf2[i] * uf2[j])
    L = torch.stack(L, dim=0)
    # M_ij = 2Δ² ( filter2(|S̄| S̄_ij) - 4 |S̄_test| S̄test_ij )   (α=2 test-filter ratio -> 4)
    M = []
    for c in range(6):
        m1 = _gaussian_filter((Smag * S[c]).unsqueeze(0), tf)[0]
        M.append(2.0 * (delta ** 2) * (m1 - 4.0 * S2mag * S2[c]))
    M = torch.stack(M, dim=0)
    # global Cs² = <L_ij M_ij> / <M_ij M_ij> (off-diag ×2), plane-averaged (single scalar)
    wt = torch.tensor([1., 1., 1., 2., 2., 2.], device=u_filt.device)[:, None, None, None]
    num = (wt * L * M).sum()
    den = (wt * M * M).sum().clamp_min(1e-20)
    cs2 = (num / den).clamp(0.0, cs2_cap)
    nu_t = cs2 * (delta ** 2) * Smag                          # eddy viscosity field
    tau = torch.stack([-2.0 * nu_t * S[c] for c in range(6)], dim=0)
    # Make it EXACTLY deviatoric (trace-free): subtract 1/3 tr from the diagonal. The raw
    # -2 nu_t S has trace -2 nu_t tr(S) = -2 nu_t (div u), which is only ~0 (the filtered
    # field carries a small residual divergence), not machine-zero. The SGS model's physical
    # content is its deviatoric part, so remove the trace explicitly.
    tr = (tau[0] + tau[1] + tau[2]) / 3.0
    tau[0] = tau[0] - tr
    tau[1] = tau[1] - tr
    tau[2] = tau[2] - tr
    return tau


def _solve_pressure_spectral(u, omega_z: float = 0.0):
    """Exact kinematic pressure from velocity, by solving the pressure Poisson equation:

        lap(p) = -d_i d_j (u_i u_j) + 2 Omega . curl(u)

    Delegates to the SAME operator that generated the corpus pressure channel and that is certified
    by the D4 velocity-pressure consistency check -- so this bound is the corpus's own ground truth
    recomputed, not a reimplementation that could drift. WHICH operator that is depends on the case:
    `solver.operators.pressure_hat` for non-rotating runs, but `physics_ext.operators_ext.
    pressure_hat_rotating` for rotating ones -- `run_rotating_hit.py` writes the corpus with the
    latter, noting "plain pressure_hat would be the wrong non-rotating pressure".

    ★ omega_z (bug #7, found 2026-07-15 by the real-corpus metric range, P0): this used to ALWAYS
    call the non-rotating operator while advertising itself as the exact bound. Measured against the
    corpus's own stored pressure:

        ou_robust_kf4    (Omega=0)    nRMSE = 8.1e-08   <- bound holds; it IS the ground truth
        rotating_ro0p2   (Omega=2.5)  nRMSE = 9.997e-01 <- 99.97% wrong

    On rotating cases the "exact upper bound" anchoring the whole P4 task was solving a different
    equation than the data obeys. Two of q's six configs are rotating. Pass omega_z from
    benchmark_metrics.omega_z_for_case(case).

    u: (3,N,N,N) -> (1,N,N,N), both in PHYSICAL units.

    ★ Scale (this is why the signature takes physical u): p is QUADRATIC in u, but the Coriolis
    source is LINEAR in u. Under a normalized velocity u_n = u/s the two terms therefore rescale by
    DIFFERENT powers (s^2 vs s^1), so no single factor can convert solve(u_n) back -- the old
    `pred * vel_scale**2 / pres_scale` in main() is only valid at Omega=0. Rather than carry a
    two-term rescale, the caller now de-normalizes u first and we solve in physical units, where the
    equation is just true. Getting this wrong is the same class of bug as the SGS Pi units.
    """
    from solver.grids import SpectralGrid

    N = u.shape[-1]
    dev = "cuda" if u.is_cuda else "cpu"
    grid = SpectralGrid(N, device=dev, dtype="fp32")
    u_hat = torch.fft.rfftn(u.float(), dim=(-3, -2, -1)).to(grid.cdtype)
    if omega_z != 0.0:
        from physics_ext.operators_ext import pressure_hat_rotating
        omega_vec = torch.tensor([0.0, 0.0, float(omega_z)], dtype=grid.rdtype, device=dev)
        p_hat = pressure_hat_rotating(u_hat, omega_vec, grid)
    else:
        from solver.operators import pressure_hat
        p_hat = pressure_hat(u_hat, grid)
    p = torch.fft.irfftn(p_hat, s=(N, N, N), dim=(-3, -2, -1))
    return p.unsqueeze(0).to(u.dtype)


def _predict_nonnn(model_name, x, t_out, omega_z: float = 0.0, denorm_vel=None, norm_pres=None):
    """Non-NN lower-bound predictions.

    omega_z/denorm_vel/norm_pres are only used by the `poisson` P4 bound: it must solve in PHYSICAL
    units (the Coriolis source is linear in u while the nonlinear term is quadratic, so a single
    rescale of a normalized solve is wrong whenever Omega != 0 -- see _solve_pressure_spectral).
    denorm_vel: normalized velocity -> physical. norm_pres: physical pressure -> normalized.
    """
    if model_name == "persistence":          # repeat last input frame for all output steps
        last = x[:, -1:]
        return last.repeat(1, t_out, 1, 1, 1, 1)
    if model_name == "identity":             # recon/denoise floor: return the (masked/noisy) input
        return x
    if model_name == "spectral_interp":      # superres floor: spectral (Fourier) upsample handled
        return x                             # in the superres branch of main() where GT size is known
    if model_name == "smagorinsky":          # SGS closure floor: dynamic Smagorinsky from filtered u
        # x = (B,1,C,N,N,N) filtered field; predict tau_ij (B,1,6,N,N,N)
        B = x.shape[0]
        outs = []
        for b in range(B):
            uf = x[b, 0, :3]                  # velocity channels of the filtered input
            outs.append(_dynamic_smagorinsky_tau(uf))
        return torch.stack(outs, dim=0).unsqueeze(1)  # (B,1,6,N,N,N)
    if model_name == "poisson":              # P4 pressure bound: SOLVE the equation that defines p
        # x = (B,1,3,N,N,N) velocity; predict p (B,1,1,N,N,N) by solving
        #   lap(p) = -d_i d_j(u_i u_j) + 2 Omega . curl(u)
        # spectrally -- the same operator that GENERATED the corpus pressure channel (which one
        # that is depends on Omega; see _solve_pressure_spectral).
        #
        # This is not an approximation the way persistence/Smagorinsky are: it is the EXACT
        # solution, so it is an upper bound on achievable accuracy, and any learned model that
        # does not match it has simply failed to learn a solvable map. It also makes the P4 task
        # honest: without it a reader cannot tell whether a low pressure nRMSE means the model
        # learned the Poisson operator or merely fit the mean field.
        #
        # ★ That "EXACT" claim was false on rotating until 2026-07-15 (bug #7): the solve ignored
        # Omega and scored 9.997e-01 vs the corpus's own pressure. Solve in PHYSICAL units and the
        # two-power rescale problem disappears (Coriolis is linear in u, the nonlinear term is
        # quadratic -- no single factor converts a normalized solve when Omega != 0).
        if omega_z != 0.0 and (denorm_vel is None or norm_pres is None):
            raise SystemExit(
                "[eval] poisson bound on a ROTATING case needs the physical-unit converters "
                "(denorm_vel/norm_pres). Solving a normalized velocity cannot be rescaled by one "
                "factor when Omega != 0 (Coriolis is linear in u, the nonlinear term quadratic).")
        B = x.shape[0]
        outs = []
        for b in range(B):
            uv = x[b, 0, :3]
            if denorm_vel is not None:
                uv = denorm_vel(uv)
            pv = _solve_pressure_spectral(uv, omega_z=omega_z)
            if norm_pres is not None:
                pv = norm_pres(pv)
            outs.append(pv)
        return torch.stack(outs, dim=0).unsqueeze(1)  # (B,1,1,N,N,N)
    raise ValueError(model_name)


class _HaloTiledModel(torch.nn.Module):
    """Tiled forward where each tile is given `halo` voxels of REAL neighbour context.

    Why this exists: _TiledModel below feeds each 128^3 tile to the model with nothing outside it.
    A spectral model then FFTs that tile as if the TILE were the periodic box -- it is not -- so
    every tile's prediction is wrong near its own faces and the 8 tiles disagree where they meet.
    Measured on a NON-COLLAPSED fno3d (lr=1e-3, |f|=99% of the true delta): the mean jump across
    the tile boundary was 10.49x the jump at interior planes. The same measurement on the COLLAPSED
    production model (lr=1e-4, |f|=1% of the delta) reads 1.25x, i.e. "no seam" -- which is exactly
    how this went unseen: a model that predicts ~0 delta returns ~the input, and the input is
    continuous, so the seam cannot appear. The judge could only see the artifact in models that
    actually learned. CYB-q independently reproduced it on kf4 (8.18x vs 1.03x).

    The box IS periodic, so torch.roll supplies EXACT neighbour context (not padding, not a guess).
    Predict on the enlarged tile, then crop the halo away, so every returned voxel comes from a
    forward pass that had correct surroundings.

    This does not fully close the seam (measured: halo 8 -> 6.53x, 16 -> 4.34x, 32 -> 2.91x): the
    residual gap is structural, not a bug. Training happens on 128^3 patches while eval scores the
    native 256^3 field, and those two cannot both be honoured -- keep the training distribution
    (128^3 forward) and you inherit a seam; remove the seam (native 256^3 forward, measured 0.96x)
    and the forward is out of distribution (cos 0.738 -> 0.665, |f| overshoots to 108%). Opt-in via
    --eval-halo so the default protocol stays byte-identical for runs already scored.
    """

    def __init__(self, model, tile=128, halo=16):
        super().__init__()
        self.model = model
        self.tile = tile
        self.halo = halo

    def forward(self, x):
        N = x.shape[-1]
        if N <= self.tile:
            return self.model(x)
        g = N // self.tile
        h = self.halo
        out = None
        for pz in range(g):
            for py in range(g):
                for px in range(g):
                    z, yy, xx = pz * self.tile, py * self.tile, px * self.tile
                    # Roll the target block to sit at [h : h+tile) on every axis; periodicity of the
                    # box makes the surrounding h voxels the true neighbours of this tile.
                    xr = torch.roll(x, shifts=(-(z - h), -(yy - h), -(xx - h)), dims=(-3, -2, -1))
                    sub = xr[..., :self.tile + 2 * h, :self.tile + 2 * h, :self.tile + 2 * h]
                    o = self.model(sub)[..., h:h + self.tile, h:h + self.tile, h:h + self.tile]
                    if out is None:
                        out = torch.zeros(*o.shape[:3], N, N, N, dtype=o.dtype, device=o.device)
                    out[..., z:z + self.tile, yy:yy + self.tile, xx:xx + self.tile] = o
        return out


class _TiledModel(torch.nn.Module):
    """Run a model on 128^3 tiles and stitch the predictions back into the native 256^3 field.

    EVERY model evaluates through this wrapper, so the forward pass is one identical protocol for
    all baselines: predict on the same 128^3 sub-volume the model trained on, then reassemble. This
    replaces the old split where fno3d/tfno/ufno ran a native-256^3 forward while transolver and
    deeponet ran on a 128^3 patch -- two different resolutions scored side by side in one table.

    Metrics still see the full 256^3 field, which is required, not cosmetic: the energy spectrum,
    enstrophy and the Poisson residual are all computed with periodic FFTs, and a 128^3 sub-block of
    a 256^3 periodic box is NOT itself periodic, so scoring on a tile would corrupt exactly the
    physics this benchmark exists to measure.

    Tiling is the same deterministic non-overlapping 2x2x2 grid the dataset uses (grid_origin), so
    the 8 tiles reassemble the field exactly (verified: max|stitch - truth| = 0.0).

    !! SEAM WARNING (2026-07-15, CYB-3; independently reproduced by CYB-q on kf4) !!
    This docstring used to claim "seams were measured at 0.94x the interior jump, i.e. no seam --
    residual models predict a small delta on top of a continuous input field". That measurement was
    taken on a COLLAPSED model and the claim only holds while the model stays collapsed:
        collapsed fno3d (lr=1e-4, |f| = 1% of true delta):  seam 1.25x  (CYB-q on kf4: 1.03x)
        learned   fno3d (lr=1e-3, |f| = 99% of true delta): seam 10.49x (CYB-q on kf4: 8.18x)
    A model predicting ~0 delta returns ~its input, and the input is continuous -- so the seam is
    invisible precisely for the models that learned nothing. The polarity is inverted: this path
    passes broken models and fails good ones, because AR rollout re-injects the tile-edge
    discontinuity every step (measured: s19 1.87 and enstrophy_ratio 19.8 for the learned model,
    vs 1.09 / 0.97 for the collapsed one). Do not read a low seam ratio as "no seam" unless it was
    measured on a model whose |f| is comparable to the true delta.

    See _HaloTiledModel (--eval-halo) for the partial mitigation and why no clean fix exists.
    """

    def __init__(self, model, tile=128):
        super().__init__()
        self.model = model
        self.tile = tile

    def forward(self, x):
        # x: (B, t_in, C, N, N, N). Native-size input; tile only when it exceeds the tile size.
        N = x.shape[-1]
        if N <= self.tile:
            return self.model(x)
        g = N // self.tile
        out = None
        for pz in range(g):
            for py in range(g):
                for px in range(g):
                    z, yy, xx = pz * self.tile, py * self.tile, px * self.tile
                    sub = x[..., z:z + self.tile, yy:yy + self.tile, xx:xx + self.tile]
                    o = self.model(sub)                      # (B, t_out, C_out, tile, tile, tile)
                    if out is None:
                        out = torch.zeros(*o.shape[:3], N, N, N, dtype=o.dtype, device=o.device)
                    out[..., z:z + self.tile, yy:yy + self.tile, xx:xx + self.tile] = o
        return out


def _ar_rollout_streaming(model, x, y, k0, decay, residual=False, dev=None, truth_loader=None, t_out=None):
    """Autoregressive rollout of a single-step model (t_in frames -> 1 frame), unrolled to
    y.shape[1] steps, SCORING EACH FRAME AS IT IS PRODUCED so no full trajectory is ever held.

    Errors accumulate step-by-step (the physics the rollout task probes; a one-shot model never
    compounds its own error). Memory: at native 256^3 a single u,v,w,p frame is ~0.5GB, so keeping
    all 20 predicted frames + 20 truth frames (either on GPU -> CUDA OOM, or on CPU -> host-RAM OOM
    under the scope's MemoryMax) is the failure we hit. Instead we keep only the length-t_in sliding
    window on-device and, per step, compute the scalar nRMSE against that step's truth frame and
    then DROP both frames. Only the LAST predicted+truth frame is returned (physical metrics use it).

    x (B,t_in,C,...) on device; y (B,t_out,C,...) may stay on CPU — each truth frame y[:,t] is moved
    to `dev` only for the step that scores it, so the 30-frame truth trajectory (8GB at native 256^3)
    is never resident on the GPU/scope all at once. This is a memory move only; the scored values are
    bit-identical to scoring a fully-resident y. Returns (per_step_nrmse: list[float], last_pred:
    (C,...) on device, last_true: (C,...) on device).
    """
    window = x                                    # (B, t_in, C, ...) on device
    # t_out from y (eager) or explicit (lazy truth, where y is None and frames come from truth_loader)
    if t_out is None:
        t_out = y.shape[1]
    per_step = []
    last_pred = last_true = None
    for t in range(t_out):
        step = model(window)                      # (B, T_pred>=1, C, ...)
        nxt = step[:, :1]                         # single next frame (B,1,C,...)
        if residual:                              # residual: predicted delta + last input frame
            nxt = nxt + window[:, -1:]            # (train uses the same pred = x[:,-1] + f(x))
        if truth_loader is not None:
            # lazy: load ONLY this step's truth frame from the zarr (never the full 30-frame traj)
            yt = truth_loader(t).unsqueeze(0).unsqueeze(0).to(window.device)  # (1,1,C,...)
        else:
            yt = y[:, t:t + 1]                    # matching truth frame (B,1,C,...), possibly on CPU
            if dev is not None and yt.device.type != window.device.type:
                yt = yt.to(window.device)         # move ONLY this frame to device (not all 30)
        # score this frame now (scalar), then let the tensors go — nothing accumulates.
        e = (BM.nrmse_decay_safe(nxt[0, 0], yt[0, 0], k0=k0) if decay
             else BM.nrmse(nxt[0, 0], yt[0, 0]))
        per_step.append(float(e) if e is not None else float("nan"))
        last_pred, last_true = nxt[0, 0].detach(), yt[0, 0].detach()
        window = torch.cat([window[:, 1:], nxt], dim=1)  # slide window (only t_in frames on device)
    return per_step, last_pred, last_true


def _mean_std(vals):
    import statistics
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=("rollout", "superres", "recon", "pressure", "sgs"))
    ap.add_argument("--case", required=True)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="results/bench")
    ap.add_argument("--t-in", type=int, default=1)    # NS is Markovian: u(t) is a complete state
    ap.add_argument("--t-out", type=int, default=20)  # rollout AR eval = 20 steps = 1.0 T_L @ dt=0.05
    ap.add_argument("--per-frame-n", type=int, default=20,
                    help="Space tasks: equidistant frames per seed. Same as training (20) -- see "
                         "--frame-stride: sampling sparsely does not buy independence.")
    ap.add_argument("--frame-stride", type=int, default=8,
                    help="Rollout eval: frames between window starts. 8 = 0.4 T_L at dt=0.05. "
                         "Sampling more sparsely does NOT buy independence: measured N_eff is ~9 at "
                         "EVERY stride from 2 to 64, because the independent information in a test "
                         "set is set by (trajectory length x number of seeds), not by slicing "
                         "granularity. An earlier stride=64 threw away 186 of 195 samples and its "
                         "N_eff was in fact the LOWEST (8.1). 51 samples is enough to report "
                         "mean+/-std and to draw the rollout error curve (9 is not). The residual "
                         "correlation (rho~0.69) is reported honestly as N_eff, not hidden.")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--tfno-rank", type=int, default=8)
    ap.add_argument("--deeponet-p", type=int, default=128)
    ap.add_argument("--data-root", default=os.environ.get("TURBGEN_DATA_DIR", ""))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--native", action="store_true",
                    help="v2 protocol: single full-field 256^3 forward, no tiling/stitching. The "
                         "model is built at grid_n=256 and MUST have been trained at --patch 256; "
                         "scoring a 128-crop ckpt this way is out-of-distribution and invalid. "
                         "Mutually exclusive with --eval-halo (which only exists for the tiled path).")
    ap.add_argument("--eval-halo", type=int, default=0,
                    help="voxels of periodic neighbour context to give each 128^3 eval tile "
                         "(0 = the default hard-stitched tiling). The hard stitch leaves a seam that "
                         "is INVISIBLE on collapsed models (|f|~0 returns the continuous input) but "
                         "measures 10.49x the interior jump on a model that actually learned, and AR "
                         "rollout re-injects it every step. Halo mitigates but cannot remove it "
                         "(8 -> 6.53x, 16 -> 4.34x, 32 -> 2.91x); see _HaloTiledModel. Off by default "
                         "so already-scored runs stay comparable -- changing it changes the protocol.")
    ap.add_argument("--decay", action="store_true", help="use low-K-safe nRMSE (decay configs)")
    ap.add_argument("--residual", action="store_true",
                    help="rollout: model predicts a delta; add back the last frame each AR step "
                         "(pred = window[:,-1] + f(window)). MUST match how the ckpt was trained "
                         "(train_benchmark --residual) — else the rollout is meaningless.")
    ap.add_argument("--no-ar", action="store_true",
                    help="rollout: disable autoregressive unroll, do one-shot t_out prediction "
                         "instead (for a one-shot vs AR comparison; default is AR unroll).")
    ap.add_argument("--no-viz", action="store_true",
                    help="rollout NN eval: skip saving vis_panel.png (single-step 4-panel: input | "
                         "truth | pred | err). Default emits it for eye-check of predict-collapse.")
    args = ap.parse_args()
    if args.native and args.eval_halo > 0:
        raise SystemExit("[eval] --native and --eval-halo are mutually exclusive: halo is a "
                         "property of the tiled path, and native has no tiles.")
    dev = torch.device(args.device)

    # Resolution-bound models evaluate on a 128^3 patch, the SAME grid they trained on; the rest
    # evaluate at full native 256^3. eval_patch lets the test set crop.
    #   deeponet  — its trunk is built on a fixed grid, and native 256^3 also OOMs (~48GB).
    #   transolver — its positional embedding is a fixed-length learned table (512 entries) that the
    #     forward pass 1-D-interpolates to whatever patch count the RUNTIME shape gives:
    #     128^3/patch8 = 4096 patches (train, table stretched x8) vs 256^3 = 32768 (x64). Evaluating
    #     at 256^3 would therefore feed every location a positional code it never saw in training.
    #     (TOTAL_Z_LAYERS does NOT protect this: the models assign it and never read it; num_patches
    #     comes only from the input shape.) The FNO-family models carry no positional table and are
    #     genuinely resolution-invariant, so they keep native-256 eval.
    # EVERY model evaluates the same way: 128^3 tiled forward (_TiledModel) stitched back to the
    # native 256^3 field, which the metrics then score. So the dataset always serves full 256^3
    # samples and `eval_patch` stays off -- the old per-model split (native-256 for the FNO family,
    # 128^3 patch for transolver/deeponet, because transolver's 512-row pos_embed stretches x8 at
    # 128^3 but x64 at 256^3, and deeponet's trunk OOMs at 256^3) put two different resolutions in
    # one leaderboard. Tiling removes the asymmetry instead of documenting it.
    eval_patch = False
    # native-256 rollout AR eval loads the t_out (30) truth frames lazily (one per AR step) so the
    # 50-frame window (13.4GB) is never resident — the host-OOM fix. deeponet evals on a 128^3 patch
    # (small), so it keeps the eager path. Non-AR / space tasks (t_out=1) are tiny and stay eager.
    lazy_truth = (args.model is not None and args.task == "rollout"
                  and not args.no_ar and not eval_patch)
    ds = TurbgenZarrDataset(case=args.case, split="test", slice_manifest=args.slice, task=args.task,
                            t_in=args.t_in, t_out=args.t_out, per_frame_n=args.per_frame_n,
                            frame_stride=args.frame_stride,
                            data_root=args.data_root, max_samples=args.max_samples,
                            patch_size=(128 if eval_patch else None), eval_patch=eval_patch,
                            lazy_rollout_truth=lazy_truth)
    if len(ds) == 0:
        raise SystemExit(f"[eval] empty test set for {args.case}/{args.task}")
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    s0 = ds[0]
    in_ch = s0["input_dense"].shape[1]
    t_in = s0["input_dense"].shape[0]
    if s0.get("lazy_truth"):
        # lazy rollout sample: no stacked output_dense. rollout is same-channel (out==in) and the
        # truth length is the number of lazily-loaded frames.
        out_ch = in_ch
        t_out = int(s0["truth_frames"].shape[0])
    else:
        out_ch, t_out = s0["output_dense"].shape[1], s0["output_dense"].shape[0]

    model = None
    if args.model not in NONNN:
        inds = ['u', 'v', 'w', 'p'][:in_ch] if in_ch <= 4 else ['c%d' % i for i in range(in_ch)]
        # AR rollout: the model is SINGLE-STEP (trained t_out=1) and unrolled to t_out at eval,
        # so its output head must be built with OUTPUT_TIMESTEPS=1 to match the checkpoint. The
        # dataset's t_out (=30) is the unroll length / GT horizon, NOT the model's output width.
        # Non-rollout tasks (and --no-ar one-shot) build the head at the full t_out.
        model_t_out = 1 if (args.task == "rollout" and not args.no_ar) else t_out
        # eval grid must equal the grid the ckpt trained on: v2 native ckpts (--patch 256) build at
        # 256 so patch-transformers get the token count they trained with; v1 crop ckpts build at
        # 128 because _TiledModel feeds them 128^3 tiles.
        eval_grid = 256 if args.native else 128
        cfg = build_config(args, in_ch, out_ch, t_in, model_t_out, model=args.model, grid_n=eval_grid)
        model = get_model(args.model, cfg).to(dev).eval()
        if args.ckpt:
            sd = torch.load(args.ckpt, map_location=dev)
            model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
        if args.native:
            # v2 protocol: the model eats the full 256^3 field in one forward — no tiling, no
            # stitching, no seams. Only valid for ckpts TRAINED at native resolution (--patch 256);
            # a v1 crop-trained ckpt scored here is out-of-distribution and the number is invalid.
            model = model.to(dev).eval()
        else:
            # v1 tiled forward for EVERY model (no-op when the field is already <=128^3): removes
            # deeponet's native-256 OOM (~48GB) and transolver's pos_embed stretch, at the price of
            # the hard-stitch seam (measured 10.49x interior jump on a learned model; AR re-injects
            # it every step — the reason v2 exists).
            model = (_HaloTiledModel(model, tile=128, halo=args.eval_halo) if args.eval_halo > 0
                     else _TiledModel(model, tile=128)).to(dev).eval()

    nrmses, frmse_hi, ens_ratio, spec_l2, rollout_curves, freq_drift = [], [], [], [], [], []
    # finalized-metrics accumulators (per-task, filled below): fRMSE low/mid bands (all field tasks),
    # vorticity-PDF tails (velocity tasks), Poisson/∇p residual (P4), SGS τ-corr + Π/backscatter
    # (P5), correlation time (P1). Kept as lists -> mean at the end.
    frmse_lo, frmse_mid, vpdf_flat, vpdf_tail = [], [], [], []
    poisson_res, sgs_corr, sgs_pi, sgs_bsfrac, corr_times = [], [], [], [], []
    with torch.no_grad():
        for b in dl:
            x = b["input_dense"].to(dev)
            # lazy_truth (native-256 rollout AR): the sample carries truth frame INDICES, not the
            # stacked t_out trajectory (13.4GB). Build a per-step loader that pulls one truth frame
            # from the zarr — the AR loop is 1->1, so it only ever needs one truth frame at a time.
            lazy_truth = bool(b.get("lazy_truth", torch.tensor(False)).item()) if "lazy_truth" in b else False
            if lazy_truth:
                tf = b["truth_frames"][0].tolist()          # frame indices for this sample
                torigin = b["truth_origin"][0].tolist()
                truth_loader = (lambda t, _tf=tf, _o=torigin: ds.load_truth_frame(_tf[t], _o))
                y = None
                y_t_out = len(tf)
            else:
                truth_loader = None; y_t_out = None
                ar_eval = (model is not None and args.task == "rollout" and not args.no_ar)
                y = b["output_dense"] if ar_eval else b["output_dense"].to(dev)
            # K0 for the decay-safe metric: this trajectory's initial KE from the FIRST input
            # frame, in the same normalized field the metric operates on (threaded into every
            # nrmse_decay_safe call, else the low-K skip never fires — decay iron rule).
            k0 = float(0.5 * x[0, 0, :3].pow(2).mean()) if args.decay else None

            # rollout is evaluated AUTOREGRESSIVELY and STREAMED: the single-step model is unrolled
            # t_out steps, each frame scored (scalar nRMSE) and dropped as it's produced, so no full
            # 20-frame trajectory is ever held (native 256^3 x20 frames OOMs GPU *and* host RAM).
            # Only the last frame survives, for the physics metrics below. Other tasks / --no-ar
            # produce a full pred tensor and score it frame-by-frame the classic way.
            if model is not None and args.task == "rollout" and not args.no_ar:
                per_step, last_pred, last_true = _ar_rollout_streaming(
                    model, x, y, k0, args.decay, residual=args.residual, dev=dev,
                    truth_loader=truth_loader, t_out=y_t_out)
                if last_pred.shape[-4] != out_ch:
                    raise SystemExit(
                        f"[eval] INCOMPATIBLE model/task: '{args.model}' outputs {last_pred.shape[-4]} "
                        f"channels but task '{args.task}' needs {out_ch}. Refusing to score a "
                        f"broadcast/truncated result (pressure/sgs use only fno3d/tfno/deeponet).")
                nrmses.extend(v for v in per_step if v == v)      # drop nan (decay-skipped frames)
                rollout_curves.append(per_step)
                # last-frame fRMSE + physics use pv/yv = last_pred/last_true (velocity channels).
                pred_last, y_last = last_pred, last_true
            else:
                if model is not None:
                    pred = model(x)
                elif args.model == "spectral_interp" and args.task == "superres":
                    # superres input is ALREADY the trilinear-upsampled coarse field (see dataset);
                    # the interp lower-bound is just that input passed through.
                    pred = x
                else:
                    _om = BM.omega_z_for_case(args.case)
                    _dn = _np = None
                    if args.model == "poisson" and getattr(ds, "norm", None) is not None:
                        # Solve in PHYSICAL units, then map p back to the pressure normalizer's
                        # scale (that is what the target is on).
                        #
                        # This replaces a `pred * vel_scale**2 / pres_scale` rescale of a
                        # NORMALIZED solve. That shortcut is exact at Omega=0 (p is quadratic in u,
                        # so one factor suffices -- it verified to relative L2 1.2e-06 vs the corpus)
                        # but it is WRONG the moment rotation enters: the Coriolis source 2*Omega.
                        # curl(u) is LINEAR in u, so the two terms carry different powers of the
                        # velocity scale and no single factor converts them both (bug #7).
                        # (Both groups are zero-shift minmax; assert rather than silently mis-scale.)
                        _vs, _vsh = ds.norm._scale_shift("velocity")
                        _ps, _psh = ds.norm._scale_shift("pressure")
                        assert abs(_vsh) < 1e-12 and abs(_psh) < 1e-12, (
                            "poisson bound assumes zero-shift (minmax) normalizers; a shifted "
                            "normalizer breaks the physical-unit round trip.")
                        _dn = lambda uv, _s=_vs: uv * _s            # normalized vel -> physical
                        _np = lambda pv, _s=_ps: pv / _s            # physical p -> normalized
                    pred = _predict_nonnn(args.model, x, t_out,
                                          omega_z=_om, denorm_vel=_dn, norm_pres=_np)
                # channel-compat: an incompatible model (out==in hardcoded) on a channel-changing
                # task emits the wrong channel count -> refuse to score (don't silently truncate).
                if pred.shape[-4] != out_ch:
                    raise SystemExit(
                        f"[eval] INCOMPATIBLE model/task: '{args.model}' outputs {pred.shape[-4]} "
                        f"channels but task '{args.task}' needs {out_ch}. Refusing to score a "
                        f"broadcast/truncated result (pressure/sgs use only fno3d/tfno/deeponet).")
                # truth may be lazy (native-256 rollout: y is None, load per-frame via truth_loader)
                # or eager (y tensor in memory). Non-NN path must handle both — the AR NN path above
                # already does. Metrics are bit-identical either way; only truth SOURCING differs.
                n_t = t_out if y is None else y.shape[1]
                def _truth(t):
                    return (truth_loader(t).to(dev) if y is None else y[0, t])
                nonnn_curve = []          # per-step nRMSE, built here so lazy truth is loaded once
                for t in range(n_t):
                    yt = _truth(t)
                    step_nrmse = float(BM.nrmse(pred[0, t], yt))
                    nonnn_curve.append(step_nrmse)
                    e = (BM.nrmse_decay_safe(pred[0, t], yt, k0=k0) if args.decay
                         else step_nrmse)
                    if e is not None:
                        nrmses.append(float(e))
                pred_last, y_last = pred[0, -1], _truth(n_t - 1)
            # fRMSE high band (last output frame). frmse_bands returns an ABSOLUTE RMS of FFT
            # differences (no division by the truth), so it CARRIES UNITS: verified to scale
            # exactly 3.00x when both inputs are scaled 3x. Scored on normalized tensors it would
            # report physical_fRMSE/absmax, and absmax varies per config (2.50 kf4 vs 4.35
            # rotating_v2 = 1.74x), making a Tier-1 headline metric non-comparable ACROSS configs.
            # Same rule as the P4 Poisson and P5 Pi metrics: de-normalize first, by the group that
            # matches this task's output (velocity for rollout/superres/recon, pressure for P4;
            # sgs tau has no normalizer group of its own and is left as-is, see below).
            pred_phys, y_phys = _denorm_out(ds, pred_last, y_last, args.task, out_ch, dev)
            fb = BM.frmse_bands(pred_phys, y_phys)
            frmse_hi.append(fb["high"])
            # physics: enstrophy ratio + spectrum L2 + spectral drift (velocity, last frame).
            # ONLY for tasks whose output IS the velocity field (rollout/superres/recon).
            # pressure (out_ch=1) and sgs (out_ch=6 = tau_ij, NOT velocity) must be excluded —
            # applying the velocity normalizer / vorticity operator to tau is meaningless.
            # Computed in PHYSICAL space: invert the frozen normalizer so spectra/enstrophy
            # carry real units. If the dataset ran un-normalized (no sidecar), pred/y are
            # already physical.
            if args.task in ("rollout", "superres", "recon") and out_ch >= 3:
                pv, yv = pred_last[:3], y_last[:3]
                if getattr(ds, "norm", None) is not None:
                    # inverse_frame(un) with no pressure returns the velocity alone
                    pv = ds.norm.inverse_frame(pv.cpu()).to(dev)
                    yv = ds.norm.inverse_frame(yv.cpu()).to(dev)
                zt = BM.enstrophy(yv); zp = BM.enstrophy(pv)
                ens_ratio.append(float((zp / zt.clamp_min(1e-12))))
                spec_l2.append(BM.spectrum_l2(pv, yv))
                freq_drift.append(float(BM.frequency_drift(pv, yv, signed=True)))
                # Tier-1 fRMSE full bands (low/mid; high already above) + Tier-2 vorticity-PDF tails.
                # pv/yv are the PHYSICAL-space velocity, which is what fRMSE must see (it carries
                # units -- see the frmse_hi note above).
                fb_full = BM.frmse_bands(pv, yv)
                frmse_lo.append(fb_full["low"]); frmse_mid.append(fb_full["mid"])
                vt = BM.vorticity_pdf_tails(pv)
                vpdf_flat.append(vt["flatness"]); vpdf_tail.append(vt["tail_frac"])
            # P4 pressure: Poisson residual — does predicted p satisfy ∇²p = −∂_i u_j ∂_j u_i?
            # (the equation that DEFINES p). x is the velocity input (3ch); pred_last is p (1ch).
            # BOTH must be in PHYSICAL units: the residual is ‖∇²p − RHS‖/‖RHS‖ where RHS is a
            # quadratic in velocity, so a normalized p (different absmax than velocity, config-
            # dependent) would mismatch the physical-scale RHS and give a wrong, non-comparable
            # number. De-normalize velocity AND pressure with the frozen normalizer's per-group
            # inverse before scoring (velocity via inverse_frame's _VEL group, pressure via _PRES).
            if args.task == "pressure":
                uv = x[0, -1, :3]
                pv_ = pred_last[0] if pred_last.dim() == 4 else pred_last
                pv_ = pv_.float()
                if getattr(ds, "norm", None) is not None:
                    uv = ds.norm.inverse_frame(uv.cpu())[:3].to(dev)     # physical-space velocity
                    pv_ = ds.norm.inverse(pv_.cpu(), "pressure").to(dev)  # physical-space pressure
                # The physics of the CASE decides the equation. Rotating runs carry a +2Ω·curl(u)
                # Coriolis source (bug #7: the omega-blind call reported 2.75 on rotating_ro0p2 --
                # for the corpus's OWN exactly-correct pressure); Boussinesq runs carry a +∂_z b
                # buoyancy source (bug #8). Scoring either with the plain-HIT equation measures a
                # relation the data does not satisfy.
                if BM.is_buoyant_case(args.case):
                    # b is the 5th channel; it must be present or we are about to score buoyant
                    # data with the wrong equation -- refuse rather than emit a plausible number.
                    raise SystemExit(
                        f"[eval] {args.case} is a Boussinesq (buoyant) case: its pressure obeys "
                        f"lap(p) = -d_i d_j(u_i u_j) + d_z b, so scoring it needs the b field, "
                        f"which this P4 path does not currently load (the task's input is (u,v,w) "
                        f"only). Wire b through before scoring stratified, or exclude it from P4. "
                        f"See bug #8 in BM.pressure_poisson_residual. Refusing to report a number "
                        f"computed from an equation this data does not obey.")
                poisson_res.append(BM.pressure_poisson_residual(
                    uv, pv_, omega_z=BM.omega_z_for_case(args.case)))
            # P5 sgs: τ_ij correlation (LES a-priori headline) + SGS energy transfer Π/backscatter.
            # pred_last/y_last are the 6-comp stress; x is the filtered velocity input.
            if args.task == "sgs":
                tp = pred_last if pred_last.dim() == 4 else pred_last[0]
                ty = y_last if y_last.dim() == 4 else y_last[0]
                sgs_corr.append(BM.sgs_stress_correlation(tp, ty))
                # Pi = -tau_ij S_ij must be scored in PHYSICAL units, same rule as the P4 Poisson
                # residual above. Both operands are normalized here: the dataset builds tau from the
                # ALREADY-normalized frame, so tau ~ absmax^-2 while S (from velocity) ~ absmax^-1,
                # giving Pi ~ absmax^-3 -- a per-config factor that makes Pi_mean incomparable
                # across configs. De-normalize tau as a velocity SQUARED (it is filter(u_i u_j) -
                # filter(u_i)filter(u_j)) and the velocity linearly. sgs_stress_correlation
                # (Pearson) and backscatter_frac (sign-based) are scale-invariant, so they are
                # scored on the normalized tensors above and are unaffected.
                uv_sgs = x[0, -1, :3].float()
                tp_phys = tp.float()
                if getattr(ds, "norm", None) is not None:
                    vel_scale, vel_shift = ds.norm._scale_shift("velocity")
                    # Pure rescale only. 'minmax' (production) has shift=0, so u_phys = u_n*absmax
                    # and tau_phys = tau_n*absmax^2 exactly. Under 'standardize' the shift is
                    # non-zero and tau does NOT rescale by a single factor (the quadratic picks up
                    # cross terms), so refuse rather than report a silently wrong Pi.
                    assert abs(vel_shift) < 1e-12, (
                        "sgs Pi de-normalization assumes a zero-shift (minmax) velocity "
                        f"normalizer; got shift={vel_shift!r}. Pi would be wrong under a shifted "
                        "normalizer because tau is quadratic in velocity.")
                    uv_sgs = uv_sgs * vel_scale
                    tp_phys = tp_phys * (vel_scale ** 2)
                tr = BM.sgs_energy_transfer(tp_phys, uv_sgs)
                sgs_pi.append(tr["Pi_mean"]); sgs_bsfrac.append(tr["backscatter_frac"])
            # rollout_curves for the AR path is filled per-frame inside _ar_rollout_streaming;
            # the classic (--no-ar / non-NN) path appends its per-step curve here. Built above in
            # the scoring loop (nonnn_curve) so lazy truth (y is None) isn't re-stacked — identical
            # to the old BM.rollout_nrmse(pred, y) when y was eager.
            if args.task == "rollout" and (model is None or args.no_ar):
                rollout_curves.append(nonnn_curve)

    row = {
        "model": args.model, "task": args.task, "case": args.case,
        "n_test_samples": len(ds), "ckpt": args.ckpt,
        "nRMSE_mean": _mean_std(nrmses)[0], "nRMSE_std": _mean_std(nrmses)[1],
        "fRMSE_high_mean": _mean_std(frmse_hi)[0],
        "enstrophy_ratio_mean": _mean_std(ens_ratio)[0],       # physical space (denorm'd)
        "spectrum_L2_mean": _mean_std(spec_l2)[0],             # physical space (denorm'd)
        "freq_drift_mean": _mean_std(freq_drift)[0],           # spectral-centroid drift (signed)
        "physical_space_metrics": bool(getattr(ds, "norm", None) is not None),
    }
    # finalized metrics: only emit fields that were actually collected for this task (None-free rows).
    def _add(key, vals):
        if vals:
            row[key] = _mean_std(vals)[0]
    _add("fRMSE_low_mean", frmse_lo); _add("fRMSE_mid_mean", frmse_mid)      # Tier-1 full bands
    _add("vorticity_flatness_mean", vpdf_flat); _add("vorticity_tail_frac_mean", vpdf_tail)  # T2
    _add("poisson_residual_mean", poisson_res)                              # P4 moat
    _add("sgs_tau_corr_mean", sgs_corr); _add("sgs_Pi_mean", sgs_pi)        # P5 LES
    _add("sgs_backscatter_frac_mean", sgs_bsfrac)
    if rollout_curves:
        import statistics
        T = len(rollout_curves[0])
        # per-step mean over trajectories, FILTERING NaN (decay configs append NaN for low-K
        # skipped frames — c[t] can be NaN). Without this filter a single skipped trajectory
        # NaN-poisons the whole by_step curve and effective_prediction_time. If ALL trajectories
        # are NaN at step t (nothing valid), keep NaN there (honest: no data at that horizon).
        by_step = []
        for t in range(T):
            vals = [c[t] for c in rollout_curves if c[t] == c[t]]   # v==v drops NaN
            by_step.append(statistics.mean(vals) if vals else float("nan"))
        row["rollout_nRMSE_by_step"] = by_step
        # effective prediction time: roll-out step where mean nRMSE first exceeds 0.3
        # (interpolated). dt in frames; multiply by 0.05 to get T_L (dt=0.05 post redesign) if desired downstream.
        import torch as _t
        row["effective_prediction_time_frames"] = BM.effective_prediction_time(
            _t.tensor(by_step), threshold=0.3, dt=1.0)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    op = out_dir / f"{args.model}__{args.task}__{args.case}.json"
    op.write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))
    print(f"[eval] wrote {op}")

    # single-frame visualization: one PNG per (model, task, case) so you can eye-check whether
    # the model actually learned dynamics (numeric metrics can hide predict-mean/zero collapse
    # or high-frequency artifacts that a diverging colormap plot makes obvious). Only for NN
    # rollout with an actual ckpt — non-NN baselines (persistence/identity/etc) and non-rollout
    # tasks skip this. Failure is non-fatal (viz should never break the eval score).
    if model is not None and args.task == "rollout" and args.ckpt and not args.no_viz:
        try:
            _emit_viz_panel(model, ds, args, out_dir, dev)
        except Exception as e:
            print(f"[eval] viz skipped due to: {e}")


def _emit_viz_panel(model, ds, args, out_dir, dev):
    """Render a 4-panel PNG: input(last) | truth(t+1) | pred(t+1) | err. Model-independent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    sample = ds[0]
    x = sample["input_dense"].unsqueeze(0).to(dev)
    if sample.get("lazy_truth"):
        # lazy sample has no stacked output_dense; the viz only needs the first truth frame (t+1).
        y0 = ds.load_truth_frame(int(sample["truth_frames"][0]), sample["truth_origin"].tolist())
        y = y0.unsqueeze(0).unsqueeze(0).to(dev)   # (1,1,C,...)
    else:
        y = sample["output_dense"].unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model(x)
        nxt = out[:, :1]
        if args.residual:
            nxt = nxt + x[:, -1:]
    N = x.shape[-1]; z = N // 2
    inp = x[0, -1, 0, z].cpu().numpy()
    tru = y[0, 0, 0, z].cpu().numpy()
    prd = nxt[0, 0, 0, z].cpu().float().numpy()
    err = prd - tru
    v_ext = float(max(abs(tru).max(), abs(prd).max()))
    e_ext = float(abs(err).max())
    nrmse = float(np.sqrt((err ** 2).mean()) / max(np.sqrt((tru ** 2).mean()), 1e-12))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(inp, cmap='RdBu_r', vmin=-v_ext, vmax=v_ext); axes[0].set_title(f"input last  u @ z={z}")
    axes[1].imshow(tru, cmap='RdBu_r', vmin=-v_ext, vmax=v_ext); axes[1].set_title(f"TRUTH  u @ z={z}")
    axes[2].imshow(prd, cmap='RdBu_r', vmin=-v_ext, vmax=v_ext); axes[2].set_title(f"PRED   u @ z={z}")
    axes[3].imshow(err, cmap='RdBu_r', vmin=-e_ext, vmax=e_ext); axes[3].set_title(f"ERR (nRMSE={nrmse:.3f})")
    for ax in axes: ax.axis('off')
    fig.suptitle(f"{args.model} / {args.case} / rollout   single-step u velocity", fontsize=14)
    op = Path(out_dir) / "vis_panel.png"
    plt.tight_layout(); plt.savefig(op, dpi=100); plt.close()
    print(f"[eval] wrote {op}")


if __name__ == "__main__":
    main()
