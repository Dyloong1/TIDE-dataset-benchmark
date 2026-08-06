"""benchmark_metrics.py — evaluation metrics for the turbgen KDD benchmark.

Field tensors are (..., C, Z, H, W) or (C, Z, H, W); spatial dims are the last 3.
All metrics are pure torch (CPU/GPU). Physics side-metrics (vorticity, enstrophy, energy
spectrum) reuse the turbgen solver where available, else self-contained torch.fft.

Judge-first: tests/test_benchmark_metrics.py pins nRMSE/fRMSE/cRMSE on known-answer
synthetic fields (incl. identity=0, pure-shift, single-mode) before use on real data.

  nrmse(pred, true)                 -> ||pred-true|| / ||true||   (per-sample scalar)
  frmse_bands(pred, true, nbands=3) -> dict{low,mid,high}  RMS of |FFT diff| per |k| band
  crmse(field, kind)                -> conserved-quantity violation (energy / helicity)
  rollout_nrmse(pred_seq, true_seq) -> [nrmse per step]      (T,)
  energy_spectrum(u)                -> (k, E(k)) shell-averaged
  vorticity(u) / enstrophy(u)       -> ω=∇×u (spectral) / ζ=½⟨ω²⟩
"""
from __future__ import annotations

import torch


# ----------------------------------------------------------------------------- generic error
def nrmse(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalized RMSE = ||pred-true||2 / ||true||2 over all elements (per leading batch elem
    if batched on dim 0, else scalar)."""
    if pred.dim() >= 5:  # (B,...) -> per-sample
        flat = lambda x: x.reshape(x.shape[0], -1)
        num = (flat(pred) - flat(true)).norm(dim=1)
        den = flat(true).norm(dim=1).clamp_min(eps)
        return num / den
    num = (pred - true).norm()
    den = true.norm().clamp_min(eps)
    return num / den


def _band_masks(N: int, nbands: int, device):
    """|k| split into nbands equal-width bands over [0, k_nyq]. Returns list of bool masks on
    the rfftn grid (Z,H,W//2+1)."""
    kz = torch.fft.fftfreq(N, d=1.0 / N, device=device)
    kx = torch.fft.rfftfreq(N, d=1.0 / N, device=device)
    KZ = kz[:, None, None]
    KY = kz[None, :, None]
    KX = kx[None, None, :]
    kmag = torch.sqrt(KZ ** 2 + KY ** 2 + KX ** 2)
    kmax = float(kmag.max())
    edges = torch.linspace(0, kmax + 1e-6, nbands + 1, device=device)
    return [(kmag >= edges[b]) & (kmag < edges[b + 1]) for b in range(nbands)]


def frmse_bands(pred: torch.Tensor, true: torch.Tensor, nbands: int = 3) -> dict:
    """RMS of |FFT(pred)-FFT(true)| grouped into low/mid/high |k| bands (PDEBench fRMSE).
    Inputs (C,Z,H,W) or (B,C,Z,H,W); cubic grid assumed (Z=H=W=N)."""
    if pred.dim() == 4:
        pred, true = pred[None], true[None]
    B, C, Z, H, W = pred.shape
    N = Z
    pf = torch.fft.rfftn(pred.float(), dim=(-3, -2, -1))
    tf = torch.fft.rfftn(true.float(), dim=(-3, -2, -1))
    diff = (pf - tf).abs()  # (B,C,Z,H,W//2+1)
    masks = _band_masks(N, nbands, pred.device)
    names = ["low", "mid", "high"] if nbands == 3 else [f"band{i}" for i in range(nbands)]
    out = {}
    for name, m in zip(names, masks):
        sel = diff[..., m]                      # (B,C,#modes)
        out[name] = (sel.pow(2).mean(dim=-1).sqrt()).mean().item() if sel.numel() else 0.0
    return out


def crmse(field: torch.Tensor, kind: str = "energy") -> torch.Tensor:
    """Global conserved-quantity value (for comparing pred-vs-true conservation drift).
    energy  = ½ mean(|u|²)          (field = velocity (3,Z,H,W) or (B,3,...))
    helicity= mean(u · ω)           (needs velocity; ω spectral)
    Caller compares crmse(pred) vs crmse(true)."""
    if kind == "energy":
        return 0.5 * field.pow(2).sum(dim=-4).mean(dim=(-3, -2, -1))
    if kind == "helicity":
        w = vorticity(field)
        return (field * w).sum(dim=-4).mean(dim=(-3, -2, -1))
    raise ValueError(kind)


def rollout_nrmse(pred_seq: torch.Tensor, true_seq: torch.Tensor) -> torch.Tensor:
    """Per-step nRMSE over a rollout. seqs (T,C,Z,H,W) or (B,T,C,Z,H,W) -> (T,)."""
    if pred_seq.dim() == 5:
        pred_seq, true_seq = pred_seq[None], true_seq[None]
    B, T = pred_seq.shape[:2]
    out = []
    for t in range(T):
        out.append(nrmse(pred_seq[:, t], true_seq[:, t]).mean())
    return torch.stack(out)


# ----------------------------------------------------------------------------- physics
def vorticity(u: torch.Tensor) -> torch.Tensor:
    """ω = ∇×u via spectral derivatives. u (3,Z,H,W) or (B,3,Z,H,W) -> same shape.
    Reuses turbgen solver.operators.curl_hat when importable, else self-contained."""
    squeeze = u.dim() == 4
    if squeeze:
        u = u[None]
    B, _, Z, H, W = u.shape
    N = Z
    dev = u.device
    # integer wavenumbers for a [0, 2π) periodic box (d/dx = i·k, k = mode number).
    # fftfreq(N, d=1/N) already returns 0,1,..,N/2,..,-1 = integer modes — no extra 2π.
    # d/d(axis): the spatial axes are (Z,H,W)=(-3,-2,-1) with channels (u_x,u_y,u_z)=(0,1,2)
    # mapped to (axis-3, axis-2, axis-1). i·k operators, one per spatial axis:
    kz = torch.fft.fftfreq(N, d=1.0 / N, device=dev)   # full modes on the two fft axes
    kx = torch.fft.rfftfreq(N, d=1.0 / N, device=dev)  # half modes on the rfft axis (-1)
    KZ = (1j * kz)[:, None, None]     # ∂/∂z  (axis -3)
    KY = (1j * kz)[None, :, None]     # ∂/∂y  (axis -2)
    KX = (1j * kx)[None, None, :]     # ∂/∂x  (axis -1)
    uh = torch.fft.rfftn(u.float(), dim=(-3, -2, -1))
    ux, uy, uz = uh[:, 0], uh[:, 1], uh[:, 2]
    # ω = ∇×u :  ωx = ∂uz/∂y − ∂uy/∂z,  ωy = ∂ux/∂z − ∂uz/∂x,  ωz = ∂uy/∂x − ∂ux/∂y.
    # (verified bit-for-bit against solver.operators.curl_hat; the previous pairing was wrong —
    # it mismatched the k-operators to the wrong derivative axes, giving a non-physical ω that
    # flipped helicity's sign and value.)
    wx = torch.fft.irfftn(KY * uz - KZ * uy, s=(N, N, N), dim=(-3, -2, -1))
    wy = torch.fft.irfftn(KZ * ux - KX * uz, s=(N, N, N), dim=(-3, -2, -1))
    wz = torch.fft.irfftn(KX * uy - KY * ux, s=(N, N, N), dim=(-3, -2, -1))
    w = torch.stack([wx, wy, wz], dim=1).to(u.dtype)
    return w[0] if squeeze else w


def enstrophy(u: torch.Tensor) -> torch.Tensor:
    """ζ = ½⟨|ω|²⟩. u velocity -> scalar (or per-batch)."""
    w = vorticity(u)
    return 0.5 * w.pow(2).sum(dim=-4).mean(dim=(-3, -2, -1))


def energy_spectrum(u: torch.Tensor):
    """Shell-averaged 3D energy spectrum E(k). u (3,Z,H,W) -> (k_int[1..N//2], E)."""
    if u.dim() == 5:
        u = u[0]
    _, Z, H, W = u.shape
    N = Z
    dev = u.device
    uh = torch.fft.fftn(u.float(), dim=(-3, -2, -1)) / (N ** 3)
    e = 0.5 * uh.abs().pow(2).sum(dim=0)  # (Z,H,W) modal energy
    kz = torch.fft.fftfreq(N, d=1.0 / N, device=dev)
    KZ = kz[:, None, None]; KY = kz[None, :, None]; KX = kz[None, None, :]
    kmag = torch.sqrt(KZ ** 2 + KY ** 2 + KX ** 2).round().long()
    kmax = N // 2
    E = torch.zeros(kmax + 1, device=dev)
    # DISCARD modes outside the resolved sphere (|k| > N/2) rather than clamping them into the
    # last bin. The FFT grid is a CUBE, so its corners reach |k| = sqrt(3)*N/2 ~ 1.73*N/2: those
    # corner modes are ~44% of the grid and are NOT a physical shell (the sphere is the resolved
    # set). Clamping piled all of them into bin N/2, inflating it several-fold and letting that
    # one artifact bin dominate spectrum_l2 and pull the spectral centroid upward.
    keep = kmag <= kmax
    E.scatter_add_(0, kmag[keep].reshape(-1), e[keep].reshape(-1))
    ks = torch.arange(1, kmax + 1, device=dev)
    return ks, E[1:kmax + 1]


def spectrum_l2(pred_u: torch.Tensor, true_u: torch.Tensor) -> float:
    """Relative L2 error of the energy spectrum (log-space-friendly small floor)."""
    _, Ep = energy_spectrum(pred_u)
    _, Et = energy_spectrum(true_u)
    return ((Ep - Et).norm() / Et.norm().clamp_min(1e-12)).item()


# --------------------------------------------------------------- decay low-K safe nRMSE
def nrmse_decay_safe(pred: torch.Tensor, true: torch.Tensor, k_floor_frac: float = 0.05,
                     k0: float | None = None) -> torch.Tensor | None:
    """Decay-aware nRMSE: skip frames whose KE has fallen below k_floor_frac*K0 (small
    denominator blows up nRMSE — §2 decay-metric caveat). Returns None if below floor.
    pred/true velocity (3,Z,H,W)."""
    ke = 0.5 * true[:3].pow(2).mean()
    if k0 is not None and float(ke) < k_floor_frac * k0:
        return None
    return nrmse(pred, true)


# ------------------------------------------------------------- P1 rollout: spectral drift
def spectral_centroid(u: torch.Tensor) -> torch.Tensor:
    """Energy-weighted mean wavenumber k̄ = Σ k·E(k) / Σ E(k) of a velocity field.
    u (3,Z,H,W) or (B,3,Z,H,W) -> scalar (or (B,)). A model that over-smooths (loses
    high-k energy — the classic NO failure) has a k̄ that drifts DOWN vs the truth."""
    if u.dim() == 4:
        u = u[None]
    out = []
    for b in range(u.shape[0]):
        ks, E = energy_spectrum(u[b])
        denom = E.sum().clamp_min(1e-12)
        out.append((ks.float() * E).sum() / denom)
    c = torch.stack(out)
    return c[0] if c.numel() == 1 else c


def frequency_drift(pred_u: torch.Tensor, true_u: torch.Tensor, signed: bool = True) -> torch.Tensor:
    """Spectral-centroid drift = k̄(pred) − k̄(true), normalized by k̄(true).
    signed=True keeps the sign (negative = over-smoothed / high-k lost, the NO failure
    direction); signed=False returns |·|. pred/true velocity (3,Z,H,W) or batched.
    This is a P1 rollout first-class metric: it exposes spectral bias that RMSE hides."""
    cp = spectral_centroid(pred_u)
    ct = spectral_centroid(true_u)
    d = (cp - ct) / ct.clamp_min(1e-12)
    return d if signed else d.abs()


# ---------------------------------------------------------- P1 rollout: effective horizon
def effective_prediction_time(rollout_err: torch.Tensor, threshold: float = 0.3,
                              dt: float = 1.0) -> float:
    """Effective prediction horizon: the roll-out time τ at which the per-step error first
    crosses `threshold`, linearly interpolated between steps. rollout_err = (T,) sequence
    (e.g. from rollout_nrmse), monotone-ish increasing. dt = time between steps (in T_L or
    frames). Returns the last time if never crossed (full horizon survived), 0.0 if the
    first step already exceeds threshold. This is the physically-meaningful P1 headline:
    'how far can the model be trusted before error exceeds X'."""
    e = rollout_err.detach().float().flatten()
    T = e.numel()
    if T == 0:
        return 0.0
    if float(e[0]) >= threshold:
        return 0.0
    for t in range(1, T):
        if float(e[t]) >= threshold:
            # linear interp between step t-1 (below) and t (at/above)
            e0, e1 = float(e[t - 1]), float(e[t])
            frac = (threshold - e0) / max(e1 - e0, 1e-12)
            return (t - 1 + frac) * dt
    return (T - 1) * dt  # never crossed — survived the full rollout


# ============================================================================================
# Finalized-metrics additions (KDD D&B, per SOTA review): Tier-1 tau_ij corr; Tier-2 Poisson residual,
# SGS Π/backscatter, vorticity-PDF tails, correlation time. Each has a target test.
# ============================================================================================

def sgs_stress_correlation(tau_pred: torch.Tensor, tau_true: torch.Tensor) -> float:
    """Tier-1 (LES a-priori headline): Pearson correlation between predicted and true SGS
    stress tensors, pooled over all 6 components and all grid points. tau (6,N,N,N) or
    (B,6,N,N,N). Literature: >0.9 is a good a-priori SGS model. Returns a scalar in [-1,1]."""
    p = tau_pred.detach().float().flatten()
    t = tau_true.detach().float().flatten()
    p = p - p.mean(); t = t - t.mean()
    denom = (p.norm() * t.norm()).clamp_min(1e-12)
    return float((p * t).sum() / denom)


def _grad_spectral(f, N, dev):
    """Spectral gradient of a scalar field f (…,N,N,N) -> (…,3,N,N,N) ordered [d/dx, d/dy, d/dz].

    ORDER MATTERS AND IS NOT ARBITRARY. Callers index this as grads[i][j] = "derivative of u_i
    along direction j" and contract i with j, so the returned axis order must match the CHANNEL
    order of the velocity. The solver is authoritative (solver/operators.py::curl_hat takes
    "channels x,y,z"; solver/grids.py puts kx on axis -1 and kz on axis -3), i.e.

        u[0]=u_x pairs with kx = axis -1,   u[1]=u_y with ky = axis -2,   u[2]=u_z with kz = axis -3.

    This used to return [d/dz, d/dy, d/dx], which paired u_x with d/dz and silently transposed
    every gradient contraction. Measured proof on a corpus frame: div(u) = sum_i d_i u_i came out
    at 0.27 (of |grad u|^2) under the old order but 2.8e-12 under this one -- and the corpus is
    certified incompressible to ~1e-29 by the D1 gate, so the old order, not the data, was wrong.
    It made pressure_poisson_residual report 1.6 for a field that solves the Poisson equation
    EXACTLY (verified: the corpus's own D4-certified pressure scored the same 1.6).
    """
    kz = torch.fft.fftfreq(N, d=1.0 / N, device=dev)
    kx = torch.fft.rfftfreq(N, d=1.0 / N, device=dev)
    fh = torch.fft.rfftn(f.float(), dim=(-3, -2, -1))
    gz = torch.fft.irfftn((1j * kz)[:, None, None] * fh, s=(N, N, N), dim=(-3, -2, -1))  # axis -3
    gy = torch.fft.irfftn((1j * kz)[None, :, None] * fh, s=(N, N, N), dim=(-3, -2, -1))  # axis -2
    gx = torch.fft.irfftn((1j * kx)[None, None, :] * fh, s=(N, N, N), dim=(-3, -2, -1))  # axis -1
    return torch.stack([gx, gy, gz], dim=-4)


# Frame rotation rate per case, from the run yamls under experiments/phase_ext/configs/
# (`omega_z:`). Ω is NOT stored in the zarr attrs, so this table is the only machine-readable
# link between a case name and the equation its pressure actually obeys. Verified against the
# real corpus: each case's stored p is reproduced to ~3e-08 by its own Ω and to O(1) by any
# other (see pressure_poisson_residual's docstring).
#
# Anything not listed is non-rotating (Ω=0) -- true for plain HIT/decay/scalar. If you add a
# rotating case, add it here; `omega_z_for_case` cannot detect a missing entry, it will just
# return 0.0 and silently score you with the wrong equation. That is exactly bug #7.
_OMEGA_Z_BY_CASE = {
    "rotating_ro0p2_256_fp64": 2.5,       # measured Ro=0.065 (strong rotation / quasi-2D)
    "rotating_ro0p2_v2_256_fp64": 0.81,   # targets Ro~0.2 (moderate, mixed 2D/3D)
}


def omega_z_for_case(case: str) -> float:
    """Frame rotation rate for `case` (0.0 if it is not a rotating run).

    Callers of pressure_poisson_residual on corpus data MUST route Ω through this rather than
    defaulting to 0 -- see bug #7 in that function's docstring.
    """
    return _OMEGA_Z_BY_CASE.get(case, 0.0)


# Cases whose momentum equation carries a BUOYANCY term (Boussinesq): du/dt = ... + b e_z.
# Their pressure obeys  lap(p) = -d_i d_j(u_i u_j) + d_z b  -- see physics_ext.operators_ext.
# pressure_hat_boussinesq, which is what run_stratified_hit.py writes the corpus with.
#
# Unlike rotation, this cannot be captured by a scalar: the source is the FIELD d_z b, so the
# caller must pass the b channel itself (pressure_poisson_residual(..., b=...)). That is why there
# is no `b_for_case()` -- there is nothing to look up, only data to load.
_BUOYANT_CASES = {"stratified_reb40_256_fp64"}


def is_buoyant_case(case: str) -> bool:
    """True if `case` is a Boussinesq run whose pressure needs the +d_z b source (bug #8).

    Scoring one of these WITHOUT passing b applies the plain-HIT equation to buoyant data -- the
    same error as bug #7, different physics. CYB-t measured it on the real corpus: re-solving p
    from (u, b) with pressure_hat_boussinesq reproduces the stored p to rel_diff 8e-8, so the
    corpus is right and the metric was missing the term.
    """
    return case in _BUOYANT_CASES


def pressure_poisson_residual(u: torch.Tensor, p: torch.Tensor, omega_z: float = 0.0,
                              b: torch.Tensor = None) -> float:
    """Tier-2 (P4 moat): relative residual of the pressure Poisson equation
        ∇²p = −∂_i u_j ∂_j u_i  +  2 Ω·curl(u)  +  ∂_z b   (incompressible NS, unit density)
    i.e. how well a predicted p satisfies the equation that DEFINES it (not just fits values).
    u (3,N,N,N), p (N,N,N) [or with leading batch]. Returns ‖∇²p − RHS‖ / ‖RHS‖.

    ``omega_z``: frame rotation rate about z. **Must be passed for rotating cases** -- it
    defaults to 0 (non-rotating), which is correct ONLY for plain HIT.
    ``b``: the buoyancy field. **Must be passed for Boussinesq (stratified) cases** -- it defaults
    to None (no buoyancy), which is correct ONLY for non-buoyant flows. See ``is_buoyant_case``.

    ★ bug #8 (found 2026-07-15 by CYB-t, the same day as #7 and the same shape): stratified's
    corpus pressure is written by ``physics_ext.operators_ext.pressure_hat_boussinesq``, whose
    momentum equation carries +b e_z, so its pressure obeys ∇²p = −∂_i∂_j(u_i u_j) + ∂_z b. This
    metric knew only the plain-HIT equation, so it scored buoyant data with an equation that data
    does not obey. Measured on the real corpus: re-solving p from (u, b) with the Boussinesq
    operator reproduces the stored p to **rel_diff 8e-8** -- the corpus is right, the metric was
    missing the term. Rotation needed a scalar; buoyancy needs the FIELD, so it is a tensor arg.

    ★ Why omega_z exists (bug #7, found 2026-07-15 by the real-corpus metric range, P0):
    this function used to hardcode the NON-rotating equation and was applied to every case.
    On the rotating corpus that is simply the wrong equation -- the rotating-frame momentum
    equation carries -2Ω×u, whose divergence adds a +2Ω·curl(u) source (div(Ω×u) = -Ω·ω).
    The DNS side always knew this: the corpus is generated by
    ``physics_ext.operators_ext.pressure_hat_rotating``, whose own comment reads "plain
    pressure_hat would be the wrong non-rotating pressure". Only the benchmark side was wrong.

    Measured on the real corpus (frame 0), re-solving p from u and comparing to the stored p:

        case                      Ω=2.5      Ω=0.81     Ω=0 (what this fn used to assume)
        rotating_ro0p2         2.6e-08 ★    6.8e-01     1.0e+00
        rotating_ro0p2_v2      1.5e+00      3.2e-08 ★   7.1e-01

    Each case matches ONLY its own Ω, to machine precision; every wrong Ω gives O(1). With
    the old omega-blind code this metric reported a residual of **2.75** on rotating_ro0p2 --
    i.e. it declared the corpus's own exactly-correct pressure to be garbage, and would have
    scored every P4 model (and the `poisson` reference baseline) on rotating with an equation
    those fields do not obey. Two of q's six training configs are rotating.

    ★ Ω is NOT recorded in the zarr attrs -- it lives only in the run yaml
    (`experiments/phase_ext/configs/rotating_*.yaml: omega_z`). So the caller must supply it;
    this function cannot recover it from the data. See ``omega_z_for_case``.
    """
    if u.dim() == 5:
        u, p = u[0], p[0]
    if p.dim() == 4:
        p = p[0]
    N = u.shape[-1]; dev = u.device
    kz = torch.fft.fftfreq(N, d=1.0 / N, device=dev)
    kx = torch.fft.rfftfreq(N, d=1.0 / N, device=dev)
    KZ = kz[:, None, None]; KY = kz[None, :, None]; KX = kx[None, None, :]
    k2 = KZ ** 2 + KY ** 2 + KX ** 2
    ph = torch.fft.rfftn(p.float(), dim=(-3, -2, -1))
    lap_p = torch.fft.irfftn(-k2 * ph, s=(N, N, N), dim=(-3, -2, -1))       # ∇²p
    # RHS = −∂_i u_j ∂_j u_i = −Σ_ij (∂_j u_i)(∂_i u_j)
    grads = torch.stack([_grad_spectral(u[i], N, dev) for i in range(3)], dim=0)  # (i,3=j,N,N,N)
    rhs = torch.zeros(N, N, N, device=dev)
    for i in range(3):
        for j in range(3):
            rhs = rhs - grads[i, j] * grads[j, i]
    if omega_z != 0.0:
        # Coriolis source: div(-2Ω×u) = +2Ω·ω. With Ω = Ω_z ẑ this is 2 Ω_z ω_z, and ω_z is
        # the z-component of curl(u) -- taken from the SAME _grad_spectral used above, so an
        # axis-order regression there moves both terms together and cannot silently cancel.
        # grads[i, j] = ∂_j u_i  =>  ω_z = ∂_x u_y − ∂_y u_x = grads[1, 0] − grads[0, 1].
        rhs = rhs + 2.0 * omega_z * (grads[1, 0] - grads[0, 1])
    if b is not None:
        # Buoyancy source (bug #8): div(b e_z) = ∂_z b. Same _grad_spectral as above, so the axis
        # convention cannot drift between the two terms. grads-style indexing: [2] is ∂_z.
        if b.dim() == 4:
            b = b[0]
        rhs = rhs + _grad_spectral(b, N, dev)[2]
    # ∇²p has ZERO mean (k=0 mode annihilated by the Laplacian), so the k=0 component of RHS is
    # unsolvable and pressure is only defined up to a constant. Compare residual on the mean-free
    # (DC-removed) fields — the physically meaningful Poisson balance, matching pressure_hat which
    # sets the k=0 mode to 0. (A truly incompressible RHS is mean-zero; finite fields carry a small
    # DC that must not inflate the residual.)
    rhs = rhs - rhs.mean()
    lap_p = lap_p - lap_p.mean()
    return float((lap_p - rhs).norm() / rhs.norm().clamp_min(1e-12))


def _gaussian_filter_metric(field, sigma_frac=0.04):
    """Spectral Gaussian filter (matches dataset _gaussian_filter). field (…,N,N,N)."""
    N = field.shape[-1]; dev = field.device
    kz = torch.fft.fftfreq(N, d=1.0 / N, device=dev)
    kx = torch.fft.rfftfreq(N, d=1.0 / N, device=dev)
    k2 = (kz[:, None, None] ** 2 + kz[None, :, None] ** 2 + kx[None, None, :] ** 2)
    G = torch.exp(-k2 * (sigma_frac * N) ** 2 / 24.0)
    fh = torch.fft.rfftn(field.float(), dim=(-3, -2, -1)) * G
    return torch.fft.irfftn(fh, s=(N, N, N), dim=(-3, -2, -1)).to(field.dtype)


def sgs_energy_transfer(tau: torch.Tensor, u_filt: torch.Tensor, sigma_frac: float = 0.04) -> dict:
    """Tier-2 (P5): SGS energy transfer Π = −τ_ij S̄_ij and its backscatter fraction.
    Π>0 forward (dissipative), Π<0 backscatter (energy to resolved scales). A pure eddy-viscosity
    model (Smagorinsky) can't produce backscatter, so reporting the correct backscatter fraction
    is a neural-SGS selling point. tau (6,N,N,N) [11,22,33,12,13,23]; u_filt (3,N,N,N) filtered
    velocity. Returns {Pi_mean, backscatter_frac, backscatter_energy_frac}."""
    if tau.dim() == 5:
        tau, u_filt = tau[0], u_filt[0]
    N = u_filt.shape[-1]; dev = u_filt.device
    g = torch.stack([_grad_spectral(u_filt[i], N, dev) for i in range(3)], dim=0)  # (i,3=j,N,N,N)
    S = 0.5 * (g + g.transpose(0, 1))            # S̄_ij = ½(∂_j ū_i + ∂_i ū_j), (3,3,N,N,N)
    # τ_ij S̄_ij with the 6 stored comps (off-diagonals counted twice)
    idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    Pi = torch.zeros(N, N, N, device=dev)
    for c, (i, j) in enumerate(idx):
        w = 1.0 if i == j else 2.0
        Pi = Pi - w * tau[c] * S[i, j]
    bmask = Pi < 0
    return {
        "Pi_mean": float(Pi.mean()),
        "backscatter_frac": float(bmask.float().mean()),          # fraction of points with Π<0
        "backscatter_energy_frac": float((-Pi[bmask]).sum() / Pi.abs().sum().clamp_min(1e-12)),
    }


def vorticity_pdf_tails(u: torch.Tensor, sigma_thresh: float = 3.0) -> dict:
    """Tier-2: intermittency via vorticity-magnitude PDF tails. Over-smoothed models lose the
    heavy tail. u (3,N,N,N). Returns {flatness (kurtosis of ω-components), tail_frac (|ω|>Nσ)}.
    Flatness > 3 (Gaussian) signals intermittency; a smoothed prediction collapses it toward 3."""
    w = vorticity(u).flatten()
    wc = w - w.mean()
    var = wc.pow(2).mean().clamp_min(1e-12)
    flatness = float((wc.pow(4).mean() / var.pow(2)))     # kurtosis
    wmag = vorticity(u).pow(2).sum(dim=-4).sqrt().flatten()
    thr = wmag.mean() + sigma_thresh * wmag.std().clamp_min(1e-12)
    return {"flatness": flatness, "tail_frac": float((wmag > thr).float().mean())}


def correlation_time(pred_seq: torch.Tensor, true_seq: torch.Tensor,
                     thresh: float = 0.5, dt: float = 1.0) -> float:
    """Tier-2 (P1): time at which the spatial correlation between rollout and truth first drops
    below `thresh` (default 0.5), in units of dt (frames or T_E). Chaotic systems always diverge
    pointwise, so 'when does it decorrelate' is more meaningful than terminal RMSE. seqs
    (T,C,N,N,N) or (B,T,C,...). Returns the last time if it never decorrelates."""
    if pred_seq.dim() == 5:
        pred_seq, true_seq = pred_seq[None], true_seq[None]
    T = pred_seq.shape[1]
    corrs = []
    for t in range(T):
        p = pred_seq[:, t].flatten().float(); q = true_seq[:, t].flatten().float()
        p = p - p.mean(); q = q - q.mean()
        corrs.append(float((p * q).sum() / (p.norm() * q.norm()).clamp_min(1e-12)))
    for t in range(T):
        if corrs[t] < thresh:
            if t == 0:
                return 0.0
            c0, c1 = corrs[t - 1], corrs[t]
            frac = (c0 - thresh) / max(c0 - c1, 1e-12)
            return (t - 1 + frac) * dt
    return (T - 1) * dt
