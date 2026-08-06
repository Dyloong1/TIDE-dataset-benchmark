"""Objective A10 isotropy verdict computed on the RELEASED frame-set (the zarr itself).

WHY THIS EXISTS (2026-06-25): a10_corpus_verdict.py reads metrics.json's ui2_mean/uij_mean,
which are time-averages over the acceptance SAMPLING WINDOW (~spectrum_every_steps cadence),
NOT over the exported frames. For "the released data is objectively compliant on its OWN"
(a reviewer who downloads the zarr and recomputes A10 must get the same verdict), the A10
judged must be computed over EXACTLY the frames we ship. This script does that: it reads the
zarr's `u` channel frame-by-frame, computes per-frame <u_i^2> and <u_i u_j>, and pools them
across all frames of all seeds with the SAME formula as a10_corpus_verdict.py.

Pooling (matches a10_corpus_verdict.py, signed-correct):
  per-frame:  ui2[c] = mean(u[c]^2),  uij[p] = mean(u[i]*u[j])   (p over (0,1),(0,2),(1,2))
  pooled:     <ui2> = mean over all frames,  <uij> = mean over all frames (SIGNED -> the
              cross terms of independent realizations cancel; this is why pooling lowers A10)
  cross% = 100 * max|<uij>| / (2 K2 / 3),   comp% = 100 * max|3 <ui2>/(2 K2) - 1|,  K2=0.5*sum<ui2>

The frame-pool A10 (this script) and the metrics-window A10 (a10_corpus_verdict.py) sample the
SAME window at different cadences, so they should agree to within sampling noise -> reporting
both is a cross-check (agreement = double-backing; large disagreement = a bug to investigate).

    python a10_from_zarr.py <CASE>     # e.g. ou_relam90_256_fp64
    # reads $TURBGEN_DATA_DIR/corpus/<CASE>.zarr ; writes <CASE>_A10_frames.json beside it
"""
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import solver._env  # noqa: F401  (OpenMP guard, must precede torch)
import numpy as np
import torch
import zarr

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
_flags = [a for a in sys.argv[1:] if a.startswith("--")]
CASE = _args[0] if _args else "ou_relam90_256_fp64"
SEED_SCAN = "--seed-scan" in _flags   # report A10 vs N_seed curve (no-floor / sampling test)
SIGNS = "--signs" in _flags           # report per-seed SIGNED <u_i u_j> + sign balance:
#   the direct test of whether per-seed ou_seed randomized the cross-correlation signs.
#   shared ou_seed -> all same sign (don't cancel); independent -> ~half/half (cancel).
CROSS_THRESH = 2.0   # same as a10_corpus_verdict.py
COMP_THRESH = 5.0
_ROOT = Path(os.environ.get("TURBGEN_DATA_DIR", "./data"))
ZARR = _ROOT / "corpus" / f"{CASE}.zarr"
if not ZARR.exists():
    sys.exit(f"[A10-frames] no zarr at {ZARR}")

store = zarr.open_group(str(ZARR), mode="r")
if "u" not in store or "seed" not in store:
    sys.exit(f"[A10-frames] {ZARR} missing 'u' or 'seed' dataset (old/partial store?) "
             f"-- re-run corpus_to_zarr.py")
zu = store["u"]                         # (N, 3, Ng, Ng, Ng) fp32, raw physical
zseed = store["seed"][:]               # (N,) true pool number per frame
N = zu.shape[0]
if N == 0:
    sys.exit(f"[A10-frames] {ZARR} has 0 frames")
# guard: never verdict a partially-written store (unwritten frames are all-zero -> would
# bias <u_i^2>~0 toward a FALSE PASS). corpus_to_zarr sets attrs['complete']=True only after
# the last frame (mirror freeze_norm_from_zarr's guard). Refuse if absent AND last frame zero.
if not store.attrs.get("complete", False):
    _last = np.asarray(zu[N - 1])
    if float(np.abs(_last).max()) == 0.0:
        sys.exit(f"REFUSE: {ZARR} has no 'complete' flag and its last frame is all-zero "
                 f"-> partial store. Re-run corpus_to_zarr.py before judging A10.")
    print("  WARNING: no 'complete' flag (pre-flag store); last frame non-zero, proceeding")
dev = "cuda" if torch.cuda.is_available() else "cpu"
PAIRS = ((0, 1), (0, 2), (1, 2))

# per-frame moments (one frame in memory at a time -> no OOM on 256^3)
ui2 = np.zeros((N, 3), dtype=np.float64)
uij = np.zeros((N, 3), dtype=np.float64)
for i in range(N):
    u = torch.from_numpy(zu[i]).to(dev, torch.float64)   # [3,Ng,Ng,Ng]
    ui2[i] = [float((u[c] * u[c]).mean()) for c in range(3)]
    uij[i] = [float((u[a] * u[b]).mean()) for a, b in PAIRS]
    del u
    if (i + 1) % 100 == 0 or i == N - 1:
        print(f"  {i+1}/{N} frames scanned", flush=True)

seeds = sorted(set(int(s) for s in zseed))


def _verdict(ui2_mean, uij_mean):
    K2 = 0.5 * float(ui2_mean.sum())
    cross = 100.0 * max(abs(x) for x in uij_mean) / (2.0 * K2 / 3.0)
    comp = 100.0 * max(abs(3.0 * x / (2.0 * K2) - 1.0) for x in ui2_mean)
    return cross, comp


# POOLED over ALL frames of ALL seeds (each frame = one equal-weight sample; signed uij)
pooled_cross, pooled_comp = _verdict(ui2.mean(axis=0), uij.mean(axis=0))

# per-seed (each seed's own frames pooled) -> the high-variance distribution to report
per_cross, per_comp = [], []
per_uij = []   # signed per-seed <u_i u_j> vector (for --signs)
for s in seeds:
    msk = (zseed == s)
    seed_uij = uij[msk].mean(axis=0)
    c, k = _verdict(ui2[msk].mean(axis=0), seed_uij)
    per_cross.append(c)
    per_comp.append(k)
    per_uij.append(seed_uij)

# --signs: the DIRECT test that per-seed ou_seed randomized the cross-correlation signs.
# Shared ou_seed -> every seed same sign on a given <u_iu_j> (e.g. #1's <vw> all +) -> signed
# pooling can't cancel. Independent ou_seed -> signs ~half/half -> cancel to isotropic.
if SIGNS:
    per_uij_arr = np.array(per_uij)   # (n_seed, 3) for pairs (uv, uw, vw)
    names = ["uv", "uw", "vw"]
    print(f"\n=== per-seed SIGNED <u_i u_j> sign balance ({CASE}, {len(seeds)} seeds) ===")
    print("  (independent ou_seed -> ~half +/half - per pair -> signed pooling cancels)")
    for p, nm in enumerate(names):
        vals = per_uij_arr[:, p]
        npos = int((vals > 0).sum())
        nneg = int((vals < 0).sum())
        balanced = "BALANCED (signs randomized -> will cancel)" if min(npos, nneg) >= len(seeds) * 0.3 \
            else "SKEWED (common bias -> won't cancel!)"
        print(f"  <{nm}>: {npos} pos / {nneg} neg  -> {balanced}")
    print("  -> if all three are BALANCED, per-seed ou_seed worked; pooled A10 should drop.")

cross_ok = pooled_cross <= CROSS_THRESH
comp_ok = pooled_comp <= COMP_THRESH
verdict = "PASS" if (cross_ok and comp_ok) else "FAIL"

print(f"\n=== A10 on RELEASED FRAMES: {CASE} ({len(seeds)} seeds, {N} frames) ===")
print(f"  POOLED over all {N} released frames (the publishable verdict):")
print(f"    A10 cross = {pooled_cross:.2f}%  (<= {CROSS_THRESH})  -> {'OK' if cross_ok else 'FAIL'}")
print(f"    A10 comp  = {pooled_comp:.2f}%  (<= {COMP_THRESH})  -> {'OK' if comp_ok else 'FAIL'}")
print(f"  Per-seed (high-variance): cross mean {np.mean(per_cross):.2f}% +/- {np.std(per_cross):.2f}% "
      f"range [{min(per_cross):.2f},{max(per_cross):.2f}] "
      f"({sum(1 for x in per_cross if x > 2)}/{len(per_cross)} seeds individually >2%)")
print(f"  VERDICT (released frame-set): {verdict}")
if verdict == "FAIL":
    print(f"  ACTION (per the A10 ladder): {pooled_cross:.2f}% -> "
          + ("2.0-2.5% band: add this config's seeds to ~20." if pooled_cross <= 2.5
             else ">2.5%: STOP + diagnose (shared ou_seed common bias? not just undersampling)."))

# ---- optional: A10 vs N_seed scan (floor test / sampling-curve, no GPU) ----
# Pools cumulative seed subsets (1,2,4,8,...,all) and reports pooled A10 at each N.
# If A10 ~ C/sqrt(N_frames_total) cleanly -> NO floor, more seeds/frames work, and the
# curve reads off exactly how many are needed for <1.8%. If A10 flattens early -> a floor
# (e.g. a common bias all seeds share); then more sampling won't help and we investigate.
scan = []
if SEED_SCAN:
    print(f"\n=== A10 vs N_seed scan ({CASE}) ===")
    print(f"  {'N_seed':>6}{'N_frames':>9}{'A10_cross%':>11}{'A10_comp%':>10}{'C=cross*sqrt(Nfr)':>18}")
    import math as _m
    Ns = [n for n in (1, 2, 4, 6, 8, 12, 16, 20, 24, 32) if n <= len(seeds)]
    if len(seeds) not in Ns:
        Ns.append(len(seeds))
    for n in Ns:
        sub = seeds[:n]
        msk = np.isin(zseed, sub)
        nfr = int(msk.sum())
        c, k = _verdict(ui2[msk].mean(axis=0), uij[msk].mean(axis=0))
        Cfit = c * _m.sqrt(nfr)
        scan.append({"n_seed": n, "n_frames": nfr, "cross_pct": round(c, 3),
                     "comp_pct": round(k, 3), "C_fit": round(Cfit, 1)})
        print(f"  {n:>6}{nfr:>9}{c:>10.2f}%{k:>9.2f}%{Cfit:>17.1f}")
    # interpretation: does pooled A10 keep FALLING as N grows (no floor -> add sampling
    # works), or FLATTEN (floor -> common bias, more seeds won't help)? Compare the
    # last-half average to the first multi-seed point; ignore N=1 (no cancellation yet).
    multi = [r for r in scan if r["n_seed"] >= 2]
    if len(multi) >= 3:
        first = multi[0]["cross_pct"]
        last = multi[-1]["cross_pct"]
        # if A10 fell clean-ish, last << first; floor -> last ~ first despite many more frames
        ratio = last / first if first > 0 else 1.0
        nfr_ratio = multi[-1]["n_frames"] / multi[0]["n_frames"]
        ideal = 1.0 / _m.sqrt(nfr_ratio)   # pure 1/sqrt(N) would give this ratio
        if last <= 1.8:
            print(f"  -> pooled A10 reached {last:.2f}% at N={multi[-1]['n_seed']} "
                  f"({multi[-1]['n_frames']} frames) — meets <1.8%. NO floor; sampling works.")
        elif ratio <= ideal * 1.8:   # fell at least ~half as fast as ideal -> sampling-limited
            need = (last * _m.sqrt(multi[-1]['n_frames']) / 1.8) ** 2
            print(f"  -> pooled A10 still falling ({first:.2f}%->{last:.2f}% as frames "
                  f"x{nfr_ratio:.0f}; ideal x{1/ideal:.1f}). NO clear floor — sampling-limited. "
                  f"For <1.8%: need ~{need:.0f} pooled frames "
                  f"(~{need/(N/len(seeds)):.0f} seeds at this frames/seed).")
        else:
            print(f"  -> pooled A10 BARELY moved ({first:.2f}%->{last:.2f}% despite frames "
                  f"x{nfr_ratio:.0f}; 1/sqrt(N) expected x{1/ideal:.1f} drop) -> a FLOOR. "
                  f"More seeds give diminishing returns; investigate common bias "
                  f"(shared ou_seed across seeds?) before scaling up.")

out = {
    "case": CASE, "source": "released zarr frames (a10_from_zarr.py)",
    "n_seeds": len(seeds), "n_frames": N,
    "seed_scan": scan,
    "pooled_A10_cross_pct": round(pooled_cross, 3),
    "pooled_A10_comp_pct": round(pooled_comp, 3),
    "per_seed_cross_mean_pct": round(float(np.mean(per_cross)), 3),
    "per_seed_cross_std_pct": round(float(np.std(per_cross)), 3),
    "per_seed_cross_range": [round(min(per_cross), 2), round(max(per_cross), 2)],
    "n_seeds_individually_over_2pct": int(sum(1 for x in per_cross if x > 2)),
    "thresholds": {"cross": CROSS_THRESH, "comp": COMP_THRESH},
    "verdict": verdict,
}
(ZARR.parent / f"{CASE}_A10_frames.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"  wrote {ZARR.parent / f'{CASE}_A10_frames.json'}")
sys.exit(0 if verdict == "PASS" else 2)
