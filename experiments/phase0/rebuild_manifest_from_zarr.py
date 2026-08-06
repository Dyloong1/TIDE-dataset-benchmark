"""Rebuild a corpus manifest.json from the released zarr store.

Needed when the per-seed `.pt` frame dirs + results/<case>_corpus_poolN/metrics.json
were deleted (zarr-only workflow), so write_corpus_manifest.py (which counts *.pt
files) cannot reconstruct the manifest. Precedent: CYB-q rebuilt the kf4 manifest
"from pool metrics + zarr" (AGENT_RELAY.md 2026-07-06) as an uncommitted one-off; this
git-ifies that path so it is reproducible + self-checking.

Source of truth = the zarr's per-frame `seed` array (i8, one entry/frame giving the
IC pool number). accepted_seeds = unique(seed); frames_in_corpus[s] = count(seed==s);
total_frames = u.shape[0]. This is the SAME derivation a10_from_zarr.py uses
(a10_from_zarr.py:59/86/104).

make_splits.py only load-bears on `accepted_seeds` (list[int]) and
`per_seed[].{seed, frames_in_corpus}`; we also fill channels/total_frames/regime/
CAVEATS for write_dataset_manifest.py parity where derivable from the zarr alone
(enrichment fields like Re_lambda/K_drift come from per-seed metrics.json which are
GONE for zarr-only configs -> left null, honestly).

    python rebuild_manifest_from_zarr.py <CASE> [--write]

Without --write it is a DRY RUN: prints the derived summary and (if a
<CASE>_A10_frames.json sidecar exists next to the zarr) cross-checks n_seeds/n_frames
against it, but writes nothing. Judge-first: inspect the dry run, then re-run --write.
"""
import solver._env  # noqa: F401  (Windows OMP guard; must precede any torch/zarr import path)
import argparse
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import zarr
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("--write", action="store_true", help="actually write manifest.json (default: dry run)")
args = ap.parse_args()
CASE = args.case

ROOT = Path(os.environ.get("TURBGEN_DATA_DIR", "./data"))
CORPUS = ROOT / "corpus"
ZARR = CORPUS / f"{CASE}.zarr"
if not ZARR.exists():
    sys.exit(f"[rebuild] no zarr at {ZARR}")

store = zarr.open_group(str(ZARR), mode="r")

# ---- completeness guard (mirror a10_from_zarr.py:66-71 / freeze_norm_from_zarr.py:40-45)
complete = bool(store.attrs.get("complete", False))
if not complete:
    # fall back to the last-frame-nonzero heuristic the sibling scripts use
    last = store["u"][-1]
    if float(np.abs(last).max()) == 0.0:
        sys.exit(f"[rebuild] {CASE}: store not marked complete AND last frame all-zero -> refusing")
    print(f"[rebuild] WARN {CASE}: attrs.complete missing but last frame nonzero -> proceeding")

zseed = store["seed"][:]
N = int(store["u"].shape[0])
accepted = sorted(set(int(s) for s in zseed))
frames_by_seed = {int(s): int((zseed == s).sum()) for s in accepted}
assert sum(frames_by_seed.values()) == N, "per-seed counts must sum to total frames"

# 5th channel detection (theta scalar / b buoyancy), parity with corpus_to_zarr
channels = ["u", "v", "w", "p"]
extra = next((k for k in ("theta", "b") if k in store), None)
if extra:
    channels.append(extra)

# ---- frame_filter provenance DERIVED FROM THE ZARR (not hardcoded): the write_corpus_manifest
# convention is that a per-frame k_maxη>=1.5 gate, IF applied, drops sub-1.5 frames -> non-uniform
# spacing in 't'. We can't read keep_frac (metrics.json purged), but the zarr itself carries the
# truth: (a) does any exported frame have instantaneous k_maxη<1.5? (b) are frames uniformly
# spaced in 't' per seed? Decide the filter string + CAVEAT from those two measurements so the
# manifest's provenance matches the actual data, not an assumption.
_ke = store["k_max_eta"][:] if "k_max_eta" in store else None
_t = store["t"][:] if "t" in store else None
n_sub15 = int((_ke < 1.5).sum()) if _ke is not None else None
# per-seed dt uniformity (coefficient of variation of consecutive-frame dt, pooled over seeds)
dt_cov = None
if _t is not None:
    covs = []
    for s in accepted:
        ts = np.sort(_t[zseed == s])
        if ts.size > 2:
            d = np.diff(ts)
            if float(d.mean()) > 0:
                covs.append(float(d.std() / d.mean()))
    dt_cov = (sum(covs) / len(covs)) if covs else None
_uniform = (n_sub15 == 0) and (dt_cov is not None and dt_cov < 0.05)
if _uniform:
    frame_filter = ("none — VERIFIED from zarr: all exported frames have instantaneous k_maxη>=1.5 "
                    f"(0 sub-1.5 of {N}) and are near-uniformly spaced in t (dt CoV={dt_cov:.4f}). "
                    "Trajectory is Class-I by window-average k_maxη; no per-frame gate dropped frames.")
    caveats = ["Frames are (near-)uniformly spaced at the export cadence (verified dt CoV<5% from "
               "the stored 't'); loaders may treat dt as ~constant but the true 't' is stored per frame."]
else:
    frame_filter = (f"per-frame k_maxη>=1.5 gate likely applied — zarr shows {n_sub15} sub-1.5 frames "
                    f"and/or non-uniform dt (CoV={dt_cov}); frames are NOT uniformly spaced.")
    caveats = ["Frames are NOT uniformly spaced in time (a per-frame k_maxη>=1.5 gate dropped eps-peak "
               "instants). Loaders MUST use the stored 't', not assume uniform dt. High-order "
               "intermittency stats are a lower bound (right-truncated eps tail)."]

# ---- per-seed k_maxη: RECOVER it from the zarr instead of writing null ----
# This field used to be honest-null "because metrics.json was purged with the .pt frames". But the
# zarr stores per-frame k_max_eta (that is `_ke` above, already read for the frame_filter check),
# so the window average was recoverable all along -- we were discarding data we had in hand.
#
# Why it matters (found 2026-07-15 by CYB-3 on relam70, confirmed by q on rotating/scalar): the
# slice's seed-quality score standardizes each term across a config's seeds, and _z() maps an
# all-None column to all-zeros. With every enrichment field null, EVERY term died, all seeds scored
# q=0.000000, and "pick the 3 best-quality seeds" silently became "pick the 3 lowest seed numbers"
# -- while the published quality_formula still advertised a 5-term physical score. On relam70 the
# honest score changes the TRAINING SET: [0,1,2] -> [12,2,0].
#
# Only the machine holding the zarr can do this (data locality), so it has to happen here, at
# manifest-build time -- q's slice generator cannot recompute it for a corpus on another disk.
# Re_lambda / eps_mean / K_drift genuinely need metrics.json and stay null; k_maxη does not.
_kmax_by_seed = {}
if _ke is not None:
    for s in accepted:
        v = _ke[zseed == s]
        v = v[~np.isnan(v)]
        if v.size:
            _kmax_by_seed[s] = float(v.mean())

per_seed = [{"seed": s, "run": f"{CASE}_corpus" if s == 0 else f"{CASE}_corpus_pool{s}",
             "frames_in_corpus": frames_by_seed[s],
             # k_maxη: recovered from the zarr (see above). The rest genuinely need metrics.json,
             # which was deleted with the .pt frames -> honest null.
             "frame_keep_frac": None, "frames_seen": None, "frames_skipped_underresolved": None,
             "k_max_eta_window_avg": _kmax_by_seed.get(s), "Re_lambda": None, "eps_mean": None,
             "K_drift_last5TL_pct": None}
            for s in accepted]

manifest = {
    "case": CASE,
    "compute_precision": "fp64",
    "storage_precision": "fp32",
    "channels": channels,
    "pressure_source": "operators.pressure_hat (D4-certified)",
    "regime": {"note": "rebuilt from zarr; per-seed solver metrics unavailable (zarr-only, .pt purged)",
               "input": {}, "measured": {}},
    "n_seeds_accepted": len(accepted),
    "n_seeds_rejected": None,  # unknown from zarr alone (rejected seeds simply absent from store)
    "accepted_seeds": accepted,
    "rejected_seeds": [],
    "total_frames": N,
    "frame_keep_frac_min": None,
    "frame_keep_frac_max": None,
    "pooled_A10": None,
    "per_seed": per_seed,
    "frame_filter": frame_filter,
    "CAVEATS": caveats,
    "PROVENANCE": "rebuilt by rebuild_manifest_from_zarr.py from released zarr (accepted_seeds + "
                  "frames_in_corpus derived from per-frame seed array; frame_filter/CAVEATS DERIVED "
                  "from the zarr's own k_max_eta + t arrays, NOT hardcoded; NOT from .pt frame dirs)",
}

print(f"[rebuild] {CASE}: {len(accepted)} accepted seeds, {N} total frames")
print(f"[rebuild] channels = {channels}")
print(f"[rebuild] accepted_seeds = {accepted}")
fc = sorted(set(frames_by_seed.values()))
print(f"[rebuild] frames/seed = {frames_by_seed if len(fc) > 1 else f'uniform {fc[0]}'}")

# ---- cross-check against the A10_frames sidecar written by a10_from_zarr.py FROM this zarr
sidecar = CORPUS / f"{CASE}_A10_frames.json"
if sidecar.exists():
    sc = json.loads(sidecar.read_text(encoding="utf-8"))
    ok_seeds = sc.get("n_seeds") == len(accepted)
    ok_frames = sc.get("n_frames") == N
    print(f"[xcheck] sidecar n_seeds={sc.get('n_seeds')} (match={ok_seeds}), "
          f"n_frames={sc.get('n_frames')} (match={ok_frames})")
    if not (ok_seeds and ok_frames):
        sys.exit("[rebuild] ABORT: derived counts disagree with A10 sidecar")
else:
    print("[xcheck] no A10_frames sidecar to cross-check against")

if args.write:
    out = CORPUS / CASE / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[rebuild] WROTE {out}")
else:
    print("[rebuild] DRY RUN (no --write): nothing written. Re-run with --write to commit the manifest.")
