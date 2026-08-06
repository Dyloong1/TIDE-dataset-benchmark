"""q1: convert the multi-seed .pt corpus to a chunked zarr store + freeze normalization
constants.

The corpus frames (per-seed frame*.pt, 4-channel u,v,w,p fp32) are converted to a
single framework-agnostic zarr store with one chunk per frame, plus a per-channel
welford-accumulated normalization constants file (min/max/mean/std) written to the
store attrs AND a sidecar JSON. The normalizer helper (q2, solver/normalizer.py)
consumes these frozen constants to apply [-1,1] / standardize transforms at load
time; this script only COMPUTES and FREEZES them (no transform applied to stored data
-- the zarr holds raw physical fp32 values, exactly as the .pt frames).

Layout:
  corpus.zarr/
    u   : (N, 3, 256, 256, 256) fp32, chunk (1, 3, 256, 256, 256)  -- one frame/chunk
    p   : (N, 256, 256, 256)    fp32, chunk (1, 256, 256, 256)
    t   : (N,) f8     -- physical time of each frame (NON-uniform; see manifest caveat)
    k_max_eta : (N,) f8
    seed: (N,) i8     -- which independent seed each frame came from
    .attrs: norm constants per channel, provenance, caveats

    python corpus_to_zarr.py <CASE>     # default ou_relam90_256_fp64
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import solver._env  # noqa: F401
import numpy as np
import torch
import zarr
from numcodecs import Blosc

CASE = sys.argv[1] if len(sys.argv) > 1 else "ou_relam90_256_fp64"
# Storage root is cross-platform via TURBGEN_DATA_DIR (mirrors the PHYSICS_METRICS_DIR
# pattern in CLAUDE.md): CYB-q -> "/media/ydai17/T7 Shield/turbgen_data", CYB-t -> its
# Windows external drive. Falls back to the legacy D:/turbgen_data for back-compat.
import os
_ROOT = Path(os.environ.get("TURBGEN_DATA_DIR", "./data"))
SRC = _ROOT / "corpus" / CASE
OUT = _ROOT / "corpus" / f"{CASE}.zarr"

# gather frames in a deterministic order: by seed dir, then frame index. Order pool
# dirs NUMERICALLY (pool1, pool2, ..., pool10, ...) not lexicographically, so the
# stored seed index maps to the natural pool number (corpus=0, poolK=K).
# Seed-dir names are derived from <CASE> (config-general): "<CASE>_corpus" is the base
# (seed 0), "<CASE>_corpus_pool<K>" is seed K.
import re as _re
def _pool_num(d):
    m = _re.search(r"_pool(\d+)$", d.name)
    return int(m.group(1)) if m else 0   # the base "_corpus" dir -> 0
seed_dirs = [SRC / f"{CASE}_corpus"] + sorted(
    SRC.glob(f"{CASE}_corpus_pool*"), key=_pool_num)
seed_dirs = [d for d in seed_dirs if d.is_dir()]
frames = []  # (path, TRUE pool/seed number — NOT the enumeration index)
# Store the true IC pool number (_corpus->0, _corpus_poolK->K), not enumerate()'s compacted
# index. If any seed was gate-rejected, the dirs present are e.g. {0,1,2,4,5,6,7} and an
# enumeration index would relabel pool 4 as seed 3 etc. — silently breaking the by-IC split
# key (a held-out test seed could collide with a train seed). The seed array MUST carry the
# true pool number so a loader can filter the zarr by (case, seed) exactly as make_splits emits.
for d in seed_dirs:
    sn = _pool_num(d)
    for fp in sorted(d.glob("frame*.pt")):
        frames.append((fp, sn))
N = len(frames)
if N == 0:
    sys.exit(f"no frames under {SRC}")
print(f"{CASE}: {len(seed_dirs)} seeds, {N} frames -> {OUT}")

# Detect grid size + the optional 5th channel ("theta" for scalar, "b" for stratified)
# from the first frame -> config-general 4ch (u,v,w,p) or 5ch (+theta/+b).
_probe = torch.load(frames[0][0], map_location="cpu", weights_only=False)
Ngrid = int(_probe["u"].shape[-1])
EXTRA = next((k for k in ("theta", "b") if k in _probe), None)   # 5th channel name or None
del _probe
print(f"  grid N={Ngrid}, channels = u,v,w,p" + (f",{EXTRA}  (5ch)" if EXTRA else "  (4ch)"))

# DISKGUARD (parity with CYB-t 3a35771): -KeepPt retains ~300-400GB .pt per A10-pending
# config; if several pile up, a zarr write can fill the disk MID-WRITE and corrupt the store.
# Estimate the zarr footprint and abort CLEANLY (keeping .pt, losing nothing) if free space is
# short. Estimate = raw bytes (zstd typically shrinks, so raw is a conservative upper bound) +
# 50 GB headroom. shutil.disk_usage on the OUT parent (the actual target filesystem).
import shutil as _shutil
_nch = 4 + (1 if EXTRA else 0)                       # u,v,w,p (+theta/b)
_raw_bytes = N * _nch * (Ngrid ** 3) * 4             # f4
_need = int(_raw_bytes + 50 * 1024**3)               # + 50 GB margin (raw is conservative vs zstd)
_free = _shutil.disk_usage(OUT.parent).free
if _free < _need:
    sys.exit(f"DISK ABORT: need ~{_need/1024**3:.0f} GB (raw upper bound + 50 GB margin) for "
             f"{CASE}.zarr ({N} frames x {_nch}ch x {Ngrid}^3), but only {_free/1024**3:.0f} GB "
             f"free on {OUT.parent}. NOT writing zarr; .pt kept intact. Free space (delete stale "
             f"dropped-config .pt / old zarr) and re-run corpus_to_zarr.")
print(f"  diskguard OK: ~{_need/1024**3:.0f} GB needed (conservative), {_free/1024**3:.0f} GB free")

comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
store = zarr.open_group(str(OUT), mode="w")
z_u = store.create_dataset("u", shape=(N, 3, Ngrid, Ngrid, Ngrid),
                           chunks=(1, 3, Ngrid, Ngrid, Ngrid), dtype="f4", compressor=comp)
z_p = store.create_dataset("p", shape=(N, Ngrid, Ngrid, Ngrid),
                           chunks=(1, Ngrid, Ngrid, Ngrid), dtype="f4", compressor=comp)
z_extra = (store.create_dataset(EXTRA, shape=(N, Ngrid, Ngrid, Ngrid),
                                chunks=(1, Ngrid, Ngrid, Ngrid), dtype="f4", compressor=comp)
           if EXTRA else None)
z_t = store.create_dataset("t", shape=(N,), chunks=(N,), dtype="f8")
z_ke = store.create_dataset("k_max_eta", shape=(N,), chunks=(N,), dtype="f8")
z_seed = store.create_dataset("seed", shape=(N,), chunks=(N,), dtype="i8")
# Standalone Welford accumulator for the 5th channel (reuses solver.normalizer's
# vetted _GroupStats WITHOUT modifying solver/ — the scalar/buoyancy gets its OWN
# normalization group, separate from velocity/pressure).
from solver.normalizer import _GroupStats as _GS
extra_stat = _GS() if EXTRA else None

# Frozen normalization constants via CYB-q's vetted NormalizerFitter (q2). KEY: it
# normalizes the 3 velocity components JOINTLY by ONE statistic -- per-component
# scaling would manufacture anisotropy and break the A10 isotropy certification.
# (This replaces an earlier per-component Welford here, which was wrong on that point.)
from solver.normalizer import NormalizerFitter
fit = NormalizerFitter()

for i, (fp, si) in enumerate(frames):
    d = torch.load(fp, map_location="cpu", weights_only=False)
    u = d["u"]                  # torch [3,256,256,256] fp32
    p = d["p"]                  # torch [256,256,256]
    z_u[i] = u.numpy()
    z_p[i] = p.numpy()
    z_t[i] = float(d["t"])
    z_ke[i] = float(d.get("k_max_eta", np.nan))
    z_seed[i] = si
    fit.update(u, p)            # velocity joint group + pressure group (Welford, fp64)
    if EXTRA is not None:
        ex = d[EXTRA]           # torch [256,256,256] fp32 (theta or b)
        z_extra[i] = ex.numpy()
        extra_stat.update(ex)
    if (i + 1) % 200 == 0 or i == N - 1:
        print(f"  {i+1}/{N} frames written", flush=True)

# freeze both modes so downstream can pick minmax ([-1,1]) or standardize (z-score)
norm_minmax = fit.freeze("minmax", note=f"{CASE} corpus, {N} frames")
norm_standardize = fit.freeze("standardize", note=f"{CASE} corpus, {N} frames")

# 5th-channel (theta/b) frozen norm consts, written to attrs (own group, raw stored)
extra_norm = None
if EXTRA is not None:
    extra_norm = {"vmin": extra_stat.vmin, "vmax": extra_stat.vmax,
                  "mean": extra_stat.mean, "std": extra_stat.std,
                  "minmax_scale": extra_stat.absmax, "n_samples": extra_stat.n}
    store.attrs[f"norm_{EXTRA}"] = extra_norm

store.attrs["case"] = CASE
store.attrs["n_frames"] = N
# n_seeds is a PUBLISHED claim (it appears in the DATASHEET as an acceptance fact), so verify it
# instead of trusting the directory count. `len(seed_dirs)` counted DIRECTORIES: decay_hotstart_re86
# shipped attrs["n_seeds"]=8 for 8 pools that are 8 BIT-IDENTICAL copies of one trajectory (it
# warm-starts from a checkpoint, so the pools varied the seed LABEL while resuming the same state).
# Nothing caught it, and every downstream mean±std over those "8 seeds" was really n=1.
# Compare each seed's FIRST frame against seed 0's: identical first frames => same trajectory.
_first_by_seed = {}
_seed_arr = store["seed"][:]
_t_arr = store["t"][:]
for _s in sorted(set(int(x) for x in _seed_arr)):
    _idx = [i for i in range(len(_seed_arr)) if int(_seed_arr[i]) == _s]
    _idx.sort(key=lambda i: float(_t_arr[i]))
    _first_by_seed[_s] = store["u"][_idx[0]]
_seeds_sorted = sorted(_first_by_seed)
_dupes = []
for _a in range(len(_seeds_sorted)):
    for _b in range(_a + 1, len(_seeds_sorted)):
        _sa, _sb = _seeds_sorted[_a], _seeds_sorted[_b]
        if float(np.abs(_first_by_seed[_sa] - _first_by_seed[_sb]).max()) == 0.0:
            _dupes.append((_sa, _sb))
_n_independent = len(_seeds_sorted) - len({b for _, b in _dupes})
store.attrs["n_seeds"] = len(seed_dirs)
store.attrs["n_independent_trajectories"] = _n_independent
if _dupes:
    _msg = (f"{len(_dupes)} seed pair(s) share a BIT-IDENTICAL first frame {_dupes} -> these are "
            f"copies of one trajectory, not independent seeds. n_seeds={len(seed_dirs)} counts "
            f"directories; only {_n_independent} are independent. Do NOT report mean+/-std over them.")
    store.attrs["seed_independence_warning"] = _msg
    print(f"\n*** WARNING: {CASE}: {_msg}\n", flush=True)
else:
    print(f"[seeds] {CASE}: {_n_independent}/{len(_seeds_sorted)} seeds verified independent "
          f"(no bit-identical first frames)", flush=True)
store.attrs["channels"] = ["velocity (u,v,w joint)", "p"] + ([EXTRA] if EXTRA else [])
store.attrs["compute_precision"] = "fp64"
store.attrs["storage_precision"] = "fp32"
store.attrs["pressure_source"] = "operators.pressure_hat (D4-certified)"
store.attrs["normalizer"] = "solver.normalizer.FrozenNormalizer (velocity JOINT, vetted q2)"
store.attrs["CAVEATS"] = [
    "Stored values are RAW physical fp32 fields (no normalization applied). Use "
    "solver/normalizer.py FrozenNormalizer.from_json(<case>_norm_*.json) to transform "
    "at load time; velocity (u,v,w) is normalized JOINTLY (one scale) to preserve "
    "A10 component isotropy.",
    "Frames are filtered by instantaneous k_maxeta>=1.5 and are NOT uniformly spaced "
    "in time -- use the stored 't' array, do not assume uniform dt.",
    "The corpus under-samples the strong-dissipation (eps-peak) tail; high-order "
    "intermittency statistics are a lower bound (see DATASHEET.md).",
]

# sidecar JSONs (so training reads the frozen normalizer without opening zarr)
side_mm = OUT.parent / f"{CASE}_norm_minmax.json"
side_std = OUT.parent / f"{CASE}_norm_standardize.json"
norm_minmax.to_json(side_mm)
norm_standardize.to_json(side_std)

# completion flag LAST: a crash mid-conversion leaves this unset, so freeze_norm_from_
# zarr.py (and any consumer) can detect a partial store instead of trusting zeros.
store.attrs["complete"] = True

print(f"DONE: zarr at {OUT}")
print(f"  minmax norm     -> {side_mm}")
print(f"  standardize norm -> {side_std}")
