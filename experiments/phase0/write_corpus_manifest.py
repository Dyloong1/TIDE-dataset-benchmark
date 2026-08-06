"""Write the corpus manifest after run_corpus.ps1 finishes.

Collects per-seed acceptance + frame-retention stats and the pooled A10, and
records the two caveats CYB-q flagged:
  1. per-seed keep_frac (frames kept after the instantaneous k_maxη>=1.5 filter);
     a low keep_frac flags a possibly under-resolved trajectory.
  2. filtered frames are NOT uniformly spaced in time (dropped slots are eps-peak
     instants) -- each frame stores its true 't'; loaders must not assume uniform dt.

    python write_corpus_manifest.py <CASE> <accepted_csv> <rejected_csv>
"""
import json
import sys
import argparse
from pathlib import Path

# UTF-8 console: the summary print echoes the A10 cell (Chinese), which would crash
# this last pipeline step on Windows cp1252 (manifest.json itself is utf-8-safe).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).parent
RES = HERE / "results"
# Named args (NOT positional): PowerShell drops an empty `($x -join ',')` argument
# entirely, which would shift a positional rejected-list into the accepted slot.
ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("--accepted", default="")
ap.add_argument("--rejected", default="")
args = ap.parse_args()
CASE = args.case
accepted = [s for s in args.accepted.split(",") if s]
rejected = [s for s in args.rejected.split(",") if s]
import os
_ROOT = Path(os.environ.get("TURBGEN_DATA_DIR", "./data"))  # cross-platform (see corpus_to_zarr)
CORPUS = _ROOT / "corpus" / CASE

def rc_name(seed):
    return f"{CASE}_corpus" if seed == "0" else f"{CASE}_corpus_pool{seed}"

seeds = []
total_frames = 0
for s in accepted:
    rc = rc_name(s)
    mp = RES / rc / "metrics.json"
    m = json.load(open(mp, encoding="utf-8")) if mp.exists() else {}
    fdir = CORPUS / rc
    nf = len(list(fdir.glob("*.pt"))) if fdir.exists() else 0
    total_frames += nf
    seeds.append({
        "seed": int(s), "run": rc, "frames_in_corpus": nf,
        "frame_keep_frac": m.get("frame_keep_frac"),
        "frames_seen": m.get("frames_seen"),
        "frames_skipped_underresolved": m.get("frames_skipped_underresolved"),
        "k_max_eta_window_avg": m.get("k_max_eta"),
        "Re_lambda": m.get("Re_lambda"),
        "eps_mean": m.get("eps_mean"),
        "K_drift_last5TL_pct": m.get("K_drift_last5TL_pct"),
    })

# pooled A10 from the corpus record's appendix (REC = <CASE>_corpus)
pooled_a10 = None
appx = RES / f"{CASE}_corpus" / "DNS_STANDARD_APPENDIX_A.md"
if appx.exists():
    for line in appx.read_text(encoding="utf-8").splitlines():
        if line.startswith("| A10"):
            pooled_a10 = line.split("|")[2].strip()

kfs = [s["frame_keep_frac"] for s in seeds if s["frame_keep_frac"] is not None]

# ---- channel detection (config-general): probe a frame for the optional 5th channel
# (theta for scalar, b for stratified) instead of hardcoding 4ch. Mirrors corpus_to_zarr.
channels = ["u", "v", "w", "p"]
for s in accepted:
    fdir = CORPUS / rc_name(s)
    _f = sorted(fdir.glob("*.pt")) if fdir.exists() else []
    if _f:
        try:
            import torch
            _probe = torch.load(_f[0], map_location="cpu")
            extra = next((k for k in ("theta", "b") if k in _probe), None)
            if extra:
                channels.append(extra)
        except Exception:
            pass
        break

# ---- regime descriptor (S1: per-config MEASURED triplet, no orthogonal-axis claim) ----
# k_f and tau are CONFOUNDED with Re (higher-k injection / longer correlation shift eps
# -> Re). Only the pure-nu Re axis {50,70,86} is single-variable. We record the INPUT
# forcing knobs (nu/k_f/tau/sigma2 from the config) AND the MEASURED equilibrated
# (Re_lambda, eps, k_max_eta) so the catalog/paper can present measured-regime coverage
# honestly instead of implying clean orthogonal axes.
_cfgp = (Path(__file__).resolve().parents[0] / "configs" / f"{CASE}.yaml")
regime = {"note": "k_f/tau are confounded with Re; only pure-nu Re axis is single-variable"}
try:
    import yaml as _yaml
    _c = _yaml.safe_load(_cfgp.read_text(encoding="utf-8"))
    _f = (_c.get("solver", {}) or {}).get("forcing", {}) or {}
    regime["input"] = {"nu": (_c.get("solver", {}) or {}).get("nu"),
                       "k_f": _f.get("k_f"), "ou_tau": _f.get("ou_tau"),
                       "ou_sigma2": _f.get("ou_sigma2"), "forcing_type": _f.get("type")}
except Exception as _e:
    regime["input"] = {"error": f"config unreadable: {_e}"}
_rls = [s["Re_lambda"] for s in seeds if s.get("Re_lambda") is not None]
_eps = [s["eps_mean"] for s in seeds if s.get("eps_mean") is not None]
regime["measured"] = {
    "Re_lambda_mean": (sum(_rls) / len(_rls)) if _rls else None,
    "eps_mean": (sum(_eps) / len(_eps)) if _eps else None,
    "k_max_eta_window_avg": seeds[0]["k_max_eta_window_avg"] if seeds else None,
}

manifest = {
    "case": CASE,
    "compute_precision": "fp64",
    "storage_precision": "fp32",
    "channels": channels,
    "pressure_source": "operators.pressure_hat (D4-certified)",
    "regime": regime,
    "n_seeds_accepted": len(accepted),
    "n_seeds_rejected": len(rejected),
    "accepted_seeds": [int(s) for s in accepted],
    "rejected_seeds": [int(s) for s in rejected if s],
    "total_frames": total_frames,
    "frame_keep_frac_min": min(kfs) if kfs else None,
    "frame_keep_frac_max": max(kfs) if kfs else None,
    "pooled_A10": pooled_a10,
    "per_seed": seeds,
}

# frame_filter + CAVEATS are DYNAMIC: depends on whether a per-frame k_maxη floor was applied.
# Detect from keep_frac: floor=0 (no per-frame gate, run_forced_hit --frame-keta-floor 0) keeps
# every window frame -> keep_frac == 1.0; floor>=1.5 (old behavior) drops sub-1.5 frames -> <1.0.
# (CYB-t 2026-06-27: the per-frame gate is NOT required by the DNS standard — eval judges Class
# by WINDOW-AVERAGE k_maxη; instantaneous fraction is report-only. So floor=0 ships uniform-count,
# fully-DNS-compliant trajectories; we must declare the filter HONESTLY per actual floor.)
_no_perframe_filter = (kfs and min(kfs) >= 0.999)
if _no_perframe_filter:
    manifest["frame_filter"] = ("none (all window frames exported; trajectory is Class-I by "
                                "WINDOW-AVERAGE k_maxη per the DNS standard — instantaneous "
                                "k_maxη is report-only, not a per-frame gate)")
    manifest["CAVEATS"] = [
        "No per-frame resolution gate was applied: EVERY frame in the sampling window is "
        "exported, so frames ARE (near-)uniformly spaced at the export cadence. Each frame "
        "still stores its own 'k_max_eta'; SOME frames have instantaneous k_maxη<1.5 (eps-peak "
        "instants) -- this is physical (real turbulence ε fluctuates) and does NOT violate "
        "acceptance: the TRAJECTORY is Class-I by the window-average convention (DNS standard), "
        "and per-frame k_maxη is provided for OPTIONAL downstream filtering.",
        "Because the eps-PEAK (strongest-small-scale, most intermittent) instants are KEPT (not "
        "dropped), this corpus does NOT under-sample the strong-dissipation tail — high-order "
        "intermittency statistics are representative of the resolved trajectory. (Frames whose "
        "instantaneous k_maxη<1.5 sit at the grid's resolution edge for that burst; users wanting "
        "a strict per-frame Class-I subset can filter on the stored 'k_max_eta'.)",
    ]
else:
    manifest["frame_filter"] = "instantaneous k_maxη>=1.5 (per-frame Class-I resolution gate)"
    manifest["CAVEATS"] = [
        "Frames are filtered by instantaneous k_maxeta>=1.5, so they are NOT uniformly "
        "spaced in time (dropped slots are eps-peak instants). Each frame stores its "
        "true 't' AND its 'k_max_eta' -- loaders MUST use the stored 't' and not "
        "assume a uniform dt between consecutive frames.",
        "INTERMITTENCY BIAS (physical consequence, not just timing): the dropped frames "
        "are the eps-PEAK instants -- the most intermittent, strongest-small-scale "
        "moments. They are dropped because at those instants k_maxeta<1.5, i.e. the GRID "
        "cannot faithfully resolve that burst (spectral pileup at the cutoff); keeping "
        "them would feed under-resolved numerical artefacts to a model. The consequence "
        "is that this corpus SYSTEMATICALLY UNDER-SAMPLES the strong-dissipation tail: "
        "the eps distribution is right-truncated and high-order intermittency statistics "
        "(derivative flatness/kurtosis B3, high-order structure functions) are a LOWER "
        "BOUND, not unbiased. For Re_lambda~86 the truncation is mild (dropped eps ~1.3-"
        "1.4x the mean, not extreme tails), but downstream users must NOT treat this as "
        "an unbiased sample of real turbulence's extreme events.",
        "A seed with low frame_keep_frac (<0.80) kept fewer frames because its "
        "instantaneous k_maxeta dipped below 1.5 more often; such a trajectory may be "
        "intrinsically closer to the resolution edge -- inspect before publishing.",
    ]
CORPUS.mkdir(parents=True, exist_ok=True)
(CORPUS / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[manifest] {CASE}: {len(accepted)} seeds, {total_frames} frames, "
      f"keep_frac {manifest['frame_keep_frac_min']}..{manifest['frame_keep_frac_max']}, "
      f"pooled A10 = {pooled_a10}")
print(f"[manifest] wrote {CORPUS / 'manifest.json'}")
