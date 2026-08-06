"""Aggregate per-config corpus manifests into ONE top-level dataset manifest.

This is the dataset-release (KDD) manifest for the turbgen 256^3 DNS corpus. It
reads every per-config manifest written by write_corpus_manifest.py
(`$TURBGEN_DATA_DIR/corpus/<CASE>/manifest.json`) and emits a single
`$TURBGEN_DATA_DIR/corpus/dataset_manifest.json`.

CPU-only, deterministic, re-runnable. Robust to incomplete production: missing
per-config manifests are reported and skipped, so this can be re-run as more
configs land. Does NOT touch the solver or any eval script and never runs on GPU.

    python write_dataset_manifest.py

HONEST regime framing (spec S1, 2026-06-24): the three "axes" are NOT orthogonal.
Only the pure-nu Re axis {Re_lambda 50, 70, 86} is a true single-variable axis;
k_f and tau are CONFOUNDED with Re (higher-k injection / longer correlation shift
eps -> Re). The `axes` block records this honestly from the measured triplet.
"""
import os
import json
import sys
from pathlib import Path

# UTF-8 console: pooled_A10 cells contain Chinese; the summary print would crash
# on a Windows cp1252 console otherwise (the json file itself is utf-8-safe).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

SCHEMA_VERSION = "1.0"
DATASET_NAME = "turbgen-256-dns"

_ROOT = Path(os.environ.get("TURBGEN_DATA_DIR", "./data"))  # cross-platform parity
CORPUS = _ROOT / "corpus"
OUT = CORPUS / "dataset_manifest.json"

# ---- 15 frame-configs grouped by physics (CASE names, per the production spec) ----
# Ordering inside each group is deterministic (declaration order == sort order
# within group). Groups are emitted in the fixed order below.
# 2026-07-02 sync: DROPPED ou_robust_tau4_retune (abandoned 2026-06-26, never produced) +
# ADDED rotating_ro0p2_v2 (moderate-rotation 2nd point). Net still 15 (was 15 w/ dead
# tau4 -> now 15 w/ real v2). Must match make_splits.py::CONFIGS (15).
GROUPS = {
    "forced_isotropic": [
        "ou_relam90_256_fp64",
        "ou_relam70_256_fp64",
        "ou_relam50_256_fp64",
        "helical_re86_retune2_256_fp64",
        "ou_robust_tau1_256_fp64",
        "ou_robust_kf4_256_fp64",
        "ou_robust_kf3_256_fp64",
    ],
    "decay": [
        "decay_hotstart_re86",
        "decay_saffman_v2",
        "decay_batchelor_v2",
        "abc_turb_full_256_fp64",
    ],
    "extended": [
        "rotating_ro0p2_256_fp64",
        "rotating_ro0p2_v2_256_fp64",
        "scalar_sc1_256_fp64",
        "stratified_reb40_256_fp64",
    ],
}
GROUP_ORDER = ["forced_isotropic", "decay", "extended"]

# CASE -> physics_group lookup
_CASE_GROUP = {case: g for g in GROUP_ORDER for case in GROUPS[g]}


def _measured_re(manifest):
    """Pull measured Re_lambda_mean from a per-config manifest (or None)."""
    try:
        return manifest.get("regime", {}).get("measured", {}).get("Re_lambda_mean")
    except AttributeError:
        return None


def _measured_triplet(manifest):
    """(Re_lambda_mean, k_f, ou_tau) for the honest confounding table."""
    reg = manifest.get("regime", {}) or {}
    meas = reg.get("measured", {}) or {}
    inp = reg.get("input", {}) or {}
    return meas.get("Re_lambda_mean"), inp.get("k_f"), inp.get("ou_tau")


def main():
    CORPUS.mkdir(parents=True, exist_ok=True)

    found = []   # (group, case, manifest dict)
    missing = []
    for group in GROUP_ORDER:
        for case in GROUPS[group]:
            mp = CORPUS / case / "manifest.json"
            if not mp.exists():
                missing.append(case)
                continue
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
            except Exception as e:  # corrupt manifest -> treat as missing, report
                missing.append(f"{case} (unreadable: {e})")
                continue
            found.append((group, case, m))

    # Deterministic: group order (GROUP_ORDER) then alphabetical case within group.
    found.sort(key=lambda gc: (GROUP_ORDER.index(gc[0]), gc[1]))

    if missing:
        print("[dataset-manifest] MISSING per-config manifests (production "
              "incomplete; re-run as they land):")
        for c in missing:
            print(f"    - {c}")

    if not found:
        print("[dataset-manifest] 0 per-config manifests found under "
              f"{CORPUS}. Nothing to aggregate.")

    # ---- build configs array ----
    configs = []
    total_seeds = 0
    total_frames = 0
    caveats_union = []  # preserve first-seen order, dedup
    for group, case, m in found:
        n_seeds = m.get("n_seeds_accepted", 0) or 0
        n_frames = m.get("total_frames", 0) or 0
        total_seeds += n_seeds
        total_frames += n_frames

        # acceptance: probe the RIGHT appendix per physics group (they write different
        # filenames in different dirs): forced -> phase0/results/<case>_corpus/
        # DNS_STANDARD_APPENDIX_A.md; decay -> .../DECAY_APPENDIX.md; extended ->
        # phase_ext/results/<case>_corpus/DNS_STANDARD_APPENDIX_EXT.md. A single
        # phase0/.../APPENDIX_A probe (the old code) marked ALL decay+ext as "pending".
        _p0res = Path(__file__).parent / "results" / f"{case}_corpus"
        _peres = (Path(__file__).parent.parent / "phase_ext" / "results"
                  / f"{case}_corpus")
        if group == "decay":
            appx = _p0res / "DECAY_APPENDIX.md"
        elif group == "extended":
            appx = _peres / "DNS_STANDARD_APPENDIX_EXT.md"
        else:
            appx = _p0res / "DNS_STANDARD_APPENDIX_A.md"
        acceptance = ("A-group + D-group (D1-D4) passed" if appx.exists()
                      else "pending")

        # zarr + norm sidecars are SIBLINGS of the per-config dir (corpus/<CASE>.zarr,
        # corpus/<CASE>_norm_*.json), NOT inside corpus/<CASE>/ — match corpus_to_zarr.py.
        extra_ch = None
        for c in (m.get("channels") or []):
            if c in ("theta", "b"):
                extra_ch = c
        cfg_entry = {
            "case": case,
            "physics_type": group,
            "channels": m.get("channels"),
            "regime": m.get("regime"),  # input + measured copied through
            "n_seeds_accepted": n_seeds,
            "total_frames": n_frames,
            "pooled_A10": m.get("pooled_A10"),
            "frame_keep_frac_min": m.get("frame_keep_frac_min"),
            "acceptance": acceptance,
            "manifest_path": f"corpus/{case}/manifest.json",
            "zarr_path": f"corpus/{case}.zarr",
            "norm_minmax_path": f"corpus/{case}_norm_minmax.json",
            "norm_standardize_path": f"corpus/{case}_norm_standardize.json",
        }
        if extra_ch:
            cfg_entry["norm_extra_path"] = f"corpus/{case}_norm_{extra_ch}.json"
        configs.append(cfg_entry)

        for cav in (m.get("CAVEATS") or []):
            if cav not in caveats_union:
                caveats_union.append(cav)

    # ---- physics_groups summary ----
    physics_groups = {}
    for group in GROUP_ORDER:
        entries = [(case, m) for (g, case, m) in found if g == group]
        physics_groups[group] = {
            "configs": [
                {"case": case, "Re_lambda": _measured_re(m)}
                for (case, m) in entries  # already sorted (found is sorted)
            ],
            "n_configs": len(entries),
        }

    # ---- axes block (honest framing; NOT orthogonal) ----
    # Only the pure-nu Re axis {50,70,86} is single-variable. k_f and tau vary
    # the regime but are confounded with Re. Build a per-config (Re,k_f,tau)
    # table from MEASURED values so the confounding is visible, not asserted away.
    confounding_table = []
    for group, case, m in found:
        if group != "forced_isotropic":
            continue
        re_l, k_f, tau = _measured_triplet(m)
        confounding_table.append({
            "case": case, "Re_lambda_measured": re_l, "k_f": k_f, "ou_tau": tau,
        })

    axes = {
        "note": ("k_f and tau are CONFOUNDED with Re_lambda, not orthogonal "
                 "controlled axes. Only the pure-nu Re axis is single-variable; "
                 "k_f/tau change the physical forcing regime AND shift Re via "
                 "the injection (higher-k / longer-correlation -> higher eps -> "
                 "higher Re). The paper presents the Re axis as the single true "
                 "axis and k_f/tau as regime variations."),
        "true_single_variable_axis": {
            "name": "Re_lambda (pure-nu)",
            "targets": [50, 70, 86],
            "fixed": "k_f=2, ou_tau=2; only nu varies",
        },
        "regime_confounding_table": confounding_table,
    }

    # ---- top-level caveats: union of per-config + the timing note ----
    timing_note = (
        "Frame timing is NON-UNIFORM: frames are filtered (instantaneous "
        "k_maxeta>=1.5 gate; decay also K<5% tail-cut), so consecutive frames "
        "are NOT uniformly spaced in time. Loaders MUST use each frame's stored "
        "'t' and never assume a uniform dt.")
    caveats = list(caveats_union)
    if timing_note not in caveats:
        caveats.append(timing_note)

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": DATASET_NAME,
        "compute_precision": "fp64",
        "storage_precision": "fp32",
        "format": "zarr (lossless zstd)",
        "license_code": "MIT",
        "license_data": "CC-BY-4.0",
        "total_configs_found": len(found),
        "total_configs_expected": sum(len(v) for v in GROUPS.values()),
        "total_seeds": total_seeds,
        "total_frames": total_frames,
        "missing_configs": missing,
        "physics_groups": physics_groups,
        "axes": axes,
        "configs": configs,
        "note": timing_note,
        "caveats": caveats,
    }

    OUT.write_text(json.dumps(dataset, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[dataset-manifest] wrote {OUT}")
    print(f"[dataset-manifest] {len(found)}/{dataset['total_configs_expected']} "
          f"configs, {total_seeds} seeds, {total_frames} frames")

    # ---- summary table ----
    if configs:
        hdr = (f"{'config':<32} {'physics':<16} {'Re_lambda':>10} "
               f"{'seeds':>6} {'frames':>8} {'A10':>10}")
        print(hdr)
        print("-" * len(hdr))
        for c in configs:
            re_l = _measured_re(c)  # c['regime'] carries measured
            re_s = f"{re_l:.1f}" if isinstance(re_l, (int, float)) else "-"
            a10 = c.get("pooled_A10") or "-"
            print(f"{c['case']:<32} {c['physics_type']:<16} {re_s:>10} "
                  f"{c['n_seeds_accepted']:>6} {c['total_frames']:>8} {a10:>10}")
    else:
        print("[dataset-manifest] (no configs in table — 0 found)")


if __name__ == "__main__":
    main()
