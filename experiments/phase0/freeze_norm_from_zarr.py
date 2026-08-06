"""Re-freeze normalization constants from an already-written corpus.zarr using the
vetted NormalizerFitter (JOINT velocity). It does NOT touch the raw stored fields -- only
recomputes the frozen constants + overwrites the sidecar JSONs and zarr attrs.

    python freeze_norm_from_zarr.py --case <CASE> --seed-filter train --slice <benchmark_slice.json>
    python freeze_norm_from_zarr.py --case <CASE> --seed-filter all      # OOD / no train split

★ WHY --seed-filter EXISTS (the transductive leak, 2026-07-15)
--------------------------------------------------------------
The fit loop used to walk EVERY frame in the store -- including the val and test seeds. It even
read the per-frame `seed` array and then never used it to gate anything. So the published
normalization constants had seen the test set: a real (if mild) transductive leak. The sidecar's
own note said "<CASE> corpus, N frames", never "train split", which is the tell.

It was previously judged not worth fixing because correcting it means retraining every model. That
premise died when the benchmark was reset to a from-scratch rebuild: there are no results to
protect, and a train-only refit reads ~37% of the frames, so it is CHEAPER than the status quo
(~3 min/case vs ~8 min/case at a measured 0.40 s/frame).

Policy (decided 2026-07-15):
  in_dist cases  -> --seed-filter train : fit ONLY on the slice's train seeds. No leak.
  ood_test cases -> --seed-filter all   : no train split exists, and NO model ever trains on them,
                                          so fitting on their own frames leaks nothing into training.
  role=None / no manifest -> REFUSED. We do not guess (see _resolve_seeds).

★ Expect held-out frames to slightly exceed [-1,1] after this change. That is CORRECT, not a
regression: minmax divides by the TRAIN absmax, and a test frame may carry a larger extremum.
solver/normalizer.py:104-110 predicted exactly this. The note in the sidecar records the filter so
a future reader can tell which frames produced the constants.

★ Data locality: this must run on the machine that HOLDS the zarr (it streams every frame). q
cannot refit t's or CYB-3's cases, and vice versa.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import solver._env  # noqa: F401
import torch
import zarr

from solver.normalizer import NormalizerFitter, _GroupStats as _GS


def _resolve_seeds(case, slice_path, seed_filter):
    """Which seeds to fit on -> (sorted seed list or None for 'all', human-readable reason).

    REFUSES rather than guessing. The two OOD cases are NOT symmetric -- ou_relam90 carries
    role='ood_test' while stratified_reb40 is role=None/{"status":"no_manifest"} and only appears
    in the top-level ood_holdout list -- so `role` alone is not a sufficient discriminator, and a
    naive per_config[case]["train"]["seeds"] raises TypeError on both (train is None).
    """
    if seed_filter == "all":
        return None, "all frames (no seed filter)"
    if not slice_path:
        sys.exit("REFUSE: --seed-filter train needs --slice <benchmark_slice.json> to know which "
                 "seeds are train. Pass it, or use --seed-filter all if this case has no train "
                 "split (OOD).")
    sl = json.loads(Path(slice_path).read_text(encoding="utf-8"))
    per = sl.get("per_config", {}).get(case)
    if per is None:
        sys.exit(f"REFUSE: {case} is not in {slice_path}. Cannot infer its train seeds.")
    ood = set(sl.get("ood_holdout", []))
    role = per.get("role")
    train = per.get("train")
    if not isinstance(train, dict) or "seeds" not in train:
        # No train split. Only legitimate for a held-out case -- and then the caller must SAY so,
        # because "fit on everything" is the leaky default we are here to remove.
        if case in ood or role == "ood_test":
            sys.exit(f"REFUSE: {case} is OOD-held-out (role={role!r}, in ood_holdout={case in ood}) "
                     f"and has no train split, so --seed-filter train is meaningless. Re-run with "
                     f"--seed-filter all -- no model trains on this case, so fitting on its own "
                     f"frames leaks nothing into training.")
        sys.exit(f"REFUSE: {case} has no usable train split (role={role!r}, train={train!r}) and is "
                 f"not in ood_holdout. This is the {{'status': 'no_manifest'}} case -- its manifest "
                 f"is not on this disk, so this slice cannot say which seeds are train. Do NOT "
                 f"guess: get the manifest, regenerate the slice, or decide the policy explicitly.")
    seeds = sorted(int(s) for s in train["seeds"])
    return seeds, f"train seeds {seeds} (role={role!r})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="corpus case name (without .zarr)")
    ap.add_argument("--seed-filter", choices=("train", "all"), required=True,
                    help="'train' = fit ONLY on the slice's train seeds (in_dist; removes the "
                         "transductive leak). 'all' = every frame (ONLY for ood_test cases that "
                         "have no train split and that no model ever trains on).")
    ap.add_argument("--slice", default=None, help="benchmark_slice.json (required for --seed-filter train)")
    ap.add_argument("--data-root", default=os.environ.get("TURBGEN_DATA_DIR", "./data"),
                    help="corpus root (the dir CONTAINING corpus/). Default $TURBGEN_DATA_DIR.")
    ap.add_argument("--backup-dir", default=None,
                    help="where to copy the existing sidecars before overwriting "
                         "(default: <corpus>/_norm_backup/<UTC timestamp>/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be fit (case, seeds, frame count) and exit without "
                         "reading frames or writing anything")
    args = ap.parse_args()

    CASE = args.case
    ZARR = Path(args.data_root) / "corpus" / f"{CASE}.zarr"
    if not ZARR.is_dir():
        sys.exit(f"REFUSE: {ZARR} not found. Is --data-root right? (this must run on the machine "
                 f"that HOLDS the zarr -- refitting streams every frame)")

    seeds, why = _resolve_seeds(CASE, args.slice, args.seed_filter)

    store = zarr.open_group(str(ZARR), mode="r" if args.dry_run else "r+")
    zu, zp = store["u"], store["p"]
    N = zu.shape[0]
    EXTRA = next((k for k in ("theta", "b") if k in store), None)

    # Which frame indices participate. The per-frame `seed` array is what makes the filter
    # possible; it was always there, just never used to gate the fit.
    if seeds is None:
        idx = list(range(N))
    else:
        if "seed" not in store:
            sys.exit(f"REFUSE: {CASE}.zarr has no per-frame 'seed' array, so a train-only fit "
                     f"cannot be verified. Use --seed-filter all only if this case is OOD.")
        seed_arr = store["seed"][:]
        want = set(seeds)
        idx = [i for i in range(N) if int(seed_arr[i]) in want]
        missing = want - {int(s) for s in seed_arr}
        if missing:
            sys.exit(f"REFUSE: train seeds {sorted(missing)} are not present in {CASE}.zarr "
                     f"(it has {sorted({int(s) for s in seed_arr})}). The slice and the store "
                     f"disagree -- fix that before freezing constants from it.")
        if not idx:
            sys.exit(f"REFUSE: no frames matched train seeds {seeds}.")

    print(f"{CASE}.zarr: {N} frames total -> fitting on {len(idx)} ({len(idx)/N:.0%}) | {why}"
          + (f" | 5th channel '{EXTRA}'" if EXTRA else ""))
    if args.dry_run:
        print("[dry-run] no frames read, nothing written.")
        return

    # guard: never freeze constants from a partially-written store (unwritten frames are
    # all-zero and would silently corrupt min/mean/M2).
    if not store.attrs.get("complete", False):
        last = torch.from_numpy(zp[N - 1])
        if float(last.abs().max()) == 0.0:
            sys.exit(f"REFUSE: {CASE}.zarr has no 'complete' flag and its last frame is "
                     f"all-zero -> store is partial. Re-run corpus_to_zarr.py first.")
        print("  WARNING: no 'complete' flag (pre-flag store); last frame is non-zero, proceeding")

    # ---- back up the existing sidecars BEFORE touching anything ----
    # This script opens the store r+ and overwrites the sidecars in place; without a copy there is
    # no way back to the previous constants if the refit turns out wrong.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bdir = Path(args.backup_dir) if args.backup_dir else ZARR.parent / "_norm_backup" / stamp
    bdir.mkdir(parents=True, exist_ok=True)
    for old in ZARR.parent.glob(f"{CASE}_norm_*.json"):
        shutil.copy2(old, bdir / old.name)
    print(f"  backed up existing sidecars -> {bdir}")

    fit = NormalizerFitter()
    extra_stat = _GS() if EXTRA else None
    z_extra = store[EXTRA] if EXTRA else None
    for n, i in enumerate(idx):
        fit.update(torch.from_numpy(zu[i]), torch.from_numpy(zp[i]))
        if EXTRA is not None:
            extra_stat.update(torch.from_numpy(z_extra[i]))
        if (n + 1) % 200 == 0 or n == len(idx) - 1:
            print(f"  {n+1}/{len(idx)} frames scanned", flush=True)

    # The note is the audit trail: it must say WHICH frames produced these constants. The old text
    # ("<CASE> corpus, N frames (re-frozen)") would be an outright lie after a train-only fit.
    seed_txt = "all seeds" if seeds is None else f"seeds {seeds}"
    note = (f"{CASE}: fit on {len(idx)}/{N} frames from {seed_txt} "
            f"(--seed-filter {args.seed_filter}; {stamp})")
    nm = fit.freeze("minmax", note=note)
    ns = fit.freeze("standardize", note=note)

    # write via temp + atomic rename: a loader that catches a missing sidecar mid-write now RAISES
    # (turbgen_zarr_dataset._load_frozen_normalizer), so never leave the file absent.
    def _atomic(frozen, path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        frozen.to_json(tmp)
        os.replace(tmp, path)

    side_mm = ZARR.parent / f"{CASE}_norm_minmax.json"
    side_std = ZARR.parent / f"{CASE}_norm_standardize.json"
    _atomic(nm, side_mm)
    _atomic(ns, side_std)

    if EXTRA is not None:
        # 5th-channel stats live in the ZARR ATTRS (not just a sidecar). A refit that only rewrote
        # sidecars would leave theta/b on the OLD leaked constants -- the loader now raises on a
        # missing attr, but a STALE attr is silent, so it must be rewritten here.
        extra_norm = {"channel": EXTRA, "vmin": extra_stat.vmin, "vmax": extra_stat.vmax,
                      "mean": extra_stat.mean, "std": extra_stat.std, "note": note}
        store.attrs[f"norm_{EXTRA}"] = extra_norm
        side_extra = ZARR.parent / f"{CASE}_norm_{EXTRA}.json"
        tmp = side_extra.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(extra_norm, indent=2), encoding="utf-8")
        os.replace(tmp, side_extra)
        print(f"  5th channel '{EXTRA}' norm -> attrs + {side_extra}")

    store.attrs["channels"] = ["velocity (u,v,w joint)", "p"] + ([EXTRA] if EXTRA else [])
    store.attrs["normalizer"] = "solver.normalizer.FrozenNormalizer (velocity JOINT, vetted q2)"
    store.attrs["norm_fit_scope"] = note      # machine-readable provenance of the constants
    if "norm_constants" in store.attrs:
        del store.attrs["norm_constants"]   # remove the old per-component block
    store.attrs["CAVEATS"] = [
        "Stored values are RAW physical fp32 fields (no normalization applied). Load "
        f"solver.normalizer.FrozenNormalizer.from_json({CASE}_norm_minmax.json or "
        f"{CASE}_norm_standardize.json) to transform at load time; velocity (u,v,w) is "
        "normalized JOINTLY (one shared scale) to preserve A10 component isotropy. NOTE: "
        "the 'standardize' mode is UNBOUNDED (z-score, ~+/-5, pressure can reach ~-11); "
        "only 'minmax' maps to [-1,1].",
        f"Normalization constants were fit on {seed_txt} (see attrs['norm_fit_scope']). For "
        "train-only fits, held-out frames MAY slightly exceed [-1,1] under 'minmax' -- that is "
        "correct (the scale is the TRAIN absmax), not a bug.",
        "Frames are filtered by instantaneous k_maxeta>=1.5 and are NOT uniformly spaced "
        "in time -- use the stored 't' array (monotonic WITHIN each seed, resets at seed "
        "boundaries), do not assume uniform dt.",
        "The 'seed' array stores the TRUE IC pool number (_corpus->0, _corpus_poolK->K), so a "
        "loader can filter the zarr by (case, seed) exactly as the by-IC split manifest emits; "
        "rejected seeds are simply absent (no compaction/relabeling).",
        "The corpus under-samples the strong-dissipation (eps-peak) tail; high-order "
        "intermittency statistics are a lower bound (see DATASHEET.md).",
    ]
    store.attrs["complete"] = True

    print(f"DONE: {side_mm}")
    print(f"      {side_std}")
    print(f"  note: {note}")


if __name__ == "__main__":
    main()
