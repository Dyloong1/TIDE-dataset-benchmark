"""make_benchmark_slice.py — pick the benchmark's train/val/test seeds per config by
OBJECTIVE PHYSICAL QUALITY (model-independent), and write benchmark_slice.json.

WHY (2026-06-26 user decision): the NO benchmark trains/evaluates on a small slice of the
released corpus (full-corpus training won't fit the deadline). We pick WHICH seeds to use by
a quality score computed from DNS-objective quantities only — k_maxη resolution margin,
stationarity (K-drift), and per-seed A10 isotropy — NEVER from any model's loss. This is
responsible curation ("use the cleanest data to stand up the baseline"), the opposite of
cherry-picking by model performance. The standard is fixed + published here; reviewers can
recompute it. Two iron rules: (1) selection is model-independent; (2) train/val/test seeds
are disjoint (no leakage). The FULL seed set still ships with the dataset.

Quality score (higher = cleaner), per seed:
    q = w_res * z(k_maxη_window_avg)          # resolution margin, ↑ better
      + w_drift * z(-|K_drift_last5TL_pct|)   # stationarity, ↑ better (less drift)
      + w_iso  * z(-per_seed_A10_cross)       # isotropy, ↑ better (smaller cross)
      + w_splice * z(-n_dt_splices)           # frame-time continuity, ↑ better (no resumed-run
                                              #   splice; a spliced trajectory is lower quality)
      + w_frames * z(frames_in_corpus)        # trajectory length, ↑ better (more eval windows)
    + hard prefilter: frames_in_corpus must == TARGET_FRAMES (drop short seeds)
z() = standardize across a config's accepted seeds. Weights are fixed (below) and logged.

Per config: 3 best-quality seeds -> train; next 1 -> val; next 3 -> test. Disjoint by
construction. 3 TEST seeds because eval samples at a full decorrelation time (~3.3 T_L measured),
which leaves only ~3 independent rollout windows per seed -- 1 seed would give 3 samples, which is
not an error bar. Configs in OOD_HOLDOUT (from make_splits) get no train split (whole-config OOD
test; see make_splits.py / benchmark §5).

THIS SCRIPT IS THE SOURCE OF TRUTH for benchmark_slice.json. Do NOT hand-edit that file: a
hand-edited slice cannot be reproduced by a reviewer, which is the whole point of publishing the
selection rule. Change the rule here and regenerate.

    python make_benchmark_slice.py [--train-seeds 3] [--frames 150]
    reads $TURBGEN_DATA_DIR/corpus/<CASE>/manifest.json (+ <CASE>_A10_frames.json if present)
    writes $TURBGEN_DATA_DIR/corpus/benchmark_slice.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# reuse the canonical config list + OOD holdout from make_splits (single source of truth)
from make_splits import CONFIGS, OOD_HOLDOUT  # noqa: E402  (same dir)

# fixed, published quality weights
W_RES, W_DRIFT, W_ISO = 1.0, 1.0, 1.0
SLICE_VAL_SEEDS = 1
# 3 test seeds (was 1) is what makes an INDEPENDENT test set affordable. The velocity field
# decorrelates in a measured ~3.3 T_L (2026-07-15 re-measure on a fine lag grid across 3 test
# seeds: 3.2/3.3/3.5; the older 3.2 here and 3.6 in the design doc were both coarse-grid
# artifacts -- lag could only land on discrete frames), so eval steps a full decorrelation time (frame_stride=64)
# and each seed then yields only ~3 independent rollout windows: 1 seed = 3 samples (useless),
# 3 seeds = 9 genuinely independent samples. Training keeps its dense stride -- overlapping windows
# are augmentation and training does not need independence, but an ERROR BAR does.
SLICE_TEST_SEEDS = 3

# A trajectory exported by a RESUMED run carries a splice in its frame times (measured: t jumps
# 42.31 -> 46.14 while the median dt is 0.10 = a 37x step). Rollout windows are fed to the model as
# equi-spaced single steps, so a window straddling that splice asks it to advance ~37 dt in one
# step. The dataset already drops those windows, but a spliced trajectory is genuinely LOWER
# QUALITY, so the split should prefer clean seeds for train/val rather than discovering the problem
# downstream. This is a physical, model-independent property -- the same basis as the other terms.
W_SPLICE = 1.0
# Trajectory length: more frames = more (and more independent) eval windows. Keeps FULL seeds
# ranked above shorter top-up seeds when the pool mixes both.
W_FRAMES = 1.0
SPLICE_FACTOR = 3.0     # a step > 3x the seed's own median dt is a splice, not jitter
                        # (clean seeds measure max/median = 1.00; real splices measure 27x-87x)


def _z(vals):
    """standardize a list -> z-scores (0 if degenerate)."""
    import statistics
    xs = [v for v in vals if v is not None]
    if len(xs) < 2:
        return [0.0 for _ in vals]
    mu = statistics.mean(xs)
    sd = statistics.pstdev(xs) or 1.0
    return [0.0 if v is None else (v - mu) / sd for v in vals]


def _per_seed_a10(corpus_dir, case):
    """Optional per-seed A10 cross from <CASE>_A10_frames.json (a10_from_zarr --seed-scan
    writes pooled, not per-seed; we read per_seed_cross_range/mean if present, else None)."""
    j = corpus_dir / f"{case}_A10_frames.json"
    if not j.exists():
        return {}
    try:
        d = json.loads(j.read_text(encoding="utf-8"))
        # a10_from_zarr currently logs aggregate per-seed stats, not per-seed-indexed values;
        # if a per-seed map is added later, consume it here. For now return {} (A10 term skipped).
        return d.get("per_seed_cross_by_seed", {}) or {}
    except Exception:
        return {}


def _per_seed_kmaxeta(corpus_dir, case):
    """{seed: mean k_maxη} straight from the zarr. {} if the store is unreadable.

    Used only when the manifest has no per-seed k_maxη -- true for every extended-physics config
    (rotating / scalar / stratified), whose producers leave that field null. Without this the whole
    quality score collapses to zero there and seed selection becomes "lowest seed number wins".
    The zarr's per-frame `k_max_eta` is the same quantity the manifest would have recorded.
    """
    try:
        import zarr as _zarr
        import numpy as _np
        z = _zarr.open(str(corpus_dir / f"{case}.zarr"), mode="r")
        ke = _np.asarray(z["k_max_eta"][:])
        sd = _np.asarray(z["seed"][:])
        out = {}
        for s in sorted({int(x) for x in sd.tolist()}):
            v = ke[sd == s]
            v = v[~_np.isnan(v)]
            if v.size:
                out[s] = float(v.mean())
        return out
    except Exception:
        return {}


def _per_seed_splices(corpus_dir, case):
    """{seed: n_splices} from the corpus zarr frame times. {} if the store is unreadable.

    A splice is a consecutive-frame step larger than SPLICE_FACTOR x that seed's OWN median dt, so
    the test adapts to each config's frame spacing instead of hardcoding a time. Clean seeds measure
    max/median = 1.00; resumed-run splices measure 27x-87x, so the factor is not delicate.
    """
    try:
        import numpy as np
        import zarr
        store = zarr.open(str(corpus_dir / f"{case}.zarr"), mode="r")
        seed = np.asarray(store["seed"][:])
        t = np.asarray(store["t"][:])
    except Exception:
        return {}
    out = {}
    for s in sorted(set(seed.tolist())):
        ts = np.sort(t[seed == s])
        if len(ts) < 3:
            continue
        d = np.diff(ts)
        med = float(np.median(d))
        out[int(s)] = int((d > SPLICE_FACTOR * med).sum()) if med > 0 else 0
    return out


def build_slice(data_root, train_seeds, target_frames):
    corpus = Path(data_root) / "corpus"
    per_config = {}
    for case in CONFIGS:
        man_path = corpus / case / "manifest.json"
        if not man_path.exists():
            per_config[case] = {"status": "no_manifest"}
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))  # manifests are utf-8 (CAVEATS contain η etc.); cp1252 default crashes on Windows
        seeds = man.get("per_seed", []) or []
        # Prefilter: prefer FULL-frame seeds (== target_frames, uniform), and TOP UP from the
        # shorter-but-usable ones when there are not enough full seeds to fill 3 train + 1 val +
        # 3 test. `full if full else usable` was an either/or, which starved scalar_sc1: it has 4
        # full seeds (150 frames) and 4 usable ones (102-143), so the full tier shut the rest out
        # and 4 seeds cannot fill a 7-seed split -> test came back EMPTY with no error.
        # Ordering still prefers full seeds (they sort first within the same quality), so a config
        # with enough full seeds is unaffected; only starved configs reach into the usable tier.
        # EVAL_MIN_FRAMES = t_in(1) + t_out_eval(20) = 21 — the AR-eval window need at the
        # finalized Markovian protocol (was 50 when t_in was 20 and AR ran 30 steps).
        EVAL_MIN_FRAMES = 21
        need = train_seeds + SLICE_VAL_SEEDS + SLICE_TEST_SEEDS
        full = [s for s in seeds if (s.get("frames_in_corpus") or 0) >= target_frames]
        usable = [s for s in seeds if (s.get("frames_in_corpus") or 0) >= EVAL_MIN_FRAMES]
        if len(full) >= need:
            pool = full
        elif usable:
            pool = usable          # includes the full ones; quality ordering sorts them to the top
        else:
            pool = seeds           # nothing clears the bar -> keep all, flagged by n_full_frame_seeds

        a10map = _per_seed_a10(corpus, case)
        splicemap = _per_seed_splices(corpus, case)
        kvals = [s.get("k_max_eta_window_avg") for s in pool]
        # ★ Fallback to the zarr when the manifest carries no per-seed k_maxη (2026-07-15).
        # The extended-physics producers (rotating / scalar / stratified, experiments/phase_ext)
        # write the manifest KEYS but leave Re_lambda / k_max_eta_window_avg / K_drift as null,
        # while the phase0 producers (kf3/kf4/tau1/relam*) fill them in. _z() maps an all-None
        # column to all-zeros, so on those configs EVERY quality term died and "pick the best 3
        # seeds" silently degenerated into "pick the 3 lowest seed numbers" -- while the emitted
        # `quality_formula` string still advertised all five terms as live. Found by CYB-3 on
        # relam70 (all 16 seeds scored 0.000000) and confirmed on q's own disk: rotating_ro0p2 and
        # decay_hotstart score all-zero; ro0p2_v2 and scalar_sc1 score all-TIED, which degenerates
        # the same way. That is 3 of q's 6 training configs selecting seeds by seed number, under a
        # published rule that claims otherwise.
        # The data was never missing -- the zarr stores per-frame k_max_eta (0% NaN), and the
        # per-seed means are genuinely distinct (rotating: 2.40 / 2.50 / 2.58). So recover it.
        if all(v is None for v in kvals):
            kmap = _per_seed_kmaxeta(corpus, case)
            if kmap:
                kvals = [kmap.get(int(s["seed"])) for s in pool]
        dvals = [-abs(s["K_drift_last5TL_pct"]) if s.get("K_drift_last5TL_pct") is not None else None for s in pool]
        ivals = [-(a10map.get(str(s["seed"]))) if str(s.get("seed")) in a10map else None for s in pool]
        # negated: FEWER splices = higher quality. None when the zarr was unreadable -> _z gives 0
        # (term inert), so this can never make the slice depend on whether the corpus was mounted.
        svals = [-(splicemap[int(s["seed"])]) if int(s["seed"]) in splicemap else None for s in pool]
        # Longer trajectory = higher quality: it yields more (and, at the eval stride, more
        # independent) windows. This is what keeps FULL seeds ranked above the shorter usable ones
        # once the pool mixes both, so a top-up never demotes a full seed.
        fvals = [s.get("frames_in_corpus") for s in pool]
        zk, zd, zi, zs, zf = _z(kvals), _z(dvals), _z(ivals), _z(svals), _z(fvals)
        # Report which terms actually DISCRIMINATE here. A term whose input is all-None (missing)
        # or all-identical (tied) contributes z==0 to every seed: it appears in the formula string
        # but has zero effect on the ranking. Publishing "seeds chosen by a 5-term quality score"
        # while all 5 are inert is exactly what CYB-3 caught on relam70 (16 seeds, all q=0.000000
        # -> the order was just ascending seed#). So state per-config which terms were live rather
        # than assert the formula unconditionally.
        _live = {n: any(abs(x) > 1e-12 for x in z)
                 for n, z in (("k_maxeta", zk), ("K_drift", zd), ("A10", zi),
                              ("splice", zs), ("frames", zf))}
        _live_names = sorted(k for k, v in _live.items() if v)
        # Report which terms actually DISCRIMINATE on this config. A term whose input is all-None
        # (missing) or all-identical (tied) contributes z==0 to every seed -- it is present in the
        # formula string but has no effect on the ranking. Publishing "chosen by a 5-term quality
        # score" while 5 of 5 are inert is the exact failure CYB-3 caught, so the slice must state
        # per-config which terms were live rather than assert the formula unconditionally.
        _live = {n: any(abs(x) > 1e-12 for x in z)
                 for n, z in (("k_maxeta", zk), ("K_drift", zd), ("A10", zi),
                              ("splice", zs), ("frames", zf))}
        _live_names = sorted(k for k, v in _live.items() if v)
        scored = []
        for i, s in enumerate(pool):
            q = W_RES * zk[i] + W_DRIFT * zd[i] + W_ISO * zi[i] + W_SPLICE * zs[i] + W_FRAMES * zf[i]
            scored.append((q, int(s["seed"]), s.get("frames_in_corpus")))
        scored.sort(key=lambda x: (-x[0], x[1]))  # high quality first, seed# tiebreak

        is_ood = case in OOD_HOLDOUT
        # 2026-07-17 owner decision: relam90 is the flagship config with a dual role, ood_test + in_dist.
        # Split = top-3 by quality for train / next 1 for val / ALL remaining for test (not the usual 3):
        # the test set shares its seed pool with the OOD target, so in-dist level vs zero-shot transfer is directly comparable; in-dist ckpts never enter OOD rows (contract preserved).
        DUAL_ROLE = {"ou_relam90_256_fp64"}
        if is_ood and case in DUAL_ROLE:
            chosen = [s for _, s, _ in scored]
            tr = chosen[:train_seeds]
            va = chosen[train_seeds:train_seeds + SLICE_VAL_SEEDS]
            te = chosen[train_seeds + SLICE_VAL_SEEDS:]      # all remaining, shared with the OOD evaluation
            per_config[case] = {
                "role": "ood_test+in_dist",
                "quality_formula": "w_res*z(k_maxη)+w_drift*z(-|K_drift|)+w_iso*z(-A10cross)+w_splice*z(-n_dt_splices)+w_frames*z(frames) [NOTE: A10 term currently INACTIVE — per-seed A10 source not wired (a10_from_zarr logs pooled, not per-seed), so k_maxη + K_drift + splice-count are live. All terms are model-independent physical properties of the trajectory.]",
                "quality_terms_live": _live_names,
                "quality_note": "terms absent from quality_terms_live contributed z=0 to EVERY seed ""(input missing or identical across seeds) and did not affect the ranking; ""if the list is empty the order is just ascending seed#",
                "train": {"seeds": tr}, "val": {"seeds": va}, "test": {"seeds": te},
                "ranked": [{"seed": s, "q": round(q, 4), "frames": f} for q, s, f in scored],
                "n_full_frame_seeds": len(full),
                "disjoint_ok": len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0,
                "_amendment_20260717": "Flagship in-dist column added (owner decision): the OOD test set shrinks from all seeds to the remaining seeds (changed before any G1 evaluation; no measured number invalidated); in-dist and OOD share test seeds; in-dist checkpoints never enter OOD rows.",
            }
            continue
        if is_ood:
            # whole-config OOD: all (full) seeds -> test, no train/val
            per_config[case] = {
                "role": "ood_test", "quality_formula": "w_res*z(k_maxη)+w_drift*z(-|K_drift|)+w_iso*z(-A10cross)+w_splice*z(-n_dt_splices)+w_frames*z(frames) [NOTE: A10 term currently INACTIVE — per-seed A10 source not wired (a10_from_zarr logs pooled, not per-seed), so k_maxη + K_drift + splice-count are live. All terms are model-independent physical properties of the trajectory.]",
                "quality_terms_live": _live_names,
                "quality_note": "terms absent from quality_terms_live contributed z=0 to EVERY seed ""(input missing or identical across seeds) and did not affect the ranking; ""if the list is empty the order is just ascending seed#",
                "test": {"seeds": [s for _, s, _ in scored]},
                "ranked": [{"seed": s, "q": round(q, 4), "frames": f} for q, s, f in scored],
                "n_full_frame_seeds": len(full),
            }
            continue

        chosen = [s for _, s, _ in scored]
        tr = chosen[:train_seeds]
        va = chosen[train_seeds:train_seeds + SLICE_VAL_SEEDS]
        te = chosen[train_seeds + SLICE_VAL_SEEDS:train_seeds + SLICE_VAL_SEEDS + SLICE_TEST_SEEDS]
        per_config[case] = {
            "role": "in_dist",
            "quality_formula": "w_res*z(k_maxη)+w_drift*z(-|K_drift|)+w_iso*z(-A10cross)+w_splice*z(-n_dt_splices)+w_frames*z(frames) [NOTE: A10 term currently INACTIVE — per-seed A10 source not wired (a10_from_zarr logs pooled, not per-seed), so k_maxη + K_drift + splice-count are live. All terms are model-independent physical properties of the trajectory.]",
            "quality_terms_live": _live_names,
            "quality_note": "terms absent from quality_terms_live contributed z=0 to EVERY seed ""(input missing or identical across seeds) and did not affect the ranking; ""if the list is empty the order is just ascending seed#",
            "train": {"seeds": tr}, "val": {"seeds": va}, "test": {"seeds": te},
            "ranked": [{"seed": s, "q": round(q, 4), "frames": f} for q, s, f in scored],
            "n_full_frame_seeds": len(full),
            "disjoint_ok": len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0,
        }
    return {
        "schema": "benchmark_slice/v1",
        "policy": "train/val/test seeds chosen by a MODEL-INDEPENDENT physical quality score "
                  "(resolution k_maxη, stationarity |K_drift|, isotropy A10, and dt-splice count); "
                  "train=best 3, val=next 1, test=next 3; disjoint; full seed set still shipped. "
                  "3 test seeds because eval samples at a full decorrelation time (~3.3 T_L), which "
                  "leaves ~3 independent windows per seed. This file is GENERATED — regenerate with "
                  "make_benchmark_slice.py rather than hand-editing it.",
        "weights": {"w_res": W_RES, "w_drift": W_DRIFT, "w_iso": W_ISO, "w_splice": W_SPLICE, "w_frames": W_FRAMES},
        "slice_train_seeds": train_seeds, "target_frames": target_frames,
        "ood_holdout": OOD_HOLDOUT,
        "per_config": per_config,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", type=int, default=3)   # protocol: 3 train / 1 val / 3 test
    ap.add_argument("--frames", type=int, default=150)  # 2026-07-08 dt-redesign: corpus is now 150
    # frames/traj (was 100). This is the hard prefilter "frames_in_corpus must == --frames"; run it
    # AFTER the dt=0.05 re-production. On the OLD 100-frame corpus, pass --frames 100 explicitly.
    ap.add_argument("--data-root", default=os.environ.get("TURBGEN_DATA_DIR", "./data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite even if that would BLANK configs the existing slice has scored "
                         "(see the regression guard below). Use only if you really do hold every "
                         "manifest and the existing file is the stale one.")
    args = ap.parse_args()
    out = build_slice(args.data_root, args.train_seeds, args.frames)
    op = Path(args.out) if args.out else Path(args.data_root) / "corpus" / "benchmark_slice.json"

    # ---- regression guard: a slice is only as complete as the manifests on THIS disk ----
    # build_slice walks a hardcoded 15-config list and writes {"status": "no_manifest"} for every
    # config whose manifest is not local (see the CONFIGS loop above), then main() overwrites the
    # WHOLE file. So "each machine regenerates its own slice" -- which the three-machine PROMPT
    # used to say -- is mutually destructive: q holds 7 manifests, CYB-t holds ~6, CYB-3 holds 1,
    # each run blanks the other two machines' configs, same filename, last writer wins. And since
    # hand-editing the JSON is (correctly) forbidden, there is no sanctioned way to rebuild what
    # was blanked. Manifests are ~128K vs ~237G per zarr, so the fix is cheap and obvious:
    #   ship manifests to ONE machine and generate the slice there, once.
    # This guard makes the destructive case fail loudly instead of silently.
    scored = {c for c, v in out["per_config"].items()
              if isinstance(v.get("train"), dict) and "seeds" in v["train"]}
    if op.exists() and not args.force:
        try:
            prev = json.loads(op.read_text(encoding="utf-8")).get("per_config", {})
        except Exception:
            prev = {}
        prev_scored = {c for c, v in prev.items()
                       if isinstance(v.get("train"), dict) and "seeds" in v["train"]}
        lost = sorted(prev_scored - scored)
        if lost:
            raise SystemExit(
                f"[slice] REFUSING to overwrite {op}.\n"
                f"  It currently scores {len(prev_scored)} configs; this run scores {len(scored)} "
                f"and would BLANK {len(lost)} of them to no_manifest:\n"
                f"    {', '.join(lost)}\n"
                f"  Those manifests are not on this disk, so this machine cannot regenerate their\n"
                f"  seed splits -- and hand-editing the slice is forbidden, so they would be gone.\n"
                f"  Manifests are ~128K (zarrs are ~237G): copy the missing corpus/<case>/\n"
                f"  manifest.json here, or generate the slice on the machine that has them all.\n"
                f"  Use --out <other.json> to write a partial slice elsewhere, or --force if you\n"
                f"  are certain the existing file is the stale one.")

    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    n_ok = sum(1 for c in out["per_config"].values() if c.get("disjoint_ok"))
    # "blank" = genuinely has no manifest on this disk. NOT the same as "has no train split": an
    # ood_test config has no train split BY DESIGN, and counting those cried wolf on a complete
    # 15-config slice ("2 config(s) ... PARTIAL" when the 2 were just the OOD holdouts).
    blank = [c for c, v in out["per_config"].items() if v.get("status") == "no_manifest"]
    print(f"[slice] wrote {op}  ({len(out['per_config'])} configs, {n_ok} in-dist disjoint-ok)")
    if blank:
        print(f"[slice] NOTE: {len(blank)} config(s) have no local manifest and are blank here: "
              f"{', '.join(blank)}. PARTIAL slice -- do not ship it as the merged one.")
    # A term that is inert everywhere means the published quality_formula is fiction for that
    # config; surface it here so a degenerate slice cannot be generated quietly (2026-07-15).
    degen = [c for c, v in out["per_config"].items()
             if v.get("status") != "no_manifest" and not v.get("quality_terms_live")]
    if degen:
        print(f"[slice] WARNING: {len(degen)} config(s) have NO live quality term -- their seed "
              f"order is just ascending seed#, not a quality ranking: {', '.join(degen)}. "
              f"Rebuild their manifest (rebuild_manifest_from_zarr.py recovers k_maxη from the "
              f"zarr) on the machine that holds the corpus.")


if __name__ == "__main__":
    main()
