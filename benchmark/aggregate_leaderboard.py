"""Leaderboard aggregation: per-case skill scores + the three-axis table.

Pure POST-PROCESSING over the row jsons that eval_benchmark writes — reads files, never
touches protocol, models, or checkpoints, so it is safe to run/iterate mid-matrix.

Why skill instead of raw nRMSE averaging: persistence is measured per-case (kf3 0.8407 vs
relam70 0.8217, 2026-07-16), so averaging raw nRMSE across cases weights the leaderboard by
case difficulty instead of model quality. skill = 1 - nRMSE_model / nRMSE_persistence(same
case) is 0 at "no better than copying the input", positive when the model actually predicts,
and comparable across cases. Cross-case aggregation averages skill, never raw nRMSE.

Why three axes: measured 2026-07-16 that the 20-step-mean nRMSE is blind to physical
collapse (two fno3d ckpts 0.7838 vs 0.7847 while enstrophy_ratio differed 3.63 vs 1.21) —
so the table always co-reports nRMSE_mean, enstrophy_ratio_mean and fRMSE_high_mean.

Usage:
  python aggregate_leaderboard.py --roots ../checkpoints/bench_q [more roots ...] \
         [--out leaderboard.md] [--baseline persistence]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(roots):
    """Official rows only, with an IDENTITY CHECK (2026-07-17): a row is accepted only when its
    location matches its content — subdir layout requires dir name == model__task__case, flat
    layout (file directly under the root, as results_rows/<machine>/ ships) requires file stem ==
    model__task__case. Why: eval writes identically-keyed row JSONs into ablation_*/ood_*/control_*
    dirs; the old keys-only load let those silently OVERWRITE official rows (last-glob-wins). The
    dir/stem convention is what separates official rows from ablation artifacts."""
    rows = {}
    for root in roots:
        rp = Path(root)
        if not rp.is_dir():
            print(f"[aggregate] WARN root missing, skipped: {root}")
            continue
        for f in list(rp.glob("*/*.json")) + list(rp.glob("*.json")):
            if f.name == "status.json":
                continue
            try:
                r = json.loads(f.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(r, dict) or "nRMSE_mean" not in r:
                continue
            key = (r.get("model"), r.get("task"), r.get("case"))
            if None in key:
                continue
            ident = "__".join(key)
            ok = (f.parent.name == ident) if f.parent != rp else (f.stem == ident)
            if not ok:
                continue   # ablation/OOD/control artifact or mislabeled file — not an official row
            # last root wins on duplicates — pass roots oldest-first
            rows[key] = r
    return rows


def load_seed_std(seed_roots):
    """nRMSE_mean per key from SEED-repeat roots (bench_*_seed1 etc.), for the +-std display.
    Kept OUT of the main rows: aggregate has no notion of averaging rows, so mounting seed
    roots as --roots silently replaced seed-0 rows (found+fixed 2026-07-17)."""
    vals = {}
    for root in seed_roots or []:
        for key, r in load_rows([root]).items():
            vals.setdefault(key, []).append(r["nRMSE_mean"])
    return vals


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    ap.add_argument("--baseline", default="persistence",
                    help="skill reference model (must have a row per case; default persistence)")
    ap.add_argument("--seed-roots", nargs="*", default=None,
                    help="seed-repeat roots (bench_q_seed1 ...) used ONLY for +-std, never pooled")
    args = ap.parse_args()

    rows = load_rows(args.roots)
    seed_vals = load_seed_std(args.seed_roots)
    tasks = sorted({t for (_, t, _) in rows})
    lines = ["# Leaderboard (aggregated)", ""]
    missing_baseline = []

    for task in tasks:
        cases = sorted({c for (_, t, c) in rows if t == task})
        models = sorted({m for (m, t, _) in rows if t == task and m != args.baseline})
        lines += [f"## {task}", ""]

        # --- skill table (cross-case comparable) ---
        lines += [f"### skill = 1 - nRMSE/nRMSE_{args.baseline} (same case) — higher is better, 0 = persistence level", ""]
        header = "| model | " + " | ".join(cases) + " | **mean skill** |"
        lines += [header, "|" + "---|" * (len(cases) + 2)]
        for m in models:
            cells, skills = [], []
            for c in cases:
                r = rows.get((m, task, c))
                b = rows.get((args.baseline, task, c))
                if b is None:
                    missing_baseline.append((task, c))
                if r is None or b is None or not b.get("nRMSE_mean"):
                    cells.append("—")
                else:
                    s = 1.0 - r["nRMSE_mean"] / b["nRMSE_mean"]
                    skills.append(s)
                    cells.append(f"{s:+.3f}")
            mean_s = f"**{sum(skills)/len(skills):+.3f}**" if skills else "—"
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {mean_s} |")
        lines.append("")

        # --- three-axis table per case ---
        lines += ["### three-axis detail: nRMSE_mean / enstrophy_ratio / fRMSE_high (physical stability next to the mean)", ""]
        lines += ["| model | case | nRMSE_mean | ens_ratio | fRMSE_high |",
                  "|---|---|---|---|---|"]
        for m in models + [args.baseline]:
            for c in cases:
                r = rows.get((m, task, c))
                if r is None:
                    continue
                sv = seed_vals.get((m, task, c))
                nr = fmt(r.get('nRMSE_mean'))
                if sv and r.get('nRMSE_mean') is not None:
                    import statistics
                    pool = sv + [r['nRMSE_mean']]
                    nr += f" +-{statistics.pstdev(pool):.4f}(n={len(pool)})"
                lines.append(
                    f"| {m} | {c} | {nr} | "
                    f"{fmt(r.get('enstrophy_ratio_mean'), 2)} | {fmt(r.get('fRMSE_high_mean'), 2)} |")
        lines.append("")

    if missing_baseline:
        uniq = sorted(set(missing_baseline))
        lines += ["", f"WARNING missing {args.baseline} baseline rows (skill left blank): " +
                  ", ".join(f"{t}/{c}" for t, c in uniq)]

    text = "\n".join(lines) + "\n"
    if args.out:
        # utf-8 explicit: the leaderboard uses U+2212 MINUS SIGN and other non-ASCII;
        # Windows default cp936/gbk raised UnicodeEncodeError (CYB-3 hit this 2026-07-18,
        # cross-platform bug q's Linux never surfaced). Newline pinned so LF stays LF.
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print(f"[aggregate] wrote {args.out} ({len(rows)} rows, {len(tasks)} tasks)")
    else:
        print(text)


if __name__ == "__main__":
    main()
