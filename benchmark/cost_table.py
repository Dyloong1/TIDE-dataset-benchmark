"""cost_table — per-model compute-cost table for the paper (G6). Pure post-processing.

Sources, best-effort and honest about provenance:
- params: `_params_M` from production_configs_256.json (measured values, capacity-test-pinned)
- peak VRAM: `_peak_GB`/`_v2_note` annotations (Phase A measured) — quoted, not re-measured
- wall-clock per run: train.log first-line timestamp (suite writes `# <iso> <cmd>`) vs the file's
  mtime (last write ≈ training end). Runs without a parseable header are skipped and counted.

Usage: python cost_table.py --roots <run-roots...> [--out cost.md]
"""
import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_duration_h(run_dir: Path):
    log = run_dir / "train.log"
    if not log.exists():
        return None
    try:
        head = log.open(errors="ignore").readline()
        # suite header: "# 2026-07-16T17:55:29.594314+00:00" — tz-AWARE UTC. Compare against an
        # aware mtime or the local/UTC offset silently shifts every duration by hours.
        m = re.search(r"(\d{4}-\d{2}-\d{2}T[\d:.]+\+00:00)", head)
        if not m:
            return None
        t0 = datetime.fromisoformat(m.group(1))
        t1 = datetime.fromtimestamp(log.stat().st_mtime, tz=timezone.utc)
        h = (t1 - t0).total_seconds() / 3600
        return h if 0 < h < 200 else None
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models_json = json.load(open(HERE / "Model/configs/production_configs_256.json"))["models"]
    durs, skipped = {}, 0
    for root in args.roots:
        for d in Path(root).iterdir():
            if not d.is_dir() or "__" not in d.name:
                continue
            model = d.name.split("__")[0]
            if model.startswith(("ood_", "ablation_", "control_", "nsint_", "_")):
                continue
            # only finished runs: an in-flight run's train.log mtime is still advancing, so its
            # "duration" would grow every time this script runs.
            st = d / "status.json"
            try:
                if json.load(st.open()).get("status") != "done":
                    continue
            except (OSError, json.JSONDecodeError):
                continue
            h = run_duration_h(d)
            if h is None:
                skipped += 1
                continue
            durs.setdefault(model, []).append(h)

    lines = ["# Compute cost per baseline (measured)", "",
             "| model | params (M) | peak VRAM train (GB) | wall-clock / run (h, median [n]) |",
             "|---|---|---|---|"]
    for m in sorted(durs):
        cfg = models_json.get(m, {})
        params = cfg.get("_params_M", "—")
        peak = cfg.get("_peak_GB", "—")
        note = cfg.get("_v2_note", "")
        vram = f"{peak}" + (" (see _v2_note)" if note else "")
        med = statistics.median(durs[m])
        lines.append(f"| {m} | {params} | {vram} | {med:.2f} [{len(durs[m])}] |")
    lines += ["",
              f"_train.log-header-to-mtime durations; {skipped} run dirs without parseable header "
              f"skipped (counted, not silently dropped). VRAM figures are the Phase-A measured "
              f"annotations from production_configs_256.json, not re-measured here._"]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"[cost_table] wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
