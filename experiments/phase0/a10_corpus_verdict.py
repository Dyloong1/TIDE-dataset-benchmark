"""Objective A10 isotropy verdict over the corpus seeds.

A10 cross is a HIGH-VARIANCE statistic (audit + CYB-q cross-check, ~17 independent
realizations): single-seed values scatter 0.7-6% and a single value is NOT a verdict.
The verdict uses the SAMPLE-WEIGHTED POOLED value over all accepted independent seeds
(the standard's ensemble-pooling clause) and reports the per-seed mean +/- std for the
paper's A10-methodology figure. >2% (cross) or >5% (comp) on the POOLED value -> add seeds.

    python a10_corpus_verdict.py <CASE>     # e.g. ou_relam90_256_fp64
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

HERE = Path(__file__).parent
RES = HERE / "results"
CASE = sys.argv[1] if len(sys.argv) > 1 else "ou_relam90_256_fp64"

# accepted corpus seeds: <CASE>_corpus (REC) + <CASE>_corpus_pool*
dirs = [RES / f"{CASE}_corpus"] + sorted(RES.glob(f"{CASE}_corpus_pool*"))
pool = []
for d in dirs:
    mp = d / "metrics.json"
    if not mp.exists():
        continue
    m = json.load(open(mp, encoding="utf-8"))
    if "ui2_mean" in m and m.get("eps_mean", 0) > 0.03:
        pool.append(m)

if not pool:
    print(f"[A10] no accepted corpus seeds for {CASE}")
    sys.exit(1)


def cross_of(p):
    return 100.0 * max(abs(x) for x in p["uij_mean"]) / (0.5 * sum(p["ui2_mean"]) * 2 / 3)


def comp_of(p):
    K2 = 0.5 * sum(p["ui2_mean"])
    return 100.0 * max(abs(3.0 * x / (2.0 * K2) - 1.0) for x in p["ui2_mean"])


# sample-weighted pooled (the verdict number)
w = np.array([p["iso_n_samples"] for p in pool], float)
uij = np.average([p["uij_mean"] for p in pool], axis=0, weights=w)
ui2 = np.average([p["ui2_mean"] for p in pool], axis=0, weights=w)
K2 = 0.5 * ui2.sum()
pooled_cross = 100.0 * max(abs(x) for x in uij) / (2.0 * K2 / 3.0)
pooled_comp = 100.0 * max(abs(3.0 * x / (2.0 * K2) - 1.0) for x in ui2)

per_cross = [cross_of(p) for p in pool]
per_comp = [comp_of(p) for p in pool]
total_TE = float(w.sum()) * 0.0  # samples, not T_E; report samples

cross_ok = pooled_cross <= 2.0
comp_ok = pooled_comp <= 5.0
verdict = "PASS" if (cross_ok and comp_ok) else "FAIL"

print(f"=== A10 corpus verdict: {CASE} ({len(pool)} independent seeds) ===")
print(f"  POOLED (verdict, sample-weighted over {int(w.sum())} samples):")
print(f"    A10 cross = {pooled_cross:.2f}%  (threshold <=2.0)  -> {'OK' if cross_ok else 'FAIL'}")
print(f"    A10 comp  = {pooled_comp:.2f}%  (threshold <=5.0)  -> {'OK' if comp_ok else 'FAIL'}")
print(f"  Per-seed distribution (A10 cross is high-variance -- report as mean+/-std):")
print(f"    cross: mean {np.mean(per_cross):.2f}% +/- {np.std(per_cross):.2f}%  "
      f"range [{min(per_cross):.2f}, {max(per_cross):.2f}]  ({sum(1 for x in per_cross if x>2)}/{len(per_cross)} seeds individually >2%)")
print(f"    comp:  mean {np.mean(per_comp):.2f}% +/- {np.std(per_comp):.2f}%  "
      f"range [{min(per_comp):.2f}, {max(per_comp):.2f}]")
print(f"  VERDICT: {verdict}")
if verdict == "FAIL":
    print(f"  ACTION: pooled cross {pooled_cross:.2f}% > 2% -- add independent seeds "
          f"(high-variance statistic, more seeds reduce the pooled fluctuation).")

# write a small json for the catalog/manifest
out = {
    "case": CASE, "n_seeds": len(pool), "n_samples": int(w.sum()),
    "pooled_A10_cross_pct": round(pooled_cross, 3),
    "pooled_A10_comp_pct": round(pooled_comp, 3),
    "per_seed_cross_mean_pct": round(float(np.mean(per_cross)), 3),
    "per_seed_cross_std_pct": round(float(np.std(per_cross)), 3),
    "per_seed_cross_range": [round(min(per_cross), 2), round(max(per_cross), 2)],
    "n_seeds_individually_over_2pct": int(sum(1 for x in per_cross if x > 2)),
    "verdict": verdict,
}
(RES / f"{CASE}_corpus" / "A10_verdict.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print(f"  wrote {RES / f'{CASE}_corpus' / 'A10_verdict.json'}")
sys.exit(0 if verdict == "PASS" else 2)
