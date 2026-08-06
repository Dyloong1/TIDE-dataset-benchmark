"""Pool extended-physics metrics over independent IC seeds. Generalizes
phase0/pool_a10_tau4.py to the three statistic families:
  (a) A10 cross/comp  — sample-weighted moment pool (exact eval_dns_standard formula),
      the high-variance statistic that needs pooling (REPORT for anisotropic flows).
  (b) R-group (rotating): frac_2D, b_ij_max_abs  — mean +/- std over seeds.
  (c) S-group (stratified): buoyancy_flux_wb, PE_over_KE, b_ij_max_abs — mean +/- std.
The verdict is the POOLED value; per-seed mean+/-std is the error bar.

    python pool_anisotropic.py <base_case> [extra metrics.json paths ...]
e.g. python pool_anisotropic.py rotating_ro0p2_256_fp64 \
        results/rotating_ro0p2_256_fp64_seed1/metrics.json results/..._seed2/... results/..._seed3/...
(base case's own results/<base>/metrics.json is the seed0 member; extra paths add seeds.)
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: pool_anisotropic.py <base_case> [extra metrics.json ...]")
    base = sys.argv[1]
    cands = [RES / base / "metrics.json"] + [Path(p) for p in sys.argv[2:]]
    pool, labels = [], []
    for mp in cands:
        if mp.exists():
            m = json.load(open(mp, encoding="utf-8"))
            # exclude aborted / laminarized runs from the pool
            if m.get("a14_aborted") or m.get("aborted_before_sampling"):
                print(f"  skip {mp.parent.name}: aborted"); continue
            if "ui2_mean" not in m:
                print(f"  skip {mp.parent.name}: no ui2_mean"); continue
            pool.append(m); labels.append(mp.parent.name)
    if not pool:
        raise SystemExit("no usable metrics with ui2_mean")

    phys = pool[0].get("physics", "?")
    # --- (a) A10 sample-weighted pool (exact eval formula) ---
    w = np.array([m["iso_n_samples"] for m in pool], float)
    uij = np.average([m["uij_mean"] for m in pool], axis=0, weights=w)
    ui2 = np.average([m["ui2_mean"] for m in pool], axis=0, weights=w)
    K2 = 0.5 * ui2.sum()
    pooled_cross = 100.0 * max(abs(x) for x in uij) / (2.0 * K2 / 3.0)
    pooled_comp = 100.0 * max(abs(3.0 * x / (2.0 * K2) - 1.0) for x in ui2)
    per_cross = [100.0 * max(abs(x) for x in m["uij_mean"]) / (0.5 * sum(m["ui2_mean"]) * 2 / 3)
                 for m in pool]

    out = {"case": base, "physics": phys, "n_seeds": len(pool), "seeds": labels,
           "pooled_A10_cross_pct": round(pooled_cross, 3),
           "pooled_A10_comp_pct": round(pooled_comp, 3),
           "per_seed_A10_cross_pct": [round(x, 3) for x in per_cross],
           "total_iso_samples": int(w.sum())}

    print(f"=== {base} ({phys}) pooling over {len(pool)} seed(s): {labels} ===")
    print(f"  A10 cross: per-seed {[round(x,2) for x in per_cross]} "
          f"(mean {np.mean(per_cross):.2f} +/- {np.std(per_cross):.2f}%) -> POOLED {pooled_cross:.2f}% "
          f"(report; isotropic-gate <=2)")
    print(f"  A10 comp : POOLED {pooled_comp:.2f}%")

    def ms(key, label, fmt="%.4f"):
        vals = [m[key] for m in pool if key in m]
        if not vals:
            return
        mean, std = float(np.mean(vals)), float(np.std(vals))
        out[f"{key}_mean"] = mean; out[f"{key}_std"] = std
        out[f"{key}_per_seed"] = [round(float(v), 5) for v in vals]
        print(f"  {label}: " + " ".join(fmt % v for v in vals)
              + f"  -> mean {fmt % mean} +/- {fmt % std}")

    if phys == "rotating":
        ms("frac_2D", "R2 frac_2D (2D energy)")
        ms("b_ij_max_abs", "R3 anisotropy b_max")
        ms("rel_helicity", "R4 rel_helicity")
        ms("Ro", "R1 Rossby Ro")
    elif phys == "stratified":
        ms("buoyancy_flux_wb", "S4 buoyancy flux <wb>", "%.4e")
        ms("PE_over_KE", "S3 PE/KE")
        ms("Re_b", "S2 Re_b", "%.1f")
        ms("b_ij_max_abs", "S anisotropy b_max")
    elif phys == "passive_scalar":
        ms("scalar_variance", "scalar <theta^2>")
        ms("scalar_dissipation_chi", "chi", "%.4e")

    (RES / f"{base}_pool.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  wrote {RES / f'{base}_pool.json'}")
    if len(pool) < 4:
        print(f"  NOTE: only {len(pool)} seed(s); append the other seeds' metrics.json for a 4-seed pool.")


if __name__ == "__main__":
    main()
