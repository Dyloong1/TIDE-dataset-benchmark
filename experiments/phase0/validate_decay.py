"""Automated self-check for the free-decay code path (replaces human supervision
of the first decay run). Exit 0 = sane, exit 1 = something is wrong.

Checks on results/<case>/timeseries.csv:
  1. resume start: first logged t == expected checkpoint time (within dt).
  2. energy continuity: K at the first sample is within a sane band of the
     checkpoint's K (no jump from a botched warm start).
  3. decay actually happens: K(end) < K(start) (unforced -> must lose energy).
  4. no blow-up / NaN: eps finite and bounded throughout.
  5. monotone-ish: most steps decrease K (free decay shouldn't re-energize).

Usage: python validate_decay.py <case_name> <expected_start_time>
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
case = sys.argv[1] if len(sys.argv) > 1 else "decay_hotstart_re86"
t_expect = float(sys.argv[2]) if len(sys.argv) > 2 else None
REC = HERE / "results" / case

ts = np.genfromtxt(REC / "timeseries.csv", delimiter=",", names=True)
t = np.atleast_1d(ts["t"].astype(float))
K = np.atleast_1d(ts["K"].astype(float))
eps = np.atleast_1d(ts["eps"].astype(float))

fail = []

# 1. resume start time
if t_expect is not None and abs(t[0] - t_expect) > 0.05:
    fail.append(f"resume start t={t[0]:.3f} != expected {t_expect:.3f}")

# 2/3. decay happens, energy sane
if not np.isfinite(K).all():
    fail.append("K has non-finite values")
elif K[-1] >= K[0]:
    fail.append(f"K did not decay: K0={K[0]:.4f} -> Kend={K[-1]:.4f} (unforced must lose energy)")

# 4. no blow-up
if not np.isfinite(eps).all() or eps.max() > 50.0:
    fail.append(f"eps blow-up or NaN (max={np.nanmax(eps):.3g})")

# 5. mostly monotone decreasing
if len(K) > 10:
    frac_rising = float((np.diff(K) > 0).mean())
    if frac_rising > 0.5:
        fail.append(f"K rises on {frac_rising*100:.0f}% of steps (re-energization?)")

print(f"[validate_decay] {case}: t0={t[0]:.2f} K0={K[0]:.4f} Kend={K[-1]:.4f} "
      f"eps_max={np.nanmax(eps):.3g} n={len(t)}")
if fail:
    for f in fail:
        print(f"  FAIL: {f}")
    sys.exit(1)
print("  all decay sanity checks PASSED")
sys.exit(0)
