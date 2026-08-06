"""Judge the dynamics-consistency group (D1/D2/D3, proposed v1.1 additions)
from the showcase run's dynamics_residuals.json and append the D-group rows
to DNS_STANDARD_APPENDIX_A.md (append-only; the user's standard file itself
is not modified).

Proposed thresholds (frozen here):
  D1 divergence residual <(div u)^2>/<|grad u|^2> <= 1e-6   (same as A12)
  D2 NS momentum residual ||r||/||du/dt||         <= 1e-2   (time-truncation level)
  D3 half-dt probe: residual ratio in [2.5, 6]  (~4 expected for O(h^2)
     central-difference dominance) -> proves D2 is discretization truncation,
     not a model/equation error.
"""
import json
import sys
from pathlib import Path

# The D-group table contains Chinese text; Windows' default cp1252 stdout
# raises UnicodeEncodeError on print(). Force UTF-8 so the run completes (the
# .md file is appended as UTF-8 regardless; this only fixes the console echo).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).parent
RES = HERE / "results"
# Case isolation (same convention as eval_dns_standard.py): argv[1] = case name.
# D-group source dir is results/trajectory_showcase_<case>/ (falls back to the
# legacy results/trajectory_showcase/ if the per-case dir is absent); the table
# is appended to that case's own DNS_STANDARD_APPENDIX_A.md. This stops two
# routes (e.g. OU and k_f=4) from overwriting each other's D-group json/table.
_case = sys.argv[1] if len(sys.argv) > 1 else "hit_relam100_256_fp64"
_show_per_case = RES / f"trajectory_showcase_{_case}"
SHOW = _show_per_case if _show_per_case.exists() else RES / "trajectory_showcase"
SRC = SHOW / "dynamics_residuals.json"
_case_dir = RES / _case
TABLE = (_case_dir / "DNS_STANDARD_APPENDIX_A.md") if _case_dir.exists() \
    else RES / "DNS_STANDARD_APPENDIX_A.md"

d = json.load(open(SRC, encoding="utf-8"))
samples, probes = d["samples"], d["halfdt_probes"]

import numpy as np

d1 = [s["D1_div_ratio"] for s in samples]
d2 = [s["res_rel_dudt"] for s in samples]
d2_visc = [s["res_rel_visc"] for s in samples]
ratios = [p["ratio_rel_dudt"] for p in probes]
d4_pois = [s["D4_poisson_res"] for s in samples if "D4_poisson_res" in s]
d4_grad = [s["D4_gradp_res"] for s in samples if "D4_gradp_res" in s]

d1_max = max(d1)
d2_mean, d2_max = float(np.mean(d2)), max(d2)
ok1 = d1_max <= 1e-6
ok2 = d2_max <= 1e-2
ok3 = all(2.5 <= r <= 6.0 for r in ratios)
has_d4 = bool(d4_pois)
d4_pois_max = max(d4_pois) if d4_pois else float("nan")
d4_grad_max = max(d4_grad) if d4_grad else float("nan")
ok4 = has_d4 and d4_pois_max <= 1e-8 and d4_grad_max <= 1e-8

def pf(ok):
    return "pass" if ok else "**FAIL**"

rows = [
    "",
    "## D-group: dynamical consistency (proposed v1.1 addition; filled at the proposed thresholds)",
    "",
    f"Protocol: {d['protocol']}; samples = step-level frame triplets at {len(samples)} instants of the showcase run"
    f" (spatial terms are spectrally exact; the residual contains only the time-discretization truncation).",
    "",
    "| Item | Measured | Proposed threshold | Verdict |",
    "|---|---|---|---|",
    f"| D1 divergence residual (per-sample max) | {d1_max:.2e} | <=1e-6 (same as A12) | {pf(ok1)} |",
    f"| D2 NS momentum residual ||r||/||du/dt|| | mean {d2_mean:.2e}, max {d2_max:.2e}"
    f" (||r||/||nu*lap(u)|| mean {float(np.mean(d2_visc)):.2e}) | <=1e-2 | {pf(ok2)} |",
    f"| D3 step-halving convergence probe | residual ratios {['%.2f' % r for r in ratios]} (expected ~4, O(h^2) dominated)"
    f" | in [2.5, 6] | {pf(ok3)} |",
]
if has_d4:
    rows.append(
        f"| D4 velocity--pressure consistency | Poisson residual max {d4_pois_max:.1e};"
        f" grad(p) balances the irrotational advection part, max {d4_grad_max:.1e} | <=1e-8 | {pf(ok4)} |")
else:
    rows.append("| D4 velocity--pressure consistency | (samples carry no D4 fields; rerun the showcase) | <=1e-8 | MISSING |")
rows.append("")
rows.append("D4 simultaneously validates the solver pressure solve (operators.pressure_hat), "
            "which is also the generator of the released pressure channel.")
rows.append("")

# Idempotent append (review P2-3): strip any existing D-group section first so
# re-running D-group (or re-running A-group which rewrites the file, then D) never
# stacks duplicate D tables. The A-group section (before the D-group header) is kept.
_marker = "## D-group"
_legacy_marker = "## D \u7ec4"   # header written by earlier releases; kept for idempotent re-runs
if TABLE.exists():
    existing = TABLE.read_text(encoding="utf-8")
    for _m in (_marker, _legacy_marker):
        if _m in existing:
            existing = existing[:existing.index(_m)].rstrip() + "\n"
            TABLE.write_text(existing, encoding="utf-8")
            break
with open(TABLE, "a", encoding="utf-8") as fh:
    fh.write("\n" + "\n".join(rows))
print("\n".join(rows))
print(f"appended D-group to {TABLE} (idempotent)")
# D-group is non-negotiable (CLAUDE.md): a MISSING D4 channel must FAIL, not silently pass.
# forced/decay always run the showcase that emits D4; absence means a showcase/pipeline
# regression, which must block admission rather than slip through. (was `ok4 or not has_d4`.)
sys.exit(0 if (ok1 and ok2 and ok3 and ok4) else 1)
