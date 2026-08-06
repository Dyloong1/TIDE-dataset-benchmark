# Leaderboard row collection (input to the final aggregate)

Each machine copies its per-run row JSONs into this directory, preserving the
original layout, and force-adds them (gitignore exception):

    benchmark/results_rows/<machine>/<model>__<task>__<case>/<model>__<task>__<case>.json

Row JSONs only (a few KB each); no checkpoints or logs. The layout mirrors
`checkpoints/bench_*`, so `aggregate_leaderboard.py --roots` reads it directly.

## Directory conventions

- **Main-table rows** (including the non-learned reference rows) go to
  `results_rows/<machine>/`.
- **Replicate-seed rows** go to `results_rows/<machine>_seed1/`, `_seed2/`
  (independent top-level directories; the aggregate consumes them via
  `--seed-roots` and produces only the +/- std, never a main-table entry).
- **Ablation rows** (e.g. the stride-s8 mirrors) go to
  `results_rows/<machine>_ablation/` (excluded from the main aggregate).
- Layout may be flat files or subdirectories, but **the file name (or
  subdirectory name) must equal `model__task__case`** — the aggregate performs
  an identity check and rejects anything that does not match.
