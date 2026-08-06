# -*- coding: utf-8 -*-
"""Aggregate identity check + seed-root isolation pin (judge for the 2026-07-17 double bug fix).

bug1: the old load_rows keyed rows only on JSON-internal fields, so same-key rows under
      ablation_*/ood_*/control_* directories silently overwrote official rows (last glob
      wins). Fix = the directory name (or flat file name) must equal model__task__case.
bug2: the aggregate has no averaging logic, so mounting seed roots in --roots let seed2
      overwrite the seed0 main row. Fix = seed roots go through a separate --seed-roots and
      produce only the +/- std, never a main-table entry.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate_leaderboard import load_rows, load_seed_std  # noqa: E402


def _row(model, task, case, nrmse):
    return {"model": model, "task": task, "case": case, "nRMSE_mean": nrmse}


def _write(d, name, row):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(row))


def test_ablation_and_ood_dirs_are_excluded(tmp_path):
    root = tmp_path / "bench"
    ident = "fno3d__rollout__caseA"
    # official row (subdirectory name = identity)
    _write(root / ident, f"{ident}.json", _row("fno3d", "rollout", "caseA", 0.80))
    # ablation row: same internal keys, mismatched directory name -> must be rejected
    _write(root / "ablation_stride4__fno3d_caseA", f"{ident}.json",
           _row("fno3d", "rollout", "caseA", 0.99))
    # OOD row: likewise
    _write(root / "ood_caseA__fno3d__from__caseB", f"{ident}.json",
           _row("fno3d", "rollout", "caseA", 0.55))
    rows = load_rows([str(root)])
    assert rows[("fno3d", "rollout", "caseA")]["nRMSE_mean"] == 0.80, \
        "a same-key row from an ablation/OOD directory overwrote the official row -- identity check broken"


def test_flat_layout_accepted_with_stem_check(tmp_path):
    # flat-file layout of results_rows/<machine>/ (the layout machines actually push)
    root = tmp_path / "results_rows" / "CYB-t"
    ident = "transolver__rollout__caseR"
    _write(root, f"{ident}.json", _row("transolver", "rollout", "caseR", 0.74))
    _write(root, "mislabeled.json", _row("transolver", "rollout", "caseR", 0.11))  # name mismatch -> rejected
    rows = load_rows([str(root)])
    assert rows[("transolver", "rollout", "caseR")]["nRMSE_mean"] == 0.74


def test_seed_roots_never_pollute_main(tmp_path):
    main, seed = tmp_path / "main", tmp_path / "seed1"
    ident = "fno3d__rollout__caseA"
    _write(main / ident, f"{ident}.json", _row("fno3d", "rollout", "caseA", 0.80))
    _write(seed / ident, f"{ident}.json", _row("fno3d", "rollout", "caseA", 0.83))
    rows = load_rows([str(main)])
    std = load_seed_std([str(seed)])
    assert rows[("fno3d", "rollout", "caseA")]["nRMSE_mean"] == 0.80, "a seed row entered the main table"
    assert std[("fno3d", "rollout", "caseA")] == [0.83], "the seed value was not collected separately"


def test_missing_root_is_skipped_not_fatal(tmp_path):
    rows = load_rows([str(tmp_path / "does_not_exist")])
    assert rows == {}
