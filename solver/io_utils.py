"""Result-directory discipline (knowledge base 7.2): every run directory gets
a config snapshot, git hash, seed, scalar time series CSV, spectra, metrics
json — reproducible from the directory alone.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def git_hash(repo_dir: str | Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def provenance_stamp(repo_dir: str | Path | None = None) -> str:
    """One-line provenance for acceptance tables: git hash (+ '-dirty' if the tree
    has uncommitted changes), date, machine, Python/torch versions. Lets any
    acceptance table be traced to the exact code/run that produced it — so a
    verdict can be judged trustworthy or suspect after the fact (review P2-1).
    """
    import sys as _sys
    import platform as _platform
    import datetime as _dt
    repo = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[1]
    gh = git_hash(repo)
    try:
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        if dirty:
            gh += "-dirty"
    except Exception:
        pass
    try:
        import torch as _torch
        tv = _torch.__version__
    except Exception:
        tv = "no-torch"
    # date passed-free: use UTC now (acceptance tables are stamped at write time)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    py = ".".join(map(str, _sys.version_info[:3]))
    return (f"_provenance_: git `{gh}` | {now} | {_platform.node()} "
            f"| py{py} torch{tv} | eval={Path(__file__).name}+caller")


def environment_record() -> dict:
    rec = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        rec["gpu"] = torch.cuda.get_device_name(0)
    return rec


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class SpectrumLogger:
    """Accumulates (t, E_k rows); saved as one npz at the end."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.times: list[float] = []
        self.spectra: list[np.ndarray] = []
        self.k: np.ndarray | None = None

    def log(self, t: float, k, E_k) -> None:
        if self.k is None:
            self.k = np.asarray(k, dtype=np.float64)
        self.times.append(t)
        self.spectra.append(np.asarray(E_k, dtype=np.float64))

    def save(self) -> None:
        np.savez(self.path, t=np.asarray(self.times), k=self.k,
                 E=np.stack(self.spectra) if self.spectra else np.empty((0,)))


def write_meta(out_dir: str | Path, cfg_dict: dict, extra: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": cfg_dict,
        "git_hash": git_hash(Path(__file__).resolve().parents[1]),
        "environment": environment_record(),
    }
    if extra:
        meta.update(extra)
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)


def write_metrics(out_dir: str | Path, metrics: dict) -> None:
    with open(Path(out_dir) / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)


def save_snapshot(out_dir: str | Path, t: float, u_phys: torch.Tensor,
                  p_phys: torch.Tensor | None = None) -> Path:
    """Save a physical-space snapshot. Stores velocity (3 channels) and, when
    given, the kinematic pressure (4th channel) — the corpus's 4-channel format
    (u,v,w,p), matching JHTDB. Pressure comes from operators.pressure_hat, which
    is certified by the D4 velocity-pressure consistency check and tests/test_pressure.
    """
    path = Path(out_dir) / f"snapshot_t{t:08.3f}.pt"
    data = {"t": t, "u": u_phys.cpu()}
    if p_phys is not None:
        data["p"] = p_phys.cpu()
    torch.save(data, path)
    return path
