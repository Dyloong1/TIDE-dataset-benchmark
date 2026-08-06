"""Disk-space guard utilities. The 2026-06-19 incident: a verification long run
was accidentally given frame export (--frame-every-TL), wrote 621 fp32 256^3
4-channel frames = 156G, filled the root disk to 100%, and CRASHED while writing
spectra.npz/metrics.json (OSError: No space left) — truncating the sampling
window AND losing the final metrics. Both machines now guard disk before a long
run / corpus run, the same way memguard.py guards host RAM.

Usage:
    from solver.diskguard import disk_status, check_disk, estimate_run_bytes

    # before a long run: refuse to start if there isn't room for the output
    check_disk(out_dir, need_gb=estimate_run_bytes(...) / 1e9, where="forced_hit start")
    ...
    # periodically during the run (cheap), so we abort with flushed partials
    # rather than crash mid-write on a full disk
    check_disk(out_dir, need_gb=2.0, where="checkpoint")

Degrades to a no-op-with-warning if shutil.disk_usage is somehow unavailable
(never blocks a run for a missing capability).
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path


def disk_status(path: str | Path) -> dict:
    """Free/total disk space (GB) for the filesystem holding `path`. Walks up to
    the nearest existing parent (the dir may not be created yet). Empty dict if
    disk_usage is unavailable."""
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        du = shutil.disk_usage(p)
    except Exception:  # pragma: no cover
        warnings.warn("shutil.disk_usage unavailable; disk guard disabled")
        return {}
    return {
        "total_gb": du.total / 1e9,
        "free_gb": du.free / 1e9,
        "used_frac": du.used / du.total,
        "path": str(p),
    }


def estimate_run_bytes(N: int, n_checkpoints: int, dtype: str = "fp64",
                       n_frames: int = 0, frame_channels: int = 4,
                       frame_dtype: str = "fp32") -> float:
    """Rough on-disk footprint of a run, in bytes. A checkpoint stores the
    complex u_hat spectrum [3, N, N, N//2+1] in the solver dtype (16 B/complex
    fp64); a corpus frame stores frame_channels real N^3 fields, exported as
    fp32 by the gated pipeline (4 B/real). The 2026-06-19 incident was 621 fp32
    4-channel 256^3 frames = ~156G. Use to size need_gb BEFORE the run so we
    fail fast instead of mid-write."""
    cbytes = 2 * (8 if dtype == "fp64" else 4)
    fbytes = 8 if frame_dtype == "fp64" else 4
    ckpt = 3 * N * N * (N // 2 + 1) * cbytes
    frame = frame_channels * N * N * N * fbytes
    # +20% for spectra/timeseries/meta + filesystem overhead
    return (n_checkpoints * ckpt + n_frames * frame) * 1.2


class DiskAbort(OSError):
    """Raised proactively before the filesystem fills, so the run ends with a
    stack trace and flushed partials rather than a corrupt mid-write crash."""


def check_disk(path: str | Path, need_gb: float, where: str,
               warn_gb: float = 10.0) -> dict:
    """Proactive guard. Raise DiskAbort if the filesystem holding `path` has less
    than `need_gb` free. Warn (don't abort) below `warn_gb`. Returns status dict.

    Call once at run start with the full estimated footprint, then cheaply at
    each checkpoint with a small need_gb (one checkpoint + flush headroom)."""
    st = disk_status(path)
    if not st:
        return st
    if st["free_gb"] < max(need_gb, warn_gb):
        print(f"[diskguard] {where}: {st['free_gb']:.1f} GB free on {st['path']} "
              f"({st['used_frac']*100:.0f}% used), need ~{need_gb:.1f} GB",
              flush=True)
    if st["free_gb"] < need_gb:
        raise DiskAbort(
            f"{where}: only {st['free_gb']:.1f} GB free on {st['path']} but the "
            f"run needs ~{need_gb:.1f} GB — refusing to start/continue before a "
            f"disk-full crash. Free space or redirect output to a larger volume. "
            f"(Frame export belongs in the gated corpus pipeline, NOT a "
            f"verification run — see the 2026-06-19 incident.)")
    return st
