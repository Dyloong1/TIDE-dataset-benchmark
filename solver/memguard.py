"""CPU-RAM guard utilities. The phase-0 working directory was lost to a CPU OOM reboot;
the evaluation scripts load many full N^3 fields on the host, so they must be
defensive about host RAM, not just GPU VRAM.

Usage:
    from solver.memguard import ram_status, check_ram, free_cpu_caches

    check_ram("eval loop", abort_frac=0.92)   # raises before the OS OOM-kills us
    ...
    free_cpu_caches()                          # after del-ing big host tensors

The guard reads psutil if available; otherwise it degrades to a no-op that
prints a one-time warning (never blocks the run for a missing optional dep).
"""
from __future__ import annotations

import gc
import warnings

try:
    import psutil  # noqa
    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover
    _HAVE_PSUTIL = False
    warnings.warn("psutil not installed; CPU-RAM guard disabled (pip install psutil)")

import torch


def ram_status() -> dict:
    """Host RAM snapshot in GB. Empty dict if psutil is unavailable."""
    if not _HAVE_PSUTIL:
        return {}
    vm = psutil.virtual_memory()
    p = psutil.Process()
    return {
        "total_gb": vm.total / 1e9,
        "available_gb": vm.available / 1e9,
        "used_frac": vm.percent / 100.0,
        "proc_rss_gb": p.memory_info().rss / 1e9,
    }


def free_cpu_caches() -> None:
    """Release Python garbage and torch's CUDA cache. (torch keeps a CPU
    caching allocator for pinned memory only; ordinary host tensors are freed
    by the GC once their refcount drops, so an explicit gc.collect() after
    del-ing large host arrays is what actually returns RAM.)"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class RamAbort(MemoryError):
    """Raised proactively before the OS OOM-killer fires, so the run ends with
    a stack trace and partial results flushed rather than a hard reboot."""


def check_ram(where: str, abort_frac: float = 0.92, warn_frac: float = 0.85,
              try_free: bool = True) -> dict:
    """Proactive guard. If host RAM usage exceeds abort_frac, attempt to free
    caches once; if still over, raise RamAbort. Returns the status dict.

    abort_frac=0.92 on a 31.5 GB box leaves ~2.5 GB headroom — enough for the
    OS and to write out partial results, well before the Windows working-set
    pressure that triggered the earlier reboot.
    """
    st = ram_status()
    if not st:
        return st
    if st["used_frac"] >= warn_frac:
        print(f"[memguard] {where}: host RAM {st['used_frac']*100:.0f}% used, "
              f"{st['available_gb']:.1f} GB free (proc {st['proc_rss_gb']:.1f} GB)",
              flush=True)
    if st["used_frac"] >= abort_frac:
        if try_free:
            free_cpu_caches()
            st = ram_status()
        if st["used_frac"] >= abort_frac:
            raise RamAbort(
                f"{where}: host RAM at {st['used_frac']*100:.0f}% "
                f"(only {st['available_gb']:.1f} GB free) — aborting before OOM. "
                f"Reduce the number of fields held in memory or process them "
                f"in smaller batches.")
    return st


# Disk guarding moved to solver.diskguard (single source of truth, unit-tested).
# Re-export DiskAbort / disk_status here so existing memguard imports keep working.
from solver.diskguard import DiskAbort, disk_status  # noqa: E402,F401


def check_disk_space(path: str, need_gb: float, where: str = "",
                     headroom_gb: float = 20.0) -> dict:
    """DEPRECATED ALIAS — disk guarding now lives in solver.diskguard (single
    source of truth, unit-tested in test_diskguard.py). CYB-t and CYB-q both
    landed a disk guard in parallel after the 2026-06-19 incident; this thin
    wrapper preserves the memguard.check_disk_space call site by delegating to
    diskguard.check_disk (need_gb here already includes the run estimate; the
    headroom maps to diskguard's warn_gb floor)."""
    from solver.diskguard import check_disk
    return check_disk(path, need_gb=need_gb + headroom_gb, where=where)
