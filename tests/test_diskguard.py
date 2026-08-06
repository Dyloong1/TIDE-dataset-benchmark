"""diskguard must itself be vetted — it is a safety gate that aborts/permits long
runs, so a wrong threshold either crashes good runs or fails to prevent the next
156G disk-full incident. These tests pin: status reads a real fs, the byte
estimate matches the known checkpoint/frame sizes, abort fires below need, and a
generous need passes."""
import solver._env  # noqa: F401
import pytest

from solver.diskguard import (disk_status, estimate_run_bytes, check_disk,
                              DiskAbort)


def test_disk_status_reads_real_fs(tmp_path):
    st = disk_status(tmp_path)
    assert st and st["free_gb"] > 0 and 0.0 <= st["used_frac"] <= 1.0


def test_disk_status_walks_to_existing_parent(tmp_path):
    # a not-yet-created subdir resolves to its existing parent's filesystem
    st = disk_status(tmp_path / "does" / "not" / "exist")
    assert st and st["free_gb"] > 0


def test_estimate_matches_known_sizes():
    # one fp64 256^3 checkpoint (complex u_hat [3,256,256,129]) ~= 0.39 GB.
    one_ckpt = estimate_run_bytes(256, 1, "fp64") / 1.2  # strip the 20% overhead
    assert 0.35e9 < one_ckpt < 0.45e9
    # one 4-channel fp32 256^3 corpus frame ~= 0.27 GB (the incident's frame size)
    one_frame = (estimate_run_bytes(256, 0, "fp64", n_frames=1) / 1.2)
    assert 0.25e9 < one_frame < 0.30e9, one_frame
    # frames dominate: 621 frames (the incident, ~156G measured) >> checkpoints
    incident = estimate_run_bytes(256, 9, "fp64", n_frames=621) / 1e9
    assert 150 < incident < 220, incident  # would have been refused at start


def test_check_disk_aborts_when_insufficient(tmp_path):
    # demand absurdly more than any disk has -> must raise, not silently pass
    with pytest.raises(DiskAbort):
        check_disk(tmp_path, need_gb=1e9, where="test")


def test_check_disk_passes_with_headroom(tmp_path):
    # tiny need on a normal disk -> returns status, no raise
    st = check_disk(tmp_path, need_gb=0.001, where="test", warn_gb=0.0)
    assert st["free_gb"] > 0
