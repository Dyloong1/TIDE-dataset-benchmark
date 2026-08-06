"""Judge-first test for the STREAMING autoregressive rollout (the rollout eval protocol).

Design rollout protocol: train a single-step model (t_in frames -> 1 frame), then at eval unroll
it autoregressively to t_out frames, feeding each prediction back as the newest input frame.
Errors accumulate step-by-step — the physics the rollout task probes.

_ar_rollout_streaming scores each frame AS IT IS PRODUCED (scalar nRMSE vs that step's truth) and
drops it, so no full 20-frame trajectory is ever held (native 256^3 x20 frames OOMs GPU *and*
host RAM under the scope MemoryMax — the bug this replaced). Only the last frame is returned, for
the physics metrics. This pins: (1) it returns t_out per-step scalars + the last pred/truth frame;
(2) the sliding window is correct (identity single-step model on a constant field reproduces it,
per-step nRMSE all ~0); (3) errors accumulate (a model drifting from truth gives a rising nRMSE
curve); (4) the per-step scalar equals a direct nrmse on the corresponding frame.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=<root>:<root>/benchmark \
     python -m pytest tests/test_ar_unroll.py -q
"""
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("TURBGEN_REPO", str(Path.home() / "turbulence"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_benchmark import _ar_rollout_streaming  # noqa: E402
import benchmark_metrics as BM  # noqa: E402


class _LastFrame(torch.nn.Module):
    """single-step 'model': predict the last input frame unchanged (persistence-like)."""
    def forward(self, w):          # w: (B, t_in, C, ...)
        return w[:, -1:]           # (B, 1, C, ...)


class _AddDelta(torch.nn.Module):
    """single-step model: next frame = last frame + delta (deterministic linear drift)."""
    def __init__(self, delta):
        super().__init__(); self.delta = delta
    def forward(self, w):
        return w[:, -1:] + self.delta


def test_returns_perstep_and_last_frame():
    B, t_in, C, N, t_out = 1, 5, 4, 8, 7
    x = torch.randn(B, t_in, C, N, N, N)
    y = torch.randn(B, t_out, C, N, N, N)
    per_step, last_pred, last_true = _ar_rollout_streaming(_LastFrame(), x, y, k0=None, decay=False)
    assert len(per_step) == t_out
    assert last_pred.shape == (C, N, N, N) and last_true.shape == (C, N, N, N)
    assert torch.equal(last_true, y[0, -1])            # last truth frame is y's last


def test_identity_on_constant_field_scores_zero():
    # constant-in-time input + last-frame model -> every predicted frame == the truth frame
    # (truth also constant) -> per-step nRMSE all ~0.
    B, t_in, C, N, t_out = 1, 5, 4, 6, 10
    frame = torch.randn(B, 1, C, N, N, N)
    x = frame.repeat(1, t_in, 1, 1, 1, 1)
    y = frame[:, 0:1].repeat(1, t_out, 1, 1, 1, 1)     # constant truth == the frame
    per_step, last_pred, _ = _ar_rollout_streaming(_LastFrame(), x, y, k0=None, decay=False)
    assert all(abs(v) < 1e-5 for v in per_step), per_step
    assert torch.allclose(last_pred, frame[0, 0], atol=1e-6)


def test_error_accumulates_monotonically():
    # add-delta model drifts away from a constant truth by k*delta at step k -> nRMSE strictly rises.
    B, t_in, C, N, t_out = 1, 4, 3, 5, 8
    x = torch.zeros(B, t_in, C, N, N, N)               # last input frame 0
    y = torch.zeros(B, t_out, C, N, N, N)              # truth stays 0
    per_step, _, _ = _ar_rollout_streaming(_AddDelta(0.1), x, y, k0=None, decay=False)
    # pred at step k = k*delta (constant field), truth 0 -> nrmse = ||k*delta|| / ||0||... but truth
    # is 0 so nrmse normalizes by pred? Use a nonzero truth to keep nrmse well-defined instead:
    y2 = torch.ones(B, t_out, C, N, N, N)
    per_step2, _, _ = _ar_rollout_streaming(_AddDelta(0.1), torch.ones(B, t_in, C, N, N, N), y2,
                                            k0=None, decay=False)
    # pred_k = 1 + k*0.1, truth 1 -> error grows with k -> nRMSE strictly increasing
    assert all(per_step2[i] < per_step2[i + 1] for i in range(len(per_step2) - 1)), per_step2


def test_perstep_matches_direct_nrmse():
    # the streamed scalar at each step must equal a direct nrmse(pred_frame, truth_frame).
    B, t_in, C, N, t_out = 1, 3, 2, 4, 4
    x = torch.randn(B, t_in, C, N, N, N)
    y = torch.randn(B, t_out, C, N, N, N)
    # reproduce the unroll by hand and score each frame directly
    win = x.clone(); ref = []
    for t in range(t_out):
        nxt = win[:, -1:]
        ref.append(float(BM.nrmse(nxt[0, 0], y[0, t])))
        win = torch.cat([win[:, 1:], nxt], dim=1)
    per_step, _, _ = _ar_rollout_streaming(_LastFrame(), x, y, k0=None, decay=False)
    for a, b in zip(per_step, ref):
        assert abs(a - b) < 1e-5, (a, b)


if __name__ == "__main__":
    for fn in [test_returns_perstep_and_last_frame, test_identity_on_constant_field_scores_zero,
               test_error_accumulates_monotonically, test_perstep_matches_direct_nrmse]:
        fn(); print(f"  PASS {fn.__name__}")
    print("\n4/4 streaming AR-unroll tests passed")
