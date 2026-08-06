"""Judge-first test for residual (delta) rollout prediction — the fix for predict-mean collapse.

Root cause (measured): absolute-frame regression collapses to 'predict the mean' (pred std ~0.02
vs truth 0.15), because consecutive turbulence frames are highly correlated so the trivial basin
(loss ~= var(y)) dominates. Residual makes the model predict a delta and add back the last input
frame: pred = x[:,-1] + f(x). This test pins that arithmetic (train side) and the AR-unroll side.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=<root>:<root>/benchmark python -m pytest tests/test_residual.py -q
"""
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("TURBGEN_REPO", str(Path.home() / "turbulence"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_benchmark import _predict  # noqa: E402
from eval_benchmark import _ar_rollout_streaming  # noqa: E402


class _Zero(torch.nn.Module):
    """model that outputs ~0 (the collapse case). With residual, pred should = last input frame."""
    def __init__(self, t_out=1):
        super().__init__(); self.t_out = t_out
    def forward(self, x):                       # x (B,T_in,C,Z,H,W)
        B, _, C, Z, H, W = x.shape
        return torch.zeros(B, self.t_out, C, Z, H, W)


class _Const(torch.nn.Module):
    """model that outputs a constant delta d each step."""
    def __init__(self, d, t_out=1):
        super().__init__(); self.d = d; self.t_out = t_out
    def forward(self, x):
        B, _, C, Z, H, W = x.shape
        return torch.full((B, self.t_out, C, Z, H, W), self.d)


def test_predict_residual_adds_last_frame():
    # zero-delta model + residual -> pred == last input frame (persistence, the natural baseline)
    x = torch.randn(2, 5, 4, 8, 8, 8)
    pred_res = _predict(_Zero(), x, residual=True)
    assert torch.allclose(pred_res, x[:, -1:], atol=1e-6)
    # without residual, the zero model outputs zero (the collapse)
    pred_abs = _predict(_Zero(), x, residual=False)
    assert pred_abs.abs().max() < 1e-6


def test_predict_residual_nonrollout_shape_safe():
    # channel-changing outputs (e.g. pressure 4->1) must NOT get residual (shapes incompatible)
    class _P(torch.nn.Module):
        def forward(self, x):
            B = x.shape[0]; Z = x.shape[-1]
            return torch.ones(B, 1, 1, Z, Z, Z)      # out C=1 != in C=4
    x = torch.randn(1, 1, 4, 6, 6, 6)
    out = _predict(_P(), x, residual=True)
    assert out.shape[2] == 1 and float(out.mean()) == 1.0  # unchanged, no last-frame added


def test_ar_unroll_residual_persistence_is_stable():
    # zero-delta model unrolled WITH residual = persistence: every predicted frame == the last
    # input frame (the window's newest), so a constant-in-time truth gives ~0 error and no blow-up.
    B, t_in, C, N, t_out = 1, 5, 4, 8, 6
    frame = torch.randn(B, 1, C, N, N, N)
    x = frame.repeat(1, t_in, 1, 1, 1, 1)             # constant-in-time input
    y = frame[:, 0:1].repeat(1, t_out, 1, 1, 1, 1)    # constant truth
    per, last_p, _ = _ar_rollout_streaming(_Zero(), x, y, k0=None, decay=False, residual=True)
    assert all(abs(v) < 1e-5 for v in per), per       # persistence tracks a constant field
    assert torch.allclose(last_p, frame[0, 0], atol=1e-5)


def test_ar_unroll_residual_accumulates_drift():
    # constant +delta model with residual: frame_k = last + k*delta -> linear drift (real AR
    # accumulation, the physics rollout probes). error grows monotonically vs a constant truth.
    B, t_in, C, N, t_out = 1, 4, 3, 6, 6
    x = torch.zeros(B, t_in, C, N, N, N)
    y = torch.zeros(B, t_out, C, N, N, N)             # truth stays 0
    per, _, _ = _ar_rollout_streaming(_Const(0.1), x, y, k0=None, decay=False, residual=True)
    # step k prediction = k*0.1 (drift), truth 0 -> error strictly increasing
    assert all(per[i] < per[i + 1] for i in range(len(per) - 1)), per


if __name__ == "__main__":
    for fn in [test_predict_residual_adds_last_frame, test_predict_residual_nonrollout_shape_safe,
               test_ar_unroll_residual_persistence_is_stable, test_ar_unroll_residual_accumulates_drift]:
        fn(); print(f"  PASS {fn.__name__}")
    print("\n4/4 residual tests passed")
