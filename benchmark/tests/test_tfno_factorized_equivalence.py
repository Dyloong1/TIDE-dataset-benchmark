# -*- coding: utf-8 -*-
"""TFNO factorized apply vs dense reconstruction: mathematical-identity pin (correctness judge for the 2026-07-17 performance optimization).

_apply_factorized and _compl_mul3d(x, _reconstruct(c)) are the same multilinear form under
different association orders, so their outputs must agree within fp rounding. If either path
is broken (axis mix-up, factor misalignment, wrong corner slicing), this test goes red.
The model forward hard-codes fp32 (the production dtype), so we test in fp32 with a
threshold set by fp32 association-order rounding (rel < 1e-5).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Model.baselines.tfno_model import TuckerSpectralConv3d  # noqa: E402


@pytest.mark.parametrize("shape", [(2, 6, 16, 16, 16), (1, 5, 24, 24, 24)])
def test_factorized_equals_reconstruct(shape):
    torch.manual_seed(0)
    B, C, N, _, _ = shape
    conv = TuckerSpectralConv3d(C, C, modes_z=6, modes_h=6, modes_w=6, rank=4)
    x = torch.randn(B, C, N, N, N)

    conv.factorized_apply = False
    y_dense = conv(x)
    conv.factorized_apply = True
    y_fact = conv(x)

    diff = (y_dense - y_fact).abs().max().item()
    scale = y_dense.abs().max().item()
    assert diff / max(scale, 1e-30) < 1e-5, (
        f"factorized path diverged from dense reconstruct: rel max diff {diff/scale:.3e} "
        f" (the two paths must be the same mathematical object)")


def test_factorized_is_the_default():
    conv = TuckerSpectralConv3d(4, 4, 6, 6, 6, rank=3)
    assert getattr(conv, "factorized_apply", True) is True, (
        "factorized apply should be the default (since 2026-07-17); disabling requires factorized_apply=False explicitly")


def test_gradients_flow_identically():
    """Gradients w.r.t. the same parameters must also agree between the two paths (the optimizer sees the same loss surface)."""
    torch.manual_seed(1)
    conv = TuckerSpectralConv3d(4, 4, 6, 6, 6, rank=3)
    x = torch.randn(1, 4, 12, 12, 12)

    grads = {}
    for mode in (False, True):
        conv.zero_grad(set_to_none=True)
        conv.factorized_apply = mode
        conv(x).pow(2).sum().backward()
        grads[mode] = [p.grad.clone() for p in conv.parameters() if p.grad is not None]
    assert len(grads[False]) == len(grads[True])
    for gd, gf in zip(grads[False], grads[True]):
        rel = (gd - gf).abs().max() / gd.abs().max().clamp_min(1e-30)
        assert rel < 1e-4, f"gradient mismatch: rel {rel:.3e}"
