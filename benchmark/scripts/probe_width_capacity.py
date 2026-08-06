# -*- coding: utf-8 -*-
"""Width capacity probe (methodology-appendix material; reproducible evidence
behind the 2026-07-17 width rollback decision).

Conclusion: TFNO's FNO_WIDTH is a capacity knob. A constant parameter count
(the Tucker spectral core stays ~30.27M regardless of width) does not mean
constant capacity: width is the feature-channel dimension flowing between
spectral convolutions, and a rank-18 core expresses a different map space over
24 vs 40 channels. Criterion = single-sample overfitting (can the model fit an
arbitrary target), which is closer to true capacity than parameter counting.

Reproduce: python benchmark/scripts/probe_width_capacity.py
(CPU is enough, ~3 min; imports the production TFNO3d + production JSON
hyperparameters, sweeps width only.)
Measured 2026-07-17: w64 0.00140 / w40 0.00188 / w32 0.00457 / w24 0.01209 /
w16 0.11767 -> capacity decreases monotonically with width; w24 is 6.4x worse
than w40. Practical consequence: a relam50 w24 run lost to persistence on
rollout (nRMSE 1.28), which triggered the rollback to w40.
"""
import json
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Model.baselines.tfno_model import TFNO3d  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "Model" / "configs" / "production_configs_256.json"


def make(width, seed=0):
    torch.manual_seed(seed)
    d = {k: v for k, v in json.load(open(CFG))["models"]["tfno"].items() if not k.startswith("_")}
    d["FNO_WIDTH"] = width
    d.update(INPUT_TIMESTEPS=1, OUTPUT_TIMESTEPS=1, IN_CHANNELS=4, OUT_CHANNELS=4)
    return TFNO3d(types.SimpleNamespace(**d))


def overfit_loss(width, n_steps=80, n=16, lr=2e-3):
    """Loss after n_steps of single-sample overfitting (lower = more capacity). 16^3 random field; production hyperparameters, only width varies."""
    torch.manual_seed(1)
    x, y = torch.randn(1, 1, 4, n, n, n), torch.randn(1, 1, 4, n, n, n)
    m = make(width)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = (m(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    return loss.item()


if __name__ == "__main__":
    print("width | params(total) | overfit loss (80 steps, lower = more capacity)")
    for w in (64, 48, 40, 32, 24, 16):
        m = make(w)
        p = sum(q.numel() for q in m.parameters())
        print(f"w{w:2d}  | {p:,} | {overfit_loss(w):.5f}")
