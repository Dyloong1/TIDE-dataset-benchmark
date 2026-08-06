"""Every baseline must evaluate through the SAME tiled forward protocol.

Why: eval used to split by model -- fno3d/tfno/ufno ran a native-256^3 forward while transolver
and deeponet ran on a 128^3 patch (transolver's 512-row pos_embed stretches x8 at 128^3 but x64 at
256^3; deeponet's dense trunk OOMs at 256^3 ~48GB). That put two different eval resolutions in one
leaderboard. _TiledModel runs every model on the 128^3 tiles it trained on and stitches the 8
predictions back to the native field, so the forward protocol is uniform AND the metrics still see
a true 256^3 periodic box (a 128^3 sub-block of a periodic box is NOT periodic, so spectra /
enstrophy / Poisson residuals computed on a tile would be wrong).
"""
import sys
from pathlib import Path

import pytest
import torch

_BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH.parent))

from eval_benchmark import _TiledModel  # noqa: E402


class _Echo(torch.nn.Module):
    """Identity-ish model: returns the input's first frame, so stitching is exactly checkable."""
    def forward(self, x):                      # (B,t_in,C,Z,H,W) -> (B,1,C,Z,H,W)
        return x[:, :1]


class _Recorder(torch.nn.Module):
    """Records the spatial size it is called with."""
    def __init__(self):
        super().__init__()
        self.sizes = []

    def forward(self, x):
        self.sizes.append(int(x.shape[-1]))
        return x[:, :1]


def test_tiled_stitch_is_exact_for_identity():
    """8 tiles of a 256^3 field must reassemble the field bit-exactly."""
    x = torch.randn(1, 1, 2, 32, 32, 32)       # small stand-in for 256^3 (tile=16 -> 2x2x2)
    tiled = _TiledModel(_Echo(), tile=16)
    out = tiled(x)
    assert out.shape == (1, 1, 2, 32, 32, 32)
    assert torch.equal(out, x[:, :1]), "stitched output must equal the field the tiles came from"


def test_model_only_ever_sees_tile_sized_input():
    """The wrapped model must be called at the tile size, never at the native size -- that is the
    whole point (transolver/deeponet are only valid at the grid they trained on)."""
    rec = _Recorder()
    tiled = _TiledModel(rec, tile=16)
    tiled(torch.randn(1, 1, 2, 32, 32, 32))
    assert rec.sizes, "model was never called"
    assert set(rec.sizes) == {16}, f"model saw non-tile sizes: {sorted(set(rec.sizes))}"
    assert len(rec.sizes) == 8, f"expected 2x2x2=8 tiles, got {len(rec.sizes)}"


def test_no_tiling_when_field_fits():
    """A field already at or below the tile size must pass through in ONE call (no wasted tiling)."""
    rec = _Recorder()
    tiled = _TiledModel(rec, tile=16)
    tiled(torch.randn(1, 1, 2, 16, 16, 16))
    assert rec.sizes == [16], f"expected a single native call, got {rec.sizes}"


def test_channel_changing_output_is_stitched():
    """Tasks where C_out != C_in (pressure 3->1, sgs 3->6) must stitch on the OUTPUT channels."""
    class _Squeeze(torch.nn.Module):
        def forward(self, x):                  # (B,t,3,...) -> (B,1,1,...)
            return x[:, :1, :1]
    out = _TiledModel(_Squeeze(), tile=16)(torch.randn(1, 1, 3, 32, 32, 32))
    assert out.shape == (1, 1, 1, 32, 32, 32), f"got {tuple(out.shape)}"
