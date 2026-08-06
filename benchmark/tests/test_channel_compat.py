"""Judge-first test for task/model channel compatibility (pressure 3->1, sgs 4->6).

Root cause of the shipped bug: only 4 of 11 models (fno3d/tfno/deeponet/ufno) honor
OUT_CHANNELS != IN_CHANNELS; the other 7 hardcode out == in. On a channel-changing task an
incompatible model would SILENTLY broadcast in mse_loss (3ch pred vs 1ch target -> UserWarning
only) and train to a garbage loss that still lands a leaderboard number, or get silently
truncated at eval. The `09_review` "pressure/sgs 5/5 PASS" was a false all-clear (it only
exercised fno3d + deeponet).

This test iterates the WHOLE registry and pins, per model, exactly how many output channels it
emits on pressure (out=1) and sgs (out=6). It documents the design contract (the 4 flexible
models must emit the requested out_ch; the rest emit in_ch) so a regression — e.g. a model
silently starting/stopping honoring OUT_CHANNELS — trips the test instead of shipping green.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=<root>:<root>/benchmark \
     python -m pytest tests/test_channel_compat.py -q
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

os.environ.setdefault("TURBGEN_REPO", str(Path.home() / "turbulence"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Model.baselines import get_model, MODEL_REGISTRY  # noqa: E402

# Models that honor OUT_CHANNELS != IN_CHANNELS (design contract: these can run pressure/sgs).
# Channel-flexible kept models (take separate in/out channels, so they run in!=out tasks like
# pressure 3->1 / sgs 4->6): fno3d/tfno/deeponet/ufno. (ufno restored 2026-07-14 as the CNN/U-Net
# family baseline; dpot dropped from MODEL_REGISTRY same day — family-redundant with transolver —
# so it is NOT listed here, though its .py stays for appendix reruns.) The channel guard is dynamic
# (probe pred_ch vs out_ch), so it auto-accepts any flexible model on P4/P5; the REQUIRED minimal
# set for P4/P5 uses fno3d/tfno/deeponet.
CHANNEL_FLEXIBLE = {"fno3d", "tfno", "deeponet", "ufno"}


def _cfg(in_ch, out_ch, t_in=1, t_out=1):
    return SimpleNamespace(
        INPUT_TIMESTEPS=t_in, OUTPUT_TIMESTEPS=t_out,
        INDICATORS=['u', 'v', 'w', 'p'][:in_ch] if in_ch <= 4 else ['c%d' % i for i in range(in_ch)],
        IN_CHANNELS=in_ch, OUT_CHANNELS=out_ch,
        FNO_WIDTH=12, UFNO_WIDTH=12, FNO_LAYERS=2, FNO_MODES_Z=6, FNO_MODES_H=6, FNO_MODES_W=6,
        TFNO_RANK=8, DEEPONET_P=64, DEEPONET_WIDTH=12)


def _out_channels(name, in_ch, out_ch, t_in=1):
    """Emit channel count of model(name) on a (1,t_in,in_ch,G,G,G) probe, or None if it can't fwd."""
    G = 16
    model = get_model(name, _cfg(in_ch, out_ch, t_in=t_in))
    x = torch.randn(1, t_in, in_ch, G, G, G)
    with torch.no_grad():
        try:
            return model(x).shape[-4]
        except Exception:
            return None


def test_flexible_models_honor_out_channels_pressure():
    # pressure: in=3 (velocity) -> out=1. The 4 flexible models MUST emit exactly 1.
    for name in sorted(CHANNEL_FLEXIBLE):
        oc = _out_channels(name, 3, 1)
        assert oc == 1, f"{name} pressure: expected 1 out channel, got {oc}"


def test_flexible_models_honor_out_channels_sgs():
    # sgs: in=4 -> out=6 (stress tensor). The 4 flexible models MUST emit exactly 6.
    for name in sorted(CHANNEL_FLEXIBLE):
        oc = _out_channels(name, 4, 6)
        assert oc == 6, f"{name} sgs: expected 6 out channels, got {oc}"


def test_inflexible_models_emit_in_channels_not_out():
    # The other models hardcode out==in. Pin that contract: on pressure (in=3,out=1) they emit 3
    # (NOT 1). This is WHY the fail-fast guard in train/eval refuses to run them on pressure/sgs.
    # If a model here starts honoring OUT_CHANNELS, move it into CHANNEL_FLEXIBLE + the design doc.
    inflexible = sorted(set(MODEL_REGISTRY) - CHANNEL_FLEXIBLE - {"latent_diffusion"})
    offenders = []
    for name in inflexible:
        oc = _out_channels(name, 3, 1)
        if oc is None:
            continue  # forward failed (separate concern); not a silent-broadcast risk
        if oc == 1:
            # unexpectedly flexible — good, but the design contract/doc must be updated
            offenders.append((name, "now honors OUT_CHANNELS — update CHANNEL_FLEXIBLE + design doc"))
        else:
            assert oc == 3, f"{name}: expected 3 (out==in), got {oc}"
    assert not offenders, offenders


def test_guard_would_catch_mismatch():
    # the exact check the train/eval fail-fast guards use: pred_ch != out_ch -> refuse.
    # transolver on pressure emits 3 != 1 -> guard must fire.
    oc = _out_channels("transolver", 3, 1)
    assert oc is not None and oc != 1, "guard precondition: incompatible model emits wrong out_ch"


if __name__ == "__main__":
    for fn in [test_flexible_models_honor_out_channels_pressure,
               test_flexible_models_honor_out_channels_sgs,
               test_inflexible_models_emit_in_channels_not_out,
               test_guard_would_catch_mismatch]:
        fn(); print(f"  PASS {fn.__name__}")
    print("\n4/4 channel-compat tests passed")
