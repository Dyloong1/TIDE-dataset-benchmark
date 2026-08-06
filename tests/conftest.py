import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Reference implementation (physics_metrics) used as ground truth in cross-checks.
# Its location is machine-specific (it lives in the NeurIPS repo, not here), so it
# is NOT hard-coded — each machine points to it via the PHYSICS_METRICS_DIR env var.
# When unset/missing, the cross-check test importorskips itself (see test_grids.py).
#   Linux  example: export PHYSICS_METRICS_DIR=/home/ydai17/NeurIPS/ablations/vae_32_patch/code
#   Windows example: $env:PHYSICS_METRICS_DIR='C:\Users\ychen226\Yilong\NeurIPS\ablations\vae_32_patch\code'
_pm_dir = os.environ.get("PHYSICS_METRICS_DIR")
if _pm_dir and Path(_pm_dir).is_dir():
    sys.path.insert(0, _pm_dir)

import pytest  # noqa: E402
import torch  # noqa: E402


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
