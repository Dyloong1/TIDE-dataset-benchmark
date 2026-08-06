"""
Baseline Models for 3D Turbulence Benchmark (re-curated 2026-07-13: 11 -> 7 learned).

Curation: representatives per method family (avoid same-family duplicates) + physics
lower-bounds (non-NN, in eval_benchmark.py). Rationale for the 2026-07-13 trim:

  FNO family:        FNO3d (classic anchor) + TFNO3d (newest/standard Tucker variant).
                     UFNO3d dropped (FNO+UNet, older, redundant with the two above).
  Attention operator: Transolver3D (Physics-Attention, ICML'24) + DPOT3D (denoising-pretrained
                     operator transformer, ICML'24, high-impact PDE foundation-model arch,
                     trained from scratch as a NO baseline). Factformer3D dropped (older, and
                     pricier at 14.6GB; transolver+dpot represent the attention family).
  Operator-learning: DeepONet3D (branch+trunk foundation; the ONLY resolution-bound model,
                     a differentiation point — kept regardless of family size).
  Generative:        PDERefiner3D (classic refiner) + ACDM3D (pure AR diffusion, turbulence).
                     FlowRefiner dropped (duplicates PDE-Refiner's refiner role);
                     LatentDiffusion3D dropped (two-stage, most expensive, harness unsupported).

DROPPED from the registry (files kept on disk, importable by their modules for appendix /
future-work reruns, simply absent from MODEL_REGISTRY so the curated benchmark never picks
them): UFNO3d, DPOT3D, Factformer3D, LatentDiffusion3D, FMHybridRefiner3D (flow_refiner),
plus the earlier-dropped DiTRefiner3D / FMRefinerUNet3D / FMRefinerDiT3D / UNetBaseline2D.
"""

from .deeponet_model import DeepONet3D
from .fno3d_model import FNO3d
from .tfno_model import TFNO3d
from .ufno_model import UFNO3d
from .transolver_model import Transolver3D
from .dpot_model import DPOT3D
from .factformer_model import Factformer3D
from .pde_refiner_model import PDERefiner3D
from .acdm_model import ACDM3D
from .latent_diffusion_model import LatentDiffusion3D
from .fm_hybrid_model import FMHybridRefiner3D  # in-house FlowRefiner

__all__ = [
    'DeepONet3D',
    'FNO3d',
    'TFNO3d',
    'UFNO3d',
    'Transolver3D',
    'DPOT3D',
    'Factformer3D',
    'PDERefiner3D',
    'ACDM3D',
    'LatentDiffusion3D',
    'FMHybridRefiner3D',
]

# Benchmark model registry (re-curated 2026-07-13, 7 learned models). Physics lower-bounds
# (spectral-interp, dynamic-Smagorinsky, persistence, identity) are non-NN and handled in
# eval_benchmark.py, not here. Dropped models keep their imports above (appendix/legacy) but
# are absent here so the curated benchmark never picks them.
MODEL_REGISTRY = {
    # FNO / spectral family (2): classic anchor + newest standard variant
    'fno3d': FNO3d,
    'tfno': TFNO3d,
    # CNN / U-Net family (1): U-FNO = spectral conv + U-Net multi-scale decoder. Restored
    # 2026-07-14 to give the benchmark a multi-scale-convolution representative (a D&B reviewer
    # expects a U-Net-family baseline like PDEBench/The Well/PDEArena carry); it breaks the
    # spectral-only monoculture on superres/recon/sgs. Takes separate in/out channels, so it runs
    # every task including pressure (3->1) and sgs (4->6).
    'ufno': UFNO3d,
    # attention operator (1): Physics-Attention transformer (ICML'24). dpot (DPOT, ICML'24, the
    # other attention-family model) stays importable below as an appendix baseline but is not in
    # the curated set — trimmed 2026-07-14 as family-redundant with transolver.
    'transolver': Transolver3D,
    # dpot (DPOT, ICML'24, attention-family): re-registered 2026-07-24 for the final-48h ext
    # appendix (5th baseline column). It was importable but never in MODEL_REGISTRY, so
    # --models dpot filtered to empty and train_benchmark rejected it as an invalid choice.
    # Config comes from production_configs_256.json "dpot" key (DPOT_DIM/DEPTH/HEADS/...).
    'dpot': DPOT3D,
    # operator-learning foundation (1): only resolution-bound model
    'deeponet': DeepONet3D,
    # generative (2): classic refiner + pure AR diffusion
    'pde_refiner': PDERefiner3D,
    'acdm': ACDM3D,
}


def get_model(model_name, config):
    """Get a benchmark model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](config)
