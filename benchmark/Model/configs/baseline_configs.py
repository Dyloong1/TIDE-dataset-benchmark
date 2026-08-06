"""
Configuration for Baseline Models

Each baseline has its own set of hyperparameters optimized for
the 3D turbulence prediction task.
"""

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class BaselineConfig:
    """Base configuration for all models."""
    # Data dimensions
    INPUT_TIMESTEPS: int = 5
    OUTPUT_TIMESTEPS: int = 5
    INDICATORS: List[str] = None
    TOTAL_Z_LAYERS: int = 128

    def __post_init__(self):
        if self.INDICATORS is None:
            self.INDICATORS = ['u', 'v', 'w', 'p']


@dataclass
class UFNOConfig(BaselineConfig):
    """U-FNO Configuration."""
    # U-FNO specific
    UFNO_WIDTH: int = 32  # Base channel width
    FNO_MODES_Z: int = 16
    FNO_MODES_H: int = 24
    FNO_MODES_W: int = 24


@dataclass
class TransolverConfig(BaselineConfig):
    """Transolver Configuration."""
    # Transolver specific
    TRANSOLVER_DIM: int = 256
    TRANSOLVER_DEPTH: int = 6
    TRANSOLVER_HEADS: int = 8
    TRANSOLVER_SLICES: int = 16  # Number of physics slices
    TRANSOLVER_PATCH_SIZE: Tuple[int, int, int] = (8, 8, 8)


@dataclass
class DPOTConfig(BaselineConfig):
    """DPOT Configuration."""
    # DPOT specific
    DPOT_DIM: int = 256
    DPOT_DEPTH: int = 8
    DPOT_HEADS: int = 8
    DPOT_SPECTRAL_MODES: int = 32
    DPOT_PATCH_SIZE: Tuple[int, int, int] = (4, 8, 8)


@dataclass
class FactformerConfig(BaselineConfig):
    """Factformer Configuration."""
    # Factformer specific
    FACTFORMER_DIM: int = 128
    FACTFORMER_DEPTH: int = 6
    FACTFORMER_HEADS: int = 8
    FACTFORMER_DOWNSAMPLE: int = 4


@dataclass
class PDERefinerConfig(BaselineConfig):
    """PDE-Refiner Configuration."""
    # PDE-Refiner specific
    REFINER_HIDDEN: int = 64
    REFINER_STEPS: int = 3  # Number of refinement steps
    REFINER_TIME_DIM: int = 128
    REFINER_CH_MULTS: Tuple[int, ...] = (1, 2, 2, 4)
    REFINER_N_BLOCKS: int = 2
    MIN_NOISE_STD: float = 4e-7


@dataclass
class DiTRefinerConfig(BaselineConfig):
    """DiT-Refiner Configuration."""
    DIT_HIDDEN_DIM: int = 512
    DIT_NUM_HEADS: int = 8
    DIT_NUM_LAYERS: int = 8
    DIT_PATCH_SIZE: Tuple[int, int, int] = (4, 8, 8)
    DIT_MLP_RATIO: float = 4.0
    DIT_DROPOUT: float = 0.0
    REFINER_STEPS: int = 3
    REFINER_TIME_DIM: int = 128
    MIN_NOISE_STD: float = 4e-7


@dataclass
class FMHybridConfig(BaselineConfig):
    """FM-Hybrid Refiner Configuration."""
    REFINER_HIDDEN: int = 64
    REFINER_STEPS: int = 3      # K=3 refinement steps
    REFINER_TIME_DIM: int = 128
    MIN_NOISE_STD: float = 4e-7
    SIGMA_SCHEDULE: str = 'ddpm'  # 'ddpm', 'ddpm_large', or 'linear'
    ODE_STEPS: int = 10           # N substeps per refinement k at inference
    NOISE_TYPE: str = 'white'     # Flow matching prior noise type
    NOISE_SPECTRUM_PATH: str = None  # Path to precomputed spectrum .pt file
    SIGMA_MAX: float = None   # For fixed_range: max sigma (default 0.01 if None)
    SIGMA_MIN: float = None   # For fixed_range: min sigma (default 0.001 if None)


@dataclass
class UNet2DConfig(BaselineConfig):
    """2D UNet Baseline Configuration (slice-wise K=0 prediction).

    hidden=112 gives ~53M params, comparable to the 3D UNet baseline (~50M).
    """
    UNET2D_HIDDEN: int = 112
    UNET2D_CH_MULTS: Tuple[int, ...] = (1, 2, 2, 4)
    UNET2D_N_BLOCKS: int = 2


@dataclass
class ACDMConfig(BaselineConfig):
    """ACDM Configuration (Autoregressive Conditional Diffusion Model)."""
    ACDM_DIM: int = 96               # Base channel dimension (~48M params)
    ACDM_DIM_MULTS: Tuple[int, ...] = (1, 2, 2)  # Channel multipliers per level
    ACDM_CONVNEXT_MULT: int = 2      # ConvNext expansion factor
    ACDM_DIFF_STEPS: int = 100       # Diffusion timesteps
    ACDM_SCHEDULE: str = 'cosine'    # Beta schedule: 'cosine' or 'linear'
    ACDM_COND_MODE: str = 'clean'    # Conditioning: 'clean' or 'noisy'


@dataclass
class LatentDiffusionConfig(BaselineConfig):
    """Latent Diffusion Configuration (CoNFiLD-style)."""
    LD_LATENT_CH: int = 8            # Latent channels
    LD_ENC_CHANNELS: Tuple[int, ...] = (64, 128, 256)  # Encoder channel progression
    LD_DIFF_STEPS: int = 100         # Diffusion steps in latent space
    LD_SCHEDULE: str = 'cosine'      # Beta schedule
    LD_DDPM_HIDDEN: int = 64         # Latent DDPM U-Net hidden channels
    LD_DDPM_CH_MULTS: Tuple[int, ...] = (1, 2, 2)  # Latent U-Net channel mults
    LD_STAGE: str = 'ae'             # Training stage: 'ae' or 'ldm'


@dataclass
class FMRefinerUNetConfig(BaselineConfig):
    """FM-Refiner U-Net Configuration."""
    REFINER_HIDDEN: int = 64
    FM_STEPS: int = 3  # Number of Euler steps at inference
    REFINER_TIME_DIM: int = 128


@dataclass
class FMRefinerDiTConfig(BaselineConfig):
    """FM-Refiner DiT Configuration."""
    DIT_HIDDEN_DIM: int = 512
    DIT_NUM_HEADS: int = 8
    DIT_NUM_LAYERS: int = 8
    DIT_PATCH_SIZE: Tuple[int, int, int] = (4, 8, 8)
    DIT_MLP_RATIO: float = 4.0
    DIT_DROPOUT: float = 0.0
    FM_STEPS: int = 3
    REFINER_TIME_DIM: int = 128


# Preset configurations for different compute budgets
BASELINE_CONFIGS = {
    # U-FNO variants
    'ufno_small': UFNOConfig(UFNO_WIDTH=24, FNO_MODES_Z=12, FNO_MODES_H=16, FNO_MODES_W=16),
    'ufno_base': UFNOConfig(UFNO_WIDTH=32, FNO_MODES_Z=16, FNO_MODES_H=24, FNO_MODES_W=24),
    'ufno_large': UFNOConfig(UFNO_WIDTH=48, FNO_MODES_Z=20, FNO_MODES_H=32, FNO_MODES_W=32),

    # Transolver variants
    'transolver_small': TransolverConfig(TRANSOLVER_DIM=192, TRANSOLVER_DEPTH=4, TRANSOLVER_SLICES=8),
    'transolver_base': TransolverConfig(TRANSOLVER_DIM=256, TRANSOLVER_DEPTH=6, TRANSOLVER_SLICES=16),
    'transolver_large': TransolverConfig(TRANSOLVER_DIM=384, TRANSOLVER_DEPTH=8, TRANSOLVER_SLICES=24),

    # DPOT variants
    'dpot_small': DPOTConfig(DPOT_DIM=192, DPOT_DEPTH=6, DPOT_SPECTRAL_MODES=24),
    'dpot_base': DPOTConfig(DPOT_DIM=256, DPOT_DEPTH=8, DPOT_SPECTRAL_MODES=32),
    'dpot_large': DPOTConfig(DPOT_DIM=384, DPOT_DEPTH=10, DPOT_SPECTRAL_MODES=48),

    # Factformer variants
    'factformer_small': FactformerConfig(FACTFORMER_DIM=96, FACTFORMER_DEPTH=4),
    'factformer_base': FactformerConfig(FACTFORMER_DIM=128, FACTFORMER_DEPTH=6),
    'factformer_large': FactformerConfig(FACTFORMER_DIM=192, FACTFORMER_DEPTH=8),

    # PDE-Refiner variants
    'pde_refiner_small': PDERefinerConfig(REFINER_HIDDEN=48, REFINER_STEPS=2, REFINER_CH_MULTS=(1, 2, 2), REFINER_N_BLOCKS=1),
    'pde_refiner_base': PDERefinerConfig(REFINER_HIDDEN=64, REFINER_STEPS=3, REFINER_CH_MULTS=(1, 2, 2, 4), REFINER_N_BLOCKS=2),
    'pde_refiner_large': PDERefinerConfig(REFINER_HIDDEN=96, REFINER_STEPS=4, REFINER_CH_MULTS=(1, 2, 2, 4), REFINER_N_BLOCKS=2),

    # DiT-Refiner variants
    'dit_refiner_small': DiTRefinerConfig(DIT_HIDDEN_DIM=384, DIT_NUM_HEADS=6, DIT_NUM_LAYERS=6),
    'dit_refiner_base': DiTRefinerConfig(DIT_HIDDEN_DIM=512, DIT_NUM_HEADS=8, DIT_NUM_LAYERS=8),
    'dit_refiner_large': DiTRefinerConfig(DIT_HIDDEN_DIM=768, DIT_NUM_HEADS=12, DIT_NUM_LAYERS=12),

    # FM-Refiner U-Net variants
    'fm_refiner_unet_small': FMRefinerUNetConfig(REFINER_HIDDEN=48, FM_STEPS=3),
    'fm_refiner_unet_base': FMRefinerUNetConfig(REFINER_HIDDEN=64, FM_STEPS=3),
    'fm_refiner_unet_large': FMRefinerUNetConfig(REFINER_HIDDEN=96, FM_STEPS=3),

    # FM-Refiner DiT variants
    'fm_refiner_dit_small': FMRefinerDiTConfig(DIT_HIDDEN_DIM=384, DIT_NUM_HEADS=6, DIT_NUM_LAYERS=6),
    'fm_refiner_dit_base': FMRefinerDiTConfig(DIT_HIDDEN_DIM=512, DIT_NUM_HEADS=8, DIT_NUM_LAYERS=8),
    'fm_refiner_dit_large': FMRefinerDiTConfig(DIT_HIDDEN_DIM=768, DIT_NUM_HEADS=12, DIT_NUM_LAYERS=12),

    # FM-Hybrid Refiner variants
    'fm_hybrid_small': FMHybridConfig(REFINER_HIDDEN=48, REFINER_STEPS=2),
    'fm_hybrid_base': FMHybridConfig(REFINER_HIDDEN=64, REFINER_STEPS=3),
    'fm_hybrid_large': FMHybridConfig(REFINER_HIDDEN=96, REFINER_STEPS=4),

    # FM-Hybrid size ablation (K=4, ddpm_large, white noise)
    'fm_hybrid_size_tiny': FMHybridConfig(REFINER_HIDDEN=16, REFINER_STEPS=4, SIGMA_SCHEDULE='ddpm_large', ODE_STEPS=10),
    'fm_hybrid_size_small': FMHybridConfig(REFINER_HIDDEN=32, REFINER_STEPS=4, SIGMA_SCHEDULE='ddpm_large', ODE_STEPS=10),

    # 2D UNet Baseline
    'unet_2d_base': UNet2DConfig(),
    'unet_2d_small': UNet2DConfig(UNET2D_HIDDEN=96, UNET2D_CH_MULTS=(1, 2, 2, 4)),

    # ACDM variants
    'acdm_base': ACDMConfig(),
    'acdm_small': ACDMConfig(ACDM_DIM=64, ACDM_DIM_MULTS=(1, 2, 2)),

    # Latent Diffusion variants
    'latent_diffusion_base': LatentDiffusionConfig(),
    'latent_diffusion_small': LatentDiffusionConfig(LD_ENC_CHANNELS=(48, 96, 192), LD_DDPM_HIDDEN=48),
}


def get_baseline_config(model_name: str, size: str = 'base') -> BaselineConfig:
    """
    Get configuration for a baseline model.

    Args:
        model_name: 'ufno', 'transolver', 'dpot', 'factformer', 'pde_refiner'
        size: 'small', 'base', 'large'

    Returns:
        Configuration object
    """
    config_key = f"{model_name}_{size}"
    if config_key not in BASELINE_CONFIGS:
        raise ValueError(f"Unknown config: {config_key}. Available: {list(BASELINE_CONFIGS.keys())}")
    return BASELINE_CONFIGS[config_key]
