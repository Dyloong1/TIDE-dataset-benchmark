"""
Configuration for Baseline Models - Lite Version (Memory Efficient)

Designed to fit in ~20GB GPU memory for 128x128x128 data with batch_size=1.
These are smaller models but still competitive for comparison.
"""

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class BaselineConfig:
    """Base configuration for all models."""
    INPUT_TIMESTEPS: int = 5
    OUTPUT_TIMESTEPS: int = 5
    INDICATORS: List[str] = None
    TOTAL_Z_LAYERS: int = 128

    def __post_init__(self):
        if self.INDICATORS is None:
            self.INDICATORS = ['u', 'v', 'w', 'p']


# =============================================================================
# U-FNO Lite: Reduced from 1.46B to ~50M params
# =============================================================================
@dataclass
class UFNOConfigLite(BaselineConfig):
    """U-FNO Configuration - Memory efficient."""
    UFNO_WIDTH: int = 24         # Reduced from 48
    FNO_MODES_Z: int = 16        # Reduced from 24
    FNO_MODES_H: int = 16        # Reduced from 32
    FNO_MODES_W: int = 16        # Reduced from 32


# =============================================================================
# Transolver Lite: Reduced dim and depth
# =============================================================================
@dataclass
class TransolverConfigLite(BaselineConfig):
    """Transolver Configuration - Memory efficient."""
    TRANSOLVER_DIM: int = 384    # Reduced from 720
    TRANSOLVER_DEPTH: int = 8    # Reduced from 15
    TRANSOLVER_HEADS: int = 8    # Reduced from 12
    TRANSOLVER_SLICES: int = 32  # Reduced from 46
    TRANSOLVER_PATCH_SIZE: Tuple[int, int, int] = (4, 4, 4)  # Larger patches = fewer tokens


# =============================================================================
# DPOT Lite: Reduced significantly
# =============================================================================
@dataclass
class DPOTConfigLite(BaselineConfig):
    """DPOT Configuration - Memory efficient."""
    DPOT_DIM: int = 384          # Reduced from 936
    DPOT_DEPTH: int = 8          # Reduced from 15
    DPOT_HEADS: int = 8          # Reduced from 12
    DPOT_SPECTRAL_MODES: int = 64  # Reduced from 117
    DPOT_PATCH_SIZE: Tuple[int, int, int] = (4, 4, 4)  # Larger patches


# =============================================================================
# Factformer Lite: Reduced dim and depth
# =============================================================================
@dataclass
class FactformerConfigLite(BaselineConfig):
    """Factformer Configuration - Memory efficient."""
    FACTFORMER_DIM: int = 384    # Reduced from 768
    FACTFORMER_DEPTH: int = 8    # Reduced from 16
    FACTFORMER_HEADS: int = 8    # Reduced from 12
    FACTFORMER_DOWNSAMPLE: int = 2  # Downsample to reduce memory


# =============================================================================
# PDE-Refiner Lite: Already small, keep similar
# =============================================================================
@dataclass
class PDERefinerConfigLite(BaselineConfig):
    """PDE-Refiner Configuration - Memory efficient."""
    REFINER_HIDDEN: int = 64     # Reduced from 96
    REFINER_STEPS: int = 2       # Same
    REFINER_TIME_DIM: int = 64   # Reduced from 128
    REFINER_CH_MULTS: tuple = (1, 2, 2)  # 3 levels for memory
    REFINER_N_BLOCKS: int = 1
    MIN_NOISE_STD: float = 4e-7


# =============================================================================
# DiT-Refiner Lite
# =============================================================================
@dataclass
class DiTRefinerConfigLite(BaselineConfig):
    """DiT-Refiner Configuration - Memory efficient."""
    DIT_HIDDEN_DIM: int = 256
    DIT_NUM_HEADS: int = 4
    DIT_NUM_LAYERS: int = 4
    DIT_PATCH_SIZE: Tuple[int, int, int] = (4, 8, 8)
    DIT_MLP_RATIO: float = 4.0
    DIT_DROPOUT: float = 0.0
    REFINER_STEPS: int = 2
    REFINER_TIME_DIM: int = 64
    MIN_NOISE_STD: float = 4e-7


# =============================================================================
# FM-Refiner U-Net Lite
# =============================================================================
@dataclass
class FMRefinerUNetConfigLite(BaselineConfig):
    """FM-Refiner U-Net Configuration - Memory efficient."""
    REFINER_HIDDEN: int = 64
    FM_STEPS: int = 3
    REFINER_TIME_DIM: int = 64


# =============================================================================
# FM-Refiner DiT Lite
# =============================================================================
@dataclass
class FMRefinerDiTConfigLite(BaselineConfig):
    """FM-Refiner DiT Configuration - Memory efficient."""
    DIT_HIDDEN_DIM: int = 256
    DIT_NUM_HEADS: int = 4
    DIT_NUM_LAYERS: int = 4
    DIT_PATCH_SIZE: Tuple[int, int, int] = (4, 8, 8)
    DIT_MLP_RATIO: float = 4.0
    DIT_DROPOUT: float = 0.0
    FM_STEPS: int = 3
    REFINER_TIME_DIM: int = 64


# =============================================================================
# FM-Hybrid Refiner Lite
# =============================================================================
@dataclass
class FMHybridConfigLite(BaselineConfig):
    """FM-Hybrid Refiner Configuration - Memory efficient."""
    REFINER_HIDDEN: int = 64
    REFINER_STEPS: int = 2
    REFINER_TIME_DIM: int = 64
    MIN_NOISE_STD: float = 4e-7
    SIGMA_SCHEDULE: str = 'ddpm'
    ODE_STEPS: int = 10
    NOISE_TYPE: str = 'white'
    NOISE_SPECTRUM_PATH: str = None


# =============================================================================
# 2D UNet Baseline Lite
# =============================================================================
@dataclass
class UNet2DConfigLite(BaselineConfig):
    """2D UNet Baseline Configuration - Memory efficient."""
    UNET2D_HIDDEN: int = 112
    UNET2D_CH_MULTS: tuple = (1, 2, 2, 4)
    UNET2D_N_BLOCKS: int = 2


# =============================================================================
# ACDM Lite
# =============================================================================
@dataclass
class ACDMConfigLite(BaselineConfig):
    """ACDM Configuration - Memory efficient."""
    ACDM_DIM: int = 96
    ACDM_DIM_MULTS: tuple = (1, 2, 2)
    ACDM_CONVNEXT_MULT: int = 2
    ACDM_DIFF_STEPS: int = 100
    ACDM_SCHEDULE: str = 'cosine'
    ACDM_COND_MODE: str = 'clean'


# =============================================================================
# Latent Diffusion Lite
# =============================================================================
@dataclass
class LatentDiffusionConfigLite(BaselineConfig):
    """Latent Diffusion Configuration - Memory efficient."""
    LD_LATENT_CH: int = 8
    LD_ENC_CHANNELS: tuple = (64, 128, 256)
    LD_DIFF_STEPS: int = 100
    LD_SCHEDULE: str = 'cosine'
    LD_DDPM_HIDDEN: int = 64
    LD_DDPM_CH_MULTS: tuple = (1, 2, 2)
    LD_STAGE: str = 'ae'


# Registry for lite configs
BASELINE_CONFIGS_LITE = {
    'ufno': UFNOConfigLite(),
    'transolver': TransolverConfigLite(),
    'dpot': DPOTConfigLite(),
    'factformer': FactformerConfigLite(),
    'pde_refiner': PDERefinerConfigLite(),
    'dit_refiner': DiTRefinerConfigLite(),
    'fm_refiner_unet': FMRefinerUNetConfigLite(),
    'fm_refiner_dit': FMRefinerDiTConfigLite(),
    'fm_hybrid': FMHybridConfigLite(),
    'unet_2d': UNet2DConfigLite(),
    'acdm': ACDMConfigLite(),
    'latent_diffusion': LatentDiffusionConfigLite(),
}


def get_baseline_config_lite(model_name: str) -> BaselineConfig:
    """Get memory-efficient configuration for a baseline model."""
    if model_name not in BASELINE_CONFIGS_LITE:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(BASELINE_CONFIGS_LITE.keys())}")
    return BASELINE_CONFIGS_LITE[model_name]
