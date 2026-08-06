"""
Configuration for Baseline Models - 30GB VRAM Target

Adjusted configurations to use approximately 30GB GPU memory
for fair comparison across all models.
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
# U-FNO: Base uses ~12GB, scale up ~2.5x
# =============================================================================
@dataclass
class UFNOConfig30G(BaselineConfig):
    """U-FNO Configuration for ~30GB VRAM."""
    UFNO_WIDTH: int = 48  # Keep moderate for CPU memory
    FNO_MODES_Z: int = 24  # Keep moderate
    FNO_MODES_H: int = 32  # Keep moderate
    FNO_MODES_W: int = 32  # Keep moderate


# =============================================================================
# Transolver: Base uses ~1.5GB, need much larger model
# =============================================================================
@dataclass
class TransolverConfig30G(BaselineConfig):
    """Transolver Configuration for ~30GB VRAM."""
    TRANSOLVER_DIM: int = 720  # Tuned for ~30GB (divisible by 12)
    TRANSOLVER_DEPTH: int = 15  # Tuned depth
    TRANSOLVER_HEADS: int = 12
    TRANSOLVER_SLICES: int = 46  # Tuned slices
    TRANSOLVER_PATCH_SIZE: Tuple[int, int, int] = (2, 2, 2)  # Smaller patches = more tokens


# =============================================================================
# DPOT: Keep larger patches to avoid OOM
# =============================================================================
@dataclass
class DPOTConfig30G(BaselineConfig):
    """DPOT Configuration for ~30GB VRAM."""
    DPOT_DIM: int = 936  # Tuned for ~30GB (divisible by 12)
    DPOT_DEPTH: int = 15  # Tuned depth
    DPOT_HEADS: int = 12
    DPOT_SPECTRAL_MODES: int = 117
    DPOT_PATCH_SIZE: Tuple[int, int, int] = (2, 2, 4)  # 64*32*16 = 32768 tokens


# =============================================================================
# Factformer: Increase significantly
# =============================================================================
@dataclass
class FactformerConfig30G(BaselineConfig):
    """Factformer Configuration for ~30GB VRAM."""
    FACTFORMER_DIM: int = 768  # Moderate for CPU memory
    FACTFORMER_DEPTH: int = 16  # Moderate depth
    FACTFORMER_HEADS: int = 12
    FACTFORMER_DOWNSAMPLE: int = 1  # No downsampling


# =============================================================================
# PDE-Refiner: Optimize for 30GB
# =============================================================================
@dataclass
class PDERefinerConfig30G(BaselineConfig):
    """PDE-Refiner Configuration for ~30GB VRAM."""
    REFINER_HIDDEN: int = 96  # Moderate for memory
    REFINER_STEPS: int = 2  # Fewer steps
    REFINER_TIME_DIM: int = 128
    REFINER_CH_MULTS: tuple = (1, 2, 2, 4)
    REFINER_N_BLOCKS: int = 2
    MIN_NOISE_STD: float = 4e-7


# =============================================================================
# DiT-Refiner: Optimized for 30GB
# =============================================================================
@dataclass
class DiTRefinerConfig30G(BaselineConfig):
    """DiT-Refiner Configuration for ~30GB VRAM."""
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
class FMRefinerUNetConfig30G(BaselineConfig):
    """FM-Refiner U-Net Configuration for ~30GB VRAM."""
    REFINER_HIDDEN: int = 96
    FM_STEPS: int = 3
    REFINER_TIME_DIM: int = 128


@dataclass
class FMRefinerDiTConfig30G(BaselineConfig):
    """FM-Refiner DiT Configuration for ~30GB VRAM."""
    DIT_HIDDEN_DIM: int = 512
    DIT_NUM_HEADS: int = 8
    DIT_NUM_LAYERS: int = 8
    DIT_PATCH_SIZE: Tuple[int, int, int] = (4, 8, 8)
    DIT_MLP_RATIO: float = 4.0
    DIT_DROPOUT: float = 0.0
    FM_STEPS: int = 3
    REFINER_TIME_DIM: int = 128


@dataclass
class FMHybridConfig30G(BaselineConfig):
    """FM-Hybrid Refiner Configuration for ~30GB VRAM."""
    REFINER_HIDDEN: int = 96
    REFINER_STEPS: int = 2
    REFINER_TIME_DIM: int = 128
    MIN_NOISE_STD: float = 4e-7
    SIGMA_SCHEDULE: str = 'ddpm'
    ODE_STEPS: int = 10
    NOISE_TYPE: str = 'white'
    NOISE_SPECTRUM_PATH: str = None


# Registry for 30GB configs
BASELINE_CONFIGS_30G = {
    'ufno': UFNOConfig30G(),
    'transolver': TransolverConfig30G(),
    'dpot': DPOTConfig30G(),
    'factformer': FactformerConfig30G(),
    'pde_refiner': PDERefinerConfig30G(),
    'dit_refiner': DiTRefinerConfig30G(),
    'fm_refiner_unet': FMRefinerUNetConfig30G(),
    'fm_refiner_dit': FMRefinerDiTConfig30G(),
    'fm_hybrid': FMHybridConfig30G(),
}


def get_baseline_config_30g(model_name: str) -> BaselineConfig:
    """Get 30GB-optimized configuration for a baseline model."""
    if model_name not in BASELINE_CONFIGS_30G:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(BASELINE_CONFIGS_30G.keys())}")
    return BASELINE_CONFIGS_30G[model_name]
