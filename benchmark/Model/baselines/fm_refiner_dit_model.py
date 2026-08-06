"""
FM-Refiner DiT: Flow Matching with Diffusion Transformer backbone.

Combines the DiT3D backbone (global self-attention) with Flow Matching
(straight interpolation paths + Euler ODE integration).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dit_refiner_model import DiT3D
from .fm_scheduler import FMScheduler


class FMRefinerDiT3D(nn.Module):
    """
    Flow Matching Refiner with DiT3D backbone.

    Same interface as PDERefiner3D but uses DiT backbone + Flow Matching.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dimensions
        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        self.total_z = getattr(config, 'TOTAL_Z_LAYERS', 128)

        # DiT parameters
        hidden_dim = getattr(config, 'DIT_HIDDEN_DIM', 512)
        num_heads = getattr(config, 'DIT_NUM_HEADS', 8)
        num_layers = getattr(config, 'DIT_NUM_LAYERS', 8)
        patch_size = getattr(config, 'DIT_PATCH_SIZE', (4, 8, 8))
        mlp_ratio = getattr(config, 'DIT_MLP_RATIO', 4.0)
        dropout = getattr(config, 'DIT_DROPOUT', 0.0)

        # FM parameters
        self.fm_steps = getattr(config, 'FM_STEPS', 3)
        self.time_dim = getattr(config, 'REFINER_TIME_DIM', 128)

        # Channel dimensions
        self.cond_channels = self.t_in * self.in_channels
        self.out_channels = self.t_out * self.in_channels

        # FM scheduler
        self.scheduler = FMScheduler(num_steps=self.fm_steps)

        # Spatial dimensions for pos_embed initialization
        h_size = getattr(config, 'H_SIZE', 128)
        w_size = getattr(config, 'W_SIZE', 128)
        input_size = (self.total_z, h_size, w_size)

        # DiT backbone
        self.model = DiT3D(
            in_channels=self.out_channels,
            cond_channels=self.cond_channels,
            out_channels=self.out_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            patch_size=patch_size,
            time_dim=self.time_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            input_size=input_size,
        )

        self._print_info()

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[FM-Refiner DiT] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.in_channels}, Z, H, W)")
        print(f"  FM steps: {self.fm_steps}")
        print(f"  Patch size: {self.model.patch_size}")
        print(f"  Total parameters: {total_params:,}")

    def get_training_loss(self, input_dense, output_dense):
        """Compute Flow Matching velocity matching loss."""
        B, T_in, C, Z, H, W = input_dense.shape
        device = input_dense.device

        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')
        y = rearrange(output_dense, 'b t c z h w -> b (t c) z h w')

        # Get FM training tuple
        x_t, t, v_target = self.scheduler.get_train_tuple(y)

        # Network prediction: velocity
        # Scale t by 1000 for sinusoidal embedding
        v_pred = self.model(x_t, x, t * 1000.0)

        loss = F.mse_loss(v_pred, v_target)
        return loss

    def forward(self, input_dense, global_timesteps=None):
        """Inference: Euler ODE integration from noise to prediction."""
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = self.t_out
        device = input_dense.device

        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')

        # Start from noise
        y = torch.randn(B, self.out_channels, Z, H, W, dtype=input_dense.dtype, device=device)

        # Euler integration
        for i in range(self.fm_steps):
            t_i = torch.full((B,), i / self.fm_steps, device=device)
            v_pred = self.model(y, x, t_i * 1000.0)
            y = self.scheduler.euler_step(y, v_pred, self.scheduler.dt)

        output = rearrange(y, 'b (t c) z h w -> b t c z h w', t=T_out, c=C)

        return output
