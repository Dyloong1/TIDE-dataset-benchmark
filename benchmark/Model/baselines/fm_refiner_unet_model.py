"""
FM-Refiner U-Net: Flow Matching with 3D U-Net backbone.

Replaces the diffusion-based refinement in PDE-Refiner with
Flow Matching (straight interpolation paths + Euler ODE integration).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .pde_refiner_model import RefinerUNet3D
from .fm_scheduler import FMScheduler


class FMRefinerUNet3D(nn.Module):
    """
    Flow Matching Refiner with 3D U-Net backbone.

    Same interface as PDERefiner3D but uses Flow Matching instead of diffusion.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dimensions
        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        self.total_z = getattr(config, 'TOTAL_Z_LAYERS', 128)

        # FM parameters
        hidden_channels = getattr(config, 'REFINER_HIDDEN', 64)
        self.fm_steps = getattr(config, 'FM_STEPS', 3)
        self.time_dim = getattr(config, 'REFINER_TIME_DIM', 128)

        # Channel dimensions
        self.cond_channels = self.t_in * self.in_channels
        self.out_channels = self.t_out * self.in_channels

        # FM scheduler
        self.scheduler = FMScheduler(num_steps=self.fm_steps)

        # U-Net backbone (reuse RefinerUNet3D from pde_refiner_model)
        self.model = RefinerUNet3D(
            in_channels=self.out_channels,
            cond_channels=self.cond_channels,
            out_channels=self.out_channels,
            hidden_channels=hidden_channels,
            gradient_checkpointing=True,
        )

        self._print_info()

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[FM-Refiner U-Net] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.in_channels}, Z, H, W)")
        print(f"  FM steps: {self.fm_steps}")
        print(f"  Total parameters: {total_params:,}")

    def get_training_loss(self, input_dense, output_dense):
        """
        Compute Flow Matching velocity matching loss.

        Args:
            input_dense: [B, T_in, C, Z, H, W]
            output_dense: [B, T_out, C, Z, H, W] ground truth

        Returns:
            loss: MSE loss on velocity prediction
        """
        B, T_in, C, Z, H, W = input_dense.shape
        device = input_dense.device

        # Prepare condition
        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')
        y = rearrange(output_dense, 'b t c z h w -> b (t c) z h w')

        # Get FM training tuple
        x_t, t, v_target = self.scheduler.get_train_tuple(y)

        # Network prediction: velocity
        # Scale t by 1000 for sinusoidal embedding (same range as diffusion timesteps)
        v_pred = self.model(x_t, x, t * 1000.0)

        # MSE loss on velocity
        loss = F.mse_loss(v_pred, v_target)
        return loss

    def forward(self, input_dense, global_timesteps=None):
        """
        Inference: Euler ODE integration from noise to prediction.

        Args:
            input_dense: [B, T_in, C, Z, H, W]

        Returns:
            output: [B, T_out, C, Z, H, W]
        """
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = self.t_out
        device = input_dense.device

        # Prepare condition
        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')

        # Start from noise
        y = torch.randn(B, self.out_channels, Z, H, W, dtype=input_dense.dtype, device=device)

        # Euler integration
        for i in range(self.fm_steps):
            t_i = torch.full((B,), i / self.fm_steps, device=device)
            v_pred = self.model(y, x, t_i * 1000.0)
            y = self.scheduler.euler_step(y, v_pred, self.scheduler.dt)

        # Reshape output
        output = rearrange(y, 'b (t c) z h w -> b t c z h w', t=T_out, c=C)

        return output
