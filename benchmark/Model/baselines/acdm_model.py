"""
ACDM: Autoregressive Conditional Diffusion Model (3D adaptation)

Reference: Lippe et al., "Benchmarking Autoregressive Conditional Diffusion Models
           for Turbulent Flow Prediction", Neural Networks 2025
Paper: https://arxiv.org/abs/2309.01745
GitHub: https://github.com/tum-pbs/autoreg-pde-diffusion

Key design (faithful to original):
- ConvNext U-Net backbone (no attention — removed for 3D memory)
- Cosine beta schedule, 100 diffusion steps
- "Clean" conditioning: clean input frames concatenated with noisy target
- smooth_l1_loss on noise prediction
- DDPM reverse sampling at inference

3D adaptation changes:
- All Conv2d → Conv3d (kernel 7→7³, 3→3³, etc.)
- Attention layers removed (2M tokens at 128³ infeasible)
- dim_mults adjusted from (1,1,1) to (1,2,2) for 3D memory
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops import rearrange


# =============================================================================
# Beta schedules (from model_diffusion_blocks.py)
# =============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    """Linear schedule with reference adjustment for 500 steps."""
    beta_start = 0.0001 * (500 / timesteps)
    beta_end = 0.02 * (500 / timesteps)
    return torch.linspace(beta_start, beta_end, timesteps)


# =============================================================================
# Sinusoidal position embeddings (from model_diffusion_blocks.py)
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings matching ACDM original."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# =============================================================================
# ConvNextBlock3D — faithful port of ACDM's ConvNextBlock to 3D
# =============================================================================

class ConvNextBlock3D(nn.Module):
    """
    ConvNext block for 3D, ported from ACDM model_diffusion_blocks.py.

    Structure: depthwise_conv → time_emb → GroupNorm → expand_conv → GELU → GroupNorm → contract_conv + residual
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if time_emb_dim is not None else None
        )

        # Depthwise separable conv (7³ kernel, matching original's 7×7)
        self.ds_conv = nn.Conv3d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv3d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv3d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if self.mlp is not None and time_emb is not None:
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


# =============================================================================
# Down/Upsample (matching ACDM's strided conv approach)
# =============================================================================

def Downsample3D(dim):
    """Strided conv downsample, matching ACDM Conv2d(dim, dim, 4, 2, 1)."""
    return nn.Conv3d(dim, dim, 4, 2, 1)


def Upsample3D(dim):
    """Transposed conv upsample, matching ACDM ConvTranspose2d(dim, dim, 4, 2, 1)."""
    return nn.ConvTranspose3d(dim, dim, 4, 2, 1)


# =============================================================================
# ACDMUnet3D — 3D U-Net with ConvNext blocks (no attention)
# =============================================================================

class ACDMUnet3D(nn.Module):
    """
    3D U-Net with ConvNext blocks, ported from ACDM's Unet class.

    Changes from original:
    - All Conv2d → Conv3d
    - Attention layers removed (infeasible at 128³)
    - dim_mults adjusted for 3D memory constraints
    """
    def __init__(
        self,
        dim=64,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 2),
        channels=40,
        convnext_mult=2,
        gradient_checkpointing=True,
    ):
        super().__init__()
        self.channels = channels
        self.gradient_checkpointing = gradient_checkpointing

        init_dim = init_dim if init_dim is not None else dim // 3 * 2
        self.init_conv = nn.Conv3d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embeddings (matching ACDM: dim * 4)
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder (down path)
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock3D(dim_in, dim_out, time_emb_dim=time_dim, mult=convnext_mult),
                ConvNextBlock3D(dim_out, dim_out, time_emb_dim=time_dim, mult=convnext_mult),
                Downsample3D(dim_out) if not is_last else nn.Identity(),
            ]))

        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock3D(mid_dim, mid_dim, time_emb_dim=time_dim, mult=convnext_mult)
        self.mid_block2 = ConvNextBlock3D(mid_dim, mid_dim, time_emb_dim=time_dim, mult=convnext_mult)

        # Decoder (up path)
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ConvNextBlock3D(dim_out * 2, dim_in, time_emb_dim=time_dim, mult=convnext_mult),
                ConvNextBlock3D(dim_in, dim_in, time_emb_dim=time_dim, mult=convnext_mult),
                Upsample3D(dim_in) if not is_last else nn.Identity(),
            ]))

        out_dim = out_dim if out_dim is not None else channels
        self.final_conv = nn.Sequential(
            ConvNextBlock3D(dim, dim, mult=convnext_mult),
            nn.Conv3d(dim, out_dim, 1),
        )

    def _forward_block(self, block, x, t):
        """Forward through a ConvNextBlock3D with optional gradient checkpointing."""
        if self.gradient_checkpointing and self.training:
            return torch_checkpoint(block, x, t, use_reentrant=False)
        return block(x, t)

    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time)

        h = []

        # Downsample
        for block1, block2, downsample in self.downs:
            x = self._forward_block(block1, x, t)
            x = self._forward_block(block2, x, t)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self._forward_block(self.mid_block1, x, t)
        x = self._forward_block(self.mid_block2, x, t)

        # Upsample
        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = self._forward_block(block1, x, t)
            x = self._forward_block(block2, x, t)
            x = upsample(x)

        return self.final_conv(x)


# =============================================================================
# ACDM3D — full model with DDPM training and inference
# =============================================================================

class ACDM3D(nn.Module):
    """
    Autoregressive Conditional Diffusion Model adapted for 3D turbulence.

    Training: noise prediction with smooth_l1_loss, "clean" conditioning
    Inference: DDPM reverse sampling (100 steps)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dimensions
        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        self.total_z = getattr(config, 'TOTAL_Z_LAYERS', 128)

        # ACDM parameters
        dim = getattr(config, 'ACDM_DIM', 64)
        dim_mults = getattr(config, 'ACDM_DIM_MULTS', (1, 2, 2))
        convnext_mult = getattr(config, 'ACDM_CONVNEXT_MULT', 2)
        self.timesteps = getattr(config, 'ACDM_DIFF_STEPS', 100)
        schedule = getattr(config, 'ACDM_SCHEDULE', 'cosine')
        self.cond_mode = getattr(config, 'ACDM_COND_MODE', 'clean')

        # Channel dimensions
        self.cond_channels = self.t_in * self.in_channels   # 5*4 = 20
        self.data_channels = self.t_out * self.in_channels   # 5*4 = 20
        total_channels = self.cond_channels + self.data_channels  # 40

        # Build noise schedule
        if schedule == 'cosine':
            betas = cosine_beta_schedule(self.timesteps)
        elif schedule == 'linear':
            betas = linear_beta_schedule(self.timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers (matching ACDM's buffer shapes)
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_posterior_variance',
                             torch.sqrt(betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)))

        # Build U-Net
        self.unet = ACDMUnet3D(
            dim=dim,
            dim_mults=dim_mults,
            channels=total_channels,
            convnext_mult=convnext_mult,
            gradient_checkpointing=True,
        )

        self._print_info()

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[ACDM 3D] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.in_channels}, Z, H, W)")
        print(f"  Diffusion steps: {self.timesteps}")
        print(f"  Conditioning mode: {self.cond_mode}")
        print(f"  Total parameters: {total_params:,}")

    def _expand_t(self, vals, t, target):
        """Expand schedule values to match target tensor shape."""
        out = vals[t]
        while len(out.shape) < len(target.shape):
            out = out.unsqueeze(-1)
        return out

    def get_training_loss(self, input_dense, output_dense):
        """
        ACDM training loss with "clean" conditioning mode.

        Faithful to model_diffusion.py lines 98-104:
        - Only noise the target data (NOT the conditioning)
        - noise_target = concat([conditioning, data_noise])
        - model_input = concat([conditioning, noisy_data])
        - loss = smooth_l1_loss(noise_target, predicted_noise)

        Args:
            input_dense: [B, T_in, C, Z, H, W]
            output_dense: [B, T_out, C, Z, H, W]
        Returns:
            loss: scalar
        """
        B = input_dense.shape[0]
        device = input_dense.device

        # Flatten timesteps into channels
        cond = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')    # (B, 20, Z, H, W)
        target = rearrange(output_dense, 'b t c z h w -> b (t c) z h w')  # (B, 20, Z, H, W)

        # Sample random timestep
        t = torch.randint(0, self.timesteps, (B,), device=device).long()

        # "clean" mode: only noise the target, keep conditioning clean
        data_noise = torch.randn_like(target)
        sqrt_abar = self._expand_t(self.sqrt_alphas_cumprod, t, target)
        sqrt_1m_abar = self._expand_t(self.sqrt_one_minus_alphas_cumprod, t, target)
        target_noisy = sqrt_abar * target + sqrt_1m_abar * data_noise

        # UNet input: [clean_cond, noisy_target]
        model_input = torch.cat([cond, target_noisy], dim=1)

        # Predict noise (output has same channels as input: 40)
        predicted_noise = self.unet(model_input, t)

        # Loss only on DATA channels (last 20), not conditioning channels
        # This focuses the model on learning noise prediction rather than
        # wasting capacity on reconstructing the clean conditioning signal
        pred_data = predicted_noise[:, self.cond_channels:]
        loss = F.smooth_l1_loss(pred_data, data_noise)
        return loss

    @torch.no_grad()
    def forward(self, input_dense, global_timesteps=None):
        """
        DDPM reverse sampling inference.

        Faithful to model_diffusion.py lines 120-161:
        - Start from random noise for data channels
        - 100 reverse steps with DDPM posterior sampling
        - "clean" conditioning at each step

        Args:
            input_dense: [B, T_in, C, Z, H, W]
        Returns:
            output: [B, T_out, C, Z, H, W]
        """
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = self.t_out
        device = input_dense.device

        # Flatten conditioning
        cond = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')  # (B, 20, Z, H, W)

        # Start from random noise for data channels
        data_noise = torch.randn(B, self.data_channels, Z, H, W,
                                 dtype=input_dense.dtype, device=device)

        # Reverse diffusion: DDPM on data channels only, cond stays clean
        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            # UNet sees [clean_cond, noisy_data]
            model_input = torch.cat([cond, data_noise], dim=1)
            predicted_noise = self.unet(model_input, t)

            # Extract noise prediction for DATA channels only
            pred_noise_data = predicted_noise[:, self.cond_channels:]

            # DDPM mean on data channels only
            sqrt_recip_a = self._expand_t(self.sqrt_recip_alphas, t, data_noise)
            beta = self._expand_t(self.betas, t, data_noise)
            sqrt_1m_abar = self._expand_t(self.sqrt_one_minus_alphas_cumprod, t, data_noise)

            data_noise = sqrt_recip_a * (data_noise - beta * pred_noise_data / sqrt_1m_abar)

            # Add posterior noise (except at final step)
            if i != 0:
                sqrt_post_var = self._expand_t(self.sqrt_posterior_variance, t, data_noise)
                data_noise = data_noise + sqrt_post_var * torch.randn_like(data_noise)

        # Reshape to output format
        output = rearrange(data_noise, 'b (t c) z h w -> b t c z h w', t=T_out, c=C)
        return output
