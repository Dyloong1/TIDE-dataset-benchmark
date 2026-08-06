"""
Latent Diffusion Model for 3D Turbulence Prediction (CoNFiLD-style)

Inspired by: CoNFiLD — Conditional Neural Field Latent Diffusion
Reference: Qu et al., Nature Communications 15, 10416 (2024)
Paper: https://www.nature.com/articles/s41467-024-54712-1

Architecture:
- 3D Convolutional Autoencoder: 128³ → 16³ latent → 128³
- Latent DDPM: standard DDPM in compressed 16³ latent space
- Two-stage training: (1) train AE, (2) freeze AE + train latent DDPM

Adaptation from CoNFiLD:
- CoNFiLD uses coordinate-based neural fields for irregular meshes
- We use standard 3D conv autoencoder since our data is on regular grids
- Core idea preserved: compress to latent space, diffuse in latent space
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops import rearrange

from .pde_refiner_model import RefinerUNet3D


# =============================================================================
# Beta schedule (shared with ACDM)
# =============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# =============================================================================
# ResBlock3D for Autoencoder (no time conditioning)
# =============================================================================

class AEResBlock3D(nn.Module):
    """Residual block for autoencoder (no time embedding)."""
    def __init__(self, in_channels, out_channels, n_groups=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


# =============================================================================
# 3D Convolutional Encoder: 128³ → 16³
# =============================================================================

class ConvEncoder3D(nn.Module):
    """
    3D convolutional encoder with 3 downsample levels.

    128³ → 64³ → 32³ → 16³
    Channels: input → enc_channels[0] → enc_channels[1] → enc_channels[2] → latent_ch
    """
    def __init__(self, in_channels, latent_ch, enc_channels=(64, 128, 256),
                 gradient_checkpointing=True):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        # Input projection
        self.input_conv = nn.Conv3d(in_channels, enc_channels[0], 3, padding=1)

        # Downsample levels
        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch_in = enc_channels[0]
        for ch_out in enc_channels:
            self.levels.append(nn.Sequential(
                AEResBlock3D(ch_in, ch_out),
                AEResBlock3D(ch_out, ch_out),
            ))
            self.downsamples.append(nn.Conv3d(ch_out, ch_out, 4, stride=2, padding=1))
            ch_in = ch_out

        # Final projection to latent
        self.output_conv = nn.Sequential(
            nn.GroupNorm(1, enc_channels[-1]),
            nn.SiLU(),
            nn.Conv3d(enc_channels[-1], latent_ch, 3, padding=1),
        )

    def forward(self, x):
        x = self.input_conv(x)
        for level, down in zip(self.levels, self.downsamples):
            if self.gradient_checkpointing and self.training:
                x = torch_checkpoint(level, x, use_reentrant=False)
            else:
                x = level(x)
            x = down(x)
        x = self.output_conv(x)
        return x


# =============================================================================
# 3D Convolutional Decoder: 16³ → 128³
# =============================================================================

class ConvDecoder3D(nn.Module):
    """
    3D convolutional decoder with 3 upsample levels.

    16³ → 32³ → 64³ → 128³
    Channels: latent_ch → dec_channels[-1] → ... → dec_channels[0] → output
    """
    def __init__(self, latent_ch, out_channels, dec_channels=(256, 128, 64),
                 gradient_checkpointing=True):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        # Input projection from latent
        self.input_conv = nn.Sequential(
            nn.Conv3d(latent_ch, dec_channels[0], 3, padding=1),
            nn.SiLU(),
        )

        # Upsample levels
        self.upsamples = nn.ModuleList()
        self.levels = nn.ModuleList()

        ch_in = dec_channels[0]
        for ch_out in dec_channels:
            self.upsamples.append(nn.ConvTranspose3d(ch_in, ch_out, 4, stride=2, padding=1))
            self.levels.append(nn.Sequential(
                AEResBlock3D(ch_out, ch_out),
                AEResBlock3D(ch_out, ch_out),
            ))
            ch_in = ch_out

        # Final projection to output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(1, dec_channels[-1]),
            nn.SiLU(),
            nn.Conv3d(dec_channels[-1], out_channels, 3, padding=1),
        )

    def forward(self, z):
        x = self.input_conv(z)
        for up, level in zip(self.upsamples, self.levels):
            x = up(x)
            if self.gradient_checkpointing and self.training:
                x = torch_checkpoint(level, x, use_reentrant=False)
            else:
                x = level(x)
        x = self.output_conv(x)
        return x


# =============================================================================
# LatentDiffusion3D — full model
# =============================================================================

class LatentDiffusion3D(nn.Module):
    """
    Latent Diffusion Model for 3D turbulence prediction.

    Two-stage training:
    - Stage 'ae': train autoencoder (MSE reconstruction)
    - Stage 'ldm': freeze AE, train latent DDPM

    Inference: encode input → DDPM reverse in latent space → decode
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dimensions
        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))

        # Latent diffusion parameters
        self.latent_ch = getattr(config, 'LD_LATENT_CH', 8)
        enc_channels = getattr(config, 'LD_ENC_CHANNELS', (64, 128, 256))
        self.timesteps = getattr(config, 'LD_DIFF_STEPS', 100)
        schedule = getattr(config, 'LD_SCHEDULE', 'cosine')
        ddpm_hidden = getattr(config, 'LD_DDPM_HIDDEN', 64)
        ddpm_ch_mults = getattr(config, 'LD_DDPM_CH_MULTS', (1, 2, 2))

        # Channel dimensions
        self.flat_channels = self.t_out * self.in_channels  # 20
        self.cond_channels_flat = self.t_in * self.in_channels  # 20

        # Stage tracking
        self.stage = getattr(config, 'LD_STAGE', 'ae')

        # Autoencoder
        self.encoder = ConvEncoder3D(
            in_channels=self.flat_channels,
            latent_ch=self.latent_ch,
            enc_channels=enc_channels,
            gradient_checkpointing=True,
        )
        self.decoder = ConvDecoder3D(
            latent_ch=self.latent_ch,
            out_channels=self.flat_channels,
            dec_channels=tuple(reversed(enc_channels)),
            gradient_checkpointing=True,
        )

        # Latent DDPM (small U-Net operating at 16³)
        self.latent_unet = RefinerUNet3D(
            in_channels=self.latent_ch,
            cond_channels=self.latent_ch,  # encoded condition
            out_channels=self.latent_ch,
            hidden_channels=ddpm_hidden,
            ch_mults=ddpm_ch_mults,
            n_blocks=2,
            gradient_checkpointing=False,  # 16³ is small enough
        )

        # Fourier dim for time embedding in latent_unet
        self.time_multiplier = 1000.0

        # Build noise schedule
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        self._print_info()

    def _print_info(self):
        ae_params = sum(p.numel() for p in self.encoder.parameters()) + \
                    sum(p.numel() for p in self.decoder.parameters())
        ddpm_params = sum(p.numel() for p in self.latent_unet.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[Latent Diffusion 3D] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.in_channels}, Z, H, W)")
        print(f"  Latent: {self.latent_ch} channels at 16³")
        print(f"  Diffusion steps: {self.timesteps}")
        print(f"  Stage: {self.stage}")
        print(f"  AE parameters: {ae_params:,}")
        print(f"  Latent DDPM parameters: {ddpm_params:,}")
        print(f"  Total parameters: {total_params:,}")

    def set_stage(self, stage):
        """Switch between 'ae' and 'ldm' training stages."""
        self.stage = stage
        if stage == 'ldm':
            # Freeze autoencoder
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            self.decoder.eval()
            print("[Latent Diffusion] Stage 'ldm': AE frozen, training latent DDPM only")
        elif stage == 'ae':
            # Unfreeze autoencoder
            for p in self.encoder.parameters():
                p.requires_grad = True
            for p in self.decoder.parameters():
                p.requires_grad = True
            print("[Latent Diffusion] Stage 'ae': training autoencoder")

    def _expand_t(self, vals, t, target):
        """Expand schedule values to match target shape."""
        out = vals[t]
        while len(out.shape) < len(target.shape):
            out = out.unsqueeze(-1)
        return out

    def get_training_loss(self, input_dense, output_dense):
        """
        Training loss depends on stage:
        - 'ae': MSE reconstruction loss on both input and output
        - 'ldm': DDPM noise prediction loss in latent space

        Args:
            input_dense: [B, T_in, C, Z, H, W]
            output_dense: [B, T_out, C, Z, H, W]
        Returns:
            loss: scalar
        """
        B = input_dense.shape[0]
        device = input_dense.device

        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')
        y = rearrange(output_dense, 'b t c z h w -> b (t c) z h w')

        if self.stage == 'ae':
            # Autoencoder reconstruction loss
            z_x = self.encoder(x)
            z_y = self.encoder(y)
            x_recon = self.decoder(z_x)
            y_recon = self.decoder(z_y)
            loss = F.mse_loss(x_recon, x) + F.mse_loss(y_recon, y)
            return loss

        elif self.stage == 'ldm':
            # Latent DDPM training
            with torch.no_grad():
                z_cond = self.encoder(x)    # (B, latent_ch, 16, 16, 16)
                z_target = self.encoder(y)  # (B, latent_ch, 16, 16, 16)

            # Sample random timestep
            t = torch.randint(0, self.timesteps, (B,), device=device).long()

            # Add noise to target latent
            noise = torch.randn_like(z_target)
            sqrt_abar = self._expand_t(self.sqrt_alphas_cumprod, t, z_target)
            sqrt_1m_abar = self._expand_t(self.sqrt_one_minus_alphas_cumprod, t, z_target)
            z_noisy = sqrt_abar * z_target + sqrt_1m_abar * noise

            # Predict noise (latent U-Net conditioned on encoded input)
            t_emb = t.float() * self.time_multiplier
            noise_pred = self.latent_unet(z_noisy, z_cond, t_emb)

            # MSE loss on noise prediction
            loss = F.mse_loss(noise_pred, noise)
            return loss

        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    def forward(self, input_dense, global_timesteps=None):
        """
        Inference:
        - 'ae' stage: return reconstruction (for AE validation)
        - 'ldm' stage: encode → DDPM reverse in latent → decode

        Args:
            input_dense: [B, T_in, C, Z, H, W]
        Returns:
            output: [B, T_out, C, Z, H, W]
        """
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = self.t_out
        device = input_dense.device

        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')

        if self.stage == 'ae':
            # AE reconstruction for validation
            z = self.encoder(x)
            recon = self.decoder(z)
            output = rearrange(recon, 'b (t c) z h w -> b t c z h w', t=T_out, c=C)
            return output

        # LDM inference: encode condition → DDPM reverse → decode
        with torch.no_grad():
            z_cond = self.encoder(x)  # (B, latent_ch, 16, 16, 16)

        # Start from noise in latent space
        z_shape = z_cond.shape
        z = torch.randn(z_shape, dtype=input_dense.dtype, device=device)

        # DDPM reverse sampling in latent space (very fast at 16³)
        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            t_emb = t.float() * self.time_multiplier

            # Predict noise
            noise_pred = self.latent_unet(z, z_cond, t_emb)

            # DDPM mean
            sqrt_recip_a = self._expand_t(self.sqrt_recip_alphas, t, z)
            beta = self._expand_t(self.betas, t, z)
            sqrt_1m_abar = self._expand_t(self.sqrt_one_minus_alphas_cumprod, t, z)

            mean = sqrt_recip_a * (z - beta * noise_pred / sqrt_1m_abar)

            # Posterior sampling
            if i > 0:
                post_var = self._expand_t(self.posterior_variance, t, z)
                z = mean + torch.sqrt(post_var.clamp(min=1e-20)) * torch.randn_like(z)
            else:
                z = mean

        # Decode latent to output
        y_pred = self.decoder(z)
        output = rearrange(y_pred, 'b (t c) z h w -> b t c z h w', t=T_out, c=C)
        return output
