"""
2D UNet Baseline — slice-wise K=0 prediction for 3D turbulence.

Predicts each xy z-slice independently:
- Input (B, T=5, C=4, Z=128, H=128, W=128) is reshaped by merging Z into the
  batch dim, producing (B*Z, 20, H, W).
- A 2D UNet predicts each 2D slice independently.
- Output is reshaped back to (B, T=5, C=4, Z, H, W) for downstream loss / eval.

No z-coupling: slices are completely independent. The stacking back to 3D is
only for computing physics consistency (divergence, NS residual, etc.) and for
compatibility with the existing train/eval pipeline.

Architecture mirrors RefinerUNet3D (pre-norm ResBlock + learned down/up conv +
skip concat), but with Conv2d instead of Conv3d. Same interface as the
FM-Hybrid K=0 wrapper so `train_baseline.py` and `evaluate_ar.py` work without
changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops import rearrange

from .pde_refiner_model import zero_module, fourier_embedding


# ============================================================================
# 2D building blocks — direct analogs of the 3D ones
# ============================================================================

class ResidualBlock2D(nn.Module):
    """Pre-norm residual block with time conditioning (2D Conv)."""

    def __init__(self, in_channels, out_channels, cond_channels, n_groups=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = zero_module(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.act = nn.GELU()
        self.cond_emb = nn.Linear(cond_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class Downsample2D(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class MiddleBlock2D(nn.Module):
    def __init__(self, n_channels, cond_channels, n_groups=1):
        super().__init__()
        self.res1 = ResidualBlock2D(n_channels, n_channels, cond_channels, n_groups)
        self.res2 = ResidualBlock2D(n_channels, n_channels, cond_channels, n_groups)

    def forward(self, x, emb):
        x = self.res1(x, emb)
        x = self.res2(x, emb)
        return x


# ============================================================================
# RefinerUNet2D — 2D analog of RefinerUNet3D
# ============================================================================

class RefinerUNet2D(nn.Module):
    """2D U-Net with the same encoder/middle/decoder structure as RefinerUNet3D."""

    def __init__(self, in_channels, cond_channels, out_channels,
                 hidden_channels=128, ch_mults=(1, 2, 2, 4), n_blocks=2,
                 n_groups=1, gradient_checkpointing=False):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.gradient_checkpointing = gradient_checkpointing
        n_resolutions = len(ch_mults)

        time_embed_dim = hidden_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.fourier_dim = hidden_channels

        total_in = in_channels + cond_channels
        self.image_proj = nn.Conv2d(total_in, hidden_channels, kernel_size=3, padding=1)

        # ---- Encoder ----
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [hidden_channels]
        ch_in = hidden_channels
        for i in range(n_resolutions):
            ch_out = hidden_channels * ch_mults[i]
            blocks = nn.ModuleList()
            for j in range(n_blocks):
                blocks.append(ResidualBlock2D(ch_in, ch_out, time_embed_dim, n_groups))
                ch_in = ch_out
                channels.append(ch_out)
            self.down_blocks.append(blocks)
            if i < n_resolutions - 1:
                self.downsamples.append(Downsample2D(ch_out))
                channels.append(ch_out)
            else:
                self.downsamples.append(None)

        # ---- Middle ----
        self.middle = MiddleBlock2D(ch_in, time_embed_dim, n_groups)

        # ---- Decoder ----
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(n_resolutions)):
            ch_out = hidden_channels * ch_mults[i]
            if i < n_resolutions - 1:
                self.upsamples.append(Upsample2D(ch_in))
            else:
                self.upsamples.append(None)

            blocks = nn.ModuleList()
            for j in range(n_blocks + 1):
                skip_ch = channels.pop()
                blocks.append(ResidualBlock2D(ch_in + skip_ch, ch_out, time_embed_dim, n_groups))
                ch_in = ch_out
            self.up_blocks.append(blocks)

        self.final_norm = nn.GroupNorm(n_groups, ch_in)
        self.final_act = nn.GELU()
        self.final_conv = zero_module(nn.Conv2d(ch_in, out_channels, kernel_size=3, padding=1))

    def forward(self, y_noised, condition, timestep):
        """
        Args:
            y_noised: (B, C_out, H, W)
            condition: (B, C_cond, H, W)
            timestep: (B,) float
        Returns:
            (B, C_out, H, W)
        """
        t_emb = fourier_embedding(timestep, self.fourier_dim)
        t_emb = self.time_embed(t_emb)

        h = torch.cat([y_noised, condition], dim=1)
        h = self.image_proj(h)

        use_ckpt = self.gradient_checkpointing and self.training

        skips = [h]
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                if use_ckpt:
                    h = torch_checkpoint(block, h, t_emb, use_reentrant=False)
                else:
                    h = block(h, t_emb)
                skips.append(h)
            if downsample is not None:
                h = downsample(h)
                skips.append(h)

        if use_ckpt:
            h = torch_checkpoint(self.middle, h, t_emb, use_reentrant=False)
        else:
            h = self.middle(h, t_emb)

        for blocks, upsample in zip(self.up_blocks, self.upsamples):
            if upsample is not None:
                h = upsample(h)
            for block in blocks:
                skip = skips.pop()
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
                h = torch.cat([h, skip], dim=1)
                if use_ckpt:
                    h = torch_checkpoint(block, h, t_emb, use_reentrant=False)
                else:
                    h = block(h, t_emb)

        h = self.final_conv(self.final_act(self.final_norm(h)))
        return h


# ============================================================================
# UNetBaseline2D — wrapper matching FMHybridRefiner3D interface
# ============================================================================

class UNetBaseline2D(nn.Module):
    """
    Slice-wise 2D UNet baseline for 3D turbulence prediction.

    Internally merges the Z dimension into the batch dimension so the 2D UNet
    processes all 128 z-slices of a sample in a single forward pass. The Z
    dimension is restored before returning so train/eval code sees a 3D tensor.

    Training objective is pure MSE regression (K=0 only — no FM refinement).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        self.total_z = getattr(config, 'TOTAL_Z_LAYERS', 128)

        hidden_channels = getattr(config, 'UNET2D_HIDDEN', 128)
        ch_mults = getattr(config, 'UNET2D_CH_MULTS', (1, 2, 2, 4))
        n_blocks = getattr(config, 'UNET2D_N_BLOCKS', 2)

        self.cond_channels = self.t_in * self.in_channels      # 20
        self.out_channels = self.t_out * self.in_channels      # 20

        self.model = RefinerUNet2D(
            in_channels=self.out_channels,
            cond_channels=self.cond_channels,
            out_channels=self.out_channels,
            hidden_channels=hidden_channels,
            ch_mults=ch_mults,
            n_blocks=n_blocks,
            gradient_checkpointing=True,  # needed for 128-slice effective batch
        )

        # Z-chunking for memory management: process Z slices in chunks instead of all at once
        self.z_chunk = getattr(config, 'UNET2D_Z_CHUNK', 32)

        self._print_info()

    def _print_info(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"\n[UNet Baseline 2D] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.in_channels}, Z, H, W)")
        print(f"  Z merged into batch (effective batch {self.total_z}x per sample)")
        print(f"  Total parameters: {total:,}")

    def _predict_slices(self, input_dense):
        """
        Core 2D forward: run UNet on every z-slice.

        To avoid OOM with effective batch = B*Z=128 on 128x128 slices,
        we process Z in chunks of size `self.z_chunk` (default 32).

        Args:
            input_dense: (B, T_in, C, Z, H, W)
        Returns:
            pred: (B, T_out, C, Z, H, W)
        """
        B, T_in, C, Z, H, W = input_dense.shape

        # Merge time*channel and Z into batch: (B*Z, T*C, H, W)
        x_all = rearrange(input_dense, 'b t c z h w -> (b z) (t c) h w')
        N = x_all.shape[0]  # = B*Z

        chunk = self.z_chunk if self.z_chunk else N
        pred_chunks = []
        for i in range(0, N, chunk):
            x_chunk = x_all[i:i+chunk]
            zeros_in = torch.zeros(x_chunk.shape[0], self.out_channels, H, W,
                                   dtype=x_chunk.dtype, device=x_chunk.device)
            t_zero = torch.zeros(x_chunk.shape[0], dtype=x_chunk.dtype, device=x_chunk.device)
            pred_chunk = self.model(zeros_in, x_chunk, t_zero)
            pred_chunks.append(pred_chunk)
        pred_2d = torch.cat(pred_chunks, dim=0)  # (B*Z, T_out*C, H, W)

        pred = rearrange(pred_2d, '(b z) (t c) h w -> b t c z h w',
                         b=B, z=Z, t=self.t_out, c=self.in_channels)
        return pred

    def get_training_loss(self, input_dense, output_dense):
        """
        Pure MSE regression loss. Matches CustomMSELoss reduction used by
        FMHybridRefiner3D at K=0 (avg spatial, sum T*C, avg batch) so RMSE
        numbers are directly comparable with the 3D baseline.
        """
        pred = self._predict_slices(input_dense)  # (B, T, C, Z, H, W)

        # Collapse T*C for the CustomMSELoss reduction
        pred_flat = rearrange(pred, 'b t c z h w -> b (t c) z h w')
        gt_flat = rearrange(output_dense, 'b t c z h w -> b (t c) z h w')

        loss = F.mse_loss(pred_flat, gt_flat, reduction='none')
        loss = loss.mean(dim=tuple(range(2, loss.ndim)))  # avg spatial
        loss = loss.sum(dim=1)                            # sum T*C
        loss = loss.mean()                                # avg batch
        return loss

    def forward(self, input_dense, global_timesteps=None,
                num_refine_steps=None, ode_steps=None):
        """Inference: slice-wise K=0 prediction. Extra kwargs are ignored."""
        return self._predict_slices(input_dense)
