"""
U-FNO: U-Net Enhanced Fourier Neural Operator

Combines FNO's spectral convolution with U-Net's multi-scale architecture.
Reference: https://github.com/neuraloperator/neuraloperator

Key improvements over vanilla FNO:
1. Multi-scale feature extraction via downsampling/upsampling
2. Skip connections preserve high-frequency details
3. Better gradient flow for deeper networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import torch.utils.checkpoint
from einops import rearrange
import math


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution Layer (same as FNO)."""

    def __init__(self, in_channels, out_channels, modes_z, modes_h, modes_w):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_z = modes_z
        self.modes_h = modes_h
        self.modes_w = modes_w

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_z, modes_h, modes_w, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_z, modes_h, modes_w, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_z, modes_h, modes_w, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_z, modes_h, modes_w, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bizxy,iozxy->bozxy", input, weights)

    def forward(self, x):
        B, C, Z, H, W = x.shape
        orig_dtype = x.dtype          # preserve input dtype (AMP/half safety), like fno3d
        # Complex FFT doesn't support half precision, cast to float32
        x_float = x.float()
        x_ft = fft.rfftn(x_float, dim=[-3, -2, -1])
        Wf = W // 2 + 1

        out_ft = torch.zeros(
            B, self.out_channels, Z, H, Wf,
            dtype=torch.cfloat, device=x.device
        )

        # Clamp modes to the ACTUAL rfft extent of this (possibly down-sampled) tensor.
        # A fixed mode count (e.g. floored at 4) can exceed the spectral extent of a
        # small/deep-U-Net grid (e.g. 4^3 -> rfft W=3 < modes_w=4), which slices a
        # smaller block than the weights and crashes the einsum. Clamp keeps it safe
        # at any resolution; at the 256^3 production grid the modes are unchanged.
        #
        # For the FULL-spectrum axes Z, H we keep a LOW block [:m] and a HIGH block [-m:]
        # (positive and negative frequencies). These MUST NOT overlap, so m is capped at
        # HALF the axis (Z//2 / H//2); otherwise [:m] and [-m:] would double-write out_ft
        # with the wrong weights on a small grid. W is an rfft axis (non-negative freqs
        # only, single [:mw] block), so it is capped at the rfft extent Wf.
        mz = min(self.modes_z, Z // 2)
        mh = min(self.modes_h, H // 2)
        mw = min(self.modes_w, Wf)

        # If any axis has collapsed below one spectral mode (a degenerate deep-U-Net grid,
        # e.g. Z=1), there is no spectral content to mix — skip the spectral conv (out_ft
        # stays zero; the UFNOBlock's parallel local 1x1 conv still carries the signal).
        if mz >= 1 and mh >= 1 and mw >= 1:
            out_ft[:, :, :mz, :mh, :mw] = \
                self.compl_mul3d(x_ft[:, :, :mz, :mh, :mw], self.weights1[:, :, :mz, :mh, :mw])
            out_ft[:, :, -mz:, :mh, :mw] = \
                self.compl_mul3d(x_ft[:, :, -mz:, :mh, :mw], self.weights2[:, :, :mz, :mh, :mw])
            out_ft[:, :, :mz, -mh:, :mw] = \
                self.compl_mul3d(x_ft[:, :, :mz, -mh:, :mw], self.weights3[:, :, :mz, :mh, :mw])
            out_ft[:, :, -mz:, -mh:, :mw] = \
                self.compl_mul3d(x_ft[:, :, -mz:, -mh:, :mw], self.weights4[:, :, :mz, :mh, :mw])

        x = fft.irfftn(out_ft, s=(Z, H, W))
        return x.to(orig_dtype)       # match input dtype so + local_conv doesn't clash under AMP


class UFNOBlock(nn.Module):
    """U-FNO Block with spectral conv + local conv + skip connection."""

    def __init__(self, in_channels, out_channels, modes_z, modes_h, modes_w):
        super().__init__()
        self.spectral_conv = SpectralConv3d(in_channels, out_channels, modes_z, modes_h, modes_w)
        self.local_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.GELU()

        # Skip connection projection if dimensions change
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)
        x = self.norm(x1 + x2)
        x = self.activation(x + residual)
        return x


class DownBlock(nn.Module):
    """Downsampling block for U-FNO encoder."""

    def __init__(self, in_channels, out_channels, modes_z, modes_h, modes_w):
        super().__init__()
        self.conv = UFNOBlock(in_channels, out_channels, modes_z, modes_h, modes_w)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block for U-FNO decoder with skip connection."""

    def __init__(self, in_channels, skip_channels, out_channels, modes_z, modes_h, modes_w):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = UFNOBlock(in_channels + skip_channels, out_channels, modes_z, modes_h, modes_w)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UFNO3d(nn.Module):
    """
    U-FNO: U-Net Enhanced Fourier Neural Operator for 3D turbulence prediction.

    Architecture:
    - Encoder: Downsampling path with spectral convolutions
    - Bottleneck: Deep spectral processing at low resolution
    - Decoder: Upsampling path with skip connections

    Input: (B, T_in=5, C=4, Z, H, W)
    Output: (B, T_out=5, C=4, Z, H, W)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dimensions
        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        # in/out channels: honor OUT_CHANNELS != IN_CHANNELS (P4 pressure 3->1, P5 sgs 4->6),
        # same contract as fno3d/tfno/deeponet. Falls back to len(INDICATORS) for same-channel tasks.
        self.in_channels = getattr(config, 'IN_CHANNELS', None) or len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        self.out_channels = getattr(config, 'OUT_CHANNELS', None) or self.in_channels
        self.total_z = getattr(config, 'TOTAL_Z_LAYERS', 128)
        self.grad_ckpt = bool(getattr(config, 'GRAD_CKPT', False))

        # U-FNO parameters
        base_width = getattr(config, 'UFNO_WIDTH', 32)
        self.modes_z = getattr(config, 'FNO_MODES_Z', 16)
        self.modes_h = getattr(config, 'FNO_MODES_H', 24)
        self.modes_w = getattr(config, 'FNO_MODES_W', 24)

        # Encoder/Decoder channel progression
        self.enc_channels = [base_width, base_width*2, base_width*4]
        self.dec_channels = [base_width*4, base_width*2, base_width]

        # Lifting
        lifting_in = self.t_in * self.in_channels
        self.lifting = nn.Sequential(
            nn.Conv3d(lifting_in, base_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(base_width, base_width, kernel_size=1)
        )

        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        in_ch = base_width
        for out_ch in self.enc_channels:
            modes_z = max(4, self.modes_z // (2 ** len(self.encoder)))
            modes_h = max(4, self.modes_h // (2 ** len(self.encoder)))
            modes_w = max(4, self.modes_w // (2 ** len(self.encoder)))
            self.encoder.append(DownBlock(in_ch, out_ch, modes_z, modes_h, modes_w))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            UFNOBlock(self.enc_channels[-1], self.enc_channels[-1], 4, 4, 4),
            UFNOBlock(self.enc_channels[-1], self.enc_channels[-1], 4, 4, 4),
        )

        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        in_ch = self.enc_channels[-1]
        for i, out_ch in enumerate(self.dec_channels):
            skip_ch = self.enc_channels[-(i+1)]
            modes_z = max(4, self.modes_z // (2 ** (len(self.dec_channels) - i - 1)))
            modes_h = max(4, self.modes_h // (2 ** (len(self.dec_channels) - i - 1)))
            modes_w = max(4, self.modes_w // (2 ** (len(self.dec_channels) - i - 1)))
            self.decoder.append(UpBlock(in_ch, skip_ch, out_ch, modes_z, modes_h, modes_w))
            in_ch = out_ch

        # Projection (output uses out_channels, may differ from in_channels)
        projection_out = self.t_out * self.out_channels
        self.projection = nn.Sequential(
            nn.Conv3d(base_width, base_width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(base_width * 2, projection_out, kernel_size=1)
        )

        self._print_info()

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[U-FNO] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.out_channels}, Z, H, W)")
        print(f"  Encoder channels: {self.enc_channels}")
        print(f"  Decoder channels: {self.dec_channels}")
        print(f"  Total parameters: {total_params:,}")

    def forward(self, input_dense, global_timesteps=None):
        """
        Forward pass.
        """
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = self.t_out

        # Reshape: (B, T*C, Z, H, W)
        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')

        # Lifting
        x = self.lifting(x)

        # Encoder. Multi-scale U-Net activations are the largest of the FNO family (6.8GB at
        # 128^3, ~8x at native 256^3) — recompute per block when GRAD_CKPT is set.
        ckpt_on = self.grad_ckpt and self.training
        skips = []
        for enc_block in self.encoder:
            if ckpt_on:
                x, skip = torch.utils.checkpoint.checkpoint(enc_block, x, use_reentrant=False)
            else:
                x, skip = enc_block(x)
            skips.append(skip)

        # Bottleneck
        if ckpt_on:
            x = torch.utils.checkpoint.checkpoint(self.bottleneck, x, use_reentrant=False)
        else:
            x = self.bottleneck(x)

        # Decoder
        for i, dec_block in enumerate(self.decoder):
            skip = skips[-(i+1)]
            if ckpt_on:
                x = torch.utils.checkpoint.checkpoint(dec_block, x, skip, use_reentrant=False)
            else:
                x = dec_block(x, skip)

        # Projection
        x = self.projection(x)

        # Reshape output (out_channels, may differ from input C for P4/P5)
        output = rearrange(x, 'b (t c) z h w -> b t c z h w', t=T_out, c=self.out_channels)

        return output
