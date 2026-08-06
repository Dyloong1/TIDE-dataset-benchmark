"""
DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training

Reference: Hao et al., ICML 2024
Paper: https://arxiv.org/abs/2403.03542
GitHub: https://github.com/thu-ml/DPOT

Key Innovations:
1. Auto-regressive denoising pre-training (inject noise, predict clean next step)
2. Fourier-based attention for efficient spectral learning
3. Scalable architecture (7M to 1B parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from einops import rearrange, repeat
import math


class FourierAttention(nn.Module):
    """
    Fourier-based Attention Layer (core of DPOT).

    Applies attention in frequency domain for more efficient learning
    of kernel integral transforms in PDEs.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 use_spectral=True, spectral_modes=16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_spectral = use_spectral
        self.spectral_modes = spectral_modes

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_spectral:
            # Learnable spectral filter
            self.spectral_filter = nn.Parameter(
                torch.randn(num_heads, spectral_modes) * 0.02
            )

    def forward(self, x):
        """
        Args:
            x: (B, N, C)
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape

        # Standard QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_spectral and N > self.spectral_modes:
            # Spectral attention: apply in frequency domain
            # Cast to float32 for complex FFT (half precision not supported)
            q_float = q.float()
            k_float = k.float()
            v_float = v.float()
            q_fft = fft.rfft(q_float, dim=-2)
            k_fft = fft.rfft(k_float, dim=-2)
            v_fft = fft.rfft(v_float, dim=-2)

            # Only keep low-frequency modes
            modes = min(self.spectral_modes, q_fft.shape[-2])

            # Apply learnable spectral filter
            filter_weights = F.softplus(self.spectral_filter[:, :modes]).unsqueeze(0).unsqueeze(-1)

            q_low = q_fft[..., :modes, :] * filter_weights
            k_low = k_fft[..., :modes, :] * filter_weights
            v_low = v_fft[..., :modes, :]

            # Attention in frequency domain (use magnitude for softmax)
            attn_fft = (q_low @ k_low.transpose(-2, -1)) * self.scale
            # Softmax on magnitude, keep complex phase. NaN-guard: torch.exp(1j*angle()) has an
            # undefined/NaN gradient where the complex magnitude is exactly 0 (phase of 0 is
            # undefined). Reconstruct the unit-phase from the (real,imag) parts with an eps-padded
            # magnitude so both value and gradient stay finite under bf16->fp32.
            attn_magnitude = attn_fft.abs()
            attn_weights = attn_magnitude.softmax(dim=-1)
            unit_phase = attn_fft / attn_magnitude.clamp_min(1e-12)   # unit-magnitude phase, NaN-safe at 0
            attn_fft = attn_weights * unit_phase
            attn_fft = self.attn_drop(attn_fft.real) + 1j * self.attn_drop(attn_fft.imag)

            out_fft = attn_fft @ v_low

            # Pad and inverse FFT
            out_padded = torch.zeros_like(q_fft)
            out_padded[..., :modes, :] = out_fft
            x_out = fft.irfft(out_padded, n=N, dim=-2)
            # Cast back to original dtype
            x_out = x_out.to(q.dtype)
        else:
            # Standard attention for short sequences
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_out = attn @ v

        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out


class DPOTBlock(nn.Module):
    """DPOT Transformer Block with Fourier Attention."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0.,
                 use_spectral=True, spectral_modes=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FourierAttention(
            dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop,
            use_spectral=use_spectral, spectral_modes=spectral_modes
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DPOT3D(nn.Module):
    """
    DPOT: Denoising Pretraining Operator Transformer for 3D Turbulence.

    Adapts DPOT architecture to our 3D turbulence prediction task.
    Key difference from original: we don't use denoising pretraining here,
    but leverage the Fourier-attention architecture.

    Input: (B, T_in=5, C=4, Z, H, W)
    Output: (B, T_out=5, C=4, Z, H, W)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dimensions
        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        # honor OUT_CHANNELS != IN_CHANNELS so DPOT can run channel-changing tasks (pressure
        # 3->1, sgs 4->6); defaults to in_channels for same-channel tasks (rollout/superres/recon).
        self.out_channels = getattr(config, 'OUT_CHANNELS', None) or self.in_channels
        self.total_z = getattr(config, 'TOTAL_Z_LAYERS', 128)

        # DPOT parameters
        self.embed_dim = getattr(config, 'DPOT_DIM', 256)
        self.depth = getattr(config, 'DPOT_DEPTH', 8)
        self.num_heads = getattr(config, 'DPOT_HEADS', 8)
        self.spectral_modes = getattr(config, 'DPOT_SPECTRAL_MODES', 32)
        self.patch_size = getattr(config, 'DPOT_PATCH_SIZE', (4, 8, 8))

        # Patch embedding
        self.patch_z, self.patch_h, self.patch_w = self.patch_size
        in_dim = self.t_in * self.in_channels * self.patch_z * self.patch_h * self.patch_w
        self.patch_embed = nn.Linear(in_dim, self.embed_dim)

        # Learnable position embedding - pre-allocate for expected size
        # For DNS data: Z=64, H=64, W=64 with patch_size=(4,8,8) -> 16*8*8=1024 patches
        max_patches = 1024
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Time embedding (for potential temporal modeling)
        self.time_embed = nn.Sequential(
            nn.Linear(self.t_in, self.embed_dim // 4),
            nn.GELU(),
            nn.Linear(self.embed_dim // 4, self.embed_dim)
        )

        # DPOT blocks
        self.blocks = nn.ModuleList([
            DPOTBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.,
                drop=0.1,
                attn_drop=0.,
                use_spectral=True,
                spectral_modes=self.spectral_modes
            ) for _ in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Output projection
        out_dim = self.t_out * self.out_channels * self.patch_z * self.patch_h * self.patch_w
        self.output_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, out_dim)
        )

        self._print_info()

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[DPOT] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.out_channels}, Z, H, W)")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Depth: {self.depth}")
        print(f"  Num heads: {self.num_heads}")
        print(f"  Spectral modes: {self.spectral_modes}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Total parameters: {total_params:,}")

    def _get_pos_embed(self, num_patches, device):
        """Get positional embedding, interpolate if size differs."""
        pos_embed = self.pos_embed.to(device)
        if pos_embed.shape[1] != num_patches:
            # Interpolate position embedding to match actual patch count
            pos_embed = F.interpolate(
                pos_embed.transpose(1, 2),  # (1, C, N)
                size=num_patches,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (1, N', C)
        return pos_embed

    def patchify(self, x):
        """Convert input to patches."""
        B, T, C, Z, H, W = x.shape
        pz, ph, pw = self.patch_size

        # Pad if necessary
        pad_z = (pz - Z % pz) % pz
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_z > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_z))
            Z, H, W = Z + pad_z, H + pad_h, W + pad_w

        nz, nh, nw = Z // pz, H // ph, W // pw
        x = x.reshape(B, T, C, nz, pz, nh, ph, nw, pw)
        x = x.permute(0, 3, 5, 7, 1, 2, 4, 6, 8)
        x = x.reshape(B, nz * nh * nw, T * C * pz * ph * pw)

        return x, (Z, H, W), (nz, nh, nw)

    def unpatchify(self, x, orig_shape, padded_shape, grid_shape):
        """Convert patches back to volume."""
        B = x.shape[0]
        T, C = self.t_out, self.out_channels  # output head emits out_channels (pressure 1 / sgs 6)
        Z, H, W = padded_shape
        nz, nh, nw = grid_shape
        pz, ph, pw = self.patch_size

        x = x.reshape(B, nz, nh, nw, T, C, pz, ph, pw)
        x = x.permute(0, 4, 5, 1, 6, 2, 7, 3, 8)
        x = x.reshape(B, T, C, Z, H, W)

        orig_z, orig_h, orig_w = orig_shape
        x = x[:, :, :, :orig_z, :orig_h, :orig_w]

        return x

    def forward(self, input_dense, global_timesteps=None):
        """Forward pass."""
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = self.t_out
        orig_shape = (Z, H, W)

        # Patchify input
        x, padded_shape, grid_shape = self.patchify(input_dense)
        num_patches = x.shape[1]

        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embedding
        pos_embed = self._get_pos_embed(num_patches, x.device)
        x = x + pos_embed

        # Add time embedding (global to all patches)
        if global_timesteps is not None:
            time_feat = self.time_embed(global_timesteps[:, :self.t_in].float())
            x = x + time_feat.unsqueeze(1)

        # DPOT blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Output projection
        x = self.output_proj(x)

        # Unpatchify
        output = self.unpatchify(x, orig_shape, padded_shape, grid_shape)

        return output
