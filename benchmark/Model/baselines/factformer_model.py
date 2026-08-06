"""
Factformer: Factorized Transformer for Multi-Dimensional PDE Learning

Reference: Li et al., NeurIPS 2023
GitHub: https://github.com/thuml/Neural-Solver-Library

Key Innovation: Factorized Attention
- Decomposes multi-dimensional attention into separate 1D attention along each axis
- Reduces O(N^2) complexity to O(N^(2/d)) for d-dimensional data
- Efficient for high-dimensional PDEs like 3D turbulence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class AxialAttention(nn.Module):
    """
    Axial Attention: Apply attention along a single axis.

    For 3D data (Z, H, W), we can apply attention along Z, H, or W independently.
    This reduces complexity from O(ZHW)^2 to O(Z^2 + H^2 + W^2).
    """

    def __init__(self, dim, axis, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.axis = axis  # 0=Z, 1=H, 2=W
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: (B, Z, H, W, C)
        Returns:
            (B, Z, H, W, C)
        """
        B, Z, H, W, C = x.shape

        # Reshape based on axis
        if self.axis == 0:  # Attention along Z
            x = rearrange(x, 'b z h w c -> (b h w) z c')
            N = Z
        elif self.axis == 1:  # Attention along H
            x = rearrange(x, 'b z h w c -> (b z w) h c')
            N = H
        else:  # Attention along W
            x = rearrange(x, 'b z h w c -> (b z h) w c')
            N = W

        BHW = x.shape[0]

        # Standard attention
        qkv = self.qkv(x).reshape(BHW, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BHW, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reshape back
        if self.axis == 0:
            x = rearrange(x, '(b h w) z c -> b z h w c', b=B, h=H, w=W)
        elif self.axis == 1:
            x = rearrange(x, '(b z w) h c -> b z h w c', b=B, z=Z, w=W)
        else:
            x = rearrange(x, '(b z h) w c -> b z h w c', b=B, z=Z, h=H)

        return x


class FactorizedAttention(nn.Module):
    """
    Factorized Attention: Sequential axial attention along all axes.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.z_attn = AxialAttention(dim, axis=0, num_heads=num_heads,
                                      qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.h_attn = AxialAttention(dim, axis=1, num_heads=num_heads,
                                      qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.w_attn = AxialAttention(dim, axis=2, num_heads=num_heads,
                                      qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x):
        """
        Args:
            x: (B, Z, H, W, C)
        Returns:
            (B, Z, H, W, C)
        """
        x = x + self.z_attn(x)
        x = x + self.h_attn(x)
        x = x + self.w_attn(x)
        return x


class FactformerBlock(nn.Module):
    """Factformer Transformer Block with Factorized Attention."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FactorizedAttention(
            dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop
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
        """
        Args:
            x: (B, Z, H, W, C)
        """
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Factformer3D(nn.Module):
    """
    Factformer for 3D Turbulence Prediction.

    Uses factorized attention (axial attention) for efficient 3D processing.
    Complexity: O(Z^2 + H^2 + W^2) instead of O((ZHW)^2)

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
        self.total_z = getattr(config, 'TOTAL_Z_LAYERS', 128)

        # Factformer parameters
        self.embed_dim = getattr(config, 'FACTFORMER_DIM', 128)
        self.depth = getattr(config, 'FACTFORMER_DEPTH', 6)
        self.num_heads = getattr(config, 'FACTFORMER_HEADS', 8)
        self.downsample_factor = getattr(config, 'FACTFORMER_DOWNSAMPLE', 4)

        # Input projection (with spatial downsampling for memory efficiency)
        in_dim = self.t_in * self.in_channels
        self.input_proj = nn.Sequential(
            nn.Conv3d(in_dim, self.embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(self.embed_dim // 2, self.embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        # Learnable position embedding
        self.pos_embed = None

        # Factformer blocks
        self.blocks = nn.ModuleList([
            FactformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.,
                drop=0.1,
                attn_drop=0.
            ) for _ in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Output projection (with upsampling)
        out_dim = self.t_out * self.in_channels
        self.output_proj = nn.Sequential(
            nn.ConvTranspose3d(self.embed_dim, self.embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(self.embed_dim // 2, out_dim, kernel_size=4, stride=2, padding=1),
        )

        self._print_info()

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[Factformer] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.in_channels}, Z, H, W)")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Depth: {self.depth}")
        print(f"  Num heads: {self.num_heads}")
        print(f"  Downsample factor: {self.downsample_factor}")
        print(f"  Total parameters: {total_params:,}")

    def forward(self, input_dense, global_timesteps=None):
        """Forward pass."""
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = self.t_out

        # Reshape input: (B, T*C, Z, H, W)
        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')

        # Input projection with downsampling
        x = self.input_proj(x)  # (B, embed_dim, Z', H', W')

        # Get spatial dimensions after downsampling
        _, _, Zd, Hd, Wd = x.shape

        # Reshape for factorized attention: (B, Z', H', W', C)
        x = rearrange(x, 'b c z h w -> b z h w c')

        # Factformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape back: (B, C, Z', H', W')
        x = rearrange(x, 'b z h w c -> b c z h w')

        # Output projection with upsampling
        x = self.output_proj(x)

        # Handle size mismatch
        if x.shape[2:] != (Z, H, W):
            x = F.interpolate(x, size=(Z, H, W), mode='trilinear', align_corners=False)

        # Reshape output: (B, T, C, Z, H, W)
        output = rearrange(x, 'b (t c) z h w -> b t c z h w', t=T_out, c=C)

        return output
