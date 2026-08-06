"""
Transolver: A Fast Transformer Solver for PDEs on General Geometries

Reference: Wu et al., ICML 2024 Spotlight
Paper: https://arxiv.org/abs/2402.02366
GitHub: https://github.com/thuml/Transolver

Key Innovation: Physics-Attention
- Adaptively split discretized domain into learnable slices
- Points with similar physical states are grouped together
- More efficient than standard attention on large meshes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class PhysicsAttention(nn.Module):
    """
    Physics-Attention Layer (core innovation of Transolver).

    Instead of computing attention between all N points,
    learns to group points into K slices based on physical states,
    then applies cross-attention between slices.
    """

    def __init__(self, dim, num_heads=8, num_slices=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_slices = num_slices
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Slice projection: learn to group points into slices
        self.slice_proj = nn.Linear(dim, num_slices)

        # QKV for slice-level attention
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Slice aggregation and distribution
        self.slice_agg = nn.Linear(dim, dim)
        self.slice_dist = nn.Linear(dim, dim)

        # Learnable softmax temperature -- official Transolver divides the slice logits by a
        # learnable temperature (init 0.5, clamped to [0.1, 5]); official is per-head because its
        # slice weights are per-head, ours are shared across heads so a scalar is the equivalent.
        self.temperature = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x: (B, N, C) - N is number of spatial points
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        K = self.num_slices

        # Step 1: Compute slice assignments (soft assignment)
        # (B, N, K) - probability of each point belonging to each slice
        slice_weights = F.softmax(self.slice_proj(x) / torch.clamp(self.temperature, 0.1, 5.0), dim=-1)

        # Step 2: Aggregate points into slices -- weighted MEAN, not weighted sum. The official
        # implementation divides by the per-slice weight total (slice_norm + 1e-5); without it a
        # slice token scales with N/K (=64 at native 256^3 / patch16), which for correlated inputs
        # amplifies the common mode ~64x per block and was measured to blow the residual stream up
        # from rms 1.4 (block 0) to 55 (block 11) at init (CYB-3 gradient-flow probe, 2026-07-16).
        slice_tokens = torch.einsum('bnk,bnc->bkc', slice_weights, x)
        slice_norm = slice_weights.sum(dim=1)  # (B, K)
        slice_tokens = slice_tokens / (slice_norm.unsqueeze(-1) + 1e-5)

        # Step 3: Apply self-attention among slices
        # This is much cheaper than attention among all N points
        qkv = self.qkv(slice_tokens).reshape(B, K, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        slice_out = (attn @ v).transpose(1, 2).reshape(B, K, C)
        slice_out = self.slice_agg(slice_out)

        # Step 4: Distribute slice features back to points
        # (B, N, C) - weighted sum of slice features
        out = torch.einsum('bnk,bkc->bnc', slice_weights, slice_out)
        out = self.slice_dist(out)

        # Residual connection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class TransolverBlock(nn.Module):
    """Transolver Transformer Block with Physics-Attention."""

    def __init__(self, dim, num_heads=8, num_slices=8, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PhysicsAttention(
            dim, num_heads=num_heads, num_slices=num_slices,
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

        # LayerScale (per-channel residual-branch damping, init 1e-2; CaiT/modern-ViT standard).
        # Not in the official repo, but required here: at standard init the 12-block stack buries
        # the input signal under accumulated random branch outputs and attenuates head gradients
        # ~1000x -- the model then cannot fit even ONE sample's delta direction (cos 0.002 after
        # 300 steps) while the identical embed+head WITHOUT blocks learns (cos 0.18 and climbing).
        # Keeping branches damped-but-trainable preserves the input pathway (CYB-3 audit 2026-07-16).
        self.ls1 = nn.Parameter(torch.full((dim,), 1e-2))
        self.ls2 = nn.Parameter(torch.full((dim,), 1e-2))

    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class Transolver3D(nn.Module):
    """
    Transolver for 3D Turbulence Prediction.

    Adapts Transolver to our 3D turbulence prediction task.
    - Treats 3D volume as a sequence of spatial tokens
    - Uses Physics-Attention for efficient global modeling
    - Outputs full 3D prediction

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

        # Transolver parameters
        self.embed_dim = getattr(config, 'TRANSOLVER_DIM', 256)
        self.depth = getattr(config, 'TRANSOLVER_DEPTH', 6)
        self.num_heads = getattr(config, 'TRANSOLVER_HEADS', 8)
        self.num_slices = getattr(config, 'TRANSOLVER_SLICES', 16)
        self.patch_size = getattr(config, 'TRANSOLVER_PATCH_SIZE', (8, 8, 8))

        # Patch embedding
        self.patch_z, self.patch_h, self.patch_w = self.patch_size
        in_dim = self.t_in * self.in_channels * self.patch_z * self.patch_h * self.patch_w
        self.patch_embed = nn.Linear(in_dim, self.embed_dim)

        # Positional embedding (learnable). TRANSOLVER_MAX_PATCHES must equal the ACTUAL token
        # count of the training field ((N/patch)^3; v2 native-256 with patch8 -> 32768, patch16 ->
        # 4096) -- the old hardcoded 512 was sized for a 128^3 field and _get_pos_embed silently
        # stretched it 8-64x by linear interpolation at native 256^3, i.e. the model trained with a
        # nearly information-free positional signal. NOTE the table is a Parameter and counts
        # toward the 28-30M capacity band: at dim 384, 32768 tokens cost 12.6M parameters by
        # themselves, so patch/dim retuning must budget for it (v2 Phase A, 2026-07-16).
        max_patches = int(getattr(config, 'TRANSOLVER_MAX_PATCHES', 512))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transolver blocks
        self.blocks = nn.ModuleList([
            TransolverBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                num_slices=self.num_slices,
                mlp_ratio=4.,
                drop=0.1,
                attn_drop=0.
            ) for _ in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Output projection
        out_dim = self.t_out * self.in_channels * self.patch_z * self.patch_h * self.patch_w
        self.output_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, out_dim)
        )

        self._print_info()

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[Transolver] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.in_channels}, Z, H, W)")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Depth: {self.depth}")
        print(f"  Num heads: {self.num_heads}")
        print(f"  Num slices: {self.num_slices}")
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
        """
        Convert input to patches.
        (B, T, C, Z, H, W) -> (B, num_patches, patch_dim)
        """
        B, T, C, Z, H, W = x.shape
        pz, ph, pw = self.patch_size

        # Pad if necessary
        pad_z = (pz - Z % pz) % pz
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_z > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_z))
            Z, H, W = Z + pad_z, H + pad_h, W + pad_w

        # Reshape to patches
        nz, nh, nw = Z // pz, H // ph, W // pw
        x = x.reshape(B, T, C, nz, pz, nh, ph, nw, pw)
        x = x.permute(0, 3, 5, 7, 1, 2, 4, 6, 8)  # (B, nz, nh, nw, T, C, pz, ph, pw)
        x = x.reshape(B, nz * nh * nw, T * C * pz * ph * pw)

        return x, (Z, H, W), (nz, nh, nw)

    def unpatchify(self, x, orig_shape, padded_shape, grid_shape):
        """
        Convert patches back to volume.
        (B, num_patches, patch_dim) -> (B, T, C, Z, H, W)
        """
        B = x.shape[0]
        T, C = self.t_out, self.in_channels
        Z, H, W = padded_shape
        nz, nh, nw = grid_shape
        pz, ph, pw = self.patch_size

        x = x.reshape(B, nz, nh, nw, T, C, pz, ph, pw)
        x = x.permute(0, 4, 5, 1, 6, 2, 7, 3, 8)  # (B, T, C, nz, pz, nh, ph, nw, pw)
        x = x.reshape(B, T, C, Z, H, W)

        # Remove padding
        orig_z, orig_h, orig_w = orig_shape
        x = x[:, :, :, :orig_z, :orig_h, :orig_w]

        return x

    def forward(self, input_dense, global_timesteps=None):
        """
        Forward pass.
        """
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

        # Transolver blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Output projection
        x = self.output_proj(x)

        # Unpatchify
        output = self.unpatchify(x, orig_shape, padded_shape, grid_shape)

        return output
