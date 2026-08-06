"""
DiT-Refiner: Diffusion Transformer backbone for PDE-Refiner

Replaces the 3D U-Net backbone with a Diffusion Transformer (DiT),
keeping the PDE-Refiner diffusion schedule (v-prediction, K=3 steps).

Architecture: 3D Patchify -> DiT Blocks (adaLN-Zero) -> Unpatchify
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from torch.utils.checkpoint import checkpoint

from .pde_refiner_model import fourier_embedding, PDERefinerScheduler


class PatchEmbed3D(nn.Module):
    """3D patch embedding using Conv3d with kernel=stride=patch_size."""

    def __init__(self, in_channels, embed_dim, patch_size=(4, 8, 8)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, Z, H, W]
        x = self.proj(x)  # [B, embed_dim, Z//pz, H//ph, W//pw]
        B, C, Zp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, embed_dim]
        return x, (Zp, Hp, Wp)


class DiTBlock(nn.Module):
    """
    Transformer block with adaLN-Zero modulation (Peebles & Xie, 2023).

    Conditioning signal modulates LayerNorm parameters (scale, shift, gate)
    for both the attention and MLP sub-layers.
    """

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

        # adaLN-Zero: projects conditioning to 6 modulation parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        # Initialize gate parameters to zero (zero-initialization trick)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        """
        Args:
            x: [B, N, D] token sequence
            c: [B, D] conditioning embedding
        """
        # Compute modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Attention branch with adaLN modulation
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h

        # MLP branch with adaLN modulation
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class DiT3D(nn.Module):
    """
    Diffusion Transformer for 3D volumes.

    Patchifies the input, processes through DiT blocks, then unpatchifies.
    """

    def __init__(self, in_channels, cond_channels, out_channels,
                 hidden_dim=512, num_heads=8, num_layers=8,
                 patch_size=(4, 8, 8), time_dim=128, mlp_ratio=4.0,
                 dropout=0.0, use_gradient_checkpointing=True,
                 input_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.use_gradient_checkpointing = use_gradient_checkpointing

        total_in = in_channels + cond_channels
        patch_dim = total_in * patch_size[0] * patch_size[1] * patch_size[2]

        # Patch embedding
        self.patch_embed = PatchEmbed3D(total_in, hidden_dim, patch_size)

        # Time embedding (fourier_dim → MLP → hidden_dim)
        self.fourier_dim = time_dim
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Final layer
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        out_patch_dim = out_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.final_linear = nn.Linear(hidden_dim, out_patch_dim)

        # Positional embedding
        if input_size is not None:
            Zp = input_size[0] // patch_size[0]
            Hp = input_size[1] // patch_size[1]
            Wp = input_size[2] // patch_size[2]
            num_patches = Zp * Hp * Wp
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self._cached_grid_size = (Zp, Hp, Wp)
        else:
            self.pos_embed = None
            self._cached_grid_size = None

    def _get_pos_embed(self, num_patches, device):
        """Get or create positional embedding for given number of patches."""
        if self.pos_embed is None or self.pos_embed.shape[1] != num_patches:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.hidden_dim, device=device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        return self.pos_embed

    def _init_pos_embed(self, grid_size, device):
        """Initialize positional embedding for a specific grid size."""
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        if self._cached_grid_size != grid_size:
            self._cached_grid_size = grid_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.hidden_dim, device=device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def unpatchify(self, x, grid_size):
        """Reshape tokens back to 3D volume."""
        Zp, Hp, Wp = grid_size
        pz, ph, pw = self.patch_size
        C = self.out_channels

        x = x.reshape(x.shape[0], Zp, Hp, Wp, pz, ph, pw, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # [B, C, Zp, pz, Hp, ph, Wp, pw]
        x = x.reshape(x.shape[0], C, Zp * pz, Hp * ph, Wp * pw)
        return x

    def forward(self, y_noised, condition, timestep):
        """
        Args:
            y_noised: [B, C_out, Z, H, W]
            condition: [B, C_cond, Z, H, W]
            timestep: [B,] float timestep
        Returns:
            [B, C_out, Z, H, W]
        """
        # Time embedding
        t_emb = fourier_embedding(timestep.float(), self.fourier_dim)
        t_emb = self.time_embed(t_emb)  # [B, hidden_dim]

        # Concatenate and patchify
        x = torch.cat([y_noised, condition], dim=1)  # [B, C_in+C_cond, Z, H, W]
        x, grid_size = self.patch_embed(x)  # [B, N, hidden_dim]

        # Add positional embedding
        if self.pos_embed is None or self.pos_embed.shape[1] != x.shape[1]:
            self._init_pos_embed(grid_size, x.device)
        x = x + self.pos_embed.to(x.device)

        # DiT blocks
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, t_emb, use_reentrant=False)
            else:
                x = block(x, t_emb)

        # Final layer with adaLN modulation
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        x = self.final_norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_linear(x)

        # Unpatchify
        x = self.unpatchify(x, grid_size)
        return x


class DiTRefiner3D(nn.Module):
    """
    DiT-Refiner for 3D Turbulence Prediction.

    Uses DiT3D backbone with PDE-Refiner diffusion schedule.
    Same interface as PDERefiner3D.
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

        # PDE-Refiner parameters
        self.num_refine_steps = getattr(config, 'REFINER_STEPS', 3)
        self.time_dim = getattr(config, 'REFINER_TIME_DIM', 128)
        self.min_noise_std = getattr(config, 'MIN_NOISE_STD', 4e-7)
        self.time_multiplier = 1000.0 / self.num_refine_steps

        # Channel dimensions
        self.cond_channels = self.t_in * self.in_channels
        self.out_channels = self.t_out * self.in_channels

        # Scheduler
        self.scheduler = PDERefinerScheduler(
            num_refine_steps=self.num_refine_steps,
            min_noise_std=self.min_noise_std,
            prediction_type="v_prediction"
        )

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
        print(f"\n[DiT-Refiner] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.in_channels}, Z, H, W)")
        print(f"  Refinement steps: {self.num_refine_steps}")
        print(f"  Patch size: {self.model.patch_size}")
        print(f"  Total parameters: {total_params:,}")

    def get_training_loss(self, input_dense, output_dense):
        """Compute diffusion training loss with v-prediction objective."""
        B, T_in, C, Z, H, W = input_dense.shape
        device = input_dense.device

        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')
        y = rearrange(output_dense, 'b t c z h w -> b (t c) z h w')

        k = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=device)

        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        noise_factor = alphas_cumprod[k]
        noise_factor = noise_factor.view(-1, *[1 for _ in range(y.ndim - 1)])
        signal_factor = 1 - noise_factor

        noise = torch.randn_like(y)
        y_noised = self.scheduler.add_noise(y, noise, k)

        pred = self.model(y_noised, x, k.float() * self.time_multiplier)

        target = (noise_factor ** 0.5) * noise - (signal_factor ** 0.5) * y

        # CustomMSELoss: avg spatial, sum T*C, avg batch
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.mean(dim=tuple(range(2, loss.ndim)))  # avg spatial
        loss = loss.sum(dim=1)  # sum T*C
        loss = loss.mean()  # avg batch
        return loss

    def forward(self, input_dense, global_timesteps=None, num_refine_steps=None):
        """Inference: Iterative denoising from random noise."""
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = self.t_out
        device = input_dense.device

        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')

        y_noised = torch.randn(B, self.out_channels, Z, H, W, dtype=input_dense.dtype, device=device)

        for k in self.scheduler.timesteps:
            time = torch.zeros(B, dtype=input_dense.dtype, device=device) + k
            pred = self.model(y_noised, x, time * self.time_multiplier)
            y_noised = self.scheduler.step(pred, k.item(), y_noised).prev_sample

        output = rearrange(y_noised, 'b (t c) z h w -> b t c z h w', t=T_out, c=C)

        return output
