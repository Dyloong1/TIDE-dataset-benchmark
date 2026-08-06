"""
PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers

Reference: Lippe et al., NeurIPS 2023
Paper: https://arxiv.org/abs/2308.05732
GitHub: https://github.com/microsoft/pdearena

Key Innovation: Diffusion-based Iterative Refinement
- Uses exponentially decreasing noise schedule (betas from reversed sigma)
- v-prediction objective for stable training
- DDPM stochastic posterior sampling during inference
- Pre-norm residual blocks with zero-init on conv2 and output layer

This implementation follows the official pdearena implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops import rearrange
import math


# ============================================================================
# Utility functions
# ============================================================================

def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def fourier_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings (official: cos first, then sin).

    Args:
        timesteps: (N,) tensor of timestep indices (may be fractional)
        dim: embedding dimension
        max_period: controls minimum frequency
    Returns:
        (N, dim) positional embedding
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ============================================================================
# PDERefinerScheduler — correct DDPM schedule matching official
# ============================================================================

class PDERefinerScheduler:
    """
    DDPM scheduler for PDE-Refiner matching the official pdearena implementation.

    Noise schedule:
        betas = [min_noise_std^(k/K) for k in reversed(range(K+1))]
        alphas = 1 - betas
        alphas_cumprod = cumprod(alphas)

    This gives alphas_cumprod[0] ≈ 1.0 (clean) and alphas_cumprod[K] = 0.0 (pure noise).
    Inference iterates timesteps [K, K-1, ..., 1, 0] (K+1 evaluations, descending).
    """

    def __init__(self, num_refine_steps, min_noise_std=4e-7, prediction_type="v_prediction"):
        self.num_train_timesteps = num_refine_steps + 1
        self.num_refine_steps = num_refine_steps
        self.min_noise_std = min_noise_std
        self.prediction_type = prediction_type

        # Compute betas from reversed sigma schedule (matching official line 93)
        betas = [min_noise_std ** (k / num_refine_steps) for k in reversed(range(num_refine_steps + 1))]
        self.betas = torch.tensor(betas, dtype=torch.float64)

        # Compute alphas and alphas_cumprod
        alphas = 1.0 - self.betas
        self.alphas = alphas.float()
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).float()

        # Inference timesteps: descending [K, K-1, ..., 1, 0]
        self.timesteps = torch.arange(num_refine_steps, -1, -1)

        self.config = type('Config', (), {
            'num_train_timesteps': self.num_train_timesteps,
            'prediction_type': prediction_type,
        })()

        print(f"\n[PDERefinerScheduler] Noise schedule initialized:")
        print(f"  Refinement steps K={num_refine_steps}")
        print(f"  min_noise_std={min_noise_std}")
        print(f"  betas: {[f'{b:.6g}' for b in betas]}")
        print(f"  alphas_cumprod: {[f'{a:.6g}' for a in self.alphas_cumprod.tolist()]}")
        print(f"  timesteps (inference): {self.timesteps.tolist()}")

    def add_noise(self, original_samples, noise, timesteps):
        """Add noise: x_t = sqrt(abar_t) * x_0 + sqrt(1-abar_t) * eps."""
        device = original_samples.device
        abar = self.alphas_cumprod.to(device)

        sqrt_abar = abar[timesteps] ** 0.5
        sqrt_one_minus_abar = (1 - abar[timesteps]) ** 0.5

        while len(sqrt_abar.shape) < len(original_samples.shape):
            sqrt_abar = sqrt_abar.unsqueeze(-1)
            sqrt_one_minus_abar = sqrt_one_minus_abar.unsqueeze(-1)

        return sqrt_abar * original_samples + sqrt_one_minus_abar * noise

    def step(self, model_output, timestep, sample):
        """
        DDPM posterior sampling step (stochastic, fixed_small variance).

        For v-prediction:
            pred_x0 = sqrt(abar_t) * x_t - sqrt(1-abar_t) * v
        Then DDPM posterior:
            mean = coeff1 * pred_x0 + coeff2 * x_t
            variance = beta_t * (1 - abar_{t-1}) / (1 - abar_t)
            x_{t-1} = mean + sqrt(variance) * noise  (noise=0 at t=0)
        """
        device = sample.device
        t = timestep

        abar_t = self.alphas_cumprod[t].to(device)
        alpha_t = self.alphas[t].to(device)
        beta_t = self.betas[t].to(device).float()

        # Previous alpha_bar (t-1). If t=0, abar_{-1} = 1.0 (fully clean)
        if t > 0:
            abar_t_prev = self.alphas_cumprod[t - 1].to(device)
        else:
            abar_t_prev = torch.tensor(1.0, device=device)

        sqrt_abar_t = abar_t ** 0.5
        sqrt_one_minus_abar_t = (1 - abar_t) ** 0.5

        # v-prediction: recover x_0
        if self.prediction_type == "v_prediction":
            pred_x0 = sqrt_abar_t * sample - sqrt_one_minus_abar_t * model_output
        else:
            pred_x0 = (sample - sqrt_one_minus_abar_t * model_output) / (sqrt_abar_t + 1e-8)

        # DDPM posterior mean
        one_minus_abar_t = (1 - abar_t).clamp(min=1e-10)
        coeff1 = (abar_t_prev ** 0.5) * beta_t / one_minus_abar_t
        coeff2 = (alpha_t ** 0.5) * (1 - abar_t_prev) / one_minus_abar_t

        mean = coeff1 * pred_x0 + coeff2 * sample

        # DDPM fixed_small variance
        if t > 0:
            variance = beta_t * (1 - abar_t_prev) / one_minus_abar_t
            variance = variance.clamp(min=1e-20)
            noise = torch.randn_like(sample)
            prev_sample = mean + (variance ** 0.5) * noise
        else:
            prev_sample = mean

        return type('StepOutput', (), {'prev_sample': prev_sample})()


# ============================================================================
# ResidualBlock — pre-norm, zero-init conv2 (matching official)
# ============================================================================

class ResidualBlock3D(nn.Module):
    """
    Pre-norm residual block with time conditioning.
    Pattern: norm → act → conv1 → (add time emb) → norm → act → zero_init(conv2)
    """

    def __init__(self, in_channels, out_channels, cond_channels, n_groups=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = zero_module(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
        self.act = nn.GELU()

        self.cond_emb = nn.Linear(cond_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
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


# ============================================================================
# Downsample / Upsample — learned conv (matching official)
# ============================================================================

class Downsample3D(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# MiddleBlock — two ResBlocks (matching official)
# ============================================================================

class MiddleBlock3D(nn.Module):
    def __init__(self, n_channels, cond_channels, n_groups=1):
        super().__init__()
        self.res1 = ResidualBlock3D(n_channels, n_channels, cond_channels, n_groups)
        self.res2 = ResidualBlock3D(n_channels, n_channels, cond_channels, n_groups)

    def forward(self, x, emb):
        x = self.res1(x, emb)
        x = self.res2(x, emb)
        return x


# ============================================================================
# RefinerUNet3D — full architecture matching official twod_unet.py
# ============================================================================

class RefinerUNet3D(nn.Module):
    """
    3D U-Net for PDE-Refiner, matching official architecture.

    Structure:
        image_proj(3x3x3) →
        [ResBlock×n_blocks + Downsample]×n_levels →
        MiddleBlock(ResBlock + ResBlock) →
        [Upsample + ResBlock×(n_blocks+1)]×n_levels →
        norm → act → zero_init_conv(3x3x3)

    Skip connections via concatenation.
    """

    def __init__(self, in_channels, cond_channels, out_channels,
                 hidden_channels=64, ch_mults=(1, 2, 2, 4), n_blocks=2,
                 n_groups=1, gradient_checkpointing=False):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.gradient_checkpointing = gradient_checkpointing
        n_resolutions = len(ch_mults)

        # Time embedding: dim = hidden_channels * 4 (matching official)
        time_embed_dim = hidden_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.fourier_dim = hidden_channels

        # Input projection
        total_in = in_channels + cond_channels
        self.image_proj = nn.Conv3d(total_in, hidden_channels, kernel_size=3, padding=1)

        # ---- Encoder (Down path) ----
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        channels = [hidden_channels]
        ch_in = hidden_channels
        for i in range(n_resolutions):
            ch_out = hidden_channels * ch_mults[i]
            blocks = nn.ModuleList()
            for j in range(n_blocks):
                blocks.append(ResidualBlock3D(ch_in, ch_out, time_embed_dim, n_groups))
                ch_in = ch_out
                channels.append(ch_out)  # append after EACH block
            self.down_blocks.append(blocks)

            if i < n_resolutions - 1:
                self.downsamples.append(Downsample3D(ch_out))
                channels.append(ch_out)
            else:
                self.downsamples.append(None)

        # ---- Middle ----
        self.middle = MiddleBlock3D(ch_in, time_embed_dim, n_groups)

        # ---- Decoder (Up path) ----
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in reversed(range(n_resolutions)):
            ch_out = hidden_channels * ch_mults[i]

            if i < n_resolutions - 1:
                self.upsamples.append(Upsample3D(ch_in))
            else:
                self.upsamples.append(None)

            blocks = nn.ModuleList()
            for j in range(n_blocks + 1):
                skip_ch = channels.pop()
                blocks.append(ResidualBlock3D(ch_in + skip_ch, ch_out, time_embed_dim, n_groups))
                ch_in = ch_out
            self.up_blocks.append(blocks)

        # ---- Output ----
        self.final_norm = nn.GroupNorm(n_groups, ch_in)
        self.final_act = nn.GELU()
        self.final_conv = zero_module(nn.Conv3d(ch_in, out_channels, kernel_size=3, padding=1))

    def forward(self, y_noised, condition, timestep):
        """
        Args:
            y_noised: (B, C_out, Z, H, W)
            condition: (B, C_cond, Z, H, W)
            timestep: (B,) float timestep values
        Returns:
            (B, C_out, Z, H, W) v-prediction
        """
        t_emb = fourier_embedding(timestep, self.fourier_dim)
        t_emb = self.time_embed(t_emb)

        h = torch.cat([y_noised, condition], dim=1)
        h = self.image_proj(h)

        use_ckpt = self.gradient_checkpointing and self.training

        # Encoder
        skips = [h]
        for i, (blocks, downsample) in enumerate(zip(self.down_blocks, self.downsamples)):
            for block in blocks:
                if use_ckpt:
                    h = torch_checkpoint(block, h, t_emb, use_reentrant=False)
                else:
                    h = block(h, t_emb)
                skips.append(h)
            if downsample is not None:
                h = downsample(h)
                skips.append(h)

        # Middle
        if use_ckpt:
            h = torch_checkpoint(self.middle, h, t_emb, use_reentrant=False)
        else:
            h = self.middle(h, t_emb)

        # Decoder
        for i, (blocks, upsample) in enumerate(zip(self.up_blocks, self.upsamples)):
            if upsample is not None:
                h = upsample(h)
            for block in blocks:
                skip = skips.pop()
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)
                h = torch.cat([h, skip], dim=1)
                if use_ckpt:
                    h = torch_checkpoint(block, h, t_emb, use_reentrant=False)
                else:
                    h = block(h, t_emb)

        h = self.final_conv(self.final_act(self.final_norm(h)))
        return h


# ============================================================================
# PDERefiner3D — wrapper
# ============================================================================

class PDERefiner3D(nn.Module):
    """
    PDE-Refiner for 3D Turbulence Prediction.

    Matches the official pdearena implementation:
    - Correct noise schedule via reversed sigma betas
    - v-prediction objective with CustomMSELoss reduction
    - DDPM stochastic posterior sampling during inference
    - K+1 inference evaluations (timesteps: K, K-1, ..., 0)

    Input: (B, T_in=5, C=4, Z, H, W)
    Output: (B, T_out=5, C=4, Z, H, W)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        self.total_z = getattr(config, 'TOTAL_Z_LAYERS', 128)

        hidden_channels = getattr(config, 'REFINER_HIDDEN', 64)
        self.num_refine_steps = getattr(config, 'REFINER_STEPS', 3)
        self.min_noise_std = getattr(config, 'MIN_NOISE_STD', 4e-7)

        ch_mults = getattr(config, 'REFINER_CH_MULTS', (1, 2, 2, 4))
        n_blocks = getattr(config, 'REFINER_N_BLOCKS', 2)

        # Time multiplier: 1000/K (matching official line 101)
        self.time_multiplier = 1000.0 / self.num_refine_steps

        self.cond_channels = self.t_in * self.in_channels
        self.out_channels = self.t_out * self.in_channels

        self.scheduler = PDERefinerScheduler(
            num_refine_steps=self.num_refine_steps,
            min_noise_std=self.min_noise_std,
            prediction_type="v_prediction",
        )

        self.model = RefinerUNet3D(
            in_channels=self.out_channels,
            cond_channels=self.cond_channels,
            out_channels=self.out_channels,
            hidden_channels=hidden_channels,
            ch_mults=ch_mults,
            n_blocks=n_blocks,
            gradient_checkpointing=True,
        )

        self._print_info()

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[PDE-Refiner] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.in_channels}, Z, H, W)")
        print(f"  Refinement steps: {self.num_refine_steps}")
        print(f"  time_multiplier: {self.time_multiplier:.2f}")
        print(f"  Min noise std: {self.min_noise_std}")
        print(f"  Total parameters: {total_params:,}")

    def get_training_loss(self, input_dense, output_dense):
        """
        Compute diffusion training loss with v-prediction objective.

        Loss reduction matches official CustomMSELoss:
        - Average over spatial dims
        - Sum over time*channels (T*C merged)
        - Average over batch
        """
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
        """
        Inference: DDPM stochastic sampling from random noise.
        K+1 steps: timesteps [K, K-1, ..., 1, 0] (descending).
        """
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = self.t_out
        device = input_dense.device

        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')

        y_noised = torch.randn(
            size=(B, self.out_channels, Z, H, W),
            dtype=input_dense.dtype,
            device=device,
        )

        for k in self.scheduler.timesteps:
            time = torch.zeros(B, dtype=input_dense.dtype, device=device) + k
            pred = self.model(y_noised, x, time * self.time_multiplier)
            y_noised = self.scheduler.step(pred, k.item(), y_noised).prev_sample

        output = rearrange(y_noised, 'b (t c) z h w -> b t c z h w', t=T_out, c=C)
        return output
