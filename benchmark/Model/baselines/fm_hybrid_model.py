"""
FM-Hybrid Refiner: Flow Matching Hybrid with PDE-Refiner Structure.

Combines PDE-Refiner's k=0 MSE prediction (from zeros) with
Flow Matching velocity matching for k>=1 refinement steps.

Key design:
- k=0: predict signal u(t) from zeros input (MSE loss, identical to PDE-Refiner)
- k>=1: predict NORMALIZED velocity (= -noise), so all k levels have unit-variance
  targets and contribute equally to training gradients.

Three sigma schedule options:
- 'ddpm': Derived from PDE-Refiner's DDPM cumprod (exponential decay)
- 'ddpm_large': Same formula with min_noise_std=0.01 (larger noise range)
- 'linear': Linear interpolation between DDPM-derived endpoints

Loss normalization:
  v_target = (y_clean - y_perturbed) / sigma_k = -noise  (unit variance for all k)
  At inference: multi-step ODE integration from tau=0 to tau=1 with N Euler substeps

This ensures balanced gradient contribution from all refinement steps,
solving the loss imbalance where k=0 dominated k>=2 by 1000-150000x.

Inference alignment: Training samples tau~U[0,1] along the flow path.
Multi-step ODE inference evaluates the model at tau=0, 1/N, 2/N, ..., (N-1)/N,
matching the training distribution. Single-step (N=1) causes mismatch because
the model only sees tau=0 (full noise) but was trained on all tau levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from .pde_refiner_model import RefinerUNet3D
from noise_ablation.noise_generators import generate_noise as _generate_noise


def _compute_ddpm_sigmas(min_noise_std, K):
    """
    Compute FM-Hybrid refinement sigmas from PDE-Refiner's DDPM schedule.

    Uses the same betas -> cumprod -> abar chain, then extracts
    sigma_k = sqrt(1 - abar[K-k]) for k=1,...,K (decreasing order).

    Args:
        min_noise_std: minimum noise standard deviation (e.g., 4e-7)
        K: number of refinement steps

    Returns:
        list of K sigma values (decreasing: sigma_1 > sigma_2 > ... > sigma_K)
    """
    if K == 0:
        return []

    # PDE-Refiner betas formula (reversed order)
    betas = [min_noise_std ** (k / K) for k in reversed(range(K + 1))]
    alphas = [1.0 - b for b in betas]

    # Cumulative product
    abar = [alphas[0]]
    for i in range(1, len(alphas)):
        abar.append(abar[-1] * alphas[i])

    # Extract sigmas for FM-Hybrid (k=1,...,K, decreasing)
    # k=1 gets the largest sigma, k=K gets the smallest
    sigmas = []
    for k in range(1, K + 1):
        idx = K - k  # Maps k=1 -> abar[K-1], k=K -> abar[0]
        sigma = math.sqrt(max(1.0 - abar[idx], 0.0))
        sigmas.append(sigma)

    return sigmas


class FMHybridScheduler:
    """
    Flow Matching scheduler for FM-Hybrid refinement.

    Supports three sigma schedule types:
    - 'ddpm': Derived from PDE-Refiner's DDPM cumprod schedule
    - 'ddpm_large': Same formula with min_noise_std=0.01
    - 'linear': Linear interpolation between DDPM-derived endpoints

    All schedules produce decreasing sigmas: sigma_1 > sigma_2 > ... > sigma_K.
    Only min_noise_std and K control the schedule — no hardcoded values.

    Loss normalization: velocity target is divided by sigma_k so all k levels
    produce unit-variance targets. Inference multiplies prediction by sigma_k.
    """

    def __init__(self, num_refine_steps=3, min_noise_std=4e-7, sigma_schedule='ddpm',
                 noise_type='white', noise_kwargs=None, sigma_max=None, sigma_min=None):
        self.K = num_refine_steps
        self.min_noise_std = min_noise_std
        self.sigma_schedule = sigma_schedule
        self.noise_type = noise_type
        self.noise_kwargs = noise_kwargs or {}
        self.sigma_max_override = sigma_max
        self.sigma_min_override = sigma_min

        # Compute sigmas based on schedule type
        if num_refine_steps == 0:
            self.sigmas = torch.tensor([0.0], dtype=torch.float32)
        else:
            sigma_values = self._compute_sigmas(sigma_schedule, min_noise_std, num_refine_steps)
            # Index 0 is placeholder for k=0 (MSE step), indices 1..K are refinement sigmas
            self.sigmas = torch.tensor([0.0] + sigma_values, dtype=torch.float32)

        print(f"\n[FMHybridScheduler] Initialized:")
        print(f"  Refinement steps K={num_refine_steps}")
        print(f"  Sigma schedule: {sigma_schedule}")
        print(f"  Noise type: {noise_type}")
        print(f"  min_noise_std={min_noise_std}")
        if num_refine_steps > 0:
            print(f"  Refinement sigmas (k=1..K): {[f'{s:.6f}' for s in self.sigmas[1:].tolist()]}")
        print(f"  Loss normalization: ON (v_target = -noise, unit variance)")

    def _compute_sigmas(self, schedule, min_noise_std, K):
        """Compute K refinement sigmas based on schedule type."""
        if schedule == 'ddpm':
            return _compute_ddpm_sigmas(min_noise_std, K)

        elif schedule == 'ddpm_large':
            # Same DDPM formula but with larger min_noise_std for wider spread
            return _compute_ddpm_sigmas(0.01, K)

        elif schedule == 'linear':
            # Linear interpolation between DDPM-derived endpoints
            ddpm_sigmas = _compute_ddpm_sigmas(min_noise_std, K)
            if K == 1:
                return ddpm_sigmas
            sigma_max = ddpm_sigmas[0]   # largest (k=1)
            sigma_min = ddpm_sigmas[-1]  # smallest (k=K)
            return [
                sigma_max - (sigma_max - sigma_min) * (k - 1) / (K - 1)
                for k in range(1, K + 1)
            ]

        elif schedule == 'fixed_range':
            # Fixed sigma range independent of K.
            # Defaults: sigma_max=0.01, sigma_min=0.001. Overridable via config.
            # More steps = finer refinement within the SAME noise range.
            # This decouples noise level from number of steps, unlike DDPM
            # where sigma_1 grows from 0.0006 (K=1) to 0.43 (K=8).
            sigma_max = self.sigma_max_override if self.sigma_max_override is not None else 0.01
            sigma_min = self.sigma_min_override if self.sigma_min_override is not None else 0.001
            if K == 1:
                return [sigma_max]
            import numpy as np
            return list(np.logspace(
                np.log10(sigma_max), np.log10(sigma_min), K
            ))

        else:
            raise ValueError(f"Unknown sigma_schedule: {schedule}. "
                             f"Choose from 'ddpm', 'ddpm_large', 'linear', 'fixed_range'")

    def get_train_tuple_refine(self, y_clean, k):
        """
        FM normalized velocity matching for refinement step k>=1.

        Defines a flow from y_perturbed = y_clean + sigma_k * noise to y_clean.
        Samples a random interpolation point along this straight path.

        The velocity target is NORMALIZED by sigma_k so all k levels have
        unit-variance targets (equivalent to epsilon/noise prediction).

        Args:
            y_clean: [B, C, Z, H, W] clean target
            k: int, refinement step index (k>=1)

        Returns:
            x_tau: [B, C, Z, H, W] interpolated sample along path
            tau: [B,] interpolation parameter in [0, 1]
            v_target_normalized: [B, C, Z, H, W] = -noise (unit variance)
        """
        B = y_clean.shape[0]
        device = y_clean.device
        sigma_k = self.sigmas[k].item()

        # Noise proxy for coarse prediction error
        noise = _generate_noise(self.noise_type, y_clean.shape, y_clean.device,
                                **self.noise_kwargs)
        y_perturbed = y_clean + sigma_k * noise

        # Random interpolation point tau in [0, 1]
        tau = torch.rand(B, device=device)
        tau_expand = tau.view(-1, *[1 for _ in range(y_clean.ndim - 1)])

        # Interpolate between perturbed and clean
        x_tau = (1 - tau_expand) * y_perturbed + tau_expand * y_clean

        # Normalized velocity target: (y_clean - y_perturbed) / sigma_k = -noise
        # This has unit variance regardless of sigma_k
        v_target_normalized = -noise

        return x_tau, tau, v_target_normalized

    def refine_step(self, x, v_pred_normalized, sigma_k, dt=1.0):
        """
        Euler step with denormalization.

        x_new = x + v_pred_normalized * sigma_k * dt

        The model predicts normalized velocity (= -noise), so we scale
        by sigma_k to get the actual displacement. dt controls step size
        for multi-step ODE integration (dt=1/N for N substeps).
        """
        return x + v_pred_normalized * sigma_k * dt


class FMHybridRefiner3D(nn.Module):
    """
    FM-Hybrid Refiner for 3D Turbulence Prediction.

    Same architecture and interface as PDERefiner3D, but replaces
    diffusion denoising (k>=1) with Flow Matching velocity prediction.

    k=0: MSE prediction from zeros input (identical to PDE-Refiner)
    k>=1: FM normalized velocity matching (predict -noise, scale by sigma_k)

    Loss normalization ensures balanced gradient contribution:
    - k=0 MSE loss ~ signal_variance * T*C
    - k>=1 velocity loss ~ 1.0 * T*C (unit variance noise prediction)

    Input: (B, T_in=5, C=4, Z, H, W)
    Output: (B, T_out=5, C=4, Z, H, W)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dimensions (identical to PDERefiner3D)
        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        self.total_z = getattr(config, 'TOTAL_Z_LAYERS', 128)

        # FM-Hybrid parameters
        hidden_channels = getattr(config, 'REFINER_HIDDEN', 64)
        self.num_refine_steps = getattr(config, 'REFINER_STEPS', 3)  # K
        self.time_dim = getattr(config, 'REFINER_TIME_DIM', 128)
        self.min_noise_std = getattr(config, 'MIN_NOISE_STD', 4e-7)
        sigma_schedule = getattr(config, 'SIGMA_SCHEDULE', 'ddpm')
        self.ode_steps = getattr(config, 'ODE_STEPS', 10)  # N substeps per refinement k

        # Noise type for flow matching prior
        noise_type = getattr(config, 'NOISE_TYPE', 'white')
        noise_spectrum_path = getattr(config, 'NOISE_SPECTRUM_PATH', None)
        noise_kwargs = {}
        if noise_spectrum_path is not None:
            noise_kwargs['spectrum_data_path'] = noise_spectrum_path

        # Time multiplier: maps k to timestep range for sinusoidal embedding
        # Matches PDE-Refiner convention: 1000/K
        if self.num_refine_steps > 0:
            self.time_multiplier = 1000.0 / self.num_refine_steps
        else:
            self.time_multiplier = 1000.0  # K=0: only k=0 step, multiplied by 0

        # Channel dimensions
        self.cond_channels = self.t_in * self.in_channels
        self.out_channels = self.t_out * self.in_channels

        # FM-Hybrid scheduler
        sigma_max = getattr(config, 'SIGMA_MAX', None)
        sigma_min = getattr(config, 'SIGMA_MIN', None)
        self.scheduler = FMHybridScheduler(
            num_refine_steps=self.num_refine_steps,
            min_noise_std=self.min_noise_std,
            sigma_schedule=sigma_schedule,
            noise_type=noise_type,
            noise_kwargs=noise_kwargs,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

        # U-Net backbone (same RefinerUNet3D as PDE-Refiner)
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
        print(f"\n[FM-Hybrid Refiner] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.in_channels}, Z, H, W)")
        print(f"  Refinement steps: {self.num_refine_steps}")
        print(f"  ODE substeps per k: {self.ode_steps}")
        print(f"  Sigma schedule: {self.scheduler.sigma_schedule}")
        print(f"  Noise type: {self.scheduler.noise_type}")
        print(f"  Time multiplier: {self.time_multiplier:.1f}")
        print(f"  Loss normalization: ON")
        print(f"  Total parameters: {total_params:,}")

    def get_training_loss(self, input_dense, output_dense):
        """
        Compute FM-Hybrid training loss with normalized velocity targets.

        Samples random k in {0,...,K} per batch item:
        - k=0: MSE prediction from zeros (signal reconstruction)
        - k>=1: normalized FM velocity matching (predict -noise, unit variance)

        Uses CustomMSELoss reduction: avg spatial, sum T*C, avg batch.

        Args:
            input_dense: [B, T_in, C, Z, H, W] input history
            output_dense: [B, T_out, C, Z, H, W] ground truth

        Returns:
            loss: scalar training loss
        """
        B, T_in, C, Z, H, W = input_dense.shape
        device = input_dense.device

        # Reshape to (B, T*C, Z, H, W)
        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')
        y = rearrange(output_dense, 'b t c z h w -> b (t c) z h w')

        # Sample random refinement step k in {0,...,K} for each batch item
        # For fixed_range schedule: guarantee 50% k=0 to preserve base prediction quality
        # (uniform sampling gives k=0 only 1/(K+1) probability, starving it at large K)
        if self.scheduler.sigma_schedule == 'fixed_range' and self.num_refine_steps > 1:
            # 50% chance k=0, 50% uniform over k=1..K
            coin = torch.rand(B, device=device)
            k_refine = torch.randint(1, self.num_refine_steps + 1, (B,), device=device)
            k = torch.where(coin < 0.5, torch.zeros_like(k_refine), k_refine)
        else:
            k = torch.randint(0, self.num_refine_steps + 1, (B,), device=device)

        total_loss = 0.0
        for b in range(B):
            kb = k[b].item()
            y_b = y[b:b+1]  # (1, T*C, Z, H, W)
            x_b = x[b:b+1]  # (1, T*C, Z, H, W)

            if kb == 0:
                # k=0: MSE prediction from zeros input
                zeros_input = torch.zeros_like(y_b)
                k_time = torch.zeros(1, device=device)
                pred = self.model(zeros_input, x_b, k_time * self.time_multiplier)
                # CustomMSELoss: avg spatial, sum channels, avg batch
                loss = F.mse_loss(pred, y_b, reduction='none')
                loss = loss.mean(dim=tuple(range(2, loss.ndim)))  # avg spatial (Z,H,W)
                loss = loss.sum(dim=1)  # sum T*C channels
                loss = loss.mean()  # avg batch (=1 here)
                total_loss = total_loss + loss
            else:
                # k>=1: normalized FM velocity matching
                # v_target_normalized = -noise (unit variance)
                x_tau, tau, v_target_norm = self.scheduler.get_train_tuple_refine(y_b, kb)
                k_time = torch.full((1,), float(kb), device=device)
                v_pred = self.model(x_tau, x_b, k_time * self.time_multiplier)
                # CustomMSELoss reduction (target is unit variance)
                loss = F.mse_loss(v_pred, v_target_norm, reduction='none')
                loss = loss.mean(dim=tuple(range(2, loss.ndim)))  # avg spatial
                loss = loss.sum(dim=1)  # sum channels
                loss = loss.mean()  # avg batch
                total_loss = total_loss + loss

        return total_loss / B

    def forward(self, input_dense, global_timesteps=None, num_refine_steps=None,
                ode_steps=None):
        """
        Inference: k=0 MSE prediction, then K FM refinement steps.

        For k>=1: Multi-step ODE integration along the flow path.
        The flow goes from y_perturbed (tau=0) to y_clean (tau=1).
        We integrate with N Euler substeps to match the training distribution
        where the model sees inputs at all tau in [0,1].

        At each substep n: tau = n/N, the current state has effective noise
        level (1-tau)*sigma_k, matching what the model was trained on.

        Args:
            input_dense: [B, T_in, C, Z, H, W]
            global_timesteps: unused, for interface compatibility
            num_refine_steps: optional override for K
            ode_steps: optional override for N (substeps per refinement k)

        Returns:
            output: [B, T_out, C, Z, H, W]
        """
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = self.t_out
        device = input_dense.device

        # Reshape condition
        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')

        # Step k=0: MSE initial prediction (from zeros)
        zeros_input = torch.zeros(
            B, self.out_channels, Z, H, W,
            dtype=input_dense.dtype, device=device
        )
        k_time = torch.zeros(B, device=device)
        y_coarse = self.model(zeros_input, x, k_time * self.time_multiplier)

        # Steps k=1,...,K: FM refinement via multi-step ODE integration
        y = y_coarse
        K = num_refine_steps if num_refine_steps is not None else self.num_refine_steps
        N = ode_steps if ode_steps is not None else self.ode_steps
        dt = 1.0 / N  # step size in tau space

        for k in range(1, K + 1):
            sigma_k = self.scheduler.sigmas[k].item()

            # Start: perturb current prediction (tau=0, full noise level sigma_k)
            noise = _generate_noise(self.scheduler.noise_type, y.shape, device,
                                    **self.scheduler.noise_kwargs)
            y_current = y + sigma_k * noise

            # Multi-step Euler integration from tau=0 to tau=1
            k_time = torch.full((B,), float(k), device=device)
            for n in range(N):
                # Predict normalized velocity at current point
                v_pred_norm = self.model(y_current, x, k_time * self.time_multiplier)
                # Euler step: move dt along the flow
                y_current = self.scheduler.refine_step(
                    y_current, v_pred_norm, sigma_k, dt=dt
                )

            y = y_current

        # Reshape output
        output = rearrange(y, 'b (t c) z h w -> b t c z h w', t=T_out, c=C)

        return output
