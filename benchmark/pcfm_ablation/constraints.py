"""
Constraint framework for physics-constrained inference.

Handles constraint target computation from ground truth,
residual evaluation, and projection pipelines.

All physics operations are done in PHYSICAL (denormalized) space.
The project() method handles denormalize → constrain → renormalize.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict

from .physics_utils import (
    load_norm_stats_tensors,
    denormalize_field, renormalize_field,
    extract_velocity, set_velocity,
    leray_projection_3d,
    compute_divergence_3d, divergence_stats,
    compute_kinetic_energy, compute_momentum,
    correct_momentum, correct_energy,
)


@dataclass
class ConstraintConfig:
    """Which physics constraints to apply."""
    use_divergence_free: bool = False
    use_momentum: bool = False
    use_energy: bool = False
    use_ic: bool = False  # Enforce first predicted timestep continuity


class ConstraintSet:
    """Manages physics constraints and their target values.

    Targets are computed per-sample from ground truth at evaluation time.
    The project() method applies all active constraints in the correct order.
    """

    def __init__(self, config: ConstraintConfig, norm_stats_dict: dict,
                 grid_shape: tuple, device: str = 'cpu'):
        """
        Args:
            config: Which constraints to apply
            norm_stats_dict: {'u': {'mean': ..., 'std': ...}, ...}
            grid_shape: (Z, H, W)
            device: torch device
        """
        self.config = config
        self.grid_shape = grid_shape
        self.device = device

        # Precompute normalization tensors
        self.norm_mean, self.norm_std = load_norm_stats_tensors(
            norm_stats_dict, device=device
        )

        # Per-sample targets (set before each evaluation)
        self.target_energy = None     # [B, T]
        self.target_momentum = None   # [B, T, 3]
        self.ic_frame = None          # [B, C, Z, H, W] last input frame (normalized)

    def set_targets_from_gt(self, gt_output_normalized, input_dense=None):
        """Compute constraint targets from ground truth.

        Always computes energy and momentum targets (for residual evaluation),
        but only applies them as constraints if config flags are set.

        Args:
            gt_output_normalized: [B, T, C=4, Z, H, W] normalized GT output
            input_dense: [B, T_in, C=4, Z, H, W] normalized input (for IC)
        """
        device = gt_output_normalized.device

        # Denormalize GT to physical space
        gt_phys = denormalize_field(gt_output_normalized, self.norm_mean.to(device),
                                     self.norm_std.to(device))

        B, T, C, Z, H, W = gt_phys.shape

        # Always compute energy/momentum targets for residual evaluation
        vel_phys = gt_phys[:, :, :3]  # [B, T, 3, Z, H, W]
        energies = []
        momenta = []
        for t in range(T):
            ek = compute_kinetic_energy(vel_phys[:, t])  # [B]
            energies.append(ek)
            mom = compute_momentum(vel_phys[:, t])  # [B, 3]
            momenta.append(mom)
        self.target_energy = torch.stack(energies, dim=1).to(device)  # [B, T]
        self.target_momentum = torch.stack(momenta, dim=1).to(device)  # [B, T, 3]

        if self.config.use_ic and input_dense is not None:
            # Last input frame as IC target (normalized, will be used directly)
            self.ic_frame = input_dense[:, -1].to(device)  # [B, C, Z, H, W]

    def compute_residuals(self, pred_normalized):
        """Compute constraint residuals for evaluation.

        Args:
            pred_normalized: [B, T, C=4, Z, H, W] normalized prediction

        Returns:
            dict of residual metrics
        """
        device = pred_normalized.device
        pred_phys = denormalize_field(pred_normalized, self.norm_mean.to(device),
                                      self.norm_std.to(device))
        B, T, C, Z, H, W = pred_phys.shape
        residuals = {}

        # Divergence
        div_maxs, div_means = [], []
        for t in range(T):
            vel = pred_phys[:, t, :3]
            d_max, d_mean = divergence_stats(vel)
            div_maxs.append(d_max)
            div_means.append(d_mean)
        residuals['div_max'] = torch.stack(div_maxs, dim=1).mean().item()
        residuals['div_mean'] = torch.stack(div_means, dim=1).mean().item()

        # Energy error (always computed — targets set in set_targets_from_gt)
        vel_phys = pred_phys[:, :, :3]
        pred_energies = []
        for t in range(T):
            ek = compute_kinetic_energy(vel_phys[:, t])
            pred_energies.append(ek)
        pred_energy = torch.stack(pred_energies, dim=1)
        if self.target_energy is not None:
            energy_err = ((pred_energy - self.target_energy.to(device)).abs()
                          / (self.target_energy.to(device).abs() + 1e-12))
            residuals['energy_err_pct'] = energy_err.mean().item() * 100
        else:
            residuals['energy_err_pct'] = float('nan')

        # Momentum error (always computed)
        pred_momenta = []
        for t in range(T):
            mom = compute_momentum(vel_phys[:, t])
            pred_momenta.append(mom)
        pred_mom = torch.stack(pred_momenta, dim=1)
        if self.target_momentum is not None:
            mom_diff = (pred_mom - self.target_momentum.to(device)).abs()
            # Use mean velocity difference (remove domain volume factor)
            # This gives a physically interpretable absolute error
            domain_vol = (2 * 3.14159265358979) ** 3
            mean_vel_err = mom_diff / domain_vol  # [B, T, 3]
            residuals['momentum_mean_vel_err'] = mean_vel_err.mean().item()
        else:
            residuals['momentum_mean_vel_err'] = float('nan')

        return residuals

    def project_single_frame(self, frame_normalized, timestep_idx=0):
        """Apply all active constraints to a single frame.

        Order: IC → Leray → Momentum → Energy
        IC is applied first (in normalized space), then physics constraints
        are applied on the resulting field. This ensures the IC frame also
        satisfies divergence-free and other physics constraints.

        Args:
            frame_normalized: [B, C=4, Z, H, W] normalized prediction for one timestep
            timestep_idx: which timestep (for per-timestep targets)

        Returns:
            frame_projected: [B, C=4, Z, H, W] normalized, constrained
        """
        device = frame_normalized.device
        mean = self.norm_mean.to(device)
        std = self.norm_std.to(device)

        # 0. IC enforcement (overwrite first timestep before physics constraints)
        if self.config.use_ic and self.ic_frame is not None and timestep_idx == 0:
            frame_normalized = self.ic_frame.to(device)

        # Denormalize to physical space
        frame_phys = denormalize_field(frame_normalized, mean, std)

        # Extract velocity (u, v, w) — only constrain velocity, not pressure
        vel_phys = frame_phys[:, :3]  # [B, 3, Z, H, W]

        # 1. Leray projection (divergence-free)
        if self.config.use_divergence_free:
            vel_phys = leray_projection_3d(vel_phys)

        # 2. Momentum correction (adjusts mean velocity)
        if self.config.use_momentum and self.target_momentum is not None:
            target_mom = self.target_momentum[:, timestep_idx].to(device)
            vel_phys = correct_momentum(vel_phys, target_mom)

        # 3. Energy scaling (uniform scaling preserves div-free and momentum direction)
        if self.config.use_energy and self.target_energy is not None:
            target_ek = self.target_energy[:, timestep_idx].to(device)
            vel_phys = correct_energy(vel_phys, target_ek)

        # Reassemble full field with corrected velocity + original pressure
        frame_phys = set_velocity(frame_phys, vel_phys, ndim=5)

        # Renormalize
        frame_projected = renormalize_field(frame_phys, mean, std)

        return frame_projected

    def project(self, pred_normalized):
        """Apply constraints to full prediction [B, T, C, Z, H, W].

        Projects each timestep independently.
        """
        B, T, C, Z, H, W = pred_normalized.shape
        result = pred_normalized.clone()
        for t in range(T):
            result[:, t] = self.project_single_frame(result[:, t], timestep_idx=t)
        return result

    def project_merged(self, y_merged, T=5, C=4):
        """Apply constraints to merged tensor [B, T*C, Z, H, W].

        Used inside samplers where the tensor is in merged (channel-stacked) format.
        Reshapes to [B, T, C, Z, H, W], projects, reshapes back.
        """
        B = y_merged.shape[0]
        Z, H, W = y_merged.shape[2:]
        y_tc = y_merged.reshape(B, T, C, Z, H, W)
        y_proj = self.project(y_tc)
        return y_proj.reshape(B, T * C, Z, H, W)

    def project_velocity_only_merged(self, y_merged, T=5, C=4):
        """Apply only Leray projection to velocity channels in merged format.

        Faster than full project() — only does divergence-free projection.
        Used for D8c/D8f experiments (Leray every step).
        """
        if not self.config.use_divergence_free:
            return y_merged

        B = y_merged.shape[0]
        Z, H, W = y_merged.shape[2:]
        device = y_merged.device
        mean = self.norm_mean.to(device)
        std = self.norm_std.to(device)

        y_tc = y_merged.reshape(B, T, C, Z, H, W)
        result = y_tc.clone()

        for t in range(T):
            frame = y_tc[:, t]  # [B, C, Z, H, W]
            frame_phys = denormalize_field(frame, mean, std)
            vel_phys = frame_phys[:, :3]
            vel_proj = leray_projection_3d(vel_phys)
            frame_phys = set_velocity(frame_phys, vel_proj, ndim=5)
            result[:, t] = renormalize_field(frame_phys, mean, std)

        return result.reshape(B, T * C, Z, H, W)

    def leray_on_velocity_field(self, v_pred_merged, T=5, C=4):
        """Apply Leray projection to velocity prediction (model output).

        Used for D8f: project the velocity field v_θ before Euler step.
        v_pred is in normalized space and represents model output (not physical velocity).

        We denormalize, project velocity channels, renormalize.
        """
        return self.project_velocity_only_merged(v_pred_merged, T=T, C=C)
