"""
Constrained inference samplers for PCFM ablation experiments.

All samplers share the same interface:
    sampler(model, input_dense, constraint_set) -> prediction [B, T, C, Z, H, W]

Samplers that modify the ODE loop replicate the FM-Hybrid forward() logic
internally rather than calling model.forward(), to intercept at each substep.
"""

import torch
import time
from einops import rearrange
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """Base class for all inference samplers."""

    def __init__(self, **kwargs):
        self.nfe = 0  # Network function evaluations counter
        self.wall_time = 0.0

    @abstractmethod
    def __call__(self, model, input_dense, constraint_set):
        """Run inference.

        Args:
            model: FMHybridRefiner3D instance
            input_dense: [B, T_in, C, Z, H, W] normalized input
            constraint_set: ConstraintSet instance (may be None for B0)

        Returns:
            prediction: [B, T_out, C, Z, H, W] normalized output
        """
        pass

    def _get_model_params(self, model):
        """Extract ODE loop parameters from model."""
        K = model.num_refine_steps
        N = model.ode_steps
        dt = 1.0 / N
        time_mult = model.time_multiplier
        scheduler = model.scheduler
        unet = model.model  # RefinerUNet3D backbone
        T_out = model.t_out
        C = model.in_channels
        return K, N, dt, time_mult, scheduler, unet, T_out, C

    def _k0_prediction(self, model, input_dense):
        """Run k=0 MSE prediction (shared by all samplers).

        Returns:
            y_coarse: [B, T*C, Z, H, W] merged tensor
            x_cond: [B, T_in*C, Z, H, W] condition tensor
        """
        B, T_in, C, Z, H, W = input_dense.shape
        device = input_dense.device

        x_cond = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')
        zeros_input = torch.zeros(B, model.t_out * C, Z, H, W,
                                  dtype=input_dense.dtype, device=device)
        k_time = torch.zeros(B, device=device)
        y_coarse = model.model(zeros_input, x_cond, k_time * model.time_multiplier)
        self.nfe += 1
        return y_coarse, x_cond


# ---------------------------------------------------------------------------
# B0: Vanilla FM (no constraints)
# ---------------------------------------------------------------------------

class VanillaSampler(BaseSampler):
    """B0: Standard FM-Hybrid inference without any constraints."""

    def __call__(self, model, input_dense, constraint_set):
        self.nfe = 0
        t0 = time.time()

        # Just call model.forward() directly
        with torch.no_grad():
            output = model(input_dense)

        K = model.num_refine_steps
        N = model.ode_steps
        self.nfe = 1 + K * N  # k=0 + K refinement steps × N substeps
        self.wall_time = time.time() - t0
        return output


# ---------------------------------------------------------------------------
# B4: Terminal-only projection
# ---------------------------------------------------------------------------

class TerminalProjectionSampler(BaseSampler):
    """B4: Run vanilla inference, then project final output once."""

    def __call__(self, model, input_dense, constraint_set):
        self.nfe = 0
        t0 = time.time()

        with torch.no_grad():
            output = model(input_dense)

        K = model.num_refine_steps
        N = model.ode_steps
        self.nfe = 1 + K * N

        # Project final output
        output = constraint_set.project(output)

        self.wall_time = time.time() - t0
        return output


# ---------------------------------------------------------------------------
# B1/C1: PCFM Sampler (every-step shooting + projection + OT inverse)
# ---------------------------------------------------------------------------

class PCFMSampler(BaseSampler):
    """PCFM: Shooting → Projection → OT Inverse → Relaxation at each ODE step.

    At each substep n of refinement step k:
    1. Standard Euler step
    2. Shoot to terminal: extrapolate current velocity to τ=1
    3. Project terminal estimate to satisfy constraints
    4. Map back via OT interpolation: x̃_τ = (1-τ)*x₀ + τ*x̃₁
    5. Optional relaxation correction
    """

    def __init__(self, relaxation_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.relaxation_lambda = relaxation_lambda

    def __call__(self, model, input_dense, constraint_set):
        self.nfe = 0
        t0 = time.time()

        K, N, dt, time_mult, scheduler, unet, T_out, C_ch = self._get_model_params(model)

        with torch.no_grad():
            # k=0: MSE prediction
            y, x_cond = self._k0_prediction(model, input_dense)

            # k=1..K: Constrained ODE refinement
            for k in range(1, K + 1):
                sigma_k = scheduler.sigmas[k].item()
                B = y.shape[0]
                device = y.device

                # Initialize with noise
                noise = torch.randn_like(y)
                y_0 = y + sigma_k * noise  # Initial noised state (τ=0)
                y_current = y_0.clone()

                k_time = torch.full((B,), float(k), device=device)

                for n in range(N):
                    tau = n / N
                    tau_next = (n + 1) / N

                    # Model prediction
                    v_pred = unet(y_current, x_cond, k_time * time_mult)
                    self.nfe += 1

                    # Standard Euler step
                    y_euler = y_current + v_pred * sigma_k * dt

                    # --- PCFM constraint correction ---
                    # 1. Shoot to terminal: estimate x₁ from current velocity
                    remaining_tau = 1.0 - tau_next
                    y_terminal_est = y_euler + remaining_tau * v_pred * sigma_k

                    # 2. Project terminal estimate
                    y_terminal_proj = constraint_set.project_merged(
                        y_terminal_est, T=T_out, C=C_ch
                    )

                    # 3. OT inverse: map back to current time
                    y_corrected = (1.0 - tau_next) * y_0 + tau_next * y_terminal_proj

                    # 4. Relaxation: blend between uncorrected Euler and
                    # PCFM-corrected result.
                    # λ=1.0 → pure PCFM correction (default for B1)
                    # λ=0.0 → pure Euler (no correction, used by C8)
                    # 0<λ<1 → partial correction
                    lam = self.relaxation_lambda
                    y_current = (1.0 - lam) * y_euler + lam * y_corrected

                y = y_current

        output = rearrange(y, 'b (t c) z h w -> b t c z h w', t=T_out, c=C_ch)
        self.wall_time = time.time() - t0
        return output


# ---------------------------------------------------------------------------
# C6: Hard Projection Sampler (project intermediate state directly)
# ---------------------------------------------------------------------------

class HardProjectionSampler(BaseSampler):
    """C6: Euler step → direct projection on intermediate state at each substep."""

    def __call__(self, model, input_dense, constraint_set):
        self.nfe = 0
        t0 = time.time()

        K, N, dt, time_mult, scheduler, unet, T_out, C_ch = self._get_model_params(model)

        with torch.no_grad():
            y, x_cond = self._k0_prediction(model, input_dense)

            for k in range(1, K + 1):
                sigma_k = scheduler.sigmas[k].item()
                B = y.shape[0]
                device = y.device

                noise = torch.randn_like(y)
                y_current = y + sigma_k * noise

                k_time = torch.full((B,), float(k), device=device)

                for n in range(N):
                    v_pred = unet(y_current, x_cond, k_time * time_mult)
                    self.nfe += 1

                    # Euler step
                    y_current = y_current + v_pred * sigma_k * dt

                    # Hard projection directly on intermediate state
                    y_current = constraint_set.project_merged(
                        y_current, T=T_out, C=C_ch
                    )

                y = y_current

        output = rearrange(y, 'b (t c) z h w -> b t c z h w', t=T_out, c=C_ch)
        self.wall_time = time.time() - t0
        return output


# ---------------------------------------------------------------------------
# C8: PCFM without Relaxation (λ=0)
# ---------------------------------------------------------------------------

class PCFMNoRelaxSampler(PCFMSampler):
    """C8: Same as PCFM but without relaxation correction."""

    def __init__(self, **kwargs):
        super().__init__(relaxation_lambda=0.0, **kwargs)


# ---------------------------------------------------------------------------
# C11: CCFM Sampler (time-adaptive constraint tightening)
# ---------------------------------------------------------------------------

class CCFMSampler(BaseSampler):
    """CCFM: Chance-Constrained Flow Matching.

    At each ODE step, tighten constraint bounds based on time schedule φ(τ).
    No shooting, no OT inverse — directly project intermediate state
    with progressively tighter constraints.

    The key idea: at time τ, the intermediate state x_τ = (1-τ)*x₀ + τ*x₁
    has noise contribution (1-τ)*x₀. The constraint tightening accounts for
    this noise by relaxing constraints early (when noise is large) and
    tightening late (when close to clean).

    Schedule: φ(τ) = τ^n where n controls tightening speed.
    φ(0) = 0 (no constraint at start), φ(1) = 1 (fully constrained at end).
    n=1: linear tightening. n>1: delayed tightening. n<1: aggressive early tightening.
    """

    def __init__(self, schedule_n=1.0, **kwargs):
        super().__init__(**kwargs)
        self.schedule_n = schedule_n

    def __call__(self, model, input_dense, constraint_set):
        self.nfe = 0
        t0 = time.time()

        K, N, dt, time_mult, scheduler, unet, T_out, C_ch = self._get_model_params(model)

        with torch.no_grad():
            y, x_cond = self._k0_prediction(model, input_dense)

            for k in range(1, K + 1):
                sigma_k = scheduler.sigmas[k].item()
                B = y.shape[0]
                device = y.device

                noise = torch.randn_like(y)
                y_current = y + sigma_k * noise

                k_time = torch.full((B,), float(k), device=device)

                for n in range(N):
                    tau = n / N
                    tau_next = (n + 1) / N

                    v_pred = unet(y_current, x_cond, k_time * time_mult)
                    self.nfe += 1

                    # Euler step
                    y_current = y_current + v_pred * sigma_k * dt

                    # CCFM: time-adaptive projection
                    # φ(τ) controls constraint strength: 0 at start, 1 at end
                    phi = tau_next ** self.schedule_n

                    # Blend between unconstrained and fully constrained:
                    # y_proj = project(y_current)
                    # y_current = (1-φ)*y_current + φ*y_proj
                    # When φ≈0 (early): mostly unconstrained
                    # When φ≈1 (late): fully constrained
                    if phi > 1e-6:
                        y_proj = constraint_set.project_merged(
                            y_current, T=T_out, C=C_ch
                        )
                        y_current = (1.0 - phi) * y_current + phi * y_proj

                y = y_current

        output = rearrange(y, 'b (t c) z h w -> b t c z h w', t=T_out, c=C_ch)
        self.wall_time = time.time() - t0
        return output


# ---------------------------------------------------------------------------
# D8c: Leray Every Step (divergence-free only, every ODE substep)
# ---------------------------------------------------------------------------

class LerayEveryStepSampler(BaseSampler):
    """D8c: Apply Leray projection after every ODE substep."""

    def __call__(self, model, input_dense, constraint_set):
        self.nfe = 0
        t0 = time.time()

        K, N, dt, time_mult, scheduler, unet, T_out, C_ch = self._get_model_params(model)

        with torch.no_grad():
            y, x_cond = self._k0_prediction(model, input_dense)

            for k in range(1, K + 1):
                sigma_k = scheduler.sigmas[k].item()
                B = y.shape[0]
                device = y.device

                noise = torch.randn_like(y)
                y_current = y + sigma_k * noise

                k_time = torch.full((B,), float(k), device=device)

                for n in range(N):
                    v_pred = unet(y_current, x_cond, k_time * time_mult)
                    self.nfe += 1
                    y_current = y_current + v_pred * sigma_k * dt

                    # Leray projection only (divergence-free)
                    y_current = constraint_set.project_velocity_only_merged(
                        y_current, T=T_out, C=C_ch
                    )

                y = y_current

        output = rearrange(y, 'b (t c) z h w -> b t c z h w', t=T_out, c=C_ch)
        self.wall_time = time.time() - t0
        return output


# ---------------------------------------------------------------------------
# D8d: Leray + OT Inverse
# ---------------------------------------------------------------------------

class LerayOTSampler(BaseSampler):
    """D8d: Leray projection + OT inverse mapping at each substep.

    Similar to PCFM but only uses Leray for divergence-free (no GN).
    """

    def __call__(self, model, input_dense, constraint_set):
        self.nfe = 0
        t0 = time.time()

        K, N, dt, time_mult, scheduler, unet, T_out, C_ch = self._get_model_params(model)

        with torch.no_grad():
            y, x_cond = self._k0_prediction(model, input_dense)

            for k in range(1, K + 1):
                sigma_k = scheduler.sigmas[k].item()
                B = y.shape[0]
                device = y.device

                noise = torch.randn_like(y)
                y_0 = y + sigma_k * noise
                y_current = y_0.clone()

                k_time = torch.full((B,), float(k), device=device)

                for n in range(N):
                    tau_next = (n + 1) / N

                    v_pred = unet(y_current, x_cond, k_time * time_mult)
                    self.nfe += 1

                    y_euler = y_current + v_pred * sigma_k * dt

                    # Shoot to terminal
                    remaining = 1.0 - tau_next
                    y_term = y_euler + remaining * v_pred * sigma_k

                    # Leray projection on terminal estimate
                    y_term_proj = constraint_set.project_velocity_only_merged(
                        y_term, T=T_out, C=C_ch
                    )

                    # OT inverse
                    y_current = (1.0 - tau_next) * y_0 + tau_next * y_term_proj

                y = y_current

        output = rearrange(y, 'b (t c) z h w -> b t c z h w', t=T_out, c=C_ch)
        self.wall_time = time.time() - t0
        return output


# ---------------------------------------------------------------------------
# D8e: Leray for div-free + PCFM for remaining constraints
# ---------------------------------------------------------------------------

class LerayPlusPCFMSampler(BaseSampler):
    """D8e: Leray handles divergence-free, PCFM handles energy/momentum/IC.

    Split projection: efficient Leray for high-dim div-free constraint,
    GN/PCFM for low-dim global constraints.
    """

    def __init__(self, **kwargs):
        # Remove relaxation_lambda from kwargs if passed (not used by this sampler)
        kwargs.pop('relaxation_lambda', None)
        super().__init__(**kwargs)

    def __call__(self, model, input_dense, constraint_set):
        self.nfe = 0
        t0 = time.time()

        K, N, dt, time_mult, scheduler, unet, T_out, C_ch = self._get_model_params(model)

        with torch.no_grad():
            y, x_cond = self._k0_prediction(model, input_dense)

            for k in range(1, K + 1):
                sigma_k = scheduler.sigmas[k].item()
                B = y.shape[0]
                device = y.device

                noise = torch.randn_like(y)
                y_0 = y + sigma_k * noise
                y_current = y_0.clone()

                k_time = torch.full((B,), float(k), device=device)

                for n in range(N):
                    tau_next = (n + 1) / N

                    v_pred = unet(y_current, x_cond, k_time * time_mult)
                    self.nfe += 1

                    y_euler = y_current + v_pred * sigma_k * dt

                    # Shoot to terminal
                    remaining = 1.0 - tau_next
                    y_term = y_euler + remaining * v_pred * sigma_k

                    # Full projection (Leray + momentum + energy + IC)
                    # project_merged → project_single_frame handles Leray internally
                    y_term_proj = constraint_set.project_merged(
                        y_term, T=T_out, C=C_ch
                    )

                    # OT inverse
                    y_current = (1.0 - tau_next) * y_0 + tau_next * y_term_proj

                y = y_current

        output = rearrange(y, 'b (t c) z h w -> b t c z h w', t=T_out, c=C_ch)
        self.wall_time = time.time() - t0
        return output


# ---------------------------------------------------------------------------
# D8f: Leray on velocity field (project model output before Euler step)
# ---------------------------------------------------------------------------

class LerayVFieldSampler(BaseSampler):
    """D8f: Apply Leray to model's velocity output before Euler step.

    Instead of projecting the state, project the predicted velocity field
    so the ODE always evolves along divergence-free directions.
    If the initial state is div-free, all subsequent states will be too.
    """

    def __call__(self, model, input_dense, constraint_set):
        self.nfe = 0
        t0 = time.time()

        K, N, dt, time_mult, scheduler, unet, T_out, C_ch = self._get_model_params(model)

        with torch.no_grad():
            y, x_cond = self._k0_prediction(model, input_dense)

            # Note: Do NOT project k=0 output here — the ablation tests
            # the effect of projecting velocity *during* the ODE loop only.
            # Projecting k=0 would confound the comparison with other samplers.

            for k in range(1, K + 1):
                sigma_k = scheduler.sigmas[k].item()
                B = y.shape[0]
                device = y.device

                noise = torch.randn_like(y)
                y_current = y + sigma_k * noise

                k_time = torch.full((B,), float(k), device=device)

                for n in range(N):
                    v_pred = unet(y_current, x_cond, k_time * time_mult)
                    self.nfe += 1

                    # Project velocity field to be divergence-free
                    v_pred = constraint_set.leray_on_velocity_field(
                        v_pred, T=T_out, C=C_ch
                    )

                    y_current = y_current + v_pred * sigma_k * dt

                y = y_current

        output = rearrange(y, 'b (t c) z h w -> b t c z h w', t=T_out, c=C_ch)
        self.wall_time = time.time() - t0
        return output


# ---------------------------------------------------------------------------
# Sampler registry
# ---------------------------------------------------------------------------

SAMPLER_REGISTRY = {
    'vanilla': VanillaSampler,
    'terminal_projection': TerminalProjectionSampler,
    'pcfm': PCFMSampler,
    'hard_projection': HardProjectionSampler,
    'pcfm_no_relax': PCFMNoRelaxSampler,
    'ccfm': CCFMSampler,
    'leray_every_step': LerayEveryStepSampler,
    'leray_ot': LerayOTSampler,
    'leray_plus_pcfm': LerayPlusPCFMSampler,
    'leray_vfield': LerayVFieldSampler,
}
