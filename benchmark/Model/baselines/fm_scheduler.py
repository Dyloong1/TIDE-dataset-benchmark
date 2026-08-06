"""
Flow Matching Scheduler for PDE Refinement.

Implements conditional flow matching with straight interpolation paths
and Euler ODE integration for sampling.

Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
"""

import torch


class FMScheduler:
    """
    Flow Matching scheduler for PDE refinement.

    Training: sample t ~ U(0,1), interpolate x_t = (1-t)*z + t*y_clean,
              velocity target v = y_clean - z
    Inference: Euler integration from noise (t=0) to data (t=1) in K steps
    """

    def __init__(self, num_steps=3):
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps

    def get_train_tuple(self, y_clean):
        """
        Sample training tuple for flow matching.

        Args:
            y_clean: [B, C, Z, H, W] clean target

        Returns:
            x_t: [B, C, Z, H, W] interpolated sample
            t: [B,] sampled time
            v_target: [B, C, Z, H, W] velocity target
        """
        B = y_clean.shape[0]
        device = y_clean.device

        # Sample t ~ Uniform(0, 1)
        t = torch.rand(B, device=device)
        t_expand = t.view(-1, *[1 for _ in range(y_clean.ndim - 1)])

        # Sample noise
        z = torch.randn_like(y_clean)

        # Interpolate: x_t = (1-t)*z + t*y_clean
        x_t = (1 - t_expand) * z + t_expand * y_clean

        # Velocity target: v = y_clean - z
        v_target = y_clean - z

        return x_t, t, v_target

    def euler_step(self, x_t, v_pred, dt):
        """Single Euler integration step."""
        return x_t + v_pred * dt

    def sample(self, model, condition, shape, device, dtype=torch.float32):
        """
        Full sampling loop: K Euler steps from noise to prediction.

        Args:
            model: network that takes (x_t, condition, t) and returns velocity
            condition: [B, C_cond, Z, H, W] conditioning signal
            shape: output shape [B, C_out, Z, H, W]
            device: torch device
            dtype: torch dtype

        Returns:
            x: [B, C_out, Z, H, W] generated sample
        """
        x = torch.randn(shape, device=device, dtype=dtype)

        for i in range(self.num_steps):
            t_i = torch.full((shape[0],), i / self.num_steps, device=device)
            v_pred = model(x, condition, t_i)
            x = self.euler_step(x, v_pred, self.dt)

        return x
