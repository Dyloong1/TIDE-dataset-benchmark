"""Plain FNO3d — the classic Fourier Neural Operator (Li et al. 2021), no U-Net.

The canonical FNO anchor for the turbulence benchmark: a stack of spectral-conv layers
(global) + pointwise conv (local) at FULL resolution, no down/up-sampling. This is the
most-cited operator-learning baseline and the natural reference point against which U-FNO
(adds a U-Net) and TFNO (tensorizes the spectral weights) are improvements.

Interface matches the other baselines (config-attr driven, getattr-with-defaults):
  input_dense : (B, T_in, C, Z, H, W)  ->  output : (B, T_out, C, Z, H, W)
Self-contained (only torch + einops), no neuraloperator dependency.
"""
import torch
import torch.nn as nn
import torch.fft as fft
import torch.utils.checkpoint
from einops import rearrange


class SpectralConv3d(nn.Module):
    """3D spectral convolution: multiply the lowest Fourier modes by learned complex
    weights (the 4 corner blocks of the rfftn spectrum, matching the FNO paper /
    the U-FNO implementation in this repo so results are comparable)."""

    def __init__(self, in_channels, out_channels, modes_z, modes_h, modes_w):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_z = modes_z
        self.modes_h = modes_h
        self.modes_w = modes_w
        scale = 1.0 / (in_channels * out_channels)
        # 4 corner blocks (±z, ±h, +w) of the rfftn half-spectrum
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_z, modes_h, modes_w, dtype=torch.cfloat))
            for _ in range(4)
        ])

    @staticmethod
    def _compl_mul3d(inp, w):
        return torch.einsum("bizxy,iozxy->bozxy", inp, w)

    def forward(self, x):
        x_float = x.float()
        B = x_float.shape[0]
        x_ft = fft.rfftn(x_float, dim=[-3, -2, -1])
        Z, H, Wf = x_ft.shape[-3], x_ft.shape[-2], x_ft.shape[-1]
        mz = min(self.modes_z, Z // 2)
        mh = min(self.modes_h, H // 2)
        mw = min(self.modes_w, Wf)
        out_ft = torch.zeros(B, self.out_channels, Z, H, Wf, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :mz, :mh, :mw] = self._compl_mul3d(x_ft[:, :, :mz, :mh, :mw], self.weights[0][:, :, :mz, :mh, :mw])
        out_ft[:, :, -mz:, :mh, :mw] = self._compl_mul3d(x_ft[:, :, -mz:, :mh, :mw], self.weights[1][:, :, :mz, :mh, :mw])
        out_ft[:, :, :mz, -mh:, :mw] = self._compl_mul3d(x_ft[:, :, :mz, -mh:, :mw], self.weights[2][:, :, :mz, :mh, :mw])
        out_ft[:, :, -mz:, -mh:, :mw] = self._compl_mul3d(x_ft[:, :, -mz:, -mh:, :mw], self.weights[3][:, :, :mz, :mh, :mw])
        x_out = fft.irfftn(out_ft, s=(x.shape[-3], x.shape[-2], x.shape[-1]), dim=[-3, -2, -1])
        return x_out.to(x.dtype)


class FNOLayer(nn.Module):
    """One FNO layer: spectral conv (global) + 1x1 conv (local) + GELU, full-resolution."""

    def __init__(self, width, modes_z, modes_h, modes_w):
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes_z, modes_h, modes_w)
        self.local = nn.Conv3d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral(x) + self.local(x))


class FNO3d(nn.Module):
    """Plain FNO: lifting -> N spectral layers (full res) -> projection.

    Input  (B, T_in,  C, Z, H, W)
    Output (B, T_out, C, Z, H, W)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = getattr(config, 'IN_CHANNELS', None) or len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        # output channel count may differ from input (pressure 3->1, sgs 4->6); default = in
        self.out_channels = getattr(config, 'OUT_CHANNELS', None) or self.in_channels
        width = getattr(config, 'FNO_WIDTH', getattr(config, 'UFNO_WIDTH', 32))
        n_layers = getattr(config, 'FNO_LAYERS', 4)
        self.grad_ckpt = bool(getattr(config, 'GRAD_CKPT', False))
        mz = getattr(config, 'FNO_MODES_Z', 16)
        mh = getattr(config, 'FNO_MODES_H', 24)
        mw = getattr(config, 'FNO_MODES_W', 24)

        lifting_in = self.t_in * self.in_channels
        self.lifting = nn.Sequential(
            nn.Conv3d(lifting_in, width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=1),
        )
        self.layers = nn.ModuleList([FNOLayer(width, mz, mh, mw) for _ in range(n_layers)])
        projection_out = self.t_out * self.out_channels
        self.projection = nn.Sequential(
            nn.Conv3d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width * 2, projection_out, kernel_size=1),
        )
        self._print_info(width, n_layers)

    def _print_info(self, width, n_layers):
        total = sum(p.numel() for p in self.parameters())
        print(f"\n[FNO3d] width={width} layers={n_layers} "
              f"in=({self.t_in},{self.in_channels}) out=({self.t_out},{self.out_channels}) params={total:,}")

    def forward(self, input_dense, global_timesteps=None):
        B, T_in, C, Z, H, W = input_dense.shape
        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')
        x = self.lifting(x)
        for layer in self.layers:
            if self.grad_ckpt and self.training:
                # native-256^3 training: activations are 8x the 128^3 budget; recompute per
                # block instead of storing. use_reentrant=False is required under bf16 autocast.
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.projection(x)
        return rearrange(x, 'b (t c) z h w -> b t c z h w', t=self.t_out, c=self.out_channels)
