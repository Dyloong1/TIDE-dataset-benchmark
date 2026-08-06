"""TFNO3d — Tensorized Fourier Neural Operator (Kossaifi et al., arXiv:2310.00120).

Same architecture as plain FNO3d, but the spectral-conv weight tensor
W[in, out, mz, mh, mw] (complex) is parameterized by a low-rank **Tucker factorization**
instead of being stored densely. This cuts spectral parameters from
O(in*out*mz*mh*mw) to O(rank^5 + rank*(in+out+mz+mh+mw)) and is the standard recent FNO
variant used as a baseline in The Well / PDEBench. Self-contained (torch + einops).

Interface matches the other baselines:
  input_dense (B, T_in, C, Z, H, W) -> output (B, T_out, C, Z, H, W)
"""
import torch
import torch.nn as nn
import torch.fft as fft
import torch.utils.checkpoint
from einops import rearrange


class TuckerSpectralConv3d(nn.Module):
    """Spectral conv with a Tucker-factorized complex weight per corner block.

    A dense corner weight is W[i,o,z,h,w]. We store a Tucker core G[r,r,r,r,r] (complex)
    plus 5 real factor matrices (one per mode: in, out, mz, mh, mw) and reconstruct
    W = G x1 Fin x2 Fout x3 Fz x4 Fh x5 Fw on the fly. rank << dims = low-rank weight.
    """

    def __init__(self, in_channels, out_channels, modes_z, modes_h, modes_w, rank=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_z, self.modes_h, self.modes_w = modes_z, modes_h, modes_w
        r = min(rank, in_channels, out_channels, modes_z, modes_h, modes_w)
        self.rank = max(2, r)
        scale = 1.0 / (in_channels * out_channels)
        # one Tucker decomposition per corner block (4 corners, like FNO3d)
        self.cores = nn.ParameterList([
            nn.Parameter(scale * torch.rand(self.rank, self.rank, self.rank, self.rank, self.rank, dtype=torch.cfloat))
            for _ in range(4)
        ])
        # real factor matrices: each corner gets its own 5 factors (in, out, mz, mh, mw),
        # stored flat as 4*5=20 params; corner c uses self._fac[c*5 : c*5+5].
        def _factors():
            return [
                nn.Parameter(torch.randn(in_channels, self.rank) * 0.1),
                nn.Parameter(torch.randn(out_channels, self.rank) * 0.1),
                nn.Parameter(torch.randn(modes_z, self.rank) * 0.1),
                nn.Parameter(torch.randn(modes_h, self.rank) * 0.1),
                nn.Parameter(torch.randn(modes_w, self.rank) * 0.1),
            ]
        self._fac = nn.ParameterList()
        for _ in range(4):
            for p in _factors():
                self._fac.append(p)

    def _reconstruct(self, c):
        """Rebuild dense corner weight from Tucker core+factors for corner c (0..3)."""
        core = self.cores[c]
        f = self._fac[c * 5:(c + 1) * 5]  # [Fin, Fout, Fz, Fh, Fw]
        Fin, Fout, Fz, Fh, Fw = [x.to(core.dtype) for x in f]
        # W[i,o,z,h,w] = sum_{abcde} core[a,b,c,d,e] Fin[i,a] Fout[o,b] Fz[z,c] Fh[h,d] Fw[w,e]
        w = torch.einsum('abcde,ia->ibcde', core, Fin)
        w = torch.einsum('ibcde,ob->iocde', w, Fout)
        w = torch.einsum('iocde,zc->iozde', w, Fz)
        w = torch.einsum('iozde,hd->iozhe', w, Fh)
        w = torch.einsum('iozhe,we->iozhw', w, Fw)
        return w  # (in, out, mz, mh, mw) complex

    @staticmethod
    def _compl_mul3d(inp, w):
        return torch.einsum("bizxy,iozxy->bozxy", inp, w)

    def _apply_factorized(self, x_slice, c, mz, mh, mw):
        """Contract the input through the Tucker factors WITHOUT reconstructing the dense
        corner weight. Mathematically identical to _compl_mul3d(x, _reconstruct(c)) — the
        same multilinear form evaluated in a different association order (fp round-off only;
        pinned by test_tfno_factorized_equivalence). Why: the dense path rebuilds W and runs
        an in×out contraction over the mode grid every forward AND re-does all of it in the
        GRAD_CKPT backward recompute; this path streams through rank-width tensors instead
        (rank≤12 vs width≈40), cutting both the reconstruct einsum chain and the peak
        activations of the spectral step."""
        core = self.cores[c]
        Fin, Fout, Fz, Fh, Fw = [p.to(core.dtype) for p in self._fac[c * 5:(c + 1) * 5]]
        # k[a, o_r, z, h, w]: rank-space mode kernel (tiny: r^2 * modes^3)
        k = torch.einsum("abcde,zc,hd,we->abzhw",
                         core, Fz[:mz], Fh[:mh], Fw[:mw])
        xa = torch.einsum("bizhw,ia->bazhw", x_slice, Fin)     # channels -> rank_in
        t = torch.einsum("bazhw,arzhw->brzhw", xa, k)          # rank-space pointwise mode mul
        return torch.einsum("brzhw,or->bozhw", t, Fout)        # rank_out -> channels

    def forward(self, x):
        x_float = x.float()
        B = x_float.shape[0]
        x_ft = fft.rfftn(x_float, dim=[-3, -2, -1])
        Z, H, Wf = x_ft.shape[-3], x_ft.shape[-2], x_ft.shape[-1]
        mz = min(self.modes_z, Z // 2)
        mh = min(self.modes_h, H // 2)
        mw = min(self.modes_w, Wf)
        out_ft = torch.zeros(B, self.out_channels, Z, H, Wf, dtype=torch.cfloat, device=x.device)
        if getattr(self, "factorized_apply", True):
            # default path since 2026-07-17: stream through factors, never build dense W
            # (math-identical to the reconstruct path; see _apply_factorized docstring)
            out_ft[:, :, :mz, :mh, :mw] = self._apply_factorized(x_ft[:, :, :mz, :mh, :mw], 0, mz, mh, mw)
            out_ft[:, :, -mz:, :mh, :mw] = self._apply_factorized(x_ft[:, :, -mz:, :mh, :mw], 1, mz, mh, mw)
            out_ft[:, :, :mz, -mh:, :mw] = self._apply_factorized(x_ft[:, :, :mz, -mh:, :mw], 2, mz, mh, mw)
            out_ft[:, :, -mz:, -mh:, :mw] = self._apply_factorized(x_ft[:, :, -mz:, -mh:, :mw], 3, mz, mh, mw)
        else:
            w0 = self._reconstruct(0)[:, :, :mz, :mh, :mw]
            w1 = self._reconstruct(1)[:, :, :mz, :mh, :mw]
            w2 = self._reconstruct(2)[:, :, :mz, :mh, :mw]
            w3 = self._reconstruct(3)[:, :, :mz, :mh, :mw]
            out_ft[:, :, :mz, :mh, :mw] = self._compl_mul3d(x_ft[:, :, :mz, :mh, :mw], w0)
            out_ft[:, :, -mz:, :mh, :mw] = self._compl_mul3d(x_ft[:, :, -mz:, :mh, :mw], w1)
            out_ft[:, :, :mz, -mh:, :mw] = self._compl_mul3d(x_ft[:, :, :mz, -mh:, :mw], w2)
            out_ft[:, :, -mz:, -mh:, :mw] = self._compl_mul3d(x_ft[:, :, -mz:, -mh:, :mw], w3)
        x_out = fft.irfftn(out_ft, s=(x.shape[-3], x.shape[-2], x.shape[-1]), dim=[-3, -2, -1])
        return x_out.to(x.dtype)


class TFNOLayer(nn.Module):
    def __init__(self, width, modes_z, modes_h, modes_w, rank):
        super().__init__()
        self.spectral = TuckerSpectralConv3d(width, width, modes_z, modes_h, modes_w, rank)
        self.local = nn.Conv3d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral(x) + self.local(x))


class TFNO3d(nn.Module):
    """Tensorized FNO: plain FNO with Tucker-factorized spectral weights.

    Input  (B, T_in,  C, Z, H, W)
    Output (B, T_out, C, Z, H, W)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = getattr(config, 'IN_CHANNELS', None) or len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        self.out_channels = getattr(config, 'OUT_CHANNELS', None) or self.in_channels
        width = getattr(config, 'FNO_WIDTH', getattr(config, 'UFNO_WIDTH', 32))
        n_layers = getattr(config, 'FNO_LAYERS', 4)
        self.grad_ckpt = bool(getattr(config, 'GRAD_CKPT', False))
        mz = getattr(config, 'FNO_MODES_Z', 16)
        mh = getattr(config, 'FNO_MODES_H', 24)
        mw = getattr(config, 'FNO_MODES_W', 24)
        rank = getattr(config, 'TFNO_RANK', 8)

        lifting_in = self.t_in * self.in_channels
        self.lifting = nn.Sequential(
            nn.Conv3d(lifting_in, width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=1),
        )
        self.layers = nn.ModuleList([TFNOLayer(width, mz, mh, mw, rank) for _ in range(n_layers)])
        projection_out = self.t_out * self.out_channels
        self.projection = nn.Sequential(
            nn.Conv3d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width * 2, projection_out, kernel_size=1),
        )
        self._print_info(width, n_layers, rank)

    def _print_info(self, width, n_layers, rank):
        total = sum(p.numel() for p in self.parameters())
        print(f"\n[TFNO3d] width={width} layers={n_layers} tucker_rank={rank} "
              f"in=({self.t_in},{self.in_channels}) out=({self.t_out},{self.out_channels}) params={total:,}")

    def forward(self, input_dense, global_timesteps=None):
        B, T_in, C, Z, H, W = input_dense.shape
        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')
        x = self.lifting(x)
        for layer in self.layers:
            if self.grad_ckpt and self.training:
                # native-256^3: measured 19.2GB forward at width 40; storing activations for
                # backward exceeds 32GB. Recompute per block; use_reentrant=False for bf16 autocast.
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.projection(x)
        return rearrange(x, 'b (t c) z h w -> b t c z h w', t=self.t_out, c=self.out_channels)
