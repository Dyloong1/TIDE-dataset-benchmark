"""DeepONet3D — Deep Operator Network (Lu et al. 2021), the foundational operator-learning
baseline, adapted to field-to-field 3D turbulence.

Classic DeepONet learns an operator G: input-function -> output-function via
  G(u)(y) = sum_k  branch_k(u) * trunk_k(y)
where the BRANCH net encodes the input field into p coefficients and the TRUNK net encodes
a query coordinate y into p basis values; their inner product gives the output at y.

For a 3D field-to-field task we:
  - branch: a small 3D CNN over the input field -> global latent -> (C_out * p) coefficients
  - trunk : an MLP over the (z,h,w) coordinate grid -> p basis per point
  - output: einsum over the p latent dim -> per-channel field, reshaped to (B,T_out,C,Z,H,W).

★ Native 256^3 support (2026-07-16, protocol v2 turn):
  A dense trunk over ZHW=16.7M points at hidden=128 needs ~25 GB of activations for backward
  (3 MLP layers x fp32 x 128 hidden). To fit native 256^3 in 32 GB, the trunk is chunked over
  the ZHW dimension into pieces of DEEPONET_TRUNK_CHUNK points at a time. Each chunk:
    - computes its trunk output (chunk, p)
    - einsums with the shared branch (B, c_out, p) into (B, c_out, chunk)
    - writes into the output field at the chunk's slice
  During TRAINING each chunk is wrapped in torch.utils.checkpoint.checkpoint(...) so trunk
  activations are recomputed on the backward pass — reduces peak memory to O(chunk) instead
  of O(ZHW). This is EXACT (not sampled): every output point sees the full branch. Chunking
  is a memory-layout choice, not a numerical approximation.

Interface matches the other baselines:
  input_dense (B, T_in, C, Z, H, W) -> output (B, T_out, C, Z, H, W)
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from einops import rearrange


class DeepONet3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        # OUT_CHANNELS != IN_CHANNELS for cross-channel tasks: P4 pressure (3->1),
        # P5 sgs (4->6). train_benchmark.py passes OUT_CHANNELS=out_ch; honor it
        # (fixes the bug where DeepONet ignored OUT_CHANNELS and always emitted
        # t_out*in_channels, crashing P4/P5). Same fix FNO3d/TFNO already have.
        self.out_channels = getattr(config, 'OUT_CHANNELS', None) or self.in_channels
        self.p = getattr(config, 'DEEPONET_P', 128)          # number of basis functions
        width = getattr(config, 'DEEPONET_WIDTH', 64)
        trunk_hidden = getattr(config, 'DEEPONET_TRUNK_HIDDEN', 128)
        # Trunk chunk size — number of ZHW points per einsum step.
        # 262144 = 64^3 points; at p=128 that is a 128 MB activation per layer, so 3 layers +
        # einsum output (~ 4 MB) fits comfortably. Tune down for tighter memory or up for speed.
        self.trunk_chunk = int(getattr(config, 'DEEPONET_TRUNK_CHUNK', 262144))
        # Explicit training-time gradient checkpointing switch. Default True on native 256^3
        # where activation memory would otherwise exceed 32 GB. Turn off for tiny grids.
        self.grad_ckpt = bool(getattr(config, 'DEEPONET_GRAD_CKPT', True))

        c_in = self.t_in * self.in_channels
        self.c_out = self.t_out * self.out_channels

        # BRANCH: 3D CNN encoder -> global avg pool -> (c_out * p) coefficients
        self.branch_cnn = nn.Sequential(
            nn.Conv3d(c_in, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(width, width * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(width * 2, width * 2, 3, stride=2, padding=1), nn.GELU(),
        )
        self.branch_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(width * 2, width * 4), nn.GELU(),
            nn.Linear(width * 4, self.c_out * self.p),
        )

        # TRUNK: MLP over normalized (z,h,w) coords -> p basis per point
        self.trunk = nn.Sequential(
            nn.Linear(3, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, self.p), nn.GELU(),
        )
        self.bias = nn.Parameter(torch.zeros(self.c_out))
        self._coord_cache = {}
        self._print_info(width)

    def _print_info(self, width):
        total = sum(p.numel() for p in self.parameters())
        print(f"\n[DeepONet3D] p={self.p} width={width} "
              f"in=({self.t_in},{self.in_channels}) out=({self.t_out},{self.out_channels}) "
              f"trunk_chunk={self.trunk_chunk} grad_ckpt={self.grad_ckpt} params={total:,}")

    def _coords(self, Z, H, W, device, dtype):
        # Note: the coord cache uses (Z,H,W,device,dtype) as key; do NOT depend on training-mode
        # flags here so switching train/eval reuses the same buffer.
        key = (Z, H, W, device, dtype)
        if key not in self._coord_cache:
            zs = torch.linspace(0, 1, Z, device=device, dtype=dtype)
            hs = torch.linspace(0, 1, H, device=device, dtype=dtype)
            ws = torch.linspace(0, 1, W, device=device, dtype=dtype)
            gz, gh, gw = torch.meshgrid(zs, hs, ws, indexing='ij')
            self._coord_cache[key] = torch.stack([gz, gh, gw], dim=-1).reshape(-1, 3)  # (ZHW, 3)
        return self._coord_cache[key]

    def _trunk_and_einsum(self, coords_chunk, b):
        # Helper for checkpointing: returns (B, c_out, chunk).
        # coords_chunk (chunk, 3); b (B, c_out, p). No side effects (no writes into a shared buffer).
        t_chunk = self.trunk(coords_chunk)                            # (chunk, p)
        return torch.einsum('bcp,np->bcn', b, t_chunk)                # (B, c_out, chunk)

    def forward(self, input_dense, global_timesteps=None):
        B, T_in, C, Z, H, W = input_dense.shape
        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')

        # BRANCH -> (B, c_out, p). Full branch is shared across all chunks (no redundant recompute).
        b = self.branch_cnn(x)
        b = self.branch_head(b).view(B, self.c_out, self.p)

        # TRUNK + EINSUM chunked over ZHW. Each chunk sees the SAME branch (b), so this is exactly
        # equivalent to the dense path — no numerical approximation, just memory layout.
        coords = self._coords(Z, H, W, x.device, x.dtype)               # (ZHW, 3)
        ZHW = coords.shape[0]
        chunk = self.trunk_chunk if self.trunk_chunk > 0 else ZHW
        # Pre-allocate output (B, c_out, ZHW). At 256^3 rollout (c_out=4) this is ~128 MB fp32,
        # ~64 MB fp16 — cheap next to trunk activations, so we hold it once.
        out = torch.empty((B, self.c_out, ZHW), device=x.device, dtype=x.dtype)
        # NOT gated on x.requires_grad: dataloader inputs never require grad (they are data), so
        # that clause made the gate ALWAYS False in real training -> no checkpointing -> 30GB OOM
        # (2026-07-16 q smoke). use_reentrant=False checkpoints fine when only the PARAMETERS
        # need grad; the requires_grad-input constraint belongs to the legacy reentrant path.
        use_ckpt = self.grad_ckpt and self.training
        for start in range(0, ZHW, chunk):
            end = min(start + chunk, ZHW)
            coords_chunk = coords[start:end]                             # (chunk, 3)
            if use_ckpt:
                # checkpoint recomputes the trunk MLP on the backward pass, so the ~3 layers'
                # activations for `chunk` points are released. use_reentrant=False is the newer,
                # recommended path; it does not require the input to have requires_grad=True.
                out_chunk = ckpt.checkpoint(
                    self._trunk_and_einsum, coords_chunk, b,
                    use_reentrant=False,
                )
            else:
                out_chunk = self._trunk_and_einsum(coords_chunk, b)      # (B, c_out, chunk)
            out[:, :, start:end] = out_chunk
        out = out + self.bias[None, :, None]
        out = out.view(B, self.c_out, Z, H, W)
        # reshape by OUT_channels (not input C) — c_out = t_out * out_channels, so P4/P5
        # (out_channels != C) reshape correctly instead of crashing.
        return rearrange(out, 'b (t c) z h w -> b t c z h w', t=self.t_out, c=self.out_channels)
