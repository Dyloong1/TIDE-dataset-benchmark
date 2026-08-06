"""TurbgenZarrDataset — reads the 256^3 turbgen corpus (zarr) and emits task-specific
(input, output) pairs for the KDD benchmark. ONE Dataset class, parameterized by `task`:

  rollout   : input = T_in consecutive frames, output = next T_out frames (same seed)
  superres  : input = spectrally down-sampled field (×factor), output = full field
  recon     : input = randomly-masked field (keep_frac kept), output = full field
  pressure  : input = (u,v,w), output = p
  sgs       : input = Gaussian-filtered field, output = subgrid stress tau_ij (6 comps)

Reads `$TURBGEN_DATA_DIR/corpus/<CASE>.zarr` (layout from corpus_to_zarr.py:
  u (N,3,N³) f4, p (N,N³) f4, optional theta/b (N,N³), t (N,), k_max_eta (N,), seed (N,)
  attrs: complete, channels, norm_velocity, norm_p, ...).

Split / seed selection comes from a benchmark_slice.json manifest (made by
make_benchmark_slice.py): which (case, seed) go to this split. **Trajectory-atomic**:
a given (case, seed) lives in exactly one split — frames never leak across splits.

Normalization: FrozenNormalizer rebuilt from the zarr attrs (velocity JOINT, pressure
separate), applied per frame.

Returns a dict compatible with the existing harness:
  {'input_dense': (T_in, C, Z, H, W), 'output_dense': (T_out, C, Z, H, W),
   'task': str, 'case': str, 'seed': int, 'global_timesteps': (T_in+T_out,) or task meta}

Spectral ops (down-sample for superres, Gaussian filter + tau for sgs) reuse the turbgen
solver via TURBGEN_REPO on sys.path; they are pure torch.fft so run on CPU or GPU.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# turbgen repo for FrozenNormalizer (+ optional spectral helpers). Env override, else sibling.
_TURBGEN = os.environ.get("TURBGEN_REPO", str(Path.home() / "turbulence"))
if _TURBGEN not in sys.path:
    sys.path.insert(0, _TURBGEN)

VALID_TASKS = ("rollout", "superres", "recon", "pressure", "sgs")


def _load_frozen_normalizer(zpath, mode="minmax"):
    """Load the FrozenNormalizer from the sidecar JSON that corpus_to_zarr.py writes next
    to the zarr (`<CASE>_norm_minmax.json` / `_norm_standardize.json`). This is the
    published, judge-tested artifact (velocity JOINT, pressure separate).

    RAISES if the sidecar is missing. It used to `return None`, and the caller then trained
    on RAW PHYSICAL fields (velocity ~O(1), pressure ~O(10)) with normalize=True still set and
    not one warning anywhere -- a silent 2.7-GPU-day loss. That is the same failure shape as the
    val=0 bug and the skip-reads-as-pass bug: an absence that produces a green run instead of an
    error. It becomes live the moment anyone re-freezes normalizers (the P2 train-only refit
    rewrites every sidecar), so the window is real, not theoretical.

    If you genuinely want raw fields, ask for them: TurbgenZarrDataset(normalize=False).
    """
    from solver.normalizer import FrozenNormalizer  # noqa: E402
    case = Path(zpath).stem  # "<CASE>" from "<CASE>.zarr"
    side = Path(zpath).parent / f"{case}_norm_{mode}.json"
    if not side.exists():
        raise FileNotFoundError(
            f"normalizer sidecar not found: {side}\n"
            f"  normalize=True was requested, so training/eval would otherwise silently run on RAW "
            f"PHYSICAL fields and every number would be wrong with no error.\n"
            f"  Regenerate it with:  python experiments/phase0/freeze_norm_from_zarr.py {case}\n"
            f"  If you actually want raw fields, pass normalize=False explicitly.")
    return FrozenNormalizer.from_json(side)


class _ScalarChannelNorm:
    """Normalizer for the 5th channel (theta for scalar / b for stratified). The frozen
    stats live in the ZARR ATTRS (`norm_theta` / `norm_b`), written by corpus_to_zarr.py as
    its OWN group (separate from velocity/pressure), NOT in the sidecar JSON. This reads
    those attrs and applies the SAME minmax / standardize convention as solver.normalizer's
    FrozenNormalizer, so theta/b are on a comparable scale to u,v,w,p. Kept here (consumer
    side) rather than extending FrozenNormalizer, matching the producer's "5th channel is a
    separate group, solver/ untouched" contract.

    minmax     -> x / absmax          (symmetric [-1,1]; theta/b are ~zero-mean sign-symmetric)
    standardize-> (x - mean) / std    (z-score)
    """

    def __init__(self, stats: dict, mode: str = "minmax"):
        self.mode = mode
        # producer writes {"mean","std","vmin","vmax", and either "minmax_scale" or "absmax"}
        self.mean = float(stats.get("mean", 0.0))
        self.std = float(stats.get("std", 1.0)) or 1.0
        absmax = stats.get("minmax_scale", stats.get("absmax"))
        if absmax is None:  # derive from vmin/vmax if the explicit key is absent
            absmax = max(abs(float(stats.get("vmin", 0.0))), abs(float(stats.get("vmax", 0.0))))
        self.absmax = float(absmax) or 1.0

    @classmethod
    def from_zarr_attrs(cls, store, extra_key, mode="minmax"):
        """Build from `store.attrs['norm_<extra_key>']`. RAISES if the attr is absent.

        Same reason as _load_frozen_normalizer: returning None here shipped the 5th channel
        (theta / b) UN-normalized while u,v,w,p were normalized -- one tensor, two scales, no
        error. Note these stats live in the ZARR ATTRS, not the sidecar, so a refit that only
        rewrites sidecars leaves theta/b on the OLD (leaked) constants; that mismatch is exactly
        what this raise exists to surface.
        """
        stats = store.attrs.get(f"norm_{extra_key}")
        if not stats:
            raise KeyError(
                f"zarr attr 'norm_{extra_key}' is missing, but this case has a {extra_key} channel "
                f"and normalize=True was requested. Channel {extra_key} would be served RAW while "
                f"u,v,w,p are normalized -- one sample, two scales, no error raised.\n"
                f"  Re-freeze it with: python experiments/phase0/freeze_norm_from_zarr.py <CASE>\n"
                f"  (it writes the 5th-channel stats into the zarr attrs, not the sidecar).")
        return cls(stats, mode)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "standardize":
            return (x - self.mean) / self.std
        return x / self.absmax

    def inverse(self, xn: torch.Tensor) -> torch.Tensor:
        if self.mode == "standardize":
            return xn * self.std + self.mean
        return xn * self.absmax


def _spectral_downsample(field, factor):
    """Down-sample a periodic field by `factor` via spectral truncation (anti-aliased).
    field: (C, N, N, N) -> (C, N//f, N//f, N//f)."""
    C, N = field.shape[0], field.shape[-1]
    n = N // factor
    fh = torch.fft.rfftn(field.float(), dim=(-3, -2, -1))
    # keep lowest n modes in each dim (corners of the half-spectrum)
    k = n // 2
    out = torch.zeros(C, n, n, n // 2 + 1, dtype=fh.dtype)
    out[:, :k, :k, :k + 1] = fh[:, :k, :k, :k + 1]
    out[:, -k:, :k, :k + 1] = fh[:, -k:, :k, :k + 1]
    out[:, :k, -k:, :k + 1] = fh[:, :k, -k:, :k + 1]
    out[:, -k:, -k:, :k + 1] = fh[:, -k:, -k:, :k + 1]
    low = torch.fft.irfftn(out, s=(n, n, n), dim=(-3, -2, -1)) * (1.0 / factor ** 3)
    return low.to(field.dtype)


def _gaussian_filter(field, sigma_frac=0.04):
    """Spectral Gaussian filter (LES-style) at full resolution. field (C,N,N,N).

    G(k) = exp(-k^2 Δ^2 / 24), the standard LES Gaussian. `sigma_frac` is the filter width Δ as a
    FRACTION OF THE BOX, so Δ = sigma_frac * 2π (the domain is [0, 2π)) -- the same length unit as
    the integer wavenumbers k below. sigma_frac=0.04 puts the cutoff at k_c = π/Δ ≈ 12, inside the
    inertial range, and keeps ~84% of the energy: a filter that removes the small scales and leaves
    the large ones, which is what "resolved field" means.

    This used to compute `delta = sigma_frac * N` -- GRID POINTS, while k is in integer modes. The
    units did not match, so Δ came out 40x too large and the filter annihilated the whole field, not
    just its small scales. Measured on a corpus frame: G(k=1) = 0.013 (98.7% of even the LARGEST
    scale removed), G(k=2) = 2.6e-08, energy retained 0.0005%, <|u|> 0.146 -> 0.0003. The "filtered
    field" fed to the SGS task was numerically a zero field, its τ target was built from that zero
    field, and the Smagorinsky reference came out 8000x too small with correlation -0.002 -- which
    would have made the P5 headline (neural closure beats Smagorinsky, and reproduces backscatter)
    a strawman: backscatter was 0 because τ was ~0, not because eddy-viscosity cannot backscatter.
    """
    C, N = field.shape[0], field.shape[-1]
    dev = field.device  # build the filter on field's device (f may be a GPU-cached frame)
    k = torch.fft.fftfreq(N, d=1.0 / N, device=dev)
    kz, ky = torch.meshgrid(k, k, indexing="ij")
    kx = torch.fft.rfftfreq(N, d=1.0 / N, device=dev)
    KZ = kz[:, :, None]
    KY = ky[:, :, None]
    KX = kx[None, None, :]
    k2 = KZ ** 2 + KY ** 2 + KX ** 2
    delta = sigma_frac * 2.0 * math.pi      # box-fraction -> same units as the integer k above
    G = torch.exp(-k2 * (delta ** 2) / 24.0)  # Gaussian filter transfer fn
    fh = torch.fft.rfftn(field.float(), dim=(-3, -2, -1)) * G
    return torch.fft.irfftn(fh, s=(N, N, N), dim=(-3, -2, -1)).to(field.dtype)


class TurbgenZarrDataset(Dataset):
    def __init__(self, case, split, slice_manifest, task="rollout",
                 t_in=1, t_out=1, frame_stride=4, per_frame_n=20, normalize=True, norm_mode="minmax",
                 sr_factor=4, recon_keep_frac=0.05, sgs_sigma_frac=0.04,
                 data_root=None, max_samples=None, patch_size=None, cache_frames=0,
                 cache_device="cpu", eval_patch=False, lazy_rollout_truth=False):
        assert task in VALID_TASKS, f"task must be one of {VALID_TASKS}"
        # lazy_rollout_truth: for native-256 rollout EVAL, do not materialize the t_out truth frames.
        # Eval is a 1->1 AR iteration that only ever needs the input frame plus ONE truth frame at a
        # time, so the truth is returned as frame INDICES and loaded one at a time in the streaming
        # loop. Values are byte-identical to the stacked path.
        #
        # Scale note: this was introduced when t_in was 20, where the input window alone was 5.4GB
        # and the full 50-frame sample was 13.4GB -- it was then the fix for a native-256 host OOM.
        # At the finalized t_in=1 the input is ~0.27GB, so the INPUT side of that argument is gone;
        # what remains is the truth side, still worth it (20 truth frames x 0.27GB = 5.4GB that
        # never has to be resident at once).
        self.lazy_rollout_truth = bool(lazy_rollout_truth)
        self.case, self.split, self.task = case, split, task
        self.t_in, self.t_out, self.frame_stride = t_in, t_out, frame_stride
        # per_frame_n: space tasks (superres/recon/pressure/sgs) draw EXACTLY this many frames
        # per seed, equidistant in time (np.linspace), NOT stride-based. This makes the per-seed
        # sample count deterministic and CONFIG-INDEPENDENT — relam90 (147-250 frames/seed) and
        # kf4 (150) both yield per_frame_n frames, so training cost + cross-config comparability
        # do not drift with trajectory length (protocol: design doc §110/§114).
        self.per_frame_n = per_frame_n
        # frame LRU cache: rollout re-decompresses the SAME frames across overlapping sliding
        # windows every epoch (zstd decompress of a 256^3 frame is ~90ms — the real bottleneck,
        # not disk). Caching the decompressed+normalized frame collapses that to one decompress
        # per unique frame. cache_frames=0 disables (default, memory-safe); set it to at least
        # t_in+t_out so a single sample never evicts its own frames. Each cached frame is
        # C*N^3*4 bytes (~0.5GB at 256^3) — size it to fit RAM (esp. with num_workers>0, where
        # every worker holds its own cache).
        self._cache_cap = int(cache_frames or 0)
        # cache device: 'cuda' keeps decompressed fp16 frames RESIDENT ON THE GPU. At 256^3 a
        # frame is 0.13GB fp16; 100 train frames = 13GB, which fits the 5090's 32GB alongside a
        # 128^3-patch training step (~3-5GB). This is the real fix for the I/O bottleneck: decompress
        # each unique frame ONCE into VRAM, then every epoch reads + crops on-GPU with zero disk,
        # zero CPU decompress, zero H2D copies. MUST use num_workers=0 (CUDA tensors can't cross a
        # DataLoader worker fork). 'cpu' caches in host RAM (careful: 30GB box OOMs near full cache).
        self._cache_device = cache_device
        self._frame_cache = {}  # i -> (C,N,N,N) fp16 tensor on cache_device, insertion-ordered LRU
        # patch_size: crop a (patch_size)^3 sub-block per sample DURING TRAINING to fit 256^3
        # in memory (full-field 256^3 training OOMs on 32GB). Applied ONLY when split=='train';
        # val/test see the full native field (resolution-invariant NO models -> clean 256^3
        # eval, no tiling). Periodic wrap-around crop (the box is 2pi-periodic). None = full field.
        # patch normally applies to train only (val/test see full native 256^3). eval_patch=True
        # lets test/val ALSO crop — needed for deeponet, which is resolution-dependent (flattens
        # the field, native-256^3 OOMs 48GB) so it is evaluated on a 128^3 patch (leaderboard marks
        # it the sole non-native-256 model; see design doc). All other models eval at full 256^3.
        self.patch_size = patch_size if (patch_size and (split == "train" or eval_patch)) else None
        # Local raw-frame decode cache (IO layer ONLY — bytes identical to a zarr read).
        # Why: the zarr stores ONE FULL FRAME PER CHUNK, so every 128^3 patch read decompresses
        # the whole 268MB zstd chunk anyway (~0.5s); crop training touches each frame 8 crops x
        # 25 epochs = 200 times -> measured ~30 min/epoch pure decompress (ufno, 2026-07-17).
        # Fix: on first touch, dump the frame's RAW fp32 arrays to a local .npy once, then serve
        # every later touch as a memmap slice (milliseconds). fp32 verbatim -> crop values are
        # bit-identical to the direct zarr path (pinned by test_local_frame_cache_bitident).
        # Gated by a FLAG FILE (not env) so already-running suite subprocesses pick it up on
        # their next run without a restart: the file holds the cache directory path.
        self._raw_cache = None
        _flag = Path(__file__).resolve().parents[2] / "checkpoints" / ".local_frame_cache"
        if self.patch_size and _flag.exists():
            try:
                cdir = Path(_flag.read_text().strip()) / case
                cdir.mkdir(parents=True, exist_ok=True)
                self._raw_cache = cdir
            except OSError:
                self._raw_cache = None   # unreadable flag/dir -> silently passthrough (IO layer)
        self.normalize = normalize
        self.sr_factor, self.recon_keep_frac, self.sgs_sigma_frac = sr_factor, recon_keep_frac, sgs_sigma_frac

        import zarr
        root = Path(data_root or os.environ.get("TURBGEN_DATA_DIR", "")) / "corpus"
        self.zpath = root / f"{case}.zarr"
        if not self.zpath.exists():
            raise FileNotFoundError(f"corpus zarr not found: {self.zpath}")
        self.store = zarr.open(str(self.zpath), mode="r")
        if not bool(self.store.attrs.get("complete", False)):
            raise RuntimeError(f"REFUSE: {self.zpath} attrs['complete'] != True (partial store)")

        self.grid_N = int(self.store["u"].shape[-1])   # native field side (256), for the patch grid
        self.channels = list(self.store.attrs.get("channels", []))
        self.has_extra = any(c in ("theta", "b") for c in self._extra_keys())
        self.norm = _load_frozen_normalizer(self.zpath, norm_mode) if normalize else None
        # 5th-channel (theta/b) normalizers, one per extra key, from the zarr attrs.
        # {} when un-normalized or the config has no 5th channel. inverse used by eval denorm.
        self.extra_norm = ({ek: _ScalarChannelNorm.from_zarr_attrs(self.store, ek, norm_mode)
                            for ek in self._extra_keys()} if normalize else {})

        # which seeds belong to this split (trajectory-atomic), from the slice manifest
        with open(slice_manifest, encoding="utf-8") as f:
            man = json.load(f)
        self.split_seeds = set(self._seeds_for_split(man, case, split))

        seed_arr = np.asarray(self.store["seed"][:])
        t_arr = np.asarray(self.store["t"][:])
        self._t_arr = t_arr  # kept for the rollout dt-uniformity check in _build_index
        # group frame indices by seed, ordered by time within each seed
        self.by_seed = {}
        for i in range(len(seed_arr)):
            s = int(seed_arr[i])
            if s in self.split_seeds:
                self.by_seed.setdefault(s, []).append(i)
        for s in self.by_seed:
            self.by_seed[s].sort(key=lambda i: float(t_arr[i]))

        self.samples = self._build_index()
        if max_samples:
            # Cap by BASE sample (trajectory/frame-window), NOT by the patch-expanded list, so the
            # eval-samples cap means the same # of trajectories for every model. Otherwise deeponet
            # (eval_patch -> 8 patches/field) would get max_samples/8 trajectories x 8 patches while
            # native-256 models (1 patch) get max_samples trajectories — different, incomparable
            # test sets. Keep the first max_samples DISTINCT (seed,frames) groups, all their patches.
            seen, kept, base_ct = set(), [], 0
            for smp in self.samples:
                key = (smp[0], tuple(smp[1]))     # (seed, frames) identifies the base sample
                if key not in seen:
                    if base_ct >= max_samples:
                        break
                    seen.add(key); base_ct += 1
                kept.append(smp)
            self.samples = kept

    def _extra_keys(self):
        return [k for k in ("theta", "b") if k in self.store]

    @staticmethod
    def _seeds_for_split(man, case, split):
        """Pull seed list for (case, split) from a benchmark_slice.json manifest.
        Schema (made by make_benchmark_slice.py): man['per_config'][case][split] = [seeds]."""
        pc = (man.get("per_config") or {}).get(case) or {}
        v = pc.get(split)
        if isinstance(v, dict):
            return v.get("seeds", [])
        return v or []

    def _build_index(self):
        """Enumerate samples (protocol design doc §2.5). rollout -> single-step sliding windows
        of (t_in+t_out) within a seed, step=frame_stride; per-frame tasks (superres/recon/
        pressure/sgs) -> per_frame_n EQUIDISTANT frames per seed (config-independent count).

        When patch training is on (patch_size set, train or eval_patch), each base sample is
        expanded into a DETERMINISTIC grid of (grid_N/patch_size)^3 patches — patch `pid` always
        has the same origin (see _grid_origin). This makes every baseline see byte-identical
        (seed,frame,patch) pairs (comparability contract) and makes len(dataset) match the
        documented patch-pair count. patch_id=None means 'use the full field'."""
        base = []
        if self.task == "rollout":
            need = self.t_in + self.t_out
            for s, frames in self.by_seed.items():
                for start in range(0, len(frames) - need + 1, self.frame_stride):
                    w = frames[start:start + need]
                    if self._window_has_dt_jump(s, w):
                        # A few trajectories were produced by a resumed run and carry a single
                        # clean splice in their exported frame times (e.g. t: ...42.31 -> 46.14
                        # while the median dt is 0.10). A rollout window is fed to the model as
                        # equi-spaced single steps, so a window STRADDLING that splice asks the
                        # model to advance ~37 dt in one step and scores it against a truth frame
                        # it could not predict. Such a window is not a hard case, it is an invalid
                        # one. We drop only the straddling windows (never the seed, never a frame,
                        # never the data): the rule is objective, applied identically to every
                        # config, and configs with uniform dt lose exactly nothing.
                        continue
                    base.append((s, w))
            # dt-uniformity check: rollout windows are consecutive STORED frames fed to the model
            # as if equi-spaced in time. If a config dropped frames (per-frame k_maxη gate), the
            # true dt is non-uniform and "single-step" spans a variable Δt. We do NOT resample
            # (that would change the data caliber) — we WARN once so the caller knows this config's
            # rollout is variable-step. The true per-frame 't' is shipped in global_timesteps.
            if base:
                cov = self._rollout_dt_cov(base)
                if cov is not None and cov > 0.05:
                    import warnings
                    warnings.warn(
                        f"[TurbgenZarrDataset] {self.case}/rollout: frame dt is NON-UNIFORM "
                        f"(CoV={cov:.3f} > 0.05) — windows span variable Δt but are trained as "
                        f"single-step. Use global_timesteps if a time-aware model is needed.",
                        stacklevel=2)
        else:
            # exactly per_frame_n frames, equidistant across the seed (NOT stride-based, so the
            # count is deterministic regardless of trajectory length — comparability across configs)
            for s, frames in self.by_seed.items():
                n = min(self.per_frame_n, len(frames))
                picks = np.linspace(0, len(frames) - 1, n).astype(int)
                for j in picks:
                    base.append((s, [frames[int(j)]]))
        # expand each base sample over the deterministic patch grid (or a single full-field entry)
        n_patch = self._n_patches()
        if n_patch <= 1:
            return [(s, fr, None) for (s, fr) in base]
        return [(s, fr, pid) for (s, fr) in base for pid in range(n_patch)]

    def _n_patches(self):
        """Number of non-overlapping patches tiling a grid_N^3 field at patch_size (1 = no patch)."""
        if not self.patch_size:
            return 1
        g = self.grid_N // self.patch_size          # patches per axis (2 for 128^3 in 256^3)
        return max(1, g) ** 3

    def _grid_origin(self, patch_id, N):
        """Deterministic origin of patch `patch_id` on the regular (N/patch_size)^3 tile grid.
        Same patch_id -> same origin for every model, so patches are identical across baselines."""
        p = self.patch_size
        g = max(1, N // p)
        pz, rem = divmod(patch_id, g * g)
        py, px = divmod(rem, g)
        return [pz * p, py * p, px * p]

    def _patch_origin(self, patch_id, N):
        """Deterministic patch origin: None (patch off / full field) -> None; else grid tile.
        Same (patch_id) -> same origin for every model -> patches identical across baselines."""
        if patch_id is None or not self.patch_size:
            return None
        return self._grid_origin(patch_id, N)

    # A window is invalid if any internal step exceeds JUMP_FACTOR x the config's own median dt.
    # Relative to the config's median (never an absolute time), so it adapts to each config's
    # frame spacing. 3x is far above real jitter (clean configs measure max/median = 1.00) and far
    # below a real splice (measured 27x-87x), so the rule is insensitive to this exact value.
    _DT_JUMP_FACTOR = 3.0

    def _frame_ordinal(self, seed, row):
        """Position of zarr `row` within its seed's time-ordered frames (0-based).

        Root-invariant physical identity: frame k of seed s is the k-th frame of that trajectory
        whether the store holds 3 seeds or 8. Used to key the recon mask -- see _build_sample.
        Falls back to the raw row if the seed is unknown (cannot happen for a built sample).
        """
        cache = getattr(self, "_ord_cache", None)
        if cache is None:
            cache = self._ord_cache = {
                s: {int(r): i for i, r in enumerate(frames)}
                for s, frames in self.by_seed.items()
            }
        return cache.get(seed, {}).get(int(row), int(row))

    def _median_dt(self):
        """Median consecutive-frame dt over each seed independently (cached). None if no t."""
        if getattr(self, "_median_dt_cache", None) is not None:
            return self._median_dt_cache
        t = getattr(self, "_t_arr", None)
        if t is None:
            return None
        d = []
        for _s, frames in self.by_seed.items():
            ts = np.sort(np.asarray([float(t[i]) for i in frames]))
            d.extend(np.diff(ts).tolist())
        self._median_dt_cache = float(np.median(d)) if d else None
        return self._median_dt_cache

    def _window_has_dt_jump(self, seed, frames):
        """True if this rollout window straddles a frame-time splice (see _build_index)."""
        t = getattr(self, "_t_arr", None)
        med = self._median_dt()
        if t is None or not med or med <= 0:
            return False
        ts = [float(t[i]) for i in frames]
        return any(ts[k + 1] - ts[k] > self._DT_JUMP_FACTOR * med for k in range(len(ts) - 1))

    def _rollout_dt_cov(self, idx):
        """Coefficient of variation of consecutive-frame dt across all rollout windows
        (pooled). ~0 => uniform spacing; >0.05 => frames were gated/dropped. None if no t."""
        t = getattr(self, "_t_arr", None)
        if t is None:
            return None
        deltas = []
        for _s, frames in idx:
            ts = [float(t[i]) for i in frames]
            deltas.extend(ts[k + 1] - ts[k] for k in range(len(ts) - 1))
        if len(deltas) < 2:
            return None
        arr = np.asarray(deltas)
        m = float(arr.mean())
        return float(arr.std() / m) if m > 0 else None

    def __len__(self):
        return len(self.samples)

    def _raw_cached(self, key, i):
        """Return a fp32 memmap of frame i's array `key`, populating the local cache on first
        touch. None when the cache is off or disk-guarded. Cached bytes are the zarr's fp32
        values verbatim (np.save of the decoded array) — slicing the memmap is bit-identical
        to slicing the zarr read."""
        if self._raw_cache is None:
            return None
        f = self._raw_cache / f"{key}_{i}.npy"
        if not f.exists():
            import shutil
            if shutil.disk_usage(self._raw_cache).free < 50 * 2**30:
                return None                      # disk guard: never fill the local drive
            a = np.asarray(self.store[key][i], dtype=np.float32)
            tmp = f.with_suffix(".tmp.npy")
            np.save(tmp, a)
            tmp.replace(f)                       # atomic-ish: readers never see partial files
        return np.load(f, mmap_mode="r")

    def _load_frame(self, i, origin=None):
        """Load frame i as a tensor [u,v,w,p(,extra)], normalized if requested.

        origin=None -> full field (C, N, N, N), the eval / full-field path (LRU-cacheable).
        origin=[z0,y0,x0] -> read ONLY the patch_size^3 sub-box directly from the zarr
        (C, p, p, p). At 256^3 a full frame is 201MB but a 128^3 patch is ~17MB, so slicing
        in the zarr read decompresses ~8x fewer chunks — this is the training-side I/O fix
        (avoids the 878GB/epoch full-frame re-reads that OOM any frame cache). Bit-identical to
        read-full-then-crop: normalization is a per-element affine with frozen constants (no
        spatial reduction), so crop(normalize(full)) == normalize(read_patch(full)) element-wise.
        Deterministic grid patches (origins 0/128 for 128-into-256) are tile-aligned and never
        wrap, so a plain slice equals _crop_patch's periodic crop.

        The direct-patch path deliberately BYPASSES the LRU frame cache: the cache stores full
        fp16 frames keyed by i alone, so mixing it with fp32 patch reads would return inconsistent
        shape+precision for the same (i) and break the bit-identical-across-models contract. Patch
        reads decompress little enough that the cache isn't needed for patched training anyway."""
        i = int(i)
        if origin is not None:
            p = self.patch_size
            z0, y0, x0 = origin
            # NOTE the old "~8x less data" claim was wrong for this corpus: chunks are ONE FULL
            # FRAME, so a patch slice decompresses the whole chunk regardless. The raw cache
            # below is the actual fix; without it this path is ~0.5s/patch of pure zstd.
            arr = self._raw_cached("u", i), self._raw_cached("p", i)
            if arr[0] is not None:
                u = torch.from_numpy(np.array(arr[0][:, z0:z0 + p, y0:y0 + p, x0:x0 + p]))
                pr = torch.from_numpy(np.array(arr[1][z0:z0 + p, y0:y0 + p, x0:x0 + p]))
            else:
                u = torch.from_numpy(np.asarray(
                    self.store["u"][i, :, z0:z0 + p, y0:y0 + p, x0:x0 + p])).float()   # (3,p,p,p)
                pr = torch.from_numpy(np.asarray(
                    self.store["p"][i, z0:z0 + p, y0:y0 + p, x0:x0 + p])).float()      # (p,p,p)
            if self.norm is not None:
                u, pr = self.norm.apply_frame(u, pr)   # per-element affine — same result on a patch
            chans = [u, pr.unsqueeze(0)]
            for ek in self._extra_keys():
                cached = self._raw_cached(ek, i)
                e = (torch.from_numpy(np.array(cached[z0:z0 + p, y0:y0 + p, x0:x0 + p]))
                     if cached is not None else
                     torch.from_numpy(np.asarray(
                         self.store[ek][i, z0:z0 + p, y0:y0 + p, x0:x0 + p])).float())
                # extra_norm is {} iff normalize=False (both are gated on `normalize` at
                # construction, and from_zarr_attrs now RAISES rather than returning None),
                # so a present key means a real normalizer. Do NOT re-gate on self.norm:
                # that tied theta/b to the VELOCITY sidecar and silently served the 5th
                # channel raw whenever the two disagreed.
                sn = self.extra_norm.get(ek)
                if sn is not None:
                    e = sn.apply(e)
                chans.append(e.unsqueeze(0))
            return torch.cat(chans, dim=0)   # (C,p,p,p) fp32, no cache
        if self._cache_cap and i in self._frame_cache:
            f = self._frame_cache.pop(i); self._frame_cache[i] = f  # mark most-recently-used
            return f.float()   # cache stores fp16 (on cache_device); upcast for downstream ops
        # native full-frame reads go through the raw disk cache too (2026-07-17): v2 native
        # training/eval re-reads full frames every epoch, and each zarr read is a whole-chunk
        # zstd decompress (~0.5s CPU-bound). The cached .npy holds the zarr's fp32 decode
        # verbatim, so this is bit-identical to the direct read (pinned by
        # test_local_frame_cache_bitident); measured root cause of the t-machine 700s/ep
        # rollout epochs (crop tasks hit the cache, native tasks bypassed it).
        ru, rp = self._raw_cached("u", i), self._raw_cached("p", i)
        u = (torch.from_numpy(np.array(ru)) if ru is not None
             else torch.from_numpy(np.asarray(self.store["u"][i])).float())   # (3,N,N,N)
        p = (torch.from_numpy(np.array(rp)) if rp is not None
             else torch.from_numpy(np.asarray(self.store["p"][i])).float())   # (N,N,N)
        if self.norm is not None:
            u, p = self.norm.apply_frame(u, p)
        chans = [u, p.unsqueeze(0)]
        for ek in self._extra_keys():
            re_ = self._raw_cached(ek, i)
            e = (torch.from_numpy(np.array(re_)) if re_ is not None
                 else torch.from_numpy(np.asarray(self.store[ek][i])).float())
            sn = self.extra_norm.get(ek) if self.norm is not None else None
            if sn is not None:                 # normalize theta/b like u,v,w,p (attrs stats)
                e = sn.apply(e)
            chans.append(e.unsqueeze(0))
        f = torch.cat(chans, dim=0)  # (C,N,N,N) on CPU
        if self._cache_cap:
            # store fp16 on cache_device (0.13GB/frame at 256^3). Values are normalized to O(1),
            # so fp16 round-trip is well within bf16-AMP precision; crop/model run fp32/bf16 after
            # the .float() upcast on read. On 'cuda' the frame lives in VRAM (zero re-decompress,
            # zero H2D thereafter). Return VIA the cache so every frame — hit or miss — comes back
            # on the SAME device (else torch.stack mixes cpu+cuda frames and errors).
            self._frame_cache[i] = f.half().to(self._cache_device)
            while len(self._frame_cache) > self._cache_cap:
                self._frame_cache.pop(next(iter(self._frame_cache)))  # evict least-recently-used
            return self._frame_cache[i].float()
        return f

    def _stack_frames(self, frames, origin):
        """Stack a list of frame indices into (len,C,...), filling a PRE-ALLOCATED buffer frame by
        frame (not torch.stack([...]), which holds the whole list AND its stacked copy at once — a
        ~2x transient that OOMs at native 256^3). Peak = final size + one frame."""
        f0 = self._load_frame(int(frames[0]), origin)
        out = torch.empty((len(frames),) + f0.shape, dtype=f0.dtype)
        out[0] = f0
        del f0
        for j in range(1, len(frames)):
            out[j] = self._load_frame(int(frames[j]), origin)
        return out

    def load_truth_frame(self, frame_idx, origin=None):
        """Load one truth frame (normalized, same path as training) for lazy eval rollout — the AR
        loop calls this per step so the t_out truth trajectory never sits in memory all at once.
        origin=None or [-1,-1,-1] -> full field. Returns (C,...)."""
        if origin is not None and (isinstance(origin, (list, tuple)) and origin[0] < 0):
            origin = None
        return self._load_frame(int(frame_idx), origin)

    def _crop_patch(self, field, origin):
        """Periodic wrap-around crop of a (...,N,N,N) tensor to (...,p,p,p) at `origin`.
        The box is 2pi-periodic, so the window wraps — no edge padding, every crop is a valid
        sub-field. field: (C,N,N,N) or (T,C,N,N,N); crops the last 3 dims."""
        if origin is None or self.patch_size is None:
            return field
        p, N = self.patch_size, field.shape[-1]
        idx = [torch.arange(o, o + p) % N for o in origin]           # wrap-around indices
        return field[..., idx[0], :, :][..., :, idx[1], :][..., :, :, idx[2]]

    def __getitem__(self, k):
        s, frames, patch_id = self.samples[k]
        # origin encodes patch_size-set AND patch_id-not-None; when set, _load_frame reads ONLY the
        # patch sub-box from the zarr (bit-identical to read-full-then-crop, ~8x less I/O) and we do
        # NOT call _crop_patch again (that would double-crop). When None (eval / full field),
        # _load_frame returns the full frame and _crop_patch is a no-op.
        origin = self._patch_origin(patch_id, self.grid_N)
        if self.task == "rollout":
            in_frames = frames[:self.t_in]
            out_frames = frames[self.t_in:self.t_in + self.t_out]
            # eval 1->1 AR path: return ONLY the t_in input window as a tensor; the t_out truth
            # frames go back as INDICES so the AR loop loads them one at a time (see
            # load_truth_frame). This keeps the resident sample at ~t_in frames (5.4GB) instead of
            # t_in+t_out (13.4GB) at native 256^3 — the host-OOM fix. Values byte-identical.
            inp = self._stack_frames(in_frames, origin)          # (t_in,C,p,p,p)
            if self.lazy_rollout_truth:
                return {"input_dense": inp,
                        "truth_frames": torch.tensor(list(out_frames)),  # indices, loaded lazily
                        "truth_origin": torch.tensor(origin if origin is not None else [-1, -1, -1]),
                        "lazy_truth": True, "task": self.task, "case": self.case, "seed": s,
                        "global_timesteps": torch.tensor([float(self.store["t"][i]) for i in frames])}
            out = self._stack_frames(out_frames, origin)         # (t_out,C,p,p,p)
            return {"input_dense": inp, "output_dense": out, "task": self.task,
                    "case": self.case, "seed": s,
                    "global_timesteps": torch.tensor([float(self.store["t"][i]) for i in frames])}

        f = self._load_frame(frames[0], origin)  # (C,p,p,p) patch, or (C,N,N,N) full when origin None
        C = f.shape[0]
        if self.task == "pressure":
            inp = f[:3].unsqueeze(0)                       # (1,3,N,N,N) velocity
            out = f[3:4].unsqueeze(0)                      # (1,1,N,N,N) pressure
        elif self.task == "superres":
            # SR setup: down-sample (spectral, anti-aliased) then trilinear-UPSAMPLE back to
            # full grid, so the model input is a coarsened full-res field and it must restore
            # the lost high-frequency content (standard operator-SR framing; keeps input/output
            # the same grid so any field-to-field model applies).
            low = _spectral_downsample(f, self.sr_factor)            # (C,n,n,n)
            up = torch.nn.functional.interpolate(
                low.unsqueeze(0), size=f.shape[-3:], mode="trilinear", align_corners=False)[0]
            inp = up.unsqueeze(0)                                    # (1,C,N,N,N) coarsened
            out = f.unsqueeze(0)                                     # (1,C,N,N,N) full
        elif self.task == "recon":
            # DETERMINISTIC mask (comparability contract): derive the RNG from (seed, frame, patch)
            # so every model + every epoch sees the SAME observed-pixel set at index k. A global
            # unseeded torch.rand would give a different mask each __getitem__ call -> models A and B
            # would be scored on different observations at the same index (leaderboard not comparable).
            # generator lives on CPU (CUDA generators can't cross a DataLoader worker fork); mask is
            # built on CPU then moved to f.device (f may be a GPU-cached frame).
            #
            # The key MUST be root-invariant. It used to use frames[0], a ZARR ROW INDEX, which is a
            # function of which seeds a given store happens to hold: for seed 4 of kf4 the same
            # physical frame is row 600 in the full corpus but row 150 in the train-only NVMe stage.
            # That produced statistically INDEPENDENT masks for the same logical sample across roots
            # (measured overlap 0.0496 ~ chance) while the truth data was bit-identical -- silently
            # breaking the contract above, and now reachable in one run because train reads the NVMe
            # stage while val reads the full corpus. Key on the frame's ORDINAL WITHIN ITS SEED,
            # which is physical identity and independent of storage layout.
            _ord = self._frame_ordinal(s, frames[0])
            g = torch.Generator().manual_seed((int(s) * 1_000_003 + int(_ord) * 1009
                                               + int(patch_id or 0)) & 0x7FFFFFFF)
            mask = (torch.rand(1, 1, *f.shape[1:], generator=g) < self.recon_keep_frac).float().to(f.device)
            inp = (f.unsqueeze(0) * mask)                  # masked field (most zeros)
            out = f.unsqueeze(0)
        elif self.task == "sgs":
            fbar = _gaussian_filter(f, self.sgs_sigma_frac)
            tau = self._subgrid_stress(f[:3], self.sgs_sigma_frac)  # (6,N,N,N)
            inp = fbar.unsqueeze(0)
            out = tau.unsqueeze(0)
        else:
            raise ValueError(self.task)
        return {"input_dense": inp, "output_dense": out, "task": self.task,
                "case": self.case, "seed": s,
                "global_timesteps": torch.tensor([float(self.store["t"][frames[0]])])}

    @staticmethod
    def _subgrid_stress(u, sigma_frac):
        """SGS stress tau_ij = filter(u_i u_j) - filter(u_i) filter(u_j), 6 unique comps.
        u: (3,N,N,N). Returns (6,N,N,N) ordered [11,22,33,12,13,23]."""
        ub = _gaussian_filter(u, sigma_frac)
        comps = []
        pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        for (i, j) in pairs:
            uiuj = _gaussian_filter((u[i] * u[j]).unsqueeze(0), sigma_frac)[0]
            comps.append(uiuj - ub[i] * ub[j])
        return torch.stack(comps, dim=0)
