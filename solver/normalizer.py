"""Frozen normalizer for the published corpus (zarr/hdf5 conversion stage).

The foundation-model corpus stores 4-channel physical-space frames (u, v, w, p)
as saved by io_utils.save_snapshot. Before training, channels are normalized to a
fixed range; for the dataset to be reproducible and the eval judges to run in
PHYSICAL units, the normalization constants must be FROZEN ONCE over the whole
corpus and published alongside it. This module computes those constants by a
streaming pass over frames (the corpus does not fit in memory), freezes them to a
JSON sidecar, and provides exact inverse so any normalized field maps back to
physical units before it hits eval_dns_standard / eval_dynamics_residuals.

Design choices (physics-motivated, both explicit):
  * The three velocity components (u, v, w) are normalized JOINTLY by a single
    statistic, NOT per-component. The dataset certifies component isotropy (A10):
    <u^2> = <v^2> = <w^2>. Normalizing each component by its own min/max/std would
    manufacture anisotropy in the normalized field. So velocity uses one shared
    scale; pressure is a distinct physical quantity and gets its own.
  * Two modes: 'minmax' -> [-1, 1] (symmetric, since HIT velocity is zero-mean and
    sign-symmetric), 'standardize' -> zero-mean unit-variance (z-score).

The frozen JSON is the publishable artifact. Stats are accumulated in fp64
regardless of frame dtype (sums over 256^3 * thousands of frames need the range).

Judge-first: covered by tests/test_normalizer.py (round-trip exactness, joint-vs-
per-channel, streaming==batch, freeze/load idempotence). Do not use on corpus
frames until those pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch

# channel groups in the (u, v, w, p) corpus layout
_VEL = "velocity"      # joint over u, v, w
_PRES = "pressure"     # p alone
_GROUPS = (_VEL, _PRES)

_EPS = 1e-30           # guard against divide-by-zero on a degenerate (constant) channel


@dataclass
class _GroupStats:
    """Streaming accumulators for one channel group (fp64).

    Variance via Welford's PARALLEL (chunked) algorithm: each frame is a chunk,
    merged into the running (mean, M2) by the numerically stable Chan et al. merge.
    The naive sum-of-squares form (sumsq/n - mean^2) suffers catastrophic
    cancellation when mean^2 >> variance (e.g. pressure with a large nonzero mean,
    small fluctuation) and silently returns std=0 -> divide-by-eps -> garbage.
    Welford is exact to fp64 regardless of mean magnitude.
    """
    n: int = 0                      # number of scalar samples seen
    vmin: float = float("inf")
    vmax: float = float("-inf")
    mean: float = 0.0              # running mean
    M2: float = 0.0               # running sum of squared deviations from the mean

    def update(self, x: torch.Tensor) -> None:
        xf = x.to(torch.float64)
        nb = xf.numel()
        if nb == 0:
            return
        self.vmin = min(self.vmin, float(xf.min()))
        self.vmax = max(self.vmax, float(xf.max()))
        # chunk stats (fp64): mean and M2 = sum((x-mean_b)^2)
        mean_b = float(xf.mean())
        M2_b = float(((xf - mean_b) ** 2).sum())
        # Chan et al. parallel merge of (n, mean, M2) with (nb, mean_b, M2_b)
        if self.n == 0:
            self.n, self.mean, self.M2 = nb, mean_b, M2_b
            return
        delta = mean_b - self.mean
        tot = self.n + nb
        self.mean += delta * nb / tot
        self.M2 += M2_b + delta * delta * self.n * nb / tot
        self.n = tot

    @property
    def std(self) -> float:
        if self.n == 0:
            return 0.0
        # population variance (M2/n, not M2/(n-1)): the corpus is the whole
        # population for normalization, and with ~3000*256^3 samples the n vs n-1
        # difference is negligible anyway. Tests compare against std(unbiased=False).
        var = self.M2 / self.n
        return float(var) ** 0.5 if var > 0 else 0.0

    @property
    def absmax(self) -> float:
        return max(abs(self.vmin), abs(self.vmax))


@dataclass
class FrozenNormalizer:
    """Immutable, publishable normalization. Apply / inverse / to-from JSON.

    mode='minmax':      x_n = x / absmax                  -> [-1, 1] for IN-SAMPLE
                        x   = x_n * absmax                    data only (see note)
    mode='standardize': x_n = (x - mean) / std
                        x   = x_n * std + mean

    NOTE on minmax bounds: x_n lands in [-1, 1] only for fields whose |value| does
    not exceed the absmax seen during fit. The corpus is fit and applied on the
    SAME frames, so in-distribution that holds; a held-out frame with a larger
    extremum than any training frame will exceed [-1, 1] (this is inherent to a
    frozen min/max normalizer, not a bug). Downstream code must NOT assume a hard
    [-1,1] clamp on unseen data.

    We use symmetric absmax (not (x-min)/(max-min)) for velocity so that the
    physical zero stays at normalized zero and the sign symmetry of HIT velocity
    is preserved; the same form is applied to pressure for consistency.
    """
    mode: str
    stats: dict          # group -> {"absmax", "mean", "std", "vmin", "vmax", "n"}
    # provenance for the published artifact
    n_frames: int = 0
    note: str = ""

    # ---- apply / inverse -------------------------------------------------
    def _scale_shift(self, group: str) -> tuple[float, float]:
        s = self.stats[group]
        if self.mode == "minmax":
            return max(s["absmax"], _EPS), 0.0          # divide by absmax, no shift
        elif self.mode == "standardize":
            return max(s["std"], _EPS), s["mean"]       # divide by std, shift by mean
        raise ValueError(f"unknown mode {self.mode!r}")

    def apply(self, x: torch.Tensor, group: str) -> torch.Tensor:
        scale, shift = self._scale_shift(group)
        return (x - shift) / scale

    def inverse(self, x_n: torch.Tensor, group: str) -> torch.Tensor:
        scale, shift = self._scale_shift(group)
        return x_n * scale + shift

    def apply_frame(self, u: torch.Tensor, p: torch.Tensor | None = None):
        """Normalize a corpus frame. u: [3,...] velocity, p: [...] pressure (opt)."""
        un = self.apply(u, _VEL)
        if p is None:
            return un
        return un, self.apply(p, _PRES)

    def inverse_frame(self, un: torch.Tensor, pn: torch.Tensor | None = None):
        u = self.inverse(un, _VEL)
        if pn is None:
            return u
        return u, self.inverse(pn, _PRES)

    # ---- persistence (the published artifact) ---------------------------
    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "FrozenNormalizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**d)


class NormalizerFitter:
    """Streaming accumulator: feed frames one at a time, then .freeze().

    Usage (zarr conversion stage):
        fit = NormalizerFitter()
        for frame in corpus_frames:            # each a dict {"u":[3,N,N,N],"p":[N,N,N]}
            fit.update(frame["u"], frame.get("p"))
        norm = fit.freeze(mode="minmax")       # FrozenNormalizer
        norm.to_json("corpus_normalization.json")
    """

    def __init__(self) -> None:
        self._g = {_VEL: _GroupStats(), _PRES: _GroupStats()}
        self._n_frames = 0
        self._saw_pressure = False

    def update(self, u: torch.Tensor, p: torch.Tensor | None = None) -> None:
        if u.shape[0] != 3:
            raise ValueError(f"velocity must be [3,...], got {tuple(u.shape)}")
        # Reject NaN/Inf BEFORE accumulating: a single bad frame in the 3000+ frame
        # corpus pass would silently set absmax=inf (-> all fields normalize to 0)
        # or mean/std=NaN (-> all fields NaN), corrupting the published constants
        # with no error. Fail loud at the offending frame instead.
        if not torch.isfinite(u).all():
            raise ValueError("velocity frame contains NaN/Inf — refusing to "
                             "accumulate (would corrupt frozen normalizer)")
        self._g[_VEL].update(u)                 # all three components into ONE group
        if p is not None:
            if not torch.isfinite(p).all():
                raise ValueError("pressure frame contains NaN/Inf — refusing to "
                                 "accumulate (would corrupt frozen normalizer)")
            self._g[_PRES].update(p)
            self._saw_pressure = True
        self._n_frames += 1

    def freeze(self, mode: str = "minmax", note: str = "") -> FrozenNormalizer:
        if mode not in ("minmax", "standardize"):
            raise ValueError(f"unknown mode {mode!r}")
        if self._n_frames == 0:
            raise RuntimeError("no frames seen; nothing to freeze")
        stats = {}
        for grp in _GROUPS:
            g = self._g[grp]
            if g.n == 0:                        # pressure may be absent in a 3-channel corpus
                continue
            stats[grp] = {
                "absmax": g.absmax, "mean": g.mean, "std": g.std,
                "vmin": g.vmin, "vmax": g.vmax, "n": g.n,
            }
        return FrozenNormalizer(mode=mode, stats=stats,
                                n_frames=self._n_frames, note=note)
