"""Extended-physics config dataclasses + a loader that augments solver's config
with the new-physics blocks. Kept separate so solver/config.py is never touched.

The YAML for an extended run is the SAME format solver.config.load_config reads,
plus optional top-level blocks `rotation:`, `scalar:`, `stratification:` parsed here.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

from solver.config import ExperimentConfig, dump_config, load_config


@dataclass
class RotationConfig:
    enabled: bool = False
    omega_x: float = 0.0
    omega_y: float = 0.0
    omega_z: float = 0.0       # rotation axis; z-only is the standard rotating-HIT setup

    @property
    def omega_vec(self):
        return (self.omega_x, self.omega_y, self.omega_z)

    @property
    def omega_mag(self) -> float:
        return (self.omega_x**2 + self.omega_y**2 + self.omega_z**2) ** 0.5


@dataclass
class ScalarConfig:
    enabled: bool = False
    kappa: float = 0.0         # scalar diffusivity; Sc = nu/kappa. Set kappa=nu for Sc=1
                               # (Batchelor scale eta_B = eta/sqrt(Sc) = eta -> 256^3 resolvable).
    mean_grad: float = 0.0     # uniform background gradient dTheta/dz (drives stationary variance
                               # via the -w*mean_grad production term); 0 = freely-decaying scalar.
    b_rms: float = 0.1         # initial scalar rms (for the random IC)
    spectrum_power: float = 2.0  # initial scalar low-k spectrum slope
    seed: int = 0


@dataclass
class StratificationConfig:
    enabled: bool = False
    N_sq: float = 1.0          # squared Brunt-Vaisala frequency N^2 (stratification strength)
    kappa: float = 0.0         # buoyancy diffusivity; Pr = nu/kappa. kappa=nu -> Pr=1
    seed: int = 0


def _block(d: dict, key: str, cls):
    raw = d.get(key) or {}
    fields = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in raw.items() if k in fields})


def load_ext_config(path):
    """Return (ExperimentConfig, RotationConfig, ScalarConfig, StratificationConfig).

    The ExperimentConfig is loaded by solver's own loader (unchanged); the three
    extra blocks are parsed here. Missing blocks default to disabled.
    """
    cfg: ExperimentConfig = load_config(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return (cfg,
            _block(raw, "rotation", RotationConfig),
            _block(raw, "scalar", ScalarConfig),
            _block(raw, "stratification", StratificationConfig))


def dump_ext_config(cfg, rot, scal, strat, out_dir):
    """Snapshot the FULL extended config (base + ext blocks) so the snapshot is
    reloadable by load_ext_config.

    The base solver.config.dump_config drops the rotation/scalar/stratification
    blocks (it only knows ExperimentConfig) -> a snapshot written by it reloads
    with all ext blocks disabled, breaking eval_dynamics_ext / run_ext_showcase
    which rebuild the solver FROM the snapshot. This helper writes the base
    snapshot (full base-field fidelity, incl. any cfg.seed override) and merges
    the enabled ext blocks back into config_snapshot.yaml.
    """
    out = Path(out_dir)
    dump_config(cfg, out)   # base fields (reflects cfg.seed override)
    snap = out / "config_snapshot.yaml"
    with open(snap, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    for key, blk in (("rotation", rot), ("scalar", scal), ("stratification", strat)):
        if blk is not None and getattr(blk, "enabled", False):
            d[key] = dataclasses.asdict(blk)
    with open(snap, "w", encoding="utf-8") as fh:
        yaml.safe_dump(d, fh, sort_keys=False, allow_unicode=True)
