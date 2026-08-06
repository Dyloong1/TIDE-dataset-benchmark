"""Dataclass configuration + yaml loading / snapshot dumping.

Every experiment is driven by a yaml file (see experiments/phase0/configs/).
The loaded config is re-dumped verbatim into the results directory so that
any result can be reproduced from its snapshot alone.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ForcingConfig:
    type: str = "none"          # none | negative_damping | energy_preserving | fixed_power | fixed_power_damped | stochastic_ou
    k_f: float = 2.0            # forcing band upper edge: |k| < k_f (<= k_f for energy_preserving)
    k_f_low: float = 0.0        # forcing band LOWER edge: k_f_low <= |k| (annular band, fixed_power).
                                # >0 excludes the lowest shells from injection entirely, so the
                                # largest box scales are never pumped (root-cause fix for the k=1
                                # runaway, vs a post-hoc cap). 0 = inject from k>0 as before.
    eps_w: float = 0.1          # fixed power injection rate (negative_damping / fixed_power)
    E_f0: float = 0.4           # fixed band energy (energy_preserving)
    # fixed_power_damped: large-scale soft cap on the lowest shells to stop the
    # k=1 runaway accumulation that destabilizes deterministic band forcing
    # (k=1 has no larger scale to cascade to; fixed-power gain compounds it).
    # Damping engages ONLY above a per-shell energy ceiling, so normal stationary
    # state is untouched and steady eps == eps_w (k_max*eta lock) is preserved.
    k_damp: float = 2.0         # apply the soft cap to shells with |k| < k_damp
    e_cap_factor: float = 1.5   # ceiling = e_cap_factor * initial per-shell energy of those shells
    # Eswaran-Pope stochastic (Ornstein-Uhlenbeck) forcing (stochastic_ou):
    ou_tau: float = 1.0         # OU correlation time T_F (~ one large-eddy turnover)
    ou_sigma2: float = 0.05     # OU variance per forced mode (sets the input power)
    ou_seed: int = 1234         # RNG seed for the forcing process (separate from IC seed)
    helicity_sign: int = 1      # helical_ou: +1 / -1 helicity sign to inject (breaks mirror symmetry)


@dataclass
class SolverConfig:
    N: int = 256                # grid points per dimension, domain (2pi)^3
    nu: float = 2.7e-3          # kinematic viscosity
    dtype: str = "fp32"         # fp32 | fp64 (single code path)
    device: str = "cuda"
    scheme: str = "rk3"         # rk3 (Williamson low-storage) | rk4
    cfl: float = 0.4
    dt_max: float = 5e-3        # cap so a weak initial field cannot blow dt up
    dealias: bool = True
    dealias_shape: str = "cubic"   # cubic | spherical (see grids.SpectralGrid)
    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    # Large-scale linear damping (Rayleigh friction) -alpha*u for |k| < ls_damp_kcut.
    # A standard physical device (linear friction / hypofriction) to drain the
    # largest scales, tested as the last literature-backed attempt to stabilize
    # deterministic band forcing (vs. modifying the forcing band itself).
    # alpha=0 -> off (default; does not affect any existing run).
    ls_damp_alpha: float = 0.0     # damping rate (1/time); ~0.1-0.5 typical
    ls_damp_kcut: float = 2.0      # damp shells with |k| < ls_damp_kcut


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    solver: SolverConfig = field(default_factory=SolverConfig)
    # initial condition
    ic: str = "taylor_green"    # taylor_green | random_solenoidal | abc_flow | vortex_tubes
    seed: int = 0
    k_p: float = 3.0            # peak wavenumber of E(k) ~ k^p exp(-2(k/k_p)^2)
    u_rms: float = 0.7          # target rms velocity per component for random IC
    ic_spectrum_power: float = 4.0  # low-k slope p of the random IC (4=Batchelor,
                                    # 2=Saffman); only affects decaying runs
    # abc_flow params (Beltrami amplitudes)
    abc_A: float = 1.0
    abc_B: float = 1.0
    abc_C: float = 1.0
    abc_perturb: float = 0.0    # broadband seed amplitude (frac of u_rms) to
                                # trigger ABC instability; 0 = pure (laminar) ABC
    # vortex_tubes params (antiparallel Lamb-Oseen reconnection IC)
    vt_sep: float = 1.5708      # tube separation in y (~pi/2)
    vt_core: float = 0.4        # Gaussian core radius
    vt_circ: float = 1.0        # circulation
    vt_perturb: float = 0.02    # z-wiggle amplitude seeding reconnection
    # run control
    t_end: float = 20.0
    log_every_steps: int = 1    # scalar diagnostics cadence
    spectrum_every_steps: int = 50
    snapshot_times: list = field(default_factory=list)  # physical times to save fields
    out_dir: str = ""           # results directory (set by runner if empty)


def _from_dict(cls, d: dict):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if dataclasses.is_dataclass(f.type) or f.name in ("solver", "forcing"):
            sub_cls = {"solver": SolverConfig, "forcing": ForcingConfig}[f.name]
            v = _from_dict(sub_cls, v)
        kwargs[f.name] = v
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _from_dict(ExperimentConfig, raw)


def dump_config(cfg: ExperimentConfig, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = dataclasses.asdict(cfg)
    with open(out / "config_snapshot.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(d, fh, sort_keys=False, allow_unicode=True)
    with open(out / "config_snapshot.json", "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)
