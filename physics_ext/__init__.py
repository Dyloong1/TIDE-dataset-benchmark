"""physics_ext — isolated extended-physics package for the turbgen DNS solver.

ISOLATION CONTRACT (project hard rule): this package EXTENDS the validated core
`solver/` by import/subclassing ONLY. It NEVER modifies any file under `solver/`
or `experiments/phase0/`, so the 10-config isotropic-NS corpus pipeline stays
bit-identical (regression guarantee: `git diff solver/` must be empty, and the
86 existing tests stay green).

Three physics axes, all still incompressible Navier-Stokes ("water"), MHD
explicitly excluded (would break the pure-NS dataset positioning):
  - rotating.py   : RotatingSolver  — NS + Coriolis -2 Omega x u (pure NS)
  - scalar.py     : ScalarTransport — passive scalar advected by an unchanged NS flow
  - boussinesq.py : BoussinesqSolver — active scalar (buoyancy) + back-reaction

Run before importing torch elsewhere (OpenMP guard), same as `solver`.
"""
import solver._env  # noqa: F401  (must run before torch is imported elsewhere)

from .rotating import RotatingSolver  # noqa: E402,F401

__all__ = ["RotatingSolver"]
