"""
Round 1 PCFM ablation experiment configurations.

Each experiment is defined by:
- sampler_name: key in SAMPLER_REGISTRY
- sampler_kwargs: dict of sampler constructor arguments
- constraint_config: ConstraintConfig specifying active constraints
- description: human-readable description

All experiments use the same pre-trained FM-Hybrid model.
Only inference-time strategy changes.
"""

from .constraints import ConstraintConfig
from .samplers import SAMPLER_REGISTRY


def _exp(sampler_name, constraint_config, description, **sampler_kwargs):
    """Helper to create experiment definition."""
    return {
        'sampler_name': sampler_name,
        'sampler_kwargs': sampler_kwargs,
        'constraint_config': constraint_config,
        'description': description,
    }


# Constraint presets
NO_CONSTRAINTS = ConstraintConfig()
IC_ONLY = ConstraintConfig(use_ic=True)
DIV_ONLY = ConstraintConfig(use_divergence_free=True)
IC_DIV = ConstraintConfig(use_ic=True, use_divergence_free=True)
IC_DIV_MOM = ConstraintConfig(use_ic=True, use_divergence_free=True, use_momentum=True)
IC_DIV_ENERGY = ConstraintConfig(use_ic=True, use_divergence_free=True, use_energy=True)
FULL_CONSTRAINTS = ConstraintConfig(use_ic=True, use_divergence_free=True,
                                     use_momentum=True, use_energy=True)


# ============================================================================
# Round 1 Experiments
# ============================================================================

ROUND1_EXPERIMENTS = {}

# ---------------------------------------------------------------------------
# Priority 1: Baselines (B0, B1, B4)
# ---------------------------------------------------------------------------

ROUND1_EXPERIMENTS['B0_vanilla'] = _exp(
    'vanilla', NO_CONSTRAINTS,
    'Vanilla FM: no constraints, pure flow matching sampling',
)

ROUND1_EXPERIMENTS['B1_pcfm_default'] = _exp(
    'pcfm', FULL_CONSTRAINTS,
    'PCFM default: shooting + projection + OT inverse (full correction)',
    relaxation_lambda=1.0,
)

ROUND1_EXPERIMENTS['B4_terminal_only'] = _exp(
    'terminal_projection', FULL_CONSTRAINTS,
    'Terminal-only projection: sample freely, project once at end',
)

# ---------------------------------------------------------------------------
# Priority 2: Correction strategies (C1, C6, C8, C11)
# C1 = B1 (PCFM every step), so we skip duplicate
# ---------------------------------------------------------------------------

ROUND1_EXPERIMENTS['C6_hard_projection'] = _exp(
    'hard_projection', FULL_CONSTRAINTS,
    'Hard projection: Euler step -> direct project at each substep (PDM style)',
)

ROUND1_EXPERIMENTS['C8_pcfm_no_relax'] = {
    'alias_of': 'B1_pcfm_default',
    'description': 'C8 = B1 (no separate relaxation step in current implementation)',
}

ROUND1_EXPERIMENTS['C11_ccfm_default'] = _exp(
    'ccfm', FULL_CONSTRAINTS,
    'CCFM: time-adaptive constraint tightening, schedule n=1.0',
    schedule_n=1.0,
)

# ---------------------------------------------------------------------------
# Priority 3: Constraint stacking (D2a-e)
# All use terminal projection to isolate constraint contribution
# ---------------------------------------------------------------------------

ROUND1_EXPERIMENTS['D2a_ic_only'] = _exp(
    'terminal_projection', IC_ONLY,
    'D2a: Terminal projection with IC constraint only',
)

ROUND1_EXPERIMENTS['D2b_ic_div'] = _exp(
    'terminal_projection', IC_DIV,
    'D2b: Terminal projection with IC + divergence-free',
)

ROUND1_EXPERIMENTS['D2c_ic_div_momentum'] = _exp(
    'terminal_projection', IC_DIV_MOM,
    'D2c: Terminal projection with IC + div-free + momentum conservation',
)

ROUND1_EXPERIMENTS['D2d_ic_div_energy'] = _exp(
    'terminal_projection', IC_DIV_ENERGY,
    'D2d: Terminal projection with IC + div-free + energy conservation',
)

ROUND1_EXPERIMENTS['D2e_full_constraints'] = _exp(
    'terminal_projection', FULL_CONSTRAINTS,
    'D2e: Terminal projection with all constraints (= B4)',
)

# ---------------------------------------------------------------------------
# Priority 4: Leray usage (D8a-f)
# ---------------------------------------------------------------------------

ROUND1_EXPERIMENTS['D8a_no_leray'] = _exp(
    'vanilla', NO_CONSTRAINTS,
    'D8a: No Leray, no constraints (= B0)',
)

ROUND1_EXPERIMENTS['D8b_leray_terminal'] = _exp(
    'terminal_projection', DIV_ONLY,
    'D8b: Leray projection at terminal only (div-free)',
)

ROUND1_EXPERIMENTS['D8c_leray_every_step'] = _exp(
    'leray_every_step', DIV_ONLY,
    'D8c: Leray projection after every ODE substep',
)

ROUND1_EXPERIMENTS['D8d_leray_ot'] = _exp(
    'leray_ot', DIV_ONLY,
    'D8d: Leray projection + OT inverse at each substep',
)

ROUND1_EXPERIMENTS['D8e_leray_plus_pcfm'] = _exp(
    'leray_plus_pcfm', FULL_CONSTRAINTS,
    'D8e: Leray for div-free + PCFM for energy/momentum/IC',
    relaxation_lambda=1.0,
)

ROUND1_EXPERIMENTS['D8f_leray_vfield'] = _exp(
    'leray_vfield', DIV_ONLY,
    'D8f: Leray on velocity field output (project v_theta before Euler step)',
)

# ---------------------------------------------------------------------------
# Priority 5: PCFM vs CCFM direct comparison (D9a)
# ---------------------------------------------------------------------------

ROUND1_EXPERIMENTS['D9a1_pcfm_full'] = _exp(
    'pcfm', FULL_CONSTRAINTS,
    'D9a1: PCFM with full constraints (= B1)',
    relaxation_lambda=1.0,
)

ROUND1_EXPERIMENTS['D9a2_ccfm_full'] = _exp(
    'ccfm', FULL_CONSTRAINTS,
    'D9a2: CCFM with full constraints',
    schedule_n=1.0,
)

ROUND1_EXPERIMENTS['D9a3_ccfm_leray'] = {
    'alias_of': 'D9a2_ccfm_full',
    'description': 'D9a3 = D9a2 (CCFM with full constraints; no separate Leray mechanism)',
}

ROUND1_EXPERIMENTS['D9a4_pcfm_leray'] = _exp(
    'leray_plus_pcfm', ConstraintConfig(use_ic=True, use_divergence_free=True,
                                          use_energy=True, use_momentum=True),
    'D9a4: PCFM for energy/momentum/IC + Leray for div-free (= D8e)',
    relaxation_lambda=1.0,
)


def get_experiments(filter_str=None):
    """Get experiments, optionally filtered by name substring.

    Args:
        filter_str: if provided, only return experiments whose name contains this string

    Returns:
        dict of {name: experiment_def}
    """
    if filter_str is None:
        return ROUND1_EXPERIMENTS

    return {k: v for k, v in ROUND1_EXPERIMENTS.items() if filter_str in k}


def get_unique_experiments():
    """Remove duplicate experiments (same sampler + constraints).

    B0=D8a, B1=C1=D9a1, B4=D2e — keep the first occurrence.
    """
    # First pass: map each config key to its first experiment name
    key_to_first = {}  # config_key -> first experiment name
    for name, exp in ROUND1_EXPERIMENTS.items():
        if 'alias_of' in exp and 'sampler_name' not in exp:
            continue  # Skip pre-defined aliases
        key = (exp['sampler_name'],
               exp['constraint_config'].use_divergence_free,
               exp['constraint_config'].use_momentum,
               exp['constraint_config'].use_energy,
               exp['constraint_config'].use_ic,
               tuple(sorted(exp['sampler_kwargs'].items())))
        if key not in key_to_first:
            key_to_first[key] = name

    # Second pass: build unique dict, marking duplicates as aliases
    unique = {}
    for name, exp in ROUND1_EXPERIMENTS.items():
        # Pre-defined aliases (e.g. C8, D9a3)
        if 'alias_of' in exp and 'sampler_name' not in exp:
            unique[name] = exp
            continue
        key = (exp['sampler_name'],
               exp['constraint_config'].use_divergence_free,
               exp['constraint_config'].use_momentum,
               exp['constraint_config'].use_energy,
               exp['constraint_config'].use_ic,
               tuple(sorted(exp['sampler_kwargs'].items())))
        first = key_to_first[key]
        if first == name:
            unique[name] = exp
        else:
            unique[name] = {**exp, 'alias_of': first}
    return unique
