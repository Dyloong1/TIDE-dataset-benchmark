"""
Live-updating result tables for PCFM ablation experiments.

Creates and updates markdown tables at:
- results/pcfm_ablation/results_dns.md
- results/pcfm_ablation/results_jhu.md

Each channel (u, v, w, p) gets its own table section with MSE/MAE/RMSE/RelL2/SSIM.
Physics metrics (divergence, energy, momentum) appear in a separate table.
Single-step and AR results are shown in separate sections.
"""

import json
import re
from pathlib import Path
import math


RESULTS_DIR = Path('results/pcfm_ablation')
CHANNELS = ['u', 'v', 'w', 'p']

# Regex to detect AR round entries like B0_vanilla_R1, B0_vanilla_R2
_AR_ROUND_RE = re.compile(r'_R\d+$')


def format_val(v, fmt='.6f'):
    """Format a value, handling NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{v:{fmt}}'


def _parse_row(json_path):
    """Parse a single JSON result file into a row dict."""
    with open(json_path) as f:
        data = json.load(f)
    name = json_path.stem
    row = {'name': name}
    for ch in CHANNELS:
        row[f'{ch}_mse'] = data.get(f'{ch}_mse', float('nan'))
        row[f'{ch}_mae'] = data.get(f'{ch}_mae', float('nan'))
        row[f'{ch}_rmse'] = data.get(f'{ch}_rmse', float('nan'))
        row[f'{ch}_rel_l2'] = data.get(f'{ch}_rel_l2', float('nan'))
        row[f'{ch}_ssim'] = data.get(f'{ch}_ssim', float('nan'))
    row['div_max'] = data.get('div_max', float('nan'))
    row['div_mean'] = data.get('div_mean', float('nan'))
    row['energy_err_pct'] = data.get('energy_err_pct', float('nan'))
    row['momentum_mean_vel_err'] = data.get('momentum_mean_vel_err',
                                             data.get('momentum_err_pct', float('nan')))
    row['wall_time'] = data.get('wall_time', float('nan'))
    row['nfe'] = data.get('nfe', 0)
    return row


def _render_tables(rows, lines):
    """Render per-channel + physics tables for a set of rows."""
    if not rows:
        return

    # Per-channel tables
    for ch in CHANNELS:
        ch_label = {'u': 'u (x-velocity)', 'v': 'v (y-velocity)',
                    'w': 'w (z-velocity)', 'p': 'p (pressure)'}[ch]
        cols = ['Experiment', 'MSE', 'MAE', 'RMSE', 'RelL2', 'SSIM']
        lines.append(f'### {ch_label}\n')
        lines.append('| ' + ' | '.join(cols) + ' |')
        lines.append('|' + '|'.join(['---'] * len(cols)) + '|')
        for r in rows:
            vals = [
                r['name'],
                format_val(r[f'{ch}_mse']),
                format_val(r[f'{ch}_mae']),
                format_val(r[f'{ch}_rmse']),
                format_val(r[f'{ch}_rel_l2'], '.4f'),
                format_val(r[f'{ch}_ssim'], '.4f'),
            ]
            lines.append('| ' + ' | '.join(vals) + ' |')
        lines.append('')

    # Physics + timing table
    phys_cols = ['Experiment', 'max|div|', 'mean|div|', 'dE(%)', 'dMeanVel',
                 'Time(s)', 'NFE']
    lines.append('### Physics & Timing\n')
    lines.append('| ' + ' | '.join(phys_cols) + ' |')
    lines.append('|' + '|'.join(['---'] * len(phys_cols)) + '|')
    for r in rows:
        vals = [
            r['name'],
            format_val(r['div_max'], '.2e'),
            format_val(r['div_mean'], '.2e'),
            format_val(r['energy_err_pct'], '.2f'),
            format_val(r['momentum_mean_vel_err'], '.2e'),
            format_val(r['wall_time'], '.1f'),
            str(r['nfe']),
        ]
        lines.append('| ' + ' | '.join(vals) + ' |')
    lines.append('')


def update_result_table(data_source):
    """Re-read all JSON results and regenerate the markdown table.

    Args:
        data_source: 'dns' or 'jhu'
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_dir = RESULTS_DIR / data_source
    json_dir.mkdir(parents=True, exist_ok=True)

    # Collect all results, split into single-step vs AR
    single_rows = []
    ar_rows = []
    for json_path in sorted(json_dir.glob('*.json')):
        # Skip _ar.json summary files (per-round entries are saved separately)
        if json_path.stem.endswith('_ar'):
            continue
        row = _parse_row(json_path)
        if _AR_ROUND_RE.search(row['name']):
            ar_rows.append(row)
        else:
            single_rows.append(row)

    if not single_rows and not ar_rows:
        return

    lines = []
    ds_label = 'TGV (DNS)' if data_source == 'dns' else 'JHU (DNS128)'
    lines.append(f'# PCFM Ablation Results — {ds_label}\n')

    # Single-step section
    if single_rows:
        lines.append('## Single-Step Evaluation\n')
        _render_tables(single_rows, lines)

    # AR section
    if ar_rows:
        lines.append('## Autoregressive Rollout\n')
        _render_tables(ar_rows, lines)

    # Write markdown
    md_path = RESULTS_DIR / f'results_{data_source}.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  [Results table updated: {md_path}]')


def save_experiment_result(exp_name, data_source, metrics):
    """Save single experiment result as JSON and update table.

    Args:
        exp_name: experiment name (e.g., 'B0_vanilla')
        data_source: 'dns' or 'jhu'
        metrics: dict with per-channel metrics ({ch}_mse/mae/rmse/rel_l2/ssim),
                 physics (div_max, div_mean, energy_err_pct, momentum_mean_vel_err),
                 and timing (wall_time, nfe).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_dir = RESULTS_DIR / data_source
    json_dir.mkdir(parents=True, exist_ok=True)

    json_path = json_dir / f'{exp_name}.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    update_result_table(data_source)
