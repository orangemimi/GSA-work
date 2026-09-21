"""
Incrementally add new SNR levels (between 500 and ∞) to the existing dataset.
Generates scatter points and computes all metrics for the new cases,
then appends to cases.csv, scatter_points.npz, and metrics_full.parquet.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── New SNR levels to add ──
NEW_SNRS = [750, 1000, 2000, 5000, 10000]

# ── Load all code from S1 and S2 notebooks via exec ──
def _extract_cell_source(nb_path: str, cell_indices: list[int]) -> str:
    with open(nb_path) as f:
        nb = json.load(f)
    parts = []
    for i in cell_indices:
        src = ''.join(nb['cells'][i]['source'])
        parts.append(src)
    return '\n\n'.join(parts)


S1_DIR = Path('output/S1')
S2_DIR = Path('output/S2')

# ── Execute S1 cells to get shape configs and generation functions ──
s1_code = _extract_cell_source(
    'D_S1_scatterplot_generation.ipynb',
    [1, 3, 5, 7, 9, 13]  # imports, constants, shape_configs, eval, sampling, generate
)
s1_ns: dict[str, Any] = {}
exec(s1_code, s1_ns)

shape_configs = s1_ns['shape_configs']
generate_scatter_xy = s1_ns['generate_scatter_xy']
N_PER_CASE = s1_ns['N_PER_CASE']
N_SIGNAL_REPLICATES = s1_ns['N_SIGNAL_REPLICATES']
FAMILY_LABELS = s1_ns['FAMILY_LABELS']

print(f'Loaded {len(shape_configs)} shape configs from S1')
print(f'New SNR levels: {NEW_SNRS}')

# ── Load existing data ──
existing_cases = pd.read_csv(S1_DIR / 'cases.csv', low_memory=False)
max_case_id = int(existing_cases['case_id'].max())
print(f'Existing: {len(existing_cases):,} cases, max case_id={max_case_id}')

# Check no overlap
existing_snrs = set()
for s in existing_cases['snr'].unique():
    existing_snrs.add(float('inf') if str(s) == 'inf' else float(s))
new_snrs_to_add = [s for s in NEW_SNRS if s not in existing_snrs]
if not new_snrs_to_add:
    print('All requested SNR levels already exist! Nothing to do.')
    exit(0)
print(f'SNR levels to add (not yet in data): {new_snrs_to_add}')

# ── Build new cases ──
new_rows: list[dict[str, Any]] = []
next_id = max_case_id + 1

for cfg in shape_configs:
    labels = FAMILY_LABELS[cfg.family_id]
    for snr in new_snrs_to_add:
        for rep in range(N_SIGNAL_REPLICATES):
            new_rows.append({
                'case_id': next_id,
                'shape_config_id': cfg.shape_config_id,
                'family_id': cfg.family_id,
                'family_name': labels['family_name'],
                'variant_name': cfg.variant_name,
                'variant_level': cfg.variant_level,
                'params_json': json.dumps(cfg.params, sort_keys=True),
                'snr': float(snr),
                'spread_pattern': 'constant',
                'x_distribution': 'even',
                'category': 'true_null' if cfg.family_id == 'Null' else 'mean_only',
                'replicate': rep,
                'n': N_PER_CASE,
                **{k: v for k, v in labels.items() if k != 'family_name'},
            })
            next_id += 1

new_cases_df = pd.DataFrame(new_rows)
print(f'\nNew cases to generate: {len(new_cases_df):,}')
print(f'  shape configs × {len(new_snrs_to_add)} SNR × {N_SIGNAL_REPLICATES} reps = '
      f'{len(shape_configs)} × {len(new_snrs_to_add)} × {N_SIGNAL_REPLICATES} = {len(new_cases_df)}')

# ── Generate scatter points for new cases ──
print('\n=== Generating scatter points ===')
n_new = len(new_cases_df)
x_new = np.empty((n_new, N_PER_CASE), dtype=np.float32)
y_new = np.empty((n_new, N_PER_CASE), dtype=np.float32)

# We need to pass new_cases_df to generate_scatter_xy
# The function looks up case by case_id, so we set index
new_cases_indexed = new_cases_df.copy()

t0 = time.time()
for idx in range(n_new):
    cid = int(new_cases_df.iloc[idx]['case_id'])
    x_new[idx], y_new[idx] = generate_scatter_xy(cid, cases=new_cases_indexed)
    if (idx + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (n_new - idx - 1) / rate
        print(f'  {idx+1:>6,}/{n_new:,}  ({rate:.0f}/s, ETA {eta:.0f}s)')

elapsed = time.time() - t0
print(f'Generated {n_new:,} scatter sets in {elapsed:.1f}s')

# ── Compute all metrics ──
print('\n=== Computing metrics ===')

# Execute S2 cells for metric functions
s2_code = _extract_cell_source(
    'D_S2_metrics_computation.ipynb',
    [1, 5, 6, 7, 8]  # imports, vectorised, helpers, per-case, wrapper
)
s2_ns: dict[str, Any] = {
    'x_all': x_new, 'y_all': y_new, 'n_cases': n_new,
}
exec(s2_code, s2_ns)

vectorised_pearson = s2_ns['vectorised_pearson']
vectorised_spearman = s2_ns['vectorised_spearman']
compute_all_per_case = s2_ns['compute_all_per_case']

# Phase 1: vectorised Pearson + Spearman
t0 = time.time()
x64 = x_new.astype(np.float64)
y64 = y_new.astype(np.float64)
pearson_new = vectorised_pearson(x64, y64)
spearman_new = vectorised_spearman(x64, y64)
del x64, y64
print(f'Phase 1 (Pearson+Spearman): {time.time()-t0:.1f}s')

# Phase 2: per-case metrics
per_case_results = []
t0 = time.time()
for i in range(n_new):
    x = x_new[i].astype(np.float64)
    y = y_new[i].astype(np.float64)
    per_case_results.append(compute_all_per_case(x, y))
    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (n_new - i - 1) / rate
        print(f'  {i+1:>6,}/{n_new:,}  ({rate:.0f}/s, ETA {eta/60:.1f}min)')

elapsed = time.time() - t0
print(f'Phase 2 (per-case metrics): {elapsed:.1f}s ({n_new/elapsed:.0f}/s)')

metrics_new = pd.DataFrame(per_case_results)
metrics_new.insert(0, 'case_id', new_cases_df['case_id'].values)
metrics_new['pearson_r'] = pearson_new
metrics_new['spearman_rho'] = spearman_new

# ── Append to existing files ──
print('\n=== Saving ===')

# 1. Append cases.csv
combined_cases = pd.concat([existing_cases, new_cases_df], ignore_index=True)
combined_cases.to_csv(S1_DIR / 'cases.csv', index=False)
print(f'cases.csv: {len(existing_cases):,} → {len(combined_cases):,} rows')

# 2. Append scatter_points.npz
pts = np.load(S1_DIR / 'scatter_points.npz')
x_combined = np.concatenate([pts['x'], x_new], axis=0)
y_combined = np.concatenate([pts['y'], y_new], axis=0)
del pts
np.savez_compressed(S1_DIR / 'scatter_points.npz', x=x_combined, y=y_combined)
size_mb = (S1_DIR / 'scatter_points.npz').stat().st_size / 1e6
print(f'scatter_points.npz: shape {x_combined.shape}  ({size_mb:.1f} MB)')
del x_combined, y_combined

# 3. Append metrics_full.parquet
existing_metrics = pd.read_parquet(S2_DIR / 'metrics_full.parquet')
combined_metrics = pd.concat([existing_metrics, metrics_new], ignore_index=True)
combined_metrics.to_parquet(S2_DIR / 'metrics_full.parquet', index=False)
print(f'metrics_full.parquet: {len(existing_metrics):,} → {len(combined_metrics):,} rows')

# ── Summary ──
print(f'\n=== Done ===')
print(f'Added {len(new_snrs_to_add)} SNR levels: {new_snrs_to_add}')
print(f'Generated {n_new:,} new cases')
total_snrs = combined_cases['snr'].nunique()
print(f'Total SNR levels now: {total_snrs}')
print(f'Total cases now: {len(combined_cases):,}')
