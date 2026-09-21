import json
from pathlib import Path


path = Path("case/caseA/S4_CMIP6_spatial_relationship_classification.ipynb")
notebook = json.loads(path.read_text())


def replace_cell(cell_id, source, cell_type="code"):
    cell = next(item for item in notebook["cells"] if item.get("id") == cell_id)
    cell["cell_type"] = cell_type
    cell["metadata"] = {}
    cell["source"] = source.splitlines(keepends=True)
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    else:
        cell.pop("execution_count", None)
        cell.pop("outputs", None)


replace_cell(
    "branch-hexbin-intro",
    """## 12. Active branch detector: multi-extent Hexbin

This is the only active static-branch calculation in this notebook. It does not use model/domain names or hand-entered positive labels. For every panel it evaluates the full data extent and a lightly clipped 0.25–99.75% extent, preventing a few RAW outliers from compressing the populated density field. Within each extent, a Branch requires two populated and compact conditional density peaks, a low-density valley, persistence across neighboring x slabs, and agreement from at least 2 of 4 Hexbin grid sizes (18, 24, 32, 40).

Results are reported as **Branch** (≥2 grid sizes agree in either extent), **Candidate** (one-grid evidence), or **No branch**. Coverage and path roughness are retained as diagnostics rather than hard rejection rules. The method is applied to all 5 models × 6 domains × 3 variable pairs × raw/trimmed versions = 180 panels. These are geometric branch labels, not dynamical bifurcation tests.
""",
    "markdown",
)

replace_cell(
    "branch-hexbin-run",
    """import importlib
import sys

if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

import branch_benchmark as _branch_benchmark
_branch_benchmark = importlib.reload(_branch_benchmark)
from branch_benchmark import detect_multiextent_hexbin, plot_branch_result

HEXBIN_GRIDSIZES = (18, 24, 32, 40)
HEXBIN_EXTENT_QUANTILES = (0.0, 0.0025)
HEXBIN_REQUIRED_SUPPORT = 2

HEXBIN_CASE_DATA = {}
HEXBIN_RESULT_OBJECTS = {}
_hexbin_rows = []
_hexbin_started = _time.perf_counter()

for run in RUNS:
    model = run['model']
    frame = run['clim']
    for x_var, y_var, pair_label in VARIABLE_PAIRS:
        x_all = np.asarray(frame[x_var], dtype=float)
        y_all = np.asarray(frame[y_var], dtype=float)
        finite_all = np.isfinite(x_all) & np.isfinite(y_all)
        x_lo, x_hi = np.quantile(x_all[finite_all], [0.01, 0.99])
        y_lo, y_hi = np.quantile(y_all[finite_all], [0.01, 0.99])

        for domain_key, _, domain_filter in DOMAINS:
            subset = frame if domain_filter is None else frame.loc[domain_filter(frame)]
            x_domain = np.asarray(subset[x_var], dtype=float)
            y_domain = np.asarray(subset[y_var], dtype=float)
            finite = np.isfinite(x_domain) & np.isfinite(y_domain)
            trim_keep = (
                finite
                & (x_domain >= x_lo) & (x_domain <= x_hi)
                & (y_domain >= y_lo) & (y_domain <= y_hi)
            )

            for version, keep in [('raw', finite), ('trimmed', trim_keep)]:
                x_values = x_domain[keep]
                y_values = y_domain[keep]
                result = detect_multiextent_hexbin(
                    x_values,
                    y_values,
                    extent_quantiles=HEXBIN_EXTENT_QUANTILES,
                    gridsizes=HEXBIN_GRIDSIZES,
                    required_support=HEXBIN_REQUIRED_SUPPORT,
                )
                grid_support = int(round(result.score * len(HEXBIN_GRIDSIZES)))
                if result.branch_detected:
                    branch_status, status_code = 'Branch', 2
                elif grid_support >= 1:
                    branch_status, status_code = 'Candidate', 1
                else:
                    branch_status, status_code = 'No branch', 0
                extent_runs = (result.geometry or {}).get('extent_runs', [])
                extent_votes = int(sum(item['branch_detected'] for item in extent_runs))
                selected_clip = (result.geometry or {}).get('selected_extent_quantile', np.nan)
                key = (model, domain_key, pair_label, version)
                HEXBIN_CASE_DATA[key] = (x_values, y_values)
                HEXBIN_RESULT_OBJECTS[key] = result
                _hexbin_rows.append({
                    'model': model,
                    'domain': domain_key,
                    'pair': pair_label,
                    'version': version,
                    'n_points': len(x_values),
                    'branch_status': branch_status,
                    'status_code': status_code,
                    'grid_support': grid_support,
                    'extent_votes': extent_votes,
                    'selected_clip_quantile': selected_clip,
                    **result.to_dict(),
                })

HEXBIN_RESULTS_DF = pd.DataFrame(_hexbin_rows)
print(
    f'Multi-extent Hexbin completed: {len(HEXBIN_RESULTS_DF)} panels in '
    f'{_time.perf_counter() - _hexbin_started:.1f}s'
)
""",
)

replace_cell(
    "branch-hexbin-summary",
    """HEXBIN_SUMMARY_DF = (
    HEXBIN_RESULTS_DF
    .groupby(['pair', 'version'], as_index=False)
    .agg(
        panels=('branch_status', 'size'),
        branches=('branch_status', lambda values: int((values == 'Branch').sum())),
        candidates=('branch_status', lambda values: int((values == 'Candidate').sum())),
        median_grid_support=('grid_support', 'median'),
        median_x_coverage=('x_coverage', 'median'),
    )
)

HEXBIN_DETECTIONS_DF = (
    HEXBIN_RESULTS_DF.loc[
        HEXBIN_RESULTS_DF['branch_status'] != 'No branch',
        ['model', 'domain', 'pair', 'version', 'branch_status', 'n_points',
         'grid_support', 'extent_votes', 'selected_clip_quantile',
         'x_coverage', 'min_branch_weight', 'branch_separation'],
    ]
    .sort_values(['branch_status', 'pair', 'version', 'model', 'domain'])
)
HEXBIN_BRANCHES_DF = HEXBIN_DETECTIONS_DF.loc[
    HEXBIN_DETECTIONS_DF['branch_status'] == 'Branch'
].copy()
HEXBIN_CANDIDATES_DF = HEXBIN_DETECTIONS_DF.loc[
    HEXBIN_DETECTIONS_DF['branch_status'] == 'Candidate'
].copy()

print('Multi-extent Hexbin summary')
display(HEXBIN_SUMMARY_DF.round(3))
print(f'Branch panels: {len(HEXBIN_BRANCHES_DF)}')
display(HEXBIN_BRANCHES_DF.round(3))
print(f'Candidate panels: {len(HEXBIN_CANDIDATES_DF)}')
display(HEXBIN_CANDIDATES_DF.round(3))

if WRITE_OUTPUTS:
    HEXBIN_RESULTS_DF.to_csv(
        S4_OUTPUT_DIR / 'S4_branch_hexbin_all_cases_results.csv', index=False,
    )
    HEXBIN_SUMMARY_DF.to_csv(
        S4_OUTPUT_DIR / 'S4_branch_hexbin_all_cases_summary.csv', index=False,
    )
    HEXBIN_BRANCHES_DF.to_csv(
        S4_OUTPUT_DIR / 'S4_branch_hexbin_detected_cases.csv', index=False,
    )
    HEXBIN_CANDIDATES_DF.to_csv(
        S4_OUTPUT_DIR / 'S4_branch_hexbin_candidate_cases.csv', index=False,
    )
    HEXBIN_RESULTS_DF.loc[HEXBIN_RESULTS_DF['pair'] == 'P → Q'].to_csv(
        S4_OUTPUT_DIR / 'S4_branch_hexbin_P_Q_results.csv', index=False,
    )
""",
)

replace_cell(
    "branch-hexbin-stability-figure",
    """from matplotlib.colors import ListedColormap

# Compact all-case Branch/Candidate heatmap.
_hexbin_status_cmap = ListedColormap(['#F3F4F6', '#F59E0B', '#2563EB'])
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
for row_index, version in enumerate(['raw', 'trimmed']):
    for col_index, (_, _, pair_label) in enumerate(VARIABLE_PAIRS):
        ax = axes[row_index, col_index]
        matrix = (
            HEXBIN_RESULTS_DF.loc[
                (HEXBIN_RESULTS_DF['pair'] == pair_label)
                & (HEXBIN_RESULTS_DF['version'] == version)
            ]
            .pivot(index='model', columns='domain', values='status_code')
            .reindex(index=MODELS, columns=[item[0] for item in DOMAINS])
            .fillna(0).astype(int)
        )
        ax.imshow(matrix, vmin=0, vmax=2, cmap=_hexbin_status_cmap, aspect='auto')
        for row_i in range(matrix.shape[0]):
            for col_i in range(matrix.shape[1]):
                value = int(matrix.iloc[row_i, col_i])
                ax.text(
                    col_i, row_i, {0: '–', 1: 'C', 2: 'B'}[value],
                    ha='center', va='center', fontsize=10,
                    color='white' if value else '#4B5563', fontweight='bold',
                )
        ax.set_xticks(range(len(DOMAINS)), [item[0] for item in DOMAINS], rotation=35, ha='right')
        ax.set_yticks(range(len(MODELS)), MODELS)
        n_branch = int((matrix.to_numpy() == 2).sum())
        n_candidate = int((matrix.to_numpy() == 1).sum())
        ax.set_title(
            f'{pair_label} · {version.upper()} · B={n_branch}, C={n_candidate}',
            fontsize=10, fontweight='bold',
        )
        ax.grid(False)
fig.suptitle('Multi-extent Hexbin: Branch (B) and Candidate (C)', fontsize=15, fontweight='bold')
HEXBIN_HEATMAP_PATH = FIGURE_DIR / 'S4_branch_hexbin_all_cases_heatmap.png'
if WRITE_OUTPUTS:
    fig.savefig(HEXBIN_HEATMAP_PATH, dpi=180, bbox_inches='tight')
    print(f'Saved {HEXBIN_HEATMAP_PATH}')
plt.show()
plt.close(fig)


def draw_hexbin_case_grid(pair_label, version, show=False):
    fig, axes = plt.subplots(5, 6, figsize=(18, 14), constrained_layout=True)
    for row_index, model in enumerate(MODELS):
        for col_index, (domain_key, domain_label, _) in enumerate(DOMAINS):
            ax = axes[row_index, col_index]
            key = (model, domain_key, pair_label, version)
            x_values, y_values = HEXBIN_CASE_DATA[key]
            result = HEXBIN_RESULT_OBJECTS[key]
            row = HEXBIN_RESULTS_DF.loc[
                (HEXBIN_RESULTS_DF['model'] == model)
                & (HEXBIN_RESULTS_DF['domain'] == domain_key)
                & (HEXBIN_RESULTS_DF['pair'] == pair_label)
                & (HEXBIN_RESULTS_DF['version'] == version)
            ].iloc[0]
            plot_branch_result(ax, x_values, y_values, result)
            status = row['branch_status']
            color = {'Branch': '#B91C1C', 'Candidate': '#B45309', 'No branch': '#374151'}[status]
            ax.set_title(
                f\"{model} · {domain_label}\\n{status} · support {int(row['grid_support'])}/4\",
                fontsize=9, color=color, fontweight='bold',
            )
            if row_index == len(MODELS) - 1:
                ax.set_xlabel(pair_label.split(' → ')[0], fontsize=8)
            if col_index == 0:
                ax.set_ylabel(pair_label.split(' → ')[1], fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(
        f'{pair_label} · {version.upper()} · multi-extent Hexbin branch detector',
        fontsize=16, fontweight='bold',
    )
    output_path = FIGURE_DIR / f'S4_branch_hexbin_{PAIR_SLUGS[pair_label]}_{version}.png'
    if WRITE_OUTPUTS:
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
    if show:
        print(f'Saved {output_path}')
        plt.show()
    plt.close(fig)
    return output_path


HEXBIN_GRID_PATHS = {}
for _, _, pair_label in VARIABLE_PAIRS:
    for version in ['raw', 'trimmed']:
        key = (pair_label, version)
        HEXBIN_GRID_PATHS[key] = draw_hexbin_case_grid(
            pair_label, version,
            show=(pair_label == 'P → Q'),
        )


# Local threshold audit without visual labels.
_hexbin_stability_rows = []
for valley_depth in [0.30, 0.35, 0.40]:
    for relative_peak in [0.13, 0.15, 0.17]:
        for model in MODELS:
            for domain_key, _, _ in DOMAINS:
                for version in ['raw', 'trimmed']:
                    key = (model, domain_key, 'P → Q', version)
                    x_values, y_values = HEXBIN_CASE_DATA[key]
                    result = detect_multiextent_hexbin(
                        x_values, y_values,
                        extent_quantiles=HEXBIN_EXTENT_QUANTILES,
                        gridsizes=HEXBIN_GRIDSIZES,
                        required_support=HEXBIN_REQUIRED_SUPPORT,
                        min_valley_depth=valley_depth,
                        min_relative_peak=relative_peak,
                    )
                    _hexbin_stability_rows.append({
                        'model': model,
                        'domain': domain_key,
                        'version': version,
                        'min_valley_depth': valley_depth,
                        'min_relative_peak': relative_peak,
                        'branch_detected': result.branch_detected,
                        'grid_support': int(round(result.score * len(HEXBIN_GRIDSIZES))),
                    })

HEXBIN_PQ_STABILITY_DF = pd.DataFrame(_hexbin_stability_rows)
HEXBIN_PQ_STABILITY_SUMMARY_DF = (
    HEXBIN_PQ_STABILITY_DF.groupby(['model', 'domain', 'version'], as_index=False)
    .agg(
        detected_settings=('branch_detected', 'sum'),
        total_settings=('branch_detected', 'size'),
    )
)
HEXBIN_PQ_STABILITY_SUMMARY_DF['detection_rate'] = (
    HEXBIN_PQ_STABILITY_SUMMARY_DF['detected_settings']
    / HEXBIN_PQ_STABILITY_SUMMARY_DF['total_settings']
)
print('P→Q threshold sensitivity: panels detected in at least one setting')
display(HEXBIN_PQ_STABILITY_SUMMARY_DF.loc[
    HEXBIN_PQ_STABILITY_SUMMARY_DF['detected_settings'] > 0
].round(3))

if WRITE_OUTPUTS:
    HEXBIN_PQ_STABILITY_DF.to_csv(
        S4_OUTPUT_DIR / 'S4_branch_hexbin_P_Q_threshold_stability.csv', index=False,
    )
    HEXBIN_PQ_STABILITY_SUMMARY_DF.to_csv(
        S4_OUTPUT_DIR / 'S4_branch_hexbin_P_Q_threshold_summary.csv', index=False,
    )
""",
)

path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
