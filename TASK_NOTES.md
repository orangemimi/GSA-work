# Task Notes

Updated: 2026-05-19

## Current Goal

Refactor and rerun the response-curve analysis from `explore/no0/form_260518_curve.ipynb` into a clean, self-contained analysis folder without overwriting the original notebook or previous outputs.

The new analysis computes Form candidate metrics using the Shape, Strength, and Tightness framework for:

- Warm-wet zone (`HW`) transpiration (`T`) with inputs `P` and `LAI`, for CLASSIC and LPJ-GUESS.
- Soil evaporation (`E_soil`) across `HW`, `HD`, `CW`, and `CD`, with inputs `P` and `LAI`, for CLASSIC and LPJ-GUESS.

## Key Decisions

- Created a new output folder at `explore/no0/form_metrics_260519/`.
- Preserved the original data-reading logic and variable maps from `form_260518_curve.ipynb`.
- Added fallback data paths so the notebook can run from the project root `/Users/mimi/Documents/Code/Github/GSA-work/`.
- Reused the seven fitting methods from the existing notebook:
  - Equal-width bins
  - Equal-count bins
  - LOWESS
  - Polynomial degree 3
  - GAM
  - Piecewise linear
  - Symbolic regression
- Made fitting robust by logging method success/failure instead of stopping the full batch.
- Used min-max normalized fitted `x` and fitted `y` for slope-based Shape and Strength metrics.
- Computed Tightness metrics from one residual definition: `residual = y_raw - interpolated_y_fit`.
- Avoided extrapolation for Tightness metrics by only evaluating raw points inside the fitted curve's x range.
- Kept formal U-test and Lind-Mehlum fields as `NaN`; implemented geometric turning-point proxies and a two-line slope proxy instead.
- Added a separate CSV formula dictionary so each metric output column can be traced to its formula or rule.

## Changed Files

Created:

- `TASK_NOTES.md`
- `explore/no0/form_metrics_260519/form_metrics_analysis.ipynb`
- `explore/no0/form_metrics_260519/curves/fitted_curves_all.csv`
- `explore/no0/form_metrics_260519/metrics/form_candidate_metrics_all.csv`
- `explore/no0/form_metrics_260519/metrics/form_candidate_metrics_HW_T.csv`
- `explore/no0/form_metrics_260519/metrics/form_candidate_metrics_all_zones_E_soil.csv`
- `explore/no0/form_metrics_260519/tables/fitting_method_log.csv`
- `explore/no0/form_metrics_260519/tables/form_candidate_metric_formulas.csv`
- `explore/no0/form_metrics_260519/tables/selected_metric_summary.csv`
- `explore/no0/form_metrics_260519/tables/shape_label_counts.csv`
- `explore/no0/form_metrics_260519/tables/shape_label_counts_by_relation.csv`
- `explore/no0/form_metrics_260519/tables/strength_summary_metrics.csv`
- `explore/no0/form_metrics_260519/tables/tightness_summary_metrics.csv`
- PNG and PDF figures under `explore/no0/form_metrics_260519/figures/`.

Important outputs already verified:

- `form_candidate_metrics_all.csv`: 140 rows x 120 columns.
- `form_candidate_metrics_HW_T.csv`: 28 rows x 120 columns.
- `form_candidate_metrics_all_zones_E_soil.csv`: 112 rows x 120 columns.
- `fitted_curves_all.csv`: 291,253 rows x 10 columns.
- `fitting_method_log.csv`: 140 successful fits, no failures or unavailable methods.
- `form_candidate_metric_formulas.csv`: documents all 120 metric output columns, plus notation rows.

Original file intentionally not modified:

- `explore/no0/form_260518_curve.ipynb`

## Known Issues / Caveats

- The system `/usr/bin/python3` environment does not have the required scientific packages such as pandas, matplotlib, scipy, statsmodels, sklearn, pygam, or gplearn.
- The analysis was executed successfully with `/Users/mimi/miniconda3/envs/pip39/bin/python`.
- If opening the notebook interactively, use a kernel backed by the `pip39` environment or install the missing dependencies in the active kernel.
- Symbolic regression is available and succeeded, but it is the slowest fitting method; the full run took about 7.2 minutes.
- Strength metrics were updated after review: `normalized_amplitude` now equals `amplitude_norm_by_y_sd = raw_amplitude / SD(y_raw)`, and `amplitude_norm_by_y_range` is kept separately.
- Strength SD-normalized net-change and total-variation columns were added: `net_change_by_sd`, `absolute_net_change_by_sd`, `abs_net_change_by_sd`, and `total_variation_by_sd`.
- Accumulated response strength now uses accumulated absolute normalized variation, `cumsum(abs(diff(y_fit_norm)))`, rather than an unweighted cumulative sum of local slopes. Legacy `accumulated_slope_*` columns are kept but now store this accumulated variation definition.
- Tightness metrics were updated to match the three-dimension metric summary: overall and local `nrmse_by_sd`, `buffer_width90_by_sd`, `buffer_width95_by_sd`, and early/middle/late SD-normalized RMSE and buffer-width metrics are now included.
- Selected-method wide CSV outputs were added for Shape, Strength, and Tightness. These keep only `Equal-width bins`, `Equal-count bins`, `LOWESS`, and `Generalized Additive Model (GAM)` as method columns.
- Older `no0` metric outputs are not directly comparable to the new `form_metrics_260519` metrics without checking both data source and formula definitions.
- The new notebook was run from the project root and used `data/preprocessed/preprocessed_with_zones` via fallback. Older `no0` notebooks often used `explore/no0/preprocessed/preprocessed_with_zones` or `explore/data/preprocessed/no0/preprocessed_with_zones`.
- The data path difference changes sample counts. For example, root data has CLASSIC `HW` = 12,340 rows and LPJ-GUESS `HW` = 12,538 rows, while the older `no0` data has CLASSIC `HW` = 12,047 rows and LPJ-GUESS `HW` = 12,149 rows.
- Formal U-test and Lind-Mehlum tests are not implemented; their output fields are intentionally `NaN`.
- The derived `shape_label` is rule-based and meant for interpretation/visualization only; the candidate metrics are the primary outputs.
- Previous metric inventory and HW/T old-vs-current comparison were saved to:
  - `explore/no0/form_metrics_260519/tables/no0_previous_metric_inventory.csv`
  - `explore/no0/form_metrics_260519/tables/HW_T_old_vs_current_common_metrics.csv`
- Current mapping against `/Users/mimi/Downloads/three_dimensions_metrics_summary.csv` was saved to:
  - `explore/no0/form_metrics_260519/tables/three_dimensions_metrics_current_mapping.csv`
- Selected-method wide tables were saved to:
  - `explore/no0/form_metrics_260519/tables/shape_metrics_wide_selected_methods.csv`
  - `explore/no0/form_metrics_260519/tables/strength_metrics_wide_selected_methods.csv`
  - `explore/no0/form_metrics_260519/tables/tightness_metrics_wide_selected_methods.csv`

## Already Ruled Out

- Do not overwrite or edit `explore/no0/form_260518_curve.ipynb`.
- Do not overwrite earlier output folders from previous analyses.
- Do not use the bare system `python3` for reruns.
- Do not install or change dependencies unless explicitly requested.
- Do not make unavailable or failed fitting methods fatal to the full analysis.
- Do not extrapolate fitted curves beyond their fitted x range for residual-based Tightness metrics.
- Do not treat formal U-shape tests as implemented unless a robust implementation is added later.
- Do not assume older `form_output/shape_metrics.csv`, `strength_metrics.csv`, and `tightness_metrics.csv` are independent earlier metrics; `form_260518_curve.ipynb` cell 13 reads `form_metrics_260519/metrics/form_candidate_metrics_all.csv` and splits it into those three tables.
- Do not mix residual-based Tightness with bin-based data certainty unless explicitly adding a separate bin-certainty module; bin-based certainty is a useful old metric family but is not the same residual definition.
