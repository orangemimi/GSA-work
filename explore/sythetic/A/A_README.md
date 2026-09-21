# A/ — Synthetic Diagnostic Framework Pipeline

## Model

```
Y = f(X) + σ(X) · ε
```

Four relationship categories:

| f(X)     | σ(X)     | Category        | Detectable by              |
|----------|----------|-----------------|----------------------------|
| Constant | Constant | True Null       | Nothing (baseline)         |
| Varying  | Constant | Mean-only       | Pearson, Spearman, dcor…   |
| Constant | Varying  | Variance-only   | Spread-sensitive metrics   |
| Varying  | Varying  | Mean + Variance | Both types                 |

## Pipeline

```
A_S1  →  A_S1.1  →  A_S2  →  A_S2.1  →  A_S3  →  A_S3.1
 ↓                    ↓                    ↓
output/S1/         output/S2/           output/S3/
```

### A_S1: Data Generation

Generates scatter data for 22 function families × 12 SNR levels × 4 spread patterns × 8 x-distributions.

**Two output file sets** in `output/S1/`:
- `cases.csv` + `scatter_points.npz` — 114,176 main cases
- `null_expanded_cases.csv` + `null_expanded_points.npz` — 6,880 expanded null cases (for FPR calibration)

### A_S1.1: Visualization

Six figures showing representative examples from each category with f(X) overlay and ±2σ(X) envelopes.

### A_S2: Deterministic Metrics

Computes ~60 metrics on each case (merged main + expanded null = ~121K cases):
- Pearson r, Spearman ρ, covariance
- Distance correlation and covariance
- MINE (MIC, MAS, MEV, MCN)
- LOWESS and GAM curve fits
- Bin-based (η², amplitude, buffer widths)
- Slope-based (endpoint, polyfit, segmented)
- Distribution (KS, Wasserstein)

Output: `output/S2/metrics_full.parquet`

### A_S2.1: Observed Metric Distribution Visualization

Seven figures exploring how metric values behave across categories:
1. Category violin plots (core 4 metrics)
2. Metric response vs SNR (mean-only)
3. Function family heatmap
4. Metric correlation matrix
5. True Null vs Variance-only separation
6. Spread pattern effect
7. X-distribution influence

Output: `output/S2/viz/`

### A_S3: Permutation Test

500-permutation null distribution for all metrics, in 5 phases:

| Phase | Metrics | Mode |
|-------|---------|------|
| 1 | Core 4 + slopes + bins + covariance (~30 metrics) | Full |
| 2 | Distance covariance + correlation | Full |
| 3 | KS + Wasserstein (equal-width + equal-count) | Full |
| 4 | MINE (MIC/MAS/MEV/MCN) | Subset |
| 5 | LOWESS R² | Subset |

Joint test: max-Z across all metrics → p-value → classification (detectable / uncertain / not_detectable).

Output: `output/S3/permutation_all.parquet` + `output/S3/checkpoints/phase[1-5].npz`

### A_S3.1: Permutation Test Results Visualization

Nine figures evaluating the testing framework:
1. Classification overview (detection rates by category)
2. p-value uniformity under True Null (histogram, QQ, CDF)
3. FPR calibration (by α, spread pattern, x-distribution)
4. Null distribution examples (observed vs null Z-scores)
5. Detection power by SNR
6. Detection by function family
7. Metric contribution to joint test (driving metrics)
8. Variance-only detection analysis
9. Metric group ablation

Output: `output/S3/viz/`

## Directory Structure

```
A/
├── A_README.md
├── A_S1_scatterplot_generation.ipynb
├── A_S1.1_scatterplot_visualization.ipynb
├── A_S2_metrics_computation.ipynb
├── A_S2.1_observed_distribution.ipynb
├── A_S3_permutation.ipynb
├── A_S3.1_null_distribution.ipynb
└── output/
    ├── S1/
    │   ├── cases.csv
    │   ├── scatter_points.npz
    │   ├── null_expanded_cases.csv
    │   ├── null_expanded_points.npz
    │   └── viz/  (figures from S1.1)
    ├── S2/
    │   ├── metrics_full.parquet
    │   └── viz/  (figures from S2.1)
    └── S3/
        ├── permutation_all.parquet
        ├── checkpoints/
        └── viz/  (figures from S3.1)
```
