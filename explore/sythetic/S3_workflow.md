# Permutation-Based Relationship Detection — Workflow

## Goal

Given a pair of variables (x, y) with n observations, determine whether
there is a **statistically detectable relationship** between them — regardless
of the relationship's functional form (linear, monotonic, nonlinear, etc.).

This method was developed and validated on synthetic data and is intended
for application to **Earth System Model (ESM) input–output pairs**.

---

## 1. Data Generation (upstream, synthetic only)

Each synthetic case is a (x, y) pair defined by a three-knob model:

```
y = f(x) + ε(x),    x ~ p(x)
```

| Knob | Options | Count |
|------|---------|-------|
| f(x) — function family | F01–F22 + Null | 23 |
| SNR — signal-to-noise ratio | 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, ∞ | 12 |
| ε(x) — noise spread pattern | constant, increasing, decreasing, middle_high | 4 |
| p(x) — x distribution | even, left_dense, right_dense, center_dense, clusters_2/3/4/5 | 8 |

**Total: 114,176 cases** (including 128 Null cases where f(x)=0).

### 1.1 Four Categories of Cases

| Category | f(x) | σ(x) | What exists |
|----------|------|------|-------------|
| **True Null** | constant (=0) | constant | No relationship at all |
| **Mean-only** | ≠ constant | constant | Mean response depends on x |
| **Variance-only** | constant (=0) | varies with x | Only noise spread depends on x |
| **Mean + Variance** | ≠ constant | varies with x | Both mean and spread depend on x |

The permutation test detects **any statistical dependence** between x and y
(H₀: x ⊥ y), so it correctly flags both mean-only and variance-only cases
as "detectable". Variance-only cases are not false positives — they contain
real x–y dependence (in variance, not in mean).

---

## 2. Metric Selection

Four metrics are computed per case, chosen for complementary sensitivity:

| Metric | Symbol | Captures | Range |
|--------|--------|----------|-------|
| Absolute Pearson correlation | \|r\| | Linear dependence | [0, 1] |
| Absolute Spearman correlation | \|ρ\| | Monotonic dependence | [0, 1] |
| Distance correlation | dcor | Arbitrary dependence (incl. nonlinear, variance) | [0, 1] |
| Eta-squared (equal-width bins) | η² | Mean-response signal across x-regions | [0, 1] |

All four are **non-negative**: higher value = stronger relationship.

**Why 4 metrics?** Cross-metric null correlation analysis (S5, Plot 10) shows
η² has low correlation with the other three (r ≈ 0.27–0.44), meaning it
provides substantially independent information. The joint test leverages this
complementarity to detect relationships that any single metric might miss.

---

## 3. Permutation Test Procedure

### 3.1 Null Hypothesis

**H₀: x and y are statistically independent.**

Under H₀, shuffling y while keeping x fixed should produce metric values
that are comparable to the observed value.

### 3.2 Permutation Protocol

For each case i:

1. **Compute observed metrics** on the original (x, y):
   - M_obs = {|r|_obs, |ρ|_obs, dcor_obs, η²_obs}

2. **Generate 500 permutations** of y (deterministic seeds):
   - For each permutation k (k = 1, …, 500):
     - Shuffle y → y_perm
     - Compute M_null^(k) = {|r|_k, |ρ|_k, dcor_k, η²_k}

3. **Result**: per case, per metric:
   - 1 observed value
   - 500 null values (the null distribution)

### 3.3 Computation Phases

| Phase | Metrics | Method | Time (114K cases) | Checkpoint |
|-------|---------|--------|-----|------------|
| Phase 1 | \|r\|, \|ρ\|, η² | Vectorised | ~20 min | `_perm_phase1.npz` |
| Phase 2 | dcor | Loop (n×n distance matrix per perm) | ~15 h | `_perm_phase2.npz` |
| Phase 3 | Z-scores + joint test | Vectorised | ~1 min | final parquet |

---

## 4. Z-Score Normalization

Raw metric values are not comparable across metrics (different scales,
different null baselines). Normalize each to a Z-score:

```
Z_m = (M_obs − median(M_null)) / IQR(M_null)
```

Where:
- median(M_null) = median of the 500 null values
- IQR(M_null) = Q75 − Q25 of the 500 null values

**Fallback**: if IQR < 1e-12, use std(M_null) × 1.35 instead.

**Why median/IQR instead of mean/std?**
The null distributions are right-skewed (all metrics are non-negative
absolute values; S5 Plot 2 confirms skewness ≈ 1 across all metrics),
so median/IQR is more robust than mean/std.

**Validation (S5 Plot 5)**: Under 4,032 True Null cases, the Z-scores
have median = 0.00 and IQR = 1.00 for all four metrics, confirming the
normalization is correctly calibrated.

Typical null distribution statistics:

| Metric | Null median | Null IQR | Null shape |
|--------|-------------|----------|------------|
| \|r\| | ~0.030 | ~0.037 | Right-skewed |
| \|ρ\| | ~0.030 | ~0.037 | Right-skewed |
| dcor | ~0.067 | ~0.021 | Right-skewed |
| η² | ~0.015 | ~0.010 | Right-skewed |

---

## 5. Joint Test

### 5.1 Test Statistic

Combine the four Z-scores into a single joint statistic using the max:

```
T_joint = max(Z_|r|, Z_|ρ|, Z_dcor, Z_η²)
```

**Rationale**: if *any* metric shows strong departure from the null, the
relationship is detectable. The max captures this without assuming which
type of dependence is present.

### 5.2 Joint p-value

The p-value accounts for multiple testing by comparing T_joint against the
*joint* null distribution of T:

```
For each permutation k:
    T_null^(k) = max(Z_|r|^(k), Z_|ρ|^(k), Z_dcor^(k), Z_η²^(k))

p = (#{T_null^(k) ≥ T_obs} + 1) / (500 + 1)
```

This is exact (non-parametric) and inherently controls for the correlation
structure between metrics.

**Note**: The minimum achievable p-value with 500 permutations is 1/501 ≈ 0.002.

### 5.3 Classification

| p-value range | Classification | Meaning |
|---------------|----------------|---------|
| p ≤ 0.05 | **detectable** | Reject H₀ — relationship exists |
| 0.05 < p < 0.10 | **uncertain** | Inconclusive |
| p ≥ 0.10 | **not_detectable** | Cannot reject H₀ |

---

## 6. Output

### 6.1 Final Table

File: `generated_scatterplot_data/full/S3/permutation_test.parquet`

| Column | Description |
|--------|-------------|
| case_id | Case identifier (1-indexed) |
| pearson_obs, spearman_obs, dcor_obs, eta2_obs | Observed metric values |
| {metric}_null_median | Median of the 500 null values |
| {metric}_null_iqr | IQR of the 500 null values |
| z_pearson, z_spearman, z_dcor, z_eta2 | Z-scores |
| T_joint | Joint test statistic = max(Z) |
| p_value | Joint permutation p-value |
| classification | detectable / uncertain / not_detectable |

Shape: 114,176 rows × 20 columns (18 MB).

### 6.2 Raw Null Distributions

Available in checkpoint files for deeper analysis:

| File | Contents | Size |
|------|----------|------|
| `full/S3/_perm_phase1.npz` | pearson/spearman/eta2 obs + null (114K × 500) | 615 MB |
| `full/S3/_perm_phase2.npz` | dcor obs + null (114K × 500) | 196 MB |

---

## 7. Results Summary (Synthetic Data)

### 7.1 Overall Detection

| Classification | Count | Fraction |
|----------------|-------|----------|
| detectable | 113,127 | 99.1% |
| not_detectable | 810 | 0.7% |
| uncertain | 239 | 0.2% |

### 7.2 Detection Rate by SNR (Signal cases only)

| SNR | Detection rate |
|-----|---------------|
| 0.1 | 96.5% |
| 0.3 | 98.2% |
| 0.5 | 98.6% |
| 1.0 | 99.1% |
| ≥2.0 | >99.3% |

Even at the weakest SNR=0.1 (noise is 10× the signal), the joint test
detects 96.5% of relationships. Effect size analysis (S5 Plot 9) shows
the observed metric value is 5–6× the null baseline even at SNR=0.1.

### 7.3 Which Metric Drives Detection?

T_joint is driven by whichever metric has the highest Z-score:

| Driver metric | Fraction of Signal cases |
|---------------|-------------------------|
| η² | 80.5% |
| dcor | 18.9% |
| \|r\| / \|ρ\| | 0.6% |

η² dominates because its null IQR is very tight (~0.010), amplifying even
small departures into large Z-scores.

---

## 8. Calibration Validation (S4)

### 8.1 The Problem

The original S3 dataset has only 32 True Null cases (f=0, constant noise),
yielding an observed FP rate of 4/32 = 12.5% with 95% CI [5.0%, 28.1%].
This wide CI makes it impossible to determine whether the method is
properly calibrated at the 5% significance level.

### 8.2 Validation Approach

S4 generates **4,000 additional True Null cases** (500 per x_distribution,
8 distributions) with y = N(0,1) independent of x, then runs the identical
permutation test. Results are combined with S3's 32 True Null cases.

### 8.3 Results (Combined: 4,032 True Null cases)

**Overall FP rate: 217/4,032 = 5.38%, 95% CI [4.73%, 6.12%]**

The 5% target falls within the CI. **The method is correctly calibrated.**

The earlier 12.5% (4/32) was random fluctuation from insufficient sample size.

| x_distribution | FP rate | n |
|----------------|---------|---|
| center_dense | 4.4% | 504 |
| left_dense | 4.2% | 504 |
| clusters_3 | 4.4% | 504 |
| clusters_4 | 5.0% | 504 |
| even | 5.6% | 504 |
| clusters_2 | 5.6% | 504 |
| right_dense | 6.4% | 504 |
| clusters_5 | 7.0% | 504 |

All x_distributions are within acceptable range. clusters_5 is slightly
elevated (7%), possibly due to η² bin instability with 5 clusters, but
not a systematic concern.

### 8.4 FP Driver Breakdown

Among the 217 false positives:
- dcor drives 56% (119 cases)
- η² drives 30% (65 cases)
- |Pearson| drives 9% (19 cases)
- |Spearman| drives 5% (10 cases)

### 8.5 p-value Calibration

Under True Null, p-values should follow Uniform(0,1). S5 Plot 8 confirms:
- Histogram is flat around density = 1
- CDF closely follows the diagonal
- QQ-plot shows near-perfect agreement

The permutation p-values are accurately calibrated — when the test reports
p = 0.03, there truly is only a ~3% chance of seeing that result under H₀.

---

## 9. Heteroscedastic Null Cases

### The Observation

Among the 128 Null cases in S3 (f=0, all 4 spread patterns), the overall
"FP" rate is 75.8% (97/128):

| Null spread_pattern | Detection rate | Primary driver |
|---------------------|----------------|----------------|
| constant | 12.5% (→ validated at ~5%) | random |
| increasing | 97% | dcor |
| decreasing | 100% | dcor |
| middle_high | 94% | dcor |

### Why This Is Not a Bug

In Null cases with heteroscedastic noise, y = 0 + ε(x) where Var(ε) depends
on x. The permutation test correctly detects that **y's variance depends on x**,
which is a real form of statistical dependence. dcor is sensitive to all
forms of dependence (not just mean response), so it flags these correctly.

### The Four-Category Framework

| Category | f(x) | σ(x) | Permutation test says | Correct? |
|----------|------|------|----------------------|----------|
| True Null | =0 | constant | not_detectable (~95%) | ✓ |
| Mean-only | ≠0 | constant | detectable | ✓ |
| Variance-only | =0 | varies | detectable | ✓ (real dependence) |
| Mean+Variance | ≠0 | varies | detectable | ✓ |

The test answers "is there **any** statistical dependence?" — not "is there
a mean-response relationship?" For ESM applications, this is the appropriate
first-level screening: if two variables are statistically independent, there
is nothing further to analyze.

---

## 10. Detection Power & Metric Ablation (S6)

S6 performs comprehensive detection power analysis using S3's results and
raw null distributions. No permutations are rerun — metric ablation
reconstructs joint tests from stored NPZ arrays.

### 10.1 Four-Category Performance

Each of the 114,176 cases is classified into one of four categories based
on whether f(x) and σ(x) are constant or not:

| Category | Definition | Expected outcome |
|----------|-----------|-----------------|
| True Null | f=0, σ=const | not_detectable (FP if detected) |
| Mean-only | f≠0, σ=const | detectable |
| Variance-only | f=0, σ≠const | detectable (real dependence) |
| Mean+Variance | f≠0, σ≠const | detectable |

### 10.2 Mean-only Analysis

- **Function × SNR heatmap**: detection rate for each of the 22 function
  families at each of the 12 SNR levels. Reveals which functional forms
  are hardest to detect (e.g., high-frequency oscillations at low SNR).
- **SNR curves**: per-function and overall detection rate vs SNR. Even at
  SNR=0.1, overall detection exceeds 96%.
- **Driver by SNR**: at low SNR, η² and dcor dominate; at high SNR,
  Pearson/Spearman become competitive as linear approximation improves.

### 10.3 Variance-only Analysis

Compares Z-score distributions of variance-only cases vs True Null.
dcor shows the strongest separation — it is the primary detector of
heteroscedastic dependence, while Pearson and Spearman remain near null.

### 10.4 Metric Ablation

Reconstructs the joint test T_S = max_{m∈S} Z_m for 12 metric subsets
using the stored null arrays (no recomputation needed):

- **Single metrics**: pearson / spearman / dcor / eta2 alone
- **Pairs and triples**: pearson+spearman, pearson+spearman+eta2, etc.
- **Leave-one-out**: no dcor / no eta2 / no pearson / no spearman

For each subset, reports detection rate per category and True Null FPR.

### 10.5 Unique Contribution

U_m = P(full test detects AND test-without-m does not). Quantifies each
metric's irreplaceable contribution to detection power.

### 10.6 Additional Analyses

- **X distribution influence**: detection rate by x_distribution, with
  focus on low-SNR regime where distribution shape matters most.
- **Error analysis**: profiles false negatives (signal cases classified
  as not_detectable) by SNR, function family, x_distribution, and
  spread_pattern.
- **Threshold sensitivity**: FPR and power across α from 0.001 to 0.20,
  showing the FPR–power tradeoff curve.

---

## 11. Application to ESM Data

### 11.1 How to Use

For any ESM input–output pair (x, y) with n observations:

1. Compute 4 observed metrics: |Pearson|, |Spearman|, dcor, η²
2. Shuffle y 500 times, compute 4 metrics each time → null distribution
3. Z-score normalize → joint test → p-value
4. p ≤ 0.05 → detectable relationship; p ≥ 0.10 → no evidence of relationship

### 11.2 Practical Considerations

- **Sample size**: Validated at n=500. Larger n increases power (almost any
  relationship becomes detectable); smaller n decreases power. The dcor
  computation is O(n²), so n > 5000 may need subsampling.
- **Multiple testing**: If testing many variable pairs (e.g., 50 inputs ×
  50 outputs = 2,500 pairs), apply FDR correction (Benjamini-Hochberg)
  to the p-values.
- **After detection**: The permutation test only answers yes/no. Further
  analysis (relationship type, functional form, effect size) requires
  additional steps beyond this workflow.

---

## 12. File Map

```
Notebooks:
  S1_scatterplot_generation_workflow.ipynb   ← Data generation (114K cases)
  S2_metrics_computation_full.ipynb          ← Descriptive metrics computation
  S3_permutation_test.ipynb                  ← Permutation test (3 phases)
  S3.2_visualization_analysis.ipynb          ← S3 null distribution analysis
  S4_Null_validate.ipynb                     ← True Null calibration (4K cases)
  S5_visualization_analysis.ipynb            ← Combined visualization & analysis
  S6_detection_power_analysis.ipynb          ← Detection power & metric ablation

Documentation:
  S3_workflow.md                             ← This document
  S3_feedback.md                             ← Agent feedback on S3

Data:
  generated_scatterplot_data/
  ├── cases.csv                              ← Case metadata (114K rows)
  ├── scatter_points.npz                     ← Raw (x, y) arrays
  └── full/
      ├── S3/
      │   ├── permutation_test.parquet       ← S3 results (114K × 20)
      │   ├── _perm_phase1.npz              ← Phase 1 checkpoint (615 MB)
      │   ├── _perm_phase2.npz              ← Phase 2 checkpoint (196 MB)
      │   ├── viz/                           ← S3.2_visualization plots
      │   └── viz_analysis/                  ← S3.2_visualization_analysis plots
      ├── S4_null_validate/
      │   ├── null_validate.parquet          ← S4 results (4K rows)
      │   ├── _null_phase1.npz              ← S4 Phase 1 checkpoint
      │   ├── _null_phase2.npz              ← S4 Phase 2 checkpoint
      │   └── null_meta.csv                  ← S4 case metadata
      ├── S5_viz/                            ← S5 combined analysis plots
      └── S6_power/                           ← S6 detection power plots
```
