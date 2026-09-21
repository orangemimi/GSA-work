"""Sensitivity test for equal-frequency conditional Dip branch screening.

This is an analysis script only; it does not change the S4 production logic.
It compares Dip tests on raw y with Dip tests on residuals from a local
quadratic trend inside each equal-frequency x bin.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diptest import diptest
from statsmodels.nonparametric.smoothers_lowess import lowess


CASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/Volumes/mimi-T9/CMIP6")
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["CESM2", "CNRM-CM6-1", "CanESM5", "GFDL-CM4", "CMCC-CM2-SR5"]
DOMAINS = ["all_land", "WW", "WD", "CW", "CD", "LI"]
PAIRS = [
    ("P",          "Q", "P → Q"),
    ("ET",         "Q", "ET → Q"),
    ("hfls",       "Q", "hfls → Q"),
    ("hfss",       "Q", "hfss → Q"),
    ("tran",       "Q", "tran → Q"),
    ("evspsblsoi", "Q", "evspsblsoi → Q"),
    ("mrros",      "Q", "mrros → Q"),
    ("mrso",       "Q", "mrso → Q"),
    ("mrsos",      "Q", "mrsos → Q"),
    ("lai",        "Q", "lai → Q"),
    ("tas",        "Q", "tas → Q"),
    ("prsn",       "Q", "prsn → Q"),
    ("rlds",       "Q", "rlds → Q"),
    ("rlus",       "Q", "rlus → Q"),
    ("rsds",       "Q", "rsds → Q"),
    ("rsus",       "Q", "rsus → Q"),
]
K_VALUES = (6, 10, 14)


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def longest_run(flags: np.ndarray) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return int(best)


def local_quadratic_residual(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Remove the within-bin center trend without fitting multiple branches."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    scale = np.ptp(x)
    if len(x) < 8 or not np.isfinite(scale) or scale <= 0:
        return y - np.median(y)
    z = (x - np.mean(x)) / scale
    degree = 2 if len(np.unique(z)) >= 3 else 1
    coefficients = np.polyfit(z, y, degree)
    return y - np.polyval(coefficients, z)


def binned_lowess_residual(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residual from one robust, deliberately smooth centerline per panel."""
    order = np.argsort(x, kind="mergesort")
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    n_knots = min(60, max(12, len(x_sorted) // 100))
    groups = np.array_split(np.arange(len(x_sorted)), n_knots)
    knot_x = np.asarray([np.median(x_sorted[g]) for g in groups])
    knot_y = np.asarray([np.median(y_sorted[g]) for g in groups])
    smooth = lowess(knot_y, knot_x, frac=0.30, it=2, return_sorted=True)
    unique_x, unique_indices = np.unique(smooth[:, 0], return_index=True)
    fitted_sorted = np.interp(x_sorted, unique_x, smooth[unique_indices, 1])
    residual_sorted = y_sorted - fitted_sorted
    residual = np.empty_like(residual_sorted)
    residual[order] = residual_sorted
    return residual


def equal_frequency_dip(x: np.ndarray, y: np.ndarray, n_bins: int, variant: str) -> pd.DataFrame:
    global_residual = binned_lowess_residual(x, y) if variant == "global_residual" else None
    order = np.argsort(x, kind="mergesort")
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    if global_residual is not None:
        global_residual = global_residual[order]
    rows = []
    for bin_index, indices in enumerate(np.array_split(np.arange(len(x)), n_bins)):
        xb = x[indices]
        yb = y[indices]
        if variant == "raw_y":
            values = yb
        elif variant == "global_residual":
            values = global_residual[indices]
        else:
            values = local_quadratic_residual(xb, yb)
        if len(values) < 20 or np.ptp(values) <= 1e-12:
            dip, pvalue = 0.0, 1.0
        else:
            dip, pvalue = diptest(values)
        rows.append({
            "bin": bin_index,
            "n": len(indices),
            "x_min": float(np.min(xb)),
            "x_mid": float(np.median(xb)),
            "x_max": float(np.max(xb)),
            "dip": float(dip),
            "pvalue": float(pvalue),
            "zero_fraction": float(np.mean(np.isclose(yb, 0.0, atol=1e-10))),
        })
    result = pd.DataFrame(rows)
    result["qvalue"] = bh_adjust(result["pvalue"].to_numpy())
    result["significant"] = result["qvalue"] < 0.05
    return result


def load_runs() -> dict[str, pd.DataFrame]:
    runs = {}
    for model in MODELS:
        paths = sorted((DATA_ROOT / model / "historical").glob(
            "*/*/land/zones/zone_climatology_1985_2014.parquet"
        ))
        if not paths:
            raise FileNotFoundError(f"No zone climatology for {model}")
        frame = pd.read_parquet(paths[0])
        runs[model] = frame.rename(columns={"R": "Q"})
    return runs


def build_cases(runs: dict[str, pd.DataFrame], pairs=None):
    if pairs is None:
        pairs = PAIRS
    cases = {}
    for model, frame in runs.items():
        for x_name, y_name, pair in pairs:
            if x_name not in frame.columns or y_name not in frame.columns:
                continue
            finite_all = np.isfinite(frame[x_name]) & np.isfinite(frame[y_name])
            x_all = frame.loc[finite_all, x_name].to_numpy(float)
            y_all = frame.loc[finite_all, y_name].to_numpy(float)
            x_lo, x_hi = np.quantile(x_all, [0.01, 0.99])
            y_lo, y_hi = np.quantile(y_all, [0.01, 0.99])
            for domain in DOMAINS:
                subset = frame if domain == "all_land" else frame.loc[frame["analysis_zone"] == domain]
                x = subset[x_name].to_numpy(float)
                y = subset[y_name].to_numpy(float)
                finite = np.isfinite(x) & np.isfinite(y)
                trimmed = finite & (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
                cases[(model, domain, pair, "raw")] = (x[finite], y[finite])
                cases[(model, domain, pair, "trimmed")] = (x[trimmed], y[trimmed])
    return cases


def run_analysis(cases):
    detail_rows = []
    summary_rows = []
    for key, (x, y) in cases.items():
        model, domain, pair, version = key
        if len(x) < 80:
            continue
        for n_bins in K_VALUES:
            for variant in ("raw_y", "local_residual", "global_residual"):
                detail = equal_frequency_dip(x, y, n_bins, variant)
                detail_rows.append(detail.assign(
                    model=model, domain=domain, pair=pair, version=version,
                    n_bins=n_bins, variant=variant,
                ))
                flags = detail["significant"].to_numpy(bool)
                summary_rows.append({
                    "model": model,
                    "domain": domain,
                    "pair": pair,
                    "version": version,
                    "n_points": len(x),
                    "n_bins": n_bins,
                    "variant": variant,
                    "n_significant": int(flags.sum()),
                    "significant_fraction": float(flags.mean()),
                    "longest_run": longest_run(flags),
                    "max_dip": float(detail["dip"].max()),
                    "median_dip": float(detail["dip"].median()),
                    "min_q": float(detail["qvalue"].min()),
                    "zero_fraction": float(np.mean(np.isclose(y, 0.0, atol=1e-10))),
                })
    return pd.concat(detail_rows, ignore_index=True), pd.DataFrame(summary_rows)


def plot_pq_atlas(cases, detail, variant="global_residual", version="trimmed", n_bins=10):
    fig, axes = plt.subplots(5, 6, figsize=(18, 14), constrained_layout=True)
    for row, model in enumerate(MODELS):
        for col, domain in enumerate(DOMAINS):
            ax = axes[row, col]
            x, y = cases[(model, domain, "P → Q", version)]
            ax.hexbin(x, y, gridsize=34, bins="log", mincnt=1, cmap="Greys", alpha=0.82)
            selected = detail.loc[
                (detail["model"] == model) & (detail["domain"] == domain)
                & (detail["pair"] == "P → Q") & (detail["version"] == version)
                & (detail["n_bins"] == n_bins) & (detail["variant"] == variant)
            ].sort_values("bin")
            flags = selected["significant"].to_numpy(bool)
            for item in selected.loc[selected["significant"]].itertuples():
                ax.axvspan(item.x_min, item.x_max, color="#EF4444", alpha=0.16, lw=0)
            max_dip = selected["dip"].max() if len(selected) else np.nan
            ax.set_title(
                f"{model} · {domain}\nq<.05: {int(flags.sum())}/{n_bins}; run={longest_run(flags)}; maxD={max_dip:.3f}",
                fontsize=8.5,
                color="#B91C1C" if flags.any() else "#374151",
                fontweight="bold",
            )
            if row == 4:
                ax.set_xlabel("P", fontsize=8)
            if col == 0:
                ax.set_ylabel("Q", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
    fig.suptitle(
        f"P → Q · {version.upper()} · equal-frequency residual Dip test\n"
        "red x-ranges: within-panel BH q < 0.05",
        fontsize=16, fontweight="bold",
    )
    output = FIGURE_DIR / f"S4_residual_dip_P_Q_{version}_K{n_bins}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_heatmap(summary, version="trimmed", n_bins=10):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), constrained_layout=True)
    for row, variant in enumerate(("raw_y", "local_residual", "global_residual")):
        for col, (_, _, pair) in enumerate(PAIRS):
            ax = axes[row, col]
            selected = summary.loc[
                (summary["version"] == version) & (summary["n_bins"] == n_bins)
                & (summary["variant"] == variant) & (summary["pair"] == pair)
            ]
            matrix = selected.pivot(index="model", columns="domain", values="n_significant").reindex(
                index=MODELS, columns=DOMAINS
            )
            image = ax.imshow(matrix, vmin=0, vmax=n_bins, cmap="YlOrRd", aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    value = int(matrix.iloc[i, j])
                    ax.text(j, i, str(value), ha="center", va="center",
                            color="white" if value >= n_bins / 2 else "#374151", fontweight="bold")
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            variant_label = {
                "raw_y": "raw y", "local_residual": "within-bin residual",
                "global_residual": "global LOWESS residual",
            }[variant]
            ax.set_title(f"{pair} · {variant_label}", fontsize=10)
            ax.grid(False)
    fig.colorbar(image, ax=axes, shrink=0.75, label=f"BH-significant bins out of {n_bins}")
    fig.suptitle(f"Equal-frequency Dip test · {version.upper()} · K={n_bins}", fontsize=15, fontweight="bold")
    output = FIGURE_DIR / f"S4_residual_dip_all_pairs_{version}_K{n_bins}_heatmap.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def add_visual_labels(summary: pd.DataFrame) -> pd.DataFrame:
    reference_path = OUTPUT_DIR / "S4_branch_visual_reference_features_P_Q_trimmed.csv"
    reference = pd.read_csv(reference_path)[["model", "domain", "label", "target"]]
    selected = summary.loc[
        (summary["pair"] == "P → Q") & (summary["version"] == "trimmed")
    ].copy()
    return selected.merge(reference, on=["model", "domain"], how="left", validate="many_to_one")


def main():
    print("Loading CMIP6 zone climatologies...")
    runs = load_runs()
    cases = build_cases(runs)
    print(f"Built {len(cases)} panels")
    detail, summary = run_analysis(cases)
    labelled = add_visual_labels(summary)

    detail_path = OUTPUT_DIR / "S4_residual_dip_bin_details.csv"
    summary_path = OUTPUT_DIR / "S4_residual_dip_panel_summary.csv"
    labelled_path = OUTPUT_DIR / "S4_residual_dip_P_Q_trimmed_visual_comparison.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    labelled.to_csv(labelled_path, index=False)

    atlas = plot_pq_atlas(cases, detail)
    heatmap = plot_heatmap(summary)

    print("\nTRIMMED panel counts with at least one BH-significant bin (K=10):")
    table = (summary.loc[(summary["version"] == "trimmed") & (summary["n_bins"] == 10)]
             .assign(any_sig=lambda d: d["n_significant"] > 0)
             .groupby(["pair", "variant"])["any_sig"].sum().unstack())
    print(table)
    print("\nP→Q labelled comparison, K=10:")
    cols = ["model", "domain", "label", "variant", "n_significant", "longest_run", "max_dip", "min_q"]
    print(labelled.loc[labelled["n_bins"] == 10, cols].to_string(index=False))
    print("\nSensitivity: number of P→Q panels with >=1 significant bin:")
    sensitivity = (labelled.assign(any_sig=lambda d: d["n_significant"] > 0)
                   .groupby(["variant", "n_bins"])["any_sig"].sum().unstack())
    print(sensitivity)
    print(f"\nSaved {detail_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {labelled_path}")
    print(f"Saved {atlas}")
    print(f"Saved {heatmap}")


if __name__ == "__main__":
    main()
