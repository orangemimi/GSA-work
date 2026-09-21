"""Calibrate one geometric KDE peak-valley cutoff to the accepted baseline.

The accepted hard-window result supplies reference Branch and No-branch panels.
Ambiguous Two-band and Candidate panels are excluded from cutoff calibration.
No null distribution, p-value, or statistical significance test is used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from branch_benchmark import _robust_location_scale, _window_slices
from test_kde_peak_valley_score import multiband_score
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS
from test_sliding_kde_modal_summary import stratified_cap


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
REFERENCE_PATH = OUTPUT_DIR / "S4_sliding_kde_modal_tracks_results.csv"
CSV_PATH = OUTPUT_DIR / "S4_calibrated_peak_valley_score_results.csv"
FIGURE_PATH = FIGURE_DIR / "S4_calibrated_peak_valley_score_heatmap.png"


def extract_panel(x, y):
    x, y = stratified_cap(x, y)
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    y_center, y_scale = _robust_location_scale(y)
    y_scaled = (y - y_center) / y_scale
    points = []
    for start, stop in _window_slices(len(x), window_frac=0.06):
        estimate = multiband_score(y_scaled[start:stop])
        points.append({
            "x_min": float(x[start]),
            "x_mid": float(np.median(x[start:stop])),
            "x_max": float(x[stop - 1]),
            "score": float(estimate["score"]),
            "modes": tuple(value * y_scale + y_center for value in estimate["modes"]),
        })
    y_lo, y_hi = np.quantile(y, [0.005, 0.995])
    return {
        "points": points,
        "x_range": max(float(np.max(x) - np.min(x)), 1e-12),
        "y_range": max(float(y_hi - y_lo), 1e-12),
    }


def longest_run(panel, cutoff):
    jump_limit = 0.20 * panel["y_range"]
    runs, current = [], []
    previous_modes = None
    for point in panel["points"]:
        modes = np.asarray(point["modes"], dtype=float)
        usable = bool(point["score"] > cutoff and np.all(np.isfinite(modes)))
        connected = bool(
            usable and previous_modes is not None
            and np.max(np.abs(modes - previous_modes)) <= jump_limit
        )
        if usable:
            if current and not connected:
                runs.append(current)
                current = []
            current.append(point)
            previous_modes = modes
        else:
            if current:
                runs.append(current)
                current = []
            previous_modes = None
    if current:
        runs.append(current)
    return max(runs, key=len, default=[])


def classify_panel(panel, cutoff):
    run = longest_run(panel, cutoff)
    run_length = len(run)
    if run:
        coverage = float((run[-1]["x_max"] - run[0]["x_min"]) / panel["x_range"])
        separation = np.asarray([point["modes"][1] - point["modes"][0] for point in run])
        edge = min(2, run_length)
        relative_change = float(
            abs(np.mean(separation[-edge:]) - np.mean(separation[:edge]))
            / max(float(np.max(separation)), 1e-12)
        )
        median_score = float(np.median([point["score"] for point in run]))
        x_start, x_end = run[0]["x_min"], run[-1]["x_max"]
    else:
        coverage = relative_change = median_score = 0.0
        x_start = x_end = np.nan

    if run_length >= 3 and relative_change >= 0.25 and coverage >= 0.10:
        status, branch_type = "Branch", "extended fork"
    elif run_length == 2:
        status, branch_type = "Branch", "local side branch"
    elif run_length >= 3:
        status, branch_type = "Two-band", "stable/ambiguous two-band"
    elif run_length == 1:
        status, branch_type = "Candidate", "one high-score window"
    else:
        status, branch_type = "No branch", "none"
    return {
        "status": status,
        "branch_type": branch_type,
        "run_length": run_length,
        "x_coverage": coverage,
        "relative_separation_change": relative_change,
        "median_peak_valley_score": median_score,
        "x_start": x_start,
        "x_end": x_end,
    }


def balanced_accuracy(reference, predicted):
    reference = np.asarray(reference, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    sensitivity = float(np.mean(predicted[reference])) if np.any(reference) else 0.0
    specificity = float(np.mean(~predicted[~reference])) if np.any(~reference) else 0.0
    return 0.5 * (sensitivity + specificity), sensitivity, specificity


def choose_cutoff(panel_data, references):
    observed_scores = np.asarray([
        point["score"] for panel in panel_data.values() for point in panel["points"]
    ])
    positive_scores = observed_scores[observed_scores > 0]
    candidates = np.unique(np.r_[
        0.0,
        np.quantile(positive_scores, np.linspace(0.02, 0.98, 160)),
    ])
    definitive = references.loc[references["status"].isin(["Branch", "No branch"])].copy()
    rows = []
    for cutoff in candidates:
        predicted = []
        truth = []
        for item in definitive.itertuples():
            key = (item.model, item.domain, item.pair, item.version)
            predicted.append(classify_panel(panel_data[key], float(cutoff))["status"] == "Branch")
            truth.append(item.status == "Branch")
        score, sensitivity, specificity = balanced_accuracy(truth, predicted)
        rows.append({
            "cutoff": float(cutoff),
            "balanced_accuracy": score,
            "sensitivity": sensitivity,
            "specificity": specificity,
        })
    calibration = pd.DataFrame(rows)
    # Prefer the more conservative cutoff when balanced accuracy ties.
    best = calibration.sort_values(
        ["balanced_accuracy", "specificity", "cutoff"], ascending=[False, False, False]
    ).iloc[0]
    calibration.to_csv(OUTPUT_DIR / "S4_peak_valley_cutoff_calibration.csv", index=False)
    return float(best["cutoff"]), best, definitive


def plot_heatmap(results, cutoff, figure_path=FIGURE_PATH,
                 title_prefix="Calibrated single-score KDE peak-valley tracks"):
    codes = {"No branch": 0, "Candidate": 1, "Two-band": 2, "Branch": 3}
    letters = {"No branch": "–", "Candidate": "C", "Two-band": "T", "Branch": "B"}
    cmap = ListedColormap(["#F3F4F6", "#F59E0B", "#7C3AED", "#2563EB"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.7), constrained_layout=True)
    for row, version in enumerate(("raw", "trimmed")):
        for col, (_, _, pair) in enumerate(PAIRS):
            ax = axes[row, col]
            selected = results.loc[
                (results["pair"] == pair) & (results["version"] == version)
            ].copy()
            selected["code"] = selected["status"].map(codes)
            matrix = selected.pivot(index="model", columns="domain", values="code").reindex(
                index=MODELS, columns=DOMAINS
            )
            statuses = selected.pivot(index="model", columns="domain", values="status").reindex(
                index=MODELS, columns=DOMAINS
            )
            ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    status = statuses.iloc[i, j]
                    ax.text(j, i, letters[status], ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if codes[status] >= 2 else "#374151")
            counts = selected["status"].value_counts()
            ax.set_title(
                f"{pair} · {version.upper()} · B={counts.get('Branch', 0)}, "
                f"T={counts.get('Two-band', 0)}, C={counts.get('Candidate', 0)}",
                fontsize=10.5, fontweight="bold",
            )
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.tick_params(labelsize=9)
            ax.set_xticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle(
        f"{title_prefix} · P₀={cutoff:.4f}",
        fontsize=15.5, fontweight="bold",
    )
    fig.savefig(figure_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    cases = build_cases(load_runs())
    panel_data = {}
    for index, (key, (x, y)) in enumerate(cases.items(), start=1):
        panel_data[key] = extract_panel(x, y)
        if index % 15 == 0:
            print(f"Extracted {index}/{len(cases)}", flush=True)

    references = pd.read_csv(REFERENCE_PATH)
    cutoff, calibration_best, definitive = choose_cutoff(panel_data, references)
    rows = []
    for key, panel in panel_data.items():
        model, domain, pair, version = key
        rows.append({
            "model": model, "domain": domain, "pair": pair, "version": version,
            "peak_valley_cutoff": cutoff,
            **classify_panel(panel, cutoff),
        })
    results = pd.DataFrame(rows)
    results.to_csv(CSV_PATH, index=False)
    plot_heatmap(results, cutoff)

    merged = definitive.merge(
        results[["model", "domain", "pair", "version", "status"]],
        on=["model", "domain", "pair", "version"], suffixes=("_reference", "_single"),
    )
    score, sensitivity, specificity = balanced_accuracy(
        merged["status_reference"] == "Branch", merged["status_single"] == "Branch"
    )
    print(f"\nSelected cutoff P0={cutoff:.6f}")
    print(f"Calibration balanced accuracy={score:.3f}; sensitivity={sensitivity:.3f}; specificity={specificity:.3f}")
    print("\nCounts")
    print(results.groupby(["pair", "version", "status"]).size().to_string())
    print("\nReference disagreements")
    print(merged.loc[
        (merged["status_reference"] == "Branch") != (merged["status_single"] == "Branch"),
        ["model", "domain", "pair", "version", "status_reference", "status_single"],
    ].to_string(index=False))
    print("\nFocus panels")
    print(results.loc[
        (
            (results["pair"] == "mrros → Q") & (results["model"] == "CanESM5")
            & (results["domain"] == "CW")
        )
        | (
            (results["pair"] == "P → Q") & (results["domain"] == "LI")
            & results["model"].isin(["CESM2", "GFDL-CM4", "CMCC-CM2-SR5"])
        ),
        ["model", "domain", "pair", "version", "status", "branch_type",
         "run_length", "median_peak_valley_score"],
    ].to_string(index=False))
    print(f"\nSaved {CSV_PATH}")
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
