"""Test adjacent-scale Hexbin persistence for localized two-ridge structure.

Unlike a 5-of-6 majority over coarse and fine grids, this diagnostic asks
whether the same x location is supported by at least three consecutive grid
resolutions.  It is a branch-like candidate screen, not yet a split/merge
topology classifier.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from branch_benchmark import detect_multires_hexbin
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
GRIDSIZES = (24, 32, 40, 48, 56, 64, 72, 80)
EXTENTS = (0.0, 0.0025)
MIN_ADJACENT_RUN = 3


def longest_true_run(flags):
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return int(best)


def one_extent(x, y):
    result = detect_multires_hexbin(
        x, y, gridsizes=GRIDSIZES, required_support=len(GRIDSIZES) + 1,
    )
    resolutions = (result.geometry or {}).get("resolutions", [])
    x_min, x_max = float(np.min(x)), float(np.max(x))
    span = max(x_max - x_min, 1e-12)
    intervals_by_scale = []
    candidate_locations = []
    for resolution in resolutions:
        intervals = []
        for window in resolution.get("windows", []):
            if not window.get("flag", False):
                continue
            left = (window["x_min"] - x_min) / span
            right = (window["x_max"] - x_min) / span
            intervals.append((left, right))
            candidate_locations.extend((left, (left + right) / 2, right))
        intervals_by_scale.append(intervals)

    best_run = best_total = 0
    best_location = np.nan
    best_flags = [False] * len(GRIDSIZES)
    for location in candidate_locations:
        flags = [
            any(left - 1e-10 <= location <= right + 1e-10
                for left, right in intervals)
            for intervals in intervals_by_scale
        ]
        run = longest_true_run(flags)
        total = int(sum(flags))
        if (run, total) > (best_run, best_total):
            best_run, best_total = run, total
            best_location = float(location)
            best_flags = flags
    return {
        "adjacent_run": best_run,
        "total_support": best_total,
        "location": best_location,
        "location_x": x_min + best_location * span if np.isfinite(best_location) else np.nan,
        "scale_flags": best_flags,
    }


def panel_result(x, y):
    runs = []
    for quantile in EXTENTS:
        if quantile <= 0:
            keep = np.ones(len(x), dtype=bool)
        else:
            x_lo, x_hi = np.quantile(x, [quantile, 1 - quantile])
            y_lo, y_hi = np.quantile(y, [quantile, 1 - quantile])
            keep = (
                (x >= x_lo) & (x <= x_hi)
                & (y >= y_lo) & (y <= y_hi)
            )
        run = one_extent(x[keep], y[keep])
        run["quantile"] = quantile
        runs.append(run)
    return max(
        runs,
        key=lambda item: (item["adjacent_run"], item["total_support"], -item["quantile"]),
    )


def plot_heatmap(results):
    palette = ListedColormap(["#F3F4F6", "#FDE68A", "#F59E0B", "#2563EB"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 8.5], palette.N)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True)
    for row, version in enumerate(("raw", "trimmed")):
        for col, (_, _, pair) in enumerate(PAIRS):
            ax = axes[row, col]
            selected = results.loc[
                (results["pair"] == pair) & (results["version"] == version)
            ]
            matrix = selected.pivot(
                index="model", columns="domain", values="adjacent_run"
            ).reindex(index=MODELS, columns=DOMAINS)
            total = selected.pivot(
                index="model", columns="domain", values="total_support"
            ).reindex(index=MODELS, columns=DOMAINS)
            ax.imshow(matrix, cmap=palette, norm=norm, aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    run = int(matrix.iloc[i, j])
                    candidate = run >= MIN_ADJACENT_RUN
                    color = "white" if candidate else "#374151"
                    ax.text(j, i, f"{'C' if candidate else '-'}\nrun={run}, n={int(total.iloc[i,j])}",
                            ha="center", va="center", fontsize=7.5,
                            fontweight="bold", color=color)
            n_candidates = int((matrix >= MIN_ADJACENT_RUN).to_numpy().sum())
            ax.set_title(
                f"{pair} · {version.upper()} · candidates={n_candidates}",
                fontsize=10.5, fontweight="bold",
            )
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.tick_params(labelsize=8)
    fig.suptitle(
        "Adjacent-scale Hexbin persistence\n"
        "C = same-x two-ridge evidence at >=3 consecutive grids; grids=24..80",
        fontsize=14.5, fontweight="bold",
    )
    path = FIGURE_DIR / "S4_adjacent_scale_hexbin_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    cases = build_cases(load_runs())
    rows = []
    for key, (x, y) in cases.items():
        model, domain, pair, version = key
        result = panel_result(x, y)
        rows.append({
            "model": model,
            "domain": domain,
            "pair": pair,
            "version": version,
            "adjacent_run": result["adjacent_run"],
            "total_support": result["total_support"],
            "consensus_x": result["location_x"],
            "selected_extent_quantile": result["quantile"],
            "scale_flags": "".join("1" if flag else "0" for flag in result["scale_flags"]),
            "candidate": result["adjacent_run"] >= MIN_ADJACENT_RUN,
        })
    results = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "S4_adjacent_scale_hexbin_results.csv"
    results.to_csv(csv_path, index=False)
    figure_path = plot_heatmap(results)

    print("Candidate counts:")
    print(results.groupby(["pair", "version"])["candidate"].sum().to_string())
    print("\nHighlighted panel:")
    print(results.loc[
        (results["model"] == "CanESM5")
        & (results["domain"] == "CW")
        & (results["pair"] == "mrros → Q")
        & (results["version"] == "trimmed")
    ].to_string(index=False))
    print("\nP->Q candidates:")
    print(results.loc[
        (results["pair"] == "P → Q") & results["candidate"],
        ["version", "model", "domain", "adjacent_run", "total_support", "scale_flags"],
    ].to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
