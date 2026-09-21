"""A/B test: equal-width versus equal-frequency x windows for tracked Hexbin.

The production detector keeps its existing equal-width default.  This script
calls the same detector with ``x_window_mode='equal_frequency'`` and writes
separate diagnostic outputs.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from branch_benchmark import detect_tracked_multiextent_hexbin, plot_branch_result
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
GRIDSIZES = (16, 22, 28, 34, 40, 46)
EXTENT_QUANTILES = (0.0, 0.0025)
REQUIRED_SUPPORT = 3
STATUS_CODE = {"No branch": 0, "Candidate": 1, "Two-band": 2, "Branch": 3}


def run_equal_frequency(cases):
    objects = {}
    rows = []
    started = time.perf_counter()
    for index, (key, (x, y)) in enumerate(cases.items(), 1):
        model, domain, pair, version = key
        result = detect_tracked_multiextent_hexbin(
            x,
            y,
            extent_quantiles=EXTENT_QUANTILES,
            gridsizes=GRIDSIZES,
            required_support=REQUIRED_SUPPORT,
            x_window_mode="equal_frequency",
        )
        geometry = result.geometry or {}
        best_track = geometry.get("best_track") or {}
        status = geometry.get("track_status", "No branch")
        objects[key] = result
        rows.append({
            "model": model,
            "domain": domain,
            "pair": pair,
            "version": version,
            "n_points": len(x),
            "track_status": status,
            "status_code": STATUS_CODE[status],
            "branch_support": int(geometry.get("branch_support", 0)),
            "band_support": int(geometry.get("band_support", 0)),
            "selected_clip_quantile": geometry.get("selected_extent_quantile", np.nan),
            "track_windows": int(best_track.get("n_windows", 0)),
            "relative_separation_change": best_track.get("relative_separation_change", np.nan),
            "separation_correlation": best_track.get("separation_correlation", np.nan),
            **result.to_dict(),
        })
        if index % 30 == 0:
            print(f"  {index}/{len(cases)} panels; {time.perf_counter() - started:.1f}s")
    return objects, pd.DataFrame(rows)


def draw_atlas(cases, objects, results, pair="P → Q", version="trimmed"):
    fig, axes = plt.subplots(5, 6, figsize=(18, 14), constrained_layout=True)
    colors = {
        "Branch": "#B91C1C", "Two-band": "#6D28D9",
        "Candidate": "#B45309", "No branch": "#374151",
    }
    for row, model in enumerate(MODELS):
        for col, domain in enumerate(DOMAINS):
            ax = axes[row, col]
            key = (model, domain, pair, version)
            x, y = cases[key]
            result = objects[key]
            item = results.loc[
                (results["model"] == model) & (results["domain"] == domain)
                & (results["pair"] == pair) & (results["version"] == version)
            ].iloc[0]
            plot_branch_result(ax, x, y, result)
            status = item["track_status"]
            ax.set_title(
                f"{model} · {domain}\n{status} · branch {int(item['branch_support'])}/6 · tracks {int(item['band_support'])}/6",
                fontsize=8.5, color=colors[status], fontweight="bold",
            )
            if row == 4:
                ax.set_xlabel(pair.split(" → ")[0], fontsize=8)
            if col == 0:
                ax.set_ylabel(pair.split(" → ")[1], fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{pair} · {version.upper()} · tracked Hexbin with equal-frequency x windows",
        fontsize=16, fontweight="bold",
    )
    path = FIGURE_DIR / f"S4_branch_tracked_equal_frequency_{pair.replace(' → ', '_')}_{version}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_heatmap(results):
    cmap = ListedColormap(["#F3F4F6", "#F59E0B", "#7C3AED", "#2563EB"])
    symbols = {0: "–", 1: "C", 2: "T", 3: "B"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    for row, version in enumerate(("raw", "trimmed")):
        for col, (_, _, pair) in enumerate(PAIRS):
            ax = axes[row, col]
            selected = results.loc[(results["pair"] == pair) & (results["version"] == version)]
            matrix = selected.pivot(index="model", columns="domain", values="status_code").reindex(
                index=MODELS, columns=DOMAINS
            ).fillna(0).astype(int)
            ax.imshow(matrix, vmin=0, vmax=3, cmap=cmap, aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    value = int(matrix.iloc[i, j])
                    ax.text(j, i, symbols[value], ha="center", va="center",
                            color="white" if value else "#4B5563", fontweight="bold")
            counts = matrix.stack().value_counts()
            ax.set_title(
                f"{pair} · {version.upper()} · B={counts.get(3, 0)}, T={counts.get(2, 0)}, C={counts.get(1, 0)}",
                fontsize=10, fontweight="bold",
            )
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.grid(False)
    fig.suptitle("Tracked Hexbin · equal-frequency x windows", fontsize=15, fontweight="bold")
    path = FIGURE_DIR / "S4_branch_tracked_equal_frequency_all_cases_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def compare_with_equal_width(equal_frequency):
    baseline_path = OUTPUT_DIR / "S4_branch_tracked_all_cases_results.csv"
    baseline = pd.read_csv(baseline_path)[
        ["model", "domain", "pair", "version", "track_status", "branch_support", "band_support"]
    ].rename(columns={
        "track_status": "equal_width_status",
        "branch_support": "equal_width_branch_support",
        "band_support": "equal_width_band_support",
    })
    comparison = baseline.merge(
        equal_frequency[
            ["model", "domain", "pair", "version", "track_status", "branch_support", "band_support"]
        ].rename(columns={
            "track_status": "equal_frequency_status",
            "branch_support": "equal_frequency_branch_support",
            "band_support": "equal_frequency_band_support",
        }),
        on=["model", "domain", "pair", "version"], how="inner", validate="one_to_one",
    )
    return comparison


def main():
    print("Loading cases...")
    cases = build_cases(load_runs())
    print(f"Running equal-frequency detector on {len(cases)} panels...")
    objects, results = run_equal_frequency(cases)
    comparison = compare_with_equal_width(results)

    results_path = OUTPUT_DIR / "S4_branch_tracked_equal_frequency_all_cases_results.csv"
    comparison_path = OUTPUT_DIR / "S4_branch_tracked_equal_width_vs_frequency.csv"
    results.to_csv(results_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    atlas_path = draw_atlas(cases, objects, results)
    heatmap_path = draw_heatmap(results)

    print("\nEqual-frequency status counts:")
    print(results.groupby(["pair", "version", "track_status"]).size().unstack(fill_value=0))
    print("\nEqual-width -> equal-frequency transitions:")
    print(pd.crosstab(comparison["equal_width_status"], comparison["equal_frequency_status"]))
    print("\nP→Q TRIMMED changed panels:")
    changed = comparison.loc[
        (comparison["pair"] == "P → Q") & (comparison["version"] == "trimmed")
        & (comparison["equal_width_status"] != comparison["equal_frequency_status"])
    ]
    print(changed.to_string(index=False))
    print(f"\nSaved {results_path}")
    print(f"Saved {comparison_path}")
    print(f"Saved {atlas_path}")
    print(f"Saved {heatmap_path}")


if __name__ == "__main__":
    main()
