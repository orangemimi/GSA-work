"""Simple location-independent Hexbin branch consensus diagnostic.

A panel is a Branch when a valid two-peak conditional density occurs at the
same x location in at least five of six Hexbin resolutions.  The local pair is
provided by the existing density guards (valley, branch mass, peak height,
separation and compactness).  No x=0 or y=0 boundary condition is used.
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
GRIDSIZES = (16, 22, 28, 34, 40, 46)
EXTENTS = (0.0, 0.0025)
REQUIRED_SUPPORT = 5


def one_extent_consensus(x, y):
    result = detect_multires_hexbin(
        x, y, gridsizes=GRIDSIZES, required_support=REQUIRED_SUPPORT,
    )
    resolutions = (result.geometry or {}).get("resolutions", [])
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_span = max(x_max - x_min, 1e-12)
    intervals_by_scale = []
    candidates = []
    for resolution in resolutions:
        intervals = []
        for window in resolution.get("windows", []):
            if not window.get("flag", False):
                continue
            left = (window["x_min"] - x_min) / x_span
            right = (window["x_max"] - x_min) / x_span
            intervals.append((left, right, window))
            candidates.extend((left, (left + right) / 2, right))
        intervals_by_scale.append(intervals)

    best_support = 0
    best_location = np.nan
    for location in candidates:
        support = sum(
            any(left - 1e-10 <= location <= right + 1e-10
                for left, right, _ in intervals)
            for intervals in intervals_by_scale
        )
        if support > best_support:
            best_support = support
            best_location = float(location)

    supporting_windows = []
    if np.isfinite(best_location):
        for resolution, intervals in zip(resolutions, intervals_by_scale):
            matches = [
                window for left, right, window in intervals
                if left - 1e-10 <= best_location <= right + 1e-10
            ]
            if matches:
                supporting_windows.append({
                    "gridsize": resolution["gridsize"],
                    "window": max(matches, key=lambda item: item.get("strength", 0.0)),
                })
    return {
        "support": int(best_support),
        "location": best_location,
        "location_x": float(x_min + best_location * x_span)
        if np.isfinite(best_location) else np.nan,
        "resolutions": resolutions,
        "supporting_windows": supporting_windows,
    }


def panel_consensus(x, y):
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
        run = one_extent_consensus(x[keep], y[keep])
        run.update({
            "quantile": quantile,
            "x": x[keep],
            "y": y[keep],
        })
        runs.append(run)
    return max(runs, key=lambda item: (item["support"], -item["quantile"]))


def draw_heatmap(results):
    palette = ListedColormap(["#F3F4F6", "#F59E0B", "#2563EB"])
    norm = BoundaryNorm([-0.5, 2.5, 4.5, 6.5], palette.N)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True)
    for row, version in enumerate(("raw", "trimmed")):
        for col, (_, _, pair) in enumerate(PAIRS):
            ax = axes[row, col]
            selected = results.loc[
                (results["pair"] == pair) & (results["version"] == version)
            ]
            matrix = selected.pivot(
                index="model", columns="domain", values="scale_support"
            ).reindex(index=MODELS, columns=DOMAINS)
            ax.imshow(matrix, cmap=palette, norm=norm, aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    support = int(matrix.iloc[i, j])
                    label = "B" if support >= REQUIRED_SUPPORT else "-"
                    color = "white" if support >= 3 else "#374151"
                    ax.text(j, i, f"{label}\n{support}/6", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=color)
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.tick_params(labelsize=8)
            ax.set_title(
                f"{pair} · {version.upper()} · B={(matrix >= REQUIRED_SUPPORT).to_numpy().sum()}",
                fontsize=11, fontweight="bold",
            )
    fig.suptitle(
        "Local Hexbin branch consensus\n"
        "B = valid two-ridge geometry at the same x location in at least 5/6 grid sizes",
        fontsize=15, fontweight="bold",
    )
    path = FIGURE_DIR / "S4_local_consensus_hexbin_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_pq_atlas(cases, objects, version):
    fig, axes = plt.subplots(5, 6, figsize=(18, 14), constrained_layout=True)
    for row, model in enumerate(MODELS):
        for col, domain in enumerate(DOMAINS):
            ax = axes[row, col]
            x, y = cases[(model, domain, "P → Q", version)]
            item = objects[(model, domain, "P → Q", version)]
            run = item["run"]
            ax.hexbin(run["x"], run["y"], gridsize=34, bins="log", mincnt=1,
                      cmap="Greys", alpha=0.80)
            if item["branch"]:
                points = sorted(
                    [(entry["window"]["x_mid"], *entry["window"]["modes"][:2])
                     for entry in run["supporting_windows"]],
                    key=lambda value: value[0],
                )
                if points:
                    xp = np.asarray([value[0] for value in points])
                    lower = np.asarray([value[1] for value in points])
                    upper = np.asarray([value[2] for value in points])
                    ax.scatter(xp, lower, s=22, color="#DC2626", zorder=5)
                    ax.scatter(xp, upper, s=22, color="#2563EB", zorder=5)
                color, label = "#B91C1C", "Branch"
            else:
                color, label = "#374151", "No branch"
            ax.set_title(
                f"{model} · {domain}\n{label} · support={item['support']}/6",
                fontsize=8.5, color=color, fontweight="bold",
            )
            if row == 4:
                ax.set_xlabel("P", fontsize=8)
            if col == 0:
                ax.set_ylabel("Q", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
    fig.suptitle(
        f"P → Q · {version.upper()} · local cross-scale Hexbin consensus",
        fontsize=16, fontweight="bold",
    )
    path = FIGURE_DIR / f"S4_local_consensus_hexbin_P_Q_{version}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    cases = build_cases(load_runs())
    objects = {}
    records = []
    for key, (x, y) in cases.items():
        model, domain, pair, version = key
        run = panel_consensus(x, y)
        branch = run["support"] >= REQUIRED_SUPPORT
        objects[key] = {
            "run": run, "support": run["support"], "branch": branch,
        }
        records.append({
            "model": model,
            "domain": domain,
            "pair": pair,
            "version": version,
            "n_points": len(x),
            "scale_support": run["support"],
            "consensus_x": run["location_x"],
            "selected_extent_quantile": run["quantile"],
            "branch": branch,
        })
    results = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "S4_local_consensus_hexbin_results.csv"
    results.to_csv(csv_path, index=False)
    heatmap_path = draw_heatmap(results)
    atlas_paths = [draw_pq_atlas(cases, objects, version) for version in ("raw", "trimmed")]

    print("Counts:")
    print(results.groupby(["pair", "version"])["branch"].sum().to_string())
    print("\nBranches:")
    print(results.loc[results["branch"], [
        "pair", "version", "model", "domain", "scale_support", "consensus_x",
    ]].sort_values(["pair", "version", "model", "domain"]).to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {heatmap_path}")
    for path in atlas_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
