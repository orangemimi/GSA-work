"""Detect and draw the boundary-branch subtype in P -> Q TRIMMED panels.

A boundary branch has a dominant ridge at the lower Q boundary and a positive
upper arm.  The panel-level cutoff is selected from the largest gap in lower-
boundary mass across the 30 P -> Q panels, rather than hand-tuned labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"


def boundary_mass(y):
    y = np.asarray(y, dtype=float)
    lower, upper = float(np.min(y)), float(np.max(y))
    width = max(upper - lower, 1e-12)
    cutoff = lower + 0.01 * width
    return float(np.mean(y <= cutoff)), cutoff, lower, upper


def largest_gap_threshold(values):
    ordered = np.sort(np.asarray(values, dtype=float))
    gaps = np.diff(ordered)
    index = int(np.argmax(gaps))
    return float((ordered[index] + ordered[index + 1]) / 2), float(gaps[index])


def clean_boundary_tracks(x, y, boundary_cutoff, y_lower, y_upper):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_width = max(y_upper - y_lower, 1e-12)
    lower = y <= boundary_cutoff
    upper = y > y_lower + 0.05 * y_width
    if lower.sum() < 20 or upper.sum() < 20:
        return None

    upper_start, upper_end = np.quantile(x[upper], [0.05, 0.95])
    lower_start, lower_end = np.quantile(x[lower], [0.05, 0.95])
    upper_core = upper & (x >= upper_start) & (x <= upper_end)
    order = np.argsort(x[upper_core], kind="mergesort")
    xu = x[upper_core][order]
    yu = y[upper_core][order]
    if len(xu) < 30:
        return None
    groups = np.array_split(np.arange(len(xu)), min(9, max(4, len(xu) // 80)))
    knot_x = np.asarray([np.median(xu[group]) for group in groups])
    knot_y = np.asarray([np.median(yu[group]) for group in groups])
    upper_track = lowess(knot_y, knot_x, frac=0.55, it=2, return_sorted=True)

    lower_core = lower & (x >= lower_start) & (x <= lower_end)
    lower_y = float(np.median(y[lower_core])) if lower_core.any() else float(np.median(y[lower]))
    lower_track_x = np.asarray([lower_start, lower_end])
    lower_track_y = np.asarray([lower_y, lower_y])
    return {
        "upper_x": upper_track[:, 0],
        "upper_y": upper_track[:, 1],
        "lower_x": lower_track_x,
        "lower_y": lower_track_y,
        "x_start": min(lower_start, upper_start),
        "x_end": max(lower_end, upper_end),
        "upper_fraction": float(np.mean(upper)),
    }


def main():
    cases = build_cases(load_runs())
    records = []
    for model in MODELS:
        for domain in DOMAINS:
            x, y = cases[(model, domain, "P → Q", "trimmed")]
            mass, cutoff, lower, upper = boundary_mass(y)
            records.append({
                "model": model, "domain": domain, "n_points": len(x),
                "boundary_mass": mass, "boundary_cutoff": cutoff,
                "y_lower": lower, "y_upper": upper,
            })
    results = pd.DataFrame(records)
    threshold, gap = largest_gap_threshold(results["boundary_mass"])
    results["boundary_threshold"] = threshold
    results["largest_gap"] = gap
    results["boundary_branch"] = results["boundary_mass"] > threshold

    fig, axes = plt.subplots(5, 6, figsize=(18, 14), constrained_layout=True)
    for row, model in enumerate(MODELS):
        for col, domain in enumerate(DOMAINS):
            ax = axes[row, col]
            x, y = cases[(model, domain, "P → Q", "trimmed")]
            item = results.loc[(results["model"] == model) & (results["domain"] == domain)].iloc[0]
            ax.hexbin(x, y, gridsize=34, bins="log", mincnt=1, cmap="Greys", alpha=0.80)
            if item["boundary_branch"]:
                tracks = clean_boundary_tracks(
                    x, y, item["boundary_cutoff"], item["y_lower"], item["y_upper"],
                )
                if tracks is not None:
                    ax.plot(tracks["lower_x"], tracks["lower_y"], "o-", color="#DC2626", lw=2.0, ms=4)
                    ax.plot(tracks["upper_x"], tracks["upper_y"], "o-", color="#2563EB", lw=2.0, ms=4)
                status, color = "Boundary branch", "#B91C1C"
            else:
                status, color = "No boundary branch", "#374151"
            ax.set_title(
                f"{model} · {domain}\n{status} · boundary mass={item['boundary_mass']:.1%}",
                fontsize=8.5, color=color, fontweight="bold",
            )
            if row == 4:
                ax.set_xlabel("P", fontsize=8)
            if col == 0:
                ax.set_ylabel("Q", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
    fig.suptitle(
        f"P → Q · TRIMMED · boundary-branch detector\n"
        f"data-derived boundary-mass cutoff={threshold:.1%} (largest gap={gap:.1%})",
        fontsize=16, fontweight="bold",
    )
    figure_path = FIGURE_DIR / "S4_boundary_branch_P_Q_trimmed.png"
    csv_path = OUTPUT_DIR / "S4_boundary_branch_P_Q_trimmed.csv"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    results.to_csv(csv_path, index=False)

    print(results.loc[results["boundary_branch"], [
        "model", "domain", "boundary_mass", "boundary_threshold",
    ]].to_string(index=False))
    print(f"Saved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
