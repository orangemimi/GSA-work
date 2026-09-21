"""Evaluate coordinate-swap rescue for the 2-D KDE branch detector.

The forward x -> y result remains authoritative.  Swapping x and y is used
only when the forward result is ``No branch``; any reverse detection then
becomes ``Candidate`` rather than being promoted directly to Branch/Two-band.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from test_2d_kde_conditional_tracks import classify_panel
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
CSV_PATH = OUTPUT_DIR / "S4_2d_kde_bidirectional_candidate_results.csv"
FIGURE_PATH = FIGURE_DIR / "S4_2d_kde_bidirectional_candidate_heatmap.png"


def combine(forward, reverse):
    """Keep forward classifications; use reverse only to rescue a candidate."""
    if forward["status"] != "No branch":
        return forward["status"], "forward"
    if reverse["status"] != "No branch":
        return "Candidate", f"reverse:{reverse['status']}"
    return "No branch", "none"


def plot_heatmap(results):
    codes = {"No branch": 0, "Candidate": 1, "Two-band": 2, "Branch": 3}
    letters = {"No branch": "–", "Candidate": "C", "Two-band": "T", "Branch": "B"}
    cmap = ListedColormap(["#F3F4F6", "#F59E0B", "#7C3AED", "#2563EB"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.7), constrained_layout=True)
    for row, version in enumerate(("raw", "trimmed")):
        for col, (_, _, pair) in enumerate(PAIRS):
            ax = axes[row, col]
            selected = results[
                (results["pair"] == pair) & (results["version"] == version)
            ].copy()
            selected["code"] = selected["status"].map(codes)
            matrix = selected.pivot(
                index="model", columns="domain", values="code"
            ).reindex(index=MODELS, columns=DOMAINS)
            statuses = selected.pivot(
                index="model", columns="domain", values="status"
            ).reindex(index=MODELS, columns=DOMAINS)
            sources = selected.pivot(
                index="model", columns="domain", values="source"
            ).reindex(index=MODELS, columns=DOMAINS)
            ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    status = statuses.iloc[i, j]
                    rescued = str(sources.iloc[i, j]).startswith("reverse:")
                    label = "C*" if rescued else letters[status]
                    ax.text(
                        j, i, label, ha="center", va="center", fontsize=10.5,
                        fontweight="bold",
                        color="white" if codes[status] >= 2 else "#374151",
                    )
            counts = selected["status"].value_counts()
            rescued_n = int(selected["source"].str.startswith("reverse:").sum())
            ax.set_title(
                f"{pair} · {version.upper()} · B={counts.get('Branch', 0)}, "
                f"T={counts.get('Two-band', 0)}, C={counts.get('Candidate', 0)} "
                f"(C*={rescued_n})",
                fontsize=10.2, fontweight="bold",
            )
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.set_xticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle(
        "Bidirectional 2-D KDE: forward result + reverse-only Candidate rescue\n"
        "C* = detected only after swapping x and y",
        fontsize=14.5, fontweight="bold",
    )
    fig.savefig(FIGURE_PATH, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    cases = build_cases(load_runs())
    rows = []
    for index, (key, (x, y)) in enumerate(cases.items(), start=1):
        model, domain, pair, version = key
        forward = classify_panel(x, y)
        reverse = classify_panel(y, x)
        status, source = combine(forward, reverse)
        rows.append({
            "model": model,
            "domain": domain,
            "pair": pair,
            "version": version,
            "status": status,
            "source": source,
            "forward_status": forward["status"],
            "forward_run_length": forward["run_length"],
            "forward_coverage": forward.get("x_coverage", 0.0),
            "reverse_status": reverse["status"],
            "reverse_run_length": reverse["run_length"],
            "reverse_coverage": reverse.get("x_coverage", 0.0),
            "reverse_valley_depth": reverse.get("median_valley_depth", 0.0),
            "reverse_branch_mass": reverse.get("median_branch_mass", 0.0),
            "reverse_mode_separation": reverse.get("median_mode_separation", 0.0),
        })
        if index % 15 == 0:
            print(f"Processed {index}/{len(cases)}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(CSV_PATH, index=False)
    plot_heatmap(results)

    rescued = results[results["source"].str.startswith("reverse:")]
    print("\nCombined counts")
    print(results.groupby(["pair", "version", "status"]).size().to_string())
    print("\nReverse-only rescues")
    print(rescued.groupby(["pair", "version", "reverse_status"]).size().to_string())
    print("\nRescued panels")
    print(rescued[[
        "model", "domain", "pair", "version", "source",
        "reverse_run_length", "reverse_coverage", "reverse_valley_depth",
        "reverse_branch_mass", "reverse_mode_separation",
    ]].to_string(index=False))
    print(f"\nSaved {CSV_PATH}")
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
