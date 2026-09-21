"""Apply the P->Q boundary-branch score unchanged to all S4 panels.

This is a diagnostic script.  The score cutoff is learned once from the
largest gap among the 30 P->Q TRIMMED panels and is then frozen for every
pair/version, so groups without a boundary branch are not forced to contain
one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from test_boundary_branch_pq import boundary_mass, largest_gap_threshold
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"


def main():
    cases = build_cases(load_runs())
    rows = []
    for (model, domain, pair, version), (x, y) in cases.items():
        mass, cutoff, lower, upper = boundary_mass(y)
        width = max(upper - lower, 1e-12)
        upper_count = int(np.sum(y > lower + 0.05 * width))
        rows.append({
            "model": model,
            "domain": domain,
            "pair": pair,
            "version": version,
            "n_points": len(x),
            "boundary_mass": mass,
            "boundary_cutoff": cutoff,
            "y_lower": lower,
            "y_upper": upper,
            "upper_count": upper_count,
            "upper_fraction": upper_count / len(y),
        })

    result = pd.DataFrame(rows)
    reference = result.loc[
        (result["pair"] == "P → Q") & (result["version"] == "trimmed"),
        "boundary_mass",
    ]
    threshold, reference_gap = largest_gap_threshold(reference)
    result["boundary_threshold"] = threshold
    result["reference_gap"] = reference_gap
    result["boundary_branch"] = (
        (result["boundary_mass"] > threshold)
        & (result["upper_count"] >= 20)
    )

    csv_path = OUTPUT_DIR / "S4_boundary_branch_all_cases.csv"
    result.to_csv(csv_path, index=False)

    pair_labels = [pair for _, _, pair in PAIRS]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True)
    for row, version in enumerate(("raw", "trimmed")):
        for col, pair in enumerate(pair_labels):
            ax = axes[row, col]
            subset = result.loc[
                (result["pair"] == pair) & (result["version"] == version)
            ]
            matrix = subset.pivot(
                index="model", columns="domain", values="boundary_mass"
            ).reindex(index=MODELS, columns=DOMAINS)
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
            labels = subset.pivot(
                index="model", columns="domain", values="boundary_branch"
            ).reindex(index=MODELS, columns=DOMAINS)
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    value = matrix.iloc[i, j]
                    mark = "B" if bool(labels.iloc[i, j]) else "-"
                    color = "white" if value >= 0.55 else "#374151"
                    ax.text(j, i, f"{mark}\n{value:.0%}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=color)
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.set_title(
                f"{pair} · {version.upper()} · B={int(labels.to_numpy().sum())}",
                fontsize=11, fontweight="bold",
            )
            ax.tick_params(labelsize=8)
    fig.colorbar(image, ax=axes, shrink=0.78, label="mass in lowest 1% of Q range")
    fig.suptitle(
        "Boundary-branch score applied unchanged to all panels\n"
        f"B if boundary mass > {threshold:.1%}; cutoff learned once from P→Q TRIMMED",
        fontsize=15, fontweight="bold",
    )
    figure_path = FIGURE_DIR / "S4_boundary_branch_all_cases_heatmap.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    positives = result.loc[result["boundary_branch"], [
        "pair", "version", "model", "domain", "n_points",
        "boundary_mass", "upper_fraction",
    ]].sort_values(["pair", "version", "model", "domain"])
    counts = result.groupby(["pair", "version"])["boundary_branch"].sum()
    print(f"Frozen threshold: {threshold:.6f}; reference gap: {reference_gap:.6f}")
    print("\nCounts:")
    print(counts.to_string())
    print("\nPositive panels:")
    print(positives.to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
