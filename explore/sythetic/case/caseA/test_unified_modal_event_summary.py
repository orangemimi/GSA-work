"""Reclassify existing hard-window modal tracks by one unified event rule.

Branch means that the estimated conditional mode count contains a connected
1 <-> 2 transition.  This isolates the effect of the proposed topology rule;
it does not change or retune the upstream KDE peak detector.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from test_residual_dip import DOMAINS, MODELS, PAIRS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
SOURCE = OUTPUT_DIR / "S4_sliding_kde_modal_tracks_results.csv"
CSV_PATH = OUTPUT_DIR / "S4_unified_modal_event_results.csv"
FIGURE_PATH = OUTPUT_DIR / "figures" / "S4_unified_modal_event_heatmap.png"


def main():
    results = pd.read_csv(SOURCE)
    # A run of at least two connected double-mode windows, surrounded somewhere
    # by single-mode windows, is the direct discrete 1 <-> 2 event definition.
    results["modal_event"] = (
        (results["run_length"] >= 2)
        & (results["run_length"] < results["n_windows"])
    )
    results["status_unified"] = np.where(results["modal_event"], "Branch", "No branch")
    results.to_csv(CSV_PATH, index=False)

    cmap = ListedColormap(["#F3F4F6", "#2563EB"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.7), constrained_layout=True)
    for row, version in enumerate(("raw", "trimmed")):
        for col, (_, _, pair) in enumerate(PAIRS):
            ax = axes[row, col]
            selected = results.loc[
                (results["pair"] == pair) & (results["version"] == version)
            ].copy()
            matrix = selected.pivot(
                index="model", columns="domain", values="modal_event"
            ).reindex(index=MODELS, columns=DOMAINS).astype(int)
            ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    detected = bool(matrix.iloc[i, j])
                    ax.text(j, i, "B" if detected else "–",
                            ha="center", va="center", fontsize=11,
                            fontweight="bold", color="white" if detected else "#374151")
            ax.set_title(
                f"{pair} · {version.upper()} · B={int(matrix.to_numpy().sum())}",
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
        "Unified modal-event rule: Branch = connected M(x) 1 ↔ 2 transition",
        fontsize=15.5, fontweight="bold",
    )
    fig.savefig(FIGURE_PATH, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(results.groupby(["pair", "version", "status_unified"]).size().to_string())
    print(f"\nSaved {CSV_PATH}")
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
