"""Mass-weighted Candidate summary for the tracked six-scale Hexbin detector.

Branch and Two-band still require the original 3/6 scale consensus.  Weak
1--2-scale evidence is shown as Candidate only when its density-track mass is
at least one typical x-window mass.  This keeps the familiar B/T/C summary but
downweights sparse terminal tracks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from branch_benchmark import detect_tracked_multiextent_hexbin
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
STATUS_CODE = {"No branch": 0, "Candidate": 1, "Two-band": 2, "Branch": 3}


def weighted_summary(cases):
    rows = []
    for index, (key, (x, y)) in enumerate(cases.items(), 1):
        model, domain, pair, version = key
        result = detect_tracked_multiextent_hexbin(x, y, x_window_mode="equal_width")
        geometry = result.geometry or {}
        selected_quantile = geometry.get("selected_extent_quantile")
        resolutions = geometry.get("resolutions", [])

        evidence_mass = 0.0
        typical_window_masses = []
        for resolution in resolutions:
            windows = resolution.get("windows", [])
            if windows:
                typical_window_masses.append(1.0 / len(windows))
            tracking = resolution.get("tracking", {})
            mass = float(resolution.get("cluster", {}).get("mass_fraction", 0.0) or 0.0)
            if tracking.get("topological_branch", False):
                evidence_mass += mass
            elif tracking.get("continuous", False):
                evidence_mass += 0.5 * mass

        # This is derived from the actual six grids rather than hand-tuned:
        # evidence must represent at least one typical x window's point mass.
        reference_mass = float(np.median(typical_window_masses)) if typical_window_masses else 1.0
        original_status = geometry.get("track_status", "No branch")
        if original_status in {"Branch", "Two-band"}:
            weighted_status = original_status
        elif evidence_mass >= reference_mass:
            weighted_status = "Candidate"
        else:
            weighted_status = "No branch"

        rows.append({
            "model": model,
            "domain": domain,
            "pair": pair,
            "version": version,
            "n_points": len(x),
            "original_status": original_status,
            "weighted_status": weighted_status,
            "status_code": STATUS_CODE[weighted_status],
            "branch_support": int(geometry.get("branch_support", 0)),
            "band_support": int(geometry.get("band_support", 0)),
            "evidence_mass": evidence_mass,
            "reference_mass": reference_mass,
            "mass_ratio": evidence_mass / max(reference_mass, 1e-12),
            "selected_clip_quantile": selected_quantile,
        })
        if index % 30 == 0:
            print(f"  {index}/{len(cases)}")
    return pd.DataFrame(rows)


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
                    ax.text(j, i, symbols[value], ha="center", va="center", fontsize=10,
                            color="white" if value else "#4B5563", fontweight="bold")
            counts = matrix.stack().value_counts()
            ax.set_title(
                f"{pair} · {version.upper()} · B={counts.get(3, 0)}, "
                f"T={counts.get(2, 0)}, C={counts.get(1, 0)}",
                fontsize=10, fontweight="bold",
            )
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.grid(False)
    fig.suptitle(
        "Mass-weighted Tracked Hexbin: Branch (B), Two-band (T), Candidate (C)",
        fontsize=15, fontweight="bold",
    )
    path = FIGURE_DIR / "S4_branch_tracked_mass_weighted_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Loading and running weighted six-scale Hexbin summary...")
    cases = build_cases(load_runs())
    results = weighted_summary(cases)
    csv_path = OUTPUT_DIR / "S4_branch_tracked_mass_weighted_results.csv"
    results.to_csv(csv_path, index=False)
    figure_path = draw_heatmap(results)

    print("\nWeighted status counts:")
    print(results.groupby(["pair", "version", "weighted_status"]).size().unstack(fill_value=0))
    print("\nOriginal -> weighted transitions:")
    print(pd.crosstab(results["original_status"], results["weighted_status"]))
    print("\nP→Q TRIMMED retained candidates:")
    selected = results.loc[
        (results["pair"] == "P → Q") & (results["version"] == "trimmed")
        & (results["weighted_status"] != "No branch")
    ]
    print(selected[[
        "model", "domain", "original_status", "weighted_status",
        "branch_support", "band_support", "evidence_mass", "reference_mass", "mass_ratio",
    ]].to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
