"""B/T/C heatmap with confidence-weighted Candidate colour.

Structural labels are not deleted for low global mass.  Track mass controls
only the orange saturation of Candidate cells.  A Candidate needs at least one
split/merge vote, or repeated local modal pairs at four or more of the six
grids when no continuous track survives.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from branch_benchmark import detect_tracked_multiextent_hexbin
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
STATUS_CODE = {"No branch": 0, "Candidate": 1, "Two-band": 2, "Branch": 3}
BASE_RGB = {
    "No branch": np.array([243, 244, 246]) / 255,
    "Candidate": np.array([245, 158, 11]) / 255,
    "Two-band": np.array([124, 58, 237]) / 255,
    "Branch": np.array([37, 99, 235]) / 255,
}


def collect_results(cases):
    rows = []
    for index, (key, (x, y)) in enumerate(cases.items(), 1):
        model, domain, pair, version = key
        result = detect_tracked_multiextent_hexbin(x, y, x_window_mode="equal_width")
        geometry = result.geometry or {}
        resolutions = geometry.get("resolutions", [])

        tracked_mass = 0.0
        repeated_local_mass = 0.0
        repeated_local_support = 0
        typical_window_masses = []
        for resolution in resolutions:
            windows = resolution.get("windows", [])
            if windows:
                typical_window_masses.append(1.0 / len(windows))
            cluster = resolution.get("cluster", {})
            cluster_mass = float(cluster.get("mass_fraction", 0.0) or 0.0)
            tracking = resolution.get("tracking", {})
            if tracking.get("topological_branch", False):
                tracked_mass += cluster_mass
            elif tracking.get("continuous", False):
                tracked_mass += 0.5 * cluster_mass
            if int(cluster.get("count", 0) or 0) >= 2:
                repeated_local_support += 1
                repeated_local_mass += cluster_mass

        reference_mass = float(np.median(typical_window_masses)) if typical_window_masses else 1.0
        original_status = geometry.get("track_status", "No branch")
        branch_support = int(geometry.get("branch_support", 0))
        band_support = int(geometry.get("band_support", 0))
        if original_status in {"Branch", "Two-band"}:
            display_status = original_status
            confidence_mass = tracked_mass
            candidate_source = "tracked"
        elif branch_support >= 1:
            # Candidate means at least one scale saw actual split/merge
            # topology; a merely broken or wide second band is not enough.
            display_status = "Candidate"
            confidence_mass = tracked_mass
            candidate_source = "branch-vote"
        elif repeated_local_support >= 4:
            # A fallback for localized structures that repeatedly produce two
            # modes but fail the cross-window connection at every scale.
            display_status = "Candidate"
            confidence_mass = repeated_local_mass
            candidate_source = "repeated-local"
        else:
            display_status = "No branch"
            confidence_mass = 0.0
            candidate_source = "none"

        confidence = float(np.clip(confidence_mass / max(reference_mass, 1e-12), 0.0, 1.0))
        rows.append({
            "model": model,
            "domain": domain,
            "pair": pair,
            "version": version,
            "n_points": len(x),
            "original_status": original_status,
            "display_status": display_status,
            "status_code": STATUS_CODE[display_status],
            "branch_support": branch_support,
            "band_support": band_support,
            "repeated_local_support": repeated_local_support,
            "tracked_mass": tracked_mass,
            "repeated_local_mass": repeated_local_mass,
            "reference_mass": reference_mass,
            "confidence": confidence,
            "candidate_source": candidate_source,
        })
        if index % 30 == 0:
            print(f"  {index}/{len(cases)}")
    return pd.DataFrame(rows)


def blend_with_white(rgb, strength):
    return 1.0 - strength * (1.0 - rgb)


def draw_heatmap(results):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    symbols = {"No branch": "–", "Candidate": "C", "Two-band": "T", "Branch": "B"}
    for row, version in enumerate(("raw", "trimmed")):
        for col, (_, _, pair) in enumerate(PAIRS):
            ax = axes[row, col]
            selected = results.loc[(results["pair"] == pair) & (results["version"] == version)].copy()
            selected = selected.set_index(["model", "domain"])
            rgba = np.ones((len(MODELS), len(DOMAINS), 4), dtype=float)
            labels = np.empty((len(MODELS), len(DOMAINS)), dtype=object)
            statuses = np.empty((len(MODELS), len(DOMAINS)), dtype=object)
            for i, model in enumerate(MODELS):
                for j, domain in enumerate(DOMAINS):
                    item = selected.loc[(model, domain)]
                    status = item["display_status"]
                    statuses[i, j] = status
                    labels[i, j] = symbols[status]
                    if status == "Candidate":
                        # Keep weak candidates visible while making their lower
                        # confidence explicit through a paler orange.
                        strength = 0.35 + 0.65 * float(item["confidence"])
                        rgba[i, j, :3] = blend_with_white(BASE_RGB[status], strength)
                    else:
                        rgba[i, j, :3] = BASE_RGB[status]
                    rgba[i, j, 3] = 1.0
            ax.imshow(rgba, aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    status = statuses[i, j]
                    ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=10,
                            color="white" if status in {"Branch", "Two-band", "Candidate"} else "#4B5563",
                            fontweight="bold")
            counts = pd.Series(statuses.ravel()).value_counts()
            ax.set_title(
                f"{pair} · {version.upper()} · B={counts.get('Branch', 0)}, "
                f"T={counts.get('Two-band', 0)}, C={counts.get('Candidate', 0)}",
                fontsize=10, fontweight="bold",
            )
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.grid(False)
    fig.suptitle(
        "Tracked Hexbin: Branch (B), Two-band (T), Candidate (C)\n"
        "Candidate colour intensity = density-track confidence",
        fontsize=15, fontweight="bold",
    )
    path = FIGURE_DIR / "S4_branch_tracked_confidence_weighted_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Running confidence-weighted six-scale Hexbin summary...")
    cases = build_cases(load_runs())
    results = collect_results(cases)
    csv_path = OUTPUT_DIR / "S4_branch_tracked_confidence_weighted_results.csv"
    results.to_csv(csv_path, index=False)
    figure_path = draw_heatmap(results)

    print("\nDisplay status counts:")
    print(results.groupby(["pair", "version", "display_status"]).size().unstack(fill_value=0))
    print("\nOriginal -> display transitions:")
    print(pd.crosstab(results["original_status"], results["display_status"]))
    print("\nRequested cases:")
    requested = [
        ("P → Q", "raw", "GFDL-CM4", "LI"),
        ("P → Q", "raw", "CMCC-CM2-SR5", "LI"),
        ("P → Q", "trimmed", "GFDL-CM4", "LI"),
        ("mrros → Q", "trimmed", "CESM2", "LI"),
        ("mrros → Q", "trimmed", "CMCC-CM2-SR5", "WW"),
    ]
    mask = np.zeros(len(results), dtype=bool)
    for pair, version, model, domain in requested:
        mask |= ((results["pair"] == pair) & (results["version"] == version)
                 & (results["model"] == model) & (results["domain"] == domain))
    print(results.loc[mask, [
        "model", "domain", "pair", "version", "original_status", "display_status",
        "branch_support", "band_support", "repeated_local_support", "confidence", "candidate_source",
    ]].to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
