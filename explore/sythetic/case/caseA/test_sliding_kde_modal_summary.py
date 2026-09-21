"""Summarise simple sliding-window conditional KDE modal tracks.

This is a diagnostic script.  It does not modify the production S4 notebook.
For every panel it asks whether two conditional y modes persist and connect
across neighbouring, equal-count x windows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from branch_benchmark import detect_modal_kde
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MAX_POINTS = 6000
MAX_TRACK_JUMP_FRACTION = 0.20


def stratified_cap(x, y, max_points=MAX_POINTS):
    """Cap runtime while retaining the full ordered x distribution."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    if len(x) <= max_points:
        return x, y
    positions = np.rint(np.linspace(0, len(x) - 1, max_points)).astype(int)
    return x[positions], y[positions]


def connected_flag_runs(windows, y):
    """Split double-mode windows at gaps or implausibly large mode jumps."""
    y_lo, y_hi = np.quantile(y, [0.005, 0.995])
    jump_limit = MAX_TRACK_JUMP_FRACTION * max(float(y_hi - y_lo), 1e-12)
    runs, current = [], []
    previous_index = None
    previous_modes = None
    for index, window in enumerate(windows):
        modes = np.asarray(window.get("modes", (np.nan, np.nan)), dtype=float)
        usable = bool(window.get("flag")) and len(modes) >= 2 and np.all(np.isfinite(modes[:2]))
        connected = False
        if usable and previous_index is not None:
            connected = bool(
                index == previous_index + 1
                and np.max(np.abs(modes[:2] - previous_modes[:2])) <= jump_limit
            )
        if usable:
            if current and not connected:
                runs.append(current)
                current = []
            current.append(window)
            previous_index = index
            previous_modes = modes
        else:
            if current:
                runs.append(current)
                current = []
            previous_index = None
            previous_modes = None
    if current:
        runs.append(current)
    return runs


def classify_panel(x, y):
    x_use, y_use = stratified_cap(x, y)
    result = detect_modal_kde(x_use, y_use)
    windows = (result.geometry or {}).get("windows", [])
    runs = connected_flag_runs(windows, y_use)
    run = max(runs, key=len, default=[])
    run_length = len(run)

    rho = 0.0
    relative_change = 0.0
    if run_length >= 2:
        x_mid = np.asarray([item["x_mid"] for item in run])
        separation = np.asarray([item["modes"][1] - item["modes"][0] for item in run])
        relative_change = float(
            abs(np.mean(separation[-min(2, run_length):]) - np.mean(separation[:min(2, run_length)]))
            / max(float(np.max(separation)), 1e-12)
        )
        if run_length >= 3 and np.ptp(separation) > 1e-12:
            rho = float(spearmanr(x_mid, separation).statistic)
            if not np.isfinite(rho):
                rho = 0.0

    if run:
        x_start = float(run[0]["x_min"])
        x_end = float(run[-1]["x_max"])
        median_valley = float(np.nanmedian([item["valley_depth"] for item in run]))
        median_weight = float(np.nanmedian([item["min_weight"] for item in run]))
    else:
        x_start = x_end = median_valley = median_weight = np.nan
    x_range = max(float(np.max(x_use) - np.min(x_use)), 1e-12)
    coverage = float((x_end - x_start) / x_range) if run else 0.0

    # Two branch morphologies need different persistence evidence:
    # 1) an extended fork, whose separation changes over a meaningful x span;
    # 2) a short side branch, visible as a deep, lower-mass secondary mode.
    extended_fork = bool(
        run_length >= 3
        and relative_change >= 0.25
        and coverage >= 0.10
    )
    local_side_branch = bool(
        run_length == 2
        and median_valley >= 0.30
        and median_weight <= 0.25
    )
    if extended_fork:
        status = "Branch"
        branch_type = "extended fork"
    elif local_side_branch:
        status = "Branch"
        branch_type = "local side branch"
    elif run_length >= 3:
        status = "Two-band"
        branch_type = "stable/ambiguous two-band"
    elif run_length == 2:
        status = "Candidate"
        branch_type = "weak local candidate"
    else:
        status = "No branch"
        branch_type = "none"
    return {
        "status": status,
        "branch_type": branch_type,
        "n_points": len(x),
        "n_points_used": len(x_use),
        "n_windows": len(windows),
        "n_double_mode_windows": int(sum(bool(item.get("flag")) for item in windows)),
        "run_length": run_length,
        "x_start": x_start,
        "x_end": x_end,
        "x_coverage": coverage,
        "median_valley_depth": median_valley,
        "median_min_branch_weight": median_weight,
        "separation_rho": rho,
        "relative_separation_change": relative_change,
    }


def plot_heatmap(results):
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
                    color = "white" if codes[status] >= 2 else "#374151"
                    ax.text(j, i, letters[status], ha="center", va="center",
                            fontsize=11, fontweight="bold", color=color)
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
        "Sliding-window KDE Modal Tracks: Branch (B), Two-band (T), Candidate (C)",
        fontsize=15.5, fontweight="bold",
    )
    path = FIGURE_DIR / "S4_sliding_kde_modal_tracks_heatmap.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    cases = build_cases(load_runs())
    rows = []
    total = len(cases)
    for index, (key, (x, y)) in enumerate(cases.items(), start=1):
        model, domain, pair, version = key
        rows.append({
            "model": model,
            "domain": domain,
            "pair": pair,
            "version": version,
            **classify_panel(x, y),
        })
        if index % 15 == 0:
            print(f"Processed {index}/{total}", flush=True)
    results = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "S4_sliding_kde_modal_tracks_results.csv"
    results.to_csv(csv_path, index=False)
    figure_path = plot_heatmap(results)
    print("\nCounts:")
    print(results.groupby(["pair", "version", "status"]).size().to_string())
    print("\nP -> Q non-null:")
    print(results.loc[
        (results["pair"] == "P → Q") & (results["status"] != "No branch"),
        ["model", "domain", "version", "status", "run_length", "x_coverage",
         "separation_rho", "relative_separation_change"],
    ].to_string(index=False))
    print("\nHighlighted CanESM5 / CW / mrros -> Q / trimmed:")
    print(results.loc[
        (results["model"] == "CanESM5") & (results["domain"] == "CW")
        & (results["pair"] == "mrros → Q") & (results["version"] == "trimmed")
    ].to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
