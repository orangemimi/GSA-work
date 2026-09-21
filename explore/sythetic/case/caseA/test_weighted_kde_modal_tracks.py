"""Adaptive Gaussian-x weighted conditional KDE modal tracks.

This diagnostic replaces hard equal-count x windows with smooth, adaptive
Gaussian weights.  It is inspired by SiZer's local weighting, but estimates
conditional density modes rather than the significance of a mean derivative.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from branch_benchmark import _robust_location_scale
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS
from test_sliding_kde_modal_summary import stratified_cap


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

N_CENTERS = 61
CENTER_QUANTILES = np.linspace(0.02, 0.98, N_CENTERS)
BANDWIDTH_FACTORS = (0.75, 1.00, 1.25)
BANDWIDTH_SUPPORT = 2


def weighted_quantile(values, quantiles, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= max(float(np.sum(weights)), 1e-12)
    return np.interp(np.asarray(quantiles), cumulative, values)


def adaptive_x_weights(x, x0, target_count):
    """Choose Gaussian bandwidth so its effective n matches target_count."""
    distance = np.abs(x - x0)
    target = min(max(float(target_count), 30.0), float(len(x)))
    positive = distance[distance > 1e-12]
    if not len(positive):
        weights = np.ones(len(x), dtype=float)
        return np.ones(len(x), dtype=bool), weights, 1.0, float(len(x))

    lower = max(float(np.min(positive)) * 1e-6, 1e-12)
    upper = max(float(np.max(positive)) * 10.0, lower * 10.0)
    # Log-scale bisection is stable across highly skewed control variables.
    for _ in range(45):
        hx = float(np.sqrt(lower * upper))
        trial = np.exp(-0.5 * (distance / hx) ** 2)
        effective_n = float(np.sum(trial) ** 2 / max(np.sum(trial ** 2), 1e-12))
        if effective_n < target:
            lower = hx
        else:
            upper = hx
    hx = upper
    local = distance <= 4.0 * hx
    weights = np.exp(-0.5 * (distance[local] / hx) ** 2)
    effective_n = float(np.sum(weights) ** 2 / max(np.sum(weights ** 2), 1e-12))
    return local, weights, hx, effective_n


def weighted_mode_pair(y_scaled, weights, bandwidth_factor):
    lo, hi = weighted_quantile(y_scaled, [0.005, 0.995], weights)
    keep = (y_scaled >= lo) & (y_scaled <= hi)
    values = y_scaled[keep]
    local_weights = weights[keep]
    if len(values) < 30 or hi - lo <= 1e-10:
        return None
    try:
        kde = gaussian_kde(
            values,
            weights=local_weights,
            bw_method=lambda obj: obj.scotts_factor() * bandwidth_factor,
        )
    except (np.linalg.LinAlgError, ValueError):
        return None
    grid = np.linspace(lo, hi, 500)
    density = kde(grid)
    peaks = list(find_peaks(
        density,
        prominence=max(0.025 * float(density.max()), 1e-12),
        distance=10,
    )[0])
    if density[0] > density[1] and density[0] >= 0.05 * density.max():
        peaks.insert(0, 0)
    if density[-1] > density[-2] and density[-1] >= 0.05 * density.max():
        peaks.append(len(grid) - 1)
    peaks = sorted(set(peaks))
    candidates = []
    total_weight = max(float(np.sum(local_weights)), 1e-12)
    for left_index in range(len(peaks) - 1):
        for right_index in range(left_index + 1, len(peaks)):
            left, right = peaks[left_index], peaks[right_index]
            valley = left + int(np.argmin(density[left:right + 1]))
            low_peak = min(float(density[left]), float(density[right]))
            valley_depth = 1.0 - float(density[valley]) / max(low_peak, 1e-12)
            cut = grid[valley]
            left_weight = float(np.sum(local_weights[values <= cut])) / total_weight
            right_weight = float(np.sum(local_weights[values > cut])) / total_weight
            min_weight = min(left_weight, right_weight)
            separation = float(grid[right] - grid[left])
            valid = bool(
                min_weight >= 0.08
                and valley_depth >= 0.15
                and separation >= 0.80
            )
            candidates.append({
                "valid": valid,
                "modes": (float(grid[left]), float(grid[right])),
                "valley_depth": valley_depth,
                "min_weight": min_weight,
                "separation": separation,
                "strength": low_peak,
            })
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["valid"], item["strength"]))


def connected_runs(points, y_range):
    jump_limit = 0.20 * max(float(y_range), 1e-12)
    runs, current = [], []
    previous_index = None
    previous_modes = None
    for index, point in enumerate(points):
        modes = np.asarray(point.get("modes", (np.nan, np.nan)), dtype=float)
        usable = bool(point.get("flag")) and np.all(np.isfinite(modes[:2]))
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
            current.append(point)
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
    x, y = stratified_cap(x, y)
    _, y_scale = _robust_location_scale(y)
    y_center = float(np.median(y))
    y_scaled = (y - y_center) / y_scale
    y_lo, y_hi = np.quantile(y, [0.005, 0.995])
    target_count = min(len(x), max(120, int(round(0.06 * len(x)))))

    points = []
    for q, x0 in zip(CENTER_QUANTILES, np.quantile(x, CENTER_QUANTILES)):
        local, weights, hx, effective_n = adaptive_x_weights(x, x0, target_count)
        candidates = [
            weighted_mode_pair(y_scaled[local], weights, factor)
            for factor in BANDWIDTH_FACTORS
        ]
        valid = [item for item in candidates if item is not None and item["valid"]]
        flag = len(valid) >= BANDWIDTH_SUPPORT
        if valid:
            low = float(np.median([item["modes"][0] for item in valid]) * y_scale + y_center)
            high = float(np.median([item["modes"][1] for item in valid]) * y_scale + y_center)
            valley = float(np.median([item["valley_depth"] for item in valid]))
            min_weight = float(np.median([item["min_weight"] for item in valid]))
        else:
            low = high = valley = min_weight = np.nan
        points.append({
            "q": float(q),
            "x_mid": float(x0),
            "flag": flag,
            "modes": (low, high),
            "valley_depth": valley,
            "min_weight": min_weight,
            "bandwidth_support": len(valid),
            "hx": hx,
            "effective_n": effective_n,
        })

    runs = connected_runs(points, y_hi - y_lo)
    run = max(runs, key=lambda items: items[-1]["q"] - items[0]["q"], default=[])
    if run:
        # Include one local kernel footprint; evaluation-grid density must not
        # determine the inferred x support of a modal track.
        kernel_mass = float(target_count / max(len(x), 1))
        mass_coverage = min(1.0, float(run[-1]["q"] - run[0]["q"] + kernel_mass))
        separation = np.asarray([item["modes"][1] - item["modes"][0] for item in run])
        edge = min(2, len(run))
        relative_change = float(
            abs(np.mean(separation[-edge:]) - np.mean(separation[:edge]))
            / max(float(np.max(separation)), 1e-12)
        )
        median_valley = float(np.nanmedian([item["valley_depth"] for item in run]))
        median_weight = float(np.nanmedian([item["min_weight"] for item in run]))
        x_start, x_end = float(run[0]["x_mid"]), float(run[-1]["x_mid"])
    else:
        mass_coverage = relative_change = 0.0
        median_valley = median_weight = x_start = x_end = np.nan

    extended_fork = bool(
        len(run) >= 3
        and mass_coverage >= 0.10
        and relative_change >= 0.25
    )
    local_side_branch = bool(
        len(run) >= 2
        and mass_coverage >= 0.04
        and median_valley >= 0.30
        and median_weight <= 0.25
    )
    if extended_fork:
        status, branch_type = "Branch", "extended fork"
    elif local_side_branch:
        status, branch_type = "Branch", "local side branch"
    elif mass_coverage >= 0.10:
        status, branch_type = "Two-band", "stable/ambiguous two-band"
    elif mass_coverage >= 0.04:
        status, branch_type = "Candidate", "weak local candidate"
    else:
        status, branch_type = "No branch", "none"

    return {
        "status": status,
        "branch_type": branch_type,
        "n_points": len(x),
        "n_centers": len(points),
        "n_double_mode_centers": int(sum(point["flag"] for point in points)),
        "track_points": len(run),
        "x_start": x_start,
        "x_end": x_end,
        "mass_coverage": mass_coverage,
        "relative_separation_change": relative_change,
        "median_valley_depth": median_valley,
        "median_min_branch_weight": median_weight,
        "median_effective_n": float(np.median([item["effective_n"] for item in points])),
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
        "Adaptive Gaussian-x Weighted KDE Modal Tracks",
        fontsize=15.5, fontweight="bold",
    )
    path = FIGURE_DIR / "S4_weighted_kde_modal_tracks_heatmap.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    cases = build_cases(load_runs())
    rows = []
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
            print(f"Processed {index}/{len(cases)}", flush=True)
    results = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "S4_weighted_kde_modal_tracks_results.csv"
    results.to_csv(csv_path, index=False)
    figure_path = plot_heatmap(results)
    print("\nCounts")
    print(results.groupby(["pair", "version", "status"]).size().to_string())
    print("\nFocus panels")
    print(results.loc[
        (
            (results["pair"] == "mrros → Q")
            & (results["model"] == "CanESM5") & (results["domain"] == "CW")
        )
        | (
            (results["pair"] == "P → Q") & (results["domain"] == "LI")
            & results["model"].isin(["CESM2", "GFDL-CM4", "CMCC-CM2-SR5"])
        ),
        ["model", "domain", "pair", "version", "status", "branch_type",
         "track_points", "mass_coverage", "median_valley_depth",
         "median_min_branch_weight"],
    ].to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
