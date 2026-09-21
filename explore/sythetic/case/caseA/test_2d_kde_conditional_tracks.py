"""Global 2-D KDE conditional-density branch detector.

The full panel is first represented by five globally smoothed 2-D densities in
rank-transformed x and physical y.  Equal-frequency x columns are then
normalised to estimate p(y | x-bin).  Each bandwidth independently detects and
tracks lower-mode/valley/upper-mode triplets and applies the same fork geometry
and 3/5 bandwidth vote used by the sliding-window detector.

Only Branch, Candidate, and No branch are returned.  Persistent multimodality
without opening/closing is retained as a diagnostic flag, not a fourth class.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS
from test_sliding_kde_modal_summary import stratified_cap


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
CSV_PATH = OUTPUT_DIR / "S4_2d_kde_unified_branch_rules_results.csv"
FIGURE_PATH = FIGURE_DIR / "S4_2d_kde_unified_branch_rules_heatmap.png"
FOCUS_PATH = FIGURE_DIR / "S4_2d_kde_unified_branch_rules_focus.png"

BANDWIDTH_FACTORS = (0.65, 0.80, 1.00, 1.25, 1.50)
MIN_BRANCH_VOTES = 3
HIGH_CONFIDENCE_BRANCH_VOTES = 4
MIN_CANDIDATE_BRANCH_VOTES = 2
MIN_CANDIDATE_SUPPORT_VOTES = 2
MIN_CANDIDATE_RELATIVE_MODE_SEPARATION = 1.0
MIN_VALLEY_DEPTH = 0.30
MIN_BRANCH_MASS = 0.10
MIN_TWO_COLUMN_FORK_SCORE = 0.15
MIN_THREE_COLUMN_FORK_SCORE = 0.20
MIN_COLUMN_SUPPORT = 30
MAX_TRIPLET_JUMP = 0.50


def _profile_pair(profile, y_centers):
    """Return the strongest two-mode/valley geometry in one density column."""
    if not np.all(np.isfinite(profile)) or float(np.sum(profile)) <= 0:
        return None
    density = np.asarray(profile, dtype=float)
    peaks = list(find_peaks(density, distance=max(2, len(density) // 40))[0])
    if density[0] > density[1] and density[0] >= 0.05 * density.max():
        peaks.insert(0, 0)
    if density[-1] > density[-2] and density[-1] >= 0.05 * density.max():
        peaks.append(len(density) - 1)
    peaks = sorted(set(peaks))
    best = None
    total = max(float(np.sum(density)), 1e-12)
    for left_i in range(len(peaks) - 1):
        for right_i in range(left_i + 1, len(peaks)):
            left, right = peaks[left_i], peaks[right_i]
            valley = left + int(np.argmin(density[left:right + 1]))
            low_peak = min(float(density[left]), float(density[right]))
            persistence = max(low_peak - float(density[valley]), 0.0)
            depth = persistence / max(low_peak, 1e-12)
            mass = min(
                float(np.sum(density[:valley + 1]) / total),
                float(np.sum(density[valley + 1:]) / total),
            )
            separation = float(y_centers[right] - y_centers[left])
            candidate = {
                "strength": persistence / max(float(np.max(density)), 1e-12),
                "low": float(y_centers[left]),
                "valley": float(y_centers[valley]),
                "high": float(y_centers[right]),
                "valley_depth": depth,
                "branch_mass": mass,
                "separation": separation,
                "boundary_peak": bool(left == 0 or right == len(density) - 1),
            }
            if best is None or candidate["strength"] > best["strength"]:
                best = candidate
    return best


def _density_surfaces(x, y):
    """Approximate five global 2-D KDEs on equal-frequency x columns.

    Smoothing in the x direction is performed in empirical-rank space.  This
    gives every conditional column comparable support and avoids sparse-tail
    columns controlling the branch decision.
    """
    x_lo, x_hi = np.quantile(x, [0.005, 0.995])
    y_lo, y_hi = np.quantile(y, [0.005, 0.995])
    if x_hi - x_lo <= 1e-12 or y_hi - y_lo <= 1e-12:
        return None
    nx_target = int(np.clip(len(x) // 80, 12, 32))
    ny = 112
    keep = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    x_edges = np.unique(np.quantile(
        x[keep], np.linspace(0.0, 1.0, nx_target + 1),
    ))
    if len(x_edges) < 4:
        return None
    nx = len(x_edges) - 1
    y_edges = np.linspace(y_lo, y_hi, ny + 1)
    histogram, _, _ = np.histogram2d(x[keep], y[keep], bins=(x_edges, y_edges))
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    y_bin = (y_hi - y_lo) / ny
    scott = max(int(np.sum(keep)), 2) ** (-1.0 / 6.0)
    sigma_x = max(0.8, scott * nx / np.sqrt(12.0))
    sigma_y = max(0.8, scott * float(np.std(y[keep])) / y_bin)
    surfaces = []
    for factor in BANDWIDTH_FACTORS:
        smooth = gaussian_filter(
            histogram,
            sigma=(factor * sigma_x, factor * sigma_y),
            mode="constant",
        )
        row_sum = smooth.sum(axis=1, keepdims=True)
        conditional = np.divide(
            smooth,
            row_sum,
            out=np.zeros_like(smooth),
            where=row_sum > 0,
        )
        surfaces.append(conditional)
    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "histogram": histogram,
        "surfaces": surfaces,
    }


def _column_geometry(surface_data, surface, bandwidth_factor):
    """Extract independently gated peak/valley triplets for one bandwidth."""
    points = []
    histogram = surface_data["histogram"]
    x_centers = surface_data["x_centers"]
    y_centers = surface_data["y_centers"]
    for column, x_mid in enumerate(x_centers):
        support = float(histogram[column].sum())
        estimate = _profile_pair(surface[column], y_centers)
        if estimate is not None:
            low = float(estimate["low"])
            valley = float(estimate["valley"])
            high = float(estimate["high"])
            depth = float(estimate["valley_depth"])
            mass = float(estimate["branch_mass"])
            separation = float(estimate["separation"])
        else:
            low = valley = high = np.nan
            depth = mass = separation = 0.0
        flag = bool(
            support >= MIN_COLUMN_SUPPORT
            and estimate is not None
            and depth >= MIN_VALLEY_DEPTH
            and mass >= MIN_BRANCH_MASS
        )
        points.append({
            "x_mid": float(x_mid),
            "support": support,
            "flag": flag,
            "low": low,
            "valley": valley,
            "high": high,
            "valley_depth": depth,
            "branch_mass": mass,
            "separation": separation,
            "bandwidth_factor": float(bandwidth_factor),
        })
    return points


def _connected_runs(points, flag_key="flag"):
    runs, current, previous = [], [], None
    for point in points:
        triplet = np.asarray([point["low"], point["valley"], point["high"]])
        usable = bool(
            point[flag_key] and np.all(np.isfinite(triplet))
            and triplet[0] < triplet[1] < triplet[2]
        )
        connected = False
        if usable and previous is not None:
            old = np.asarray([previous["low"], previous["valley"], previous["high"]])
            reference_gap = np.median([old[2] - old[0], triplet[2] - triplet[0]])
            jump = float(np.max(np.abs(triplet - old)) / max(reference_gap, 1e-12))
            connected = bool(jump <= MAX_TRIPLET_JUMP)
        if usable:
            if current and not connected:
                runs.append(current)
                current = []
            current.append(point)
            previous = point
        else:
            if current:
                runs.append(current)
                current = []
            previous = None
    if current:
        runs.append(current)
    return runs


def _classify_bandwidth(surface_data, surface, x, y, coverage_mode, factor):
    """Classify one complete global 2-D KDE surface."""
    points = _column_geometry(surface_data, surface, factor)
    runs = _connected_runs(points, "flag")
    run = max(runs, key=lambda item: (len(item), item[-1]["x_mid"] - item[0]["x_mid"]), default=[])
    run_length = len(run)
    x_sorted = np.sort(x)
    if run:
        x_edges = surface_data["x_edges"]
        first_col = points.index(run[0])
        last_col = points.index(run[-1])
        run_x_lo = float(x_edges[first_col])
        run_x_hi = float(x_edges[last_col + 1])
        if coverage_mode == "point_count":
            n_in = int(np.sum((x_sorted >= run_x_lo) & (x_sorted <= run_x_hi)))
            coverage = float(n_in / max(len(x_sorted), 1))
        else:
            x_range = max(float(x_edges[-1] - x_edges[0]), 1e-12)
            coverage = float((run_x_hi - run_x_lo) / x_range)
        separations = np.asarray([item["high"] - item["low"] for item in run])
        q05, q95 = np.quantile(y, [0.05, 0.95])
        y90_range = max(float(q95 - q05), 1e-12)
        median_relative_mode_separation = float(np.median(separations) / y90_range)
        relative_opening = float((np.max(separations) - np.min(separations)) / y90_range)
        fork_score = float(min(median_relative_mode_separation, relative_opening))
        edge = min(2, run_length)
        relative_change = float(
            abs(np.mean(separations[-edge:]) - np.mean(separations[:edge]))
            / max(float(np.max(separations)), 1e-12)
        )
        median_depth = float(np.median([item["valley_depth"] for item in run]))
        median_mass = float(np.median([item["branch_mass"] for item in run]))
        median_separation = float(np.median([item["separation"] for item in run]))
        x_start, x_end = run[0]["x_mid"], run[-1]["x_mid"]
    else:
        coverage = relative_change = median_depth = median_mass = median_separation = 0.0
        median_relative_mode_separation = 0.0
        relative_opening = fork_score = 0.0
        x_start = x_end = np.nan

    if coverage_mode == "point_count":
        cov_branch, cov_candidate = 0.30, 0.15
    else:
        cov_branch, cov_candidate = 0.10, 0.05
    if run_length == 2:
        fork_threshold = MIN_TWO_COLUMN_FORK_SCORE
    elif run_length >= 3:
        fork_threshold = MIN_THREE_COLUMN_FORK_SCORE
    else:
        fork_threshold = np.nan
    reliable_fork = bool(
        run_length >= 2
        and fork_score >= fork_threshold
        and coverage >= cov_candidate
    )
    if reliable_fork and run_length >= 3 and coverage >= cov_branch:
        status, branch_type = "Branch", "extended global-KDE fork"
    elif reliable_fork:
        status, branch_type = "Candidate", "local global-KDE fork"
    else:
        status, branch_type = "No branch", "none"
    stable_multimodal = bool(
        run_length >= 3
        and coverage >= cov_candidate
        and median_relative_mode_separation >= 0.20
        and not reliable_fork
    )
    return {
        "status": status,
        "branch_type": branch_type,
        "n_x_columns": len(points),
        "n_valid_columns": int(sum(item["flag"] for item in points)),
        "run_length": run_length,
        "x_start": x_start,
        "x_end": x_end,
        "x_coverage": coverage,
        "coverage_mode": coverage_mode,
        "relative_separation_change": relative_change,
        "median_valley_depth": median_depth,
        "median_branch_mass": median_mass,
        "median_mode_separation": median_separation,
        "median_relative_mode_separation": median_relative_mode_separation,
        "relative_opening": relative_opening,
        "fork_score": fork_score,
        "stable_multimodal": stable_multimodal,
        "bandwidth_factor": float(factor),
    }, {"points": points, "runs": runs, "run": run}


def _vote_bandwidth_results(results):
    n_branch = sum(item["status"] == "Branch" for item in results)
    n_support = sum(item["status"] in {"Branch", "Candidate"} for item in results)
    supported_sep = [
        float(item["median_relative_mode_separation"])
        for item in results
        if item["status"] in {"Branch", "Candidate"}
        and np.isfinite(item["median_relative_mode_separation"])
    ]
    support_median_sep = float(np.median(supported_sep)) if supported_sep else 0.0
    if n_branch >= MIN_BRANCH_VOTES:
        status = "Branch"
        branch_type = (
            "global-KDE multiband fork (high confidence)"
            if n_branch >= HIGH_CONFIDENCE_BRANCH_VOTES
            else "global-KDE multiband majority fork"
        )
    elif (
        (n_branch >= MIN_CANDIDATE_BRANCH_VOTES or n_support >= MIN_CANDIDATE_SUPPORT_VOTES)
        and support_median_sep >= MIN_CANDIDATE_RELATIVE_MODE_SEPARATION
    ):
        status, branch_type = "Candidate", "global-KDE multiband candidate"
    else:
        status, branch_type = "No branch", "none"
    return status, branch_type, n_branch, n_support, support_median_sep


def classify_panel(x, y, return_geometry=False, coverage_mode="x_range"):
    """Fit global 2-D KDEs, classify each bandwidth, then vote 3/5."""
    x, y = stratified_cap(x, y)
    surface_data = _density_surfaces(x, y)
    if surface_data is None:
        result = {"status": "No branch", "branch_type": "none", "run_length": 0,
                  "coverage_mode": coverage_mode}
        return (result, None) if return_geometry else result
    per_bandwidth, geometries = [], []
    for factor, surface in zip(BANDWIDTH_FACTORS, surface_data["surfaces"]):
        item, geometry = _classify_bandwidth(
            surface_data, surface, x, y, coverage_mode, factor,
        )
        per_bandwidth.append(item)
        geometries.append(geometry)
    status, branch_type, n_branch, n_support, support_median_sep = (
        _vote_bandwidth_results(per_bandwidth)
    )
    priority = {"No branch": 0, "Candidate": 1, "Branch": 2}
    selected_index = max(
        range(len(per_bandwidth)),
        key=lambda index: (
            priority[per_bandwidth[index]["status"]],
            per_bandwidth[index]["run_length"],
            per_bandwidth[index]["fork_score"],
            -abs(BANDWIDTH_FACTORS[index] - 1.0),
        ),
    )
    result = dict(per_bandwidth[selected_index])
    result.update({
        "status": status,
        "branch_type": branch_type,
        "n_points": len(x),
        "coverage_mode": coverage_mode,
        "bandwidth_vote_pattern": "|".join(item["status"] for item in per_bandwidth),
        "bandwidth_branch_votes": int(n_branch),
        "bandwidth_support_votes": int(n_support),
        "bandwidth_support_median_relative_mode_separation": support_median_sep,
        "candidate_relative_mode_separation_threshold": MIN_CANDIDATE_RELATIVE_MODE_SEPARATION,
        "high_confidence_branch": bool(status == "Branch" and n_branch >= HIGH_CONFIDENCE_BRANCH_VOTES),
        "selected_bandwidth_factor": float(BANDWIDTH_FACTORS[selected_index]),
        "stable_multimodal": bool(any(item["stable_multimodal"] for item in per_bandwidth)),
    })
    for factor, item in zip(BANDWIDTH_FACTORS, per_bandwidth):
        result[f"bandwidth_{factor:.2f}_status"] = item["status"]
    if return_geometry:
        geometry = geometries[selected_index]
        return result, {
            "surface": surface_data,
            "points": geometry["points"],
            "run": geometry["run"],
            "bandwidth_results": per_bandwidth,
            "bandwidth_geometries": geometries,
            "selected_bandwidth_factor": float(BANDWIDTH_FACTORS[selected_index]),
        }
    return result


def plot_heatmap(results):
    codes = {"No branch": 0, "Candidate": 1, "Branch": 2}
    letters = {"No branch": "–", "Candidate": "C", "Branch": "B"}
    cmap = ListedColormap(["#F3F4F6", "#F59E0B", "#2563EB"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    n_pairs = len(PAIRS)
    fig, axes = plt.subplots(
        2, n_pairs, figsize=(max(14, 3.8 * n_pairs), 7.7),
        constrained_layout=True, squeeze=False,
    )
    for row, version in enumerate(("raw", "trimmed")):
        for col, (_, _, pair) in enumerate(PAIRS):
            ax = axes[row, col]
            selected = results[(results["pair"] == pair) & (results["version"] == version)].copy()
            selected["code"] = selected["status"].map(codes)
            matrix = selected.pivot(index="model", columns="domain", values="code").reindex(index=MODELS, columns=DOMAINS)
            statuses = selected.pivot(index="model", columns="domain", values="status").reindex(index=MODELS, columns=DOMAINS)
            ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    status = statuses.iloc[i, j]
                    ax.text(j, i, letters[status], ha="center", va="center", fontsize=11,
                            fontweight="bold", color="white" if codes[status] >= 2 else "#374151")
            counts = selected["status"].value_counts()
            ax.set_title(f"{pair} · {version.upper()} · B={counts.get('Branch', 0)}, "
                         f"C={counts.get('Candidate', 0)}",
                         fontsize=10.5, fontweight="bold")
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.set_xticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle(
        "2-D KDE tracks: run-level relative peak separation",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(FIGURE_PATH, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_focus(cases):
    focus = [
        ("CESM2", "all_land", "P → Q", "raw"),
        ("CESM2", "LI", "P → Q", "raw"),
        ("GFDL-CM4", "LI", "P → Q", "raw"),
        ("CanESM5", "CW", "mrros → Q", "raw"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    for ax, key in zip(axes, focus):
        result, geometry = classify_panel(*cases[key], return_geometry=True)
        surface = geometry["surface"]
        selected_index = BANDWIDTH_FACTORS.index(geometry["selected_bandwidth_factor"])
        density = surface["surfaces"][selected_index]
        ax.pcolormesh(surface["x_edges"], surface["y_edges"], np.log1p(density.T * 1000),
                      shading="auto", cmap="Reds")
        run = geometry["run"]
        if run:
            xx = [item["x_mid"] for item in run]
            ax.plot(xx, [item["low"] for item in run], color="#2563EB", lw=2, marker="o", ms=3)
            ax.plot(xx, [item["high"] for item in run], color="#2563EB", lw=2, marker="o", ms=3)
            ax.plot(xx, [item["valley"] for item in run], color="#111827", lw=1.4, ls="--")
        ax.set_title(f"{key[0]} · {key[1]}\n{key[2]} · {result['status']}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("Q")
    fig.suptitle("Conditional 2-D KDE with tracked density peaks and valley", fontsize=14, fontweight="bold")
    fig.savefig(FOCUS_PATH, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    cases = build_cases(load_runs())
    rows = []
    for index, (key, (x, y)) in enumerate(cases.items(), start=1):
        model, domain, pair, version = key
        rows.append({"model": model, "domain": domain, "pair": pair, "version": version,
                     **classify_panel(x, y)})
        if index % 15 == 0:
            print(f"Processed {index}/{len(cases)}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(CSV_PATH, index=False)
    plot_heatmap(results)
    plot_focus(cases)
    print("\nCounts")
    print(results.groupby(["pair", "version", "status"]).size().to_string())
    print("\nFocus panels")
    print(results.loc[
        ((results["pair"] == "P → Q") & (results["domain"] == "LI"))
        | ((results["pair"] == "P → Q") & (results["domain"] == "all_land"))
        | ((results["pair"] == "mrros → Q") & (results["model"] == "CanESM5") & (results["domain"] == "CW")),
        ["model", "domain", "pair", "version", "status", "run_length", "x_coverage",
         "median_valley_depth", "median_branch_mass", "median_mode_separation",
         "median_relative_mode_separation"],
    ].to_string(index=False))
    print(f"\nSaved {CSV_PATH}")
    print(f"Saved {FIGURE_PATH}")
    print(f"Saved {FOCUS_PATH}")


if __name__ == "__main__":
    main()
