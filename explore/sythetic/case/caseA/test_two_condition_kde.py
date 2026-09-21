"""Peak-valley-peak tracked multi-bandwidth KDE branch detector.

Each of three KDE bandwidths independently performs the complete detection
pipeline: local bimodality within x windows, ordered peak/valley tracking across
windows, and geometric fork classification. A local window passes one
bandwidth when its relative valley depth is at least 0.30 and its smaller-side
mass is at least 0.10.

Within each bandwidth, adjacent windows are connected only when their lower
peak, valley, and upper peak form a continuous ordered triplet. The fork score
combines both the typical separation and its opening/closing amplitude:

    R_fork = min(median(d) / y90_range, (max(d) - min(d)) / y90_range)

where d is the peak-to-peak distance along a connected run and y90_range is
Q95(y) - Q05(y). Two-window local candidates require R_fork >= 0.15; tracks
with at least three windows require R_fork >= 0.20. Each bandwidth therefore
votes Branch, Candidate, or No branch. The final consensus is Branch for at
least 2/3 Branch votes. Candidate requires at least 1/3 Branch-or-Candidate
support votes and a median supported relative mode separation of at least
0.5. No branch is returned otherwise. All three Branch votes are retained
as a high-confidence flag.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from branch_benchmark import _robust_location_scale, _window_slices
from test_kde_peak_valley_score import (
    peak_valley_at_bandwidth,
    peak_valley_at_bandwidths,
)
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS
from test_sliding_kde_modal_summary import stratified_cap


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
CSV_PATH = OUTPUT_DIR / "S4_scatter_window_kde_unified_branch_rules_results.csv"
FIGURE_PATH = FIGURE_DIR / "S4_scatter_window_kde_unified_branch_rules_heatmap.png"

BANDWIDTH_FACTORS = (0.65, 1.00, 1.50)
MIN_BRANCH_VOTES = 2
HIGH_CONFIDENCE_BRANCH_VOTES = 3
MIN_CANDIDATE_BRANCH_VOTES = 1
MIN_CANDIDATE_SUPPORT_VOTES = 1
MIN_CANDIDATE_RELATIVE_MODE_SEPARATION = 0.5
MIN_VALLEY_DEPTH = 0.30
MIN_BRANCH_MASS = 0.10
MIN_TWO_WINDOW_FORK_SCORE = 0.15
MIN_THREE_WINDOW_FORK_SCORE = 0.20
# Legacy side-peak-ratio gates (kept for comparison, inactive):
# MAX_SIDE_PEAK_RATIO = 0.50
# MAX_SIDE_PEAK_RATIO = 0.30
# Legacy global-y-scale peak-distance gate (kept for comparison, inactive):
# MIN_MODE_SEPARATION = 1.00
MAX_TRIPLET_JUMP = 0.50
USE_TRIPLET_JUMP = False


def _pair_from_estimate(estimate, bandwidth_factor):
    """Convert a peak-valley estimate into the cached per-bandwidth record."""
    present = bool(estimate["score"] > 0)
    depth = float(estimate["valley_depth"]) if present else 0.0
    pair_score = float(estimate.get("pair_score", 0.0)) if present else 0.0
    mass = float(estimate["min_weight"]) if present else 0.0
    separation = (
        float(abs(estimate["modes"][1] - estimate["modes"][0]))
        if present else 0.0
    )
    height_ratio = pair_score / max(depth, 1e-12) if present else 1.0
    modes = tuple(float(value) for value in estimate["modes"]) if present else (np.nan, np.nan)
    valley = float(estimate["valley"]) if present else np.nan
    return {
        "flag": bool(
            present
            and depth >= MIN_VALLEY_DEPTH
            and mass >= MIN_BRANCH_MASS
            # Legacy scale-dependent gate (inactive):
            # and median_separation >= MIN_MODE_SEPARATION
        ),
        "modes": modes,
        "valley": valley,
        "median_valley_depth": depth,
        "median_peak_valley_score": pair_score,
        "median_branch_mass": mass,
        "median_mode_separation": separation,
        "median_peak_height_ratio": height_ratio,
        "n_bandwidth_pairs": int(present),
        "bandwidth_factor": float(bandwidth_factor),
    }


def single_band_pair(values, bandwidth_factor):
    """Return one bandwidth's local peak-valley decision.

    The historical ``median_*`` field names are retained for compatibility
    with downstream diagnostics; each now contains the selected bandwidth's
    value rather than a median across bandwidths.
    """
    return _pair_from_estimate(
        peak_valley_at_bandwidth(values, bandwidth_factor),
        bandwidth_factor,
    )


def single_band_pairs(values, bandwidth_factors=BANDWIDTH_FACTORS):
    """Shared-grid peak-valley decisions for every detector bandwidth."""
    estimates = peak_valley_at_bandwidths(values, bandwidth_factors)
    return [
        _pair_from_estimate(estimate, factor)
        for estimate, factor in zip(estimates, bandwidth_factors)
    ]


def connected_triplet_runs(points):
    """Connect only ordered lower-peak/valley/upper-peak triplets."""
    runs, current = [], []
    previous = None
    for point in points:
        triplet = np.asarray(
            [point["modes"][0], point["valley"], point["modes"][1]],
            dtype=float,
        )
        usable = bool(
            point["flag"]
            and np.all(np.isfinite(triplet))
            and triplet[0] < triplet[1] < triplet[2]
        )
        connected = False
        jump = np.nan
        if usable and previous is not None:
            previous_triplet = np.asarray(
                [previous["modes"][0], previous["valley"], previous["modes"][1]],
                dtype=float,
            )
            reference_gap = float(np.median([
                previous_triplet[2] - previous_triplet[0],
                triplet[2] - triplet[0],
            ]))
            jump = float(
                np.max(np.abs(triplet - previous_triplet))
                / max(reference_gap, 1e-12)
            )
            connected = True if not USE_TRIPLET_JUMP else bool(jump <= MAX_TRIPLET_JUMP)
        point["triplet_jump"] = jump
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


def _classify_bandwidth_points(points, x, y, coverage_mode, bandwidth_factor):
    """Run the existing cross-window geometry rule for one KDE bandwidth."""
    runs = connected_triplet_runs(points[::2]) + connected_triplet_runs(points[1::2])
    run = max(
        runs,
        key=lambda r: (len(r), r[-1]["x_max"] - r[0]["x_min"]),
        default=[],
    )
    run_length = len(run)
    if run:
        if coverage_mode == "point_count":
            n_in = int(np.sum((x >= run[0]["x_min"]) & (x <= run[-1]["x_max"])))
            coverage = float(n_in / max(len(x), 1))
        else:
            x_range = max(float(np.max(x) - np.min(x)), 1e-12)
            coverage = float((run[-1]["x_max"] - run[0]["x_min"]) / x_range)
        separation = np.asarray([item["modes"][1] - item["modes"][0] for item in run])
        q05, q95 = np.quantile(y, [0.05, 0.95])
        y90_range = max(float(q95 - q05), 1e-12)
        median_relative_mode_separation = float(
            np.median(separation) / y90_range
        )
        relative_opening = float(
            (np.max(separation) - np.min(separation)) / y90_range
        )
        fork_score = float(min(median_relative_mode_separation, relative_opening))
        edge = min(2, run_length)
        relative_change = float(
            abs(np.mean(separation[-edge:]) - np.mean(separation[:edge]))
            / max(float(np.max(separation)), 1e-12)
        )
        median_depth = float(np.median([item["median_valley_depth"] for item in run]))
        median_pair_score = float(np.median([
            item["median_peak_valley_score"] for item in run
        ]))
        median_mass = float(np.median([item["median_branch_mass"] for item in run]))
        median_mode_separation = float(np.median([
            item["median_mode_separation"] for item in run
        ]))
        median_peak_height_ratio = float(np.median([
            item["median_peak_height_ratio"] for item in run
        ]))
        finite_jumps = [item["triplet_jump"] for item in run if np.isfinite(item["triplet_jump"])]
        median_triplet_jump = float(np.median(finite_jumps)) if finite_jumps else 0.0
        x_start, x_end = run[0]["x_min"], run[-1]["x_max"]
    else:
        coverage = relative_change = median_depth = median_mass = 0.0
        median_pair_score = 0.0
        median_mode_separation = 0.0
        median_relative_mode_separation = 0.0
        relative_opening = 0.0
        fork_score = 0.0
        median_peak_height_ratio = 1.0
        median_triplet_jump = 0.0
        x_start = x_end = np.nan

    # Coverage controls Branch versus Candidate after the same geometric fork
    # test. A two-window run is allowed only as Candidate and uses the slightly
    # lower 0.15 threshold; runs of three or more windows use 0.20.
    if coverage_mode == "point_count":
        _cov_branch, _cov_candidate = 0.30, 0.15
    else:  # x_range
        _cov_branch, _cov_candidate = 0.10, 0.05

    if run_length == 2:
        fork_score_threshold = MIN_TWO_WINDOW_FORK_SCORE
    elif run_length >= 3:
        fork_score_threshold = MIN_THREE_WINDOW_FORK_SCORE
    else:
        fork_score_threshold = np.nan

    reliable_fork = bool(
        run_length >= 2
        and fork_score >= fork_score_threshold
        and coverage >= _cov_candidate
    )
    if reliable_fork and run_length >= 3 and coverage >= _cov_branch:
        status, branch_type = "Branch", "extended fork"
    elif reliable_fork:
        status, branch_type = "Candidate", "local fork"
    else:
        status, branch_type = "No branch", "none"

    # Legacy peak-height-asymmetry Candidate rules (inactive):
    # elif run_length >= 3 and median_peak_height_ratio <= MAX_SIDE_PEAK_RATIO:
    #     status, branch_type = "Candidate", "persistent local side branch"
    # elif run_length >= 2 and coverage >= _cov_short and median_peak_height_ratio <= MAX_SIDE_PEAK_RATIO:
    #     status, branch_type = "Candidate", "short local side branch"

    x_mids = np.asarray([p["x_mid"] for p in points])
    y_meds = np.asarray([p["y_median"] for p in points])
    max_curvature = 0.0
    curvature_x = np.nan
    l_shape = False
    if len(y_meds) >= 5:
        dx = np.diff(x_mids)
        dy = np.diff(y_meds)
        slopes = dy / np.maximum(dx, 1e-12)
        slope_scale = max(float(np.percentile(np.abs(slopes), 90)), 1e-12)
        slope_change = np.abs(np.diff(slopes)) / slope_scale
        max_idx = int(np.argmax(slope_change))
        max_curvature = float(slope_change[max_idx])
        curvature_x = float(x_mids[max_idx + 1])
        l_shape = bool(max_curvature >= 2.0)

    # L-shape classification disabled — curvature metric needs refinement
    # if l_shape and status == "No branch":
    #     status, branch_type = "Candidate", "L-shape curvature"

    result = {
        "status": status,
        "branch_type": branch_type,
        "bandwidth_factor": float(bandwidth_factor),
        "n_points": len(x),
        "n_windows": len(points),
        "n_valid_windows": int(sum(point["flag"] for point in points)),
        "run_length": run_length,
        "x_start": x_start,
        "x_end": x_end,
        "x_coverage": coverage,
        "coverage_mode": coverage_mode,
        "relative_separation_change": relative_change,
        "relative_opening": relative_opening,
        "fork_score": fork_score,
        "fork_score_threshold": fork_score_threshold,
        "median_valley_depth": median_depth,
        "median_peak_valley_score": median_pair_score,
        "median_branch_mass": median_mass,
        "median_mode_separation": median_mode_separation,
        "median_relative_mode_separation": median_relative_mode_separation,
        "median_peak_height_ratio": median_peak_height_ratio,
        "median_triplet_jump": median_triplet_jump,
        "max_curvature": max_curvature,
        "curvature_x": curvature_x,
        "l_shape": l_shape,
    }
    return result, {"points": points, "runs": runs, "run": run}


def _vote_bandwidth_results(results):
    """Combine three per-bandwidth classifications by 2/3 voting."""
    n_branch = sum(item["status"] == "Branch" for item in results)
    n_support = sum(item["status"] in {"Branch", "Candidate"} for item in results)
    supported_separations = [
        float(item["median_relative_mode_separation"])
        for item in results
        if item["status"] in {"Branch", "Candidate"}
        and np.isfinite(item["median_relative_mode_separation"])
    ]
    support_median_separation = (
        float(np.median(supported_separations))
        if supported_separations else 0.0
    )
    if n_branch >= MIN_BRANCH_VOTES:
        status = "Branch"
        branch_type = (
            "multiband consensus fork (high confidence)"
            if n_branch >= HIGH_CONFIDENCE_BRANCH_VOTES
            else "multiband majority fork"
        )
    elif (
        (
            n_branch >= MIN_CANDIDATE_BRANCH_VOTES
            or n_support >= MIN_CANDIDATE_SUPPORT_VOTES
        )
        and support_median_separation
        >= MIN_CANDIDATE_RELATIVE_MODE_SEPARATION
    ):
        status = "Candidate"
        branch_type = (
            "multiband branch-vote candidate"
            if n_branch >= MIN_CANDIDATE_BRANCH_VOTES
            else "multiband support candidate"
        )
    else:
        status, branch_type = "No branch", "none"
    return status, branch_type, n_branch, n_support, support_median_separation


def prepare_panel_xy(x, y):
    """Apply the detector's deterministic cap and x ordering once."""
    x, y = stratified_cap(x, y)
    order = np.argsort(x, kind="mergesort")
    return x[order], y[order]


def fit_panel_kde(x, y, min_window=40):
    """Fit all local KDEs without applying the downstream branch rules.

    The returned peak/valley summaries are the expensive reusable layer.  A
    caller may persist ``bandwidth_points`` and later pass it to
    :func:`classify_fitted_panel` after changing gate, geometry, or vote
    thresholds.  Changing the window layout, bandwidth factors, or peak/valley
    extraction itself still requires refitting this layer.
    """
    x, y = prepare_panel_xy(x, y)
    y_center, y_scale = _robust_location_scale(y)
    y_scaled = (y - y_center) / y_scale

    bandwidth_points = {factor: [] for factor in BANDWIDTH_FACTORS}
    for start, stop in _window_slices(
        len(x), window_frac=0.06, step_frac=0.5, min_size=min_window,
    ):
        base = {
            "x_min": float(x[start]),
            "x_mid": float(np.median(x[start:stop])),
            "x_max": float(x[stop - 1]),
            "y_median": float(np.median(y[start:stop])),
        }
        window_values = y_scaled[start:stop]
        for factor, estimate in zip(BANDWIDTH_FACTORS, single_band_pairs(window_values)):
            bandwidth_points[factor].append({
                **base,
                **estimate,
                "modes": tuple(
                    value * y_scale + y_center for value in estimate["modes"]
                ),
                "valley": estimate["valley"] * y_scale + y_center,
            })
    return {
        "x": x,
        "y": y,
        "bandwidth_points": bandwidth_points,
        "min_window": int(min_window),
    }


def fit_panel_cache_rows(case_id, x, y, min_window=40, cache_version=1):
    """Pickleable helper: fit one panel and flatten it to cache rows."""
    fitted = fit_panel_kde(x, y, min_window=min_window)
    rows = []
    for factor, window_points in fitted["bandwidth_points"].items():
        for window_index, point in enumerate(window_points):
            mode_low, mode_high = point["modes"]
            rows.append({
                "case_id": str(case_id),
                "cache_version": int(cache_version),
                "min_window": int(min_window),
                "bandwidth_factor": float(factor),
                "window_index": int(window_index),
                "x_min": point["x_min"],
                "x_mid": point["x_mid"],
                "x_max": point["x_max"],
                "y_median": point["y_median"],
                "mode_low": mode_low,
                "mode_high": mode_high,
                "valley": point["valley"],
                "median_valley_depth": point["median_valley_depth"],
                "median_peak_valley_score": point["median_peak_valley_score"],
                "median_branch_mass": point["median_branch_mass"],
                "median_mode_separation": point["median_mode_separation"],
                "median_peak_height_ratio": point["median_peak_height_ratio"],
                "n_bandwidth_pairs": point["n_bandwidth_pairs"],
            })
    return rows


def fit_panel_cache_rows_from_arrays(
    case_ids, point_indices, x_all, y_all, min_window=40, cache_version=1,
):
    """Fit a batch of panels from shared ``x_all`` / ``y_all`` arrays.

    Joblib memmaps the point arrays once per worker; each task only ships
    case identifiers and integer row indices.
    """
    rows = []
    for case_id, point_index in zip(case_ids, point_indices):
        rows.extend(fit_panel_cache_rows(
            case_id,
            np.asarray(x_all[int(point_index)], dtype=float),
            np.asarray(y_all[int(point_index)], dtype=float),
            min_window=min_window,
            cache_version=cache_version,
        ))
    return rows


def _points_with_current_local_gate(points):
    """Copy cached KDE summaries and apply the current local-mode gate."""
    gated = []
    for original in points:
        point = dict(original)
        modes = np.asarray(point.get("modes", (np.nan, np.nan)), dtype=float)
        present = bool(
            int(point.get("n_bandwidth_pairs", 0)) > 0
            and modes.size == 2
            and np.all(np.isfinite(modes))
            and np.isfinite(point.get("valley", np.nan))
        )
        point["flag"] = bool(
            present
            and float(point.get("median_valley_depth", 0.0)) >= MIN_VALLEY_DEPTH
            and float(point.get("median_branch_mass", 0.0)) >= MIN_BRANCH_MASS
        )
        gated.append(point)
    return gated


def classify_fitted_panel(
    fitted, return_geometry=False, coverage_mode="x_range",
):
    """Classify a reusable :func:`fit_panel_kde` result without KDE refitting."""
    if coverage_mode not in {"x_range", "point_count"}:
        raise ValueError("coverage_mode must be 'x_range' or 'point_count'")

    x = np.asarray(fitted["x"], dtype=float)
    y = np.asarray(fitted["y"], dtype=float)
    cached_points = fitted["bandwidth_points"]

    per_bandwidth = []
    geometries = []
    for factor in BANDWIDTH_FACTORS:
        points = _points_with_current_local_gate(cached_points.get(factor, []))
        result, geometry = _classify_bandwidth_points(
            points, x, y, coverage_mode, factor,
        )
        per_bandwidth.append(result)
        geometries.append(geometry)

    (
        status,
        branch_type,
        n_branch,
        n_support,
        support_median_separation,
    ) = _vote_bandwidth_results(per_bandwidth)
    status_priority = {"No branch": 0, "Candidate": 1, "Branch": 2}
    selected_index = max(
        range(len(per_bandwidth)),
        key=lambda index: (
            status_priority[per_bandwidth[index]["status"]],
            per_bandwidth[index]["run_length"],
            per_bandwidth[index]["x_coverage"],
            per_bandwidth[index]["fork_score"],
            -abs(BANDWIDTH_FACTORS[index] - 1.0),
        ),
    )
    selected = dict(per_bandwidth[selected_index])
    selected.update({
        "status": status,
        "branch_type": branch_type,
        "bandwidth_factors": tuple(float(value) for value in BANDWIDTH_FACTORS),
        "bandwidth_vote_pattern": "|".join(
            item["status"] for item in per_bandwidth
        ),
        "bandwidth_branch_votes": int(n_branch),
        "bandwidth_support_votes": int(n_support),
        "bandwidth_support_median_relative_mode_separation": float(
            support_median_separation
        ),
        "candidate_relative_mode_separation_threshold": float(
            MIN_CANDIDATE_RELATIVE_MODE_SEPARATION
        ),
        "bandwidth_vote_threshold": MIN_BRANCH_VOTES,
        "high_confidence_branch": bool(
            status == "Branch" and n_branch >= HIGH_CONFIDENCE_BRANCH_VOTES
        ),
        "selected_bandwidth_factor": float(BANDWIDTH_FACTORS[selected_index]),
    })
    for factor, item in zip(BANDWIDTH_FACTORS, per_bandwidth):
        selected[f"bandwidth_{factor:.2f}_status"] = item["status"]

    if return_geometry:
        selected_geometry = geometries[selected_index]
        return selected, {
            "points": selected_geometry["points"],
            "runs": selected_geometry["runs"],
            "run": selected_geometry["run"],
            "selected_bandwidth_factor": float(BANDWIDTH_FACTORS[selected_index]),
            "bandwidth_geometries": geometries,
            "bandwidth_results": per_bandwidth,
        }
    return selected


def classify_panel(x, y, return_geometry=False, coverage_mode="x_range", min_window=40):
    """Fit local KDEs, then classify the three bandwidth tracks by 2/3 voting."""
    fitted = fit_panel_kde(x, y, min_window=min_window)
    return classify_fitted_panel(
        fitted,
        return_geometry=return_geometry,
        coverage_mode=coverage_mode,
    )


def plot_heatmap(results):
    codes = {"No branch": 0, "Candidate": 1, "Two-band": 2, "Branch": 3}
    letters = {"No branch": "–", "Candidate": "C", "Two-band": "T", "Branch": "B"}
    cmap = ListedColormap(["#F3F4F6", "#F59E0B", "#7C3AED", "#2563EB"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    n_pairs = len(PAIRS)
    fig, axes = plt.subplots(
        2, n_pairs, figsize=(max(14, 3.8 * n_pairs), 7.7),
        constrained_layout=True, squeeze=False,
    )
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
                    ax.text(j, i, letters[status], ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if codes[status] >= 2 else "#374151")
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
        "Scatter-window KDE fork tracks · five independent bandwidths · 3/5 vote "
        "(P′ ≥ 0.30; R_fork ≥ 0.15 for 2 windows, ≥ 0.20 for 3+)",
        fontsize=15.2, fontweight="bold",
    )
    fig.savefig(FIGURE_PATH, dpi=190, bbox_inches="tight")
    plt.close(fig)


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
    results.to_csv(CSV_PATH, index=False)
    plot_heatmap(results)
    print("\nCounts")
    print(results.groupby(["pair", "version", "status"]).size().to_string())
    print("\nFocus panels")
    print(results.loc[
        (
            (results["pair"] == "mrros → Q") & (results["model"] == "CanESM5")
            & (results["domain"] == "CW")
        )
        | (
            (results["pair"] == "P → Q") & (results["domain"] == "LI")
            & results["model"].isin(["CESM2", "GFDL-CM4", "CMCC-CM2-SR5"])
        ),
        ["model", "domain", "pair", "version", "status", "branch_type",
         "bandwidth_vote_pattern", "bandwidth_branch_votes",
         "bandwidth_support_votes", "high_confidence_branch",
         "run_length", "median_valley_depth", "median_branch_mass",
         "median_mode_separation"],
    ].to_string(index=False))
    print(f"\nSaved {CSV_PATH}")
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
