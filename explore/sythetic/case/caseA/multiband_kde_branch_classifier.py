"""Five-bandwidth conditional-KDE branch classifier.

The classifier separates the expensive local KDE fit from the inexpensive
geometric/voting rules.  Each bandwidth independently labels a panel as
``Branch``, ``Candidate`` or ``No branch``.  The five-bandwidth consensus is:

* Branch: at least 3 explicit Branch votes.
* Candidate: fewer than 3 Branch votes, at least 2 support votes
  (Branch or Candidate), at least 1 explicit Branch vote, and median relative
  mode separation >= 0.90 among supporting bandwidths.
* No branch: everything else.

The two-support-vote rule is the core Candidate criterion.  The explicit
Branch vote and separation gates suppress weak, poorly separated KDE modes.
Triplet-jump filtering is intentionally not used.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from branch_benchmark import _robust_location_scale, _window_slices
from test_kde_peak_valley_score import peak_valley_at_bandwidths
from test_sliding_kde_modal_summary import stratified_cap


CLASSIFIER_VERSION = "5bw-consensus-v1"

# Five Scott-bandwidth multipliers.  The widest bandwidth is retained as an
# oversmoothing stability check even though it rarely changes the final vote.
BANDWIDTH_FACTORS = (0.65, 0.80, 1.00, 1.25, 1.50)

# Per-window bimodality gates.
MIN_VALLEY_DEPTH = 0.30
MIN_BRANCH_MASS = 0.10

# Per-bandwidth connected-fork gates.
MIN_TWO_WINDOW_FORK_SCORE = 0.15
MIN_THREE_WINDOW_FORK_SCORE = 0.20

# Five-bandwidth consensus gates.
MIN_BRANCH_VOTES = 3
MIN_CANDIDATE_SUPPORT_VOTES = 2
MIN_CANDIDATE_BRANCH_VOTES = 1
MIN_CANDIDATE_RELATIVE_MODE_SEPARATION = 0.90

DEFAULT_MIN_WINDOW = 40
DEFAULT_MIN_N = 30

STATUS_TO_GROUP = {
    "Branch": "Branch",
    "Candidate": "Candidate branch",
    "No branch": "No branch",
}


def classifier_signature() -> dict[str, Any]:
    """Return the fit/classification settings used for diagnostics and caches."""
    return {
        "classifier_version": CLASSIFIER_VERSION,
        "bandwidth_factors": BANDWIDTH_FACTORS,
        "min_valley_depth": MIN_VALLEY_DEPTH,
        "min_branch_mass": MIN_BRANCH_MASS,
        "min_two_window_fork_score": MIN_TWO_WINDOW_FORK_SCORE,
        "min_three_window_fork_score": MIN_THREE_WINDOW_FORK_SCORE,
        "min_branch_votes": MIN_BRANCH_VOTES,
        "min_candidate_support_votes": MIN_CANDIDATE_SUPPORT_VOTES,
        "min_candidate_branch_votes": MIN_CANDIDATE_BRANCH_VOTES,
        "min_candidate_relative_mode_separation": (
            MIN_CANDIDATE_RELATIVE_MODE_SEPARATION
        ),
        "triplet_jump_filter": False,
    }


def _peak_valley_record(estimate: dict[str, Any], factor: float) -> dict[str, Any]:
    present = bool(estimate["score"] > 0)
    depth = float(estimate["valley_depth"]) if present else 0.0
    pair_score = float(estimate.get("pair_score", 0.0)) if present else 0.0
    mass = float(estimate["min_weight"]) if present else 0.0
    separation = (
        float(abs(estimate["modes"][1] - estimate["modes"][0]))
        if present else 0.0
    )
    height_ratio = pair_score / max(depth, 1e-12) if present else 1.0
    modes = (
        tuple(float(value) for value in estimate["modes"])
        if present else (np.nan, np.nan)
    )
    valley = float(estimate["valley"]) if present else np.nan
    return {
        "flag": bool(
            present
            and depth >= MIN_VALLEY_DEPTH
            and mass >= MIN_BRANCH_MASS
        ),
        "modes": modes,
        "valley": valley,
        "median_valley_depth": depth,
        "median_peak_valley_score": pair_score,
        "median_branch_mass": mass,
        "median_mode_separation": separation,
        "median_peak_height_ratio": height_ratio,
        "n_bandwidth_pairs": int(present),
        "bandwidth_factor": float(factor),
    }


def _all_bandwidth_records(values: np.ndarray) -> list[dict[str, Any]]:
    estimates = peak_valley_at_bandwidths(values, BANDWIDTH_FACTORS)
    return [
        _peak_valley_record(estimate, factor)
        for estimate, factor in zip(estimates, BANDWIDTH_FACTORS)
    ]


def prepare_panel_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove invalid values, cap deterministically, and sort by x."""
    return stratified_cap(x, y)


def fit_kde(
    x: np.ndarray,
    y: np.ndarray,
    min_window: int = DEFAULT_MIN_WINDOW,
) -> dict[str, Any]:
    """Fit reusable local KDE peak/valley summaries at all five bandwidths."""
    x, y = prepare_panel_xy(x, y)
    if len(x) < DEFAULT_MIN_N:
        return {
            "x": x,
            "y": y,
            "bandwidth_points": {factor: [] for factor in BANDWIDTH_FACTORS},
            "bandwidth_factors": BANDWIDTH_FACTORS,
            "min_window": int(min_window),
        }

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
        records = _all_bandwidth_records(y_scaled[start:stop])
        for factor, record in zip(BANDWIDTH_FACTORS, records):
            bandwidth_points[factor].append({
                **base,
                **record,
                "modes": tuple(
                    value * y_scale + y_center for value in record["modes"]
                ),
                "valley": record["valley"] * y_scale + y_center,
            })

    return {
        "x": x,
        "y": y,
        "bandwidth_points": bandwidth_points,
        "bandwidth_factors": BANDWIDTH_FACTORS,
        "min_window": int(min_window),
    }


def _apply_local_gate(points: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
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
            and float(point.get("median_valley_depth", 0.0))
            >= MIN_VALLEY_DEPTH
            and float(point.get("median_branch_mass", 0.0))
            >= MIN_BRANCH_MASS
        )
        gated.append(point)
    return gated


def _connected_runs(points: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Track ordered peak-valley-peak triplets; no jump threshold is applied."""
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_usable = False
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
        if usable:
            if current and not previous_usable:
                runs.append(current)
                current = []
            current.append(point)
        elif current:
            runs.append(current)
            current = []
        previous_usable = usable
    if current:
        runs.append(current)
    return runs


def _classify_one_bandwidth(
    points: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    coverage_mode: str,
    factor: float,
) -> dict[str, Any]:
    # Preserve the historical odd/even window tracks used by the CMIP6 pilot.
    runs = _connected_runs(points[::2]) + _connected_runs(points[1::2])
    run = max(
        runs,
        key=lambda item: (len(item), item[-1]["x_max"] - item[0]["x_min"]),
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
        separation = np.asarray([
            item["modes"][1] - item["modes"][0] for item in run
        ])
        q05, q95 = np.quantile(y, [0.05, 0.95])
        y90_range = max(float(q95 - q05), 1e-12)
        median_relative_separation = float(np.median(separation) / y90_range)
        relative_opening = float(
            (np.max(separation) - np.min(separation)) / y90_range
        )
        fork_score = float(min(median_relative_separation, relative_opening))
        x_start = float(run[0]["x_min"])
        x_end = float(run[-1]["x_max"])
    else:
        coverage = 0.0
        median_relative_separation = 0.0
        relative_opening = 0.0
        fork_score = 0.0
        x_start = x_end = np.nan

    if coverage_mode == "point_count":
        coverage_branch, coverage_candidate = 0.30, 0.15
    else:
        coverage_branch, coverage_candidate = 0.10, 0.05

    if run_length == 2:
        fork_threshold = MIN_TWO_WINDOW_FORK_SCORE
    elif run_length >= 3:
        fork_threshold = MIN_THREE_WINDOW_FORK_SCORE
    else:
        fork_threshold = np.nan

    reliable_fork = bool(
        run_length >= 2
        and fork_score >= fork_threshold
        and coverage >= coverage_candidate
    )
    if reliable_fork and run_length >= 3 and coverage >= coverage_branch:
        status = "Branch"
    elif reliable_fork:
        status = "Candidate"
    else:
        status = "No branch"

    return {
        "status": status,
        "bandwidth_factor": float(factor),
        "run_length": int(run_length),
        "x_start": x_start,
        "x_end": x_end,
        "x_coverage": coverage,
        "relative_opening": relative_opening,
        "fork_score": fork_score,
        "fork_score_threshold": fork_threshold,
        "median_relative_mode_separation": median_relative_separation,
    }


def _vote(results: list[dict[str, Any]]) -> tuple[str, int, int, float]:
    branch_votes = sum(item["status"] == "Branch" for item in results)
    support_votes = sum(
        item["status"] in {"Branch", "Candidate"} for item in results
    )
    supported_separations = [
        float(item["median_relative_mode_separation"])
        for item in results
        if item["status"] in {"Branch", "Candidate"}
        and np.isfinite(item["median_relative_mode_separation"])
    ]
    median_separation = (
        float(np.median(supported_separations))
        if supported_separations else 0.0
    )

    if branch_votes >= MIN_BRANCH_VOTES:
        status = "Branch"
    elif (
        support_votes >= MIN_CANDIDATE_SUPPORT_VOTES
        and branch_votes >= MIN_CANDIDATE_BRANCH_VOTES
        and median_separation >= MIN_CANDIDATE_RELATIVE_MODE_SEPARATION
    ):
        status = "Candidate"
    else:
        status = "No branch"
    return status, branch_votes, support_votes, median_separation


def classify_from_fitted(
    fitted: dict[str, Any],
    coverage_mode: str = "x_range",
) -> dict[str, Any]:
    """Apply the inexpensive geometry and five-bandwidth voting stage."""
    if coverage_mode not in {"x_range", "point_count"}:
        raise ValueError("coverage_mode must be 'x_range' or 'point_count'")

    x = np.asarray(fitted["x"], dtype=float)
    y = np.asarray(fitted["y"], dtype=float)
    if len(x) < DEFAULT_MIN_N:
        return {
            "group": "Uncertain",
            "status": "insufficient_data",
            "bandwidth_factors": BANDWIDTH_FACTORS,
        }

    cached_factors = tuple(float(value) for value in fitted.get(
        "bandwidth_factors", fitted["bandwidth_points"].keys(),
    ))
    if set(cached_factors) != set(BANDWIDTH_FACTORS):
        raise ValueError(
            "Fitted KDE bandwidths do not match the five-bandwidth classifier: "
            f"cached={cached_factors}, expected={BANDWIDTH_FACTORS}"
        )

    results = []
    for factor in BANDWIDTH_FACTORS:
        points = _apply_local_gate(fitted["bandwidth_points"].get(factor, []))
        results.append(_classify_one_bandwidth(
            points, x, y, coverage_mode, factor,
        ))

    status, branch_votes, support_votes, median_separation = _vote(results)
    priority = {"No branch": 0, "Candidate": 1, "Branch": 2}
    selected = max(
        results,
        key=lambda item: (
            priority[item["status"]],
            item["run_length"],
            item["x_coverage"],
            item["fork_score"],
            -abs(item["bandwidth_factor"] - 1.0),
        ),
    )

    output = dict(selected)
    output.update({
        "group": STATUS_TO_GROUP[status],
        "status": status,
        "branch_type": {
            "Branch": "five-bandwidth branch consensus",
            "Candidate": "two-bandwidth candidate support",
            "No branch": "none",
        }[status],
        "classifier_version": CLASSIFIER_VERSION,
        "bandwidth_factors": BANDWIDTH_FACTORS,
        "bandwidth_vote_pattern": "|".join(
            item["status"] for item in results
        ),
        "bandwidth_branch_votes": int(branch_votes),
        "bandwidth_support_votes": int(support_votes),
        "bandwidth_support_median_relative_mode_separation": float(
            median_separation
        ),
        "bandwidth_vote_threshold": MIN_BRANCH_VOTES,
        "candidate_support_vote_threshold": MIN_CANDIDATE_SUPPORT_VOTES,
        "candidate_branch_vote_threshold": MIN_CANDIDATE_BRANCH_VOTES,
        "candidate_relative_mode_separation_threshold": (
            MIN_CANDIDATE_RELATIVE_MODE_SEPARATION
        ),
        "high_confidence_branch": bool(
            status == "Branch" and branch_votes == len(BANDWIDTH_FACTORS)
        ),
        "selected_bandwidth_factor": float(selected["bandwidth_factor"]),
    })
    for factor, item in zip(BANDWIDTH_FACTORS, results):
        output[f"bandwidth_{factor:.2f}_status"] = item["status"]
    return output


def detect_branch(
    x: np.ndarray,
    y: np.ndarray,
    *,
    coverage_mode: str = "x_range",
    min_window: int = DEFAULT_MIN_WINDOW,
    min_n: int = DEFAULT_MIN_N,
) -> dict[str, Any]:
    """Fit and classify one x-y panel."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < min_n:
        return {
            "group": "Uncertain",
            "status": "insufficient_data",
            "classifier_version": CLASSIFIER_VERSION,
            "bandwidth_factors": BANDWIDTH_FACTORS,
        }
    return classify_from_fitted(
        fit_kde(x, y, min_window=min_window),
        coverage_mode=coverage_mode,
    )
