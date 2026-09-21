"""Comparable branch detectors for static x-y scatter plots.

The functions in this module detect statistical/geometric branches only.  They
do not claim a dynamical bifurcation.  Every detector returns the same compact
schema so that S4 can compare methods without modifying the production
classifier in ``classifier.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import numpy as np
from scipy.signal import find_peaks
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import gaussian_kde, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

try:
    from diptest import diptest
except ImportError:  # pragma: no cover - reported in the result table
    diptest = None


@dataclass
class BranchResult:
    method: str
    branch_detected: bool
    n_branches: int
    score: float
    x_start: float = np.nan
    x_end: float = np.nan
    x_coverage: float = 0.0
    persistence: int = 0
    min_branch_weight: float = np.nan
    branch_separation: float = np.nan
    notes: str = ""
    geometry: dict[str, Any] | None = None

    def to_dict(self, include_geometry: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if not include_geometry:
            result.pop("geometry", None)
        return result


def _clean_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(xv) & np.isfinite(yv)
    return xv[keep], yv[keep]


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    q05, q95 = np.quantile(values, [0.05, 0.95])
    # IQR alone collapses on zero-inflated variables (as in land-ice runoff).
    # The central 90% range keeps the upper branch on a meaningful scale while
    # remaining much less sensitive to extremes than the standard deviation.
    scale = float(max(q75 - q25, (q95 - q05) / 3.0))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def _standardize_xy(x, y):
    xc, xs = _robust_location_scale(x)
    yc, ys = _robust_location_scale(y)
    return (x - xc) / xs, (y - yc) / ys, (xc, xs, yc, ys)


def _window_slices(n: int, window_frac=0.07, step_frac=0.5, min_size=40):
    if n < 40:
        return []
    width = min(n, max(min_size, int(round(window_frac * n))))
    step = max(1, int(round(step_frac * width)))
    starts = list(range(0, max(n - width + 1, 1), step))
    final_start = max(0, n - width)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + width) for start in starts]


def _longest_true_run(flags: list[bool]) -> tuple[int, int, int]:
    best_start, best_end, best_len = -1, -1, 0
    current_start = -1
    for i, flag in enumerate(flags + [False]):
        if flag and current_start < 0:
            current_start = i
        elif not flag and current_start >= 0:
            run_len = i - current_start
            if run_len > best_len:
                best_start, best_end, best_len = current_start, i - 1, run_len
            current_start = -1
    return best_start, best_end, best_len


def _result_from_window_flags(
    method: str,
    windows: list[dict[str, Any]],
    *,
    min_run: int = 3,
    min_coverage: float = 0.08,
    notes: str = "",
) -> BranchResult:
    if not windows:
        return BranchResult(method, False, 1, 0.0, notes="insufficient data")
    flags = [bool(item["flag"]) for item in windows]
    start, end, run_len = _longest_true_run(flags)
    full_min = min(item["x_min"] for item in windows)
    full_max = max(item["x_max"] for item in windows)
    full_range = max(full_max - full_min, 1e-12)
    if run_len:
        x_start = windows[start]["x_min"]
        x_end = windows[end]["x_max"]
        coverage = float((x_end - x_start) / full_range)
        selected = windows[start : end + 1]
        min_weight = float(np.nanmedian([w.get("min_weight", np.nan) for w in selected]))
        separation = float(np.nanmedian([w.get("separation", np.nan) for w in selected]))
        n_branches = int(max(w.get("n_modes", 2) for w in selected))
    else:
        x_start = x_end = np.nan
        coverage = 0.0
        min_weight = separation = np.nan
        n_branches = 1
    detected = bool(run_len >= min_run and coverage >= min_coverage)
    score = float(run_len / max(len(windows), 1))
    geometry = {
        "windows": windows,
        "longest_run": (start, end),
    }
    return BranchResult(
        method=method,
        branch_detected=detected,
        n_branches=n_branches if detected else 1,
        score=score,
        x_start=float(x_start),
        x_end=float(x_end),
        x_coverage=coverage,
        persistence=run_len,
        min_branch_weight=min_weight,
        branch_separation=separation,
        notes=notes,
        geometry=geometry,
    )


def detect_current_dip(x, y, n_bins=10, alpha=0.05, fraction_threshold=0.30):
    """Reproduce the current S4 conditional Dip branch gate."""
    xv, yv = _clean_xy(x, y)
    if diptest is None or len(xv) < 30:
        return BranchResult("Current conditional Dip", False, 1, 0.0, notes="diptest unavailable or insufficient data")
    edges = np.quantile(xv, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-10
    bin_id = np.digitize(xv, edges[1:-1])
    windows = []
    significant = 0
    for b in range(n_bins):
        mask = bin_id == b
        if mask.sum() < 10:
            continue
        pvalue = float(diptest(yv[mask])[1])
        flag = pvalue < alpha
        significant += int(flag)
        windows.append({
            "x_min": float(xv[mask].min()),
            "x_max": float(xv[mask].max()),
            "x_mid": float(np.median(xv[mask])),
            "flag": flag,
            "pvalue": pvalue,
            "n_modes": 2 if flag else 1,
        })
    fraction = significant / n_bins
    result = BranchResult(
        method="Current conditional Dip",
        branch_detected=bool(fraction >= fraction_threshold),
        n_branches=2 if fraction >= fraction_threshold else 1,
        score=float(fraction),
        persistence=_longest_true_run([w["flag"] for w in windows])[2],
        notes=f"significant-bin fraction={fraction:.3f}; threshold={fraction_threshold:.2f}",
        geometry={"windows": windows, "longest_run": (-1, -1)},
    )
    return result


def _kde_mode_pair(
    y_scaled: np.ndarray,
    bandwidth_factor: float,
    *,
    min_weight=0.08,
    min_valley_depth=0.15,
    min_separation=0.80,
):
    lo, hi = np.quantile(y_scaled, [0.005, 0.995])
    values = y_scaled[(y_scaled >= lo) & (y_scaled <= hi)]
    if len(values) < 30 or hi - lo <= 1e-10:
        return None
    try:
        kde = gaussian_kde(
            values,
            bw_method=lambda obj: obj.scotts_factor() * bandwidth_factor,
        )
    except np.linalg.LinAlgError:
        return None
    grid = np.linspace(lo, hi, 500)
    density = kde(grid)
    peaks = list(find_peaks(
        density,
        prominence=max(0.025 * density.max(), 1e-12),
        distance=10,
    )[0])
    if density[0] > density[1] and density[0] >= 0.05 * density.max():
        peaks.insert(0, 0)
    if density[-1] > density[-2] and density[-1] >= 0.05 * density.max():
        peaks.append(len(grid) - 1)
    peaks = sorted(set(peaks))
    candidates = []
    for left_i in range(len(peaks) - 1):
        for right_i in range(left_i + 1, len(peaks)):
            left, right = peaks[left_i], peaks[right_i]
            valley = left + int(np.argmin(density[left : right + 1]))
            low_peak = min(density[left], density[right])
            valley_depth = 1.0 - density[valley] / max(low_peak, 1e-12)
            cut = grid[valley]
            weight = min(np.mean(values <= cut), np.mean(values > cut))
            separation = float(grid[right] - grid[left])
            valid = (
                weight >= min_weight
                and valley_depth >= min_valley_depth
                and separation >= min_separation
            )
            candidates.append({
                "valid": valid,
                "modes": (float(grid[left]), float(grid[right])),
                "valley_depth": float(valley_depth),
                "min_weight": float(weight),
                "separation": separation,
                "strength": float(low_peak),
            })
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["valid"], item["strength"]))


def detect_modal_kde(
    x,
    y,
    *,
    window_frac=0.06,
    bandwidth_factors=(0.75, 1.0, 1.25),
    bandwidth_support=2,
):
    """Sliding conditional KDE modes with multi-bandwidth persistence."""
    xv, yv = _clean_xy(x, y)
    order = np.argsort(xv)
    xv, yv = xv[order], yv[order]
    _, _, (_, _, yc, ys) = _standardize_xy(xv, yv)
    windows = []
    for start, stop in _window_slices(len(xv), window_frac=window_frac):
        xw, yw = xv[start:stop], yv[start:stop]
        y_scaled = (yw - yc) / ys
        candidates = [
            _kde_mode_pair(y_scaled, factor)
            for factor in bandwidth_factors
        ]
        valid = [item for item in candidates if item is not None and item["valid"]]
        flag = len(valid) >= bandwidth_support
        if valid:
            mode_low = float(np.median([item["modes"][0] for item in valid]) * ys + yc)
            mode_high = float(np.median([item["modes"][1] for item in valid]) * ys + yc)
            min_weight = float(np.median([item["min_weight"] for item in valid]))
            separation = float(np.median([item["separation"] for item in valid]))
            valley_depth = float(np.median([item["valley_depth"] for item in valid]))
        else:
            mode_low = mode_high = min_weight = separation = valley_depth = np.nan
        windows.append({
            "x_min": float(xw.min()),
            "x_max": float(xw.max()),
            "x_mid": float(np.median(xw)),
            "flag": flag,
            "n_modes": 2 if flag else 1,
            "modes": (mode_low, mode_high),
            "min_weight": min_weight,
            "separation": separation,
            "valley_depth": valley_depth,
            "bandwidth_support": len(valid),
        })
    return _result_from_window_flags(
        "Sliding KDE modal",
        windows,
        notes=(
            f"window={window_frac:.1%} of n; "
            f"requires {bandwidth_support}/{len(bandwidth_factors)} bandwidths"
        ),
    )


def detect_sliding_gmm(
    x,
    y,
    *,
    window_frac=0.07,
    max_components=3,
    min_weight=0.10,
    min_separation=2.0,
    min_delta_bic=10.0,
):
    xv, yv = _clean_xy(x, y)
    order = np.argsort(xv)
    xv, yv = xv[order], yv[order]
    _, _, (_, _, yc, ys) = _standardize_xy(xv, yv)
    windows = []
    for start, stop in _window_slices(len(xv), window_frac=window_frac):
        xw, yw = xv[start:stop], yv[start:stop]
        values = ((yw - yc) / ys)[:, None]
        fits = []
        for k in range(1, max_components + 1):
            try:
                model = GaussianMixture(k, n_init=5, random_state=17).fit(values)
                fits.append((k, model.bic(values), model))
            except ValueError:
                continue
        if not fits:
            continue
        best_k, best_bic, best = min(fits, key=lambda item: item[1])
        one_bic = next(item[1] for item in fits if item[0] == 1)
        means = best.means_.ravel()
        sds = np.sqrt(best.covariances_.reshape(-1))
        weights = best.weights_.ravel()
        order_k = np.argsort(means)
        means, sds, weights = means[order_k], sds[order_k], weights[order_k]
        adjacent_sep = []
        if best_k >= 2:
            for i in range(best_k - 1):
                pooled = np.sqrt((sds[i] ** 2 + sds[i + 1] ** 2) / 2)
                adjacent_sep.append((means[i + 1] - means[i]) / max(pooled, 1e-12))
        effective = weights >= min_weight
        n_effective = int(effective.sum())
        separation = float(max(adjacent_sep)) if adjacent_sep else np.nan
        flag = bool(
            best_k >= 2
            and one_bic - best_bic >= min_delta_bic
            and n_effective >= 2
            and separation >= min_separation
        )
        original_modes = tuple((means * ys + yc).tolist())
        windows.append({
            "x_min": float(xw.min()),
            "x_max": float(xw.max()),
            "x_mid": float(np.median(xw)),
            "flag": flag,
            "n_modes": n_effective if flag else 1,
            "modes": original_modes,
            "min_weight": float(weights[effective].min()) if effective.any() else np.nan,
            "separation": separation,
            "delta_bic": float(one_bic - best_bic),
        })
    return _result_from_window_flags(
        "Sliding-window GMM",
        windows,
        notes=f"window={window_frac:.1%} of n; BIC + weight + separation filters",
    )


def _mixture_regression_fit(x_scaled, y_scaled, n_components, degree, seed):
    rng = np.random.default_rng(seed)
    design = np.column_stack([x_scaled ** power for power in range(degree + 1)])
    n = len(y_scaled)
    if n_components == 1:
        responsibility = np.ones((n, 1))
    else:
        feature = np.column_stack([x_scaled, y_scaled])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            labels = MiniBatchKMeans(
                n_clusters=n_components,
                n_init=1,
                random_state=seed,
                batch_size=min(1024, n),
            ).fit_predict(feature)
        responsibility = np.full((n, n_components), 0.02 / (n_components - 1))
        responsibility[np.arange(n), labels] = 0.98
        responsibility += rng.uniform(0, 1e-5, responsibility.shape)
        responsibility /= responsibility.sum(axis=1, keepdims=True)
    previous = -np.inf
    for _ in range(400):
        nk = responsibility.sum(axis=0) + 1e-9
        mixing = nk / nk.sum()
        coefficients = []
        sigmas = []
        for k in range(n_components):
            weights = responsibility[:, k]
            lhs = design.T @ (weights[:, None] * design) + 1e-7 * np.eye(design.shape[1])
            rhs = design.T @ (weights * y_scaled)
            beta = np.linalg.solve(lhs, rhs)
            residual = y_scaled - design @ beta
            sigma = max(np.sqrt(np.sum(weights * residual ** 2) / nk[k]), 0.02)
            coefficients.append(beta)
            sigmas.append(sigma)
        coefficients = np.asarray(coefficients)
        sigmas = np.asarray(sigmas)
        log_prob = np.empty_like(responsibility)
        for k in range(n_components):
            residual = (y_scaled - design @ coefficients[k]) / sigmas[k]
            log_prob[:, k] = (
                np.log(mixing[k] + 1e-300)
                - np.log(sigmas[k])
                - 0.5 * residual ** 2
                - 0.5 * np.log(2 * np.pi)
            )
        max_log = log_prob.max(axis=1, keepdims=True)
        normalizer = max_log + np.log(np.exp(log_prob - max_log).sum(axis=1, keepdims=True))
        log_likelihood = float(normalizer.sum())
        responsibility = np.exp(log_prob - normalizer)
        if abs(log_likelihood - previous) < 1e-7 * (1 + abs(log_likelihood)):
            break
        previous = log_likelihood
    n_parameters = n_components * (design.shape[1] + 1) + (n_components - 1)
    bic = -2 * log_likelihood + n_parameters * np.log(n)
    return {
        "log_likelihood": log_likelihood,
        "bic": float(bic),
        "mixing": mixing,
        "coefficients": coefficients,
        "sigmas": sigmas,
        "responsibility": responsibility,
        "degree": degree,
    }


def detect_mixture_regression(
    x,
    y,
    *,
    max_components=3,
    degree=1,
    min_weight=0.08,
    min_delta_bic=10.0,
):
    xv, yv = _clean_xy(x, y)
    if len(xv) < 80:
        return BranchResult("Mixture of regressions", False, 1, 0.0, notes="insufficient data")
    x_scaled, y_scaled, scale = _standardize_xy(xv, yv)
    xc, xs, yc, ys = scale
    fits = {}
    for k in range(1, max_components + 1):
        starts = [_mixture_regression_fit(x_scaled, y_scaled, k, degree, seed) for seed in range(6)]
        fits[k] = min(starts, key=lambda item: item["bic"])
    best_k = min(fits, key=lambda k: fits[k]["bic"])
    best = fits[best_k]
    delta_bic = float(fits[1]["bic"] - best["bic"])
    effective = best["mixing"] >= min_weight
    n_effective = int(effective.sum())
    grid_original = np.linspace(np.quantile(xv, 0.01), np.quantile(xv, 0.99), 160)
    grid_scaled = (grid_original - xc) / xs
    design_grid = np.column_stack([grid_scaled ** power for power in range(degree + 1)])
    predictions_scaled = design_grid @ best["coefficients"].T
    predictions = predictions_scaled * ys + yc
    effective_predictions = predictions[:, effective]
    if n_effective >= 2:
        ordered = np.sort(effective_predictions, axis=1)
        separation = float(np.median(np.max(np.diff(ordered, axis=1), axis=1) / ys))
    else:
        separation = np.nan
    detected = bool(
        best_k >= 2
        and n_effective >= 2
        and delta_bic >= min_delta_bic
        and separation >= 0.80
    )
    geometry = {
        "x_grid": grid_original,
        "predictions": predictions,
        "effective": effective,
        "weights": best["mixing"],
        "bics": {k: fits[k]["bic"] for k in fits},
        "responsibility": best["responsibility"],
    }
    return BranchResult(
        method="Mixture of regressions",
        branch_detected=detected,
        n_branches=n_effective if detected else 1,
        score=delta_bic,
        x_start=float(grid_original.min()) if detected else np.nan,
        x_end=float(grid_original.max()) if detected else np.nan,
        x_coverage=0.98 if detected else 0.0,
        persistence=1 if detected else 0,
        min_branch_weight=float(best["mixing"][effective].min()) if effective.any() else np.nan,
        branch_separation=separation,
        notes=f"selected K={best_k}; delta-BIC={delta_bic:.1f}; polynomial degree={degree}",
        geometry=geometry,
    )


def detect_density_ridge(
    x,
    y,
    *,
    n_x_grid=60,
    n_y_grid=140,
    bandwidth_factor=0.75,
):
    """Extract vertical ridges from a 2-D KDE on a regular grid.

    This is an x-oriented density-ridge approximation: at each x-grid column,
    local maxima along y are retained and linked by persistence.  It is useful
    when x is the control axis and avoids an optional SCMS dependency.
    """
    xv, yv = _clean_xy(x, y)
    if len(xv) < 80:
        return BranchResult("2-D KDE density ridge", False, 1, 0.0, notes="insufficient data")
    xs, ys, scale = _standardize_xy(xv, yv)
    xc, xscale, yc, yscale = scale
    keep = (
        (xs >= np.quantile(xs, 0.005)) & (xs <= np.quantile(xs, 0.995))
        & (ys >= np.quantile(ys, 0.005)) & (ys <= np.quantile(ys, 0.995))
    )
    sample = np.vstack([xs[keep], ys[keep]])
    try:
        kde = gaussian_kde(
            sample,
            bw_method=lambda obj: obj.scotts_factor() * bandwidth_factor,
        )
    except np.linalg.LinAlgError:
        return BranchResult("2-D KDE density ridge", False, 1, 0.0, notes="singular KDE")
    x_grid = np.linspace(np.quantile(xs, 0.01), np.quantile(xs, 0.99), n_x_grid)
    y_grid = np.linspace(np.quantile(ys, 0.005), np.quantile(ys, 0.995), n_y_grid)
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    windows = []
    for j, x_position in enumerate(x_grid):
        column = density[:, j]
        peaks = list(find_peaks(
            column,
            prominence=max(0.025 * column.max(), 1e-12),
            distance=5,
        )[0])
        if column[0] > column[1] and column[0] >= 0.05 * column.max():
            peaks.insert(0, 0)
        peaks = sorted(set(peaks), key=lambda idx: column[idx], reverse=True)[:3]
        peaks = sorted(peaks)
        flag = False
        separation = min_weight = np.nan
        if len(peaks) >= 2:
            best_pair = None
            for left_i in range(len(peaks) - 1):
                for right_i in range(left_i + 1, len(peaks)):
                    left, right = peaks[left_i], peaks[right_i]
                    valley = left + int(np.argmin(column[left : right + 1]))
                    depth = 1 - column[valley] / max(min(column[left], column[right]), 1e-12)
                    sep = y_grid[right] - y_grid[left]
                    relative_height = min(column[left], column[right]) / max(column.max(), 1e-12)
                    candidate = (depth >= 0.12 and sep >= 0.65 and relative_height >= 0.04, depth, sep, relative_height, left, right)
                    if best_pair is None or candidate[:4] > best_pair[:4]:
                        best_pair = candidate
            if best_pair is not None:
                flag, _, separation, min_weight, left, right = best_pair
                peaks = [left, right]
        ridge_y = tuple((y_grid[peaks] * yscale + yc).tolist()) if peaks else tuple()
        x_original = x_position * xscale + xc
        windows.append({
            "x_min": float(x_original),
            "x_max": float(x_original),
            "x_mid": float(x_original),
            "flag": bool(flag),
            "n_modes": 2 if flag else 1,
            "modes": ridge_y,
            "min_weight": float(min_weight),
            "separation": float(separation),
        })
    result = _result_from_window_flags(
        "2-D KDE density ridge",
        windows,
        min_run=4,
        min_coverage=0.08,
        notes=f"vertical KDE ridges; bandwidth factor={bandwidth_factor:.2f}",
    )
    result.geometry.update({
        "x_grid": x_grid * xscale + xc,
        "y_grid": y_grid * yscale + yc,
        "density": density,
    })
    return result


def _persistent_flag_cluster(
    flags: list[bool],
    *,
    max_gap: int = 2,
    min_density: float = 0.55,
    min_coverage: float = 0.18,
) -> dict[str, Any]:
    """Find a dense run of positive windows while tolerating short holes."""
    indices = np.flatnonzero(flags)
    empty = {
        "detected": False, "start": -1, "end": -1,
        "count": 0, "span": 0, "density": 0.0,
    }
    if len(indices) == 0:
        return empty
    groups = []
    start = 0
    for stop in range(1, len(indices) + 1):
        split = (
            stop == len(indices)
            or indices[stop] - indices[stop - 1] > max_gap + 1
        )
        if split:
            group = indices[start:stop]
            span = int(group[-1] - group[0] + 1)
            groups.append({
                "start": int(group[0]),
                "end": int(group[-1]),
                "count": int(len(group)),
                "span": span,
                "density": float(len(group) / span),
            })
            start = stop
    best = max(groups, key=lambda item: (item["count"], item["span"], item["density"]))
    need_span = max(3, int(np.ceil(min_coverage * len(flags))))
    best["detected"] = bool(
        best["count"] >= 3
        and best["span"] >= need_span
        and best["density"] >= min_density
    )
    return best


def _hexbin_resolution(
    x: np.ndarray,
    y: np.ndarray,
    gridsize: int,
    *,
    x_window_mode: str,
    min_valley_depth: float,
    min_branch_weight: float,
    min_relative_peak: float,
    min_separation: float,
    min_compactness: float,
    min_cluster_mass: float,
    max_path_roughness: float,
) -> dict[str, Any]:
    """Extract conditional modal pairs from the counts of one hexbin grid."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(1, 1))
    axis = figure.subplots()
    collection = axis.hexbin(x, y, gridsize=gridsize, mincnt=1)
    centers = np.asarray(collection.get_offsets(), dtype=float)
    counts = np.asarray(collection.get_array(), dtype=float)
    if len(centers) == 0:
        return {
            "gridsize": gridsize, "detected": False, "windows": [],
            "cluster": _persistent_flag_cluster([]),
        }

    target_x_windows = max(9, gridsize // 2)
    n_y_bins = max(9, int(round(gridsize / np.sqrt(3))))
    x_windows = None
    if x_window_mode == "equal_width":
        x_edges = np.linspace(
            centers[:, 0].min(), centers[:, 0].max(), target_x_windows + 1,
        )
    elif x_window_mode in {"equal_frequency", "equal_frequency_overlap"}:
        # Collapse the 2-D Hexbin cells to x-columns, then place boundaries
        # where cumulative Hexbin mass crosses equal-count targets.  A large
        # atom (for example x=0) is kept intact, so duplicate quantile edges
        # cannot create empty windows.
        unique_x, inverse = np.unique(centers[:, 0], return_inverse=True)
        column_mass = np.bincount(inverse, weights=counts)
        cumulative = np.cumsum(column_mass)
        span = max(float(unique_x[-1] - unique_x[0]), 1.0)
        epsilon = np.finfo(float).eps * span * 16
        if x_window_mode == "equal_frequency":
            targets = np.linspace(0.0, cumulative[-1], target_x_windows + 1)[1:-1]
            cut_indices = np.searchsorted(cumulative, targets, side="left")
            cut_indices = np.unique(cut_indices[(cut_indices > 0) & (cut_indices < len(unique_x))])
            internal_edges = (unique_x[cut_indices - 1] + unique_x[cut_indices]) / 2
            x_edges = np.concatenate((
                [unique_x[0] - epsilon], internal_edges, [unique_x[-1] + epsilon],
            ))
        else:
            # Equal-mass sliding windows with 50% overlap.  Overlap lets a
            # localized branch occupy several consecutive windows without
            # giving sparse tails smaller sample support than the dense core.
            cumulative_mid = (cumulative - 0.5 * column_mass) / cumulative[-1]
            quantile_width = 1.0 / target_x_windows
            quantile_step = quantile_width / 2.0
            starts = np.arange(
                0.0, 1.0 - quantile_width + quantile_step / 2, quantile_step,
            )
            x_windows = []
            for start_q in starts:
                stop_q = min(start_q + quantile_width, 1.0)
                left = float(np.interp(start_q, cumulative_mid, unique_x))
                right = float(np.interp(stop_q, cumulative_mid, unique_x))
                if not x_windows or (left, right) != x_windows[-1]:
                    x_windows.append((left - epsilon, right + epsilon))
    else:
        raise ValueError(
            "x_window_mode must be 'equal_width', 'equal_frequency', or "
            f"'equal_frequency_overlap', got {x_window_mode!r}"
        )
    if x_windows is None:
        x_windows = list(zip(x_edges[:-1], x_edges[1:]))
    n_x_windows = len(x_windows)
    y_edges = np.linspace(centers[:, 1].min(), centers[:, 1].max(), n_y_bins + 1)
    y_mid = (y_edges[:-1] + y_edges[1:]) / 2
    y_step = max(y_edges[1] - y_edges[0], 1e-12)
    windows = []

    for j, (x_left, x_right) in enumerate(x_windows):
        if x_window_mode == "equal_frequency_overlap" or j + 1 == n_x_windows:
            mask = (centers[:, 0] >= x_left) & (centers[:, 0] <= x_right)
        else:
            mask = (centers[:, 0] >= x_left) & (centers[:, 0] < x_right)
        histogram, _ = np.histogram(
            centers[mask, 1], bins=y_edges, weights=counts[mask]
        )
        histogram = histogram.astype(float)
        total = float(histogram.sum())
        best = None
        primary_mode = np.nan
        peak_modes = tuple()
        if total >= 20 and np.count_nonzero(histogram) >= 2:
            smoothed = np.convolve(histogram, [0.2, 0.6, 0.2], mode="same")
            prominence = max(1.0, 0.04 * float(smoothed.max()))
            peaks = list(find_peaks(smoothed, prominence=prominence, distance=2)[0])
            if smoothed[0] >= smoothed[1] and smoothed[0] >= prominence:
                peaks.insert(0, 0)
            if smoothed[-1] >= smoothed[-2] and smoothed[-1] >= prominence:
                peaks.append(len(smoothed) - 1)
            peaks = sorted(set(peaks))
            primary_mode = float(y_mid[int(np.argmax(smoothed))])
            peak_modes = tuple(float(y_mid[index]) for index in peaks)

            for left_i in range(len(peaks) - 1):
                for right_i in range(left_i + 1, len(peaks)):
                    left, right = peaks[left_i], peaks[right_i]
                    if right - left < 2:
                        continue
                    valley = left + int(np.argmin(smoothed[left : right + 1]))
                    low_peak = min(smoothed[left], smoothed[right])
                    valley_depth = 1.0 - smoothed[valley] / max(low_peak, 1e-12)
                    left_weight = histogram[: valley + 1]
                    right_weight = histogram[valley + 1 :]
                    min_weight = min(left_weight.sum(), right_weight.sum()) / total
                    relative_peak = low_peak / max(float(smoothed.max()), 1e-12)
                    separation = (right - left) / len(histogram)

                    left_y = y_mid[: valley + 1]
                    right_y = y_mid[valley + 1 :]
                    left_mean = float(np.average(left_y, weights=left_weight))
                    right_mean = float(np.average(right_y, weights=right_weight))
                    left_sd = float(np.sqrt(np.average((left_y - left_mean) ** 2, weights=left_weight)))
                    right_sd = float(np.sqrt(np.average((right_y - right_mean) ** 2, weights=right_weight)))
                    compactness = (right_mean - left_mean) / max(left_sd, right_sd, y_step / 2)
                    pooled_width = float(
                        np.sqrt((left_sd ** 2 + right_sd ** 2) / 2)
                    )
                    valid = bool(
                        valley_depth >= min_valley_depth
                        and min_weight >= min_branch_weight
                        and relative_peak >= min_relative_peak
                        and separation >= min_separation
                        and compactness >= min_compactness
                    )
                    strength = valley_depth * min_weight * relative_peak * separation
                    candidate = {
                        "valid": valid,
                        "strength": float(strength),
                        "modes": (float(y_mid[left]), float(y_mid[right])),
                        "valley_y": float(y_mid[valley]),
                        "valley_depth": float(valley_depth),
                        "min_weight": float(min_weight),
                        "relative_peak": float(relative_peak),
                        "separation": float(separation),
                        "separation_y": float(y_mid[right] - y_mid[left]),
                        "compactness": float(compactness),
                        "left_sd": left_sd,
                        "right_sd": right_sd,
                        "pooled_width": max(pooled_width, y_step / 2),
                    }
                    if best is None or (candidate["valid"], candidate["strength"]) > (best["valid"], best["strength"]):
                        best = candidate

        windows.append({
            "x_min": float(x_left),
            "x_max": float(x_right),
            "x_mid": float((x_left + x_right) / 2),
            "flag": bool(best is not None and best["valid"]),
            "n_modes": 2 if best is not None and best["valid"] else 1,
            "total_count": int(total),
            "primary_mode": primary_mode,
            "peak_modes": peak_modes,
            "y_step": float(y_step),
            **({} if best is None else best),
        })

    cluster = _persistent_flag_cluster([window["flag"] for window in windows])
    cluster["base_detected"] = bool(cluster["detected"])
    selected = []
    if cluster["start"] >= 0:
        selected = [
            window
            for window in windows[cluster["start"] : cluster["end"] + 1]
            if window["flag"]
        ]
    total_available = max(sum(window["total_count"] for window in windows), 1)
    cluster_mass = sum(window["total_count"] for window in selected) / total_available
    path_roughness = []
    for mode_index in (0, 1):
        mode_values = np.asarray([
            window["modes"][mode_index] for window in selected
        ], dtype=float)
        if len(mode_values) < 2:
            roughness = np.inf
        else:
            differences = np.diff(mode_values)
            roughness = float(
                np.sum(np.abs(differences))
                / max(abs(mode_values[-1] - mode_values[0]), y_step)
            )
        path_roughness.append(roughness)
    cluster["mass_fraction"] = float(cluster_mass)
    cluster["path_roughness"] = tuple(path_roughness)
    cluster["detected"] = bool(
        cluster["base_detected"]
        and cluster_mass >= min_cluster_mass
        and max(path_roughness, default=np.inf) <= max_path_roughness
    )
    return {
        "gridsize": int(gridsize),
        "detected": bool(cluster["detected"]),
        "windows": windows,
        "cluster": cluster,
        "hex_centers": centers,
        "hex_counts": counts,
    }


def detect_multires_hexbin(
    x,
    y,
    *,
    gridsizes=(16, 22, 28, 34, 40, 46),
    required_support=3,
    x_window_mode="equal_width",
    min_valley_depth=0.35,
    min_branch_weight=0.08,
    min_relative_peak=0.15,
    min_separation=0.12,
    min_compactness=1.5,
    min_cluster_mass=0.0,
    max_path_roughness=np.inf,
):
    """Detect persistent conditional branches in multi-resolution hexbin counts.

    A resolution votes for a branch only when two local count peaks persist in
    nearby x windows and pass valley, mass, peak-height, separation, band-
    compactness, cluster-support, and path-smoothness filters. Requiring
    agreement across resolutions suppresses grid-size artifacts, sparse tails,
    and visually broad one-band fans.
    """
    xv, yv = _clean_xy(x, y)
    if len(xv) < 80:
        return BranchResult("Multi-resolution Hexbin", False, 1, 0.0, notes="insufficient data")
    resolutions = [
        _hexbin_resolution(
            xv, yv, int(gridsize),
            x_window_mode=x_window_mode,
            min_valley_depth=min_valley_depth,
            min_branch_weight=min_branch_weight,
            min_relative_peak=min_relative_peak,
            min_separation=min_separation,
            min_compactness=min_compactness,
            min_cluster_mass=min_cluster_mass,
            max_path_roughness=max_path_roughness,
        )
        for gridsize in gridsizes
    ]
    supported = [resolution for resolution in resolutions if resolution["detected"]]
    detected = len(supported) >= required_support
    selected_windows = []
    x_starts, x_ends = [], []
    for resolution in supported:
        cluster = resolution["cluster"]
        windows = resolution["windows"]
        if cluster["start"] >= 0:
            x_starts.append(windows[cluster["start"]]["x_min"])
            x_ends.append(windows[cluster["end"]]["x_max"])
            selected_windows.extend(
                window for window in windows[cluster["start"] : cluster["end"] + 1]
                if window["flag"]
            )
    full_range = max(float(np.max(xv) - np.min(xv)), 1e-12)
    x_start = float(np.median(x_starts)) if detected else np.nan
    x_end = float(np.median(x_ends)) if detected else np.nan
    min_weight_value = float(np.median([w["min_weight"] for w in selected_windows])) if selected_windows else np.nan
    compactness_value = float(np.median([w["compactness"] for w in selected_windows])) if selected_windows else np.nan
    return BranchResult(
        method="Multi-resolution Hexbin",
        branch_detected=detected,
        n_branches=2 if detected else 1,
        score=float(len(supported) / len(resolutions)),
        x_start=x_start,
        x_end=x_end,
        x_coverage=float((x_end - x_start) / full_range) if detected else 0.0,
        persistence=max((r["cluster"]["count"] for r in supported), default=0),
        min_branch_weight=min_weight_value,
        branch_separation=compactness_value,
        notes=(
            f"resolution support={len(supported)}/{len(resolutions)}; "
            f"requires {required_support}; gridsizes={tuple(gridsizes)}; "
            f"x windows={x_window_mode}; "
            f"cluster mass>={min_cluster_mass:.1%}; "
            f"path roughness<={max_path_roughness:.2f}"
        ),
        geometry={"resolutions": resolutions, "required_support": required_support},
    )


def detect_multiextent_hexbin(
    x,
    y,
    *,
    extent_quantiles=(0.0, 0.0025),
    gridsizes=(16, 22, 28, 34, 40, 46),
    required_support=3,
    **detector_kwargs,
):
    """Run the Hexbin detector on full and lightly clipped plotting extents.

    The full extent preserves genuinely remote branches. The lightly clipped
    extent prevents a handful of raw outliers from compressing the populated
    density field into only one or two hex rows. A panel is called Branch when
    either extent has support from ``required_support`` grid sizes. One-grid
    support is retained in ``score`` so callers can report it as Candidate.
    """
    xv, yv = _clean_xy(x, y)
    if len(xv) < 80:
        return BranchResult("Multi-extent Hexbin", False, 1, 0.0, notes="insufficient data")

    extent_runs = []
    for quantile in extent_quantiles:
        quantile = float(quantile)
        if quantile <= 0:
            keep = np.ones(len(xv), dtype=bool)
            bounds = (
                float(np.min(xv)), float(np.max(xv)),
                float(np.min(yv)), float(np.max(yv)),
            )
        else:
            x_lo, x_hi = np.quantile(xv, [quantile, 1 - quantile])
            y_lo, y_hi = np.quantile(yv, [quantile, 1 - quantile])
            keep = (
                (xv >= x_lo) & (xv <= x_hi)
                & (yv >= y_lo) & (yv <= y_hi)
            )
            bounds = (float(x_lo), float(x_hi), float(y_lo), float(y_hi))
        result = detect_multires_hexbin(
            xv[keep], yv[keep],
            gridsizes=gridsizes,
            required_support=required_support,
            **detector_kwargs,
        )
        extent_runs.append({
            "quantile": quantile,
            "bounds": bounds,
            "n_points": int(keep.sum()),
            "result": result,
        })

    chosen = max(
        extent_runs,
        key=lambda item: (
            item["result"].score,
            item["result"].x_coverage,
            -item["quantile"],
        ),
    )
    chosen_result = chosen["result"]
    branch_detected = any(item["result"].branch_detected for item in extent_runs)
    extent_votes = sum(item["result"].branch_detected for item in extent_runs)
    geometry = dict(chosen_result.geometry or {})
    geometry.update({
        "selected_extent_quantile": chosen["quantile"],
        "display_bounds": chosen["bounds"],
        "extent_runs": [
            {
                "quantile": item["quantile"],
                "bounds": item["bounds"],
                "n_points": item["n_points"],
                "branch_detected": item["result"].branch_detected,
                "score": item["result"].score,
                "x_coverage": item["result"].x_coverage,
            }
            for item in extent_runs
        ],
    })
    max_grid_support = int(round(chosen_result.score * len(gridsizes)))
    return BranchResult(
        method="Multi-extent Hexbin",
        branch_detected=branch_detected,
        n_branches=2 if branch_detected else 1,
        score=float(chosen_result.score),
        x_start=chosen_result.x_start,
        x_end=chosen_result.x_end,
        x_coverage=chosen_result.x_coverage,
        persistence=chosen_result.persistence,
        min_branch_weight=chosen_result.min_branch_weight,
        branch_separation=chosen_result.branch_separation,
        notes=(
            f"max grid support={max_grid_support}/{len(gridsizes)}; "
            f"extent votes={extent_votes}/{len(extent_runs)}; "
            f"selected clip={chosen['quantile']:.4f}"
        ),
        geometry=geometry,
    )


def _track_hexbin_resolution(
    resolution: dict[str, Any],
    *,
    y_range: float,
    max_jump: float = 0.18,
    max_curve_rmse: float = 0.06,
    min_separation_change: float = 0.04,
    min_relative_change: float = 0.25,
    min_separation_correlation: float = 0.55,
) -> dict[str, Any]:
    """Test whether local modal pairs form two continuous trajectories.

    ``_hexbin_resolution`` already decides whether two conditional modes persist
    across x windows.  This second stage is deliberately stricter: it connects
    the lower and upper modes, rejects paths that jump between unrelated peaks,
    and distinguishes a changing separation (split/merge branch) from two
    approximately parallel density bands.
    """
    empty = {
        "continuous": False,
        "topological_branch": False,
        "track_type": "none",
        "n_windows": 0,
        "jump_q90": np.inf,
        "curve_rmse": np.inf,
        "separation_change": 0.0,
        "relative_separation_change": 0.0,
        "separation_correlation": 0.0,
        "x": np.asarray([], dtype=float),
        "lower": np.asarray([], dtype=float),
        "upper": np.asarray([], dtype=float),
    }
    cluster = resolution.get("cluster", {})
    if not resolution.get("detected") or cluster.get("start", -1) < 0:
        return empty

    selected = []
    for index, window in enumerate(resolution.get("windows", [])):
        in_cluster = cluster["start"] <= index <= cluster["end"]
        modes = window.get("modes", ())
        if in_cluster and window.get("flag") and len(modes) >= 2:
            selected.append((index, window, float(modes[0]), float(modes[1])))
    if len(selected) < 3:
        return {**empty, "n_windows": len(selected)}

    indices = np.asarray([item[0] for item in selected], dtype=float)
    x_values = np.asarray([item[1]["x_mid"] for item in selected], dtype=float)
    lower = np.asarray([item[2] for item in selected], dtype=float)
    upper = np.asarray([item[3] for item in selected], dtype=float)
    scale = max(float(y_range), 1e-12)
    gaps = np.maximum(np.diff(indices), 1.0)
    jumps = np.concatenate([
        np.abs(np.diff(lower)) / gaps / scale,
        np.abs(np.diff(upper)) / gaps / scale,
    ])
    jump_q90 = float(np.quantile(jumps, 0.90)) if len(jumps) else 0.0

    x_scaled = (x_values - x_values.min()) / max(float(np.ptp(x_values)), 1e-12)
    residuals = []
    for track in (lower, upper):
        degree = min(2, len(track) - 1)
        fitted = np.polyval(np.polyfit(x_scaled, track, degree), x_scaled)
        residuals.append(float(np.sqrt(np.mean((track - fitted) ** 2)) / scale))
    curve_rmse = float(max(residuals, default=np.inf))
    continuous = bool(jump_q90 <= max_jump and curve_rmse <= max_curve_rmse)

    separation = (upper - lower) / scale
    edge_width = max(1, min(3, len(separation) // 3))
    separation_start = float(np.mean(separation[:edge_width]))
    separation_end = float(np.mean(separation[-edge_width:]))
    separation_change = float(abs(separation_end - separation_start))
    relative_change = float(
        separation_change / max(separation_start, separation_end, 1e-12)
    )
    if len(separation) >= 3 and float(np.std(separation)) > 1e-12:
        separation_correlation = float(np.corrcoef(x_scaled, separation)[0, 1])
    else:
        separation_correlation = 0.0
    topological_branch = bool(
        continuous
        and separation_change >= min_separation_change
        and relative_change >= min_relative_change
        and abs(separation_correlation) >= min_separation_correlation
    )
    track_type = "branch" if topological_branch else ("two_band" if continuous else "broken")
    return {
        "continuous": continuous,
        "topological_branch": topological_branch,
        "track_type": track_type,
        "n_windows": int(len(selected)),
        "jump_q90": jump_q90,
        "curve_rmse": curve_rmse,
        "separation_change": separation_change,
        "relative_separation_change": relative_change,
        "separation_correlation": separation_correlation,
        "x": x_values,
        "lower": lower,
        "upper": upper,
    }


def _topology_track_hexbin_resolution(
    resolution: dict[str, Any],
    *,
    min_x_coverage: float = 0.12,
    min_window_density: float = 0.55,
    max_standardized_jump: float = 2.5,
    min_valley_median: float = 0.45,
    min_compactness_median: float = 3.0,
    min_relative_change: float = 0.30,
    min_abs_spearman: float = 0.80,
    min_monotone_fraction: float = 0.70,
    trunk_search_windows: int = 2,
) -> dict[str, Any]:
    """Detect a one-ridge <-> two-ridge topology at one Hexbin scale.

    Continuity is expressed using x coverage and jumps measured in local peak
    widths, so curved branches are allowed. A confirmed topology additionally
    needs a neighboring one-mode ridge connected to the narrow end of the two
    tracks. If that end lies on the sampled x boundary, the evidence is kept as
    an edge-truncated candidate rather than promoted to a confirmed branch.
    """
    empty = {
        "continuous": False,
        "topological_branch": False,
        "strong_vote": False,
        "edge_candidate": False,
        "track_type": "none",
        "direction": "none",
        "n_windows": 0,
        "x_coverage": 0.0,
        "window_density": 0.0,
        "jump_q90_local_widths": np.inf,
        "valley_median": np.nan,
        "compactness_median": np.nan,
        "relative_separation_change": 0.0,
        "separation_spearman": 0.0,
        "monotone_fraction": 0.0,
        "trunk_available": False,
        "trunk_connected": False,
        "edge_truncated": False,
        "trunk_x": np.nan,
        "trunk_y": np.nan,
        "x": np.asarray([], dtype=float),
        "lower": np.asarray([], dtype=float),
        "upper": np.asarray([], dtype=float),
    }
    windows = resolution.get("windows", [])
    cluster = resolution.get("cluster", {})
    if not resolution.get("detected") or cluster.get("start", -1) < 0:
        return empty

    selected = []
    for index, window in enumerate(windows):
        modes = window.get("modes", ())
        in_cluster = cluster["start"] <= index <= cluster["end"]
        if in_cluster and window.get("flag") and len(modes) >= 2:
            selected.append((index, window, float(modes[0]), float(modes[1])))
    if len(selected) < 3:
        return {**empty, "n_windows": len(selected)}

    indices = np.asarray([item[0] for item in selected], dtype=int)
    x_values = np.asarray([item[1]["x_mid"] for item in selected], dtype=float)
    lower = np.asarray([item[2] for item in selected], dtype=float)
    upper = np.asarray([item[3] for item in selected], dtype=float)
    widths = np.asarray([
        max(
            float(item[1].get("pooled_width", np.nan)),
            float(item[1].get("y_step", 1e-12)) / 2,
        )
        for item in selected
    ], dtype=float)
    widths[~np.isfinite(widths) | (widths <= 1e-12)] = 1e-12

    full_x_min = min(window["x_min"] for window in windows)
    full_x_max = max(window["x_max"] for window in windows)
    full_x_range = max(float(full_x_max - full_x_min), 1e-12)
    x_coverage = float(
        (selected[-1][1]["x_max"] - selected[0][1]["x_min"])
        / full_x_range
    )
    index_span = max(int(indices[-1] - indices[0] + 1), 1)
    window_density = float(len(indices) / index_span)

    index_gaps = np.maximum(np.diff(indices), 1)
    local_width = np.maximum((widths[:-1] + widths[1:]) / 2, 1e-12)
    jumps = np.concatenate([
        np.abs(np.diff(lower)) / index_gaps / local_width,
        np.abs(np.diff(upper)) / index_gaps / local_width,
    ])
    jump_q90 = float(np.quantile(jumps, 0.90)) if len(jumps) else 0.0
    continuous = bool(
        x_coverage >= min_x_coverage
        and window_density >= min_window_density
        and jump_q90 <= max_standardized_jump
    )

    valley_median = float(np.median([
        item[1].get("valley_depth", np.nan) for item in selected
    ]))
    compactness_median = float(np.median([
        item[1].get("compactness", np.nan) for item in selected
    ]))
    morphology_ok = bool(
        valley_median >= min_valley_median
        and compactness_median >= min_compactness_median
    )

    separation = upper - lower
    edge_width = max(1, min(3, len(separation) // 3))
    separation_start = float(np.mean(separation[:edge_width]))
    separation_end = float(np.mean(separation[-edge_width:]))
    signed_change = separation_end - separation_start
    relative_change = float(
        abs(signed_change)
        / max(separation_start, separation_end, 1e-12)
    )
    if len(separation) >= 3 and np.ptp(separation) > 1e-12:
        spearman_value = float(spearmanr(x_values, separation).statistic)
        if not np.isfinite(spearman_value):
            spearman_value = 0.0
    else:
        spearman_value = 0.0
    direction_sign = 1.0 if signed_change > 0 else (-1.0 if signed_change < 0 else 0.0)
    if len(separation) >= 2 and direction_sign:
        tolerance = 0.03 * max(float(np.median(separation)), 1e-12)
        monotone_fraction = float(
            np.mean(direction_sign * np.diff(separation) >= -tolerance)
        )
    else:
        monotone_fraction = 0.0
    trend_ok = bool(
        relative_change >= min_relative_change
        and abs(spearman_value) >= min_abs_spearman
        and monotone_fraction >= min_monotone_fraction
    )
    direction = "split" if direction_sign > 0 else ("merge" if direction_sign < 0 else "parallel")

    boundary_at_start = direction_sign > 0
    boundary_selected = selected[0] if boundary_at_start else selected[-1]
    boundary_index, boundary_window, boundary_lower, boundary_upper = boundary_selected
    populated_indices = [
        index for index, window in enumerate(windows)
        if window.get("total_count", 0) >= 20
    ]
    effective_first = min(populated_indices, default=0)
    effective_last = max(populated_indices, default=len(windows) - 1)
    if boundary_at_start:
        neighbor_indices = range(
            boundary_index - 1,
            max(-1, boundary_index - trunk_search_windows - 1),
            -1,
        )
        edge_truncated = boundary_index <= effective_first + 1
    else:
        neighbor_indices = range(
            boundary_index + 1,
            min(len(windows), boundary_index + trunk_search_windows + 1),
        )
        edge_truncated = boundary_index >= effective_last - 1

    trunk_window = next((
        windows[index] for index in neighbor_indices
        if windows[index].get("total_count", 0) >= 20
        and np.isfinite(windows[index].get("primary_mode", np.nan))
        and not windows[index].get("flag", False)
    ), None)
    trunk_available = trunk_window is not None
    trunk_connected = False
    trunk_x = np.nan
    trunk_y = np.nan
    if trunk_window is not None:
        trunk_x = float(trunk_window["x_mid"])
        trunk_y = float(trunk_window["primary_mode"])
        boundary_width = max(
            float(boundary_window.get("pooled_width", 0.0)),
            float(boundary_window.get("y_step", 1e-12)),
        )
        trunk_connected = bool(
            boundary_lower - boundary_width
            <= trunk_y
            <= boundary_upper + boundary_width
        )

    strong_vote = bool(
        continuous and morphology_ok and trend_ok and trunk_connected
    )
    edge_candidate = bool(
        continuous and morphology_ok and trend_ok
        and not trunk_connected and edge_truncated
    )
    if strong_vote:
        track_type = direction
    elif edge_candidate:
        track_type = f"edge-{direction}"
    elif continuous:
        track_type = "wide/two-ridge"
    else:
        track_type = "broken"
    return {
        "continuous": continuous,
        "topological_branch": strong_vote,
        "strong_vote": strong_vote,
        "edge_candidate": edge_candidate,
        "track_type": track_type,
        "direction": direction,
        "n_windows": int(len(selected)),
        "x_coverage": x_coverage,
        "window_density": window_density,
        "jump_q90_local_widths": jump_q90,
        "valley_median": valley_median,
        "compactness_median": compactness_median,
        "relative_separation_change": relative_change,
        "separation_spearman": spearman_value,
        "monotone_fraction": monotone_fraction,
        "trunk_available": trunk_available,
        "trunk_connected": trunk_connected,
        "edge_truncated": edge_truncated,
        "trunk_x": trunk_x,
        "trunk_y": trunk_y,
        "x": x_values,
        "lower": lower,
        "upper": upper,
    }


def detect_topological_multiextent_hexbin(
    x,
    y,
    *,
    extent_quantiles=(0.0, 0.0025),
    gridsizes=(16, 22, 28, 34, 40, 46),
    required_support=3,
    **detector_kwargs,
):
    """Classify confirmed, candidate, and wide/two-ridge geometries."""
    xv, yv = _clean_xy(x, y)
    if len(xv) < 80:
        return BranchResult(
            "Topological multi-extent Hexbin", False, 1, 0.0,
            notes="insufficient data",
            geometry={"track_status": "No branch"},
        )

    extent_runs = []
    for quantile_value in extent_quantiles:
        quantile_value = float(quantile_value)
        if quantile_value <= 0:
            keep = np.ones(len(xv), dtype=bool)
            bounds = (
                float(np.min(xv)), float(np.max(xv)),
                float(np.min(yv)), float(np.max(yv)),
            )
        else:
            x_lo, x_hi = np.quantile(xv, [quantile_value, 1 - quantile_value])
            y_lo, y_hi = np.quantile(yv, [quantile_value, 1 - quantile_value])
            keep = (
                (xv >= x_lo) & (xv <= x_hi)
                & (yv >= y_lo) & (yv <= y_hi)
            )
            bounds = (float(x_lo), float(x_hi), float(y_lo), float(y_hi))

        base_result = detect_multires_hexbin(
            xv[keep], yv[keep],
            gridsizes=gridsizes,
            required_support=required_support,
            **detector_kwargs,
        )
        resolutions = (base_result.geometry or {}).get("resolutions", [])
        for resolution in resolutions:
            resolution["tracking"] = _topology_track_hexbin_resolution(resolution)
        strong_support = sum(
            item["tracking"]["strong_vote"] for item in resolutions
        )
        edge_support = sum(
            item["tracking"]["edge_candidate"] for item in resolutions
        )
        continuous_support = sum(
            item["tracking"]["continuous"] for item in resolutions
        )
        evidence_support = strong_support + edge_support
        if strong_support >= required_support:
            status, status_code = "Branch", 4
        elif strong_support >= max(2, required_support - 1):
            status, status_code = "Strong candidate", 3
        elif evidence_support >= 1:
            status, status_code = "Candidate", 2
        elif continuous_support >= required_support:
            status, status_code = "Wide/two-ridge", 1
        else:
            status, status_code = "No branch", 0
        extent_runs.append({
            "quantile": quantile_value,
            "bounds": bounds,
            "n_points": int(keep.sum()),
            "base_result": base_result,
            "resolutions": resolutions,
            "strong_support": int(strong_support),
            "edge_support": int(edge_support),
            "evidence_support": int(evidence_support),
            "continuous_support": int(continuous_support),
            "status": status,
            "status_code": status_code,
        })

    chosen = max(
        extent_runs,
        key=lambda item: (
            item["status_code"], item["strong_support"],
            item["evidence_support"], item["continuous_support"],
            item["base_result"].x_coverage, -item["quantile"],
        ),
    )
    supported_tracks = [
        item["tracking"] for item in chosen["resolutions"]
        if item["tracking"]["continuous"]
    ]
    best_track = max(
        supported_tracks,
        key=lambda item: (
            item["strong_vote"], item["edge_candidate"],
            item["relative_separation_change"], item["n_windows"],
        ),
        default=None,
    )
    status = chosen["status"]
    base_result = chosen["base_result"]
    geometry = dict(base_result.geometry or {})
    geometry.update({
        "track_status": status,
        "strong_support": chosen["strong_support"],
        "edge_support": chosen["edge_support"],
        "evidence_support": chosen["evidence_support"],
        "continuous_support": chosen["continuous_support"],
        "selected_extent_quantile": chosen["quantile"],
        "display_bounds": chosen["bounds"],
        "best_track": best_track,
        "extent_runs": [
            {
                "quantile": item["quantile"],
                "bounds": item["bounds"],
                "n_points": item["n_points"],
                "status": item["status"],
                "strong_support": item["strong_support"],
                "edge_support": item["edge_support"],
                "continuous_support": item["continuous_support"],
            }
            for item in extent_runs
        ],
    })
    return BranchResult(
        method="Topological multi-extent Hexbin",
        branch_detected=status == "Branch",
        n_branches=2 if status != "No branch" else 1,
        score=float(chosen["evidence_support"] / len(gridsizes)),
        x_start=base_result.x_start if chosen["continuous_support"] else np.nan,
        x_end=base_result.x_end if chosen["continuous_support"] else np.nan,
        x_coverage=(best_track or {}).get("x_coverage", 0.0),
        persistence=(best_track or {}).get("n_windows", 0),
        min_branch_weight=base_result.min_branch_weight,
        branch_separation=(best_track or {}).get(
            "relative_separation_change", np.nan
        ),
        notes=(
            f"status={status}; strong={chosen['strong_support']}/{len(gridsizes)}; "
            f"edge={chosen['edge_support']}/{len(gridsizes)}; "
            f"continuous={chosen['continuous_support']}/{len(gridsizes)}; "
            f"selected clip={chosen['quantile']:.4f}"
        ),
        geometry=geometry,
    )


def detect_tracked_multiextent_hexbin(
    x,
    y,
    *,
    extent_quantiles=(0.0, 0.0025),
    gridsizes=(16, 22, 28, 34, 40, 46),
    required_support=3,
    **detector_kwargs,
):
    """Track Hexbin modes across x and classify split/merge versus two bands.

    The returned geometry contains ``track_status`` with one of ``Branch``,
    ``Two-band``, ``Candidate``, and ``No branch``.  A Branch needs continuous
    tracks *and* a systematic change in their separation at the configured
    number of grid sizes. Two-band likewise uses ``required_support`` but lacks
    convincing split/merge geometry. Lower support remains Candidate.
    """
    xv, yv = _clean_xy(x, y)
    if len(xv) < 80:
        return BranchResult(
            "Tracked multi-extent Hexbin", False, 1, 0.0,
            notes="insufficient data",
            geometry={"track_status": "No branch"},
        )

    extent_runs = []
    for quantile_value in extent_quantiles:
        quantile_value = float(quantile_value)
        if quantile_value <= 0:
            keep = np.ones(len(xv), dtype=bool)
            bounds = (
                float(np.min(xv)), float(np.max(xv)),
                float(np.min(yv)), float(np.max(yv)),
            )
        else:
            x_lo, x_hi = np.quantile(xv, [quantile_value, 1 - quantile_value])
            y_lo, y_hi = np.quantile(yv, [quantile_value, 1 - quantile_value])
            keep = (
                (xv >= x_lo) & (xv <= x_hi)
                & (yv >= y_lo) & (yv <= y_hi)
            )
            bounds = (float(x_lo), float(x_hi), float(y_lo), float(y_hi))

        base_result = detect_multires_hexbin(
            xv[keep], yv[keep],
            gridsizes=gridsizes,
            required_support=required_support,
            **detector_kwargs,
        )
        y_range = max(bounds[3] - bounds[2], 1e-12)
        resolutions = (base_result.geometry or {}).get("resolutions", [])
        for resolution in resolutions:
            resolution["tracking"] = _track_hexbin_resolution(
                resolution, y_range=y_range
            )
        branch_support = sum(
            item["tracking"]["topological_branch"] for item in resolutions
        )
        band_support = sum(
            item["tracking"]["continuous"] for item in resolutions
        )
        if branch_support >= required_support:
            status, status_code = "Branch", 3
        elif band_support >= required_support:
            status, status_code = "Two-band", 2
        elif band_support >= 1:
            status, status_code = "Candidate", 1
        else:
            status, status_code = "No branch", 0
        extent_runs.append({
            "quantile": quantile_value,
            "bounds": bounds,
            "n_points": int(keep.sum()),
            "base_result": base_result,
            "resolutions": resolutions,
            "branch_support": int(branch_support),
            "band_support": int(band_support),
            "status": status,
            "status_code": status_code,
        })

    chosen = max(
        extent_runs,
        key=lambda item: (
            item["status_code"], item["branch_support"], item["band_support"],
            item["base_result"].x_coverage, -item["quantile"],
        ),
    )
    status = chosen["status"]
    base_result = chosen["base_result"]
    supported_tracks = [
        item["tracking"] for item in chosen["resolutions"]
        if item["tracking"]["continuous"]
    ]
    best_track = max(
        supported_tracks,
        key=lambda item: (item["topological_branch"], item["n_windows"]),
        default=None,
    )
    # A second, deliberately strong consensus route handles clear boundary-
    # anchored fans: two resolutions must contain continuous tracks, one must
    # explicitly see split/merge topology, and that topology must show both a
    # large relative separation change and a strongly monotone trend.  This is
    # still label-free and avoids requiring an identical peak discretization at
    # two grid sizes.
    strong_separation_consensus = bool(
        status == "Two-band"
        and chosen["branch_support"] >= 1
        and chosen["band_support"] >= required_support
        and best_track is not None
        and best_track["relative_separation_change"] >= 0.50
        and abs(best_track["separation_correlation"]) >= 0.80
    )
    if strong_separation_consensus:
        status = "Branch"
    geometry = dict(base_result.geometry or {})
    geometry.update({
        "track_status": status,
        "branch_support": chosen["branch_support"],
        "band_support": chosen["band_support"],
        "selected_extent_quantile": chosen["quantile"],
        "display_bounds": chosen["bounds"],
        "best_track": best_track,
        "strong_separation_consensus": strong_separation_consensus,
        "extent_runs": [
            {
                "quantile": item["quantile"],
                "bounds": item["bounds"],
                "n_points": item["n_points"],
                "status": item["status"],
                "branch_support": item["branch_support"],
                "band_support": item["band_support"],
            }
            for item in extent_runs
        ],
    })
    return BranchResult(
        method="Tracked multi-extent Hexbin",
        branch_detected=status == "Branch",
        n_branches=2 if status in {"Branch", "Two-band", "Candidate"} else 1,
        score=float(chosen["band_support"] / len(gridsizes)),
        x_start=base_result.x_start if chosen["band_support"] else np.nan,
        x_end=base_result.x_end if chosen["band_support"] else np.nan,
        x_coverage=base_result.x_coverage if chosen["band_support"] else 0.0,
        persistence=(best_track or {}).get("n_windows", 0),
        min_branch_weight=base_result.min_branch_weight,
        branch_separation=(best_track or {}).get("relative_separation_change", np.nan),
        notes=(
            f"status={status}; branch support={chosen['branch_support']}/{len(gridsizes)}; "
            f"continuous-band support={chosen['band_support']}/{len(gridsizes)}; "
            f"strong separation consensus={strong_separation_consensus}; "
            f"selected clip={chosen['quantile']:.4f}"
        ),
        geometry=geometry,
    )


def detect_principal_tree(
    x,
    y,
    *,
    n_nodes=28,
    min_leaf_support=0.025,
):
    """Lightweight principal-graph approximation using centroid MST pruning."""
    xv, yv = _clean_xy(x, y)
    if len(xv) < max(80, n_nodes * 3):
        return BranchResult("Principal tree (MST)", False, 1, 0.0, notes="insufficient data")
    xs, ys, scale = _standardize_xy(xv, yv)
    points = np.column_stack([xs, ys])
    model = MiniBatchKMeans(
        n_clusters=n_nodes,
        n_init=10,
        random_state=23,
        batch_size=min(2048, len(points)),
    ).fit(points)
    centers = model.cluster_centers_
    support = np.bincount(model.labels_, minlength=n_nodes) / len(points)
    difference = centers[:, None, :] - centers[None, :, :]
    distance = np.sqrt(np.sum(difference ** 2, axis=2))
    tree = minimum_spanning_tree(distance).toarray()
    adjacency = (tree + tree.T) > 0
    active = np.ones(n_nodes, dtype=bool)
    # Repeatedly remove tiny leaf nodes; this suppresses outlier twigs.
    changed = True
    while changed:
        changed = False
        degree = adjacency[:, active].sum(axis=1)
        leaves = np.where(active & (degree <= 1) & (support < min_leaf_support))[0]
        if len(leaves):
            active[leaves] = False
            changed = True
    active_index = np.where(active)[0]
    active_adjacency = adjacency[np.ix_(active_index, active_index)]
    degree = active_adjacency.sum(axis=1)
    branch_nodes_local = np.where(degree >= 3)[0]
    branch_nodes = active_index[branch_nodes_local]
    n_leaves = int(np.sum(degree == 1))
    detected = bool(len(branch_nodes) >= 1 and n_leaves >= 3)
    xc, xscale, yc, yscale = scale
    original_centers = centers.copy()
    original_centers[:, 0] = centers[:, 0] * xscale + xc
    original_centers[:, 1] = centers[:, 1] * yscale + yc
    edges = []
    for i, j in zip(*np.where(np.triu(adjacency, 1))):
        if active[i] and active[j]:
            edges.append((int(i), int(j)))
    geometry = {
        "centers": original_centers,
        "edges": edges,
        "active": active,
        "support": support,
        "branch_nodes": branch_nodes,
    }
    return BranchResult(
        method="Principal tree (MST)",
        branch_detected=detected,
        n_branches=max(n_leaves - 1, 1) if detected else 1,
        score=float(len(branch_nodes)),
        x_start=float(original_centers[active, 0].min()) if detected else np.nan,
        x_end=float(original_centers[active, 0].max()) if detected else np.nan,
        x_coverage=1.0 if detected else 0.0,
        persistence=int(len(branch_nodes)),
        min_branch_weight=float(support[active].min()),
        branch_separation=np.nan,
        notes=f"{n_nodes} centroids; {len(branch_nodes)} branch nodes after leaf pruning",
        geometry=geometry,
    )


def run_all_branch_detectors(x, y) -> list[BranchResult]:
    """Run the comparable static-branch benchmark suite."""
    return [
        detect_current_dip(x, y),
        detect_modal_kde(x, y),
        detect_sliding_gmm(x, y),
        detect_mixture_regression(x, y),
        detect_density_ridge(x, y),
        detect_principal_tree(x, y),
    ]


def plot_branch_result(ax, x, y, result: BranchResult, *, point_color="#64748B"):
    """Draw one detector result on a shared scatter-plot scale."""
    xv, yv = _clean_xy(x, y)
    geometry = result.geometry or {}
    method = result.method
    if method in {"Multi-resolution Hexbin", "Multi-extent Hexbin", "Tracked multi-extent Hexbin"}:
        from matplotlib.colors import LogNorm
        resolutions = geometry.get("resolutions", [])
        chosen = max(resolutions, key=lambda item: item["gridsize"], default=None)
        gridsize = chosen["gridsize"] if chosen is not None else 32
        plot_x, plot_y = xv, yv
        bounds = geometry.get("display_bounds")
        if bounds is not None:
            x_lo, x_hi, y_lo, y_hi = bounds
            keep = (
                (xv >= x_lo) & (xv <= x_hi)
                & (yv >= y_lo) & (yv <= y_hi)
            )
            plot_x, plot_y = xv[keep], yv[keep]
        ax.hexbin(plot_x, plot_y, gridsize=gridsize, mincnt=1, norm=LogNorm(), cmap="Greys", alpha=0.72, zorder=1)
    else:
        ax.scatter(xv, yv, s=5, alpha=0.18, color=point_color, edgecolors="none", rasterized=True)
    if method in {"Sliding KDE modal", "Sliding-window GMM", "2-D KDE density ridge"}:
        windows = geometry.get("windows", [])
        y_jump_limit = 0.20 * max(
            np.quantile(yv, 0.995) - np.quantile(yv, 0.005), 1e-12
        )
        for mode_index, color in [(0, "#D62728"), (1, "#1F77B4"), (2, "#2CA02C")]:
            segments = []
            xx, yy = [], []
            previous_index = None
            for window_index, item in enumerate(windows):
                modes = item.get("modes", ())
                if item.get("flag") and len(modes) > mode_index and np.isfinite(modes[mode_index]):
                    discontinuous = (
                        previous_index is not None
                        and (
                            window_index != previous_index + 1
                            or abs(modes[mode_index] - yy[-1]) > y_jump_limit
                        )
                    )
                    if discontinuous and xx:
                        segments.append((xx, yy))
                        xx, yy = [], []
                    xx.append(item["x_mid"])
                    yy.append(modes[mode_index])
                    previous_index = window_index
                elif xx:
                    segments.append((xx, yy))
                    xx, yy = [], []
                    previous_index = None
            if xx:
                segments.append((xx, yy))
            for segment_x, segment_y in segments:
                if len(segment_x) >= 2:
                    ax.plot(segment_x, segment_y, "o-", ms=3.0, lw=1.4, color=color, zorder=5)
    elif method == "Mixture of regressions":
        x_grid = geometry.get("x_grid")
        predictions = geometry.get("predictions")
        effective = geometry.get("effective")
        weights = geometry.get("weights")
        if x_grid is not None and predictions is not None:
            palette = ["#D62728", "#1F77B4", "#2CA02C"]
            for k in range(predictions.shape[1]):
                alpha = 1.0 if effective is None or effective[k] else 0.25
                label = f"w={weights[k]:.2f}" if weights is not None else None
                ax.plot(x_grid, predictions[:, k], lw=2.0, color=palette[k % len(palette)], alpha=alpha, label=label)
            ax.legend(loc="lower right", fontsize=7, frameon=True)
    elif method == "Principal tree (MST)":
        centers = geometry.get("centers")
        active = geometry.get("active")
        if centers is not None:
            for i, j in geometry.get("edges", []):
                ax.plot(centers[[i, j], 0], centers[[i, j], 1], color="#D62728", lw=1.6, zorder=5)
            ax.scatter(centers[active, 0], centers[active, 1], s=18, color="#D62728", zorder=6)
    elif method == "Current conditional Dip":
        for item in geometry.get("windows", []):
            if item.get("flag"):
                ax.axvspan(item["x_min"], item["x_max"], color="#F59E0B", alpha=0.12)
    elif method in {
        "Multi-resolution Hexbin", "Multi-extent Hexbin",
        "Tracked multi-extent Hexbin", "Topological multi-extent Hexbin",
    }:
        resolutions = geometry.get("resolutions", [])
        detected_resolutions = [
            item for item in resolutions
            if item.get("tracking", {}).get("continuous", item.get("detected", False))
        ]
        chosen = max(
            detected_resolutions,
            key=lambda item: (
                item.get("tracking", {}).get("topological_branch", False),
                item.get("tracking", {}).get("n_windows", 0),
                item["gridsize"],
            ),
            default=None,
        )
        if chosen is not None:
            tracking = chosen.get("tracking", {})
            if tracking.get("continuous"):
                for values, color in [
                    (tracking.get("lower", []), "#D62728"),
                    (tracking.get("upper", []), "#1F77B4"),
                ]:
                    ax.plot(tracking.get("x", []), values, "o-", ms=3.5, lw=1.7, color=color, zorder=5)
            else:
                cluster = chosen["cluster"]
                for mode_index, color in [(0, "#D62728"), (1, "#1F77B4")]:
                    segment_x, segment_y = [], []
                    for window_index, item in enumerate(chosen["windows"]):
                        in_cluster = cluster["start"] <= window_index <= cluster["end"]
                        modes = item.get("modes", ())
                        if in_cluster and item.get("flag") and len(modes) > mode_index:
                            segment_x.append(item["x_mid"])
                            segment_y.append(modes[mode_index])
                        elif segment_x:
                            if len(segment_x) >= 2:
                                ax.plot(segment_x, segment_y, "o-", ms=3.5, lw=1.7, color=color, zorder=5)
                            segment_x, segment_y = [], []
                    if len(segment_x) >= 2:
                        ax.plot(segment_x, segment_y, "o-", ms=3.5, lw=1.7, color=color, zorder=5)
    track_status = geometry.get("track_status")
    verdict = track_status.upper() if track_status else ("BRANCH" if result.branch_detected else "NO BRANCH")
    verdict_color = "#B91C1C" if verdict == "BRANCH" else ("#7C3AED" if verdict == "TWO-BAND" else "#374151")
    ax.set_title(f"{method}\n{verdict}", fontsize=10, color=verdict_color, fontweight="bold")
    ax.grid(alpha=0.2)
    return ax
