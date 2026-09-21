"""Conditional-spread classifier based on equal-frequency quantile buffers.

The mean response and the conditional spread are intentionally kept separate.
For each panel this module uses all finite points (after the caller's optional
EPS filter), splits x into 15 equal-count bins, removes the local linear trend
inside each bin, and measures the residual 5--95 % width. Pearson correlation
gives the global direction, while a reversal ratio prevents profiles with
substantial movement in both directions from being called a clean spread trend.
Amplitude and edge contrast prevent tiny fluctuations from being interpreted
as a spread trend.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


CLASSIFIER_VERSION = "eq15-detrended-spread-v16"

N_BINS = 15
MIN_PANEL_N = 150
MIN_BIN_N = 10
EDGE_BINS = 3

LOW_QUANTILE = 0.05
MID_QUANTILE = 0.50
HIGH_QUANTILE = 0.95

# Diagnostics: turning-point detection (retained for diagnostics, not gating).
SPREAD_TP_DEADZONE = 0.20
PEAK_SEARCH_RADIUS = 2

# Classification thresholds.
PEARSON_THRESHOLD = 0.70
PEAKED_PROMINENCE_THRESHOLD = 0.20
PEAKED_DROP_THRESHOLD = 0.10
FLAT_AMPLITUDE_THRESHOLD = 0.50


def classifier_signature() -> dict[str, Any]:
    """Return the complete set of settings used by the classifier."""
    return {
        "classifier_version": CLASSIFIER_VERSION,
        "n_bins": N_BINS,
        "min_panel_n": MIN_PANEL_N,
        "min_bin_n": MIN_BIN_N,
        "edge_bins": EDGE_BINS,
        "quantiles": (LOW_QUANTILE, MID_QUANTILE, HIGH_QUANTILE),
        "spread_tp_deadzone": SPREAD_TP_DEADZONE,
        "peak_search_radius": PEAK_SEARCH_RADIUS,
        "pearson_threshold": PEARSON_THRESHOLD,
        "peaked_prominence_threshold": PEAKED_PROMINENCE_THRESHOLD,
        "peaked_drop_threshold": PEAKED_DROP_THRESHOLD,
        "flat_amplitude_threshold": FLAT_AMPLITUDE_THRESHOLD,
    }


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with explicit constant-vector handling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation retained as a diagnostic, not a hard gate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return 0.0
    result = spearmanr(x, y)
    value = getattr(result, "statistic", result[0])
    return float(value) if np.isfinite(value) else 0.0


def _chord_curvature(
    x: np.ndarray, w: np.ndarray
) -> tuple[float, float]:
    """Signed mean curvature relative to the first-to-last chord.

    Positive: curve above chord (concave / saturation).
    Negative: curve below chord (convex / accelerating).
    Returns (curvature, chord_r2).
    """
    n = len(x)
    if n < 3 or x[-1] == x[0]:
        return 0.0, 0.0
    w_chord = w[0] + (w[-1] - w[0]) * (x - x[0]) / (x[-1] - x[0])
    deviations = w - w_chord
    dw = abs(w[-1] - w[0])
    if dw < 1e-12:
        dw = max(abs(np.median(w)), 1e-12)
    curvature = float(np.mean(deviations[1:-1]) / dw)
    ss_res = float(np.sum(deviations ** 2))
    ss_tot = float(np.sum((w - np.mean(w)) ** 2))
    chord_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return curvature, max(chord_r2, 0.0)


def _median_smooth_3(values: np.ndarray) -> np.ndarray:
    """Suppress one-bin width spikes without blurring the 15-bin profile."""
    values = np.asarray(values, dtype=float)
    smoothed = values.copy()
    for i in range(1, len(values) - 1):
        smoothed[i] = np.median(values[i - 1:i + 2])
    return smoothed


def _spread_turning_points(
    widths: np.ndarray,
    typical_width: float,
    deadzone: float,
    *,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], int]:
    """Return analyzed widths, normalized changes, sign runs, and TP count."""
    analyzed = _median_smooth_3(widths) if smooth else np.asarray(widths, dtype=float).copy()
    scale = max(float(typical_width), 1e-12)
    delta = np.diff(analyzed) / scale
    signs = tuple(
        1 if value >= deadzone else -1
        for value in delta
        if abs(value) >= deadzone
    )
    runs: list[int] = []
    for sign in signs:
        if not runs or sign != runs[-1]:
            runs.append(sign)
    return analyzed, delta, tuple(runs), max(0, len(runs) - 1)


def _spread_reversal_ratio(
    delta: np.ndarray,
    deadzone: float,
) -> tuple[float, float, float]:
    """Return reversal ratio and retained positive/negative width variation."""
    delta = np.asarray(delta, dtype=float)
    retained = np.where(np.abs(delta) >= deadzone, delta, 0.0)
    positive = max(0.0, float(retained[retained > 0].sum()))
    negative = max(0.0, float(-retained[retained < 0].sum()))
    total = positive + negative
    ratio = float(min(positive, negative) / total) if total > 0 else 0.0
    return ratio, positive, negative


def prepare_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    eps_filter: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter a panel and sort it by x; no subsampling or KDE weighting."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if eps_filter is not None:
        valid &= (np.abs(x) > eps_filter) & (np.abs(y) > eps_filter)
    x, y = x[valid], y[valid]
    order = np.argsort(x, kind="mergesort")
    return x[order], y[order]


def _empty_result(n: int, reason: str, n_bins: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "classifier_version": CLASSIFIER_VERSION,
        "spread_label": "Insufficient",
        "spread_status": reason,
        "n": int(n),
        "n_valid_bins": 0,
        "buffer_pearson": np.nan,
        "buffer_spearman": np.nan,
        "buffer_edge_contrast": np.nan,
        "buffer_amplitude": np.nan,
        "buffer_peak_bin": np.nan,
        "buffer_peak_prominence": np.nan,
        "buffer_fractional_drop": np.nan,
        "buffer_left_pearson": np.nan,
        "buffer_right_pearson": np.nan,
        "buffer_peak_supported": False,
        "buffer_tp_count": np.nan,
        "buffer_tp_pattern": "na",
        "buffer_tp_bin": np.nan,
        "buffer_reversal_ratio": np.nan,
        "buffer_positive_variation": np.nan,
        "buffer_negative_variation": np.nan,
        "buffer_raw_reversal_ratio": np.nan,
        "buffer_raw_positive_variation": np.nan,
        "buffer_raw_negative_variation": np.nan,
        "buffer_raw_tp_count": np.nan,
        "buffer_raw_tp_pattern": "na",
        "buffer_post_peak_negative_count": np.nan,
        "buffer_post_peak_positive_count": np.nan,
        "buffer_post_peak_drop": np.nan,
        "buffer_clean_peak": False,
        "buffer_raw_irregular": False,
        "buffer_chord_curvature": np.nan,
        "buffer_chord_r2": np.nan,
        "buffer_width_ratio": np.nan,
        "spread_width_method": "local_linear_residual_q05_q95",
    }
    for j in range(1, n_bins + 1):
        for name in ("x", "q05", "q50", "q95", "width"):
            result[f"buffer_{name}_b{j:02d}"] = np.nan
        for name in ("resid_q05", "resid_q50", "resid_q95", "local_slope"):
            result[f"buffer_{name}_b{j:02d}"] = np.nan
        result[f"buffer_width_smooth_b{j:02d}"] = np.nan
        if j < n_bins:
            result[f"buffer_width_delta_b{j:02d}"] = np.nan
            result[f"buffer_width_raw_delta_b{j:02d}"] = np.nan
        result[f"buffer_n_b{j:02d}"] = 0
    return result


def classify_spread(
    x: np.ndarray,
    y: np.ndarray,
    *,
    eps_filter: float | None = None,
    n_bins: int = N_BINS,
    min_panel_n: int = MIN_PANEL_N,
    min_bin_n: int = MIN_BIN_N,
    edge_bins: int = EDGE_BINS,
    spread_tp_deadzone: float = SPREAD_TP_DEADZONE,
    peak_search_radius: int = PEAK_SEARCH_RADIUS,
    pearson_threshold: float = PEARSON_THRESHOLD,
    peaked_prominence_threshold: float = PEAKED_PROMINENCE_THRESHOLD,
    peaked_drop_threshold: float = PEAKED_DROP_THRESHOLD,
    flat_amplitude_threshold: float = FLAT_AMPLITUDE_THRESHOLD,
) -> dict[str, Any]:
    """Classify the 5--95 % conditional-width pattern of one x-y panel.

    Labels are ``Increasing``, ``Decreasing``, ``Flat``,
    ``Peaked``, ``Complex``, and ``Insufficient``.
    """
    x, y = prepare_xy(x, y, eps_filter=eps_filter)
    if n_bins < 5:
        raise ValueError("n_bins must be at least 5")
    if len(x) < max(min_panel_n, n_bins * min_bin_n):
        return _empty_result(len(x), "too_few_points", n_bins)

    index_bins = np.array_split(np.arange(len(x)), n_bins)
    centers, q05, q50, q95, widths, counts = [], [], [], [], [], []
    resid_q05, resid_q50, resid_q95, local_slopes = [], [], [], []
    for indices in index_bins:
        xb, yb = x[indices], y[indices]
        counts.append(int(len(indices)))
        if len(indices) < min_bin_n:
            centers.append(np.nan)
            q05.append(np.nan)
            q50.append(np.nan)
            q95.append(np.nan)
            widths.append(np.nan)
            resid_q05.append(np.nan)
            resid_q50.append(np.nan)
            resid_q95.append(np.nan)
            local_slopes.append(np.nan)
            continue
        center = float(np.median(xb))
        dx = xb - center
        dx_centered = dx - np.mean(dx)
        denominator_x = float(np.dot(dx_centered, dx_centered))
        if denominator_x > 0:
            slope = float(np.dot(dx_centered, yb - np.mean(yb)) / denominator_x)
        else:
            slope = 0.0
        # Use the median of y - slope*dx as a robust local level at x=center.
        local_level = float(np.median(yb - slope * dx))
        residual = yb - (local_level + slope * dx)
        rlo, rmed, rhi = np.quantile(
            residual, [LOW_QUANTILE, MID_QUANTILE, HIGH_QUANTILE]
        )
        centers.append(center)
        resid_q05.append(float(rlo))
        resid_q50.append(float(rmed))
        resid_q95.append(float(rhi))
        local_slopes.append(slope)
        # Put residual quantiles back on the original y scale for plotting.
        q05.append(float(local_level + rlo))
        q50.append(float(local_level + rmed))
        q95.append(float(local_level + rhi))
        widths.append(float(max(rhi - rlo, 0.0)))

    centers = np.asarray(centers, dtype=float)
    q05 = np.asarray(q05, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    widths = np.asarray(widths, dtype=float)
    resid_q05 = np.asarray(resid_q05, dtype=float)
    resid_q50 = np.asarray(resid_q50, dtype=float)
    resid_q95 = np.asarray(resid_q95, dtype=float)
    local_slopes = np.asarray(local_slopes, dtype=float)
    valid = np.isfinite(centers) & np.isfinite(widths)
    if valid.sum() < max(5, n_bins - 2):
        return _empty_result(len(x), "too_few_valid_bins", n_bins)

    xv, wv = centers[valid], widths[valid]
    typical_width = float(np.median(wv))
    y_scale = float(np.quantile(y, HIGH_QUANTILE) - np.quantile(y, LOW_QUANTILE))
    floor = max(1e-12, 1e-9 * max(y_scale, 1.0))
    denominator = max(typical_width, floor)

    pearson = _safe_pearson(xv, wv)
    spearman = _safe_spearman(xv, wv)
    edge_n = min(edge_bins, len(wv) // 3)
    first_width = float(np.median(wv[:edge_n]))
    last_width = float(np.median(wv[-edge_n:]))
    edge_contrast = (last_width - first_width) / denominator
    amplitude = (
        float(np.quantile(wv, 0.90) - np.quantile(wv, 0.10))
        / denominator
    )

    smoothed_widths, width_delta, tp_pattern, tp_count = _spread_turning_points(
        wv, typical_width, spread_tp_deadzone
    )
    reversal_ratio, positive_variation, negative_variation = (
        _spread_reversal_ratio(width_delta, spread_tp_deadzone)
    )
    _, raw_width_delta, raw_tp_pattern, raw_tp_count = _spread_turning_points(
        wv, typical_width, spread_tp_deadzone, smooth=False
    )
    raw_reversal_ratio, raw_positive_variation, raw_negative_variation = (
        _spread_reversal_ratio(raw_width_delta, spread_tp_deadzone)
    )

    # Locate the observed peak close to the robust smoothed maximum. This keeps
    # isolated remote spikes from defining the peak, while retaining the raw
    # post-peak order needed to distinguish a sustained contraction from a
    # one-bin tail dip.
    smooth_peak_index = int(np.argmax(smoothed_widths))
    search_lo = max(0, smooth_peak_index - peak_search_radius)
    search_hi = min(len(wv), smooth_peak_index + peak_search_radius + 1)
    peak_index = search_lo + int(np.argmax(wv[search_lo:search_hi]))
    peak_width = float(wv[peak_index])
    edge_baseline = max(first_width, last_width)
    peak_prominence = (peak_width - edge_baseline) / denominator
    left_pearson = _safe_pearson(xv[:peak_index + 1], wv[:peak_index + 1])
    right_pearson = _safe_pearson(xv[peak_index:], wv[peak_index:])

    support_level = edge_baseline + 0.50 * max(peak_width - edge_baseline, 0.0)
    near_peak = wv[max(0, peak_index - 1):min(len(wv), peak_index + 2)]
    peak_supported = bool(np.sum(near_peak >= support_level) >= 2)

    post_peak_delta = raw_width_delta[peak_index:]
    retained_post_peak = post_peak_delta[
        np.abs(post_peak_delta) >= spread_tp_deadzone
    ]
    post_peak_negative_count = int(np.sum(retained_post_peak < 0))
    post_peak_positive_count = int(np.sum(retained_post_peak > 0))
    post_peak_drop = float((wv[peak_index] - wv[-1]) / denominator)

    raw_irregular = bool(
        raw_tp_count >= 4
        and raw_reversal_ratio >= 0.25
    )

    fractional_drop = float(
        (peak_width - wv[-1]) / peak_width if peak_width > 0 else 0.0
    )
    clean_peak = False

    # Diagnostics (retained but not used for gating).
    chord_curvature, chord_r2 = _chord_curvature(xv, wv)
    width_ratio = float(last_width / first_width) if first_width > 0 else float("inf")

    # Classification: Increasing (possibly Peaked) → Decreasing → Flat → Complex.
    if pearson >= pearson_threshold:
        label = "Increasing"
    elif pearson <= -pearson_threshold:
        label = "Decreasing"
    elif amplitude < flat_amplitude_threshold:
        label = "Flat"
    else:
        label = "Complex"

    result: dict[str, Any] = {
        "classifier_version": CLASSIFIER_VERSION,
        "spread_label": label,
        "spread_status": "ok",
        "n": int(len(x)),
        "n_valid_bins": int(valid.sum()),
        "buffer_pearson": pearson,
        "buffer_spearman": spearman,
        "buffer_edge_contrast": float(edge_contrast),
        "buffer_amplitude": float(amplitude),
        "buffer_peak_bin": int(peak_index + 1),
        "buffer_peak_prominence": float(peak_prominence),
        "buffer_fractional_drop": float(fractional_drop),
        "buffer_left_pearson": float(left_pearson),
        "buffer_right_pearson": float(right_pearson),
        "buffer_peak_supported": peak_supported,
        "buffer_tp_count": int(tp_count),
        "buffer_tp_pattern": "-".join(
            "up" if sign > 0 else "down" for sign in tp_pattern
        ) or "none",
        "buffer_tp_bin": int(peak_index + 1) if clean_peak else np.nan,
        "buffer_tp_deadzone": float(spread_tp_deadzone),
        "buffer_reversal_ratio": float(reversal_ratio),
        "buffer_positive_variation": float(positive_variation),
        "buffer_negative_variation": float(negative_variation),
        "buffer_raw_reversal_ratio": float(raw_reversal_ratio),
        "buffer_raw_positive_variation": float(raw_positive_variation),
        "buffer_raw_negative_variation": float(raw_negative_variation),
        "buffer_raw_tp_count": int(raw_tp_count),
        "buffer_raw_tp_pattern": "-".join(
            "up" if sign > 0 else "down" for sign in raw_tp_pattern
        ) or "none",
        "buffer_post_peak_negative_count": post_peak_negative_count,
        "buffer_post_peak_positive_count": post_peak_positive_count,
        "buffer_post_peak_drop": post_peak_drop,
        "buffer_clean_peak": clean_peak,
        "buffer_raw_irregular": raw_irregular,
        "buffer_chord_curvature": float(chord_curvature),
        "buffer_chord_r2": float(chord_r2),
        "buffer_width_ratio": float(width_ratio),
        "spread_width_method": "local_linear_residual_q05_q95",
    }
    for j in range(n_bins):
        result[f"buffer_x_b{j + 1:02d}"] = centers[j]
        result[f"buffer_q05_b{j + 1:02d}"] = q05[j]
        result[f"buffer_q50_b{j + 1:02d}"] = q50[j]
        result[f"buffer_q95_b{j + 1:02d}"] = q95[j]
        result[f"buffer_width_b{j + 1:02d}"] = widths[j]
        result[f"buffer_resid_q05_b{j + 1:02d}"] = resid_q05[j]
        result[f"buffer_resid_q50_b{j + 1:02d}"] = resid_q50[j]
        result[f"buffer_resid_q95_b{j + 1:02d}"] = resid_q95[j]
        result[f"buffer_local_slope_b{j + 1:02d}"] = local_slopes[j]
        result[f"buffer_width_smooth_b{j + 1:02d}"] = smoothed_widths[j]
        if j < n_bins - 1:
            result[f"buffer_width_delta_b{j + 1:02d}"] = width_delta[j]
            result[f"buffer_width_raw_delta_b{j + 1:02d}"] = raw_width_delta[j]
        result[f"buffer_n_b{j + 1:02d}"] = counts[j]
    return result


def classify_job(job: dict[str, Any]) -> dict[str, Any]:
    """Classify one notebook-style job and preserve its metadata."""
    metadata = {
        key: value for key, value in job.items()
        if key not in {"x", "y"}
    }
    result = classify_spread(
        job.get("x", np.array([])),
        job.get("y", np.array([])),
        eps_filter=job.get("eps_filter"),
        n_bins=int(job.get("n_bins", N_BINS)),
        min_panel_n=int(job.get("min_panel_n", MIN_PANEL_N)),
        min_bin_n=int(job.get("min_bin_n", MIN_BIN_N)),
        edge_bins=int(job.get("edge_bins", EDGE_BINS)),
        spread_tp_deadzone=float(
            job.get("spread_tp_deadzone", SPREAD_TP_DEADZONE)
        ),
        peak_search_radius=int(
            job.get("peak_search_radius", PEAK_SEARCH_RADIUS)
        ),
        pearson_threshold=float(
            job.get("pearson_threshold", PEARSON_THRESHOLD)
        ),
        peaked_prominence_threshold=float(
            job.get("peaked_prominence_threshold", PEAKED_PROMINENCE_THRESHOLD)
        ),
        peaked_drop_threshold=float(
            job.get("peaked_drop_threshold", PEAKED_DROP_THRESHOLD)
        ),
        flat_amplitude_threshold=float(
            job.get("flat_amplitude_threshold", FLAT_AMPLITUDE_THRESHOLD)
        ),
    )
    return {**metadata, **result}


def classify_batch(
    jobs: Iterable[dict[str, Any]],
    *,
    n_jobs: int = -1,
) -> list[dict[str, Any]]:
    """Classify jobs in parallel when joblib is available."""
    jobs = list(jobs)
    if n_jobs == 1:
        return [classify_job(job) for job in jobs]
    try:
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs)(delayed(classify_job)(job) for job in jobs)
    except ImportError:  # pragma: no cover
        return [classify_job(job) for job in jobs]
