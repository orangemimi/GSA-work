"""LOWESS classifier for locally detrended conditional-spread profiles.

The existing :mod:`spread_classifier` remains the single implementation of
the data preparation step: EPS filtering, 15 equal-frequency bins, a local
linear detrend inside every bin, and the residual Q05--Q95 width ``W_j``.
This module deliberately replaces only the *shape classification* step.

The 15 raw widths are noisy quantile estimates.  We therefore fit LOWESS to
``(x_bin, W_j)`` and classify the smooth spread profile.  Raw-width diagnostics
are retained in the returned record so that smoothing does not hide unstable
profiles during review.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

import spread_classifier as _base


CLASSIFIER_VERSION = "eq15-detrended-spread-lowess-v9-secondary-veto"

N_BINS = _base.N_BINS
MIN_PANEL_N = _base.MIN_PANEL_N
MIN_BIN_N = _base.MIN_BIN_N
EDGE_BINS = _base.EDGE_BINS

# With 15 profile points, frac=0.40 uses roughly six neighbouring widths.
LOWESS_FRAC = 0.40
LOWESS_IT = 2

# A change smaller than 5% of the typical residual width is direction-neutral.
LOWESS_DELTA_DEADZONE = 0.05

# Mutually exclusive structure tree; Spearman is the only direction gate.
STABLE_AMPLITUDE_THRESHOLD = 0.50
STABLE_ZERO_FRACTION_THRESHOLD = 0.70
STABLE_NET_CHANGE_THRESHOLD = 0.15
SPEARMAN_THRESHOLD = 0.60
# Pearson / net-change remain diagnostics and unused job kwargs; they no
# longer gate direction.
PEARSON_THRESHOLD = 0.60
DIRECTION_NET_CHANGE_THRESHOLD = 0.15
STRUCTURE_REVERSAL_THRESHOLD = 0.10
STRUCTURE_SIDE_COHERENCE_THRESHOLD = 0.65
STRUCTURE_DOMINANCE_RATIO = 1.50
# Main candidate: P1 = min(both arms) ≥ 0.15.  Legacy rise/drop kwargs map here.
PEAK_PROMINENCE_THRESHOLD = 0.15
PEAK_RISE_THRESHOLD = 0.15
PEAK_DROP_THRESHOLD = 0.15
TROUGH_DROP_THRESHOLD = 0.15
TROUGH_RISE_THRESHOLD = 0.15
# Secondary reverse structure: interior and P2 ≥ 0.15; no B, no coherence.
SECONDARY_PROMINENCE_THRESHOLD = 0.15
# Veto a main peak/trough only if P1/P2 < 1.5 and the profile actually reverses.
VETO_REVERSAL_THRESHOLD = 0.15
NET_EDGE_BINS = 2


def classifier_signature() -> dict[str, Any]:
    """Return every setting that can change the LOWESS spread labels."""
    return {
        "classifier_version": CLASSIFIER_VERSION,
        "width_extractor_version": _base.CLASSIFIER_VERSION,
        "n_bins": N_BINS,
        "min_panel_n": MIN_PANEL_N,
        "min_bin_n": MIN_BIN_N,
        "edge_bins": EDGE_BINS,
        "lowess_frac": LOWESS_FRAC,
        "lowess_it": LOWESS_IT,
        "lowess_delta_deadzone": LOWESS_DELTA_DEADZONE,
        "stable_amplitude_threshold": STABLE_AMPLITUDE_THRESHOLD,
        "stable_zero_fraction_threshold": STABLE_ZERO_FRACTION_THRESHOLD,
        "stable_net_change_threshold": STABLE_NET_CHANGE_THRESHOLD,
        "spearman_threshold": SPEARMAN_THRESHOLD,
        "pearson_threshold": PEARSON_THRESHOLD,
        "direction_net_change_threshold": DIRECTION_NET_CHANGE_THRESHOLD,
        "structure_reversal_threshold": STRUCTURE_REVERSAL_THRESHOLD,
        "structure_side_coherence_threshold": STRUCTURE_SIDE_COHERENCE_THRESHOLD,
        "structure_dominance_ratio": STRUCTURE_DOMINANCE_RATIO,
        "peak_prominence_threshold": PEAK_PROMINENCE_THRESHOLD,
        "peak_rise_threshold": PEAK_RISE_THRESHOLD,
        "peak_drop_threshold": PEAK_DROP_THRESHOLD,
        "trough_drop_threshold": TROUGH_DROP_THRESHOLD,
        "trough_rise_threshold": TROUGH_RISE_THRESHOLD,
        "secondary_prominence_threshold": SECONDARY_PROMINENCE_THRESHOLD,
        "veto_reversal_threshold": VETO_REVERSAL_THRESHOLD,
        "net_edge_bins": NET_EDGE_BINS,
    }


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(x) < 3 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(x) < 3 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return 0.0
    result = spearmanr(x, y)
    value = getattr(result, "statistic", result[0])
    return float(value) if np.isfinite(value) else 0.0


def _sign_runs(delta: np.ndarray, deadzone: float) -> tuple[int, ...]:
    """Compress significant adjacent changes to ordered up/down runs."""
    runs: list[int] = []
    for value in np.asarray(delta, dtype=float):
        if not np.isfinite(value) or abs(value) < deadzone:
            continue
        sign = 1 if value > 0 else -1
        if not runs or sign != runs[-1]:
            runs.append(sign)
    return tuple(runs)


def _variation_metrics(
    delta: np.ndarray,
    deadzone: float,
) -> tuple[float, float, float]:
    retained = np.where(np.abs(delta) >= deadzone, delta, 0.0)
    positive = float(retained[retained > 0].sum())
    negative = float(-retained[retained < 0].sum())
    total = positive + negative
    reversal = min(positive, negative) / total if total > 0 else 0.0
    return float(reversal), positive, negative


def _direction_share(delta: np.ndarray, deadzone: float, sign: int) -> float:
    """Return the share of significant variation moving in ``sign``."""
    _, positive, negative = _variation_metrics(delta, deadzone)
    total = positive + negative
    if total <= 0:
        return 0.0
    wanted = positive if sign > 0 else negative
    return float(wanted / total)


def _fit_profile(
    x: np.ndarray,
    width: np.ndarray,
    *,
    frac: float,
    it: int,
) -> np.ndarray:
    """Fit LOWESS and return values in the original, x-sorted bin order."""
    x = np.asarray(x, dtype=float)
    width = np.asarray(width, dtype=float)
    if len(x) < 3:
        return width.copy()

    # Discrete drivers can give adjacent equal-frequency bins the same median
    # x.  Statsmodels LOWESS divides by local x distances, so collapse those
    # ties robustly, fit the unique centres, then interpolate back to all bins.
    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) < len(x):
        unique_width = np.array(
            [np.median(width[inverse == i]) for i in range(len(unique_x))],
            dtype=float,
        )
    else:
        unique_width = width
    if len(unique_x) < 3:
        return np.full_like(width, float(np.median(width)))
    fitted = lowess(
        unique_width,
        unique_x,
        frac=float(frac),
        it=int(it),
        return_sorted=False,
        is_sorted=True,
    )
    fitted = np.interp(x, unique_x, np.asarray(fitted, dtype=float))
    return np.maximum(fitted, 0.0)


def classify_spread(
    x: np.ndarray,
    y: np.ndarray,
    *,
    eps_filter: float | None = None,
    n_bins: int = N_BINS,
    min_panel_n: int = MIN_PANEL_N,
    min_bin_n: int = MIN_BIN_N,
    edge_bins: int = EDGE_BINS,
    lowess_frac: float = LOWESS_FRAC,
    lowess_it: int = LOWESS_IT,
    lowess_delta_deadzone: float = LOWESS_DELTA_DEADZONE,
    stable_amplitude_threshold: float = STABLE_AMPLITUDE_THRESHOLD,
    stable_zero_fraction_threshold: float = STABLE_ZERO_FRACTION_THRESHOLD,
    stable_net_change_threshold: float = STABLE_NET_CHANGE_THRESHOLD,
    spearman_threshold: float = SPEARMAN_THRESHOLD,
    pearson_threshold: float = PEARSON_THRESHOLD,
    direction_net_change_threshold: float = DIRECTION_NET_CHANGE_THRESHOLD,
    structure_reversal_threshold: float = STRUCTURE_REVERSAL_THRESHOLD,
    structure_side_coherence_threshold: float = STRUCTURE_SIDE_COHERENCE_THRESHOLD,
    structure_dominance_ratio: float = STRUCTURE_DOMINANCE_RATIO,
    peak_prominence_threshold: float = PEAK_PROMINENCE_THRESHOLD,
    peak_rise_threshold: float = PEAK_RISE_THRESHOLD,
    peak_drop_threshold: float = PEAK_DROP_THRESHOLD,
    trough_drop_threshold: float = TROUGH_DROP_THRESHOLD,
    trough_rise_threshold: float = TROUGH_RISE_THRESHOLD,
    secondary_prominence_threshold: float = SECONDARY_PROMINENCE_THRESHOLD,
    veto_reversal_threshold: float = VETO_REVERSAL_THRESHOLD,
    net_edge_bins: int = NET_EDGE_BINS,
) -> dict[str, Any]:
    """Classify one locally detrended Q05--Q95 width profile.

    Structure is a mutually exclusive tree.  A global interior peak/trough
    with P1 = min(both arms) ≥ 0.15 and both-side coherence ≥ 0.65 is the
    main candidate.  An interior reverse structure with P2 ≥ 0.15 can veto
    it to Irregular only when P1/P2 < 1.5 and R_rev ≥ 0.15.  Secondary
    extrema never promote a monotonic profile to a peak.  Direction uses
    Spearman only and is the primary label solely when the structure is
    Simple.
    """
    _ = pearson_threshold, direction_net_change_threshold
    _ = peak_rise_threshold, peak_drop_threshold
    _ = trough_drop_threshold, trough_rise_threshold
    # The base result supplies the consistently computed bins and also keeps
    # all raw diagnostics for comparison.  Its old label is intentionally
    # ignored below.
    result = _base.classify_spread(
        x,
        y,
        eps_filter=eps_filter,
        n_bins=n_bins,
        min_panel_n=min_panel_n,
        min_bin_n=min_bin_n,
        edge_bins=edge_bins,
    )
    result["width_extractor_version"] = result.get("classifier_version")
    result["classifier_version"] = CLASSIFIER_VERSION

    if result.get("spread_status") != "ok":
        result["spread_label"] = "Insufficient"
        result["spread_strength"] = "Insufficient"
        result["spread_direction"] = "Insufficient"
        result["spread_direction_confidence"] = "Not applicable"
        result["spread_structure"] = "Insufficient"
        result["spread_structure_group"] = "Insufficient"
        result["spread_display_label"] = "Insufficient"
        result["spread_lowess_status"] = result.get("spread_status", "insufficient")
        return result

    centers = np.array(
        [result.get(f"buffer_x_b{j:02d}", np.nan) for j in range(1, n_bins + 1)],
        dtype=float,
    )
    widths = np.array(
        [result.get(f"buffer_width_b{j:02d}", np.nan) for j in range(1, n_bins + 1)],
        dtype=float,
    )
    valid = np.isfinite(centers) & np.isfinite(widths)
    xv, wv = centers[valid], widths[valid]
    if len(xv) < 5:
        result["spread_label"] = "Insufficient"
        result["spread_strength"] = "Insufficient"
        result["spread_direction"] = "Insufficient"
        result["spread_direction_confidence"] = "Not applicable"
        result["spread_structure"] = "Insufficient"
        result["spread_structure_group"] = "Insufficient"
        result["spread_display_label"] = "Insufficient"
        result["spread_lowess_status"] = "too_few_valid_bins"
        return result

    smooth = _fit_profile(xv, wv, frac=lowess_frac, it=lowess_it)
    typical = max(float(np.median(wv)), 1e-12)
    normalized_delta = np.diff(smooth) / typical
    runs = _sign_runs(normalized_delta, lowess_delta_deadzone)
    reversal, positive, negative = _variation_metrics(
        normalized_delta, lowess_delta_deadzone
    )
    zero_fraction = float(np.mean(np.abs(normalized_delta) < lowess_delta_deadzone))

    pearson = _safe_pearson(xv, smooth)
    spearman = _safe_spearman(xv, smooth)
    amplitude = float(
        (np.quantile(smooth, 0.90) - np.quantile(smooth, 0.10)) / typical
    )
    roughness = float(np.median(np.abs(wv - smooth)) / typical)

    net_n = max(1, min(int(net_edge_bins), len(smooth) // 3))
    net_left_level = float(np.median(smooth[:net_n]))
    net_right_level = float(np.median(smooth[-net_n:]))
    net_change = float((net_right_level - net_left_level) / typical)

    peak_index = int(np.argmax(smooth))
    peak_width = float(smooth[peak_index])

    # Estimate the two edge baselines without ever including the peak itself.
    # This matters for a late peak (e.g. bin 13/14 of 15): the old fixed last-
    # three-bin median could contain the peak and erase a real terminal drop.
    edge_n = max(1, min(int(edge_bins), len(smooth) // 3))
    pre_peak_n = min(edge_n, peak_index)
    post_peak_n = min(edge_n, len(smooth) - peak_index - 1)
    left_level = (
        float(np.median(smooth[:pre_peak_n]))
        if pre_peak_n > 0 else peak_width
    )
    right_level = (
        float(np.median(smooth[-post_peak_n:]))
        if post_peak_n > 0 else peak_width
    )
    peak_rise = float((peak_width - left_level) / typical)
    peak_drop = float((peak_width - right_level) / typical)
    peak_prominence = float(min(peak_rise, peak_drop))
    end_high_gap = float((np.max(smooth) - smooth[-1]) / typical)
    end_low_gap = float((smooth[-1] - np.min(smooth)) / typical)

    peak_left_coherence = _direction_share(
        normalized_delta[:peak_index], lowess_delta_deadzone, +1
    )
    peak_right_coherence = _direction_share(
        normalized_delta[peak_index:], lowess_delta_deadzone, -1
    )

    peak_interior = bool(0 < peak_index < len(smooth) - 1)
    clean_peak = bool(
        peak_interior
        and peak_prominence >= peak_prominence_threshold
        and peak_left_coherence >= structure_side_coherence_threshold
        and peak_right_coherence >= structure_side_coherence_threshold
    )

    trough_index = int(np.argmin(smooth))
    trough_width = float(smooth[trough_index])
    pre_trough_n = min(edge_n, trough_index)
    post_trough_n = min(edge_n, len(smooth) - trough_index - 1)
    trough_left_level = (
        float(np.median(smooth[:pre_trough_n]))
        if pre_trough_n > 0 else trough_width
    )
    trough_right_level = (
        float(np.median(smooth[-post_trough_n:]))
        if post_trough_n > 0 else trough_width
    )
    trough_drop = float((trough_left_level - trough_width) / typical)
    trough_rise = float((trough_right_level - trough_width) / typical)
    trough_prominence = float(min(trough_drop, trough_rise))
    trough_left_coherence = _direction_share(
        normalized_delta[:trough_index], lowess_delta_deadzone, -1
    )
    trough_right_coherence = _direction_share(
        normalized_delta[trough_index:], lowess_delta_deadzone, +1
    )
    trough_interior = bool(0 < trough_index < len(smooth) - 1)
    clean_trough = bool(
        trough_interior
        and trough_prominence >= peak_prominence_threshold
        and trough_left_coherence >= structure_side_coherence_threshold
        and trough_right_coherence >= structure_side_coherence_threshold
    )

    # Step 1: Constant / Stable.  Unchanged thresholds.
    stable = bool(
        amplitude < stable_amplitude_threshold
        and abs(net_change) < stable_net_change_threshold
        and zero_fraction >= stable_zero_fraction_threshold
    )
    strength = "Stable" if stable else "Variable"

    # Step 2: global main candidate only.  Secondary reverse structure can
    # veto an impure peak/trough; it never creates a peak from a trend.
    if stable:
        structure = "Simple"
    elif clean_peak and clean_trough:
        structure = (
            "Interior peak"
            if peak_prominence >= trough_prominence
            else "Interior trough"
        )
    elif clean_peak:
        structure = "Interior peak"
    elif clean_trough:
        structure = "Interior trough"
    elif reversal < structure_reversal_threshold:
        structure = "Simple"
    else:
        structure = "Irregular"

    if structure == "Interior peak":
        structure_p1 = peak_prominence
        structure_p2 = trough_prominence if trough_interior else 0.0
    elif structure == "Interior trough":
        structure_p1 = trough_prominence
        structure_p2 = peak_prominence if peak_interior else 0.0
    else:
        structure_p1 = np.nan
        structure_p2 = np.nan

    structure_vetoed = False
    if structure in {"Interior peak", "Interior trough"}:
        if (
            structure_p2 >= secondary_prominence_threshold
            and structure_p1 < structure_dominance_ratio * structure_p2
            # and reversal >= veto_reversal_threshold
        ):
            structure = "Irregular"
            structure_vetoed = True

    structure_group = (
        "Simple" if structure == "Simple" or stable else "Complex"
    )

    # Direction: Spearman only.  Not a structure gate; never promotes Irregular.
    if spearman >= spearman_threshold:
        direction = "Increasing"
        direction_confidence = "Spearman"
    elif spearman <= -spearman_threshold:
        direction = "Decreasing"
        direction_confidence = "Spearman"
    else:
        direction = "No dominant direction"
        direction_confidence = "Not applicable"

    if stable:
        direction = "No dominant direction"
        direction_confidence = "Not applicable"
        label = "Stable"
        display_label = "Stable"
    elif structure in {"Interior peak", "Interior trough", "Irregular"}:
        label = structure
        display_label = (
            f"{structure} ({direction})"
            if direction in {"Increasing", "Decreasing"}
            else structure
        )
    elif direction in {"Increasing", "Decreasing"}:
        label = direction
        display_label = direction
    else:
        label = "No dominant direction"
        display_label = "No dominant direction"

    # Report an effective TP pattern consistent with the interpreted shape.
    # Keep the uncollapsed LOWESS sign-run result in separate diagnostic fields.
    raw_lowess_runs = runs
    if structure == "Interior peak":
        effective_runs = (1, -1)
    elif structure == "Interior trough":
        effective_runs = (-1, 1)
    elif structure == "Simple" and direction == "Increasing":
        effective_runs = (1,)
    elif structure == "Simple" and direction == "Decreasing":
        effective_runs = (-1,)
    elif stable:
        effective_runs = ()
    else:
        effective_runs = raw_lowess_runs

    # Preserve the base/raw profile metrics under explicit names before the
    # public fields are replaced by their LOWESS-profile counterparts.
    result["buffer_raw_profile_pearson"] = result.get("buffer_pearson", np.nan)
    result["buffer_raw_profile_spearman"] = result.get("buffer_spearman", np.nan)
    result["buffer_raw_profile_amplitude"] = result.get("buffer_amplitude", np.nan)

    result.update({
        "spread_label": label,
        "spread_display_label": display_label,
        "spread_strength": strength,
        "spread_direction": direction,
        "spread_direction_confidence": direction_confidence,
        "spread_structure": structure,
        "spread_structure_group": structure_group,
        "buffer_structure_p1": float(structure_p1) if np.isfinite(structure_p1) else np.nan,
        "buffer_structure_p2": float(structure_p2) if np.isfinite(structure_p2) else np.nan,
        "buffer_structure_vetoed": bool(structure_vetoed),
        "spread_status": "ok",
        "spread_lowess_status": "ok",
        "spread_profile_method": "lowess_on_15_detrended_q05_q95_widths",
        "spread_lowess_frac": float(lowess_frac),
        "spread_lowess_it": int(lowess_it),
        "spread_lowess_delta_deadzone": float(lowess_delta_deadzone),
        "buffer_pearson": float(pearson),
        "buffer_spearman": float(spearman),
        "buffer_amplitude": float(amplitude),
        "buffer_zero_fraction": zero_fraction,
        "buffer_reversal_ratio": float(reversal),
        "buffer_positive_variation": float(positive),
        "buffer_negative_variation": float(negative),
        "buffer_lowess_raw_tp_count": max(0, len(raw_lowess_runs) - 1),
        "buffer_lowess_raw_tp_pattern": "-".join(
            "up" if s > 0 else "down" for s in raw_lowess_runs
        ) or "none",
        "buffer_tp_count": max(0, len(effective_runs) - 1),
        "buffer_tp_pattern": "-".join(
            "up" if s > 0 else "down" for s in effective_runs
        ) or "none",
        "buffer_tp_bin": int(peak_index + 1) if clean_peak else np.nan,
        "buffer_peak_bin": int(peak_index + 1),
        "buffer_pre_peak_n": int(pre_peak_n),
        "buffer_post_peak_n": int(post_peak_n),
        "buffer_peak_prominence": peak_prominence,
        "buffer_peak_rise": peak_rise,
        "buffer_post_peak_drop": peak_drop,
        "buffer_end_high_gap": end_high_gap,
        "buffer_end_low_gap": end_low_gap,
        "buffer_fractional_drop": (
            float((peak_width - right_level) / peak_width) if peak_width > 0 else 0.0
        ),
        "buffer_clean_peak": clean_peak,
        "buffer_lowess_roughness": roughness,
        "buffer_net_edge_bins": int(net_n),
        "buffer_net_left_level": net_left_level,
        "buffer_net_right_level": net_right_level,
        "buffer_net_change": net_change,
        "buffer_peak_left_coherence": peak_left_coherence,
        "buffer_peak_right_coherence": peak_right_coherence,
        "buffer_trough_bin": int(trough_index + 1),
        "buffer_pre_trough_n": int(pre_trough_n),
        "buffer_post_trough_n": int(post_trough_n),
        "buffer_trough_prominence": trough_prominence,
        "buffer_trough_drop": trough_drop,
        "buffer_post_trough_rise": trough_rise,
        "buffer_trough_left_coherence": trough_left_coherence,
        "buffer_trough_right_coherence": trough_right_coherence,
        "buffer_clean_trough": clean_trough,
    })

    valid_indices = np.flatnonzero(valid)
    for local_i, source_i in enumerate(valid_indices):
        result[f"buffer_width_smooth_b{source_i + 1:02d}"] = float(smooth[local_i])
    for j in range(1, n_bins):
        result[f"buffer_width_delta_b{j:02d}"] = np.nan
    for local_i in range(len(normalized_delta)):
        source_i = valid_indices[local_i]
        next_i = valid_indices[local_i + 1]
        if next_i == source_i + 1:
            result[f"buffer_width_delta_b{source_i + 1:02d}"] = float(
                normalized_delta[local_i]
            )
    return result


def classify_job(job: dict[str, Any]) -> dict[str, Any]:
    metadata = {key: value for key, value in job.items() if key not in {"x", "y"}}
    result = classify_spread(
        job.get("x", np.array([])),
        job.get("y", np.array([])),
        eps_filter=job.get("eps_filter"),
        n_bins=int(job.get("n_bins", N_BINS)),
        min_panel_n=int(job.get("min_panel_n", MIN_PANEL_N)),
        min_bin_n=int(job.get("min_bin_n", MIN_BIN_N)),
        edge_bins=int(job.get("edge_bins", EDGE_BINS)),
        lowess_frac=float(job.get("lowess_frac", LOWESS_FRAC)),
        lowess_it=int(job.get("lowess_it", LOWESS_IT)),
        lowess_delta_deadzone=float(
            job.get("lowess_delta_deadzone", LOWESS_DELTA_DEADZONE)
        ),
        stable_amplitude_threshold=float(
            job.get("stable_amplitude_threshold", STABLE_AMPLITUDE_THRESHOLD)
        ),
        stable_zero_fraction_threshold=float(
            job.get(
                "stable_zero_fraction_threshold",
                STABLE_ZERO_FRACTION_THRESHOLD,
            )
        ),
        stable_net_change_threshold=float(
            job.get("stable_net_change_threshold", STABLE_NET_CHANGE_THRESHOLD)
        ),
        spearman_threshold=float(
            job.get("spearman_threshold", SPEARMAN_THRESHOLD)
        ),
        pearson_threshold=float(job.get("pearson_threshold", PEARSON_THRESHOLD)),
        direction_net_change_threshold=float(
            job.get("direction_net_change_threshold", DIRECTION_NET_CHANGE_THRESHOLD)
        ),
        structure_reversal_threshold=float(
            job.get("structure_reversal_threshold", STRUCTURE_REVERSAL_THRESHOLD)
        ),
        structure_side_coherence_threshold=float(
            job.get(
                "structure_side_coherence_threshold",
                STRUCTURE_SIDE_COHERENCE_THRESHOLD,
            )
        ),
        structure_dominance_ratio=float(
            job.get("structure_dominance_ratio", STRUCTURE_DOMINANCE_RATIO)
        ),
        peak_prominence_threshold=float(
            job.get("peak_prominence_threshold", PEAK_PROMINENCE_THRESHOLD)
        ),
        peak_rise_threshold=float(
            job.get("peak_rise_threshold", PEAK_RISE_THRESHOLD)
        ),
        peak_drop_threshold=float(
            job.get("peak_drop_threshold", PEAK_DROP_THRESHOLD)
        ),
        trough_drop_threshold=float(
            job.get("trough_drop_threshold", TROUGH_DROP_THRESHOLD)
        ),
        trough_rise_threshold=float(
            job.get("trough_rise_threshold", TROUGH_RISE_THRESHOLD)
        ),
        secondary_prominence_threshold=float(
            job.get(
                "secondary_prominence_threshold",
                SECONDARY_PROMINENCE_THRESHOLD,
            )
        ),
        veto_reversal_threshold=float(
            job.get("veto_reversal_threshold", VETO_REVERSAL_THRESHOLD)
        ),
        net_edge_bins=int(job.get("net_edge_bins", NET_EDGE_BINS)),
    )
    return {**metadata, **result}


def classify_batch(
    jobs: Iterable[dict[str, Any]],
    *,
    n_jobs: int = -1,
) -> list[dict[str, Any]]:
    jobs = list(jobs)
    if n_jobs == 1:
        return [classify_job(job) for job in jobs]
    try:
        from joblib import Parallel, delayed

        return Parallel(n_jobs=n_jobs)(delayed(classify_job)(job) for job in jobs)
    except ImportError:  # pragma: no cover
        return [classify_job(job) for job in jobs]
