"""Extract a few numbers from each panel's LOWESS so 510 figures are not read by eye.

Primary curve: statsmodels lowess, frac=0.2, it=3, continuous |v| > EPS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
from minepy import MINE as _MINE

EPS = 1e-4
MIN_N = 30
LOWESS_FRAC = 0.2
LOWESS_IT = 3
N_GRID = 40
N_BINS = 5
MIN_BIN = 15
CENTER_N_BINS = 10
CENTER_MIN_BIN = 10


def _iqr(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 4:
        return np.nan
    q75, q25 = np.percentile(a, [75, 25])
    return float(q75 - q25)


def _sd(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 4:
        return np.nan
    return float(np.std(a, ddof=1))


def _med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 4:
        return np.nan
    return float(np.median(a))


def _ratio(num, den, floor):
    if not np.isfinite(num) or not np.isfinite(den) or den < floor:
        return np.nan
    return float(num / den)


def center_features(x, y, n_bins=CENTER_N_BINS, min_bin=CENTER_MIN_BIN):
    """Conditional-centre diagnostics from equal-count x bins.

    Each bin contributes the median x and median y. Local slopes are
    dimensionless because both axes are scaled by their full-sample IQR.
    """
    out = {
        "center_bins_used": 0,
        "center_amplitude_A50": np.nan,
        "center_slope_typ": np.nan,
        "center_slope_peak": np.nan,
        "center_slope_signed": np.nan,
        "center_rho": np.nan,
        "center_turns": np.nan,
    }
    for k in range(1, n_bins + 1):
        out[f"center_x_b{k}"] = np.nan
        out[f"center_y_b{k}"] = np.nan
    for k in range(1, n_bins):
        out[f"center_slope_b{k}"] = np.nan

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < max(3, n_bins * min_bin):
        return out

    frame = pd.DataFrame({"x": x, "y": y})
    try:
        frame["bin"] = pd.qcut(frame["x"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return out
    grouped = (
        frame.dropna(subset=["bin"])
        .groupby("bin", observed=True)
        .agg(x=("x", "median"), y=("y", "median"), n=("y", "size"))
        .sort_index()
    )
    grouped = grouped[grouped["n"] >= min_bin]
    if len(grouped) < 3:
        return out

    x_med = grouped["x"].to_numpy(float)
    y_med = grouped["y"].to_numpy(float)
    out["center_bins_used"] = int(len(grouped))
    for k, (xm, ym) in enumerate(zip(x_med, y_med), start=1):
        if k > n_bins:
            break
        out[f"center_x_b{k}"] = float(xm)
        out[f"center_y_b{k}"] = float(ym)

    x_scale = _iqr(x)
    y_scale = _iqr(y)
    if not np.isfinite(x_scale) or x_scale <= 0:
        x_scale = float(np.ptp(x))
    if not np.isfinite(y_scale) or y_scale <= 0:
        y_scale = float(np.ptp(y))
    if not (np.isfinite(x_scale) and x_scale > 0 and np.isfinite(y_scale) and y_scale > 0):
        return out

    floor = max(1e-12, 1e-12 * y_scale)
    out["center_amplitude_A50"] = _ratio(_iqr(y_med), y_scale, floor)
    dx = np.diff(x_med) / x_scale
    dy = np.diff(y_med) / y_scale
    slopes = np.full_like(dx, np.nan, dtype=float)
    valid = np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) > 1e-12)
    slopes[valid] = dy[valid] / dx[valid]
    for k, slope in enumerate(slopes, start=1):
        if k >= n_bins:
            break
        out[f"center_slope_b{k}"] = float(slope) if np.isfinite(slope) else np.nan

    slopes_f = slopes[np.isfinite(slopes)]
    if len(slopes_f):
        abs_slopes = np.abs(slopes_f)
        out["center_slope_typ"] = float(np.median(abs_slopes))
        out["center_slope_peak"] = float(np.percentile(abs_slopes, 90))
        out["center_slope_signed"] = float(np.median(slopes_f))
        deadzone = 0.10 * float(np.median(abs_slopes))
        signs = np.sign(slopes_f[np.abs(slopes_f) > deadzone])
        out["center_turns"] = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    if np.ptp(x_med) > 0 and np.ptp(y_med) > 0:
        out["center_rho"] = float(
            pd.Series(x_med).corr(pd.Series(y_med), method="spearman")
        )
    return out


def spread_features(x, y, xs=None, ys=None, n_bins=N_BINS, min_bin=MIN_BIN):
    """Robust fan / heteroscedasticity diagnostics around the LOWESS mean.

    Bins are equal-count in x (quantiles).  The primary spread in each bin is
    IQR(y - y_hat), so a changing conditional mean is not mistaken for a
    changing cloud width.  ``spread_contrast`` is bounded near [-1, 1] and is
    therefore safer than a raw high/low ratio when the low-x spread is tiny.
    """
    nan = {
        "y_iqr_ratio": np.nan,
        "y_sd_ratio": np.nan,
        "absr_med_ratio": np.nan,
        "absr_iqr_ratio": np.nan,
        "spread_ratio": np.nan,
        "spearman_absr_x": np.nan,
        "spread_level_rel": np.nan,
        "spread_low_iqr": np.nan,
        "spread_high_iqr": np.nan,
        "spread_contrast": np.nan,
        "spread_trend_rho": np.nan,
        "spread_variation": np.nan,
        "spread_mid_prominence": np.nan,
        "spread_peak_bin": np.nan,
        "n_bin_first": 0,
        "n_bin_last": 0,
        "n_bins_used": 0,
    }
    for k in range(1, n_bins + 1):
        nan[f"spread_iqr_b{k}"] = np.nan
        nan[f"spread_rel_b{k}"] = np.nan
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < n_bins * min_bin:
        return nan
    try:
        codes = pd.qcut(x, n_bins, labels=False, duplicates="drop")
    except ValueError:
        return nan
    codes = np.asarray(codes)
    used = [c for c in np.unique(codes) if np.isfinite(c)]
    if len(used) < 2:
        return nan
    c0, c1 = int(min(used)), int(max(used))
    m0, m1 = codes == c0, codes == c1
    n0, n1 = int(m0.sum()), int(m1.sum())
    nan["n_bin_first"] = n0
    nan["n_bin_last"] = n1
    nan["n_bins_used"] = int(len(used))
    if n0 < min_bin or n1 < min_bin:
        return nan

    y_scale = _iqr(y)
    if not np.isfinite(y_scale) or y_scale <= 0:
        y_scale = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
    floor = max(1e-6 * max(y_scale, 1.0), 1e-12)

    y_iqr_ratio = _ratio(_iqr(y[m1]), _iqr(y[m0]), floor)
    y_sd_ratio = _ratio(_sd(y[m1]), _sd(y[m0]), floor)

    absr_med_ratio = np.nan
    absr_iqr_ratio = np.nan
    spear = np.nan
    spread_iqrs = []
    spread_codes = []
    if xs is not None and ys is not None:
        yhat = np.interp(x, xs, ys)
        r = y - yhat
        ar = np.abs(r)
        absr_med_ratio = _ratio(_med(ar[m1]), _med(ar[m0]), floor)
        absr_iqr_ratio = _ratio(_iqr(ar[m1]), _iqr(ar[m0]), floor)
        if np.isfinite(ar).sum() >= 30 and np.ptp(x) > 0:
            spear = float(pd.Series(x).corr(pd.Series(ar), method="spearman"))

        for j, code in enumerate(sorted(used), start=1):
            mk = codes == code
            sk = _iqr(r[mk]) if int(mk.sum()) >= min_bin else np.nan
            nan[f"spread_iqr_b{j}"] = sk
            nan[f"spread_rel_b{j}"] = (
                sk / y_scale
                if np.isfinite(sk) and np.isfinite(y_scale) and y_scale > floor
                else np.nan
            )
            if np.isfinite(sk):
                spread_iqrs.append(float(sk))
                spread_codes.append(float(j))

        if len(spread_iqrs) >= 2:
            n_edge = min(2, len(spread_iqrs) // 2)
            spread_low = float(np.median(spread_iqrs[:n_edge]))
            spread_high = float(np.median(spread_iqrs[-n_edge:]))
            spread_level_rel = (
                float(np.median(spread_iqrs) / y_scale)
                if np.isfinite(y_scale) and y_scale > floor
                else np.nan
            )
            spread_contrast = float(
                (spread_high - spread_low) / (spread_high + spread_low + floor)
            )
            if np.ptp(spread_iqrs) == 0:
                spread_trend = 0.0
            else:
                spread_trend = float(
                    pd.Series(spread_codes).corr(pd.Series(spread_iqrs), method="spearman")
                )
            spread_min = float(np.min(spread_iqrs))
            spread_max = float(np.max(spread_iqrs))
            spread_variation = float(
                (spread_max - spread_min) / (spread_max + spread_min + floor)
            )
            spread_peak_bin = int(spread_codes[int(np.argmax(spread_iqrs))])

            spread_mid_prominence = np.nan
            spread_by_bin = {
                int(code): float(value)
                for code, value in zip(spread_codes, spread_iqrs)
            }
            if 1 in spread_by_bin and n_bins in spread_by_bin:
                interior = [
                    spread_by_bin[k]
                    for k in range(2, n_bins)
                    if k in spread_by_bin
                ]
                if interior:
                    mid_max = float(np.max(interior))
                    edge_max = max(spread_by_bin[1], spread_by_bin[n_bins])
                    spread_mid_prominence = float(
                        (mid_max - edge_max) / (mid_max + edge_max + floor)
                    )
            nan.update({
                "spread_level_rel": spread_level_rel,
                "spread_low_iqr": spread_low,
                "spread_high_iqr": spread_high,
                "spread_contrast": spread_contrast,
                "spread_trend_rho": spread_trend,
                "spread_variation": spread_variation,
                "spread_mid_prominence": spread_mid_prominence,
                "spread_peak_bin": spread_peak_bin,
            })

    nan.update({
        "y_iqr_ratio": y_iqr_ratio,
        "y_sd_ratio": y_sd_ratio,
        "absr_med_ratio": absr_med_ratio,
        "absr_iqr_ratio": absr_iqr_ratio,
        "spread_ratio": absr_med_ratio,
        "spearman_absr_x": spear,
        "n_bin_first": n0,
        "n_bin_last": n1,
        "n_bins_used": int(len(used)),
    })
    return nan


LOWESS_EQ_BINS = 10
LOWESS_SHAPE_PTS = 15
SHAPE_DEADZONE = 0.02


RREV_THRESHOLD = 0.05
PEARSON_MONO_THRESHOLD = 0.50


def _count_turning_points(d, deadzone):
    """Count sign changes in d-segment sequence, ignoring deadzone segments."""
    signs = []
    for dv in d:
        if dv >= deadzone:
            signs.append(1)
        elif dv <= -deadzone:
            signs.append(-1)
    n_tp = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            n_tp += 1
    return n_tp


def lowess_shape_classify(x, y, xs, ys, n_pts=LOWESS_SHAPE_PTS,
                          deadzone=SHAPE_DEADZONE,
                          spacing="equal"):
    """Classify mean response shape from LOWESS.

    Primary classification uses reversal ratio plus EQ15 Pearson:
    R_rev >= 0.05 is Non-monotonic; otherwise |Pearson| >= 0.5 confirms
    Monotonic, and lower |Pearson| is retained as Candidate monotonic.
    Turning points are retained for secondary-shape diagnostics.
    """
    n_segs = n_pts - 1
    out = {
        "lowess_shape": "skipped",
        "lowess_shape2": "",
        "lowess_shape_flag": "",
        "ls_p0": np.nan,
        "ls_p_pos": np.nan,
        "ls_p_neg": np.nan,
        "ls_n0": 0,
        "ls_n_pos": 0,
        "ls_n_neg": 0,
        "ls_V_pos": np.nan,
        "ls_V_neg": np.nan,
        "ls_R_rev": np.nan,
        "ls_G": np.nan,
        "ls_H_U": np.nan,
        "ls_pearson": np.nan,
        "ls_abs_pearson": np.nan,
        "ls_n_tp": 0,
    }
    for k in range(1, n_segs + 1):
        out[f"ls_d{k}"] = np.nan

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[finite], y[finite]
    if len(xf) < n_pts * 3:
        out["lowess_shape"] = "too_few"
        return out

    y_scale = float(np.percentile(yf, 95) - np.percentile(yf, 5))
    if y_scale <= 0:
        y_scale = float(np.ptp(yf))
    if y_scale <= 0:
        out["lowess_shape"] = "no_variation"
        return out

    if spacing == "freq":
        pcts = np.linspace(100.0 / (2 * n_pts),
                           100.0 - 100.0 / (2 * n_pts), n_pts)
        x_eq = np.percentile(xf, pcts)
    else:
        x_eq = np.linspace(xf.min(), xf.max(), n_pts)
    if np.ptp(x_eq) <= 0:
        out["lowess_shape"] = "no_variation"
        return out
    y_eq = np.interp(x_eq, xs, ys)

    d = np.diff(y_eq) / y_scale
    for k, dv in enumerate(d, start=1):
        out[f"ls_d{k}"] = float(dv)

    n0 = int(np.sum(np.abs(d) < deadzone))
    n_pos = int(np.sum(d >= deadzone))
    n_neg = int(np.sum(d <= -deadzone))

    out["ls_n0"] = n0
    out["ls_n_pos"] = n_pos
    out["ls_n_neg"] = n_neg
    out["ls_p0"] = n0 / n_segs
    out["ls_p_pos"] = n_pos / n_segs
    out["ls_p_neg"] = n_neg / n_segs

    d_sig = d[np.abs(d) >= deadzone]
    V_pos = float(np.sum(d_sig[d_sig > 0]))
    V_neg = float(np.sum(-d_sig[d_sig < 0]))
    out["ls_V_pos"] = V_pos
    out["ls_V_neg"] = V_neg
    total_V = V_pos + V_neg
    out["ls_R_rev"] = float(min(V_pos, V_neg) / total_V) if total_V > 1e-12 else 0.0

    G = float((np.max(y_eq) - np.min(y_eq)) / y_scale)
    out["ls_G"] = G

    # --- EQ15 Pearson ---
    if np.ptp(y_eq) <= 0:
        out["ls_pearson"] = 0.0
        out["ls_abs_pearson"] = 0.0
    else:
        pr = float(np.corrcoef(x_eq, y_eq)[0, 1])
        out["ls_pearson"] = pr
        out["ls_abs_pearson"] = abs(pr)

    # --- Turning points: secondary diagnostic only ---
    n_tp = _count_turning_points(d, deadzone)
    out["ls_n_tp"] = n_tp

    # --- 1. Directionally consistent: confirmed/candidate monotonic ---
    if out["ls_R_rev"] < RREV_THRESHOLD:
        endpoint_change = float(y_eq[-1] - y_eq[0])
        direction_signal = (
            endpoint_change if abs(endpoint_change) > 1e-12
            else out["ls_pearson"]
        )
        direction = "up" if direction_signal >= 0 else "down"
        prefix = (
            "monotonic"
            if out["ls_abs_pearson"] >= PEARSON_MONO_THRESHOLD
            else "candidate_monotonic"
        )
        out["lowess_shape"] = f"{prefix}_{direction}"
        if prefix == "candidate_monotonic":
            out["lowess_shape_flag"] = "low_pearson"
        d_abs = np.abs(d)
        d_abs_nz = d_abs[d_abs >= deadzone]
        if len(d_abs_nz) >= 3:
            tail_sum = float(np.sum(sorted(d_abs_nz)[-2:]))
            if tail_sum / float(np.sum(d_abs_nz)) > 0.70:
                nz_idx = np.where(d_abs >= deadzone)[0]
                if nz_idx[0] <= 1 or nz_idx[-1] >= n_segs - 2:
                    out["lowess_shape_flag"] = "tail_driven"

        BOW_THRESHOLD = 0.15

        def _bow_classify(y_arr):
            chord = y_arr[0] + (y_arr[-1] - y_arr[0]) * np.linspace(0, 1, len(y_arr))
            e = y_arr - chord
            drop = abs(y_arr[-1] - y_arr[0])
            b = float(np.mean(np.abs(e[1:-1])) / drop) if drop > 0 else 0.0
            sign_d = 1.0 if y_arr[-1] > y_arr[0] else -1.0
            e_star = sign_d * e[1:-1]
            if b < BOW_THRESHOLD:
                cls = "linear"
            elif float(np.mean(e_star)) < 0:
                cls = "acceleration"
            else:
                cls = "saturation"
            return b, cls

        B, shape2 = _bow_classify(y_eq)
        out["ls_bow"] = B
        out["lowess_shape2"] = shape2

        i05 = max(1, int(round(0.05 * (n_pts - 1))))
        i95 = min(n_pts - 2, int(round(0.95 * (n_pts - 1))))
        y_trim = y_eq[i05:i95 + 1]
        Bp, shape2p = _bow_classify(y_trim)
        out["ls_bow_p"] = Bp
        out["lowess_shape2_p"] = shape2p

        return out

    # --- 2. Non-monotonic: substantial movement in both directions ---
    out["lowess_shape"] = "nonmonotonic"

    SYMMETRY_THRESHOLD = 0.40

    if n_tp == 1:
        first_sign = next((1 if dv >= deadzone else -1)
                          for dv in d if abs(dv) >= deadzone)
        if first_sign == -1:
            i_min = int(np.argmin(y_eq))
            left_arm = y_eq[0] - y_eq[i_min]
            right_arm = y_eq[-1] - y_eq[i_min]
        else:
            i_max = int(np.argmax(y_eq))
            left_arm = y_eq[i_max] - y_eq[0]
            right_arm = y_eq[i_max] - y_eq[-1]
        max_arm = max(left_arm, right_arm)
        sym = min(left_arm, right_arm) / max_arm if max_arm > 0 else 0.0
        if first_sign == 1:
            out["lowess_shape2"] = "inverted_U"
        else:
            if sym >= SYMMETRY_THRESHOLD:
                out["lowess_shape2"] = "U_shaped"
            else:
                out["lowess_shape2"] = "J_shaped" if y_eq[-1] > y_eq[0] else "inverted_J"
    elif n_tp == 2:
        out["lowess_shape2"] = "S_or_N"
    else:
        out["lowess_shape2"] = "complex"

    return out


def lowess_equifreq_diagnostics(x, y, xs, ys, n_bins=LOWESS_EQ_BINS):
    """S_typ and reversal ratio from LOWESS evaluated at equal-frequency x positions."""
    out = {
        "lowess_slope_typ": np.nan,
        "lowess_slope_peak": np.nan,
        "lowess_slope_signed": np.nan,
        "lowess_reversal_ratio": np.nan,
    }
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x)
    xf = x[finite]
    if len(xf) < n_bins * 3:
        return out
    pcts = np.linspace(5, 95, n_bins)
    x_eq = np.percentile(xf, pcts)
    if np.ptp(x_eq) <= 0:
        return out
    y_eq = np.interp(x_eq, xs, ys)

    x_scale = _iqr(x)
    y_scale = _iqr(y)
    if not (np.isfinite(x_scale) and x_scale > 0):
        x_scale = float(np.ptp(x))
    if not (np.isfinite(y_scale) and y_scale > 0):
        y_scale = float(np.ptp(y))
    if not (x_scale > 0 and y_scale > 0):
        return out

    dx = np.diff(x_eq) / x_scale
    dy_raw = np.diff(y_eq)
    dy = dy_raw / y_scale
    valid = np.abs(dx) > 1e-12
    slopes = np.where(valid, dy / dx, np.nan)
    slopes_f = slopes[np.isfinite(slopes)]
    if len(slopes_f) == 0:
        return out

    abs_s = np.abs(slopes_f)
    out["lowess_slope_typ"] = float(np.median(abs_s))
    out["lowess_slope_peak"] = float(np.percentile(abs_s, 90))
    out["lowess_slope_signed"] = float(np.median(slopes_f))

    pos = float(np.sum(dy_raw[dy_raw > 0]))
    neg = float(np.abs(np.sum(dy_raw[dy_raw < 0])))
    total = pos + neg
    out["lowess_reversal_ratio"] = float(min(pos, neg) / total) if total > 1e-12 else 0.0
    return out


def fit_lowess(x, y, frac=LOWESS_FRAC, it=LOWESS_IT):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < MIN_N or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return None
    try:
        z = _lowess(y, x, frac=frac, it=it, return_sorted=True)
    except Exception:
        return None
    if z is None or len(z) < 8:
        return None
    xs, ys = z[:, 0], z[:, 1]
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        return None
    return xs, ys


def _interp_grid(xs, ys, n=N_GRID):
    xg = np.linspace(float(xs.min()), float(xs.max()), n)
    yg = np.interp(xg, xs, ys)
    return xg, yg


def _sign_changes(yg, amp, rel=0.03):
    d = np.diff(yg)
    thr = rel * max(amp, 1e-12)
    d = np.where(np.abs(d) < thr, 0.0, d)
    s = np.sign(d)
    s = s[s != 0]
    if len(s) < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


def _onset_frac(xg, yg):
    span = yg[-1] - yg[0]
    if abs(span) < 1e-12:
        return np.nan
    target = yg[0] + 0.15 * span
    if span > 0:
        hit = np.where(yg >= target)[0]
    else:
        hit = np.where(yg <= target)[0]
    if len(hit) == 0:
        return np.nan
    xr = xg[-1] - xg[0]
    if xr <= 0:
        return np.nan
    return float((xg[hit[0]] - xg[0]) / xr)


def _third_slopes(xg, yg):
    xmin, xmax = float(xg[0]), float(xg[-1])
    xr = xmax - xmin
    if xr <= 0:
        return np.nan, np.nan
    cuts = xmin + np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]) * xr

    def sl(a, b):
        ya = np.interp(a, xg, yg)
        yb = np.interp(b, xg, yg)
        return (yb - ya) / (b - a)

    return float(sl(cuts[0], cuts[1])), float(sl(cuts[2], cuts[3]))


def _r2_to_curve(x, y, xs, ys):
    yhat = np.interp(x, xs, ys)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def coarse_from_rho_turns(rho, n_turns, amp):
    if not np.isfinite(amp) or amp < 0.12 or not np.isfinite(rho) or abs(rho) < 0.25:
        return "flat"
    if n_turns >= 2 and abs(rho) < 0.85:
        return "nonmonotonic"
    if rho >= 0.25:
        return "up"
    if rho <= -0.25:
        return "down"
    return "flat"


def shape_from_slopes(coarse, s0, s1, rho):
    if coarse in {"flat", "nonmonotonic", "down"}:
        return coarse
    # rising family
    s0p = s0 if np.isfinite(s0) else 0.0
    s1p = s1 if np.isfinite(s1) else 0.0
    floor = 0.05
    if s1p > 1.6 * max(s0p, floor) and s1p > 0:
        return "acceleration"
    if s0p > 1.6 * max(s1p, floor) and s0p > 0:
        return "saturation"
    if rho >= 0.7:
        return "near_linear"
    return "up"


def load_lowess_curves(path):
    """Read full LOWESS traces. Returns {(Pair, Model, Zone): (x, y_lowess)}."""
    path = Path(path)
    if not path.exists():
        return {}
    t = pd.read_parquet(path)
    out = {}
    for key, g in t.groupby(["Pair", "Model", "Zone"], sort=False):
        out[key] = (g["x"].to_numpy(dtype=float), g["y_lowess"].to_numpy(dtype=float))
    return out


def features_from_xy(x, y, frac=LOWESS_FRAC, fit=None, spacing="equal"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    out = {
        "n": int(len(x)),
        "response_iqr_abs": np.nan,
        "response_strength_A": np.nan,
        "y_iqr": np.nan,
        "amp": np.nan,
        "rho": np.nan,
        "spearman_r": np.nan,
        "mic": np.nan,
        "n_turns": np.nan,
        "onset_frac": np.nan,
        "s0": np.nan,
        "s1": np.nan,
        "s1_over_s0": np.nan,
        "r2": np.nan,
        "coarse": "skipped",
        "shape": "skipped",
        "onset": "skipped",
        "lowess_shape": "skipped",
        "lowess_shape2": "",
        "lowess_shape_flag": "",
        "ls_p0": np.nan,
        "ls_p_pos": np.nan,
        "ls_p_neg": np.nan,
        "ls_n0": 0,
        "ls_n_pos": 0,
        "ls_n_neg": 0,
        "ls_V_pos": np.nan,
        "ls_V_neg": np.nan,
        "ls_R_rev": np.nan,
        "ls_G": np.nan,
        "ls_H_U": np.nan,
        "ls_pearson": np.nan,
        "ls_abs_pearson": np.nan,
        "ls_n_tp": 0,
        "lowess_slope_typ": np.nan,
        "lowess_slope_peak": np.nan,
        "lowess_slope_signed": np.nan,
        "lowess_reversal_ratio": np.nan,
        "y_iqr_ratio": np.nan,
        "y_sd_ratio": np.nan,
        "absr_med_ratio": np.nan,
        "absr_iqr_ratio": np.nan,
        "spread_ratio": np.nan,
        "spearman_absr_x": np.nan,
        "spread_level_rel": np.nan,
        "spread_low_iqr": np.nan,
        "spread_high_iqr": np.nan,
        "spread_contrast": np.nan,
        "spread_trend_rho": np.nan,
        "spread_variation": np.nan,
        "spread_mid_prominence": np.nan,
        "spread_peak_bin": np.nan,
        "n_bin_first": 0,
        "n_bin_last": 0,
        "n_bins_used": 0,
    }
    for k in range(1, N_BINS + 1):
        out[f"spread_iqr_b{k}"] = np.nan
        out[f"spread_rel_b{k}"] = np.nan
    for k in range(1, LOWESS_SHAPE_PTS):
        out[f"ls_d{k}"] = np.nan
    out.update(center_features(x, y))
    if len(x) < MIN_N:
        out["shape"] = "too_few"
        out["coarse"] = "too_few"
        out["onset"] = "too_few"
        return out
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 10 and np.ptp(x[m]) > 0 and np.ptp(y[m]) > 0:
        out["spearman_r"] = float(
            pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman")
        )
        mine = _MINE()
        mine.compute_score(x[m].astype(float), y[m].astype(float))
        out["mic"] = float(mine.mic())
    if fit is None:
        fit = fit_lowess(x, y, frac=frac)
    if fit is None:
        out["shape"] = "fit_failed"
        out["coarse"] = "fit_failed"
        out["onset"] = "fit_failed"
        return out
    xs, ys = fit
    yhat_obs = np.interp(x, xs, ys)
    response_iqr_abs = _iqr(yhat_obs)
    y_iqr = _iqr(y)
    response_strength = (
        float(response_iqr_abs / y_iqr)
        if np.isfinite(response_iqr_abs) and np.isfinite(y_iqr) and y_iqr > 1e-12
        else np.nan
    )
    xg, yg = _interp_grid(xs, ys)
    y_scale = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
    if y_scale <= 0:
        y_scale = float(np.ptp(y)) if np.ptp(y) > 0 else 1.0
    amp = float((yg.max() - yg.min()) / y_scale)
    rho = float(pd.Series(xg).corr(pd.Series(yg), method="spearman"))
    n_turns = _sign_changes(yg, yg.max() - yg.min())
    onset = _onset_frac(xg, yg)
    s0_raw, s1_raw = _third_slopes(xg, yg)
    x_scale = float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))
    if x_scale <= 0:
        x_scale = float(np.ptp(x)) if np.ptp(x) > 0 else 1.0
    unit = y_scale / x_scale
    s0 = s0_raw / unit if np.isfinite(s0_raw) else np.nan
    s1 = s1_raw / unit if np.isfinite(s1_raw) else np.nan
    r2 = _r2_to_curve(x, y, xs, ys)
    coarse = coarse_from_rho_turns(rho, n_turns, amp)
    shape = shape_from_slopes(coarse, s0, s1, rho)
    if np.isfinite(onset) and coarse == "up":
        if onset < 0.15:
            onset_lab = "from_zero"
        elif onset > 0.30:
            onset_lab = "delayed"
        else:
            onset_lab = "mid"
    else:
        onset_lab = "na"

    out.update({
        "response_iqr_abs": response_iqr_abs,
        "response_strength_A": response_strength,
        "y_iqr": y_iqr,
        "amp": amp,
        "rho": rho,
        "n_turns": n_turns,
        "onset_frac": onset,
        "s0": s0,
        "s1": s1,
        "s1_over_s0": (s1 / s0) if (np.isfinite(s0) and abs(s0) > 1e-6) else np.nan,
        "r2": r2,
        "coarse": coarse,
        "shape": shape,
        "onset": onset_lab,
    })
    out.update(lowess_equifreq_diagnostics(x, y, xs, ys))
    out.update(lowess_shape_classify(x, y, xs, ys, spacing=spacing))
    out.update(spread_features(x, y, xs=xs, ys=ys))
    return out


def stability_label(x, y, primary=None):
    """Shape agreement at frac 0.2 / 0.3 / 0.4.

    ``high`` means the detailed shape agrees, ``medium`` means only the coarse
    family agrees, and ``low`` means even flat/up/down/nonmonotonic changes.
    """
    coarse_labs = []
    shape_labs = []
    for frac in (0.2, 0.3, 0.4):
        if frac == LOWESS_FRAC and primary is not None:
            f = primary
        else:
            f = features_from_xy(x, y, frac=frac)
        coarse_labs.append(f["coarse"])
        shape_labs.append(f["shape"])
    invalid = {"skipped", "too_few", "fit_failed"}
    valid = not any(lab in invalid for lab in coarse_labs)
    coarse_ok = valid and len(set(coarse_labs)) == 1
    shape_ok = valid and len(set(shape_labs)) == 1
    stability = "high" if shape_ok else "medium" if coarse_ok else "low"
    if not valid:
        stability = "not_available"
    return {
        "coarse_f20": coarse_labs[0],
        "coarse_f30": coarse_labs[1],
        "coarse_f40": coarse_labs[2],
        "shape_f20": shape_labs[0],
        "shape_f30": shape_labs[1],
        "shape_f40": shape_labs[2],
        "coarse_stable": bool(coarse_ok),
        "shape_stable": bool(shape_ok),
        "shape_stability": stability,
    }
