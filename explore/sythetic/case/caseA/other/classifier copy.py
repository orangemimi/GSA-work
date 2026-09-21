"""General-purpose scatter-plot relationship classifier.

Decision tree
─────────────
Gate 1  MIC ≥ 0.8 → strong global
        MIC ≥ 0.6 → intermediate global
        MIC < 0.6 → no global → "No Global Relationship"

Gate 2  |Pearson| ≥ 0.7 → Simple path
        |Pearson| < 0.7 → Complex path

Simple  power-law fit  f(x) = a·x^b + c
          R² ≥ threshold  &  |b−1| ≤ 0.05  → Linear
          R² ≥ threshold  &  |b−1| ≥ 0.5  &  autocorr ≤ 0.95 → Sat/Acc
          otherwise → falls to Complex

Complex
  1. Conditional dip test  frac_sig ≥ 0.25 → Branch
  2. SiZer turning-point consensus
       TP = 1 → U-shape
       TP = 2 → Cubic
       TP ≥ 3 → Oscillation
       TP = 0 → Hartigans' dip test on Y
                   significant → Transition
                   not significant → Uncertain
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from minepy import MINE as _MINE

try:
    from diptest import diptest as _diptest_func
except ImportError:
    _diptest_func = None


DEFAULTS: dict[str, Any] = {
    "mic_strong": 0.8,
    "mic_intermediate": 0.6,
    "pearson_threshold": 0.7,
    "r2_power_threshold": 0.5,

    "r2_autocorr_shortcut": False,
    "shortcut_autocorr_max": 0.7,

    "linear_b_tolerance": 0.1,
    "curvature_b_threshold": 0.1,
    "autocorr_threshold": 0.7,
    "use_autocorr_filter": False,



    "branch_n_bins": 10,
    "branch_frac_threshold": 0.50,
    "sizer_n_bandwidths": 4,
    "sizer_n_grid": 200,
    "sizer_h_min": 0.05,
    "sizer_h_max": 0.3, #0.18
    "sizer_interior": (0.05, 0.95),
    "sizer_min_run_frac": 0.05,
    "sizer_flat_frac": 0.1,
    "sizer_flat_ref": "max",
    "tp_support_min": 2,
}


# ═══════════════════════════════════════════════════════════
# MIC
# ═══════════════════════════════════════════════════════════

def compute_mic(x: np.ndarray, y: np.ndarray) -> float:
    mine = _MINE()
    mine.compute_score(x.astype(float), y.astype(float))
    return float(mine.mic())


# ═══════════════════════════════════════════════════════════
# Gate 1 — global relationship strength
# ═══════════════════════════════════════════════════════════

def check_global_relationship(mic: float, p: dict) -> str:
    if mic >= p["mic_strong"]:
        return "strong_global"
    if mic >= p["mic_intermediate"]:
        return "intermediate_global"
    return "no_global"


# ═══════════════════════════════════════════════════════════
# Gate 2 — Simple vs Complex
# ═══════════════════════════════════════════════════════════

def check_simple_or_complex(
    x: np.ndarray, y: np.ndarray, p: dict
) -> tuple[str, float]:
    r, _ = pearsonr(x, y)
    return ("Simple" if abs(r) >= p["pearson_threshold"] else "Complex"), float(r)


# ═══════════════════════════════════════════════════════════
# Simple path — power-law classification
# ═══════════════════════════════════════════════════════════

def _power_law(x, a, b, c):
    return a * np.power(np.maximum(x, 1e-10), b) + c


def _fit_power_law(x: np.ndarray, y: np.ndarray):
    valid = np.isfinite(x) & np.isfinite(y) & (x > 1e-6)
    if valid.sum() < 10:
        return None
    xv, yv = x[valid], y[valid]
    slope = np.polyfit(xv, yv, 1)[0]
    for b0 in (1.0, 0.5, 2.0, 0.3, 3.0):
        try:
            popt, _ = curve_fit(
                _power_law, xv, yv,
                p0=[slope, b0, np.median(yv)],
                maxfev=5000,
                bounds=([-np.inf, 0.05, -np.inf], [np.inf, 8.0, np.inf]),
            )
            y_pred = _power_law(xv, *popt)
            ss_res = np.sum((yv - y_pred) ** 2)
            ss_tot = np.sum((yv - yv.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if r2 >= 0:
                return (*popt, r2)
        except (RuntimeError, ValueError):
            continue
    return None


def _residual_autocorr(x, y, a, b, c):
    valid = np.isfinite(x) & np.isfinite(y) & (x > 1e-6)
    xv, yv = x[valid], y[valid]
    order = np.argsort(xv)
    resid = yv[order] - _power_law(xv[order], a, b, c)
    if len(resid) > 2:
        return np.corrcoef(resid[:-1], resid[1:])[0, 1]
    return 0.0


def classify_simple(x: np.ndarray, y: np.ndarray, p: dict) -> str | None:
    pw = _fit_power_law(x, y)
    if pw is None:
        return None
    a, b, c, r2 = pw
    if r2 < p["r2_power_threshold"]:
        return None
    if abs(b - 1) <= p["linear_b_tolerance"]:
        return "Linear"
    if abs(b - 1) >= p["curvature_b_threshold"]:
        if p["use_autocorr_filter"]:
            ac = _residual_autocorr(x, y, a, b, c)
            if ac > p["autocorr_threshold"]:
                return None
        return "Saturation" if b < 1 else "Acceleration"
    return None


# ═══════════════════════════════════════════════════════════
# Branch detection — conditional dip test
# ═══════════════════════════════════════════════════════════

def detect_branches(
    x: np.ndarray, y: np.ndarray, p: dict
) -> tuple[bool, float]:
    if _diptest_func is None:
        warnings.warn("diptest not installed — branch detection disabled")
        return False, 0.0
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]
    n_bins = p["branch_n_bins"]
    if len(xv) < 30:
        return False, 0.0
    bin_edges = np.quantile(xv, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1e-10
    bin_idx = np.digitize(xv, bin_edges[1:-1])
    n_sig = 0
    for b in range(n_bins):
        y_bin = yv[bin_idx == b]
        if len(y_bin) < 10:
            continue
        _, dip_p = _diptest_func(y_bin)
        if dip_p < 0.05:
            n_sig += 1
    frac = n_sig / n_bins
    return frac >= p["branch_frac_threshold"], frac


# ═══════════════════════════════════════════════════════════
# SiZer turning-point analysis
# ═══════════════════════════════════════════════════════════

def _count_stable_tps(deriv, x_positions, min_run_len, eps=0.0):
    n = len(deriv)
    if n < 2:
        return 0, []
    runs = []
    current_sign = 0
    current_start = 0
    for i in range(n):
        if np.isnan(deriv[i]) or abs(deriv[i]) < eps:
            if current_sign != 0 and (i - current_start) >= min_run_len:
                runs.append((current_sign, current_start, i - 1))
            current_sign = 0
            continue
        s = 1 if deriv[i] > 0 else -1
        if s != current_sign:
            if current_sign != 0 and (i - current_start) >= min_run_len:
                runs.append((current_sign, current_start, i - 1))
            current_sign = s
            current_start = i
    if current_sign != 0 and (n - current_start) >= min_run_len:
        runs.append((current_sign, current_start, n - 1))
    tp_count = 0
    tp_positions: list[float] = []
    for i in range(1, len(runs)):
        if runs[i][0] != runs[i - 1][0]:
            tp_count += 1
            idx_mid = (runs[i - 1][2] + runs[i][1]) // 2
            if idx_mid < len(x_positions):
                tp_positions.append(float(x_positions[idx_mid]))
    return tp_count, tp_positions


def _tp_consensus(tp_counts: list[int], min_support: int = 2) -> int | None:
    labels = [min(tp, 3) for tp in tp_counts]
    counter = Counter(labels)
    top = counter.most_common(2)
    best_label, best_count = top[0]
    if best_count < min_support:
        return None
    if len(top) > 1 and top[1][1] == best_count:
        return None
    return best_label


def count_turning_points(
    x: np.ndarray, y: np.ndarray, p: dict
) -> tuple[int | None, list[int]]:
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]
    n = len(xv)
    if n < 20:
        return None, []
    x_range = xv.max() - xv.min()
    if x_range < 1e-10:
        return None, []
    x_norm = (xv - xv.min()) / x_range

    n_grid = p["sizer_n_grid"]
    n_bw = p["sizer_n_bandwidths"]
    h_min = p["sizer_h_min"]
    h_max = p["sizer_h_max"]
    interior = p["sizer_interior"]
    min_run_frac = p["sizer_min_run_frac"]
    flat_frac = p["sizer_flat_frac"]
    flat_ref = p.get("sizer_flat_ref", "max")

    x_grid = np.linspace(0, 1, n_grid)
    bandwidths = np.geomspace(h_min, h_max, n_bw)
    interior_mask = (x_grid >= interior[0]) & (x_grid <= interior[1])
    n_interior = interior_mask.sum()
    min_run_len = max(int(min_run_frac * n_interior), 3)

    tp_counts: list[int] = []
    for h in bandwidths:
        beta1 = np.full(n_grid, np.nan)
        for gi, xg in enumerate(x_grid):
            w = np.exp(-0.5 * ((x_norm - xg) / h) ** 2)
            w_sum = w.sum()
            if w_sum < 1e-10:
                continue
            dx = x_norm - xg
            s0 = w_sum
            s1 = (w * dx).sum()
            s2 = (w * dx ** 2).sum()
            denom = s0 * s2 - s1 ** 2
            if abs(denom) < 1e-20:
                continue
            beta1[gi] = (s0 * (w * dx * yv).sum() - s1 * (w * yv).sum()) / denom

        interior_deriv = beta1[interior_mask]
        if np.any(np.isfinite(interior_deriv)):
            abs_d = np.abs(interior_deriv)
            ref = np.nanmax(abs_d) if flat_ref == "max" else np.nanmedian(abs_d)
            eps = flat_frac * ref
        else:
            eps = 0.0
        tp, _ = _count_stable_tps(
            interior_deriv, x_grid[interior_mask], min_run_len, eps
        )
        tp_counts.append(tp)

    if not tp_counts:
        return None, []
    consensus = _tp_consensus(tp_counts, p["tp_support_min"])
    return consensus, tp_counts


# ═══════════════════════════════════════════════════════════
# HD test — transition / multimodality
# ═══════════════════════════════════════════════════════════

def check_multimodal(y: np.ndarray) -> tuple[bool, float]:
    if _diptest_func is None:
        warnings.warn("diptest not installed — transition detection disabled")
        return False, 1.0
    valid = np.isfinite(y)
    yv = y[valid]
    if len(yv) < 20:
        return False, 1.0
    dip, pval = _diptest_func(yv)
    return pval < 0.05, float(pval)


# ═══════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════

def _result(
    group,
    has_global,
    simple_or_complex,
    branch,
    tp_number,
    mic,
    **diagnostics,
):
    result = {
        "group": group,
        "has_global_relationship": has_global,
        "simple_or_complex": simple_or_complex,
        "branch_structure": branch,
        "tp_number": tp_number,
        "mic": round(mic, 4) if isinstance(mic, float) else mic,
    }
    result.update(diagnostics)
    return result


def classify_relationship(
    x,
    y,
    *,
    mic: float | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify the scatter-plot relationship between *x* and *y*.

    In addition to the final classification, the returned dictionary contains
    the numerical diagnostics from every decision-tree stage that was actually
    visited. Diagnostics for stages outside the traversed path are ``None``.
    """
    p = {**DEFAULTS, **(params or {})}
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    diagnostics: dict[str, Any] = {
        "mic_gate": None,
        "pearson_r": None,
        "abs_pearson_r": None,
        "power_a": None,
        "power_b": None,
        "power_c": None,
        "power_r2": None,
        "abs_b_minus_1": None,
        "power_decision": None,
        "residual_autocorr": None,
        "complex_path_used": None,
        "branch_fraction": None,
        "sizer_tp_counts": None,
        "dip_pvalue": None,
        "transition_detected": None,
        "path_trace": None,
    }

    if len(x) < 10:
        diagnostics["path_trace"] = "insufficient_data"
        return _result(
            "Uncertain", "N/A", "N/A", "N/A", None, np.nan,
            **diagnostics,
        )

    if mic is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mic = compute_mic(x, y)

    gate1 = check_global_relationship(mic, p)
    diagnostics["mic_gate"] = gate1
    if gate1 == "no_global":
        diagnostics["path_trace"] = f"MIC:{gate1} -> No Global Relationship"
        return _result(
            "No Global Relationship", "No", "N/A", "N/A", None, mic,
            **diagnostics,
        )

    pearson_r, _ = pearsonr(x, y)
    path = "Simple" if abs(pearson_r) >= p["pearson_threshold"] else "Complex"
    diagnostics.update({
        "pearson_r": float(pearson_r),
        "abs_pearson_r": float(abs(pearson_r)),
    })

    if path == "Simple":
        pw = _fit_power_law(x, y)
        label = None
        if pw is None:
            diagnostics["power_decision"] = "fit_failed"
        else:
            a, b, c, r2 = pw
            diagnostics.update({
                "power_a": float(a),
                "power_b": float(b),
                "power_c": float(c),
                "power_r2": float(r2),
                "abs_b_minus_1": float(abs(b - 1)),
            })
            if r2 < p["r2_power_threshold"]:
                diagnostics["power_decision"] = "R2_below_threshold"
            elif abs(b - 1) <= p["linear_b_tolerance"]:
                diagnostics["power_decision"] = "Linear"
                label = "Linear"
            elif abs(b - 1) >= p["curvature_b_threshold"]:
                ac = _residual_autocorr(x, y, a, b, c)
                diagnostics["residual_autocorr"] = float(ac)
                if p["use_autocorr_filter"] and ac > p["autocorr_threshold"]:
                    diagnostics["power_decision"] = "autocorr_high"
                else:
                    label = "Saturation" if b < 1 else "Acceleration"
                    diagnostics["power_decision"] = label
            else:
                if p["r2_autocorr_shortcut"]:
                    ac = _residual_autocorr(x, y, a, b, c)
                    diagnostics["residual_autocorr"] = float(ac)
                    if ac < p["shortcut_autocorr_max"]:
                        label = "Saturation" if b < 1 else "Acceleration"
                        diagnostics["power_decision"] = f"{label}_shortcut"
                    else:
                        diagnostics["power_decision"] = "curvature_gap_autocorr_high"
                else:
                    diagnostics["power_decision"] = "curvature_gap"
        if label is not None:
            diagnostics["complex_path_used"] = False
            diagnostics["path_trace"] = (
                f"MIC:{gate1} -> Pearson:Simple -> Power:{label}"
            )
            return _result(
                label, "Yes", "Simple", "N/A", None, mic,
                **diagnostics,
            )

    diagnostics["complex_path_used"] = True
    complex_trace = f"MIC:{gate1} -> Pearson:{path}"
    if path == "Simple":
        complex_trace += (
            f" -> Power:{diagnostics['power_decision']} -> Complex:fallback"
        )
    else:
        complex_trace += " -> Complex:direct"

    is_branch, branch_fraction = detect_branches(x, y, p)
    diagnostics["branch_fraction"] = float(branch_fraction)
    if is_branch:
        diagnostics["path_trace"] = f"{complex_trace} -> Branch:Yes"
        return _result(
            "Branch", "Yes", path, "Yes", None, mic,
            **diagnostics,
        )

    tp, tp_counts = count_turning_points(x, y, p)
    diagnostics["sizer_tp_counts"] = tp_counts
    if tp is None:
        diagnostics["path_trace"] = (
            f"{complex_trace} -> Branch:No -> SiZer:no_consensus"
        )
        return _result(
            "Uncertain", "Yes", path, "No", None, mic,
            **diagnostics,
        )
    if tp == 1:
        diagnostics["path_trace"] = f"{complex_trace} -> Branch:No -> SiZer:1"
        return _result(
            "U-shape", "Yes", path, "No", 1, mic,
            **diagnostics,
        )
    if tp == 2:
        diagnostics["path_trace"] = f"{complex_trace} -> Branch:No -> SiZer:2"
        return _result(
            "Cubic", "Yes", path, "No", 2, mic,
            **diagnostics,
        )
    if tp >= 3:
        diagnostics["path_trace"] = (
            f"{complex_trace} -> Branch:No -> SiZer:{tp}"
        )
        return _result(
            "Oscillation", "Yes", path, "No", tp, mic,
            **diagnostics,
        )

    is_transition, dip_pvalue = check_multimodal(y)
    diagnostics["dip_pvalue"] = float(dip_pvalue)
    diagnostics["transition_detected"] = bool(is_transition)
    if is_transition:
        diagnostics["path_trace"] = (
            f"{complex_trace} -> Branch:No -> SiZer:0 -> Transition:Yes"
        )
        return _result(
            "Transition", "Yes", path, "No", 0, mic,
            **diagnostics,
        )

    diagnostics["path_trace"] = (
        f"{complex_trace} -> Branch:No -> SiZer:0 -> Transition:No"
    )
    return _result(
        "Uncertain", "Yes", path, "No", 0, mic,
        **diagnostics,
    )


def classify_batch(
    cases: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Classify multiple x–y pairs.

    Each entry in *cases* must have ``'x'`` and ``'y'`` keys.
    Optional: ``'name'`` (label), ``'mic'`` (pre-computed MIC).
    """
    rows = []
    for case in cases:
        result = classify_relationship(
            case["x"], case["y"],
            mic=case.get("mic"),
            params=params,
        )
        if "name" in case:
            result["name"] = case["name"]
        rows.append(result)
    df = pd.DataFrame(rows)
    if "name" in df.columns:
        cols = ["name"] + [c for c in df.columns if c != "name"]
        df = df[cols]
    return df
