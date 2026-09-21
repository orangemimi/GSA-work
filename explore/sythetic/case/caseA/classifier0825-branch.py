"""General-purpose scatter-plot relationship classifier.

Decision tree
─────────────
Gate 1  MIC/dCor 关联性预筛选
        score ≥ strong       → strong global
        score ≥ intermediate → intermediate global
        score < intermediate → "No Global Relationship"

Gate 2  Branch 检测（散点滑动窗口 KDE 或 2D KDE 条件密度，二选一）
        → Branch / Candidate / Two-band → 直接输出

Gate 3  对 No branch 的面板
        |Pearson| ≥ simple_min (0.5) → 尝试 Simple（含 0.5–0.7 中间带）
        |Pearson| < simple_min       → Complex path

Simple  power-law fit  f(x) = a·x^b + c
          R² ≥ threshold  &  |b−1| ≤ 0.2   → Linear
          R² ≥ threshold  &  0.2 < |b−1| < 0.5 → Near-linear
          R² ≥ threshold  &  |b−1| ≥ 0.5   → Saturation / Acceleration
          otherwise → No simple（Simple 试过但幂律没套上）
          linearity_score = 1 - min(|b−1| / 0.5, 1) is always stored when b exists

Complex  |Pearson| < 0.5，未走 Simple。暂不用 SiZer 分拐点。
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
    import dcor as _dcor
except ImportError:
    _dcor = None

try:
    from diptest import diptest as _diptest_func
except ImportError:
    _diptest_func = None

import test_two_condition_kde as _scatter_kde
import test_2d_kde_conditional_tracks as _kde2d


# ═══════════════════════════════════════════════════════════
# 参数配置（修改这里即可调整所有阈值）
# ═══════════════════════════════════════════════════════════

DEFAULTS: dict[str, Any] = {

    # ── Gate 1：关联性预筛选 ──────────────────────────────
    "gate1_metric": "mic",          # 预筛选指标：'mic' 或 'dcor'
    "mic_strong": 0.8,              # MIC 强关联阈值
    "mic_intermediate": 0.2,        # MIC 中等关联阈值，低于此值判定为无全局关系
    "dcor_strong": 0.8,             # dCor 强关联阈值
    "dcor_intermediate": 0.4,       # dCor 中等关联阈值，低于此值判定为无全局关系

    # ── Gate 2：Branch 检测 ──────────────────────────────
    "branch_method": "scatter",     # 分支检测方法：'scatter'（散点滑动窗口 KDE）或 'kde2d'（2D KDE 条件密度）
    "branch_coverage_mode": "x_range",  # 分支覆盖率计算方式：'x_range'（x 轴跨度比例）或 'point_count'（区间内点数比例）
    "branch_min_window": 40,            # KDE 窗口最小点数（n_window = max(0.06·n, branch_min_window)）
    "detect_behind_gate": False,            # MIC 没过的面板是否仍跑分支检测（True 会显著增加运行时间）

    # ── Gate 3：Simple / Complex 分流 ────────────────────
    "pearson_simple_min": 0.5,      # |Pearson| ≥ 此值尝试 Simple（含 0.5–0.7 中间带）
    "pearson_threshold": 0.7,       # |Pearson| ≥ 此值为强线性相关（诊断用）

    # ── Simple 路径：幂律拟合 ────────────────────────────
    "r2_power_threshold": 0.5,      # 幂律拟合 R² 最低要求
    "linear_b_tolerance": 0.2,      # |b-1| ≤ 此值判定为 Linear
    "curvature_b_threshold": 0.5,   # |b-1| ≥ 此值判定为 Saturation/Acceleration
    # 中间带 0.2 < |b-1| < 0.5 判定为 Near-linear
    # linearity_score = 1 - min(|b-1| / curvature_b_threshold, 1)
    "autocorr_threshold": 0.7,      # 残差自相关阈值，超过则拒绝幂律分类
    "use_autocorr_filter": False,   # 是否启用残差自相关过滤
    "r2_autocorr_shortcut": False,   # b 在容差和曲率之间时，用自相关快捷判定
    "shortcut_autocorr_max": 0.7,   # 快捷判定的自相关上限

    # ── Complex 路径：SiZer 转折点分析 ───────────────────
    "sizer_n_bandwidths": 4,        # SiZer 使用的带宽数量
    "sizer_n_grid": 200,            # SiZer 评估网格点数
    "sizer_h_min": 0.05,            # SiZer 最小带宽（归一化后）
    "sizer_h_max": 0.3,             # SiZer 最大带宽（归一化后）
    "sizer_interior": (0.05, 0.95), # SiZer 只在中间区域计数转折点，忽略边缘
    "sizer_min_run_frac": 0.05,     # 符号连续段最短比例（太短的忽略）
    "sizer_flat_frac": 0.1,         # 导数绝对值低于此比例×参考值视为平坦
    "sizer_flat_ref": "max",        # 平坦参考值：'max' 或 'median'
    "tp_support_min": 2,            # 转折点数需要至少这么多带宽支持才算共识

    "force_complex": False,         # 设为 True 则跳过 Simple 路径，全部走 Complex
}


# ═══════════════════════════════════════════════════════════
# Gate 1 metrics — MIC / dCor
# ═══════════════════════════════════════════════════════════

def compute_mic(x: np.ndarray, y: np.ndarray) -> float:
    mine = _MINE()
    mine.compute_score(x.astype(float), y.astype(float))
    return float(mine.mic())


def compute_dcor(x: np.ndarray, y: np.ndarray) -> float:
    if _dcor is None:
        raise ImportError("dcor package not installed — pip install dcor")
    return float(_dcor.distance_correlation(
        x.astype(float), y.astype(float),
    ))


def compute_gate1_score(
    x: np.ndarray, y: np.ndarray, metric: str,
) -> float:
    if metric == "dcor":
        return compute_dcor(x, y)
    return compute_mic(x, y)


# ═══════════════════════════════════════════════════════════
# Gate 1 — 关联性预筛选
# ═══════════════════════════════════════════════════════════

def check_global_relationship(score: float, p: dict) -> str:
    if score is None or not np.isfinite(score):
        return "no_global"
    metric = p["gate1_metric"]
    strong = p[f"{metric}_strong"]
    intermediate = p[f"{metric}_intermediate"]
    if score >= strong:
        return "strong_global"
    if score >= intermediate:
        return "intermediate_global"
    return "no_global"


# ═══════════════════════════════════════════════════════════
# Gate 2 — Branch 检测（散点 KDE / 2D KDE）
# ═══════════════════════════════════════════════════════════

def detect_branch(
    x: np.ndarray, y: np.ndarray, p: dict,
) -> dict[str, Any]:
    """用散点滑动窗口 KDE 或 2D KDE 条件密度检测分支。

    返回检测器的完整 result dict（含 status, branch_type 等）。
    """
    method = p["branch_method"]
    cov_mode = p.get("branch_coverage_mode", "x_range")
    min_win = p.get("branch_min_window", 40)
    if method == "scatter":
        return _scatter_kde.classify_panel(x, y, coverage_mode=cov_mode, min_window=min_win)
    elif method == "kde2d":
        return _kde2d.classify_panel(x, y, coverage_mode=cov_mode)
    else:
        raise ValueError(
            f"branch_method 必须是 'scatter' 或 'kde2d'，当前值: {method!r}"
        )


# ═══════════════════════════════════════════════════════════
# Gate 3 — Simple vs Complex
# ═══════════════════════════════════════════════════════════

def check_simple_or_complex(
    x: np.ndarray, y: np.ndarray, p: dict
) -> tuple[str, float]:
    r, _ = pearsonr(x, y)
    simple_min = p.get("pearson_simple_min", p["pearson_threshold"])
    return ("Simple" if abs(r) >= simple_min else "Complex"), float(r)


# ═══════════════════════════════════════════════════════════
# Simple 路径 — 幂律拟合分类
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


def linearity_score(b: float, scale: float = 0.5) -> float:
    """1 = linear, 0 = |b-1| at or beyond *scale* (default 0.5)."""
    if b is None or not np.isfinite(b) or scale <= 0:
        return np.nan
    return float(1.0 - min(abs(float(b) - 1.0) / scale, 1.0))


def shape_from_power_b(b: float, p: dict) -> str:
    """Map power-law exponent to Linear / Near-linear / Saturation / Acceleration."""
    abs_b_1 = abs(float(b) - 1.0)
    if abs_b_1 <= p["linear_b_tolerance"]:
        return "Linear"
    if abs_b_1 < p["curvature_b_threshold"]:
        return "Near-linear"
    return "Saturation" if b < 1.0 else "Acceleration"


def classify_simple(x: np.ndarray, y: np.ndarray, p: dict) -> str | None:
    pw = _fit_power_law(x, y)
    if pw is None:
        return None
    a, b, c, r2 = pw
    if r2 < p["r2_power_threshold"]:
        return None
    label = shape_from_power_b(b, p)
    if label in {"Saturation", "Acceleration"} and p["use_autocorr_filter"]:
        ac = _residual_autocorr(x, y, a, b, c)
        if ac > p["autocorr_threshold"]:
            return None
    return label


# ═══════════════════════════════════════════════════════════
# SiZer 转折点分析
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
# HD test — Transition / 多模态检测
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
    gate1_score,
    gate1_metric,
    **diagnostics,
):
    result = {
        "group": group,
        "has_global_relationship": has_global,
        "simple_or_complex": simple_or_complex,
        "branch_structure": branch,
        "tp_number": tp_number,
        "gate1_metric": gate1_metric,
        "gate1_score": round(gate1_score, 4) if isinstance(gate1_score, float) else gate1_score,
    }
    result.update(diagnostics)
    return result


def classify_relationship(
    x,
    y,
    *,
    mic: float | None = None,
    gate1_score: float | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """对 x-y 散点关系进行完整分类。

    流程：Gate 1 (MIC/dCor) → Gate 2 (Branch) → Gate 3 (Simple/Complex)

    Parameters
    ----------
    mic : float, optional
        预计算的 MIC 值（向后兼容）。
    gate1_score : float, optional
        预计算的 Gate 1 分数，优先于 *mic*。
    params : dict, optional
        覆盖 DEFAULTS 中的任意参数。
    """
    p = {**DEFAULTS, **(params or {})}
    metric = p["gate1_metric"]

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    diagnostics: dict[str, Any] = {
        "gate1_gate": None,
        "mic": None,
        "dcor": None,
        # Branch 检测
        "branch_method": p["branch_method"],
        "branch_status": None,
        "branch_type": None,
        "branch_run_length": None,
        "branch_coverage": None,
        "branch_relative_change": None,
        "branch_bandwidth_vote_pattern": None,
        "branch_bandwidth_branch_votes": None,
        "branch_bandwidth_support_votes": None,
        "branch_bandwidth_support_median_relative_mode_separation": None,
        "branch_candidate_relative_mode_separation_threshold": None,
        "branch_high_confidence": None,
        # Simple/Complex
        "pearson_r": None,
        "abs_pearson_r": None,
        "pearson_band": None,
        "power_a": None,
        "power_b": None,
        "power_c": None,
        "power_r2": None,
        "abs_b_minus_1": None,
        "linearity_score": None,
        "power_decision": None,
        "residual_autocorr": None,
        "complex_path_used": None,
        # SiZer
        "sizer_tp_counts": None,
        "dip_pvalue": None,
        "transition_detected": None,
        "path_trace": None,
    }

    if len(x) < 10:
        diagnostics["path_trace"] = "insufficient_data"
        return _result(
            "Uncertain", "N/A", "N/A", "N/A", None, np.nan, metric,
            **diagnostics,
        )

    # ── 计算 MIC 和 dCor ──
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mic_val = float(mic) if mic is not None else compute_mic(x, y)
        try:
            dcor_val = compute_dcor(x, y)
        except Exception:
            dcor_val = np.nan
    diagnostics["mic"] = round(mic_val, 4) if np.isfinite(mic_val) else np.nan
    diagnostics["dcor"] = round(dcor_val, 4) if np.isfinite(dcor_val) else np.nan

    if gate1_score is not None:
        score = float(gate1_score)
    elif metric == "dcor":
        score = diagnostics["dcor"]
    else:
        score = diagnostics["mic"]

    # ── Gate 1：关联性预筛选 ──
    gate1 = check_global_relationship(score, p)
    diagnostics["gate1_gate"] = gate1
    if gate1 == "no_global":
        diagnostics["path_trace"] = (
            f"{metric.upper()}:{gate1} -> No Global Relationship"
        )
        return _result(
            "No Global Relationship", "No", "N/A", "N/A", None,
            score, metric,
            **diagnostics,
        )

    # ── Gate 2：Branch 检测 ──
    branch_result = detect_branch(x, y, p)
    branch_status = branch_result["status"]
    diagnostics["branch_status"] = branch_status
    diagnostics["branch_type"] = branch_result.get("branch_type")
    diagnostics["branch_run_length"] = branch_result.get("run_length")
    diagnostics["branch_coverage"] = branch_result.get("x_coverage")
    diagnostics["branch_relative_change"] = branch_result.get("relative_separation_change")
    diagnostics["branch_bandwidth_vote_pattern"] = branch_result.get("bandwidth_vote_pattern")
    diagnostics["branch_bandwidth_branch_votes"] = branch_result.get("bandwidth_branch_votes")
    diagnostics["branch_bandwidth_support_votes"] = branch_result.get("bandwidth_support_votes")
    diagnostics["branch_bandwidth_support_median_relative_mode_separation"] = branch_result.get(
        "bandwidth_support_median_relative_mode_separation"
    )
    diagnostics["branch_candidate_relative_mode_separation_threshold"] = branch_result.get(
        "candidate_relative_mode_separation_threshold"
    )
    diagnostics["branch_high_confidence"] = branch_result.get("high_confidence_branch")

    if branch_status == "Branch":
        diagnostics["path_trace"] = (
            f"{metric.upper()}:{gate1} -> Branch:{branch_status}"
        )
        return _result(
            "Branch", "Yes", "N/A", "Yes", None, score, metric,
            **diagnostics,
        )
    if branch_status == "Candidate":
        diagnostics["path_trace"] = (
            f"{metric.upper()}:{gate1} -> Branch:{branch_status}"
        )
        return _result(
            "Candidate branch", "Yes", "N/A", "Candidate", None, score, metric,
            **diagnostics,
        )
    if branch_status == "Two-band":
        diagnostics["path_trace"] = (
            f"{metric.upper()}:{gate1} -> Branch:{branch_status}"
        )
        return _result(
            "Two-band", "Yes", "N/A", "Two-band", None, score, metric,
            **diagnostics,
        )

    # ── Gate 3：Simple / Complex 分流（仅对 No branch） ──
    # |r| ≥ 0.5 都先走 Simple；0.5–0.7 是中间带，拟合不上再退回 Complex。
    pearson_r, _ = pearsonr(x, y)
    abs_r = abs(pearson_r)
    simple_min = p.get("pearson_simple_min", p["pearson_threshold"])
    if p.get("force_complex"):
        path = "Complex"
        pearson_band = "forced"
    elif abs_r >= p["pearson_threshold"]:
        path = "Simple"
        pearson_band = "strong"
    elif abs_r >= simple_min:
        path = "Simple"
        pearson_band = "mid"
    else:
        path = "Complex"
        pearson_band = "low"
    diagnostics.update({
        "pearson_r": float(pearson_r),
        "abs_pearson_r": float(abs_r),
        "pearson_band": pearson_band,
    })

    pearson_tag = (
        "Simple(mid)" if pearson_band == "mid"
        else ("Simple" if path == "Simple" else "Complex")
    )
    base_trace = f"{metric.upper()}:{gate1} -> Branch:No -> Pearson:{pearson_tag}"

    # ── Simple 路径：幂律拟合 ──
    if path == "Simple":
        pw = _fit_power_law(x, y)
        label = None
        if pw is None:
            diagnostics["power_decision"] = "fit_failed"
        else:
            a, b, c, r2 = pw
            abs_b_1 = float(abs(b - 1))
            diagnostics.update({
                "power_a": float(a),
                "power_b": float(b),
                "power_c": float(c),
                "power_r2": float(r2),
                "abs_b_minus_1": abs_b_1,
                "linearity_score": linearity_score(
                    b, p["curvature_b_threshold"]
                ),
            })
            if r2 < p["r2_power_threshold"]:
                diagnostics["power_decision"] = "R2_below_threshold"
            else:
                candidate = shape_from_power_b(b, p)
                if candidate in {"Saturation", "Acceleration"}:
                    ac = _residual_autocorr(x, y, a, b, c)
                    diagnostics["residual_autocorr"] = float(ac)
                    if p["use_autocorr_filter"] and ac > p["autocorr_threshold"]:
                        diagnostics["power_decision"] = "autocorr_high"
                    else:
                        label = candidate
                        diagnostics["power_decision"] = label
                else:
                    label = candidate
                    diagnostics["power_decision"] = label
        if label is not None:
            diagnostics["complex_path_used"] = False
            diagnostics["path_trace"] = f"{base_trace} -> Power:{label}"
            return _result(
                label, "Yes", "Simple", "No", None, score, metric,
                **diagnostics,
            )

    # ── Complex / No simple（暂不用 SiZer 分拐点）──
    diagnostics["complex_path_used"] = True
    diagnostics["sizer_tp_counts"] = None
    if path == "Simple":
        diagnostics["path_trace"] = (
            f"{base_trace} -> Power:{diagnostics['power_decision']} -> No simple"
        )
        return _result(
            "No simple", "Yes", "Simple", "No", None, score, metric,
            **diagnostics,
        )
    diagnostics["path_trace"] = f"{base_trace} -> Complex"
    return _result(
        "Complex", "Yes", "Complex", "No", None, score, metric,
        **diagnostics,
    )


def classify_batch(
    cases: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """批量分类多组 x-y 散点关系。

    每个 case 必须包含 'x' 和 'y'。
    可选：'name'（标签），'gate1_score' 或 'mic'（预计算值）。
    """
    rows = []
    for case in cases:
        result = classify_relationship(
            case["x"], case["y"],
            mic=case.get("mic"),
            gate1_score=case.get("gate1_score"),
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
