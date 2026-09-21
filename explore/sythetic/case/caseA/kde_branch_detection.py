"""KDE Branch Detection — 纯分支检测分类器

仅做 Branch 检测（散点滑动窗口 KDE），不包含 MIC/dCor 预筛选，
也不包含 Simple/Complex 幂律分类路径。

检测流程:
1. 三个带宽 (0.65, 1.00, 1.50) 各自独立检测
   a. 沿 x 轴滑动窗口，每个窗口对 y 做 KDE
   b. 局部双峰判定: valley depth ≥ 0.30, min branch mass ≥ 0.10
   c. 相邻窗口连续追踪 (peak-valley-peak triplet)
   d. Fork score → 该带宽投票 Branch / Candidate / No branch
2. 三带宽投票共识
   ≥ 2/3 Branch → "Branch"
   支持票 + median mode separation ≥ 0.5 → "Candidate"
   其余 → "No branch"

输出 group:
    "Branch"           — 检测到稳定分支 (≥ 2/3 带宽投票)
    "Candidate branch" — 疑似分支
    "No branch"        — 未检测到分支
    "Uncertain"        — 数据不足

使用方式:
    # 一次性检测
    from kde_branch_detection import detect_branch
    result = detect_branch(x, y)

    # 两阶段 (缓存 KDE 拟合，快速重分类)
    from kde_branch_detection import fit_kde, classify_from_fitted
    fitted = fit_kde(x, y)          # 昂贵，可缓存
    result = classify_from_fitted(fitted)  # 廉价，改阈值后重跑
"""

from __future__ import annotations

from typing import Any

import numpy as np

from test_two_condition_kde import (
    BANDWIDTH_FACTORS,
    classify_fitted_panel as _classify_fitted,
    classify_panel as _classify_panel,
    fit_panel_kde as _fit_panel_kde,
    prepare_panel_xy,
)


DEFAULTS: dict[str, Any] = {
    "branch_coverage_mode": "x_range",
    "branch_min_window": 40,
    "min_n": 30,
}

STATUS_TO_GROUP = {
    "Branch": "Branch",
    "Candidate": "Candidate branch",
}


def _to_group(status: str) -> str:
    return STATUS_TO_GROUP.get(status, "No branch")


# ============================================================
#  一次性接口
# ============================================================

def detect_branch(
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """对 x-y 散点进行 Branch 检测（一次性完成）。"""
    p = {**DEFAULTS, **(params or {})}

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < p["min_n"]:
        return {"group": "Uncertain", "status": "insufficient_data"}

    result = _classify_panel(
        x, y,
        coverage_mode=p["branch_coverage_mode"],
        min_window=p["branch_min_window"],
    )
    return {"group": _to_group(result["status"]), **result}


# ============================================================
#  两阶段接口 (缓存 KDE → 快速重分类)
# ============================================================

def fit_kde(
    x: np.ndarray,
    y: np.ndarray,
    min_window: int = 40,
) -> dict:
    """Stage 1: 局部多带宽 KDE 拟合（昂贵，结果可缓存）。"""
    return _fit_panel_kde(x, y, min_window=min_window)


def classify_from_fitted(
    fitted: dict,
    coverage_mode: str = "x_range",
) -> dict[str, Any]:
    """Stage 2: 从已有 KDE 拟合结果做 Branch 分类（廉价）。"""
    result = _classify_fitted(fitted, coverage_mode=coverage_mode)
    return {"group": _to_group(result["status"]), **result}


# ============================================================
#  批量接口
# ============================================================

def classify_batch(
    cases: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """批量检测多组 x-y 散点的 Branch 结构。

    每个 case 须含 'x', 'y'；可选 'name' 作为标识。
    """
    rows = []
    for case in cases:
        result = detect_branch(case["x"], case["y"], params=params)
        if "name" in case:
            result["name"] = case["name"]
        rows.append(result)
    return rows
