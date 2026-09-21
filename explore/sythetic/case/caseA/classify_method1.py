"""Method 1: |Pearson| ≥ 0.80 → Simple, PL R² ≥ 0 (诊断)

最简单的基线方法: 用 EQ15 Pearson 相关系数判断单调性。
"""
import numpy as np

NAME   = "M1: |Pearson| ≥ 0.80"
SHORT  = "M1_pearson"

PEARSON_THRESH = 0.80
PLR2_THRESH    = 0.0      # 实际上不过滤


def is_simple(pearson, pl_r2, n_tp_filtered):
    """给定特征，返回是否 Simple。"""
    if not np.isfinite(pearson):
        return False
    return abs(pearson) >= PEARSON_THRESH
