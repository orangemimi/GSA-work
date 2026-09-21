"""Method 3: 所有面板 PL R² ≥ 0.85 → Simple

不依赖 TP 检测，直接用 power-law 拟合优度决定 Simple/Complex。
"""
import numpy as np

NAME   = "M3: PL R² ≥ 0.85 (all)"
SHORT  = "M3_plr2_all"

PLR2_THRESH = 0.85


def is_simple(pearson, pl_r2, n_tp_filtered):
    """给定特征，返回是否 Simple。"""
    if not np.isfinite(pl_r2):
        return False
    return pl_r2 >= PLR2_THRESH
