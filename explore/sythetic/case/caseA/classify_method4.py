import numpy as np

NAME   = "M4: |Pearson| ≥ 0.80 + PL R² ≥ 0.85"
SHORT  = "M4_pearson_plr2"

PEARSON_THRESH = 0.80
PLR2_THRESH    = 0.85


def is_simple(pearson, pl_r2, n_tp_filtered):
    if not np.isfinite(pearson) or not np.isfinite(pl_r2):
        return False
    return abs(pearson) >= PEARSON_THRESH and pl_r2 >= PLR2_THRESH
