"""LOWESS 响应曲线形状分类器

对每个 panel (变量对 × 模型 × 气候区) 的 LOWESS 曲线进行形状分类。
分类结果包含：响应强度 (R²)、一级形状 (|Pearson| 阈值)、
子形状 (Bow / 转折点 TP)，并保留各段方向比例作为诊断。

所有可调参数集中在文件开头的 "可调参数" 区域，方便统一修改。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess

# ============================================================
#  可调参数 —— 修改分类行为只需改这里
# ============================================================

# ---- LOWESS 拟合参数 ----
LOWESS_FRAC = 0.2       # LOWESS 带宽 (占数据比例)
LOWESS_IT   = 3         # LOWESS 鲁棒迭代次数
MIN_N       = 30        # 最少样本量，低于此跳过

# ---- EQ15 采样参数 ----
N_EQ_PTS    = 15        # 等距采样点数
DEADZONE    = 0.02      # 归一化相邻变化 d 的死区 (|d| < ε 视为零)

# ---- Weak 内部的 Flat 确认（先于 Monotonic / Non-monotonic）----
FLAT_FRACTION_THRESH  = 0.70  # 至少 70% 的相邻 LOWESS 变化落入死区
FLAT_AMPLITUDE_THRESH = 0.15  # LOWESS 主体振幅小于观测 y 主体范围的 15%

# ---- R² 响应强度阈值 ----
R2_WEAK     = 0.35      # R² < 此值 → Weak
R2_DETECT   = 0.60      # R² ≥ 此值 → Detectable；中间 → Uncertain

# ---- 单调性判定 ----
PEARSON_MONO_THRESH = 0.60   # |Pearson| ≥ 此值 → Monotonic；< 此值 → Non-monotonic

# ---- Bow 法 (单调子分类) ----
BOW_THRESH  = 0.15      # B < 此值 → linear；B ≥ 此值 → acceleration/saturation

# ---- 对称性 (非单调 tp=1 子分类) ----
SYM_THRESH  = 0.40      # sym ≥ 此值 → U_shaped；sym < 此值 → J/inverted_J


# ============================================================
#  LOWESS 拟合
# ============================================================

def fit_lowess(x, y, frac=LOWESS_FRAC, it=LOWESS_IT):
    """拟合 LOWESS 曲线。

    返回 (xs, ys) —— 按 x 排序的拟合曲线点；
    若样本量不足或拟合失败则返回 None。
    """
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


# ============================================================
#  R² 计算
# ============================================================

def compute_r2(x, y, xs, ys):
    """计算观测值 y 相对于 LOWESS 预测 ŷ 的 R²。"""
    yhat = np.interp(x, xs, ys)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def r2_strength(r2):
    """R² → 响应强度标签。"""
    if r2 is None or not np.isfinite(r2):
        return "Insufficient"
    if r2 < R2_WEAK:
        return "Weak"
    if r2 < R2_DETECT:
        return "Uncertain"
    return "Detectable"


# ============================================================
#  转折点计数
# ============================================================

def _count_turning_points(d, deadzone):
    """统计 d 序列中的符号变化次数 (跳过死区段)。"""
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


# ============================================================
#  Bow 法 —— 单调曲线的加速/饱和/线性子分类
# ============================================================

def _bow_classify(y_arr, bow_thresh=BOW_THRESH):
    """Bow 法：用曲线偏离首尾连线的程度判断曲率方向。

    返回 (B, 子分类)：
      B < bow_thresh → 'linear'
      B ≥ bow_thresh 且 mean(e*) < 0 → 'acceleration'
      B ≥ bow_thresh 且 mean(e*) ≥ 0 → 'saturation'
    """
    chord = y_arr[0] + (y_arr[-1] - y_arr[0]) * np.linspace(0, 1, len(y_arr))
    e = y_arr - chord
    drop = abs(y_arr[-1] - y_arr[0])
    # B = 中间点偏差均值 / 首尾落差
    b = float(np.mean(np.abs(e[1:-1])) / drop) if drop > 0 else 0.0
    # e* = 方向归一化偏差 (统一上升/下降方向)
    sign_d = 1.0 if y_arr[-1] > y_arr[0] else -1.0
    e_star = sign_d * e[1:-1]
    if b < bow_thresh:
        cls = "linear"
    elif float(np.mean(e_star)) < 0:
        cls = "acceleration"
    else:
        cls = "saturation"
    return b, cls


# ============================================================
#  单个 panel 的形状分类
# ============================================================

def classify_panel(x, y, xs, ys,
                   n_pts=N_EQ_PTS,
                   deadzone=DEADZONE,
                   flat_fraction_thresh=None,
                   flat_amplitude_thresh=None,
                   pearson_thresh=PEARSON_MONO_THRESH,
                   bow_thresh=BOW_THRESH,
                   sym_thresh=SYM_THRESH):
    """对一个 panel 的 LOWESS 曲线进行形状分类。

    参数
    ----
    x, y : 筛选后的原始散点数据 (用于计算 y_scale 和 R²)
    xs, ys : LOWESS 拟合曲线 (已按 x 排序)
    n_pts : EQ 等距采样点数 (默认 15)
    deadzone : 归一化斜率死区
    flat_fraction_thresh : Flat 所需的近零局部变化比例
    flat_amplitude_thresh : Flat 所允许的最大归一化 LOWESS 主体振幅
    pearson_thresh : |Pearson| ≥ 此值 → Monotonic
    bow_thresh : Bow 法线性/非线性阈值
    sym_thresh : U 型对称性阈值

    返回
    ----
    dict，包含所有分类结果字段
    """
    n_segs = n_pts - 1
    out = {
        "r2": np.nan,
        "strength": "Insufficient",
        "lowess_shape": "skipped",
        "lowess_shape2": "",
        "ls_pearson": np.nan,
        "ls_abs_pearson": np.nan,
        "ls_bow": np.nan,
        "ls_bow_p": np.nan,
        "lowess_shape2_p": "",
        "ls_n_tp": 0,
        "ls_p0": np.nan,
        "ls_p_pos": np.nan,
        "ls_p_neg": np.nan,
        "ls_n0": 0,
        "ls_n_pos": 0,
        "ls_n_neg": 0,
        "ls_V_pos": np.nan,
        "ls_V_neg": np.nan,
        "ls_R_rev": np.nan,
        "ls_flat_fraction": np.nan,
        "ls_flat_amplitude": np.nan,
    }

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if flat_fraction_thresh is None:
        flat_fraction_thresh = FLAT_FRACTION_THRESH
    if flat_amplitude_thresh is None:
        flat_amplitude_thresh = FLAT_AMPLITUDE_THRESH
    finite = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[finite], y[finite]

    if len(xf) < n_pts * 3:
        out["lowess_shape"] = "too_few"
        return out

    # ---- Step 1: R² ----
    out["r2"] = compute_r2(xf, yf, xs, ys)
    out["strength"] = r2_strength(out["r2"])

    # ---- Step 2: y_scale (P95 - P05) ----
    y_scale = float(np.percentile(yf, 95) - np.percentile(yf, 5))
    if y_scale <= 0:
        y_scale = float(np.ptp(yf))
    if y_scale <= 0:
        out["lowess_shape"] = "no_variation"
        return out

    # ---- Step 3: EQ 等距采样 ----
    x_eq = np.linspace(xf.min(), xf.max(), n_pts)
    if np.ptp(x_eq) <= 0:
        out["lowess_shape"] = "no_variation"
        return out
    y_eq = np.interp(x_eq, xs, ys)

    # ---- Step 4: 归一化斜率 d ----
    d = np.diff(y_eq) / y_scale
    n0 = int(np.sum(np.abs(d) < deadzone))
    n_pos = int(np.sum(d >= deadzone))
    n_neg = int(np.sum(d <= -deadzone))
    out.update({
        "ls_n0": n0,
        "ls_n_pos": n_pos,
        "ls_n_neg": n_neg,
        "ls_p0": n0 / n_segs,
        "ls_p_pos": n_pos / n_segs,
        "ls_p_neg": n_neg / n_segs,
    })

    # R_rev 只累加超过 deadzone 的实质变化；微小 LOWESS 抖动不算反向。
    d_sig = d[np.abs(d) >= deadzone]
    V_pos = float(np.sum(d_sig[d_sig > 0]))
    V_neg = float(np.sum(-d_sig[d_sig < 0]))
    total_V = V_pos + V_neg
    out["ls_V_pos"] = V_pos
    out["ls_V_neg"] = V_neg
    out["ls_R_rev"] = (
        float(min(V_pos, V_neg) / total_V) if total_V > 1e-12 else 0.0
    )

    # ---- Step 5: Weak 内部的 Flat gate ----
    # 不能只看首尾总体斜率：U / 倒 U 的首尾可能相近。这里同时要求
    # LOWESS 的主体振幅小，并且大多数相邻变化都落在 deadzone 内。
    # Uncertain / Detectable 不进入 Flat 类，直接继续做形状分类。
    flat_fraction = float(np.mean(np.abs(d) < deadzone))
    flat_amplitude = float(
        (np.percentile(y_eq, 95) - np.percentile(y_eq, 5)) / y_scale
    )
    out["ls_flat_fraction"] = flat_fraction
    out["ls_flat_amplitude"] = flat_amplitude
    if (out["strength"] == "Weak"
            and flat_fraction >= flat_fraction_thresh
            and flat_amplitude < flat_amplitude_thresh):
        out["lowess_shape"] = "flat"
        out["lowess_shape2"] = "flat"
        return out

    # ---- Step 6: EQ Pearson ----
    if np.ptp(y_eq) <= 0:
        pr = 0.0
    else:
        pr = float(np.corrcoef(x_eq, y_eq)[0, 1])
    out["ls_pearson"] = pr
    out["ls_abs_pearson"] = abs(pr)

    # ---- Step 7: TP (Non-monotonic 子形状诊断用) ----
    n_tp = _count_turning_points(d, deadzone)
    out["ls_n_tp"] = n_tp

    # ==== 一级形状: |Pearson| 判定 Monotonic / Non-monotonic ====
    if abs(pr) >= pearson_thresh:
        # ---- Monotonic ----
        endpoint_change = float(y_eq[-1] - y_eq[0])
        direction_signal = endpoint_change if abs(endpoint_change) > 1e-12 else pr
        direction = "up" if direction_signal >= 0 else "down"
        out["lowess_shape"] = f"monotonic_{direction}"

        # Bow 法子分类 (全 EQ 点)
        B, shape2 = _bow_classify(y_eq, bow_thresh)
        out["ls_bow"] = B
        out["lowess_shape2"] = shape2

        # B' (去掉首尾各 ~5% 的 EQ 点)
        i05 = max(1, int(round(0.05 * (n_pts - 1))))
        i95 = min(n_pts - 2, int(round(0.95 * (n_pts - 1))))
        y_trim = y_eq[i05:i95 + 1]
        Bp, shape2p = _bow_classify(y_trim, bow_thresh)
        out["ls_bow_p"] = Bp
        out["lowess_shape2_p"] = shape2p

        return out

    # ---- Non-monotonic ----
    out["lowess_shape"] = "nonmonotonic"

    if n_tp == 1:
        # 判断第一段的方向
        first_sign = next((1 if dv >= deadzone else -1)
                          for dv in d if abs(dv) >= deadzone)
        if first_sign == -1:
            # 先降后升 → 找谷底
            i_min = int(np.argmin(y_eq))
            left_arm = y_eq[0] - y_eq[i_min]
            right_arm = y_eq[-1] - y_eq[i_min]
        else:
            # 先升后降 → 找峰顶
            i_max = int(np.argmax(y_eq))
            left_arm = y_eq[i_max] - y_eq[0]
            right_arm = y_eq[i_max] - y_eq[-1]

        max_arm = max(left_arm, right_arm)
        sym = min(left_arm, right_arm) / max_arm if max_arm > 0 else 0.0

        if first_sign == 1:
            # 先升后降 → 一律 inverted_U
            out["lowess_shape2"] = "inverted_U"
        else:
            # 先降后升 → 按对称性区分
            if sym >= sym_thresh:
                out["lowess_shape2"] = "U_shaped"
            else:
                out["lowess_shape2"] = "J_shaped" if y_eq[-1] > y_eq[0] else "inverted_J"
    elif n_tp == 0:
        # TP=0: 死区吞掉了弱变化，用 argmax/argmin 位置判断候选形状
        i_max = int(np.argmax(y_eq))
        i_min = int(np.argmin(y_eq))
        if 0 < i_max < n_pts - 1:
            out["lowess_shape2"] = "candidate_inverted_U"
        elif 0 < i_min < n_pts - 1:
            out["lowess_shape2"] = "candidate_U_shaped"
        else:
            out["lowess_shape2"] = "complex"
    elif n_tp == 2:
        out["lowess_shape2"] = "S_or_N"
    else:
        out["lowess_shape2"] = "complex"

    return out


# ============================================================
#  LOWESS 曲线的加载与保存
# ============================================================

def load_lowess_curves(path):
    """从 parquet 文件加载已拟合的 LOWESS 曲线。

    返回 {(Pair, Model, Zone): (xs, ys)}。
    """
    path = Path(path)
    if not path.exists():
        return {}
    t = pd.read_parquet(path)
    out = {}
    for key, g in t.groupby(["Pair", "Model", "Zone"], sort=False):
        out[key] = (g["x"].to_numpy(dtype=float), g["y_lowess"].to_numpy(dtype=float))
    return out


# ============================================================
#  批量分类
# ============================================================

def classify_batch(jobs, n_jobs=-1):
    """并行对所有 panel 进行 LOWESS 拟合 + 形状分类。

    参数
    ----
    jobs : list of dict，每个 dict 包含:
        x, y       : 筛选后的散点数据
        x_var, y_var, pair_label, model, zone : 标识信息
        fit        : (xs, ys) 或 None (None 时自动拟合)
    n_jobs : joblib 并行进程数，-1 为全部 CPU

    返回
    ----
    records : list of dict (每行一个 panel 的分类结果)
    curves  : list of DataFrame 或 None (LOWESS 曲线，用于保存)
    """
    from joblib import Parallel, delayed

    def _process_one(job):
        x = np.asarray(job["x"], float)
        y = np.asarray(job["y"], float)
        rec = {
            "Pair": job["pair_label"],
            "x_var": job["x_var"],
            "y_var": job["y_var"],
            "Model": job["model"],
            "Zone": job["zone"],
            "N": len(x),
            "eps": job.get("eps", np.nan),
        }

        fit = job.get("fit")
        if fit is None:
            fit = fit_lowess(x, y)
        if fit is None:
            rec.update({
                "r2": np.nan, "strength": "Insufficient",
                "lowess_shape": "fit_failed", "lowess_shape2": "",
                "ls_pearson": np.nan, "ls_abs_pearson": np.nan,
                "ls_bow": np.nan, "ls_bow_p": np.nan,
                "lowess_shape2_p": "", "ls_n_tp": 0,
                "ls_flat_fraction": np.nan,
                "ls_flat_amplitude": np.nan,
            })
            return rec, None

        xs, ys = fit
        rec.update(classify_panel(x, y, xs, ys))

        curve_df = pd.DataFrame({
            "Pair": job["pair_label"],
            "Model": job["model"],
            "Zone": job["zone"],
            "eps": job.get("eps", np.nan),
            "x": xs,
            "y_lowess": ys,
        })
        return rec, curve_df

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_one)(job) for job in jobs
    )

    records = [r[0] for r in results]
    curves = [r[1] for r in results if r[1] is not None]
    return records, curves


def reclassify_batch(df_existing, curves_dict, raw_data_fn):
    """从已有 LOWESS 曲线重新运行形状分类 (不重新拟合)。

    参数
    ----
    df_existing : 已有的 classification DataFrame
    curves_dict : {(Pair, Model, Zone): (xs, ys)}
    raw_data_fn : 回调函数 (pair_label, model, zone, eps) → (x, y) 或 (None, None)

    返回
    ----
    df_updated : 更新了分类列的 DataFrame
    """
    df = df_existing.copy()
    cls_rows = []
    for _, rec in df.iterrows():
        key = (rec["Pair"], rec["Model"], rec["Zone"])
        eps = rec.get("eps", np.nan)
        if key not in curves_dict:
            cls_rows.append(classify_panel(
                np.array([]), np.array([]), np.array([]), np.array([])))
            continue
        xs, ys = curves_dict[key]
        x, y = raw_data_fn(rec["Pair"], rec["Model"], rec["Zone"], eps)
        if x is None:
            cls_rows.append(classify_panel(
                np.array([]), np.array([]), np.array([]), np.array([])))
            continue
        cls_rows.append(classify_panel(x, y, xs, ys))

    cls_df = pd.DataFrame(cls_rows)
    for col in cls_df.columns:
        df[col] = cls_df[col].values
    return df
