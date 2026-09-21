"""LOWESS 响应曲线形状分类器

对每个 panel (变量对 × 模型 × 气候区) 的 LOWESS 曲线进行形状分类。
分类结果包含：响应置信度 (R²)、一级形状
(|Pearson| 判定 Simple / Complex)和二级形状
(Simple 用 Bow；Complex 用转折点 TP)。

所有可调参数集中在文件开头的 "可调参数" 区域，方便统一修改。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from minepy import MINE as _MINE
from scipy.signal import find_peaks as _find_peaks
from scipy.stats import spearmanr as _spearmanr
from scipy.optimize import curve_fit as _curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess

# ============================================================
#  可调参数 —— 修改分类行为只需改这里
# ============================================================

# ---- LOWESS 拟合参数 ----
LOWESS_FRAC = 0.2       # LOWESS 带宽 (占数据比例)
LOWESS_IT   = 3         # LOWESS 鲁棒迭代次数
MIN_N       = 30        # 最少样本量，低于此跳过

# ---- 全局依赖预筛选 ----
MIC_THRESHOLD = 0.20    # MIC < 此值：不进入 Branch / 最终形状路由
MINE_ALPHA = 0.6
MINE_C = 15

# ---- LOWESS 标准化振幅（仅作并列诊断，不覆盖形状）----
FLAT_EFFECT_THRESH = 0.20
NEAR_FLAT_EFFECT_THRESH = 0.30

# ---- EQ15 采样参数 ----
N_EQ_PTS    = 15        # 等距采样点数
DEADZONE    = 0.02      # 归一化相邻变化 d 的死区 (|d| < ε 视为零)

# ---- R² 响应强度阈值 ----
R2_WEAK     = 0.35      # R² < 此值 → Weak
R2_DETECT   = 0.60      # R² ≥ 此值 → Detectable；中间 → Uncertain

# ---- Pearson (诊断指标，可选 gate) ----
PEARSON_MONO_THRESH = 0.80   # |Pearson| 阈值 (仅当 PEARSON_GATE=True 时作为前置门控)
PEARSON_SOURCE = "eq15"      # "curve" = 用 LOWESS 原始曲线 Pearson; "eq15" = 用 EQ15
PEARSON_GATE = False         # True: TP=0 且 |Pearson|≥阈值 才进 PL R²; False: TP=0 直接进 PL R²

# ---- Power-law (Simple 确认) ----
POWERLAW_R2_THRESH = 0.60    # Power-law R² ≥ 此值 → 确认 Simple；< 此值 → 降级为 Complex
POWERLAW_N_RESAMPLE = 50     # Power-law 拟合前的 EQ 重采样点数

# ---- 转折点 prominence 过滤 ----
TP_AMPLITUDE_THRESH = 0.20   # prominence < 曲线 y range 的此比例 → 忽略该 TP

# ---- Bow 法 (单调子分类) ----
BOW_THRESH  = 0.15      # B < 此值 → linear；B ≥ 此值 → acceleration/saturation/sigmoid/cubic
SIGMOID_TAU_RATIO = 0.05  # e* 偏差 < tau (= ratio × endpoint_drop) 视为零，用于 sigmoid/cubic 检测
MIN_SIGMOID_FRAC  = 0.30  # sigmoid/cubic 需少数侧占比 ≥ 30%，否则回退 acceleration/saturation

# ---- 对称性 (非单调 tp=1 子分类) ----
SYM_THRESH  = 0.40      # sym ≥ 此值 → U_shaped；sym < 此值 → J/inverted_J


# ============================================================
#  MIC 全局依赖
# ============================================================

def compute_mine_stats(x, y):
    """在筛选后的原始散点上计算 MIC / MAS / MEV。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[finite], y[finite]
    if len(xf) < MIN_N or np.ptp(xf) <= 0 or np.ptp(yf) <= 0:
        return np.nan, np.nan, np.nan
    try:
        mine = _MINE(alpha=MINE_ALPHA, c=MINE_C)
        mine.compute_score(xf, yf)
        return float(mine.mic()), float(mine.mas()), float(mine.mev())
    except Exception:
        return np.nan, np.nan, np.nan


def compute_mic(x, y):
    """在筛选后的原始散点上计算 MIC（向后兼容）。"""
    mic, _, _ = compute_mine_stats(x, y)
    return mic


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


def _count_turning_points_filtered(y_eq, deadzone, amp_thresh=TP_AMPLITUDE_THRESH):
    """Prominence-based 转折点检测。

    用 scipy.signal.find_peaks 分别找峰和谷，
    只保留 prominence ≥ amp_thresh × y_range 的极值点。
    比斜率符号变化更鲁棒：平坦段的小波动不会产生虚假 TP。

    返回 (filtered_tp_count, raw_tp_count)。
    """
    y_range = float(np.max(y_eq) - np.min(y_eq))
    if y_range < 1e-12:
        return 0, 0

    min_prom = amp_thresh * y_range

    peak_idx, peak_props = _find_peaks(y_eq, prominence=0)
    trough_idx, trough_props = _find_peaks(-y_eq, prominence=0)

    raw_tp = len(peak_idx) + len(trough_idx)

    n_peaks = int(np.sum(peak_props["prominences"] >= min_prom))
    n_troughs = int(np.sum(trough_props["prominences"] >= min_prom))
    filtered_tp = n_peaks + n_troughs

    return filtered_tp, raw_tp


def get_tp_positions(xs, ys, n_pts=N_EQ_PTS, deadzone=DEADZONE,
                     amp_thresh=TP_AMPLITUDE_THRESH):
    """返回转折点在曲线上的 (x, y) 坐标，区分保留/过滤。

    使用 prominence-based 检测：峰和谷分别通过 find_peaks 找到，
    prominence ≥ amp_thresh × y_range 的保留，其余标记为 removed。

    返回 {'kept': [(x,y), ...], 'removed': [(x,y), ...]}。
    """
    x_eq = np.linspace(xs.min(), xs.max(), n_pts)
    y_eq = np.interp(x_eq, xs, ys)
    y_range = float(np.max(y_eq) - np.min(y_eq))

    if y_range < 1e-12:
        return {"kept": [], "removed": []}

    min_prom = amp_thresh * y_range
    kept, removed = [], []

    peak_idx, peak_props = _find_peaks(y_eq, prominence=0)
    for i, idx in enumerate(peak_idx):
        pt = (float(x_eq[idx]), float(y_eq[idx]))
        if peak_props["prominences"][i] >= min_prom:
            kept.append(pt)
        else:
            removed.append(pt)

    trough_idx, trough_props = _find_peaks(-y_eq, prominence=0)
    for i, idx in enumerate(trough_idx):
        pt = (float(x_eq[idx]), float(y_eq[idx]))
        if trough_props["prominences"][i] >= min_prom:
            kept.append(pt)
        else:
            removed.append(pt)

    return {"kept": kept, "removed": removed}


# ============================================================
#  Bow 法 —— 单调曲线的加速/饱和/线性子分类
# ============================================================

def _bow_classify(y_arr, bow_thresh=BOW_THRESH, tau_ratio=SIGMOID_TAU_RATIO,
                   min_sigmoid_frac=MIN_SIGMOID_FRAC):
    """Bow 法：用曲线偏离首尾连线判断曲率方向和翻转。

    返回 (B, 子分类)：
      B < bow_thresh → 'linear'
      B ≥ bow_thresh:
        e* 全负 → 'acceleration'
        e* 全正 → 'saturation'
        e* 有翻转且少数侧 ≥ min_sigmoid_frac → 'sigmoid' / 'cubic'
        e* 有翻转但少数侧 < min_sigmoid_frac → 按多数侧回退 acceleration/saturation
    """
    chord = y_arr[0] + (y_arr[-1] - y_arr[0]) * np.linspace(0, 1, len(y_arr))
    e = y_arr - chord
    drop = abs(y_arr[-1] - y_arr[0])
    b = float(np.mean(np.abs(e[1:-1])) / drop) if drop > 0 else 0.0
    sign_d = 1.0 if y_arr[-1] > y_arr[0] else -1.0
    e_star = sign_d * e[1:-1]
    if b < bow_thresh:
        cls = "linear"
    else:
        tau = tau_ratio * drop if drop > 0 else 0.0
        sig_mask = np.abs(e_star) > tau
        sig_vals = e_star[sig_mask]
        if len(sig_vals) == 0:
            cls = "linear"
        elif np.all(sig_vals < 0):
            cls = "acceleration"
        elif np.all(sig_vals > 0):
            cls = "saturation"
        else:
            n_neg = int(np.sum(sig_vals < 0))
            n_pos = int(np.sum(sig_vals > 0))
            minority_frac = min(n_neg, n_pos) / len(sig_vals)
            if minority_frac < min_sigmoid_frac:
                cls = "acceleration" if n_neg > n_pos else "saturation"
            else:
                sig_idx = np.where(sig_mask)[0]
                neg_center = float(np.median(sig_idx[sig_vals < 0]))
                pos_center = float(np.median(sig_idx[sig_vals > 0]))
                cls = "sigmoid" if neg_center < pos_center else "cubic"
    return b, cls


# ============================================================
#  Power-law 拟合 —— 单调性二级验证
# ============================================================

def _fit_powerlaw_r2(xs, ys, n_resample=POWERLAW_N_RESAMPLE):
    """在 EQ 重采样的 LOWESS 曲线上拟合 y = a·x^b + c，返回 (R², b)。

    真正的单调曲线能很好地被 power-law 拟合 (R²>0.7)，
    而含驼峰/波动的曲线则拟合差 (R²<0.7)，以此过滤 Pearson 假阳性。
    """
    x_eq = np.linspace(xs.min(), xs.max(), n_resample)
    y_eq = np.interp(x_eq, xs, ys)
    x_shift = x_eq - x_eq.min() + 1.0
    flipped = y_eq[-1] < y_eq[0]
    if flipped:
        y_eq = -y_eq
    y_shift = y_eq - y_eq.min() + 1.0

    def _powerlaw(x, a, b, c):
        return a * np.power(x, b) + c

    try:
        popt, _ = _curve_fit(
            _powerlaw, x_shift, y_shift,
            p0=[1, 1, 0], maxfev=20000,
            bounds=([0, 0.01, -np.inf], [np.inf, 10, np.inf]),
        )
        y_pred = _powerlaw(x_shift, *popt)
        ss_res = np.sum((y_shift - y_pred) ** 2)
        ss_tot = np.sum((y_shift - y_shift.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return float(r2), float(popt[1])
    except Exception:
        return np.nan, np.nan


def fit_powerlaw_curve(xs, ys, n_resample=POWERLAW_N_RESAMPLE):
    """拟合 power-law 并返回原始坐标下的拟合曲线 (x_plot, y_plot)。

    用于在图上叠加 power-law 拟合线。若拟合失败返回 (None, None)。
    """
    x_eq = np.linspace(xs.min(), xs.max(), n_resample)
    y_eq = np.interp(x_eq, xs, ys)
    x_shift = x_eq - x_eq.min() + 1.0
    flipped = y_eq[-1] < y_eq[0]
    if flipped:
        y_eq_fit = -y_eq
    else:
        y_eq_fit = y_eq
    y_min = y_eq_fit.min()
    y_shift = y_eq_fit - y_min + 1.0

    def _powerlaw(x, a, b, c):
        return a * np.power(x, b) + c

    try:
        popt, _ = _curve_fit(
            _powerlaw, x_shift, y_shift,
            p0=[1, 1, 0], maxfev=20000,
            bounds=([0, 0.01, -np.inf], [np.inf, 10, np.inf]),
        )
        y_pred_shifted = _powerlaw(x_shift, *popt)
        y_pred = y_pred_shifted - 1.0 + y_min
        if flipped:
            y_pred = -y_pred
        return x_eq, y_pred
    except Exception:
        return None, None


# ============================================================
#  单个 panel 的形状分类
# ============================================================

def classify_panel(x, y, xs, ys,
                   n_pts=N_EQ_PTS,
                   deadzone=DEADZONE,
                   pearson_thresh=PEARSON_MONO_THRESH,
                   bow_thresh=BOW_THRESH,
                   sym_thresh=SYM_THRESH,
                   mic=None, mas=None, mev=None,
                   mic_threshold=MIC_THRESHOLD,
                   flat_threshold=FLAT_EFFECT_THRESH,
                   near_flat_threshold=NEAR_FLAT_EFFECT_THRESH):
    """对一个 panel 的 LOWESS 曲线进行形状分类。

    参数
    ----
    x, y : 筛选后的原始散点数据 (用于计算 y_scale 和 R²)
    xs, ys : LOWESS 拟合曲线 (已按 x 排序)
    n_pts : EQ 等距采样点数 (默认 15)
    deadzone : 归一化斜率死区
    pearson_thresh : |Pearson| ≥ 此值 → Simple，否则 Complex
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
        "mic": np.nan,
        "ls_mas": np.nan,
        "ls_mev": np.nan,
        "mic_pass": False,
        "ls_effect_size": np.nan,
        "ls_lowess_amplitude": np.nan,
        "ls_y_scale": np.nan,
        "flatness_class": "Insufficient",
        "lowess_shape": "skipped",
        "lowess_shape2": "",
        "lowess_detail": "",
        "direction": "",
        "shape_label": "",
        "ls_pearson": np.nan,
        "ls_abs_pearson": np.nan,
        "ls_curve_pearson": np.nan,
        "ls_curve_spearman": np.nan,
        "ls_powerlaw_r2": np.nan,
        "ls_powerlaw_b": np.nan,
        "ls_bow": np.nan,
        "ls_bow_p": np.nan,
        "lowess_shape2_p": "",
        "ls_n_tp": 0,
        "ls_n_tp_raw": 0,
        "ls_n_tp_filtered": 0,
        "ls_p0": np.nan,
        "ls_p_pos": np.nan,
        "ls_p_neg": np.nan,
        "ls_n0": 0,
        "ls_n_pos": 0,
        "ls_n_neg": 0,
        "ls_V_pos": np.nan,
        "ls_V_neg": np.nan,
        "ls_R_rev": np.nan,
    }

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[finite], y[finite]

    if len(xf) < n_pts * 3:
        out["lowess_shape"] = "too_few"
        return out

    # ---- Gate 1: MIC / MAS / MEV ----
    if mic is None:
        mic_value, mas_value, mev_value = compute_mine_stats(xf, yf)
    else:
        mic_value = float(mic)
        mas_value = float(mas) if mas is not None else np.nan
        mev_value = float(mev) if mev is not None else np.nan
    out["mic"] = mic_value
    out["ls_mas"] = mas_value
    out["ls_mev"] = mev_value
    if not np.isfinite(mic_value):
        out["mic_pass"] = False
    else:
        out["mic_pass"] = bool(mic_value >= mic_threshold)

    # ---- Step 1: R² confidence ----
    out["r2"] = compute_r2(xf, yf, xs, ys)
    out["strength"] = r2_strength(out["r2"])

    # ---- Step 2: y_scale (P95 - P05) ----
    y_scale = float(np.percentile(yf, 95) - np.percentile(yf, 5))
    if y_scale <= 0:
        y_scale = float(np.ptp(yf))
    if y_scale <= 0:
        out["lowess_shape"] = "no_variation"
        return out
    out["ls_y_scale"] = y_scale

    # ---- 独立诊断: LOWESS 振幅 / y 的稳健范围 ----
    ys_arr = np.asarray(ys, float)
    ys_finite = ys_arr[np.isfinite(ys_arr)]
    if len(ys_finite) >= 2:
        lowess_amplitude = float(np.max(ys_finite) - np.min(ys_finite))
        effect = lowess_amplitude / y_scale
        out["ls_lowess_amplitude"] = lowess_amplitude
        out["ls_effect_size"] = effect
        if effect < flat_threshold:
            out["flatness_class"] = "Flat"
        elif effect < near_flat_threshold:
            out["flatness_class"] = "Near-flat"
        else:
            out["flatness_class"] = "Non-flat"

    # ---- Step 2b: 原始 LOWESS 曲线的 Pearson / Spearman ----
    xs_arr = np.asarray(xs, float)
    if len(ys_finite) >= 5 and len(xs_arr) == len(ys_arr):
        xs_finite = xs_arr[np.isfinite(ys_arr)]
        if np.ptp(ys_finite) > 0 and np.ptp(xs_finite) > 0:
            out["ls_curve_pearson"] = float(np.corrcoef(xs_finite, ys_finite)[0, 1])
            out["ls_curve_spearman"] = float(_spearmanr(xs_finite, ys_finite).statistic)

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

    # ---- Step 5: EQ Pearson ----
    if np.ptp(y_eq) <= 0:
        pr = 0.0
    else:
        pr = float(np.corrcoef(x_eq, y_eq)[0, 1])
    out["ls_pearson"] = pr
    out["ls_abs_pearson"] = abs(pr)

    # ---- Step 6: TP (原始 + 振幅过滤) ----
    n_tp = _count_turning_points(d, deadzone)
    n_tp_filtered, n_tp_raw = _count_turning_points_filtered(y_eq, deadzone)
    out["ls_n_tp"] = n_tp
    out["ls_n_tp_raw"] = n_tp_raw
    out["ls_n_tp_filtered"] = n_tp_filtered

    # ==== Pearson 来源选择 (辅助指标) ====
    if PEARSON_SOURCE == "curve" and np.isfinite(out.get("ls_curve_pearson", np.nan)):
        pr_decision = out["ls_curve_pearson"]
    else:
        pr_decision = pr

    # ==== 形状分类: TP-first ====
    # TP=0 → (可选 Pearson gate) → PL R² → Simple；TP≥1 → Complex
    is_simple = False

    if n_tp_filtered == 0:
        pearson_ok = (not PEARSON_GATE) or (abs(pr_decision) >= pearson_thresh)
        if pearson_ok:
            pl_r2, pl_b = _fit_powerlaw_r2(xs, ys, POWERLAW_N_RESAMPLE)
            out["ls_powerlaw_r2"] = pl_r2
            out["ls_powerlaw_b"] = pl_b
            if np.isfinite(pl_r2) and pl_r2 >= POWERLAW_R2_THRESH:
                is_simple = True

    if is_simple:
        endpoint_change = float(y_eq[-1] - y_eq[0])
        direction_signal = endpoint_change if abs(endpoint_change) > 1e-12 else pr_decision
        direction = "up" if direction_signal >= 0 else "down"
        out["lowess_shape"] = "simple"
        out["lowess_detail"] = f"monotonic_{direction}"

        B, shape2 = _bow_classify(y_eq, bow_thresh)
        out["ls_bow"] = B
        out["lowess_shape2"] = shape2

        direction_arrow = "↑" if direction_signal >= 0 else "↓"
        out["direction"] = direction_arrow
        out["shape_label"] = f"{shape2}{direction_arrow}"

        i05 = max(1, int(round(0.05 * (n_pts - 1))))
        i95 = min(n_pts - 2, int(round(0.95 * (n_pts - 1))))
        y_trim = y_eq[i05:i95 + 1]
        Bp, shape2p = _bow_classify(y_trim, bow_thresh)
        out["ls_bow_p"] = Bp
        out["lowess_shape2_p"] = shape2p

        return out

    # ---- Complex: 按 filtered TP 子分类 ----
    out["lowess_shape"] = "complex"

    if n_tp_filtered == 0:
        out["lowess_shape2"] = "TP=0"
        out["lowess_detail"] = "no_significant_turn"
        tp0_dir = "↑" if y_eq[-1] > y_eq[0] else "↓"
        out["direction"] = tp0_dir
        out["shape_label"] = f"TP=0{tp0_dir}"
    elif n_tp_filtered == 1:
        out["lowess_shape2"] = "TP=1"
        min_prom = TP_AMPLITUDE_THRESH * float(np.ptp(y_eq))
        pk_i, pk_p = _find_peaks(y_eq, prominence=min_prom)
        tr_i, tr_p = _find_peaks(-y_eq, prominence=min_prom)
        is_peak = len(pk_i) > 0
        first_sign = 1 if is_peak else -1
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
            out["lowess_detail"] = "inverted_U"
            out["direction"] = "∩"
        else:
            if sym >= sym_thresh:
                out["lowess_detail"] = "U_shaped"
                out["direction"] = "U"
            else:
                if y_eq[-1] > y_eq[0]:
                    out["lowess_detail"] = "J_shaped"
                    out["direction"] = "J"
                else:
                    out["lowess_detail"] = "inverted_J"
                    out["direction"] = "iJ"
        out["shape_label"] = f"TP=1({out['direction']})"
    else:
        out["lowess_shape2"] = "TP>=2"
        out["lowess_detail"] = "S_or_N" if n_tp_filtered == 2 else "multi_turn"
        out["shape_label"] = "TP≥2"

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
    """并行对所有 panel 进行 MIC gate + LOWESS 拟合 + 形状分类。

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

        mic_threshold = float(job.get("mic_threshold", MIC_THRESHOLD))
        mic, mas, mev = compute_mine_stats(x, y)

        fit = job.get("fit")
        if fit is None:
            fit = fit_lowess(x, y)
        if fit is None:
            rec.update({
                "r2": np.nan, "strength": "Insufficient",
                "mic": mic, "ls_mas": mas, "ls_mev": mev,
                "mic_pass": bool(np.isfinite(mic) and mic >= mic_threshold),
                "ls_effect_size": np.nan,
                "ls_lowess_amplitude": np.nan,
                "ls_y_scale": np.nan,
                "flatness_class": "Insufficient",
                "lowess_shape": "fit_failed", "lowess_shape2": "",
                "lowess_detail": "", "direction": "", "shape_label": "",
                "ls_pearson": np.nan, "ls_abs_pearson": np.nan,
                "ls_curve_pearson": np.nan, "ls_curve_spearman": np.nan,
                "ls_powerlaw_r2": np.nan, "ls_powerlaw_b": np.nan,
                "ls_bow": np.nan, "ls_bow_p": np.nan,
                "lowess_shape2_p": "", "ls_n_tp": 0,
            })
            return rec, None

        xs, ys = fit
        rec.update(classify_panel(
            x, y, xs, ys, mic=mic, mas=mas, mev=mev,
            mic_threshold=mic_threshold,
        ))

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
        x, y = raw_data_fn(rec["Pair"], rec["Model"], rec["Zone"], eps)
        if x is None:
            cls_rows.append(classify_panel(
                np.array([]), np.array([]), np.array([]), np.array([])))
            continue
        mic, mas, mev = compute_mine_stats(x, y)
        if key not in curves_dict:
            row = classify_panel(
                np.array([]), np.array([]), np.array([]), np.array([]))
            row.update({"mic": mic, "ls_mas": mas, "ls_mev": mev,
                        "mic_pass": bool(np.isfinite(mic) and mic >= MIC_THRESHOLD),
                        "lowess_shape": "fit_failed"})
            cls_rows.append(row)
            continue
        xs, ys = curves_dict[key]
        cls_rows.append(classify_panel(x, y, xs, ys, mic=mic, mas=mas, mev=mev))

    cls_df = pd.DataFrame(cls_rows)
    for col in cls_df.columns:
        df[col] = cls_df[col].values
    return df
