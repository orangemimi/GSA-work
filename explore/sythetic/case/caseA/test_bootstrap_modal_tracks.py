"""Adaptive conditional modal tracks with bootstrap stability.

Diagnostic only.  Hexbin is not used for inference here.  Equal-count,
overlapping x windows adapt to dense and sparse x regions.  Within each window,
histogram KDE modes must persist across adjacent smoothing bandwidths and under
multinomial bootstrap resampling.  Stable modes are linked across x and
classified as Branch, Two-band, Candidate, or No branch.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
SIGMAS = (1.0, 1.5, 2.0, 2.75, 3.5)
N_BOOTSTRAP = 40
N_Y_BINS = 64
MIN_BOOTSTRAP_SUPPORT = 0.70


def best_pair(histogram, sigma):
    smooth = gaussian_filter1d(histogram.astype(float), sigma=sigma, mode="nearest")
    if smooth.max() <= 0:
        return None
    prominence = max(0.05 * float(smooth.max()), 0.5)
    peaks = list(find_peaks(
        smooth, prominence=prominence, distance=max(2, int(round(1.5 * sigma))),
    )[0])
    if smooth[0] >= smooth[1] and smooth[0] >= 0.10 * smooth.max():
        peaks.insert(0, 0)
    if smooth[-1] >= smooth[-2] and smooth[-1] >= 0.10 * smooth.max():
        peaks.append(len(smooth) - 1)
    peaks = sorted(set(peaks))
    candidates = []
    for left_i in range(len(peaks) - 1):
        for right_i in range(left_i + 1, len(peaks)):
            left, right = peaks[left_i], peaks[right_i]
            if right - left < max(3, int(np.ceil(2 * sigma))):
                continue
            valley = left + int(np.argmin(smooth[left : right + 1]))
            low_peak = min(smooth[left], smooth[right])
            valley_depth = 1 - smooth[valley] / max(low_peak, 1e-12)
            left_mass = histogram[: valley + 1].sum()
            right_mass = histogram[valley + 1 :].sum()
            min_weight = min(left_mass, right_mass) / max(histogram.sum(), 1)
            relative_peak = low_peak / smooth.max()
            valid = (
                valley_depth >= 0.30
                and min_weight >= 0.08
                and relative_peak >= 0.15
            )
            strength = valley_depth * min_weight * relative_peak * (right - left)
            candidates.append({
                "valid": valid,
                "left": int(left),
                "right": int(right),
                "valley_depth": float(valley_depth),
                "min_weight": float(min_weight),
                "strength": float(strength),
            })
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["valid"], item["strength"]))


def persistent_pair(histogram):
    pairs = [best_pair(histogram, sigma) for sigma in SIGMAS]
    supported = []
    for index in range(len(pairs) - 1):
        left, right = pairs[index], pairs[index + 1]
        if left is None or right is None or not left["valid"] or not right["valid"]:
            continue
        if (
            abs(left["left"] - right["left"]) <= 0.10 * len(histogram)
            and abs(left["right"] - right["right"]) <= 0.10 * len(histogram)
        ):
            supported.extend((left, right))
    if not supported:
        return None
    left = float(np.median([item["left"] for item in supported]))
    right = float(np.median([item["right"] for item in supported]))
    return {
        "left": left,
        "right": right,
        "valley_depth": float(np.median([item["valley_depth"] for item in supported])),
        "min_weight": float(np.median([item["min_weight"] for item in supported])),
    }


def bootstrap_pair(histogram, rng):
    original = persistent_pair(histogram)
    if original is None:
        return None
    total = int(histogram.sum())
    probabilities = histogram / max(total, 1)
    success = 0
    left_modes, right_modes = [], []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.multinomial(total, probabilities)
        pair = persistent_pair(sampled)
        if pair is None:
            continue
        if (
            abs(pair["left"] - original["left"]) <= 0.12 * len(histogram)
            and abs(pair["right"] - original["right"]) <= 0.12 * len(histogram)
        ):
            success += 1
            left_modes.append(pair["left"])
            right_modes.append(pair["right"])
    support = success / N_BOOTSTRAP
    return {
        **original,
        "bootstrap_support": float(support),
        "left": float(np.median(left_modes)) if left_modes else original["left"],
        "right": float(np.median(right_modes)) if right_modes else original["right"],
    }


def longest_flag_run(windows):
    best = []
    current = []
    for window in windows:
        if window["flag"]:
            current.append(window)
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    return best


def classify_panel(x, y, seed):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    n = len(x)
    width = min(n, max(80, int(round(0.15 * n))))
    step = max(1, width // 2)
    starts = list(range(0, max(n - width + 1, 1), step))
    if not starts or starts[-1] != n - width:
        starts.append(max(0, n - width))

    y_lo, y_hi = np.quantile(y, [0.0025, 0.9975])
    if y_hi <= y_lo:
        y_lo, y_hi = float(np.min(y)), float(np.max(y) + 1e-12)
    edges = np.linspace(y_lo, y_hi, N_Y_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    rng = np.random.default_rng(seed)
    windows = []
    for start in starts:
        stop = start + width
        xw = x[start:stop]
        yw = np.clip(y[start:stop], y_lo, y_hi)
        histogram, _ = np.histogram(yw, bins=edges)
        pair = bootstrap_pair(histogram, rng)
        flag = bool(pair is not None and pair["bootstrap_support"] >= MIN_BOOTSTRAP_SUPPORT)
        windows.append({
            "x_min": float(xw.min()),
            "x_mid": float(np.median(xw)),
            "x_max": float(xw.max()),
            "flag": flag,
            "bootstrap_support": pair["bootstrap_support"] if pair else 0.0,
            "lower": float(np.interp(pair["left"], np.arange(N_Y_BINS), centers)) if pair else np.nan,
            "upper": float(np.interp(pair["right"], np.arange(N_Y_BINS), centers)) if pair else np.nan,
            "valley_depth": pair["valley_depth"] if pair else np.nan,
            "min_weight": pair["min_weight"] if pair else np.nan,
        })

    run = longest_flag_run(windows)
    if len(run) < 2:
        status = "No branch"
        rho = relative_change = 0.0
    else:
        separation = np.asarray([item["upper"] - item["lower"] for item in run])
        x_mid = np.asarray([item["x_mid"] for item in run])
        relative_change = float(
            abs(separation[-1] - separation[0]) / max(float(np.max(separation)), 1e-12)
        )
        if len(run) >= 3 and np.ptp(separation) > 1e-12:
            rho = float(spearmanr(x_mid, separation).statistic)
            if not np.isfinite(rho):
                rho = 0.0
        else:
            rho = 0.0
        if len(run) >= 3 and abs(rho) >= 0.70 and relative_change >= 0.25:
            status = "Branch"
        elif len(run) >= 3:
            status = "Two-band"
        else:
            status = "Candidate"
    return {
        "status": status,
        "n_windows": len(windows),
        "run_length": len(run),
        "median_bootstrap": float(np.median([w["bootstrap_support"] for w in run])) if run else 0.0,
        "x_start": run[0]["x_min"] if run else np.nan,
        "x_end": run[-1]["x_max"] if run else np.nan,
        "separation_rho": rho,
        "relative_separation_change": relative_change,
    }


def plot_heatmap(results):
    codes = {"No branch": 0, "Candidate": 1, "Two-band": 2, "Branch": 3}
    letters = {"No branch": "-", "Candidate": "C", "Two-band": "T", "Branch": "B"}
    cmap = ListedColormap(["#F3F4F6", "#F59E0B", "#7C3AED", "#2563EB"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True)
    for row, version in enumerate(("raw", "trimmed")):
        for col, (_, _, pair) in enumerate(PAIRS):
            ax = axes[row, col]
            selected = results.loc[
                (results["pair"] == pair) & (results["version"] == version)
            ].copy()
            selected["code"] = selected["status"].map(codes)
            matrix = selected.pivot(index="model", columns="domain", values="code").reindex(
                index=MODELS, columns=DOMAINS
            )
            statuses = selected.pivot(index="model", columns="domain", values="status").reindex(
                index=MODELS, columns=DOMAINS
            )
            run = selected.pivot(index="model", columns="domain", values="run_length").reindex(
                index=MODELS, columns=DOMAINS
            )
            ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    status = statuses.iloc[i, j]
                    color = "white" if codes[status] >= 2 else "#374151"
                    ax.text(j, i, f"{letters[status]}\nrun={int(run.iloc[i,j])}",
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color=color)
            counts = selected["status"].value_counts()
            ax.set_title(
                f"{pair} · {version.upper()} · B={counts.get('Branch',0)}, "
                f"T={counts.get('Two-band',0)}, C={counts.get('Candidate',0)}",
                fontsize=10.2, fontweight="bold",
            )
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.tick_params(labelsize=8)
    fig.suptitle(
        "Adaptive conditional modal tracks with bootstrap stability",
        fontsize=15, fontweight="bold",
    )
    path = FIGURE_DIR / "S4_bootstrap_modal_tracks_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    cases = build_cases(load_runs())
    rows = []
    for index, (key, (x, y)) in enumerate(cases.items()):
        model, domain, pair, version = key
        result = classify_panel(x, y, seed=17000 + index)
        rows.append({"model": model, "domain": domain, "pair": pair, "version": version, **result})
    results = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "S4_bootstrap_modal_tracks_results.csv"
    results.to_csv(csv_path, index=False)
    figure_path = plot_heatmap(results)
    print(results.groupby(["pair", "version", "status"]).size().to_string())
    print("\nHighlighted panel:")
    print(results.loc[
        (results["model"] == "CanESM5") & (results["domain"] == "CW")
        & (results["pair"] == "mrros → Q") & (results["version"] == "trimmed")
    ].to_string(index=False))
    print("\nP->Q non-null:")
    print(results.loc[
        (results["pair"] == "P → Q") & (results["status"] != "No branch")
    ].to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
