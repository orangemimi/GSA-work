"""Single-score conditional KDE peak-valley branch pilot.

Local bimodality uses one multi-bandwidth peak-valley persistence score.  Its
cutoff is the 95th percentile under Monte Carlo Gaussian unimodal samples; the
old minimum mass, valley depth, separation, and prominence gates are absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

try:
    from numba import njit
except ImportError:  # pragma: no cover - numpy fallback
    njit = None

from branch_benchmark import _robust_location_scale, _window_slices
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS
from test_sliding_kde_modal_summary import stratified_cap


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BANDWIDTH_FACTORS = (0.65, 0.80, 1.00, 1.25, 1.50)
NULL_SAMPLE_SIZES = np.asarray([120, 180, 240, 300, 360])
N_NULL_REPLICATES = 250
KDE_GRID_SIZE = 400
_KDE_Z_CLIP = 40.0
_ZERO_PEAK = {
    "score": 0.0,
    "modes": (np.nan, np.nan),
    "valley_depth": np.nan,
    "min_weight": np.nan,
}


def _gaussian_kde_eval_numpy(samples, grid, inv_h, norm):
    dens = np.zeros(grid.size, dtype=np.float64)
    chunk = max(1, 250000 // max(grid.size, 1))
    for start in range(0, samples.size, chunk):
        sl = samples[start:start + chunk]
        z = (grid[:, None] - sl[None, :]) * inv_h
        np.clip(z, -_KDE_Z_CLIP, _KDE_Z_CLIP, out=z)
        dens += np.exp(-0.5 * z * z).sum(axis=1)
    dens *= norm
    return dens


if njit is not None:
    @njit(cache=True, nogil=True, fastmath=True)
    def _gaussian_kde_eval(samples, grid, inv_h, norm):
        n_grid = grid.shape[0]
        n = samples.shape[0]
        dens = np.empty(n_grid, dtype=np.float64)
        for i in range(n_grid):
            acc = 0.0
            gi = grid[i]
            for j in range(n):
                z = (gi - samples[j]) * inv_h
                acc += math.exp(-0.5 * z * z)
            dens[i] = acc * norm
        return dens

    @njit(cache=True, nogil=True, fastmath=True)
    def _gaussian_kde_eval_multi(samples, grid, inv_hs, norms):
        n_factor = inv_hs.shape[0]
        n_grid = grid.shape[0]
        n = samples.shape[0]
        dens = np.empty((n_factor, n_grid), dtype=np.float64)
        for f in range(n_factor):
            inv_h = inv_hs[f]
            norm = norms[f]
            for i in range(n_grid):
                acc = 0.0
                gi = grid[i]
                for j in range(n):
                    z = (gi - samples[j]) * inv_h
                    acc += math.exp(-0.5 * z * z)
                dens[f, i] = acc * norm
        return dens

    @njit(cache=True, nogil=True, fastmath=True)
    def _peak_valley_multi_core(clipped, grid, factors):
        n = clipped.shape[0]
        n_grid = grid.shape[0]
        n_f = factors.shape[0]
        out = np.empty((n_f, 7), dtype=np.float64)
        for f in range(n_f):
            out[f, 0] = 0.0
            out[f, 1] = 0.0
            out[f, 2] = np.nan
            out[f, 3] = np.nan
            out[f, 4] = np.nan
            out[f, 5] = np.nan
            out[f, 6] = np.nan

        mean = 0.0
        for j in range(n):
            mean += clipped[j]
        mean /= n
        var = 0.0
        for j in range(n):
            d = clipped[j] - mean
            var += d * d
        var /= (n - 1.0)
        if not (var > 0.0):
            return out
        std = math.sqrt(var)
        n_scott = n ** -0.2
        inv_sqrt_2pi = 0.3989422804014327
        peak_idx = np.empty(n_grid, dtype=np.int64)
        n_float = float(n)

        for f in range(n_f):
            bw = std * n_scott * factors[f]
            if not (bw > 0.0):
                continue
            inv_h = 1.0 / bw
            norm = inv_h * inv_sqrt_2pi / n_float
            density = np.empty(n_grid, dtype=np.float64)
            dens_max = 0.0
            for i in range(n_grid):
                acc = 0.0
                gi = grid[i]
                for j in range(n):
                    z = (gi - clipped[j]) * inv_h
                    acc += math.exp(-0.5 * z * z)
                val = acc * norm
                density[i] = val
                if val > dens_max:
                    dens_max = val

            n_peaks = 0
            if density[0] > density[1]:
                peak_idx[0] = 0
                n_peaks = 1
            for i in range(1, n_grid - 1):
                if density[i] > density[i - 1] and density[i] > density[i + 1]:
                    peak_idx[n_peaks] = i
                    n_peaks += 1
            if density[n_grid - 1] > density[n_grid - 2]:
                peak_idx[n_peaks] = n_grid - 1
                n_peaks += 1
            if n_peaks < 2:
                continue

            best_score = -1.0
            best_pair = 0.0
            best_ml = np.nan
            best_mh = np.nan
            best_v = np.nan
            best_vd = np.nan
            best_mw = np.nan
            for a in range(n_peaks - 1):
                left = peak_idx[a]
                for b in range(a + 1, n_peaks):
                    right = peak_idx[b]
                    vidx = left
                    vmin = density[left]
                    for k in range(left, right + 1):
                        if density[k] < vmin:
                            vmin = density[k]
                            vidx = k
                    dl = density[left]
                    dr = density[right]
                    lower = dl if dl < dr else dr
                    higher = dr if dl < dr else dl
                    persistence = lower - vmin
                    if persistence < 0.0:
                        persistence = 0.0
                    denom_max = dens_max if dens_max > 1e-12 else 1e-12
                    denom_high = higher if higher > 1e-12 else 1e-12
                    denom_low = lower if lower > 1e-12 else 1e-12
                    score = persistence / denom_max
                    pair_score = persistence / denom_high
                    cut = grid[vidx]
                    n_le = 0
                    for j in range(n):
                        if clipped[j] <= cut:
                            n_le += 1
                    w1 = n_le / n_float
                    w2 = 1.0 - w1
                    min_weight = w1 if w1 < w2 else w2
                    valley_depth = persistence / denom_low
                    if score > best_score:
                        best_score = score
                        best_pair = pair_score
                        best_ml = grid[left]
                        best_mh = grid[right]
                        best_v = cut
                        best_vd = valley_depth
                        best_mw = min_weight
            if best_score >= 0.0:
                out[f, 0] = best_score
                out[f, 1] = best_pair
                out[f, 2] = best_ml
                out[f, 3] = best_mh
                out[f, 4] = best_v
                out[f, 5] = best_vd
                out[f, 6] = best_mw
        return out
else:  # pragma: no cover
    def _gaussian_kde_eval(samples, grid, inv_h, norm):
        return _gaussian_kde_eval_numpy(samples, grid, inv_h, norm)

    def _gaussian_kde_eval_multi(samples, grid, inv_hs, norms):
        return np.vstack([
            _gaussian_kde_eval_numpy(samples, grid, float(inv_hs[i]), float(norms[i]))
            for i in range(inv_hs.shape[0])
        ])

    def _peak_valley_multi_core(clipped, grid, factors):
        densities = _gaussian_kdes_1d_scott(clipped, grid, factors)
        out = np.empty((len(factors), 7), dtype=np.float64)
        for i, density in enumerate(densities):
            item = _best_peak_valley(clipped, grid, density)
            out[i, 0] = item["score"]
            out[i, 1] = float(item.get("pair_score", 0.0))
            out[i, 2] = item["modes"][0]
            out[i, 3] = item["modes"][1]
            out[i, 4] = float(item.get("valley", np.nan))
            out[i, 5] = item["valley_depth"]
            out[i, 6] = item["min_weight"]
        return out


def _scott_inv_h_norm(samples, factor):
    n = int(samples.size)
    if n < 2:
        return None
    var = float(np.var(samples, ddof=1))
    if not np.isfinite(var) or var <= 0.0:
        return None
    bw = float(np.sqrt(var) * (n ** -0.2) * float(factor))
    if not np.isfinite(bw) or bw <= 0.0:
        return None
    inv_h = 1.0 / bw
    norm = inv_h / (n * np.sqrt(2.0 * np.pi))
    return inv_h, norm


def _gaussian_kde_1d_scott(samples, grid, factor):
    """1-D Scott Gaussian KDE, matching scipy.stats.gaussian_kde for d=1.

    ``factor`` multiplies Scott's rule exactly as
    ``bw_method=lambda obj: obj.scotts_factor() * factor``.
    """
    samples = np.ascontiguousarray(samples, dtype=np.float64)
    grid = np.ascontiguousarray(grid, dtype=np.float64)
    scale = _scott_inv_h_norm(samples, factor)
    if scale is None:
        return None
    inv_h, norm = scale
    return _gaussian_kde_eval(samples, grid, inv_h, norm)


def _gaussian_kdes_1d_scott(samples, grid, factors):
    """Evaluate the 1-D Scott KDE on one sample/grid for several bandwidths."""
    samples = np.ascontiguousarray(samples, dtype=np.float64)
    grid = np.ascontiguousarray(grid, dtype=np.float64)
    n_factor = len(factors)
    n = int(samples.size)
    if n < 2:
        return [None] * n_factor
    var = float(np.var(samples, ddof=1))
    if not np.isfinite(var) or var <= 0.0:
        return [None] * n_factor
    std = math.sqrt(var)
    n_scott = n ** -0.2
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    inv_hs = np.empty(n_factor, dtype=np.float64)
    norms = np.empty(n_factor, dtype=np.float64)
    valid = np.ones(n_factor, dtype=bool)
    any_valid = False
    for i, factor in enumerate(factors):
        bw = std * n_scott * float(factor)
        if not np.isfinite(bw) or bw <= 0.0:
            valid[i] = False
            inv_hs[i] = 1.0
            norms[i] = 0.0
        else:
            inv_hs[i] = 1.0 / bw
            norms[i] = inv_hs[i] * inv_sqrt_2pi / n
            any_valid = True
    if not any_valid:
        return [None] * n_factor
    dens = _gaussian_kde_eval_multi(samples, grid, inv_hs, norms)
    return [dens[i] if valid[i] else None for i in range(n_factor)]


def _peak_indices(density):
    peaks = list(find_peaks(density)[0])
    if density[0] > density[1]:
        peaks.insert(0, 0)
    if density[-1] > density[-2]:
        peaks.append(len(density) - 1)
    return sorted(set(int(index) for index in peaks))


def _best_peak_valley(clipped, grid, density):
    if density is None:
        return dict(_ZERO_PEAK)
    peaks = _peak_indices(density)
    if len(peaks) < 2:
        return dict(_ZERO_PEAK)
    dens_max = float(np.max(density))
    best = None
    n = float(clipped.size)
    for left_index in range(len(peaks) - 1):
        for right_index in range(left_index + 1, len(peaks)):
            left, right = peaks[left_index], peaks[right_index]
            valley = left + int(np.argmin(density[left:right + 1]))
            lower_peak = min(float(density[left]), float(density[right]))
            higher_peak = max(float(density[left]), float(density[right]))
            persistence = max(lower_peak - float(density[valley]), 0.0)
            score = persistence / max(dens_max, 1e-12)
            pair_score = persistence / max(higher_peak, 1e-12)
            cut = grid[valley]
            n_le = float(np.count_nonzero(clipped <= cut))
            min_weight = min(n_le / n, 1.0 - n_le / n)
            valley_depth = persistence / max(lower_peak, 1e-12)
            candidate = {
                "score": float(score),
                "pair_score": float(pair_score),
                "modes": (float(grid[left]), float(grid[right])),
                "valley": float(grid[valley]),
                "valley_depth": float(valley_depth),
                "min_weight": float(min_weight),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    return best or dict(_ZERO_PEAK)


def _rows_to_peak_dicts(rows):
    out = []
    for row in rows:
        score = float(row[0])
        if score > 0.0 and np.isfinite(row[2]) and np.isfinite(row[3]):
            out.append({
                "score": score,
                "pair_score": float(row[1]),
                "modes": (float(row[2]), float(row[3])),
                "valley": float(row[4]),
                "valley_depth": float(row[5]),
                "min_weight": float(row[6]),
            })
        else:
            out.append(dict(_ZERO_PEAK))
    return out


def peak_valley_at_bandwidths(values, factors=BANDWIDTH_FACTORS):
    """Shared clip/grid KDE peak-valley estimates for several bandwidths."""
    values = np.asarray(values, dtype=np.float64)
    zeros = [dict(_ZERO_PEAK) for _ in factors]
    if values.size < 30:
        return zeros
    lo, hi = np.quantile(values, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return zeros
    clipped = np.ascontiguousarray(
        values[(values >= lo) & (values <= hi)], dtype=np.float64,
    )
    if clipped.size < 30:
        return zeros
    grid = np.linspace(lo, hi, KDE_GRID_SIZE)
    factor_arr = np.ascontiguousarray(factors, dtype=np.float64)
    rows = _peak_valley_multi_core(clipped, grid, factor_arr)
    return _rows_to_peak_dicts(rows)


def peak_valley_at_bandwidth(values, factor):
    return peak_valley_at_bandwidths(values, (factor,))[0]


def _warmup_fast_kde():
    samples = np.linspace(-1.0, 1.0, 40)
    grid = np.linspace(-1.0, 1.0, 32)
    _gaussian_kde_1d_scott(samples, grid, 1.0)
    _peak_valley_multi_core(
        samples, grid, np.asarray(BANDWIDTH_FACTORS, dtype=np.float64),
    )


try:
    _warmup_fast_kde()
except Exception:
    pass


def multiband_score(values):
    estimates = peak_valley_at_bandwidths(values, BANDWIDTH_FACTORS)
    scores = np.asarray([item["score"] for item in estimates])
    score = float(np.median(scores))
    positive = [item for item in estimates if item["score"] > 0]
    if not positive:
        return {"score": 0.0, "modes": (np.nan, np.nan), "valley_depth": np.nan,
                "min_weight": np.nan}
    representative = min(positive, key=lambda item: abs(item["score"] - score))
    return {"score": score, **{key: representative[key] for key in
            ("modes", "valley_depth", "min_weight")}}


def simulate_null_thresholds(seed=20260824):
    rng = np.random.default_rng(seed)
    rows = []
    thresholds = []
    for sample_size in NULL_SAMPLE_SIZES:
        scores = np.empty(N_NULL_REPLICATES)
        for replicate in range(N_NULL_REPLICATES):
            scores[replicate] = multiband_score(rng.normal(size=sample_size))["score"]
        threshold = float(np.quantile(scores, 0.95))
        thresholds.append(threshold)
        rows.append({
            "sample_size": int(sample_size),
            "alpha": 0.05,
            "critical_score": threshold,
            "median_null_score": float(np.median(scores)),
            "n_replicates": N_NULL_REPLICATES,
        })
        print(f"Null n={sample_size}: critical score={threshold:.4f}", flush=True)
    null_table = pd.DataFrame(rows)
    null_table.to_csv(OUTPUT_DIR / "S4_peak_valley_null_thresholds.csv", index=False)
    return np.asarray(thresholds)


def longest_connected_run(points, y_range):
    jump_limit = 0.20 * max(float(y_range), 1e-12)
    runs, current = [], []
    previous_modes = None
    for point in points:
        modes = np.asarray(point["modes"], dtype=float)
        usable = bool(point["flag"]) and np.all(np.isfinite(modes))
        connected = bool(
            usable and previous_modes is not None
            and np.max(np.abs(modes - previous_modes)) <= jump_limit
        )
        if usable:
            if current and not connected:
                runs.append(current)
                current = []
            current.append(point)
            previous_modes = modes
        else:
            if current:
                runs.append(current)
                current = []
            previous_modes = None
    if current:
        runs.append(current)
    return max(runs, key=len, default=[])


def classify_panel(x, y, null_thresholds):
    x, y = stratified_cap(x, y)
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    y_center, y_scale = _robust_location_scale(y)
    y_scaled = (y - y_center) / y_scale
    points = []
    for start, stop in _window_slices(len(x), window_frac=0.06):
        estimate = multiband_score(y_scaled[start:stop])
        critical = float(np.interp(
            stop - start, NULL_SAMPLE_SIZES, null_thresholds,
            left=null_thresholds[0], right=null_thresholds[-1],
        ))
        modes = tuple(value * y_scale + y_center for value in estimate["modes"])
        points.append({
            "x_min": float(x[start]),
            "x_mid": float(np.median(x[start:stop])),
            "x_max": float(x[stop - 1]),
            "flag": bool(estimate["score"] > critical),
            "score": estimate["score"],
            "critical_score": critical,
            "modes": modes,
            "valley_depth": estimate["valley_depth"],
            "min_weight": estimate["min_weight"],
        })
    y_lo, y_hi = np.quantile(y, [0.005, 0.995])
    run = longest_connected_run(points, y_hi - y_lo)
    run_length = len(run)
    if run:
        full_range = max(float(np.max(x) - np.min(x)), 1e-12)
        coverage = float((run[-1]["x_max"] - run[0]["x_min"]) / full_range)
        separation = np.asarray([item["modes"][1] - item["modes"][0] for item in run])
        edge = min(2, run_length)
        relative_change = float(
            abs(np.mean(separation[-edge:]) - np.mean(separation[:edge]))
            / max(float(np.max(separation)), 1e-12)
        )
        median_score = float(np.median([item["score"] for item in run]))
        median_critical = float(np.median([item["critical_score"] for item in run]))
        x_start, x_end = run[0]["x_min"], run[-1]["x_max"]
    else:
        coverage = relative_change = median_score = median_critical = 0.0
        x_start = x_end = np.nan

    if run_length >= 3 and relative_change >= 0.25 and coverage >= 0.10:
        status, branch_type = "Branch", "extended fork"
    elif run_length == 2:
        # The local two-mode decision has already passed a Monte Carlo test;
        # no extra mass, valley, or separation gate is applied here.
        status, branch_type = "Branch", "local side branch"
    elif run_length >= 3:
        status, branch_type = "Two-band", "stable/ambiguous two-band"
    elif run_length == 1:
        status, branch_type = "Candidate", "one significant local window"
    else:
        status, branch_type = "No branch", "none"
    return {
        "status": status,
        "branch_type": branch_type,
        "n_points": len(x),
        "n_windows": len(points),
        "n_significant_windows": int(sum(point["flag"] for point in points)),
        "run_length": run_length,
        "x_start": x_start,
        "x_end": x_end,
        "x_coverage": coverage,
        "relative_separation_change": relative_change,
        "median_peak_valley_score": median_score,
        "median_critical_score": median_critical,
    }


def plot_heatmap(results):
    codes = {"No branch": 0, "Candidate": 1, "Two-band": 2, "Branch": 3}
    letters = {"No branch": "–", "Candidate": "C", "Two-band": "T", "Branch": "B"}
    cmap = ListedColormap(["#F3F4F6", "#F59E0B", "#7C3AED", "#2563EB"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.7), constrained_layout=True)
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
            ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    status = statuses.iloc[i, j]
                    ax.text(j, i, letters[status], ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if codes[status] >= 2 else "#374151")
            counts = selected["status"].value_counts()
            ax.set_title(
                f"{pair} · {version.upper()} · B={counts.get('Branch', 0)}, "
                f"T={counts.get('Two-band', 0)}, C={counts.get('Candidate', 0)}",
                fontsize=10.5, fontweight="bold",
            )
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.tick_params(labelsize=9)
            ax.set_xticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle(
        "Single-score multi-bandwidth KDE peak-valley persistence",
        fontsize=15.5, fontweight="bold",
    )
    path = FIGURE_DIR / "S4_kde_peak_valley_score_heatmap.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    null_thresholds = simulate_null_thresholds()
    cases = build_cases(load_runs())
    rows = []
    for index, (key, (x, y)) in enumerate(cases.items(), start=1):
        model, domain, pair, version = key
        rows.append({
            "model": model, "domain": domain, "pair": pair, "version": version,
            **classify_panel(x, y, null_thresholds),
        })
        if index % 15 == 0:
            print(f"Processed {index}/{len(cases)}", flush=True)
    results = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "S4_kde_peak_valley_score_results.csv"
    results.to_csv(csv_path, index=False)
    figure_path = plot_heatmap(results)
    print("\nCounts")
    print(results.groupby(["pair", "version", "status"]).size().to_string())
    print("\nFocus panels")
    print(results.loc[
        (
            (results["pair"] == "mrros → Q") & (results["model"] == "CanESM5")
            & (results["domain"] == "CW")
        )
        | (
            (results["pair"] == "P → Q") & (results["domain"] == "LI")
            & results["model"].isin(["CESM2", "GFDL-CM4", "CMCC-CM2-SR5"])
        ),
        ["model", "domain", "pair", "version", "status", "branch_type",
         "run_length", "x_coverage", "median_peak_valley_score",
         "median_critical_score"],
    ].to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
