"""Shared plotting helpers for the two S4 KDE branch detectors."""

from __future__ import annotations

import importlib
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess

import test_two_condition_kde as scatter_kde
import test_2d_kde_conditional_tracks as kde2d


scatter_kde = importlib.reload(scatter_kde)
kde2d = importlib.reload(kde2d)

MODELS = ["CESM2", "CNRM-CM6-1", "CanESM5", "GFDL-CM4", "CMCC-CM2-SR5"]
DOMAINS = ["all_land", "WW", "WD", "CW", "CD", "LI"]
DOMAIN_LABELS = {
    "all_land": "All land", "WW": "Wet–warm", "WD": "Dry–warm",
    "CW": "Wet–cold", "CD": "Dry–cold", "LI": "Land ice",
}
STATUS_FACECOLORS = {
    "Branch": "#DDF3E4",
    "Two-band": "#E9DDF8",
    "Candidate": "#FFF2BF",
    "No branch": "#F7F8FA",
    "No global": "#FFFFFF",
}
TRACK_COLORS = {"low": "#2563EB", "valley": "#111827", "high": "#DC2626"}
KDE_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "transparent_grey_density",
    [
        (1.00, 1.00, 1.00, 0.00),
        (0.82, 0.82, 0.82, 0.35),
        (0.45, 0.45, 0.45, 0.72),
        (0.12, 0.12, 0.12, 0.90),
    ],
)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Return panel counts by pair, version, and status."""
    return (
        results.groupby(["pair", "version", "status"]).size()
        .rename("n_panels").reset_index()
        .pivot_table(
            index=["pair", "version"], columns="status",
            values="n_panels", fill_value=0,
        )
        .astype(int)
    )


# --- Post-hoc association filter (superseded by MIC pre-filter in run_detector) ---
# def apply_association_filter(
#     results: pd.DataFrame,
#     objects: dict,
#     metrics: pd.DataFrame,
#     *,
#     mode: str = "none",
#     mic_min: float = 0.30,
#     dcor_min: float = 0.40,
#     candidate_only: bool = True,
# ) -> pd.DataFrame:
#     """Attach MIC/dCor and optionally reject weak-association candidates.
#
#     ``mode`` may be ``none``, ``mic``, ``dcor``, or ``both``.  By default the
#     association gate acts only on Candidate, leaving geometric Branch and
#     Two-band detections unchanged.
#     """
#     allowed = {"none", "mic", "dcor", "both"}
#     if mode not in allowed:
#         raise ValueError(f"mode must be one of {sorted(allowed)}, got {mode!r}")
#     keys = ["model", "domain", "pair", "version"]
#     required = set(keys + ["mic", "dcor"])
#     missing = required - set(metrics.columns)
#     if missing:
#         raise KeyError(f"association metrics missing columns: {sorted(missing)}")
#     metric_table = metrics[keys + ["mic", "dcor"]].drop_duplicates(keys)
#     if metric_table.duplicated(keys).any():
#         raise ValueError("association metrics are not unique by panel")
#     merged = results.merge(metric_table, on=keys, how="left", validate="one_to_one")
#     merged["geometric_status"] = merged["status"]
#
#     mic_pass = merged["mic"].ge(mic_min)
#     dcor_pass = merged["dcor"].ge(dcor_min)
#     if mode == "mic":
#         association_pass = mic_pass
#     elif mode == "dcor":
#         association_pass = dcor_pass
#     elif mode == "both":
#         association_pass = mic_pass & dcor_pass
#     else:
#         association_pass = pd.Series(True, index=merged.index)
#
#     eligible = (
#         merged["geometric_status"].eq("Candidate")
#         if candidate_only
#         else merged["geometric_status"].ne("No branch")
#     )
#     rejected = eligible & ~association_pass
#     merged.loc[rejected, "status"] = "No branch"
#     merged["association_filter"] = mode
#     merged["association_pass"] = association_pass
#     merged["association_rejected"] = rejected
#     merged["association_filter_scope"] = (
#         "candidate_only" if candidate_only else "all_detections"
#     )
#     merged["mic_min"] = float(mic_min)
#     merged["dcor_min"] = float(dcor_min)
#
#     for row in merged.itertuples(index=False):
#         key = (row.model, row.domain, row.pair, row.version)
#         objects[key]["geometric_status"] = row.geometric_status
#         objects[key]["status"] = row.status
#         objects[key]["mic"] = row.mic
#         objects[key]["dcor"] = row.dcor
#         objects[key]["association_pass"] = bool(row.association_pass)
#         objects[key]["association_rejected"] = bool(row.association_rejected)
#     return merged


def _association_pass(mic, dcor, mode, mic_min, dcor_min):
    """Return whether a panel passes the MIC/dCor gate."""
    mic_ok = bool(np.isfinite(mic) and mic >= mic_min)
    dcor_ok = bool(np.isfinite(dcor) and dcor >= dcor_min)
    if mode == "none":
        return True, mic_ok, dcor_ok
    if mode == "mic":
        return mic_ok, mic_ok, dcor_ok
    if mode == "dcor":
        return dcor_ok, mic_ok, dcor_ok
    if mode == "both":
        return mic_ok and dcor_ok, mic_ok, dcor_ok
    if mode == "either":
        return mic_ok or dcor_ok, mic_ok, dcor_ok
    raise ValueError(
        f"mode must be one of ['none', 'mic', 'dcor', 'both', 'either'], got {mode!r}"
    )


def _empty_geometry(method):
    if method == "scatter":
        return {"points": [], "runs": [], "run": []}
    return None


def _run_one_panel(args):
    """Worker function for parallel panel detection."""
    method, x_values, y_values, coverage_mode = args
    if method == "scatter":
        return scatter_kde.classify_panel(
            x_values, y_values, return_geometry=True, coverage_mode=coverage_mode,
        )
    return kde2d.classify_panel(
        x_values, y_values, return_geometry=True, coverage_mode=coverage_mode,
    )


def run_detector(
    cases,
    method,
    *,
    metrics=None,
    mode="none",
    mic_min=0.40,
    dcor_min=0.40,
    mic_metrics=None,
    coverage_mode="x_range",
    detect_behind_gate=False,
    n_jobs=None,
):
    """Run one detector, optionally in parallel.

    *n_jobs*: number of worker processes (default: cpu_count - 1, min 1).
    Set ``n_jobs=1`` to disable parallelism.
    """
    import multiprocessing as mp

    if method not in {"scatter", "kde2d"}:
        raise ValueError("method must be 'scatter' or 'kde2d'")
    if mic_metrics is not None and metrics is None:
        metrics = mic_metrics

    mic_lookup = {}
    dcor_lookup = {}
    if metrics is not None:
        for row in metrics.itertuples(index=False):
            key_ = (row.model, row.domain, row.pair, row.version)
            mic_lookup[key_] = row.mic
            if hasattr(row, "dcor"):
                dcor_lookup[key_] = row.dcor

    to_run = []
    skipped = []
    started = time.perf_counter()

    for key, (x_values, y_values) in cases.items():
        model, domain, pair, version = key
        panel_mic = float(mic_lookup[key]) if key in mic_lookup else np.nan
        panel_dcor = float(dcor_lookup[key]) if key in dcor_lookup else np.nan
        passed, mic_ok, dcor_ok = _association_pass(
            panel_mic, panel_dcor, mode, mic_min, dcor_min,
        )
        meta = {
            "mic": panel_mic, "dcor": panel_dcor, "gate_mode": mode,
            "mic_pass": mic_ok, "dcor_pass": dcor_ok,
            "mic_min": float(mic_min), "dcor_min": float(dcor_min),
        }
        if metrics is not None and not passed:
            if detect_behind_gate:
                to_run.append((key, x_values, y_values, meta, False))
            else:
                skipped.append((key, x_values, meta))
        else:
            to_run.append((key, x_values, y_values, meta, True))

    objects, geometries, rows = {}, {}, []

    if to_run:
        worker_args = [
            (method, x, y, coverage_mode) for (_, x, y, _, _) in to_run
        ]
        if n_jobs is None:
            n_jobs = max(1, mp.cpu_count() - 1)
        if n_jobs > 1 and len(worker_args) > 4:
            with mp.Pool(n_jobs) as pool:
                results_list = pool.map(_run_one_panel, worker_args)
        else:
            results_list = [_run_one_panel(a) for a in worker_args]

        for (key, _, _, meta, association_passed), (result, geometry) in zip(to_run, results_list):
            model, domain, pair, version = key
            if not association_passed:
                result["branch_behind_gate"] = result["status"] in ("Branch", "Candidate")
                result["status"] = "No global"
                result["association_pass"] = False
            else:
                result["branch_behind_gate"] = False
                result["association_pass"] = True
            result.update(meta)
            objects[key] = result
            geometries[key] = geometry
            rows.append({
                "model": model, "domain": domain, "pair": pair,
                "version": version, **result,
            })

    for key, x_values, meta in skipped:
        model, domain, pair, version = key
        result = {
            "status": "No global", "branch_type": "none",
            "run_length": 0, "x_coverage": 0.0,
            "coverage_mode": coverage_mode,
            "n_points": len(x_values),
            "branch_behind_gate": False,
            "association_pass": False,
        }
        result.update(meta)
        geometry = _empty_geometry(method)
        objects[key] = result
        geometries[key] = geometry
        rows.append({
            "model": model, "domain": domain, "pair": pair,
            "version": version, **result,
        })

    elapsed = time.perf_counter() - started
    n_skipped = len(skipped)
    if n_skipped:
        print(
            f"  Association gate ({mode}): {n_skipped} panels skipped "
            f"(MIC < {mic_min:.2f} and/or dCor < {dcor_min:.2f})"
        )
    n_parallel = len(to_run)
    if n_parallel:
        print(f"  Detected {n_parallel} panels in {elapsed:.1f}s (n_jobs={n_jobs})")
    return pd.DataFrame(rows), objects, geometries, elapsed


def _finish_panel(ax, model, domain, result, row_index, col_index, pair,
                  scatter_result=None, kde2d_result=None):
    status = result["status"]
    gated = result.get("branch_behind_gate", False)
    if gated:
        ax.set_facecolor("#D1D5DB")
    else:
        ax.set_facecolor(STATUS_FACECOLORS.get(status, "#FFFFFF"))
    if row_index == 0:
        ax.set_title(
            f"{DOMAIN_LABELS[domain]} ({domain})",
            fontsize=11, color="#1F2937", fontweight="bold",
        )
    if col_index == 0:
        ax.set_ylabel(f"{model}\n{pair.split(' → ')[1]}", fontsize=10)
    if row_index == len(MODELS) - 1:
        ax.set_xlabel(pair.split(" → ")[0], fontsize=10)
    mic = result.get("mic", np.nan)
    dcor = result.get("dcor", np.nan)
    mic_text = f"{mic:.2f}" if np.isfinite(mic) else "–"
    dcor_text = f"{dcor:.2f}" if np.isfinite(dcor) else "–"
    mic_flag = "✓" if result.get("mic_pass") else "✗"
    dcor_flag = "✓" if result.get("dcor_pass") else "✗"
    mic_line = f"MIC={mic_text}{mic_flag}  dCor={dcor_text}{dcor_flag}"
    branch_votes = result.get("bandwidth_branch_votes")
    support_votes = result.get("bandwidth_support_votes")
    vote_line = (
        f"\nVotes B={branch_votes}/5 S={support_votes}/5"
        if branch_votes is not None and support_votes is not None else ""
    )
    s_status = scatter_result["status"] if scatter_result is not None else None
    k_status = kde2d_result["status"] if kde2d_result is not None else None
    agree = (s_status is None or k_status is None or s_status == k_status)
    if agree:
        shown = s_status or k_status or status
        ax.text(
            0.025, 0.975, f"{shown}\n{mic_line}{vote_line}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="#1F2937",
            bbox={
                "boxstyle": "round,pad=0.22", "facecolor": "white",
                "edgecolor": "#D1D5DB", "linewidth": 0.45, "alpha": 0.82,
            },
            zorder=10,
        )
    else:
        ax.text(
            0.025, 0.975, f"{status}\n{mic_line}{vote_line}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="#1F2937",
            bbox={
                "boxstyle": "round,pad=0.22", "facecolor": "white",
                "edgecolor": "#D1D5DB", "linewidth": 0.45, "alpha": 0.82,
            },
            zorder=10,
        )
        ax.text(
            0.975, 0.975, f"S: {s_status}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            fontweight="bold", color="#166534", zorder=11,
        )
        ax.text(
            0.975, 0.90, f"K: {k_status}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            fontweight="bold", color="#1E40AF", zorder=11,
        )
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.18, linewidth=0.5)


def _draw_lowess_diagnostic(
    ax, x_values, y_values, *, frac=0.30, it=3,
    turning_threshold=3,
):
    """Overlay LOWESS and return its slope-sign-change count.

    This is deliberately a plotting diagnostic only.  It does not modify the
    branch result or any downstream decision.  The sign-change calculation
    matches the LOWESS notebook: resample the fitted curve on 50 evenly spaced
    x positions, ignore slopes smaller than 10% of the median absolute slope,
    and count changes among the remaining slope signs.
    """
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    xf, yf = x_arr[finite], y_arr[finite]
    if len(xf) < 10 or np.ptp(xf) <= 0:
        return 0

    fitted = _lowess(yf, xf, frac=frac, it=it, return_sorted=True)
    xs, ys = fitted[:, 0], fitted[:, 1]
    finite_fit = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[finite_fit], ys[finite_fit]
    if len(xs) < 10:
        return 0

    # Repeated x values are common in gridded climate data.  Collapse them
    # before interpolation so the diagnostic remains deterministic.
    unique_x, inverse = np.unique(xs, return_inverse=True)
    if len(unique_x) < 10 or np.ptp(unique_x) <= 0:
        return 0
    y_sum = np.bincount(inverse, weights=ys)
    y_count = np.bincount(inverse)
    unique_y = y_sum / np.maximum(y_count, 1)

    ax.plot(
        unique_x, unique_y, color="#B12A68", linewidth=1.35,
        linestyle="-", alpha=0.42, zorder=6,
    )

    x_uniform = np.linspace(unique_x.min(), unique_x.max(), 50)
    y_uniform = np.interp(x_uniform, unique_x, unique_y)
    slope = np.diff(y_uniform) / np.diff(x_uniform)
    slope = slope[np.isfinite(slope)]
    if len(slope) < 2:
        return 0
    slope_threshold = 0.10 * np.median(np.abs(slope))
    signs = np.zeros_like(slope, dtype=int)
    signs[slope > slope_threshold] = 1
    signs[slope < -slope_threshold] = -1
    nonzero = signs[signs != 0]
    n_sign_changes = (
        int(np.count_nonzero(np.diff(nonzero))) if len(nonzero) > 1 else 0
    )

    if n_sign_changes > turning_threshold:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#DC2626")
            spine.set_linewidth(2.2)
            spine.set_zorder(20)
    ax.text(
        0.975, 0.025, f"LOWESS turns={n_sign_changes}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2,
        color=("#DC2626" if n_sign_changes > turning_threshold else "#7A1F4D"),
        bbox={
            "boxstyle": "round,pad=0.18", "facecolor": "white",
            "edgecolor": "none", "alpha": 0.68,
        },
        zorder=21,
    )
    return n_sign_changes


def draw_scatter_grid(
    cases, objects, geometries, pair, version, figure_dir, pair_slug,
    show=True, other_objects=None, *, show_lowess=False,
    lowess_turning_threshold=3,
):
    """Draw one 5-model × 6-domain grid using the original scatter points.

    *other_objects*: result dict from the other method (e.g. kde2d).  When a
    panel's status differs, a blue label shows the other method's status.

    With *show_lowess=True*, a LOWESS curve is overlaid on every panel.  A red
    axes border marks ``n_sign_changes > lowess_turning_threshold``; this is a
    visual diagnostic and does not alter ``result``.
    """
    fig, axes = plt.subplots(5, 6, figsize=(18, 14), constrained_layout=True)
    for row_index, model in enumerate(MODELS):
        for col_index, domain in enumerate(DOMAINS):
            ax = axes[row_index, col_index]
            key = (model, domain, pair, version)
            x_values, y_values = cases[key]
            x_plot, y_plot = scatter_kde.stratified_cap(x_values, y_values)
            result, run = objects[key], geometries[key]["run"]
            ax.scatter(
                x_plot, y_plot, s=5, color="#6B7280", alpha=0.14,
                edgecolors="none", rasterized=True,
            )
            if run:
                xx = [point["x_mid"] for point in run]
                ax.plot(xx, [point["modes"][0] for point in run],
                        color=TRACK_COLORS["low"], lw=1.8, marker="o", ms=2.6)
                ax.plot(xx, [point["valley"] for point in run],
                        color=TRACK_COLORS["valley"], lw=1.2, ls="--")
                ax.plot(xx, [point["modes"][1] for point in run],
                        color=TRACK_COLORS["high"], lw=1.8, marker="o", ms=2.6)
            scatter_res = result
            kde2d_res = other_objects[key] if other_objects is not None and key in other_objects else None
            _finish_panel(ax, model, domain, result, row_index, col_index, pair,
                          scatter_result=scatter_res, kde2d_result=kde2d_res)
            if show_lowess:
                _draw_lowess_diagnostic(
                    ax, x_values, y_values,
                    turning_threshold=lowess_turning_threshold,
                )
    fig.suptitle(
        f"{pair} · {version.upper()} · scatter-window conditional KDE",
        fontsize=16, fontweight="bold",
    )
    path = Path(figure_dir) / f"S4_branch_scatter_kde_{pair_slug}_{version}_5x6.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def draw_kde2d_grid(cases, objects, geometries, pair, version, figure_dir, pair_slug, show=True, other_objects=None):
    """Draw one 5-model × 6-domain grid from conditional 2-D KDE columns.

    *other_objects*: result dict from the other method (e.g. scatter).  When a
    panel's status differs, a blue label shows the other method's status.
    """
    fig, axes = plt.subplots(5, 6, figsize=(18, 14), constrained_layout=True)
    for row_index, model in enumerate(MODELS):
        for col_index, domain in enumerate(DOMAINS):
            ax = axes[row_index, col_index]
            key = (model, domain, pair, version)
            result, geometry = objects[key], geometries[key]
            if geometry is not None and geometry.get("surface") is not None:
                surface = geometry["surface"]
                density = surface["surfaces"][2]
                ax.pcolormesh(
                    surface["x_edges"], surface["y_edges"],
                    np.log1p(1000.0 * density.T), shading="auto",
                    cmap=KDE_DENSITY_CMAP, rasterized=True,
                )
                run = geometry["run"]
                if run:
                    xx = [point["x_mid"] for point in run]
                    ax.plot(xx, [point["low"] for point in run],
                            color=TRACK_COLORS["low"], lw=1.8, marker="o", ms=2.6)
                    ax.plot(xx, [point["valley"] for point in run],
                            color=TRACK_COLORS["valley"], lw=1.2, ls="--")
                    ax.plot(xx, [point["high"] for point in run],
                            color=TRACK_COLORS["high"], lw=1.8, marker="o", ms=2.6)
            else:
                x_values, y_values = cases[key]
                x_plot, y_plot = scatter_kde.stratified_cap(x_values, y_values)
                ax.scatter(
                    x_plot, y_plot, s=5, color="#6B7280", alpha=0.14,
                    edgecolors="none", rasterized=True,
                )
            scatter_res = other_objects[key] if other_objects is not None and key in other_objects else None
            kde2d_res = result
            _finish_panel(ax, model, domain, result, row_index, col_index, pair,
                          scatter_result=scatter_res, kde2d_result=kde2d_res)
    fig.suptitle(
        f"{pair} · {version.upper()} · conditional 2-D KDE",
        fontsize=16, fontweight="bold",
    )
    path = Path(figure_dir) / f"S4_branch_2d_kde_{pair_slug}_{version}_5x6.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def draw_hexbin_grid(
    cases, objects, pair, version, figure_dir, pair_slug,
    show=True, gridsize=26, scatter_objects=None, kde2d_objects=None,
):
    """Draw one 5-model × 6-domain grid as log-count Hexbin density.

    *scatter_objects* / *kde2d_objects*: result dicts from both methods.
    When provided, cross-method comparison labels are shown on each panel.
    """
    fig, axes = plt.subplots(5, 6, figsize=(18, 14), constrained_layout=True)
    for row_index, model in enumerate(MODELS):
        for col_index, domain in enumerate(DOMAINS):
            ax = axes[row_index, col_index]
            key = (model, domain, pair, version)
            x_values, y_values = cases[key]
            result = objects[key]
            finite = np.isfinite(x_values) & np.isfinite(y_values)
            xf, yf = np.asarray(x_values)[finite], np.asarray(y_values)[finite]
            if len(xf) >= 20 and np.ptp(xf) > 0 and np.ptp(yf) > 0:
                ax.hexbin(
                    xf, yf,
                    gridsize=gridsize,
                    cmap="Greys",
                    mincnt=1,
                    norm=LogNorm(),
                    linewidths=0.0,
                    zorder=1,
                )
            else:
                ax.scatter(
                    xf, yf, s=8, color="#6B7280", alpha=0.40,
                    edgecolors="none", rasterized=True,
                )
            s_res = scatter_objects[key] if scatter_objects is not None and key in scatter_objects else None
            k_res = kde2d_objects[key] if kde2d_objects is not None and key in kde2d_objects else None
            _finish_panel(ax, model, domain, result, row_index, col_index, pair,
                          scatter_result=s_res, kde2d_result=k_res)
    fig.suptitle(
        f"{pair} · {version.upper()} · Hexbin density",
        fontsize=16, fontweight="bold",
    )
    path = Path(figure_dir) / f"S4_branch_hexbin_{pair_slug}_{version}_5x6.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def main():
    """Run both methods outside the notebook for reproducible verification."""
    from test_residual_dip import build_cases, load_runs, PAIRS

    case_dir = Path(__file__).resolve().parent
    output_dir = case_dir / "output" / "S4"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    pair_slugs = {label: f"{x_name}_{y_name}" for x_name, y_name, label in PAIRS}
    cases = build_cases(load_runs())

    for method in ("scatter", "kde2d"):
        results, objects, geometries, elapsed = run_detector(cases, method)
        print(f"\n{method}: {len(results)} panels in {elapsed:.1f}s")
        print(summarize(results).to_string())
        csv_name = (
            "S4_branch_scatter_window_kde_all_cases.csv"
            if method == "scatter"
            else "S4_branch_conditional_2d_kde_all_cases.csv"
        )
        results.to_csv(output_dir / csv_name, index=False)
        for _, _, pair in PAIRS:
            for version in ("raw", "trimmed"):
                if method == "scatter":
                    draw_scatter_grid(
                        cases, objects, geometries, pair, version,
                        figure_dir, pair_slugs[pair], show=False,
                    )
                else:
                    draw_kde2d_grid(
                        cases, objects, geometries, pair, version,
                        figure_dir, pair_slugs[pair], show=False,
                    )


if __name__ == "__main__":
    main()
