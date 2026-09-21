"""Reusable LOWESS mean-response diagnostics for F_S3.3.

The expensive LOWESS curves are cached independently from the cheap shape
classification so threshold changes in ``case/caseA/lowess_classifier.py`` do
not require refitting the curves.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _primary_shape(value: object) -> str:
    return {
        "flat": "Flat",
        "monotonic_up": "Monotonic up",
        "monotonic_down": "Monotonic down",
        "nonmonotonic": "Non-monotonic",
    }.get(str(value), "Unresolved")


def _n_pct(series: pd.Series, value: str) -> str:
    n = int((series == value).sum())
    return f"{n} ({n / max(len(series), 1):.1%})"


def _fit_case(case_a_dir: str, case_id: object, point_index: int,
              x_all: np.ndarray, y_all: np.ndarray) -> dict:
    if case_a_dir not in sys.path:
        sys.path.insert(0, case_a_dir)
    import lowess_classifier as lowess_cls
    x = np.asarray(x_all[int(point_index)], dtype=float)
    y = np.asarray(y_all[int(point_index)], dtype=float)
    fit = lowess_cls.fit_lowess(x, y)
    if fit is None:
        xs, ys = np.array([], dtype=float), np.array([], dtype=float)
    else:
        xs, ys = fit
    return {
        "case_id": str(case_id),
        "x_lowess": np.asarray(xs, dtype=float),
        "y_lowess": np.asarray(ys, dtype=float),
        "lowess_frac": float(lowess_cls.LOWESS_FRAC),
        "lowess_it": int(lowess_cls.LOWESS_IT),
    }


def _family_table(lowess_eval: pd.DataFrame, family_order: list[str],
                  family_names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for family_id in family_order:
        group = lowess_eval[lowess_eval["family_id"] == family_id]
        if group.empty:
            continue
        rows.append({
            "Family": family_id,
            "Name": family_names[family_id],
            "N": len(group),
            "Median R2": float(group["r2"].median()),
            "Weak": _n_pct(group["strength"], "Weak"),
            "Uncertain": _n_pct(group["strength"], "Uncertain"),
            "Detectable": _n_pct(group["strength"], "Detectable"),
            "Flat": _n_pct(group["primary_shape"], "Flat"),
            "Mono up": _n_pct(group["primary_shape"], "Monotonic up"),
            "Mono down": _n_pct(group["primary_shape"], "Monotonic down"),
            "Non-monotonic": _n_pct(
                group["primary_shape"], "Non-monotonic"),
            "Unresolved shape": _n_pct(
                group["primary_shape"], "Unresolved"),
        })
    return pd.DataFrame(rows)


_SUBSHAPE_COLS = [
    "linear", "saturation", "acceleration",
    "U_shaped", "inverted_U", "J_shaped", "inverted_J",
    "S_or_N", "complex",
    "candidate_U_shaped", "candidate_inverted_U",
    "flat",
]

MONO_SUBSHAPE_COLS = ["linear", "saturation", "acceleration"]
NONMONO_SUBSHAPE_COLS = [
    "U_shaped", "inverted_U", "J_shaped", "inverted_J",
    "S_or_N", "complex",
    "candidate_U_shaped", "candidate_inverted_U",
]


def _pipeline_table(lowess_eval: pd.DataFrame, r2_weak: float,
                    family_order: list[str],
                    family_names: dict[str, str]) -> pd.DataFrame:
    """Single table showing N → Weak/R²≥threshold → shape → sub-shape pipeline."""

    def _fmt(n: int, d: int) -> str:
        if d == 0:
            return ""
        return f"{n} ({n / d:.1%})"

    r2_col = f"R²≥{r2_weak}"
    rows = []
    for family_id in family_order:
        group = lowess_eval[lowess_eval["family_id"] == family_id]
        if group.empty:
            continue
        n_total = len(group)
        r2 = pd.to_numeric(group["r2"], errors="coerce")
        n_weak = int(r2.lt(r2_weak).sum())
        strong = group[r2.ge(r2_weak)]
        n_strong = len(strong)

        n_flat = int((strong["lowess_shape"] == "flat").sum())
        n_mono_up = int((strong["lowess_shape"] == "monotonic_up").sum())
        n_mono_down = int((strong["lowess_shape"] == "monotonic_down").sum())
        n_nonmono = int((strong["lowess_shape"] == "nonmonotonic").sum())
        n_mono = n_mono_up + n_mono_down

        mono_data = strong[
            strong["lowess_shape"].isin(["monotonic_up", "monotonic_down"])
        ]
        nonmono_data = strong[strong["lowess_shape"] == "nonmonotonic"]

        row = {
            "Family": family_id,
            "Name": family_names[family_id],
            "Weak": f"{n_weak} ({n_weak / n_total:.1%})",
            r2_col: f"{n_strong} ({n_strong / n_total:.1%})",
            "Flat": _fmt(n_flat, n_strong),
            "Mono up": _fmt(n_mono_up, n_strong),
            "Mono down": _fmt(n_mono_down, n_strong),
        }
        for label in MONO_SUBSHAPE_COLS:
            n = int((mono_data["lowess_shape2"] == label).sum()) if n_mono else 0
            row[label] = _fmt(n, n_mono)

        row["Non-monotonic"] = _fmt(n_nonmono, n_strong)

        if n_nonmono:
            tp = pd.to_numeric(nonmono_data["ls_n_tp"], errors="coerce")
            n_tp0 = int((tp == 0).sum())
        else:
            n_tp0 = 0
        row["tp=0"] = _fmt(n_tp0, n_nonmono)

        for label in NONMONO_SUBSHAPE_COLS:
            n = int((nonmono_data["lowess_shape2"] == label).sum()) if n_nonmono else 0
            row[label] = _fmt(n, n_nonmono)

        rows.append(row)

    df = pd.DataFrame(rows)
    drop_candidates = (
        ["Flat", "Mono down"]
        + MONO_SUBSHAPE_COLS
        + ["tp=0"]
        + NONMONO_SUBSHAPE_COLS
    )
    empty_cols = [c for c in drop_candidates if c in df.columns
                  and df[c].isin(["", "0 (0.0%)"]).all()]
    return df.drop(columns=empty_cols)


def _fork_table(lowess_eval: pd.DataFrame, fork_families: list[str],
                family_names: dict[str, str]) -> pd.DataFrame:
    visible = lowess_eval[
        lowess_eval["family_id"].isin(fork_families)
        & (lowess_eval["expected_branch"] == "Branch")
    ]
    rows = []
    for family_id in fork_families:
        group = visible[visible["family_id"] == family_id]
        mono = group["primary_shape"].isin(
            ["Monotonic up", "Monotonic down"])
        rows.append({
            "Family": family_id,
            "Name": family_names[family_id],
            "N": len(group),
            "Median R2": float(group["r2"].median()),
            "Weak": _n_pct(group["strength"], "Weak"),
            "Uncertain": _n_pct(group["strength"], "Uncertain"),
            "Detectable": _n_pct(group["strength"], "Detectable"),
            "Flat": _n_pct(group["primary_shape"], "Flat"),
            "Monotonic": f"{int(mono.sum())} ({mono.mean():.1%})",
            "Non-monotonic": _n_pct(
                group["primary_shape"], "Non-monotonic"),
        })
    return pd.DataFrame(rows)


def _metric_range_table(lowess_eval: pd.DataFrame, family_order: list[str],
                        family_names: dict[str, str]) -> pd.DataFrame:
    """Return robust and full ranges for continuous LOWESS diagnostics."""
    metrics = [
        ("R2", "r2"),
        ("Pearson rho(x, LOWESS)", "ls_pearson"),
        ("Abs Pearson", "ls_abs_pearson"),
        ("Flat-segment fraction", "ls_flat_fraction"),
        ("Flat amplitude", "ls_flat_amplitude"),
        ("Positive-segment fraction", "ls_p_pos"),
        ("Negative-segment fraction", "ls_p_neg"),
        ("Reversal ratio Rrev", "ls_R_rev"),
        ("Bow ratio", "ls_bow"),
        ("Turning points", "ls_n_tp"),
    ]
    rows = []
    for family_id in family_order:
        group = lowess_eval[lowess_eval["family_id"] == family_id]
        if group.empty:
            continue
        for metric, column in metrics:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                stats = {
                    "Min": np.nan, "P05": np.nan, "Median": np.nan,
                    "P95": np.nan, "Max": np.nan,
                }
            else:
                stats = {
                    "Min": float(values.min()),
                    "P05": float(values.quantile(0.05)),
                    "Median": float(values.median()),
                    "P95": float(values.quantile(0.95)),
                    "Max": float(values.max()),
                }
            rows.append({
                "Family": family_id,
                "Name": family_names[family_id],
                "Family N": int(len(group)),
                "Metric": metric,
                "Valid N": int(len(values)),
                **stats,
            })
    return pd.DataFrame(rows)


_CLEAN_SHAPE_TRUTH = {
    "F01": "Monotonic", "F03": "Monotonic", "F05": "Monotonic",
    "F07": "Monotonic", "F13": "Monotonic", "F15": "Monotonic",
    "F21": "Monotonic",
    "F18": "Non-monotonic", "F19": "Non-monotonic",
    "F22": "Non-monotonic",
}


def _predicted_shape_class(primary_shape: pd.Series) -> pd.Series:
    out = pd.Series("Unresolved", index=primary_shape.index, dtype=object)
    out.loc[primary_shape.isin(["Monotonic up", "Monotonic down"])] = "Monotonic"
    out.loc[primary_shape.eq("Non-monotonic")] = "Non-monotonic"
    return out


def _shape_accuracy_tables(lowess_eval: pd.DataFrame, r2_weak: float,
                           family_names: dict[str, str],
                           family_order: list[str] | None = None):
    """Evaluate shape accuracy above the R2 evidence gate.

    When *family_order* is given the per-family table includes **every**
    family in that list (families without ground truth get NaN accuracy).
    The overall row always uses only scored families.
    """
    r2_ok = pd.to_numeric(lowess_eval["r2"], errors="coerce").ge(r2_weak)

    scored = lowess_eval[
        lowess_eval["family_id"].isin(_CLEAN_SHAPE_TRUTH) & r2_ok
    ].copy()
    scored["Truth"] = scored["family_id"].map(_CLEAN_SHAPE_TRUTH)
    scored["Prediction"] = _predicted_shape_class(scored["primary_shape"])

    def one_row(g: pd.DataFrame, family: str, name: str) -> dict:
        resolved = g["Prediction"].isin(["Monotonic", "Non-monotonic"])
        has_truth = g["Truth"].notna()
        gt = g[has_truth]
        if len(gt):
            correct = gt["Prediction"].eq(gt["Truth"])
            mono = gt["Truth"].eq("Monotonic")
            nonmono = gt["Truth"].eq("Non-monotonic")
            mono_recall = float(correct[mono].mean()) if mono.any() else np.nan
            nonmono_recall = float(correct[nonmono].mean()) if nonmono.any() else np.nan
            recalls = [v for v in (mono_recall, nonmono_recall) if np.isfinite(v)]
            strict_acc = float(correct.mean())
            resolved_acc = (
                float(correct[gt["Prediction"].isin(["Monotonic", "Non-monotonic"])].mean())
                if gt["Prediction"].isin(["Monotonic", "Non-monotonic"]).any()
                else np.nan
            )
            balanced_acc = float(np.mean(recalls)) if recalls else np.nan
        else:
            strict_acc = np.nan
            resolved_acc = np.nan
            mono_recall = np.nan
            nonmono_recall = np.nan
            balanced_acc = np.nan
        return {
            "Family": family,
            "Name": name,
            "N (R2 eligible)": int(len(g)),
            "Monotonic": int(g["Prediction"].eq("Monotonic").sum()),
            "Non-monotonic": int(g["Prediction"].eq("Non-monotonic").sum()),
            "Unresolved": int(g["Prediction"].eq("Unresolved").sum()),
            "Coverage": float(resolved.mean()) if len(g) else np.nan,
            "Strict accuracy": strict_acc,
            "Resolved accuracy": resolved_acc,
            "Monotonic recall": mono_recall,
            "Non-monotonic recall": nonmono_recall,
            "Balanced accuracy": balanced_acc,
        }

    overall = pd.DataFrame([one_row(scored, "Overall", "Clean shape families")])

    if family_order is not None:
        all_eligible = lowess_eval[r2_ok].copy()
        all_eligible["Truth"] = all_eligible["family_id"].map(_CLEAN_SHAPE_TRUTH)
        all_eligible["Prediction"] = _predicted_shape_class(
            all_eligible["primary_shape"])
        rows = []
        for fid in family_order:
            g = all_eligible[all_eligible["family_id"] == fid]
            if g.empty:
                continue
            rows.append(one_row(g, fid, family_names.get(fid, fid)))
        by_family = pd.DataFrame(rows)
    else:
        by_family = pd.DataFrame([
            one_row(g, family_id, family_names.get(family_id, family_id))
            for family_id, g in scored.groupby("family_id", sort=True)
        ])
    return overall, by_family


def run_lowess_diagnostics(
    data: pd.DataFrame,
    x_all: np.ndarray,
    y_all: np.ndarray,
    *,
    case_a_dir: str | Path,
    output_dir: str | Path,
    mic_min: float,
    family_order: list[str],
    family_names: dict[str, str],
    fork_families: list[str],
    refit_lowess: bool = False,
    n_jobs: int = -1,
    pearson_thresh: float | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit/reuse LOWESS, rerun current classification, and build diagnostics."""
    case_a_dir = Path(case_a_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(case_a_dir) not in sys.path:
        sys.path.insert(0, str(case_a_dir))
    import lowess_classifier as lowess_cls
    importlib.reload(lowess_cls)
    if pearson_thresh is not None:
        lowess_cls.PEARSON_MONO_THRESH = pearson_thresh

    curve_path = output_dir / "lowess_mean_response_curves.parquet"
    feature_path = output_dir / "lowess_mean_response_features.csv"
    lowess_input = data[
        pd.to_numeric(data["MIC"], errors="coerce").ge(mic_min)
    ].copy()

    if curve_path.exists() and not refit_lowess:
        curve_cache = pd.read_parquet(curve_path)
        cached_ids = set(curve_cache["case_id"].astype(str))
        expected_ids = set(lowess_input["case_id"].astype(str))
        same_fit = (
            set(curve_cache["lowess_frac"].dropna().astype(float))
            == {float(lowess_cls.LOWESS_FRAC)}
            and set(curve_cache["lowess_it"].dropna().astype(int))
            == {int(lowess_cls.LOWESS_IT)}
        )
        if cached_ids != expected_ids or not same_fit:
            raise RuntimeError(
                "LOWESS cache differs from current cases/fit settings; "
                "set REFIT_LOWESS=True once."
            )
        print(f"Loaded LOWESS cache: {curve_path} ({len(curve_cache):,} cases)")
    else:
        print(f"Fitting LOWESS for {len(lowess_input):,} MIC-eligible cases ...")
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_case)(
                str(case_a_dir), row.case_id, row.point_index, x_all, y_all)
            for row in lowess_input.itertuples(index=False)
        )
        curve_cache = pd.DataFrame(rows)
        curve_cache.to_parquet(curve_path, index=False, compression="zstd")
        print(f"Saved LOWESS cache: {curve_path} ({len(curve_cache):,} cases)")

    curve_lookup = {
        str(row.case_id): (
            np.asarray(row.x_lowess, dtype=float),
            np.asarray(row.y_lowess, dtype=float),
        )
        for row in curve_cache.itertuples(index=False)
    }

    feature_rows = []
    for row in lowess_input.itertuples(index=False):
        x = np.asarray(x_all[int(row.point_index)], dtype=float)
        y = np.asarray(y_all[int(row.point_index)], dtype=float)
        xs, ys = curve_lookup[str(row.case_id)]
        _pt = pearson_thresh if pearson_thresh is not None else lowess_cls.PEARSON_MONO_THRESH
        if len(xs) == 0:
            result = lowess_cls.classify_panel(
                np.array([]), np.array([]), np.array([]), np.array([]),
                pearson_thresh=_pt)
        else:
            result = lowess_cls.classify_panel(x, y, xs, ys,
                                               pearson_thresh=_pt)
        feature_rows.append({
            "case_id": row.case_id,
            "family_id": row.family_id,
            "name": row.name,
            "nominal_snr": row.nominal_snr,
            "MIC": row.MIC,
            "expected_branch": row.expected,
            "variant_level": getattr(row, "variant_level", None),
            **result,
        })

    lowess_eval = pd.DataFrame(feature_rows)
    lowess_eval["primary_shape"] = lowess_eval["lowess_shape"].map(
        _primary_shape)
    lowess_eval.to_csv(feature_path, index=False)
    print(f"Saved LOWESS features: {feature_path} ({len(lowess_eval):,} cases)")

    tables = {
        "family": _family_table(
            lowess_eval, family_order, family_names),
        "strength_shape": pd.crosstab(
            lowess_eval["strength"], lowess_eval["primary_shape"],
            margins=True),
        "visible_forks": _fork_table(
            lowess_eval, fork_families, family_names),
        "metric_ranges": _metric_range_table(
            lowess_eval, family_order, family_names),
    }
    tables["shape_accuracy"], tables["shape_family_accuracy"] = (
        _shape_accuracy_tables(
            lowess_eval, float(lowess_cls.R2_WEAK), family_names)
    )
    tables["family"].to_csv(
        output_dir / "lowess_family_diagnostic.csv", index=False)
    tables["strength_shape"].to_csv(
        output_dir / "lowess_strength_shape_diagnostic.csv")
    tables["visible_forks"].to_csv(
        output_dir / "lowess_visible_branch_diagnostic.csv", index=False)
    tables["metric_ranges"].to_csv(
        output_dir / "lowess_family_metric_ranges.csv", index=False)
    tables["shape_accuracy"].to_csv(
        output_dir / "lowess_shape_accuracy.csv", index=False)
    tables["shape_family_accuracy"].to_csv(
        output_dir / "lowess_shape_family_accuracy.csv", index=False)
    return lowess_eval, tables
