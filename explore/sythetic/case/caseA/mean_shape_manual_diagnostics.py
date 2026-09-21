"""Compare CMIP6 LOWESS shape predictions with the 5x6 manual label tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["CESM2", "CNRM-CM6-1", "CanESM5", "GFDL-CM4", "CMCC-CM2-SR5"]
ZONES = ["all_land", "WW", "WD", "CW", "CD", "LI"]


def load_manual_mean_shapes(workbook: str | Path) -> pd.DataFrame:
    """Read every sheet's ``LOWESS mean shape`` 5x6 table by its label.

    The header is located by text rather than a fixed row so future notes above
    the table do not shift the extracted labels.
    """
    workbook = Path(workbook)
    xls = pd.ExcelFile(workbook)
    rows: list[dict] = []
    for sheet in xls.sheet_names:
        if sheet == "Summary":
            continue
        raw = pd.read_excel(workbook, sheet_name=sheet, header=None)
        hit = [
            int(i) for i, row in raw.iterrows()
            if row.astype(str).str.contains(
                "LOWESS mean shape", case=False, regex=False
            ).any()
        ]
        if not hit:
            continue
        title = str(raw.iat[0, 0]).split("manual labels")[0].strip()
        start = hit[0] + 2
        for i, model in enumerate(MODELS):
            for j, zone in enumerate(ZONES):
                value = raw.iat[start + i, 1 + j]
                rows.append({
                    "Sheet": sheet,
                    "Pair": title,
                    "Model": model,
                    "Zone": zone,
                    "manual_shape": str(value).strip() if pd.notna(value) else "",
                })
    return pd.DataFrame(rows)


def _manual_class(value: object) -> str:
    text = str(value).strip().lower()
    if "non-monotonic" in text or "nonmonotonic" in text:
        return "Non-monotonic"
    if "monotonic" in text:
        return "Monotonic"
    if text.startswith("flat"):
        return "Flat"
    return "Unresolved"


def _prediction_class(value: object) -> str:
    text = str(value).strip().lower()
    if text == "nonmonotonic":
        return "Non-monotonic"
    if text in {"monotonic_up", "monotonic_down"}:
        return "Monotonic"
    if text in {"candidate_monotonic_up", "candidate_monotonic_down"}:
        return "Candidate"
    if text == "flat":
        return "Flat"
    return "Unresolved"


def _summarize(group: pd.DataFrame, label: str) -> dict:
    scored = group[group["manual_class"].isin(
        ["Monotonic", "Non-monotonic"]
    )].copy()
    resolved = scored["prediction_class"].isin(
        ["Monotonic", "Non-monotonic"]
    )
    correct = scored["prediction_class"].eq(scored["manual_class"])
    mono = scored["manual_class"].eq("Monotonic")
    nonmono = scored["manual_class"].eq("Non-monotonic")
    mono_recall = float(correct[mono].mean()) if mono.any() else np.nan
    nonmono_recall = float(correct[nonmono].mean()) if nonmono.any() else np.nan
    recalls = [v for v in (mono_recall, nonmono_recall) if np.isfinite(v)]
    return {
        "Group": label,
        "Manual panels": int(len(group)),
        "R2 >= 0.35": int(group["r2_eligible"].sum()),
        "Scored shape panels": int(len(scored)),
        "Confirmed monotonic": int(scored["prediction_class"].eq("Monotonic").sum()),
        "Non-monotonic": int(scored["prediction_class"].eq("Non-monotonic").sum()),
        "Candidate": int(scored["prediction_class"].eq("Candidate").sum()),
        "Unresolved": int(scored["prediction_class"].eq("Unresolved").sum()),
        "Coverage": float(resolved.mean()) if len(scored) else np.nan,
        "Strict accuracy": float(correct.mean()) if len(scored) else np.nan,
        "Resolved accuracy": (
            float(correct[resolved].mean()) if resolved.any() else np.nan
        ),
        "Monotonic recall": mono_recall,
        "Non-monotonic recall": nonmono_recall,
        "Balanced accuracy": float(np.mean(recalls)) if recalls else np.nan,
    }


def build_manual_shape_diagnostics(
    features: pd.DataFrame,
    manual_workbook: str | Path,
    *,
    r2_weak: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return overall, pair-level, mismatches, and joined panel diagnostics."""
    labels = load_manual_mean_shapes(manual_workbook)
    use_cols = [
        "Pair", "Model", "Zone", "r2", "lowess_shape", "lowess_shape2",
        "ls_R_rev", "ls_pearson", "ls_abs_pearson", "ls_n_tp",
    ]
    joined = labels.merge(features[use_cols], on=["Pair", "Model", "Zone"], how="left")
    joined["manual_class"] = joined["manual_shape"].map(_manual_class)
    joined["prediction_class"] = joined["lowess_shape"].map(_prediction_class)
    joined["r2_eligible"] = pd.to_numeric(joined["r2"], errors="coerce").ge(r2_weak)

    # The agreed shape rule is applied only after the R2 evidence gate.
    eligible = joined[joined["r2_eligible"]].copy()
    overall = pd.DataFrame([_summarize(eligible, "Overall")])
    by_pair = pd.DataFrame([
        _summarize(group, pair)
        for pair, group in eligible.groupby("Pair", sort=False)
    ])
    mismatches = eligible[
        eligible["manual_class"].isin(["Monotonic", "Non-monotonic"])
        & eligible["prediction_class"].ne(eligible["manual_class"])
    ].copy()
    return overall, by_pair, mismatches, joined


def save_manual_shape_diagnostics(
    features: pd.DataFrame,
    manual_workbook: str | Path,
    output_dir: str | Path,
    *,
    r2_weak: float = 0.35,
):
    """Build and save the three reader-facing CMIP6 diagnostic tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall, by_pair, mismatches, joined = build_manual_shape_diagnostics(
        features, manual_workbook, r2_weak=r2_weak
    )
    overall.to_csv(output_dir / "manual_shape_overall_diagnostic.csv", index=False)
    by_pair.to_csv(output_dir / "manual_shape_by_pair_diagnostic.csv", index=False)
    mismatches.to_csv(output_dir / "manual_shape_mismatches.csv", index=False)
    joined.to_csv(output_dir / "manual_shape_panel_diagnostic.csv", index=False)
    return overall, by_pair, mismatches, joined
