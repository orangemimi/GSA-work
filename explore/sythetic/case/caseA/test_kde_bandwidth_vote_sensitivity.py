"""Sensitivity of modal-track labels to multi-bandwidth voting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import branch_benchmark
import test_sliding_kde_modal_summary as modal_summary
from test_residual_dip import build_cases, load_runs


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = CASE_DIR / "output" / "S4" / "S4_kde_bandwidth_vote_sensitivity.csv"

CONFIGURATIONS = {
    "3 bandwidths, 2/3": ((0.75, 1.00, 1.25), 2),
    "5 bandwidths, 3/5": ((0.65, 0.80, 1.00, 1.25, 1.50), 3),
    "5 bandwidths, 4/5": ((0.65, 0.80, 1.00, 1.25, 1.50), 4),
}


def main():
    cases = build_cases(load_runs())
    original_detector = modal_summary.detect_modal_kde
    rows = []
    for config_name, (factors, support) in CONFIGURATIONS.items():
        modal_summary.detect_modal_kde = lambda x, y, f=factors, s=support: (
            branch_benchmark.detect_modal_kde(
                x, y, bandwidth_factors=f, bandwidth_support=s,
            )
        )
        for index, (key, (x, y)) in enumerate(cases.items(), start=1):
            model, domain, pair, version = key
            rows.append({
                "configuration": config_name,
                "model": model,
                "domain": domain,
                "pair": pair,
                "version": version,
                **modal_summary.classify_panel(x, y),
            })
            if index % 60 == 0:
                print(f"{config_name}: {index}/{len(cases)}", flush=True)
    modal_summary.detect_modal_kde = original_detector
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nCounts")
    print(results.groupby(["configuration", "pair", "version", "status"]).size().to_string())
    focus = results.loc[
        (
            (results["pair"] == "mrros → Q")
            & (results["model"] == "CanESM5")
            & (results["domain"] == "CW")
        )
        | (
            (results["pair"] == "P → Q")
            & (results["domain"] == "LI")
            & (results["model"].isin(["CESM2", "GFDL-CM4", "CMCC-CM2-SR5"]))
        ),
        ["configuration", "model", "domain", "pair", "version", "status",
         "branch_type", "run_length", "median_valley_depth",
         "median_min_branch_weight"],
    ]
    print("\nFocus panels")
    print(focus.to_string(index=False))

    identity = ["model", "domain", "pair", "version"]
    comparison = results.pivot(index=identity, columns="configuration", values="status")
    changed = comparison.loc[comparison.nunique(axis=1) > 1]
    print(f"\nPanels whose label changes: {len(changed)}/{len(comparison)}")
    print(changed.to_string())
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
