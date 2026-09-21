"""Evaluate the convention-based fixed peak-valley cutoff P0 = 0.30."""

from pathlib import Path

import pandas as pd

from test_calibrated_peak_valley_score import extract_panel, classify_panel, plot_heatmap
from test_residual_dip import build_cases, load_runs


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
CUTOFF = 0.30
CSV_PATH = OUTPUT_DIR / "S4_fixed_peak_valley_p03_results.csv"
FIGURE_PATH = OUTPUT_DIR / "figures" / "S4_fixed_peak_valley_p03_heatmap.png"


def main():
    cases = build_cases(load_runs())
    rows = []
    for index, (key, (x, y)) in enumerate(cases.items(), start=1):
        model, domain, pair, version = key
        panel = extract_panel(x, y)
        rows.append({
            "model": model,
            "domain": domain,
            "pair": pair,
            "version": version,
            "peak_valley_cutoff": CUTOFF,
            **classify_panel(panel, CUTOFF),
        })
        if index % 15 == 0:
            print(f"Processed {index}/{len(cases)}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(CSV_PATH, index=False)
    plot_heatmap(
        results,
        CUTOFF,
        figure_path=FIGURE_PATH,
        title_prefix="Fixed geometric KDE peak-valley rule",
    )
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
         "run_length", "median_peak_valley_score"],
    ].to_string(index=False))
    print(f"\nSaved {CSV_PATH}")
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    main()
