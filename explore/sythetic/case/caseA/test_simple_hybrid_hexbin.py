"""Conservative B/T/C summary combining local consensus and topology.

This diagnostic deliberately keeps the public rule short:
  1. a two-ridge candidate must recur at the same x location in >=5/6 grids;
  2. trajectory topology labels it Branch, Two-band, or Candidate.

No boundary/zero-value feature is used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import pandas as pd

from branch_benchmark import detect_topological_multiextent_hexbin
from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"


def main():
    local = pd.read_csv(OUTPUT_DIR / "S4_local_consensus_hexbin_results.csv")
    cases = build_cases(load_runs())
    records = []
    for item in local.itertuples(index=False):
        status = "No branch"
        topology = "Not tested"
        strong = continuous = 0
        if item.scale_support >= 5:
            x, y = cases[(item.model, item.domain, item.pair, item.version)]
            result = detect_topological_multiextent_hexbin(x, y)
            geometry = result.geometry or {}
            topology = geometry.get("track_status", "No branch")
            strong = int(geometry.get("strong_support", 0))
            continuous = int(geometry.get("continuous_support", 0))
            if topology in {"Branch", "Strong candidate"}:
                status = "Branch"
            elif topology == "Wide/two-ridge":
                status = "Two-band"
            else:
                # The local structure is reproducible, but the observed track
                # is too short or lacks a visible common trunk.
                status = "Candidate"
        records.append({
            "model": item.model,
            "domain": item.domain,
            "pair": item.pair,
            "version": item.version,
            "scale_support": int(item.scale_support),
            "topology_status": topology,
            "strong_support": strong,
            "continuous_support": continuous,
            "final_status": status,
        })
    results = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "S4_simple_hybrid_hexbin_results.csv"
    results.to_csv(csv_path, index=False)

    codes = {"No branch": 0, "Candidate": 1, "Two-band": 2, "Branch": 3}
    letters = {"No branch": "-", "Candidate": "C", "Two-band": "T", "Branch": "B"}
    colors = ListedColormap(["#F3F4F6", "#F59E0B", "#7C3AED", "#2563EB"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], colors.N)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True)
    for row, version in enumerate(("raw", "trimmed")):
        for col, (_, _, pair) in enumerate(PAIRS):
            ax = axes[row, col]
            selected = results.loc[
                (results["pair"] == pair) & (results["version"] == version)
            ].copy()
            selected["code"] = selected["final_status"].map(codes)
            matrix = selected.pivot(index="model", columns="domain", values="code").reindex(
                index=MODELS, columns=DOMAINS
            )
            statuses = selected.pivot(
                index="model", columns="domain", values="final_status"
            ).reindex(index=MODELS, columns=DOMAINS)
            support = selected.pivot(
                index="model", columns="domain", values="scale_support"
            ).reindex(index=MODELS, columns=DOMAINS)
            ax.imshow(matrix, cmap=colors, norm=norm, aspect="auto")
            for i in range(len(MODELS)):
                for j in range(len(DOMAINS)):
                    status = statuses.iloc[i, j]
                    color = "white" if codes[status] >= 2 else "#374151"
                    ax.text(j, i, f"{letters[status]}\n{int(support.iloc[i,j])}/6",
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color=color)
            counts = selected["final_status"].value_counts()
            ax.set_title(
                f"{pair} · {version.upper()} · "
                f"B={counts.get('Branch',0)}, T={counts.get('Two-band',0)}, C={counts.get('Candidate',0)}",
                fontsize=10.5, fontweight="bold",
            )
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.tick_params(labelsize=8)
    fig.suptitle(
        "Simple hybrid Hexbin: cross-scale two-ridge consensus + trajectory topology\n"
        "B=split/merge, T=stable two-band, C=reproducible but truncated/uncertain",
        fontsize=14.5, fontweight="bold",
    )
    figure_path = FIGURE_DIR / "S4_simple_hybrid_hexbin_heatmap.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(results.groupby(["pair", "version", "final_status"]).size().to_string())
    print("\nNon-null classifications:")
    print(results.loc[results["final_status"] != "No branch", [
        "pair", "version", "model", "domain", "scale_support",
        "topology_status", "final_status",
    ]].sort_values(["pair", "version", "final_status", "model"]).to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
