"""Prototype topology detector on the Hexbin count surface.

The rendered plot is not analysed.  Matplotlib's Hexbin centers and counts are
rasterized into a compact density field, thresholded at several density levels,
and thinned morphologically.  A threshold votes for a geometric branch when a
skeleton junction has at least three sufficiently long incident arms.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from test_residual_dip import build_cases, load_runs, DOMAINS, MODELS, PAIRS


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output" / "S4"
FIGURE_DIR = OUTPUT_DIR / "figures"
GRIDSIZE = 40
IMAGE_SIZE = 128
LEVELS = (0.10, 0.14, 0.18, 0.23, 0.29, 0.36, 0.44)


def hex_density_image(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_lo, x_hi = np.quantile(x, [0.0025, 0.9975])
    y_lo, y_hi = np.quantile(y, [0.0025, 0.9975])
    x_span = max(float(x_hi - x_lo), 1e-12)
    y_span = max(float(y_hi - y_lo), 1e-12)
    xn = np.clip((x - x_lo) / x_span, 0, 1)
    yn = np.clip((y - y_lo) / y_span, 0, 1)

    figure = Figure(figsize=(1, 1))
    axis = figure.subplots()
    collection = axis.hexbin(xn, yn, gridsize=GRIDSIZE, mincnt=1)
    centers = np.asarray(collection.get_offsets(), dtype=float)
    counts = np.asarray(collection.get_array(), dtype=float)
    density = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    radius = max(2, int(round(IMAGE_SIZE / (2.2 * GRIDSIZE))))
    for (cx, cy), count in zip(centers, counts):
        px = int(round(cx * (IMAGE_SIZE - 1)))
        py = int(round((1 - cy) * (IMAGE_SIZE - 1)))
        if not (0 <= px < IMAGE_SIZE and 0 <= py < IMAGE_SIZE):
            continue
        value = float(np.log1p(count))
        layer = np.zeros_like(density)
        cv2.circle(layer, (px, py), radius, value, thickness=-1)
        density = np.maximum(density, layer)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=1.25, sigmaY=1.25)
    if density.max() > 0:
        density /= density.max()
    return density


def remove_small_components(mask, min_area=24):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned


def morphological_skeleton(mask):
    """One-pixel Zhang-Suen thinning of a binary density mask."""
    image = mask.astype(np.uint8).copy()
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    for _ in range(IMAGE_SIZE * 2):
        changed = False
        for subiteration in (0, 1):
            p2 = np.roll(image, -1, axis=0)
            p3 = np.roll(np.roll(image, -1, axis=0), 1, axis=1)
            p4 = np.roll(image, 1, axis=1)
            p5 = np.roll(np.roll(image, 1, axis=0), 1, axis=1)
            p6 = np.roll(image, 1, axis=0)
            p7 = np.roll(np.roll(image, 1, axis=0), -1, axis=1)
            p8 = np.roll(image, -1, axis=1)
            p9 = np.roll(np.roll(image, -1, axis=0), -1, axis=1)
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if subiteration == 0:
                condition_a = (p2 * p4 * p6) == 0
                condition_b = (p4 * p6 * p8) == 0
            else:
                condition_a = (p2 * p4 * p8) == 0
                condition_b = (p2 * p6 * p8) == 0
            remove = (
                (image == 1) & (neighbors >= 2) & (neighbors <= 6)
                & (transitions == 1) & condition_a & condition_b
            )
            remove[[0, -1], :] = False
            remove[:, [0, -1]] = False
            if np.any(remove):
                image[remove] = 0
                changed = True
        if not changed:
            break
    return image > 0


def skeleton_topology(skeleton, min_arm_length=7):
    binary = skeleton.astype(np.uint8)
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0
    degree = cv2.filter2D(binary, cv2.CV_16U, neighbor_kernel)
    raw_junction = ((binary > 0) & (degree >= 3)).astype(np.uint8)
    junction = cv2.dilate(raw_junction, np.ones((3, 3), np.uint8), iterations=1)
    junction &= binary
    n_junctions, junction_labels = cv2.connectedComponents(junction, 8)
    arms_image = binary.copy()
    arms_image[junction > 0] = 0
    n_arms, arm_labels, arm_stats, _ = cv2.connectedComponentsWithStats(arms_image, 8)

    valid_junctions = 0
    best_arms = []
    for junction_label in range(1, n_junctions):
        component = (junction_labels == junction_label).astype(np.uint8)
        neighborhood = cv2.dilate(component, np.ones((3, 3), np.uint8), iterations=1)
        touching = set(np.unique(arm_labels[neighborhood > 0]).tolist()) - {0}
        lengths = sorted(
            [int(arm_stats[label, cv2.CC_STAT_AREA]) for label in touching],
            reverse=True,
        )
        long_arms = [length for length in lengths if length >= min_arm_length]
        if len(long_arms) >= 3:
            valid_junctions += 1
            if len(long_arms) > len(best_arms):
                best_arms = long_arms
    return {
        "valid_junctions": valid_junctions,
        "best_arm_lengths": best_arms,
        "skeleton_pixels": int(binary.sum()),
    }


def detect_panel(x, y):
    density = hex_density_image(x, y)
    level_rows = []
    for level in LEVELS:
        mask = (density >= level).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1,
        )
        mask = remove_small_components(mask)
        skeleton = morphological_skeleton(mask)
        topology = skeleton_topology(skeleton)
        level_rows.append({
            "level": level,
            "branch": topology["valid_junctions"] > 0,
            **topology,
        })
    votes = int(sum(row["branch"] for row in level_rows))
    longest = current = 0
    for row in level_rows:
        current = current + 1 if row["branch"] else 0
        longest = max(longest, current)
    if longest >= 3:
        status = "Branch"
    elif longest >= 1:
        status = "Candidate"
    else:
        status = "No branch"
    return {
        "status": status,
        "branch_levels": votes,
        "longest_level_run": longest,
        "level_pattern": "".join("1" if row["branch"] else "0" for row in level_rows),
    }


def plot_heatmap(results):
    codes = {"No branch": 0, "Candidate": 1, "Branch": 2}
    letters = {"No branch": "-", "Candidate": "C", "Branch": "B"}
    cmap = ListedColormap(["#F3F4F6", "#F59E0B", "#2563EB"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
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
            run = selected.pivot(index="model", columns="domain", values="longest_level_run").reindex(
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
                f"{pair} · {version.upper()} · B={counts.get('Branch',0)}, C={counts.get('Candidate',0)}",
                fontsize=10.5, fontweight="bold",
            )
            ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
            ax.set_yticks(range(len(MODELS)), MODELS)
            ax.tick_params(labelsize=8)
    fig.suptitle(
        "Hexbin density topology: persistent skeleton junctions across density levels",
        fontsize=14.5, fontweight="bold",
    )
    path = FIGURE_DIR / "S4_hexbin_topology_skeleton_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    cases = build_cases(load_runs())
    rows = []
    for key, (x, y) in cases.items():
        model, domain, pair, version = key
        rows.append({"model": model, "domain": domain, "pair": pair, "version": version,
                     **detect_panel(x, y)})
    results = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "S4_hexbin_topology_skeleton_results.csv"
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
