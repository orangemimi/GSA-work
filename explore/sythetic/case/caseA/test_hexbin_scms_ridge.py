"""Pilot Hessian density-ridge topology on the Hexbin count surface."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from test_hexbin_topology_skeleton import (
    hex_density_image, morphological_skeleton, remove_small_components,
    skeleton_topology,
)
from test_residual_dip import build_cases, load_runs


CASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = CASE_DIR / "output" / "S4" / "figures"
SIGMAS = (1.5, 2.0, 2.75, 3.5, 4.5)


def ridge_at_scale(density, sigma):
    smooth = cv2.GaussianBlur(
        density.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma,
    )
    gy, gx = np.gradient(smooth)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    hxy = (gxy + gyx) / 2
    trace = gxx + gyy
    discriminant = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4 * hxy ** 2, 0))
    lambda_normal = (trace - discriminant) / 2

    # Eigenvector belonging to the most negative Hessian eigenvalue.
    vx = hxy.copy()
    vy = lambda_normal - gxx
    norm = np.sqrt(vx ** 2 + vy ** 2)
    fallback = norm <= 1e-12
    vx[fallback] = 1.0
    vy[fallback] = 0.0
    norm[fallback] = 1.0
    vx /= norm
    vy /= norm

    height, width = smooth.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    offset = 1.5
    plus = cv2.remap(
        smooth, (xx + offset * vx).astype(np.float32),
        (yy + offset * vy).astype(np.float32), cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    minus = cv2.remap(
        smooth, (xx - offset * vx).astype(np.float32),
        (yy - offset * vy).astype(np.float32), cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    density_gate = smooth >= 0.035 * smooth.max()
    negative = lambda_normal < 0
    curvature = -lambda_normal
    eligible = density_gate & negative
    curvature_cut = np.quantile(curvature[eligible], 0.45) if np.any(eligible) else np.inf
    local_normal_max = (smooth >= plus) & (smooth >= minus)
    ridge = eligible & local_normal_max & (curvature >= curvature_cut)

    # Connect one-cell discretization gaps, then return a one-pixel ridge graph.
    ridge = cv2.morphologyEx(
        ridge.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
    )
    ridge = remove_small_components(ridge, min_area=5)
    ridge = morphological_skeleton(ridge)
    topology = skeleton_topology(ridge, min_arm_length=5)
    return smooth, ridge, topology


def detect_ridge_topology(x, y):
    density = hex_density_image(x, y)
    runs = []
    for sigma in SIGMAS:
        smooth, ridge, topology = ridge_at_scale(density, sigma)
        runs.append({
            "sigma": sigma,
            "branch": topology["valid_junctions"] > 0,
            "ridge": ridge,
            "smooth": smooth,
            "topology": topology,
        })
    best_len = current = 0
    for item in runs:
        current = current + 1 if item["branch"] else 0
        best_len = max(best_len, current)
    status = "Branch" if best_len >= 2 else ("Candidate" if best_len == 1 else "No branch")
    best = max(
        runs,
        key=lambda item: (
            item["topology"]["valid_junctions"],
            len(item["topology"]["best_arm_lengths"]),
            sum(item["topology"]["best_arm_lengths"]),
        ),
    )
    return status, best_len, runs, density, best


def main():
    cases = build_cases(load_runs())
    pilots = [
        ("CESM2", "LI", "P → Q", "trimmed", "expected branch"),
        ("GFDL-CM4", "LI", "P → Q", "trimmed", "expected branch"),
        ("CMCC-CM2-SR5", "LI", "P → Q", "trimmed", "expected branch"),
        ("GFDL-CM4", "LI", "P → Q", "raw", "expected branch"),
        ("CMCC-CM2-SR5", "LI", "P → Q", "raw", "expected branch"),
        ("CanESM5", "CW", "mrros → Q", "trimmed", "expected local candidate"),
        ("CESM2", "WW", "P → Q", "trimmed", "negative"),
        ("GFDL-CM4", "all_land", "P → Q", "trimmed", "wide negative"),
        ("CESM2", "all_land", "hfls → Q", "trimmed", "wide negative"),
        ("CanESM5", "WW", "hfls → Q", "trimmed", "wide negative"),
        ("GFDL-CM4", "WW", "hfls → Q", "trimmed", "wide negative"),
        ("CNRM-CM6-1", "CW", "P → Q", "trimmed", "negative"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    rows = []
    for ax, key in zip(axes.ravel(), pilots):
        model, domain, pair, version, expected = key
        x, y = cases[(model, domain, pair, version)]
        status, run, scales, density, best = detect_ridge_topology(x, y)
        ax.imshow(density, cmap="Greys", origin="upper")
        ridge_y, ridge_x = np.where(best["ridge"])
        ax.scatter(ridge_x, ridge_y, s=3, color="#DC2626", alpha=0.85)
        pattern = "".join("1" if item["branch"] else "0" for item in scales)
        ax.set_title(
            f"{model} · {domain} · {pair} · {version}\n"
            f"{status}; run={run}; {pattern}; {expected}",
            fontsize=8.2, fontweight="bold",
            color="#B91C1C" if status != "No branch" else "#374151",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        rows.append((model, domain, pair, version, expected, status, run, pattern,
                     best["sigma"], best["topology"]["best_arm_lengths"]))
    fig.suptitle("Pilot: Hessian ridges on Hexbin count surfaces", fontsize=15, fontweight="bold")
    output = FIGURE_DIR / "S4_hexbin_scms_ridge_pilot.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    for row in rows:
        print(row)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
