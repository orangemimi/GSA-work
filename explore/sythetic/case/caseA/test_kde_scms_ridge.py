"""Pilot Hessian ridge topology on a continuous 2-D KDE surface."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from branch_benchmark import _standardize_xy
from test_hexbin_topology_skeleton import (
    morphological_skeleton, remove_small_components, skeleton_topology,
)
from test_residual_dip import build_cases, load_runs


CASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = CASE_DIR / "output" / "S4" / "figures"
IMAGE_SIZE = 120
SMOOTH_SIGMAS = (0.6, 1.0, 1.5, 2.1, 2.8)


def kde_density_image(x, y, seed=17):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xs, ys, _ = _standardize_xy(x, y)
    keep = (
        (xs >= np.quantile(xs, 0.0025)) & (xs <= np.quantile(xs, 0.9975))
        & (ys >= np.quantile(ys, 0.0025)) & (ys <= np.quantile(ys, 0.9975))
    )
    sample = np.column_stack([xs[keep], ys[keep]])
    if len(sample) > 5000:
        rng = np.random.default_rng(seed)
        sample = sample[rng.choice(len(sample), 5000, replace=False)]
    x_lo, x_hi = np.quantile(sample[:, 0], [0.0025, 0.9975])
    y_lo, y_hi = np.quantile(sample[:, 1], [0.0025, 0.9975])
    x_grid = np.linspace(x_lo, x_hi, IMAGE_SIZE)
    y_grid = np.linspace(y_lo, y_hi, IMAGE_SIZE)
    xx, yy = np.meshgrid(x_grid, y_grid)
    kde = gaussian_kde(
        sample.T,
        bw_method=lambda obj: obj.scotts_factor() * 0.65,
    )
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    density = np.flipud(density.astype(np.float32))
    density /= max(float(density.max()), 1e-12)
    return density


def ridge_from_kde(density, sigma):
    smooth = cv2.GaussianBlur(
        density.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma,
    )
    gy, gx = np.gradient(smooth)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    hxy = (gxy + gyx) / 2
    trace = gxx + gyy
    disc = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4 * hxy ** 2, 0))
    lambda_normal = (trace - disc) / 2

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
    offset = 1.6
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
    density_gate = smooth >= 0.025 * smooth.max()
    negative = lambda_normal < 0
    curvature = -lambda_normal
    eligible = density_gate & negative
    curvature_cut = np.quantile(curvature[eligible], 0.35) if np.any(eligible) else np.inf
    ridge = (
        eligible & (smooth >= plus) & (smooth >= minus)
        & (curvature >= curvature_cut)
    )
    ridge = cv2.morphologyEx(
        ridge.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
    )
    ridge = remove_small_components(ridge, min_area=7)
    ridge = morphological_skeleton(ridge)
    topology = skeleton_topology(ridge, min_arm_length=6)
    return smooth, ridge, topology


def detect_panel(x, y, seed=17):
    density = kde_density_image(x, y, seed=seed)
    scales = []
    for sigma in SMOOTH_SIGMAS:
        smooth, ridge, topology = ridge_from_kde(density, sigma)
        scales.append({
            "sigma": sigma,
            "branch": topology["valid_junctions"] > 0,
            "smooth": smooth,
            "ridge": ridge,
            "topology": topology,
        })
    longest = current = 0
    for item in scales:
        current = current + 1 if item["branch"] else 0
        longest = max(longest, current)
    status = "Branch" if longest >= 2 else ("Candidate" if longest == 1 else "No branch")
    best = max(
        scales,
        key=lambda item: (
            item["topology"]["valid_junctions"],
            sum(item["topology"]["best_arm_lengths"]),
        ),
    )
    return density, scales, best, status, longest


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
    for index, (ax, key) in enumerate(zip(axes.ravel(), pilots)):
        model, domain, pair, version, expected = key
        x, y = cases[(model, domain, pair, version)]
        density, scales, best, status, run = detect_panel(x, y, seed=170 + index)
        ax.imshow(density, cmap="Greys", origin="upper")
        ridge_y, ridge_x = np.where(best["ridge"])
        ax.scatter(ridge_x, ridge_y, s=4, color="#DC2626", alpha=0.9)
        pattern = "".join("1" if item["branch"] else "0" for item in scales)
        ax.set_title(
            f"{model} · {domain} · {pair} · {version}\n"
            f"{status}; run={run}; {pattern}; {expected}",
            fontsize=8.2, fontweight="bold",
            color="#B91C1C" if status != "No branch" else "#374151",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        print((model, domain, pair, version, expected, status, run, pattern,
               best["sigma"], best["topology"]["best_arm_lengths"]))
    fig.suptitle("Pilot: Hessian ridges on continuous 2-D KDE surfaces", fontsize=15, fontweight="bold")
    output = FIGURE_DIR / "S4_kde_scms_ridge_pilot.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
