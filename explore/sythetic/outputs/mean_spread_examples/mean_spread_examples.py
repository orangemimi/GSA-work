"""Create conceptual examples of mean response and spread patterns."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
BLUE = "#168bd2"
LINE = "#2d6f9f"


def sigmoid_mean(x: np.ndarray) -> np.ndarray:
    """Smooth monotonic mean response used in panels A and B."""
    return 0.8 + 7.7 / (1.0 + np.exp(-(x - 5.0) / 1.45))


def make_data(seed: int = 27, n: int = 360):
    rng = np.random.default_rng(seed)

    x_a = np.sort(rng.uniform(0, 10, n))
    mean_a = sigmoid_mean(x_a)
    y_a = mean_a + rng.normal(0, 0.95, n)

    x_b = np.sort(rng.uniform(0, 10, n))
    mean_b = sigmoid_mean(x_b)
    # Clearly heteroscedastic: uncertainty grows continuously with x.
    sigma_b = 0.18 + 0.20 * x_b
    y_b = mean_b + rng.normal(0, sigma_b, n)

    x_c = np.sort(rng.uniform(0, 10, n))
    mean_c = 5.0 + 0.60 * np.sin(0.9 * x_c)
    sigma_c = 0.18 + 0.17 * x_c
    y_c = mean_c + rng.normal(0, sigma_c, n)

    return (
        (x_a, y_a, mean_a),
        (x_b, y_b, mean_b),
        (x_c, y_c, mean_c),
    )


def r_squared(y: np.ndarray, fitted: np.ndarray) -> float:
    return 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.2, 10.5)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 11, 2))
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.tick_params(axis="both", labelsize=7, length=3, color="#777777")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")


def draw_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    title: str,
) -> None:
    ax.scatter(x, y, s=7, c=BLUE, alpha=0.92, linewidths=0)
    ax.plot(x, mean, color=LINE, lw=1.8)
    ax.text(
        0.06,
        0.89,
        rf"$R^2$ = {r_squared(y, mean):.2f}",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.set_title(title, fontsize=10, pad=7)
    style_axis(ax)


def main() -> None:
    panel_a, panel_b, panel_c = make_data()
    titles = (
        "A. Clear mean response",
        "B. Mean response + increasing spread",
        "C. Dominated by spread patterns",
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.15), sharex=True, sharey=True)
    for ax, data, title in zip(axes, (panel_a, panel_b, panel_c), titles):
        draw_panel(ax, *data, title)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.86, wspace=0.28)
    fig.savefig(OUT_DIR / "mean_response_spread_three_panels.png", dpi=300)
    fig.savefig(OUT_DIR / "mean_response_spread_three_panels.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.2, 3.15))
    draw_panel(ax, *panel_b, titles[1])
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.17, top=0.86)
    fig.savefig(OUT_DIR / "mean_response_increasing_spread.png", dpi=300)
    fig.savefig(OUT_DIR / "mean_response_increasing_spread.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
