from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "S3_polyfit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_models(x, y):
    fits = {}
    n = len(x)

    for degree, name in ((1, "Linear"), (2, "Quadratic"), (3, "Cubic")):
        coefficients = np.polyfit(x, y, degree)
        fitted = np.polyval(coefficients, x)
        rss = np.sum((y - fitted) ** 2)
        parameter_count = degree + 1
        bic = n * np.log(rss / n) + parameter_count * np.log(n)
        fits[name] = {
            "coefficients": coefficients,
            "bic": bic,
        }

    return fits


def draw_panel(ax, x, y, panel_label, expected_class):
    fits = fit_models(x, y)
    x_grid = np.linspace(x.min(), x.max(), 400)
    line_styles = {
        "Linear": {"color": "#222222", "linestyle": "-", "linewidth": 1.8},
        "Quadratic": {"color": "#D55E00", "linestyle": "--", "linewidth": 1.8},
        "Cubic": {"color": "#009E73", "linestyle": ":", "linewidth": 2.2},
    }

    ax.scatter(
        x,
        y,
        s=11,
        color="#2878B5",
        alpha=0.68,
        edgecolors="none",
        zorder=2,
        label="Observed data",
    )

    for name, fit in fits.items():
        y_grid = np.polyval(fit["coefficients"], x_grid)
        is_minimum = fit["bic"] == min(item["bic"] for item in fits.values())
        suffix = "  (minimum)" if is_minimum else ""
        ax.plot(
            x_grid,
            y_grid,
            label=f"{name}: BIC = {fit['bic']:.1f}{suffix}",
            zorder=3,
            **line_styles[name],
        )

    best_name = min(fits, key=lambda name: fits[name]["bic"])
    nonlinear_best = min(fits["Quadratic"]["bic"], fits["Cubic"]["bic"])
    delta_bic = fits["Linear"]["bic"] - nonlinear_best

    ax.set_title(
        f"{panel_label} {expected_class} relationship",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.text(
        0.02,
        0.97,
        f"Selected model: {best_name.lower()}\n"
        rf"$\Delta$BIC = {delta_bic:.1f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#222222",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(colors="#444444", labelsize=9)

    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 3, 0]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.96,
        fontsize=8.5,
        handlelength=3.0,
        borderpad=0.7,
    )


def main():
    rng = np.random.default_rng(18)
    n = 80
    x = np.linspace(-1.0, 1.0, n)

    linear_y = 0.72 * x + 0.10 + rng.normal(0.0, 0.075, n)
    nonlinear_y = (
        0.62 * x**3
        + 0.34 * x**2
        - 0.10 * x
        + 0.02
        + rng.normal(0.0, 0.075, n)
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.7),
        constrained_layout=True,
    )

    draw_panel(axes[0], x, linear_y, "(a)", "Linear")
    draw_panel(axes[1], x, nonlinear_y, "(b)", "Nonlinear")

    fig.suptitle(
        "Polynomial model selection using BIC",
        fontsize=14,
        fontweight="bold",
    )
    fig.supxlabel(
        r"Lower BIC indicates stronger model support.  "
        r"$\Delta$BIC = BIC$_{\mathrm{linear}}$"
        r" - min(BIC$_{\mathrm{quadratic}}$, BIC$_{\mathrm{cubic}}$).",
        fontsize=9.5,
    )

    png_path = OUTPUT_DIR / "bic_model_selection_scientific.png"
    svg_path = OUTPUT_DIR / "bic_model_selection_scientific.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
