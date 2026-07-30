#!/usr/bin/env python3
"""Render the latent comparison results returned from the cluster.

Reads results.json and projections.npz written by run_latent_comparison.py and
produces two figures:

  latent_variance.png    PCA explained variance -- per component and cumulative
                         -- with the autoencoder's total marked at each latent
                         width, so nonlinearity buys something or visibly does not.
  latent_space.png       2D PCA and UMAP views of each latent space, coloured by
                         day of year.

The variance figure deliberately puts PCA and the autoencoder on one axis. An
autoencoder that does not beat PCA at equal latent width has bought nothing with
its nonlinearity, and showing that plainly is the point of having a baseline.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

LATENT_COLOURS = [BLUE, ORANGE, AQUA]


def configure() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE, "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK_PRIMARY, "axes.edgecolor": BASELINE,
            "axes.spines.top": False, "axes.spines.right": False,
        }
    )


def build_variance_figure(results: dict, output_path: Path) -> Path:
    configure()
    latent_dims = results["latent_dims"]
    widest = str(max(latent_dims))
    ratios = np.array(results["pca"][widest]["explained_variance_ratio_per_component"])
    cumulative = np.array(results["pca"][widest]["cumulative_explained_variance_ratio"])

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    components = np.arange(1, len(ratios) + 1)
    left.bar(components, ratios * 100, color=BLUE, width=0.9, linewidth=0)
    left.set_xlabel("principal component", fontsize=9)
    left.set_ylabel("variance explained (%)", fontsize=9)
    left.set_title("Each component's share", fontsize=10.5, loc="left", pad=8)
    left.grid(axis="y", color=GRIDLINE, linewidth=0.6)
    left.set_axisbelow(True)
    left.tick_params(labelsize=8)
    left.annotate(
        f"PC1 alone: {ratios[0] * 100:.1f}%",
        xy=(1, ratios[0] * 100), xytext=(len(ratios) * 0.25, ratios[0] * 100 * 0.85),
        fontsize=8.5, color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )

    right.plot(components, cumulative * 100, color=BLUE, linewidth=2.0, label="PCA (cumulative)")
    right.axhline(95, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
    right.text(len(ratios) * 0.02, 95.8, "95%", fontsize=8, color=INK_MUTED)

    for colour, latent_dim in zip(LATENT_COLOURS, latent_dims):
        autoencoder = results["autoencoder"][str(latent_dim)]["validation_explained_variance_ratio"]
        pca_value = results["pca"][str(latent_dim)]["validation_explained_variance_ratio"]
        right.scatter([latent_dim], [autoencoder * 100], s=70, color=colour, zorder=5,
                      label=f"MLP-AE({latent_dim})")
        right.annotate(
            f"AE {autoencoder * 100:.1f}\nPCA {pca_value * 100:.1f}",
            xy=(latent_dim, autoencoder * 100), xytext=(6, -16),
            textcoords="offset points", fontsize=8, color=colour, linespacing=1.35,
        )

    right.set_xlabel("latent dimensions", fontsize=9)
    right.set_ylabel("variance explained (%)", fontsize=9)
    right.set_title(
        "Cumulative — and what the autoencoder achieves at the same width",
        fontsize=10.5, loc="left", pad=8,
    )
    right.grid(axis="y", color=GRIDLINE, linewidth=0.6)
    right.set_axisbelow(True)
    right.tick_params(labelsize=8)
    right.legend(fontsize=8, frameon=False, loc="lower right")

    figure.suptitle(
        f"How many independent dimensions are in {results['n_features']} MAR fields?",
        fontsize=13.5, x=0.012, ha="left", y=1.02,
    )
    figure.text(
        0.012, 0.955,
        f"{results['n_samples']:,} samples, one per ice cell per day. "
        "PCA is linear and takes seconds; the autoencoder is nonlinear. Compared at equal latent "
        "width, so any difference is attributable to nonlinearity rather than to a wider bottleneck.",
        fontsize=8.8, color=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=165, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_latent_space_figure(results: dict, projections, output_path: Path) -> Path:
    configure()
    latent_dims = results["latent_dims"]
    labels = ["input"] + [f"latent{d}" for d in latent_dims]
    titles = [f"input ({results['n_features']} fields)"] + [f"MLP-AE latent {d}" for d in latent_dims]

    methods = [m for m in ("pca2", "umap2") if f"input_{m}" in projections]
    figure, axes = plt.subplots(
        len(methods), len(labels), figsize=(4.0 * len(labels), 4.1 * len(methods)), squeeze=False
    )

    day = projections["day_of_year"]
    for row, method in enumerate(methods):
        for column, (label, title) in enumerate(zip(labels, titles)):
            axis = axes[row][column]
            key = f"{label}_{method}"
            if key not in projections:
                axis.axis("off")
                continue
            embedding = projections[key]
            # twilight is cyclic, which matches day-of-year: 31 December sits
            # next to 1 January rather than at the opposite end of the ramp.
            scatter = axis.scatter(
                embedding[:, 0], embedding[:, 1], c=day, cmap="twilight",
                s=2.5, alpha=0.5, linewidths=0, vmin=1, vmax=366,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(title, fontsize=10, loc="left", pad=6)
            if column == 0:
                axis.set_ylabel(
                    "PCA" if method == "pca2" else "UMAP", fontsize=10, color=INK_PRIMARY
                )

    colour_bar = figure.colorbar(
        scatter, ax=axes, fraction=0.012, pad=0.012, ticks=[1, 91, 182, 274, 366]
    )
    colour_bar.ax.set_yticklabels(["Jan", "Apr", "Jul", "Oct", "Dec"], fontsize=8)
    colour_bar.set_label("day of year", fontsize=8.5, color=INK_SECONDARY)
    colour_bar.outline.set_visible(False)

    figure.suptitle(
        "The latent spaces, viewed in 2D — coloured by day of year",
        fontsize=13.5, x=0.012, ha="left", y=1.085,
    )
    figure.text(
        0.012, 1.048,
        "Colour is not used by any model; it is applied afterwards. If the seasonal cycle appears as "
        "structure, the representation found it unsupervised.\nUMAP distances and cluster sizes are "
        "not metric — read it for connectivity, never for magnitude.",
        fontsize=8.8, color=INK_SECONDARY, linespacing=1.5, va="top",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=155, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    arguments = parser.parse_args()

    with open(arguments.results_dir / "results.json") as handle:
        results = json.load(handle)
    projections = np.load(arguments.results_dir / "projections.npz")

    print(f"Wrote {build_variance_figure(results, arguments.output_dir / 'latent_variance.png')}")
    print(
        f"Wrote {build_latent_space_figure(results, projections, arguments.output_dir / 'latent_space.png')}"
    )

    print("\nSummary")
    print(f"{'latent':>8}{'PCA EVR':>10}{'AE EVR':>10}{'PCA MSE':>11}{'AE MSE':>11}")
    for latent_dim in results["latent_dims"]:
        pca = results["pca"][str(latent_dim)]
        autoencoder = results["autoencoder"][str(latent_dim)]
        print(
            f"{latent_dim:>8}"
            f"{pca['validation_explained_variance_ratio']:>10.4f}"
            f"{autoencoder['validation_explained_variance_ratio']:>10.4f}"
            f"{pca['validation_mse']:>11.5f}"
            f"{autoencoder['validation_mse']:>11.5f}"
        )


if __name__ == "__main__":
    main()
