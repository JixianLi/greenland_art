#!/usr/bin/env python3
"""Render the figures for a run of run_latent_comparison.py.

Two figures, answering two different questions:

  latent_structure.png  Does the latent space organise itself into anything a
                        glaciologist would recognise? The same 150k points are
                        drawn four times -- raw input and each latent width --
                        and coloured twice: by day of year, and by surface
                        elevation. Nothing in the training told the model about
                        either. Structure that lines up with those colours was
                        found, not supplied.

  latent_variance.png   How much does the nonlinearity actually buy? PCA's
                        cumulative explained variance is the ruler; each
                        autoencoder is placed against it to read off the linear
                        width that would retain the same variance.

The scatter panels are UMAP, which preserves neighbourhoods and nothing else --
distances, cluster sizes and the gaps between clusters carry no meaning. They
are here to show that structure exists, never to measure it. Every number comes
from the variance figure instead.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

SOURCE_NOTE = (
    "Source: MAR v3.2 regional climate model, NCEPv1-forced, 20 km, 2000-2009 "
    "(daily, every 5th day). MODEL OUTPUT, NOT OBSERVATION.\n"
    "155 fields per ice-sheet cell, standardised. Latent spaces from an MLP "
    "autoencoder; 2D views by UMAP (neighbourhood-preserving: distances and "
    "cluster sizes are not meaningful)."
)

MINIMUM_ICE_PERCENT = 50.0
DAYS_PER_YEAR = 366
MONTH_START_DAY = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LABEL = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

# Cyclic, so 31 Dec sits next to 1 Jan rather than at the opposite end of the
# scale. Elevation uses cividis: the project style asks for single-hue
# sequential, and cividis is two-hue, but it is the one ramp explicitly
# optimised for deuteranopia, which the style also asks for. Legibility wins.
SEASON_COLORMAP = "twilight"
ELEVATION_COLORMAP = "cividis"

VIEW_TITLES = {
    "input": "input (155 fields)",
    "latent15": "latent 15",
    "latent30": "latent 30",
    "latent100": "latent 100",
}


def load_geometry(path: Path):
    """Pull only the small per-cell arrays; `features` is 1.6 GB and unused."""
    archive = np.load(path, allow_pickle=False)
    return archive["surface_height"], archive["ice_mask"]


def draw_season_key(cell) -> None:
    """Insets rather than the cell axes itself: a key that filled the grid cell
    would read as a panel of its own and dwarf what it is labelling."""
    cell.axis("off")
    bar = cell.inset_axes([0.06, 0.44, 0.88, 0.09])
    gradient = np.linspace(0.0, 1.0, DAYS_PER_YEAR)[np.newaxis, :]
    bar.imshow(gradient, aspect="auto", cmap=SEASON_COLORMAP, extent=(1, DAYS_PER_YEAR, 0, 1))
    bar.set_yticks([])
    bar.set_xticks(MONTH_START_DAY[::2])
    bar.set_xticklabels(MONTH_LABEL[::2], fontsize=7)
    bar.set_title("day of year", fontsize=9, pad=4)


def draw_elevation_key(cell, surface_height, ice_mask) -> None:
    cell.axis("off")
    field = np.where(ice_mask >= MINIMUM_ICE_PERCENT, surface_height, np.nan)

    map_axes = cell.inset_axes([0.05, 0.20, 0.90, 0.72])
    image = map_axes.imshow(field, origin="lower", cmap=ELEVATION_COLORMAP, aspect="equal")
    map_axes.axis("off")
    map_axes.set_title("surface elevation", fontsize=9, pad=4)

    bar_axes = cell.inset_axes([0.15, 0.13, 0.70, 0.028])
    bar = cell.figure.colorbar(image, cax=bar_axes, orientation="horizontal")
    bar.set_ticks([0, 1000, 2000, 3000])
    bar.ax.tick_params(labelsize=7)
    bar.set_label("m", fontsize=7, labelpad=1)


def draw_embedding(axes, points, colours, colormap, limits, title: str | None) -> None:
    axes.scatter(
        points[:, 0],
        points[:, 1],
        c=colours,
        cmap=colormap,
        vmin=limits[0],
        vmax=limits[1],
        s=0.4,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    axes.set_xticks([])
    axes.set_yticks([])
    # adjustable="datalim" keeps the axes box the size the grid gave it and
    # widens the data range instead. The default shrinks the box to fit the
    # data, which leaves every panel a different size and the rows misaligned.
    axes.set_aspect("equal", adjustable="datalim")
    if title:
        axes.set_title(title, fontsize=9)


def plot_structure(
    projections, geometry, views, output_path: Path, seed: int, source_note: str
) -> Path:
    surface_height, ice_mask = geometry
    cell_index = projections["cell_index"]
    day_of_year = projections["day_of_year"]
    elevation = surface_height.ravel()[cell_index]

    # Points are stored grouped by timestep, so drawing them in order buries
    # whichever season was written last. Shuffle once and share the order across
    # every panel, so overplotting is unbiased and the panels stay comparable.
    order = np.random.default_rng(seed).permutation(len(cell_index))

    figure = plt.figure(figsize=(3.0 + 2.6 * len(views), 6.0))
    grid = GridSpec(
        2, 1 + len(views), figure=figure,
        width_ratios=[0.85] + [1.0] * len(views), wspace=0.08, hspace=0.12,
    )

    draw_season_key(figure.add_subplot(grid[0, 0]))
    draw_elevation_key(figure.add_subplot(grid[1, 0]), surface_height, ice_mask)

    rows = [
        (day_of_year, SEASON_COLORMAP, (1, DAYS_PER_YEAR)),
        (elevation, ELEVATION_COLORMAP, (float(np.nanmin(elevation)), float(np.nanmax(elevation)))),
    ]
    for row, (colours, colormap, limits) in enumerate(rows):
        for column, view in enumerate(views):
            axes = figure.add_subplot(grid[row, 1 + column])
            draw_embedding(
                axes,
                projections[f"{view}_umap2"][order],
                colours[order],
                colormap,
                limits,
                VIEW_TITLES.get(view, view) if row == 0 else None,
            )

    figure.suptitle(
        "MLP autoencoder latent spaces: season and elevation were never given to the model",
        fontsize=12,
    )
    figure.text(0.01, 0.005, source_note, fontsize=7, color="0.35", va="bottom")
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_variance(results, output_path: Path, source_note: str) -> Path:
    widest = str(max(results["latent_dims"]))
    cumulative = np.array(results["pca"][widest]["cumulative_explained_variance_ratio"])
    components = np.arange(1, len(cumulative) + 1)

    figure, axes = plt.subplots(figsize=(7.5, 5.0))
    axes.plot(components, cumulative, color="#2a78d6", lw=2, label="PCA (linear)", zorder=3)

    for latent_dim in results["latent_dims"]:
        score = results["autoencoder"][str(latent_dim)]["validation_explained_variance_ratio"]
        equivalent = int(np.searchsorted(cumulative, score)) + 1
        beats = equivalent > latent_dim
        colour = "#e34948" if beats else "0.45"
        axes.plot(
            [latent_dim, equivalent], [score, score], color=colour, lw=1.0, ls="--", zorder=2
        )
        # Label the gap itself, above it when the autoencoder wins and below
        # when it loses, so the two cases stay visually distinct and neither
        # collides with the PCA curve running through the middle.
        axes.annotate(
            f"AE({latent_dim}) = PCA({equivalent})",
            xy=((latent_dim + equivalent) / 2, score),
            xytext=(0, 7 if beats else -15),
            textcoords="offset points", fontsize=8, color=colour, ha="center",
        )
        axes.scatter([latent_dim], [score], color="#e34948", s=45, zorder=4)

    axes.scatter([], [], color="#e34948", s=45, label="MLP autoencoder")
    axes.set_xlabel("latent dimensions")
    axes.set_ylabel("explained variance ratio (validation)")
    axes.set_title(
        "Nonlinear compression is worth ~3.7x at a tight bottleneck, nothing at a wide one",
        fontsize=11,
    )
    axes.set_xlim(0, len(cumulative) + 8)
    axes.set_ylim(0.28, 1.03)
    axes.grid(alpha=0.25)
    axes.legend(loc="lower right", frameon=False)
    figure.text(0.01, -0.06, source_note, fontsize=7, color="0.35", va="top")
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="output dir from a SLURM job")
    parser.add_argument(
        "--training-data", type=Path,
        default=Path("datasets/mar/mar_training_2000_2009.npz"),
        help="the npz the run was trained on; only its per-cell geometry is read",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    output_dir = arguments.output_dir or arguments.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    projections = np.load(arguments.run_dir / "projections.npz", allow_pickle=False)
    with open(arguments.run_dir / "results.json") as handle:
        results = json.load(handle)

    views = ["input"] + [f"latent{d}" for d in results["latent_dims"]]
    missing = [v for v in views if f"{v}_umap2" not in projections]
    if missing:
        raise SystemExit(f"projections.npz has no UMAP for {missing}; rerun without --skip-umap")

    # A figure that cannot be traced back to the run that produced it is the
    # provenance failure this project already paid for once.
    source_note = SOURCE_NOTE + f"\nRun {arguments.run_dir.name}."

    geometry = load_geometry(arguments.training_data)
    structure = plot_structure(
        projections, geometry, views, output_dir / "latent_structure.png",
        arguments.seed, source_note,
    )
    variance = plot_variance(results, output_dir / "latent_variance.png", source_note)
    print(f"Wrote {structure}\n      {variance}")


if __name__ == "__main__":
    main()
