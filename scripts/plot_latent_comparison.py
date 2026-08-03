#!/usr/bin/env python3
"""Render the figures for a run of run_latent_comparison.py.

Two figures, answering two different questions:

  latent_structure.png  What does the bottleneck keep? The same 150k points are
                        drawn four times -- raw input and each latent width --
                        and coloured twice, by day of year and by surface
                        elevation. Neither is a column of the input matrix, but
                        they are not equivalent evidence and the figure labels
                        the difference: elevation is recoverable from the inputs
                        exactly, day of year is not. See SEASON_CAVEAT and
                        ELEVATION_CAVEAT for the measured numbers.

  latent_variance.png   How much does the nonlinearity actually buy? PCA's
                        cumulative explained variance is the ruler; each
                        autoencoder is placed against it to read off the linear
                        width that would retain the same variance.

  reconstruction_*.png  Does what survived still look like an ice sheet? Truth
                        beside every PCA and autoencoder reconstruction, in
                        physical units on the model grid. An explained-variance
                        number cannot answer this.

Every view is drawn twice, by PCA and by UMAP, because they fail differently.
PCA axes are linear combinations of real fields and its two components are the
two highest-variance directions, so it cannot show structure that is not linear.
UMAP reaches that structure but preserves neighbourhoods and nothing else --
distances, cluster sizes and inter-cluster gaps carry no meaning. Structure
appearing in both is evidence; structure in only one is a property of the
algorithm. Neither is ever the source of a number: those come from the variance
figure and from physical_column_r2 in results.json.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

SOURCE_NOTE = (
    "Source: MAR v3.2 regional climate model, NCEPv1-forced, 20 km, 2000-2009 "
    "(daily, every 5th day). MODEL OUTPUT, NOT OBSERVATION.\n"
    "155 fields per ice-sheet cell. Latent spaces from an MLP autoencoder; 2D "
    "views by PCA and by UMAP. UMAP preserves neighbourhoods only -- its "
    "distances, cluster sizes and inter-cluster gaps are not meaningful."
)

# The two colourings are not equally impressive, and the figure has to say so.
# Elevation is in the inputs in all but name: ZZ_L00, the height of the lowest
# atmospheric level, tracks surface elevation at R^2 = 1.0000, and SP at 0.9942.
# Day of year is not an input, and no single field determines it -- the best,
# SWD, fits a one-harmonic annual cycle at R^2 = 0.875 (median over ice cells),
# and one harmonic is symmetric about the solstice, so even a perfect fit leaves
# spring and autumn indistinguishable. Recovering the date needs several fields
# and the melt-season hysteresis that breaks that symmetry.
SEASON_CAVEAT = (
    "not an input.\nNo single field fixes it:\nSWD annual cycle $R^2$=0.88,\n"
    "and one harmonic cannot\ntell spring from autumn."
)
ELEVATION_CAVEAT = (
    "effectively an input:\nZZ_L00 $R^2$=1.00, SP 0.99.\n"
    "Preserved, not discovered."
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


def resolve_example_fields(requested, available):
    present = [name for name in requested if name in available]
    missing = [name for name in requested if name not in available]
    if missing:
        print(f"skipping fields absent from this run: {', '.join(missing)}")
    if not present:
        raise SystemExit(f"none of {requested} are columns of this run")
    return present


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
    cell.text(0.5, 0.33, SEASON_CAVEAT, fontsize=6.5, color="0.35", ha="center", va="top")


def draw_elevation_key(cell, surface_height, ice_mask) -> None:
    cell.axis("off")
    field = np.where(ice_mask >= MINIMUM_ICE_PERCENT, surface_height, np.nan)

    map_axes = cell.inset_axes([0.05, 0.20, 0.90, 0.72])
    image = map_axes.imshow(field, origin="lower", cmap=ELEVATION_COLORMAP, aspect="equal")
    map_axes.axis("off")
    map_axes.set_title("surface elevation (m)", fontsize=9, pad=4)

    bar_axes = cell.inset_axes([0.15, 0.165, 0.70, 0.028])
    bar = cell.figure.colorbar(image, cax=bar_axes, orientation="horizontal")
    bar.set_ticks([0, 1000, 2000, 3000])
    bar.ax.tick_params(labelsize=7)
    cell.text(0.5, 0.085, ELEVATION_CAVEAT, fontsize=6.5, color="0.35", ha="center", va="top")


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

    colourings = {
        "season": (day_of_year, SEASON_COLORMAP, (1, DAYS_PER_YEAR)),
        "elevation": (
            elevation, ELEVATION_COLORMAP,
            (float(np.nanmin(elevation)), float(np.nanmax(elevation))),
        ),
    }
    # Both projections of the same points, one above the other, because they
    # answer different questions: PCA shows the directions carrying the most
    # variance and its axes are linear combinations of real fields, while UMAP
    # shows neighbourhood structure that PCA's two components cannot reach.
    # Agreement between them is evidence; structure in only one is a property of
    # that algorithm.
    rows = [
        ("season", "pca2", "PCA"), ("season", "umap2", "UMAP"),
        ("elevation", "pca2", "PCA"), ("elevation", "umap2", "UMAP"),
    ]

    figure = plt.figure(figsize=(3.0 + 2.6 * len(views), 11.5))
    grid = GridSpec(
        len(rows), 1 + len(views), figure=figure,
        width_ratios=[0.85] + [1.0] * len(views), wspace=0.08, hspace=0.14,
    )

    draw_season_key(figure.add_subplot(grid[0:2, 0]))
    draw_elevation_key(figure.add_subplot(grid[2:4, 0]), surface_height, ice_mask)

    for row, (colouring, projection, label) in enumerate(rows):
        colours, colormap, limits = colourings[colouring]
        for column, view in enumerate(views):
            axes = figure.add_subplot(grid[row, 1 + column])
            draw_embedding(
                axes,
                projections[f"{view}_{projection}"][order],
                colours[order],
                colormap,
                limits,
                VIEW_TITLES.get(view, view) if row == 0 else None,
            )
            if column == 0:
                axes.set_ylabel(label, fontsize=10)

    figure.suptitle(
        "A 155-field point-wise autoencoder: what survives the bottleneck",
        fontsize=12,
    )
    figure.text(0.01, 0.005, source_note, fontsize=7, color="0.35", va="bottom")
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output_path


DIVERGING = LinearSegmentedColormap.from_list(
    "anomaly", ["#2a78d6", "#f0efec", "#e34948"]
)

DEFAULT_EXAMPLE_FIELDS = ["SMB", "ME_L00", "ST", "AL"]


def to_grid(values, cell_index, grid_shape):
    field = np.full(int(np.prod(grid_shape)), np.nan)
    field[cell_index] = values
    return field.reshape(grid_shape)


def plot_reconstructions(
    examples, grid_shape, latent_dims, fields, day, output_path: Path, source_note: str
) -> Path:
    """Truth beside every reconstruction, in physical units, on the model grid.

    An explained-variance number says how much of the matrix survived; it does
    not say whether what survived still looks like an ice sheet. These maps are
    the check that a summary statistic cannot make.
    """
    names = [str(n) for n in examples["field_names"]]
    on_day = examples["day_of_year"] == day
    cells = examples["cell_index"][on_day]
    held_out = examples["is_validation"][on_day]

    methods = [("truth", "truth")]
    for latent_dim in latent_dims:
        methods += [(f"pca{latent_dim}", f"PCA({latent_dim})"),
                    (f"autoencoder{latent_dim}", f"AE({latent_dim})")]
    methods = [(key, label) for key, label in methods if key in examples]

    figure, axes_grid = plt.subplots(
        len(fields), len(methods),
        figsize=(1.55 * len(methods) + 1.0, 2.5 * len(fields)), squeeze=False,
    )
    for row, field in enumerate(fields):
        column_index = names.index(field)
        truth = examples["truth"][on_day, column_index]
        # Percentile limits, not min/max. SMB on a melt day runs to -60 mmWE at a
        # handful of margin cells, and scaling to that leaves the whole interior
        # one flat colour in every panel -- which would hide exactly the
        # differences these maps exist to show. Limits are still identical across
        # a row, so the columns remain directly comparable.
        low, high = np.nanpercentile(truth, [1.0, 99.0])
        if truth.min() < 0 and truth.max() > 0:
            span = float(max(abs(low), abs(high)))
            limits, colormap = (-span, span), DIVERGING
        else:
            limits, colormap = (float(low), float(high)), ELEVATION_COLORMAP

        for column, (key, label) in enumerate(methods):
            axes = axes_grid[row][column]
            values = examples[key][on_day, column_index]
            image = axes.imshow(
                to_grid(values, cells, grid_shape), origin="lower", cmap=colormap,
                vmin=limits[0], vmax=limits[1], aspect="equal",
            )
            axes.axis("off")
            if row == 0:
                axes.set_title(label, fontsize=9)
            if column == 0:
                axes.text(
                    -0.08, 0.5, field, transform=axes.transAxes, fontsize=10,
                    rotation=90, va="center", ha="right",
                )
                bar = figure.colorbar(image, ax=axes_grid[row], fraction=0.02, pad=0.01)
                bar.ax.tick_params(labelsize=6)
            else:
                # Error on held-out cells only. Roughly 90 % of any day was seen
                # in training, so a whole-map error would flatter every model.
                # Written with text() rather than set_xlabel because axis("off")
                # above suppresses the label along with the ticks and spines.
                error = values[held_out] - truth[held_out]
                axes.text(
                    0.5, -0.01, f"RMSE {np.sqrt(np.mean(error**2)):.3g}",
                    transform=axes.transAxes, fontsize=7, color="0.3",
                    ha="center", va="top",
                )

    figure.suptitle(
        f"Reconstruction on day {day} of {int(examples['example_year'])}: "
        f"colour limits fixed per row (1-99th pct), RMSE on held-out cells only",
        fontsize=11,
    )
    figure.text(0.01, 0.01, source_note, fontsize=7, color="0.35", va="top")
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
    parser.add_argument(
        "--example-fields", nargs="+", default=DEFAULT_EXAMPLE_FIELDS,
        help="fields to draw reconstruction maps for",
    )
    parser.add_argument(
        "--example-day", type=int, default=None,
        help="day of year for the reconstruction maps; default is the middle saved day",
    )
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
    written = [structure, variance]

    example_path = arguments.run_dir / "reconstructions.npz"
    if example_path.exists():
        examples = np.load(example_path, allow_pickle=False)
        saved_days = [int(v) for v in examples["example_days"]]
        day = arguments.example_day or saved_days[len(saved_days) // 2]
        if day not in saved_days:
            raise SystemExit(f"day {day} not saved; this run has {saved_days}")
        written.append(plot_reconstructions(
            examples, geometry[0].shape, results["latent_dims"],
            resolve_example_fields(arguments.example_fields,
                                   [str(n) for n in examples["field_names"]]),
            day, output_dir / f"reconstruction_day{day}.png", source_note,
        ))
    else:
        print(f"no reconstructions.npz in {arguments.run_dir}; skipping example maps")

    print("Wrote " + "\n      ".join(str(path) for path in written))


if __name__ == "__main__":
    main()
