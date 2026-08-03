#!/usr/bin/env python3
"""One figure per variable: the 2D views coloured by that variable's value.

The latent_structure figure colours the embedding by day of year and by surface
elevation, which asks whether the latent space recovered those two. This asks
the complementary question, for all 155 columns at once: which variables does
the latent space actually organise itself by?

Each figure is a grid -- PCA and UMAP down the rows, the raw input and every
latent width across the columns -- with every point coloured by the variable's
physical value. A variable that paints a smooth gradient across the embedding is
one the bottleneck kept. A variable that paints noise is one it discarded, and
that is worth knowing before anybody reads the latent space as physics.

Colour limits are the 1st-99th percentile of the variable over the plotted
points, identical across every panel of a figure, so the panels compare.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

SEQUENTIAL_COLORMAP = "cividis"
DIVERGING_COLORMAP = "coolwarm"

VIEW_TITLES = {
    "input": "input (155 fields)",
    "latent15": "latent 15",
    "latent30": "latent 30",
    "latent100": "latent 100",
}


def draw(values, projections, views, field, output_path: Path, source_note: str, seed: int):
    # Shared shuffle: the rows arrive grouped by timestep, so drawing in order
    # puts whichever season was written last on top of everything else.
    order = np.random.default_rng(seed).permutation(len(values))
    low, high = np.percentile(values, [1.0, 99.0])
    if low == high:
        low, high = float(values.min()), float(values.max()) or 1.0
    # Zero keeps its meaning for signed fields -- 0 C is the melt threshold, 0
    # mmWE/day separates accumulation from ablation -- so a diverging ramp is
    # centred there. TwoSlopeNorm rather than symmetric limits: surface
    # temperature spans -45..0 C, and padding it to +/-45 would spend half the
    # colour range on values that never occur.
    signed = low < 0 < high
    norm = TwoSlopeNorm(vcenter=0.0, vmin=low, vmax=high) if signed else None

    rows = ["pca2", "umap2"]
    figure, axes_grid = plt.subplots(
        len(rows), len(views), figsize=(2.7 * len(views) + 1.4, 2.9 * len(rows) + 0.9),
        squeeze=False,
    )
    for row, projection in enumerate(rows):
        for column, view in enumerate(views):
            axes = axes_grid[row][column]
            points = projections[f"{view}_{projection}"][order]
            scatter = axes.scatter(
                points[:, 0], points[:, 1], c=values[order],
                cmap=DIVERGING_COLORMAP if signed else SEQUENTIAL_COLORMAP,
                norm=norm, vmin=None if signed else low, vmax=None if signed else high,
                s=0.5, alpha=0.5, linewidths=0, rasterized=True,
            )
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_aspect("equal", adjustable="datalim")
            if row == 0:
                axes.set_title(VIEW_TITLES.get(view, view), fontsize=9)
            if column == 0:
                axes.set_ylabel(projection.removesuffix("2").upper(), fontsize=10)

    bar = figure.colorbar(scatter, ax=axes_grid, fraction=0.02, pad=0.01)
    bar.ax.tick_params(labelsize=7)
    bar.set_alpha(1.0)

    figure.suptitle(f"2D views coloured by {field}", fontsize=12)
    figure.text(0.01, 0.005, source_note, fontsize=7, color="0.35", va="top")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--training-data", type=Path, default=Path("datasets/mar/mar_training_2000_2009.npz")
    )
    parser.add_argument("--fields", nargs="+", default=None, help="default: every column")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    projections = np.load(arguments.run_dir / "projections.npz", allow_pickle=False)
    with open(arguments.run_dir / "results.json") as handle:
        results = json.load(handle)
    field_names = [str(name) for name in results["field_names"]]

    views = ["input"] + [f"latent{d}" for d in results["latent_dims"]]
    missing = [v for v in views if f"{v}_umap2" not in projections]
    if missing:
        raise SystemExit(f"projections.npz has no UMAP for {missing}; rerun without --skip-umap")

    # The plotted points are validation rows, subsampled again for the scatter.
    # Recovering their physical values means composing both index arrays, which
    # is why they are both saved.
    archive = np.load(arguments.training_data, allow_pickle=False)
    if archive["features"].shape[0] != results["n_samples"]:
        raise SystemExit(
            "this run subsampled with --max-samples, so its saved indices cannot be "
            "mapped back to the training matrix; rerun the job without --max-samples"
        )
    plotted = archive["features"][projections["validation_index"][projections["plot_index"]]]

    fields = arguments.fields or field_names
    unknown = [name for name in fields if name not in field_names]
    if unknown:
        raise SystemExit(f"not columns of this run: {', '.join(unknown)}")

    source_note = (
        "Source: MAR v3.2 regional climate model, NCEPv1-forced, 20 km, 2000-2009. "
        "MODEL OUTPUT, NOT OBSERVATION.\n"
        f"Normalization {results.get('normalization', 'zscore')}; colours are physical "
        "units, limits 1st-99th pct, identical across panels. UMAP preserves "
        "neighbourhoods only.\n"
        f"Run {arguments.run_dir.name}."
    )

    directory = arguments.run_dir / "latent_by_variable"
    directory.mkdir(parents=True, exist_ok=True)
    for field in fields:
        draw(
            plotted[:, field_names.index(field)].astype(np.float64), projections, views,
            field, directory / f"{field}.png", source_note, arguments.seed,
        )
        print(f"  {field}", flush=True)

    print(f"\nWrote {len(fields)} figures to {directory}")


if __name__ == "__main__":
    main()
