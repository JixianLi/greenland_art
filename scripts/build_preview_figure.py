#!/usr/bin/env python3
"""Static preview figure: timeline selection driving a spatial view.

Left panel is the spatial view (ten Osman 2021 core sites on Greenland in
EPSG:3413). Right panels are the temporal view (eight GISP2 chemical species
as a co-registered field matrix, plus the volcanic sulfate validation track).
The shaded band is the selected window; the map shows that window's d18O
anomaly. Interactive brushing of the same linkage is in build_preview_app.py.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from greenland_art.analysis.anomaly import multifield_novelty, standardize_fields
from greenland_art.analysis.anomaly import window_anomaly
from greenland_art.data import gisp2, osman2021
from greenland_art.visualization.greenland_map import (
    dodge_labels_vertically,
    load_greenland_rings,
    project_site_metadata,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
DIVERGING_COLD = "#2a78d6"
DIVERGING_NEUTRAL = "#f0efec"
DIVERGING_WARM = "#e34948"

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "cold_warm", [DIVERGING_COLD, DIVERGING_NEUTRAL, DIVERGING_WARM]
)

RECORD_WINDOW = (1750, 1990)
BASELINE_WINDOW = (1750, 1950)

# Independently dated eruptions used to check the chemistry against reality.
# Ice layer dates lag the eruption by the atmospheric transport and deposition
# time, so a one-to-two-year offset in the record is expected, not an error.
KNOWN_ERUPTIONS = [
    (1783, "Laki"),
    (1815, "Tambora"),
    (1912, "Katmai"),
]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "axes.linewidth": 0.8,
        }
    )


def draw_map(axis, site_anomaly, selection_window, d18o_by_site) -> None:
    for ring in load_greenland_rings():
        axis.fill(ring[:, 0], ring[:, 1], facecolor="#efeee9", edgecolor="none", zorder=1)
        axis.plot(ring[:, 0], ring[:, 1], color=BASELINE, linewidth=0.8, zorder=2)

    metadata = project_site_metadata(osman2021.load_site_metadata())
    metadata["anomaly"] = metadata["site"].map(site_anomaly)

    measured = metadata.dropna(subset=["anomaly"])
    unmeasured = metadata[metadata["anomaly"].isna()]

    limit = float(np.nanmax(np.abs(measured["anomaly"]))) if len(measured) else 1.0
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    axis.scatter(
        unmeasured["x_km"],
        unmeasured["y_km"],
        s=110,
        facecolor="#ffffff",
        edgecolor=INK_MUTED,
        linewidth=1.2,
        zorder=3,
    )
    marks = axis.scatter(
        measured["x_km"],
        measured["y_km"],
        c=measured["anomaly"],
        cmap=DIVERGING_CMAP,
        norm=norm,
        s=190,
        edgecolor=SURFACE,
        linewidth=2.0,
        zorder=4,
    )

    label_y = dodge_labels_vertically(metadata["y_km"].to_numpy(), minimum_separation_km=118.0)
    for (_, row), text_y in zip(metadata.iterrows(), label_y):
        value = row["anomaly"]
        if np.isfinite(value):
            text = f"{row['site']}  {value:+.2f}‰"
        elif row["site"] not in d18o_by_site.columns:
            text = f"{row['site']}  (accumulation only)"
        else:
            # The site-level last_year covers all variables; report the end of
            # the d18O record specifically, which can be earlier (Eurocore's
            # accumulation runs to 1763 but its d18O stops at 1741).
            d18o_end = int(d18o_by_site[row["site"]].dropna().index.max())
            text = f"{row['site']}  (δ18O ends {d18o_end})"
        axis.annotate(
            text,
            xy=(row["x_km"], row["y_km"]),
            xytext=(row["x_km"] + 105, text_y),
            fontsize=7.5,
            color=INK_SECONDARY,
            va="center",
            ha="left",
            arrowprops={"arrowstyle": "-", "color": BASELINE, "linewidth": 0.7},
            zorder=5,
        )

    colorbar = axis.figure.colorbar(
        marks, ax=axis, orientation="horizontal", fraction=0.045, pad=0.09, aspect=32
    )
    colorbar.set_label(
        f"$\\delta^{{18}}$O anomaly, {selection_window[0]}–{selection_window[1]} "
        f"vs {BASELINE_WINDOW[0]}–{BASELINE_WINDOW[1]} (‰)",
        fontsize=8,
        color=INK_SECONDARY,
    )
    colorbar.ax.tick_params(labelsize=7, color=BASELINE)
    colorbar.outline.set_visible(False)

    axis.set_aspect("equal")
    axis.set_xlim(-720, 1080)
    axis.set_ylim(-3320, -950)
    axis.set_xlabel("EPSG:3413 easting (km)", fontsize=8)
    axis.set_ylabel("EPSG:3413 northing (km)", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.set_title(
        "Where the ice cores are, and what the selected window did there",
        fontsize=10.5,
        color=INK_PRIMARY,
        loc="left",
        pad=10,
    )


def draw_field_matrix(axis, standardized, selection_window) -> None:
    limit = float(np.nanpercentile(np.abs(standardized.to_numpy()), 98))
    years = standardized.index.to_numpy()

    # shading="nearest" lets the irregular bi-yearly sample spacing set each
    # cell's width from the midpoints, rather than pretending the record is
    # evenly sampled.
    axis.pcolormesh(
        years,
        np.arange(standardized.shape[1]),
        standardized.to_numpy().T,
        cmap=DIVERGING_CMAP,
        vmin=-limit,
        vmax=limit,
        shading="nearest",
    )
    axis.set_yticks(np.arange(standardized.shape[1]))
    axis.set_yticklabels(standardized.columns, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(*RECORD_WINDOW)
    axis.tick_params(labelsize=7)
    axis.set_ylabel("GISP2 ion species", fontsize=8)
    axis.set_title(
        "Eight co-registered chemical fields — the multi-field matrix an autoencoder would consume",
        fontsize=10.5,
        color=INK_PRIMARY,
        loc="left",
        pad=8,
    )
    _mark_selection(axis, selection_window)


def draw_novelty(axis, novelty, selection_window) -> None:
    axis.fill_between(
        novelty.index, 0, novelty.to_numpy(), color=DIVERGING_COLD, alpha=0.18, linewidth=0
    )
    axis.plot(novelty.index, novelty.to_numpy(), color=DIVERGING_COLD, linewidth=1.6)
    axis.set_xlim(*RECORD_WINDOW)
    axis.set_ylim(bottom=0)
    axis.set_ylabel("novelty\n(std. dev.)", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.grid(axis="y", zorder=0)
    axis.set_title(
        "Unsupervised novelty score — the track that proposes which windows to select",
        fontsize=10.5,
        color=INK_PRIMARY,
        loc="left",
        pad=8,
    )
    _mark_selection(axis, selection_window)


def draw_volcanic_validation(axis, volcanic, selection_window) -> None:
    window = volcanic[
        (volcanic["year_ce"] >= RECORD_WINDOW[0]) & (volcanic["year_ce"] <= RECORD_WINDOW[1])
    ]
    axis.plot(
        window["year_ce"],
        window["total_sulfate_ppb"],
        color=INK_MUTED,
        linewidth=0.9,
        zorder=2,
    )
    axis.fill_between(
        window["year_ce"],
        0,
        window["volcanic_sulfate_ppb"],
        color=DIVERGING_WARM,
        alpha=0.85,
        linewidth=0,
        zorder=3,
    )

    axis.set_xlim(*RECORD_WINDOW)
    axis.set_ylim(0, float(window["total_sulfate_ppb"].max()) * 1.32)

    for year, name in KNOWN_ERUPTIONS:
        nearby = window[window["year_ce"].between(year - 3, year + 3)]
        peak = nearby.loc[nearby["volcanic_sulfate_ppb"].idxmax()]
        axis.annotate(
            f"{name} {year}",
            xy=(peak["year_ce"], peak["volcanic_sulfate_ppb"]),
            xytext=(peak["year_ce"], peak["volcanic_sulfate_ppb"] + 62),
            ha="center",
            fontsize=8,
            color=INK_PRIMARY,
            arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
            zorder=7,
        )
    axis.set_ylabel("sulfate\n(ppb)", fontsize=8)
    axis.set_xlabel("year CE", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.grid(axis="y", zorder=0)
    axis.set_title(
        "Ground truth — red is volcanic sulfate; the labelled eruptions were not used to fit anything",
        fontsize=10.5,
        color=INK_PRIMARY,
        loc="left",
        pad=8,
    )
    _mark_selection(axis, selection_window)


def _mark_selection(axis, selection_window) -> None:
    axis.axvspan(
        selection_window[0],
        selection_window[1],
        facecolor="none",
        edgecolor=INK_PRIMARY,
        linewidth=1.4,
        zorder=6,
    )


def build_figure(selection_window: tuple[int, int], output_path: Path) -> Path:
    configure_matplotlib()

    records = osman2021.load_site_records()
    d18o_by_site = osman2021.pivot_variable(records, "d18o")
    site_anomaly = window_anomaly(d18o_by_site, selection_window, BASELINE_WINDOW)

    field_matrix = gisp2.build_multifield_matrix(*RECORD_WINDOW)
    standardized = standardize_fields(field_matrix)
    novelty = multifield_novelty(field_matrix)
    volcanic = gisp2.load_volcanic_sulfate()

    figure = plt.figure(figsize=(17.5, 9.6))
    grid = figure.add_gridspec(
        3, 2, width_ratios=[1.0, 1.45], height_ratios=[1.35, 0.8, 0.9], hspace=0.42, wspace=0.16
    )

    map_axis = figure.add_subplot(grid[:, 0])
    draw_map(map_axis, site_anomaly, selection_window, d18o_by_site)

    matrix_axis = figure.add_subplot(grid[0, 1])
    draw_field_matrix(matrix_axis, standardized, selection_window)

    novelty_axis = figure.add_subplot(grid[1, 1], sharex=matrix_axis)
    draw_novelty(novelty_axis, novelty, selection_window)

    volcanic_axis = figure.add_subplot(grid[2, 1], sharex=matrix_axis)
    draw_volcanic_validation(volcanic_axis, volcanic, selection_window)

    figure.suptitle(
        f"Greenland ice cores: selecting {selection_window[0]}–{selection_window[1]} "
        "on the timeline, and reading it in space",
        fontsize=14,
        color=INK_PRIMARY,
        x=0.008,
        ha="left",
        y=0.985,
    )
    figure.text(
        0.008,
        0.012,
        "Sources: Osman et al. 2021 (PNAS) ten-site Greenland array, annual d18O; "
        "Mayewski et al. 1997 GISP2 B-core major ions; Zielinski et al. 1994 GISP2 volcanic sulfate. "
        "All records are observational — nothing here is simulated or synthetic.",
        fontsize=7.5,
        color=INK_MUTED,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1783)
    parser.add_argument("--end-year", type=int, default=1793)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "greenland_preview.png")
    arguments = parser.parse_args()

    path = build_figure((arguments.start_year, arguments.end_year), arguments.output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
