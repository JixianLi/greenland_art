#!/usr/bin/env python3
"""Atlas of every gridded field currently held, on one projection.

Six fields from three products co-registered onto the common 20 km grid, drawn
in EPSG:3413 with the ice core sites overlaid. This is the multi-field stack a
autoencoder would consume, shown as maps rather than as an array.

The Box grid is regular polar stereographic but rotated relative to EPSG:3413
(different central meridian), so the panels are drawn with pcolormesh on the
projected 2D coordinates rather than imshow on grid indices, which would shear
the outline.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from greenland_art.analysis import smb_emulator as se
from greenland_art.data import field_stack as fs
from greenland_art.data import osman2021
from greenland_art.visualization.greenland_map import (
    load_greenland_rings,
    project_site_metadata,
)
from greenland_art.visualization.projection import project_to_kilometres

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"
COLD = "#2a78d6"
NEUTRAL = "#f0efec"
WARM = "#e34948"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#eef4fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("cold_warm", [COLD, NEUTRAL, WARM])

# field -> (title, units, colour job, diverging centre or None)
PANELS = [
    ("thickness_m", "Ice thickness", "m", "sequential", None),
    ("bed_m", "Bed elevation", "m", "diverging", 0.0),
    ("surface_m", "Surface elevation", "m", "sequential", None),
    ("smb_kg_m2", "Surface mass balance, 1940–2012 mean", "kg m⁻² yr⁻¹", "diverging", 0.0),
    ("summer_temperature_c", "Summer air temperature, 1940–2012 mean", "°C", "diverging", 0.0),
    ("annual_precipitation_m", "Annual precipitation, 1940–2012 mean", "m yr⁻¹", "sequential", None),
]

SOURCE_OF = {
    "thickness_m": "BedMachine v6",
    "bed_m": "BedMachine v6",
    "surface_m": "BedMachine v6",
    "smb_kg_m2": "Box 2013",
    "summer_temperature_c": "ERA5",
    "annual_precipitation_m": "ERA5",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE, "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK_PRIMARY, "axes.edgecolor": BASELINE,
        }
    )


def draw_panel(axis, x_km, y_km, values, title, units, colour_job, centre, rings, sites=None):
    finite = values[np.isfinite(values)]
    if colour_job == "diverging":
        # Asymmetric on purpose. Bed elevation runs -1100 to +2400 m and summer
        # temperature -15.5 to +4.5 C; a symmetric span around the centre would
        # leave most of one arm unused. The centre stays at the physically
        # meaningful value (sea level, melting point), so the neutral colour
        # still means what it should.
        low = float(np.nanpercentile(finite, 1))
        high = float(np.nanpercentile(finite, 99))
        norm = TwoSlopeNorm(
            vmin=min(low, centre - 1e-6), vcenter=centre, vmax=max(high, centre + 1e-6)
        )
        cmap = DIVERGING
    else:
        norm = mpl.colors.Normalize(
            vmin=float(np.nanpercentile(finite, 1)), vmax=float(np.nanpercentile(finite, 99))
        )
        cmap = SEQUENTIAL

    for ring in rings:
        axis.plot(ring[:, 0], ring[:, 1], color=BASELINE, linewidth=0.6, zorder=1)

    mesh = axis.pcolormesh(x_km, y_km, values, cmap=cmap, norm=norm, shading="nearest", zorder=2)

    if sites is not None:
        axis.scatter(
            sites["x_km"], sites["y_km"], s=26, facecolor="none",
            edgecolor=INK_PRIMARY, linewidth=1.0, zorder=4,
        )

    axis.set_aspect("equal")
    axis.set_xlim(-750, 1100)
    axis.set_ylim(-3400, -600)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title, fontsize=10, loc="left", pad=6, color=INK_PRIMARY)

    bar = axis.figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.02)
    bar.set_label(units, fontsize=8, color=INK_SECONDARY)
    bar.ax.tick_params(labelsize=7, color=BASELINE)
    bar.outline.set_visible(False)


def build_figure(output_path: Path) -> Path:
    configure()

    stack = fs.build_field_stack()
    box = se.load_box_annual()
    x_km, y_km = project_to_kilometres(box["longitude"].values, box["latitude"].values)
    rings = load_greenland_rings()
    sites = project_site_metadata(osman2021.load_site_metadata())

    figure, axes = plt.subplots(2, 3, figsize=(15.0, 12.4))
    for axis, (field, title, units, colour_job, centre) in zip(axes.ravel(), PANELS):
        draw_panel(
            axis, x_km, y_km, stack[field].values, title, units, colour_job, centre,
            rings, sites if field == "thickness_m" else None,
        )
        axis.text(
            0.02, 0.02, SOURCE_OF[field], transform=axis.transAxes,
            fontsize=7.5, color=INK_MUTED, ha="left", va="bottom",
        )

    matrix, _ = fs.stack_to_matrix(stack, [p[0] for p in PANELS])
    figure.suptitle(
        "Every gridded field currently held, co-registered on one 20 km EPSG:3413 grid",
        fontsize=14, x=0.012, ha="left", y=1.055,
    )
    figure.text(
        0.012, 0.998,
        f"Three products on three native grids — BedMachine v6 at 150 m, Box (2013) at 5 km, "
        f"ERA5 at 0.25° — resampled to a common mesh. Open circles on the first panel are the ten "
        f"Osman ice core sites.\nTogether these form a dense {matrix.shape[0]:,} × {matrix.shape[1]} "
        "matrix: one row per ice cell, one column per field. That array is the input a multi-field "
        "autoencoder would consume.",
        fontsize=9, color=INK_SECONDARY, linespacing=1.55,
    )
    figure.text(
        0.012, 0.012,
        "All fields observational or reconstruction — none synthetic. BedMachine is a static composite; "
        "SMB is the Box reconstruction; ERA5 is reanalysis. See outputs/coverage_1940_2024.png for per-year provenance.",
        fontsize=7.5, color=INK_MUTED,
    )

    figure.subplots_adjust(top=0.945, bottom=0.035, hspace=0.02, wspace=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "field_atlas.png")
    print(f"Wrote {build_figure(parser.parse_args().output)}")


if __name__ == "__main__":
    main()
