#!/usr/bin/env python3
"""What actually backs each year of the 1940-2024 frame.

Renders the provenance table in greenland_art.data.coverage as a timeline, so
the observed span and the span that must be reconstructed are visible rather
than buried in a caption.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from greenland_art.data.coverage import (
    FRAME_FIRST_YEAR,
    FRAME_LAST_YEAR,
    SOURCES,
    TIER_OBSERVED,
    TIER_RECONSTRUCTED,
    TIER_STATIC,
    ice_observation_gap,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

TIER_COLOR = {
    TIER_OBSERVED: "#2a78d6",
    TIER_STATIC: "#eda100",
    TIER_RECONSTRUCTED: "#e34948",
}
TIER_LABEL = {
    TIER_OBSERVED: "observed",
    TIER_STATIC: "static composite (no time axis)",
    TIER_RECONSTRUCTED: "published reconstruction",
}


def build_figure(output_path: Path) -> Path:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": BASELINE,
        }
    )

    figure, axis = plt.subplots(figsize=(14.5, 6.4))
    gap_start, gap_end, gap_years = ice_observation_gap()

    axis.axvspan(
        gap_start - 0.5, gap_end + 0.5,
        facecolor="#e34948", alpha=0.07, zorder=0,
    )
    axis.annotate(
        f"no gridded ice observation of any kind\n{gap_start}–{gap_end}  ({gap_years} of "
        f"{FRAME_LAST_YEAR - FRAME_FIRST_YEAR + 1} years)",
        xy=((gap_start + gap_end) / 2, len(SOURCES) + 0.35),
        ha="center", va="bottom", fontsize=9, color="#b3302f", linespacing=1.4,
    )

    for row_index, source in enumerate(reversed(SOURCES)):
        start = max(source.first_year, FRAME_FIRST_YEAR)
        end = min(source.last_year, FRAME_LAST_YEAR)
        colour = TIER_COLOR[source.tier]

        if source.tier == TIER_STATIC:
            axis.plot(
                [start, end], [row_index, row_index],
                color=colour, linewidth=2.2, linestyle=(0, (2, 2)), zorder=3,
            )
        else:
            axis.barh(
                row_index, end - start + 1, left=start - 0.5, height=0.5,
                color=colour, edgecolor=SURFACE, linewidth=1.2, zorder=3,
            )

        axis.text(
            FRAME_FIRST_YEAR - 1.5, row_index, source.name,
            ha="right", va="center", fontsize=9, color=INK_PRIMARY,
        )
        axis.text(
            FRAME_LAST_YEAR + 1.5, row_index, source.quantity,
            ha="left", va="center", fontsize=8, color=INK_MUTED,
        )

    axis.set_yticks([])
    axis.set_ylim(-0.9, len(SOURCES) + 1.5)
    axis.set_xlim(FRAME_FIRST_YEAR - 1, FRAME_LAST_YEAR + 1)
    axis.set_xlabel("year CE", fontsize=9, color=INK_SECONDARY)
    axis.tick_params(labelsize=8, colors=INK_MUTED)
    axis.grid(axis="x", color=GRIDLINE, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)

    axis.legend(
        handles=[
            Patch(facecolor=TIER_COLOR[tier], label=TIER_LABEL[tier])
            for tier in (TIER_OBSERVED, TIER_STATIC, TIER_RECONSTRUCTED)
        ],
        loc="lower left", bbox_to_anchor=(0.0, -0.30), ncol=3,
        frameon=False, fontsize=8.5,
    )

    figure.suptitle(
        "What actually backs each year of a 1940–2024 Greenland frame",
        fontsize=13.5, x=0.012, ha="left", y=1.045,
    )
    figure.text(
        0.012, 0.905,
        "BedMachine is one static map, not a time series: its 1993–2021 span is the "
        "years of its input observations. Nothing gridded\nconstrains ice before 1992, so "
        "any 1940-onward ice product is reconstruction for most of its length.",
        fontsize=9, color=INK_SECONDARY, linespacing=1.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "coverage_1940_2024.png")
    print(f"Wrote {build_figure(parser.parse_args().output)}")


if __name__ == "__main__":
    main()
