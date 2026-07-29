#!/usr/bin/env python3
"""Interactive preview: brush the timeline, the Greenland map responds.

The brush-to-map linkage is implemented in client-side CustomJS rather than a
Python callback, so the exported HTML is genuinely interactive with no server
and no Python on the viewer's machine. A HoloViews DynamicMap would be less
code, but its callback needs a live Python process; exporting one produces a
file that looks interactive and is not.

Colour limits on the map are fixed across all selections rather than rescaled
per window, so two different windows can actually be compared.
"""

import argparse
from pathlib import Path

import numpy as np
from bokeh.events import SelectionGeometry
from bokeh.layouts import column, row
from bokeh.models import (
    BoxAnnotation,
    BoxSelectTool,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    LinearColorMapper,
)
from bokeh.plotting import figure, output_file, save

from greenland_art.analysis.anomaly import (
    multifield_novelty,
    standardize_fields,
    window_anomaly,
)
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
BASELINE_COLOR = "#c3c2b7"
LAND_FILL = "#efeee9"
DIVERGING_COLD = "#2a78d6"
DIVERGING_WARM = "#e34948"
DIVERGING_RAMP = [
    "#2a78d6", "#6da7ec", "#9ec5f4", "#cde2fb",
    "#f0efec",
    "#f8cfce", "#f2a3a2", "#ea7675", "#e34948",
]

RECORD_WINDOW = (1750, 1990)
BASELINE_WINDOW = (1750, 1950)
DEFAULT_SELECTION = (1783.0, 1793.0)

MIN_WINDOW_SAMPLES = 3
# Permil. Set from the observed spread of decadal anomalies across this array
# (98th percentile of |anomaly| = 1.55), so the ramp is not flattened by the one
# +2.0 outlier. Fixed rather than per-window: rescaling on every brush would make
# two selections look alike when they are not.
ANOMALY_COLOR_LIMIT = 1.6

TOOLBAR = "pan,wheel_zoom,box_zoom,reset,save"


def _style_axes(plot) -> None:
    plot.background_fill_color = SURFACE
    plot.border_fill_color = SURFACE
    plot.outline_line_color = None
    plot.title.text_color = INK_PRIMARY
    plot.title.text_font_size = "11px"
    plot.title.text_font_style = "normal"
    for axis in (plot.xaxis, plot.yaxis):
        axis.axis_line_color = BASELINE_COLOR
        axis.major_tick_line_color = BASELINE_COLOR
        axis.minor_tick_line_color = None
        axis.major_label_text_color = INK_MUTED
        axis.axis_label_text_color = INK_SECONDARY
        axis.axis_label_text_font_style = "normal"
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = GRIDLINE


def _cell_edges(values: np.ndarray) -> np.ndarray:
    """Midpoint widths for irregularly spaced samples."""
    midpoints = (values[:-1] + values[1:]) / 2
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def build_document():
    records = osman2021.load_site_records()
    d18o_by_site = osman2021.pivot_variable(records, "d18o")
    site_metadata = project_site_metadata(osman2021.load_site_metadata())

    baseline_slice = d18o_by_site.loc[BASELINE_WINDOW[0] : BASELINE_WINDOW[1]]
    site_names = [
        name
        for name in d18o_by_site.columns
        if baseline_slice[name].count() >= 30
    ]

    annual_data = {"year": d18o_by_site.index.to_numpy().astype(float).tolist()}
    for name in site_names:
        column_values = d18o_by_site[name].to_numpy(dtype=float)
        annual_data[name] = [None if np.isnan(v) else float(v) for v in column_values]
    annual_source = ColumnDataSource(annual_data)

    baselines = {name: float(baseline_slice[name].mean()) for name in site_names}

    # Seed the map with the default window rather than NaN: an unseeded source
    # renders every marker blank while the title claims a window is selected.
    initial_anomaly = window_anomaly(
        d18o_by_site,
        (int(DEFAULT_SELECTION[0]), int(DEFAULT_SELECTION[1])),
        BASELINE_WINDOW,
        min_window_samples=MIN_WINDOW_SAMPLES,
    )

    ordered = site_metadata.set_index("site")
    label_y = dodge_labels_vertically(
        np.array([float(ordered.loc[n, "y_km"]) for n in site_metadata["site"]]),
        minimum_separation_km=118.0,
    )
    label_y_by_site = dict(zip(site_metadata["site"], label_y))

    site_source = ColumnDataSource(
        {
            "site": list(site_names),
            "x_km": [float(ordered.loc[n, "x_km"]) for n in site_names],
            "y_km": [float(ordered.loc[n, "y_km"]) for n in site_names],
            "label_y": [label_y_by_site[n] for n in site_names],
            "anomaly": [float(initial_anomaly[n]) for n in site_names],
            "baseline": [baselines[n] for n in site_names],
        }
    )

    absent = site_metadata[~site_metadata["site"].isin(site_names)]
    absent_source = ColumnDataSource(
        {
            "site": absent["site"].tolist(),
            "x_km": absent["x_km"].astype(float).tolist(),
            "y_km": absent["y_km"].astype(float).tolist(),
            "label_y": [label_y_by_site[n] for n in absent["site"]],
        }
    )

    map_plot = _build_map(site_source, absent_source)
    heatmap_plot, selection_band_heatmap = _build_field_matrix()
    novelty_plot, selection_band_novelty = _build_novelty(heatmap_plot)
    volcanic_plot, selection_band_volcanic = _build_volcanic(heatmap_plot)

    callback = CustomJS(
        args={
            "annual": annual_source,
            "sites": site_source,
            "map_plot": map_plot,
            "bands": [
                selection_band_heatmap,
                selection_band_novelty,
                selection_band_volcanic,
            ],
            "min_samples": MIN_WINDOW_SAMPLES,
            "baseline_window": list(BASELINE_WINDOW),
        },
        code="""
        const geometry = cb_obj.geometry;
        if (geometry.x0 == null || geometry.x1 == null) { return; }
        // Snap to whole years. The records are annual, so a fractional-year
        // window is not a meaningful selection, and rounding only in the title
        // would make the label disagree with the numbers actually computed.
        const start = Math.round(Math.min(geometry.x0, geometry.x1));
        const end   = Math.round(Math.max(geometry.x0, geometry.x1));
        if (end - start < 1) { return; }

        const years = annual.data['year'];
        const names = sites.data['site'];
        const baselines = sites.data['baseline'];
        const anomalies = [];

        for (let s = 0; s < names.length; s++) {
            const series = annual.data[names[s]];
            let total = 0.0, count = 0;
            for (let i = 0; i < years.length; i++) {
                const value = series[i];
                if (years[i] >= start && years[i] <= end && value != null) {
                    total += value; count += 1;
                }
            }
            anomalies.push(count >= min_samples ? total / count - baselines[s] : NaN);
        }

        sites.data['anomaly'] = anomalies;
        sites.change.emit();

        for (const band of bands) { band.left = start; band.right = end; }
        map_plot.title.text =
            'delta-18O anomaly, ' + Math.round(start) + '-' + Math.round(end) +
            ' vs ' + baseline_window[0] + '-' + baseline_window[1];
        """,
    )
    heatmap_plot.js_on_event(SelectionGeometry, callback)

    header = Div(
        text=f"""
        <div style="font-family:system-ui,-apple-system,sans-serif;color:{INK_SECONDARY};
                    max-width:1500px;line-height:1.5">
          <h2 style="color:{INK_PRIMARY};margin:0 0 6px 0;font-weight:600">
            Greenland ice cores: a timeline that drives a map</h2>
          <p style="margin:0 0 8px 0">
            <b>Drag horizontally across the chemistry panel</b> (box-select is active) to
            choose a time window. Each core site's &delta;<sup>18</sup>O anomaly is
            recomputed for that window against its own
            {BASELINE_WINDOW[0]}&ndash;{BASELINE_WINDOW[1]} baseline, and the map repaints.
            Colour limits are fixed at &plusmn;{ANOMALY_COLOR_LIMIT}&permil; across all
            selections so two windows can be compared directly.
          </p>
          <p style="margin:0 0 8px 0">
            The novelty track is an unsupervised stand-in for a learned reconstruction
            error &mdash; it proposes windows, it does not conclude anything. The volcanic
            sulfate panel is independent ground truth: Laki 1783, Tambora 1815 and
            Katmai 1912 were never used to fit any part of this.
          </p>
          <p style="margin:0;color:{INK_MUTED};font-size:12px">
            Sources: Osman et al. 2021 (PNAS) ten-site Greenland array, annual
            &delta;<sup>18</sup>O &middot; Mayewski et al. 1997 GISP2 B-core major ions
            &middot; Zielinski et al. 1994 GISP2 volcanic sulfate. Every record shown is
            observational &mdash; nothing is simulated or synthetic.
          </p>
        </div>
        """,
        sizing_mode="stretch_width",
    )

    return column(
        header,
        row(
            map_plot,
            column(heatmap_plot, novelty_plot, volcanic_plot, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )


def _build_map(site_source, absent_source):
    plot = figure(
        width=640,
        height=760,
        match_aspect=True,
        x_range=(-720, 1150),
        y_range=(-3320, -950),
        tools=TOOLBAR,
        toolbar_location="above",
        title=(
            f"delta-18O anomaly, {DEFAULT_SELECTION[0]:.0f}-{DEFAULT_SELECTION[1]:.0f} "
            f"vs {BASELINE_WINDOW[0]}-{BASELINE_WINDOW[1]}"
        ),
        x_axis_label="EPSG:3413 easting (km)",
        y_axis_label="EPSG:3413 northing (km)",
    )
    _style_axes(plot)

    rings = load_greenland_rings()
    plot.patches(
        xs=[ring[:, 0].tolist() for ring in rings],
        ys=[ring[:, 1].tolist() for ring in rings],
        fill_color=LAND_FILL,
        line_color=BASELINE_COLOR,
        line_width=1.0,
    )

    mapper = LinearColorMapper(
        palette=DIVERGING_RAMP,
        low=-ANOMALY_COLOR_LIMIT,
        high=ANOMALY_COLOR_LIMIT,
        nan_color="#ffffff",
    )

    plot.scatter(
        x="x_km", y="y_km", source=absent_source,
        size=13, fill_color=SURFACE, line_color=INK_MUTED, line_width=1.4,
    )
    marks = plot.scatter(
        x="x_km", y="y_km", source=site_source,
        size=19,
        fill_color={"field": "anomaly", "transform": mapper},
        line_color=SURFACE, line_width=2,
    )
    for source, colour in ((site_source, INK_SECONDARY), (absent_source, INK_MUTED)):
        plot.segment(
            x0="x_km", y0="y_km", x1="x_km", y1="label_y",
            source=source, line_color=BASELINE_COLOR, line_width=0.7,
        )
        plot.text(
            x="x_km", y="label_y", text="site", source=source,
            x_offset=15, y_offset=5, text_font_size="9px", text_color=colour,
        )

    plot.add_tools(
        HoverTool(
            renderers=[marks],
            tooltips=[("site", "@site"), ("anomaly", "@anomaly{+0.000} per mil")],
        )
    )
    color_bar = ColorBar(
        color_mapper=mapper,
        title="delta-18O anomaly (per mil)",
        title_text_font_style="normal",
        title_text_color=INK_SECONDARY,
        major_label_text_color=INK_MUTED,
        background_fill_color=SURFACE,
        border_line_color=None,
        height=10,
        orientation="horizontal",
        location="bottom_center",
    )
    plot.add_layout(color_bar, "below")
    return plot


def _build_field_matrix():
    field_matrix = gisp2.build_multifield_matrix(*RECORD_WINDOW)
    standardized = standardize_fields(field_matrix)
    species = list(standardized.columns)
    years = standardized.index.to_numpy(dtype=float)
    edges = _cell_edges(years)
    widths = np.diff(edges)

    rows = []
    for species_index, name in enumerate(species):
        for year_index, year in enumerate(years):
            rows.append(
                {
                    "year": float(year),
                    "width": float(widths[year_index]),
                    "species": name,
                    "z": float(standardized.iloc[year_index, species_index]),
                    "raw": float(field_matrix.iloc[year_index, species_index]),
                }
            )
    source = ColumnDataSource({key: [r[key] for r in rows] for key in rows[0]})

    limit = float(np.nanpercentile(np.abs(standardized.to_numpy()), 98))
    mapper = LinearColorMapper(palette=DIVERGING_RAMP, low=-limit, high=limit)

    plot = figure(
        height=280,
        sizing_mode="stretch_width",
        x_range=RECORD_WINDOW,
        y_range=list(reversed(species)),
        tools=TOOLBAR,
        toolbar_location="above",
        title="Eight co-registered chemical fields — drag horizontally to select a window",
        x_axis_label="",
    )
    _style_axes(plot)
    plot.ygrid.grid_line_color = None

    glyphs = plot.rect(
        x="year", y="species", width="width", height=0.94,
        source=source, fill_color={"field": "z", "transform": mapper}, line_color=None,
    )
    plot.add_tools(
        HoverTool(
            renderers=[glyphs],
            tooltips=[
                ("year", "@year{0}"),
                ("species", "@species"),
                ("concentration", "@raw{0.00} ppb"),
                ("standardized", "@z{+0.00} sd"),
            ],
        )
    )

    box_select = BoxSelectTool(dimensions="width", persistent=True)
    plot.add_tools(box_select)
    plot.toolbar.active_drag = box_select

    band = _add_selection_band(plot)
    return plot, band


def _build_novelty(shared_x_plot):
    field_matrix = gisp2.build_multifield_matrix(*RECORD_WINDOW)
    novelty = multifield_novelty(field_matrix)

    plot = figure(
        height=170,
        sizing_mode="stretch_width",
        x_range=shared_x_plot.x_range,
        tools=TOOLBAR,
        toolbar_location=None,
        title="Unsupervised novelty score — where the multi-field state is unusual",
        y_axis_label="std. dev.",
    )
    _style_axes(plot)
    source = ColumnDataSource(
        {"year": novelty.index.to_numpy(dtype=float), "novelty": novelty.to_numpy()}
    )
    plot.varea(x="year", y1=0, y2="novelty", source=source, fill_color=DIVERGING_COLD, fill_alpha=0.18)
    line = plot.line(x="year", y="novelty", source=source, line_color=DIVERGING_COLD, line_width=1.8)
    plot.add_tools(
        HoverTool(
            renderers=[line], mode="vline",
            tooltips=[("year", "@year{0}"), ("novelty", "@novelty{0.00} sd")],
        )
    )
    return plot, _add_selection_band(plot)


def _build_volcanic(shared_x_plot):
    volcanic = gisp2.load_volcanic_sulfate()
    window = volcanic[volcanic["year_ce"].between(*RECORD_WINDOW)]

    plot = figure(
        height=190,
        sizing_mode="stretch_width",
        x_range=shared_x_plot.x_range,
        tools=TOOLBAR,
        toolbar_location=None,
        title="Ground truth — volcanic sulfate (Laki 1783, Tambora 1815, Katmai 1912)",
        x_axis_label="year CE",
        y_axis_label="sulfate (ppb)",
    )
    _style_axes(plot)
    source = ColumnDataSource(
        {
            "year": window["year_ce"].to_numpy(dtype=float),
            "total": window["total_sulfate_ppb"].to_numpy(dtype=float),
            "volcanic": window["volcanic_sulfate_ppb"].to_numpy(dtype=float),
        }
    )
    plot.varea(x="year", y1=0, y2="volcanic", source=source, fill_color=DIVERGING_WARM, fill_alpha=0.85)
    line = plot.line(x="year", y="total", source=source, line_color=INK_MUTED, line_width=1.0)
    plot.add_tools(
        HoverTool(
            renderers=[line], mode="vline",
            tooltips=[
                ("year", "@year{0}"),
                ("total sulfate", "@total{0.0} ppb"),
                ("volcanic", "@volcanic{0.0} ppb"),
            ],
        )
    )
    return plot, _add_selection_band(plot)


def _add_selection_band(plot) -> BoxAnnotation:
    band = BoxAnnotation(
        left=DEFAULT_SELECTION[0],
        right=DEFAULT_SELECTION[1],
        fill_color=INK_PRIMARY,
        fill_alpha=0.08,
        line_color=INK_PRIMARY,
        line_alpha=0.55,
        line_width=1.2,
    )
    plot.add_layout(band)
    return band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "greenland_preview.html")
    arguments = parser.parse_args()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    output_file(arguments.output, title="Greenland ice core preview", mode="inline")
    save(build_document())
    size_kb = arguments.output.stat().st_size / 1024
    print(f"Wrote {arguments.output} ({size_kb:,.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
