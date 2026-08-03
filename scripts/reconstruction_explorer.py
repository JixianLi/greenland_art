#!/usr/bin/env python3
"""Interactive reconstruction viewer: pick a time and a variable, see the error.

    uv run bokeh serve --show scripts/reconstruction_explorer.py -- \
        --run-dir outputs/<jobid>

A Bokeh *server* application rather than an exported HTML file, and that is the
whole point. Every frame is computed on demand by running the saved model over
one timestep -- 4,604 rows through a small network, a few milliseconds -- so the
six models times 155 variables times 733 timesteps that a static export would
have to precompute (103 GB) never has to exist.

Colour limits for truth and prediction are fixed per variable across the entire
decade, never rescaled to the visible timestep, so scrubbing the time slider
shows the field changing rather than the colour bar changing. The difference
panel is fixed per variable and model on the same principle.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColorBar, Div, LinearColorMapper, Select, Slider
from bokeh.palettes import Cividis256, Reds256
from bokeh.plotting import figure

from greenland_art.autoencoder import SavedRun

LIMIT_SAMPLE_TIMESTEPS = 40
PANEL_SIZE = 330


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--training-data", type=Path, default=Path("datasets/mar/mar_training_2000_2009.npz")
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args(sys.argv[1:])


class Explorer:
    def __init__(self, run: SavedRun):
        self.run = run
        self.truth_limits: dict[str, tuple[float, float]] = {}
        self.error_limits: dict[tuple[str, str], float] = {}

        self.model_select = Select(
            title="model", value=run.model_names[-1], options=run.model_names, width=170
        )
        self.field_select = Select(
            title="variable", value=run.field_names[0], options=run.field_names, width=170
        )
        self.time_slider = Slider(
            title="timestep", start=0, end=len(run.timesteps) - 1, value=0, step=1, width=760
        )
        self.caption = Div(width=760)

        self.panels, self.mappers, self.sources = {}, {}, {}
        for name, palette in (
            ("truth", Cividis256), ("prediction", Cividis256), ("|difference|", Reds256)
        ):
            mapper = LinearColorMapper(palette=palette, nan_color="#ffffff")
            panel = figure(
                title=name, width=PANEL_SIZE, height=int(PANEL_SIZE * 1.6),
                x_range=(0, run.grid_shape[1]), y_range=(0, run.grid_shape[0]),
                toolbar_location="above", match_aspect=True,
            )
            panel.axis.visible = False
            panel.grid.visible = False
            renderer = panel.image(
                image=[np.zeros(run.grid_shape)], x=0, y=0,
                dw=run.grid_shape[1], dh=run.grid_shape[0], color_mapper=mapper,
            )
            panel.add_layout(ColorBar(color_mapper=mapper, width=8), "right")
            self.panels[name], self.mappers[name] = panel, mapper
            self.sources[name] = renderer.data_source

        for widget in (self.model_select, self.field_select):
            widget.on_change("value", self._on_selection_change)
        self.time_slider.on_change("value_throttled", self._on_time_change)
        self.refresh()

    def field_limits(self, field: str) -> tuple[float, float]:
        """Percentile limits over the whole decade, computed once per variable.

        Over the full column, not the visible timestep: a colour bar that
        rescales while the slider moves makes every timestep look alike and hides
        the seasonal cycle, which is the main thing worth seeing here.
        """
        if field not in self.truth_limits:
            column_index = self.run.field_names.index(field)
            values = self.run.features[:, column_index]
            low, high = np.percentile(values, [1.0, 99.0])
            if low == high:
                low, high = float(values.min()), float(values.max()) or 1.0
            self.truth_limits[field] = (float(low), float(high))
        return self.truth_limits[field]

    def error_limit(self, model_name: str, field: str) -> float:
        """99th percentile of |error| over a sample of timesteps, cached.

        Sampled rather than exhaustive because this runs while the user waits:
        40 timesteps spread across the decade settle the colour bar to within a
        few percent of the full-record value at a fraction of the cost.
        """
        key = (model_name, field)
        if key not in self.error_limits:
            column_index = self.run.field_names.index(field)
            step = max(1, len(self.run.timesteps) // LIMIT_SAMPLE_TIMESTEPS)
            errors = []
            for year, day in self.run.timesteps[::step]:
                rows = self.run.timestep_rows(year, day)
                reconstruction = self.run.reconstruct(model_name, year, day)
                errors.append(
                    np.abs(reconstruction[:, column_index]
                           - self.run.features[rows, column_index])
                )
            self.error_limits[key] = float(np.percentile(np.concatenate(errors), 99.0)) or 1.0
        return self.error_limits[key]

    def refresh(self) -> None:
        model_name = self.model_select.value
        field = self.field_select.value
        year, day = self.run.timesteps[self.time_slider.value]
        column_index = self.run.field_names.index(field)

        truth = self.run.truth(year, day)[:, column_index]
        prediction = self.run.reconstruct(model_name, year, day)[:, column_index]
        difference = np.abs(prediction - truth)

        low, high = self.field_limits(field)
        for name, values, limits in (
            ("truth", truth, (low, high)),
            ("prediction", prediction, (low, high)),
            ("|difference|", difference, (0.0, self.error_limit(model_name, field))),
        ):
            self.mappers[name].low, self.mappers[name].high = limits
            self.sources[name].data = {
                "image": [self.run.to_grid(values, year, day)],
                "x": [0], "y": [0],
                "dw": [self.run.grid_shape[1]], "dh": [self.run.grid_shape[0]],
            }

        held_out = self.run.validation_mask
        scope = ""
        if held_out is not None:
            rows = self.run.timestep_rows(year, day)
            mask = held_out[rows]
            if mask.any():
                error = difference[mask]
                scope = (
                    f" &nbsp;|&nbsp; held-out cells here: {int(mask.sum())}, "
                    f"RMSE {np.sqrt(np.mean(error**2)):.4g}"
                )
        self.caption.text = (
            f"<b>{field}</b> &nbsp; {model_name} &nbsp; {year} day {day} "
            f"&nbsp;|&nbsp; colour limits fixed over 2000-2009 (1st-99th pct), "
            f"difference clipped at its 99th pct{scope}"
            f"<br><span style='color:#888'>MAR v3.2 regional climate model output, "
            f"NCEPv1-forced, 20 km. MODEL OUTPUT, NOT OBSERVATION. "
            f"Grid is MAR's own polar stereographic, not EPSG:3413.</span>"
        )

    def _on_selection_change(self, attr, old, new) -> None:
        self.refresh()

    def _on_time_change(self, attr, old, new) -> None:
        self.refresh()

    def layout(self):
        return column(
            row(self.model_select, self.field_select),
            self.time_slider,
            row(*[self.panels[name] for name in ("truth", "prediction", "|difference|")]),
            self.caption,
        )


arguments = parse_arguments()
explorer = Explorer(SavedRun(arguments.run_dir, arguments.training_data, arguments.device))
curdoc().add_root(explorer.layout())
curdoc().title = f"Reconstruction explorer — {arguments.run_dir.name}"
