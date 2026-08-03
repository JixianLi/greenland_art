#!/usr/bin/env python3
"""Boxplot summaries of reconstruction error, per variable, for a finished run.

At each bin on the time axis the absolute error is aggregated over the spatial
domain, so one box describes how wrong the reconstruction was across the ice
sheet at that time. Five bin definitions, because they answer different
questions and none of them subsumes the others:

  timestep    733 boxes  every (year, day) separately -- keeps year-to-year
                         differences, at the cost of being a smear at this width
  year         10 boxes  is the model drifting across the decade
  day          74 boxes  day of year pooled over years -- the seasonal shape of
                         error, where a melt-season failure would show
  year_month  120 boxes  month within year, the compromise between the two above
  month        12 boxes  calendar month pooled -- the coarsest seasonal read

Six models appear as six rows sharing an x axis, so PCA and the autoencoder can
be compared at matched latent width at every bin.

Reconstructions are recomputed from the saved checkpoints rather than read from
disk; see greenland_art.autoencoder.saved_run for why. The error statistics are
cached in the run directory, since computing them means one full pass per model.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from greenland_art.autoencoder import SavedRun

# Day-of-year boundaries of each month in a non-leap year. MAR files carry a day
# index rather than a date, and the one-day slip in leap years is far below the
# width of a monthly bin.
MONTH_START_DAY = np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
MONTH_LABEL = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

AGGREGATIONS = ("timestep", "year", "day", "year_month", "month")

STAT_NAMES = ("minimum", "q1", "median", "q3", "maximum", "mean")


def bin_labels(run, aggregation):
    """Return (per-row bin key, ordered unique keys, tick labels)."""
    years = run.year.astype(np.int64)
    days = run.day_of_year.astype(np.int64)
    months = np.searchsorted(MONTH_START_DAY, days, side="right") - 1

    if aggregation == "timestep":
        keys = years * 1000 + days
        labels = [f"{k // 1000}-{k % 1000:03d}" for k in np.unique(keys)]
    elif aggregation == "year":
        keys = years
        labels = [str(k) for k in np.unique(keys)]
    elif aggregation == "day":
        keys = days
        labels = [str(k) for k in np.unique(keys)]
    elif aggregation == "year_month":
        keys = years * 100 + months
        labels = [f"{k // 100}-{MONTH_LABEL[k % 100]}" for k in np.unique(keys)]
    else:
        keys = months
        labels = [MONTH_LABEL[k] for k in np.unique(keys)]
    return keys, np.unique(keys), labels


def compute_statistics(run, cache_path: Path, force: bool):
    """Five-number summary plus mean, per (model, aggregation, bin, variable)."""
    if cache_path.exists() and not force:
        return np.load(cache_path, allow_pickle=False)

    held_out = run.validation_mask
    if held_out is None:
        print(
            "WARNING: this run subsampled with --max-samples, so held-out rows cannot "
            "be identified. Error is computed over every cell and will flatter every "
            "model, because roughly 90 % of them were seen in training."
        )
        held_out = np.ones(len(run.features), dtype=bool)

    stored = {"held_out_only": np.array(run.validation_mask is not None)}
    rows = np.flatnonzero(held_out)

    # Group rows per bin once, by sorting. Masking the full matrix per bin would
    # be O(bins x rows) -- 733 bins against three million rows is 2.5 billion
    # comparisons per aggregation, which takes minutes rather than seconds.
    groups = {}
    for aggregation in AGGREGATIONS:
        keys, _, labels = bin_labels(run, aggregation)
        order = np.argsort(keys[rows], kind="stable")
        unique_keys, starts = np.unique(keys[rows][order], return_index=True)
        stops = np.append(starts[1:], len(order))
        groups[aggregation] = [order[start:stop] for start, stop in zip(starts, stops)]
        stored[f"{aggregation}__labels"] = np.array(labels)

    for model_name in run.model_names:
        print(f"  reconstructing {len(rows):,} rows for {model_name} ...", flush=True)
        error = run.absolute_error(model_name, rows)
        for aggregation in AGGREGATIONS:
            summary = np.zeros(
                (len(groups[aggregation]), run.features.shape[1], len(STAT_NAMES)),
                dtype=np.float32,
            )
            for position, member in enumerate(groups[aggregation]):
                selected = error[member]
                summary[position, :, :5] = np.percentile(
                    selected, [0, 25, 50, 75, 100], axis=0
                ).T
                summary[position, :, 5] = selected.mean(axis=0)
            stored[f"{aggregation}__{model_name}"] = summary
        del error

    stored["field_names"] = np.array(run.field_names)
    np.savez_compressed(cache_path, **stored)
    return np.load(cache_path, allow_pickle=False)


def draw_variable(statistics, run, aggregation, field, output_path: Path) -> Path:
    labels = [str(v) for v in statistics[f"{aggregation}__labels"]]
    column = run.field_names.index(field)
    models = run.model_names
    held_out_only = bool(statistics["held_out_only"])

    width = max(7.0, 0.16 * len(labels) + 2.0)
    figure, axes_column = plt.subplots(
        len(models), 1, figsize=(width, 1.7 * len(models) + 1.2),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, model_name in enumerate(models):
        axes = axes_column[row][0]
        summary = statistics[f"{aggregation}__{model_name}"][:, column, :]
        boxes = [
            {
                "label": label,
                # Whiskers are drawn to the true extremes rather than to a 1.5
                # IQR fence: the fence would hide exactly the worst cell on the
                # ice sheet, which is the value most worth seeing here.
                "whislo": float(summary[position, 0]),
                "q1": float(summary[position, 1]),
                "med": float(summary[position, 2]),
                "q3": float(summary[position, 3]),
                "whishi": float(summary[position, 4]),
                "mean": float(summary[position, 5]),
                "fliers": [],
            }
            for position, label in enumerate(labels)
        ]
        axes.bxp(
            boxes, showmeans=True, showfliers=False, widths=0.6,
            boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.5},
            medianprops={"color": "#e34948", "linewidth": 1.0},
            meanprops={"marker": ".", "markersize": 3, "markerfacecolor": "#2a78d6",
                       "markeredgecolor": "none"},
            capprops={"linewidth": 0.5}, flierprops={"marker": ""},
        )
        axes.set_ylabel(model_name, fontsize=8)
        axes.tick_params(labelsize=7)
        axes.grid(axis="y", alpha=0.25)

    # Symlog, because absolute error is as heavy-tailed as the fields it comes
    # from: on a linear axis the whisker reaching the worst cell on the ice sheet
    # is two orders of magnitude above the interquartile box, which collapses
    # every box to a flat line. Symlog rather than log because a perfectly
    # reconstructed cell gives exactly zero, which log cannot draw. linthresh is
    # set from the data so the linear region covers the typical error and the
    # tail is compressed above it.
    typical = np.concatenate([
        statistics[f"{aggregation}__{model}"][:, column, 2] for model in models
    ])
    positive = typical[typical > 0]
    linthresh = float(np.median(positive)) if positive.size else 1e-6
    for row in range(len(models)):
        axes_column[row][0].set_yscale("symlog", linthresh=linthresh)

    step = max(1, len(labels) // 30)
    axes_column[-1][0].set_xticks(np.arange(1, len(labels) + 1)[::step])
    axes_column[-1][0].set_xticklabels(labels[::step], rotation=90, fontsize=6)

    scope = "held-out cells only" if held_out_only else "ALL cells, ~90 % seen in training"
    figure.suptitle(
        f"{field}: absolute reconstruction error over the ice sheet, by {aggregation}\n"
        f"box = quartiles, whiskers = full spatial range, red = median, blue = mean "
        f"({scope})",
        fontsize=10,
    )
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--training-data", type=Path, default=Path("datasets/mar/mar_training_2000_2009.npz")
    )
    parser.add_argument("--fields", nargs="+", default=None, help="default: every column")
    parser.add_argument("--aggregations", nargs="+", default=list(AGGREGATIONS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recompute", action="store_true")
    arguments = parser.parse_args()

    run = SavedRun(arguments.run_dir, arguments.training_data, device=arguments.device)
    statistics = compute_statistics(
        run, arguments.run_dir / "error_statistics.npz", arguments.recompute
    )

    fields = arguments.fields or run.field_names
    unknown = [name for name in fields if name not in run.field_names]
    if unknown:
        raise SystemExit(f"not columns of this run: {', '.join(unknown)}")

    written = 0
    for field in fields:
        directory = arguments.run_dir / "error_summary" / field
        directory.mkdir(parents=True, exist_ok=True)
        for aggregation in arguments.aggregations:
            draw_variable(statistics, run, aggregation, field, directory / f"{aggregation}.png")
            written += 1
        print(f"  {field}", flush=True)

    print(f"\nWrote {written} figures to {arguments.run_dir / 'error_summary'}")


if __name__ == "__main__":
    main()
