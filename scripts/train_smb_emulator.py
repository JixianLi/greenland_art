#!/usr/bin/env python3
"""Train and evaluate the ERA5 -> Box SMB emulator, and render the result.

Produces outputs/smb_emulator.png: held-out maps, the ice-sheet total time
series, and the skill table.

The figure is built for a reader who does not work with machine learning. It
therefore leads with the comparison that decides whether the model learned
anything -- performance against simply predicting each cell's long-term average
-- rather than with a raw R^2, which is near 0.98 for a model that ignores its
inputs entirely.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from greenland_art.analysis import smb_emulator as se

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
COLD = "#2a78d6"
WARM = "#e34948"

SPLIT = se.YearSplit(train=(1940, 1994), test=(1995, 2012))


def fit_models(table, split):
    train = table[table["year"].isin(split.train_years())]
    test = table[table["year"].isin(split.test_years())]

    x_train = train[se.FEATURE_COLUMNS].to_numpy()
    y_train = train[se.TARGET_COLUMN].to_numpy()
    x_test = test[se.FEATURE_COLUMNS].to_numpy()

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x_train, y_train)
    boosted = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.08, max_depth=None, random_state=0
    ).fit(x_train, y_train)

    # Same learner, target reframed as the departure from each cell's own mean.
    anomaly_model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.08, max_depth=None, random_state=0
    ).fit(
        train[se.ANOMALY_FEATURE_COLUMNS].to_numpy(),
        train[se.ANOMALY_TARGET_COLUMN].to_numpy(),
    )
    anomaly_prediction = (
        anomaly_model.predict(test[se.ANOMALY_FEATURE_COLUMNS].to_numpy())
        + test["climatology_kg_m2"].to_numpy()
    )

    predictions = {
        "climatology (baseline)": test["climatology_kg_m2"].to_numpy(),
        "linear": ridge.predict(x_test),
        "boosting, absolute": boosted.predict(x_test),
        "boosting, anomaly": anomaly_prediction,
    }
    return test, predictions


def score(test, predictions):
    """Raw R^2, skill against climatology, and ice-sheet total error in Gt/yr."""
    truth = test[se.TARGET_COLUMN].to_numpy()
    climatology = test["climatology_kg_m2"].to_numpy()
    ice_fraction = test["ice_fraction"].to_numpy()

    climatology_mse = float(np.mean((truth - climatology) ** 2))
    rows = []
    for name, prediction in predictions.items():
        mse = float(np.mean((truth - prediction) ** 2))
        rows.append(
            {
                "model": name,
                "r2_raw": r2_score(truth, prediction),
                # 1 - MSE/MSE_climatology. Zero means no better than the
                # long-term average; negative means worse.
                "skill_vs_climatology": 1.0 - mse / climatology_mse,
                "anomaly_r2": r2_score(truth - climatology, prediction - climatology),
                "total_bias_gt": (
                    se.total_gigatonnes(prediction, ice_fraction)
                    - se.total_gigatonnes(truth, ice_fraction)
                )
                / len(np.unique(test["year"])),
            }
        )
    return rows


def annual_totals(test, prediction):
    years, totals = [], []
    for year, group in test.groupby("year"):
        mask = test["year"] == year
        years.append(int(year))
        totals.append(se.total_gigatonnes(prediction[mask.to_numpy()], group["ice_fraction"].to_numpy()))
    return np.array(years), np.array(totals)


def _to_grid(cells, values, shape):
    grid = np.full(shape[0] * shape[1], np.nan)
    grid[cells] = values
    return grid.reshape(shape)


def build_figure(table, test, predictions, rows, grid_shape, output_path):
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE, "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK_PRIMARY, "axes.edgecolor": BASELINE,
            "axes.spines.top": False, "axes.spines.right": False,
        }
    )
    figure = plt.figure(figsize=(15.5, 8.6))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.15, 0.92], hspace=0.10, wspace=0.14)

    display_year = int(test["year"].max())
    year_rows = test[test["year"] == display_year]
    cells = year_rows["cell"].to_numpy()
    truth = year_rows[se.TARGET_COLUMN].to_numpy()
    predicted = predictions["boosting, anomaly"][(test["year"] == display_year).to_numpy()]

    limit = float(np.nanpercentile(np.abs(truth), 99))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    panels = [
        ("Box reconstruction (truth)", truth, norm, "RdBu_r"),
        ("Model prediction", predicted, norm, "RdBu_r"),
        ("Difference (model − truth)", predicted - truth, TwoSlopeNorm(vmin=-limit / 2, vcenter=0.0, vmax=limit / 2), "PuOr_r"),
    ]
    for column, (title, values, panel_norm, cmap) in enumerate(panels):
        axis = figure.add_subplot(grid[0, column])
        image = axis.imshow(
            _to_grid(cells, values, grid_shape), origin="lower", cmap=cmap, norm=panel_norm
        )
        axis.set_title(title, fontsize=10.5, loc="left", pad=8)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        bar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
        bar.set_label("SMB (kg m⁻² yr⁻¹)", fontsize=8, color=INK_SECONDARY)
        bar.ax.tick_params(labelsize=7)
        bar.outline.set_visible(False)

    axis = figure.add_subplot(grid[1, :2])
    truth_years, truth_totals = annual_totals(test, test[se.TARGET_COLUMN].to_numpy())
    _, predicted_totals = annual_totals(test, predictions["boosting, anomaly"])
    _, climatology_totals = annual_totals(test, test["climatology_kg_m2"].to_numpy())

    axis.plot(truth_years, truth_totals, color=INK_PRIMARY, linewidth=2.2, label="Box reconstruction (truth)", zorder=4)
    axis.plot(truth_years, predicted_totals, color=COLD, linewidth=2.0, label="Model prediction", zorder=3)
    axis.plot(truth_years, climatology_totals, color=INK_MUTED, linewidth=1.4, linestyle=(0, (3, 2)), label="Baseline: long-term average", zorder=2)
    axis.set_ylabel("ice-sheet SMB (Gt yr⁻¹)", fontsize=9)
    axis.set_xlabel("year CE", fontsize=9)
    axis.set_xticks(np.arange(truth_years.min(), truth_years.max() + 1, 3))
    axis.tick_params(labelsize=8)
    axis.grid(axis="y", color=GRIDLINE, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(fontsize=8.5, frameon=False, loc="lower left")
    axis.set_title(
        f"Held-out years {SPLIT.test[0]}–{SPLIT.test[1]}: the model never saw any of these",
        fontsize=10.5, loc="left", pad=8,
    )

    axis = figure.add_subplot(grid[1, 2])
    axis.axis("off")
    lines = [
        "How to read this",
        "",
        "Raw R² flatters every model: Greenland's SMB",
        "pattern barely changes year to year, so even a",
        "model ignoring its inputs scores ~0.98.",
        "",
        "Skill vs baseline is the real test. It asks:",
        "does the model beat simply predicting each",
        "point's long-term average? 0 = no better.",
        "",
        "Reframing the target from absolute SMB to the",
        "departure from each point's own average roughly",
        "doubles real skill. Same model, same inputs.",
        "",
        f"{'model':<22}{'raw R²':>8}{'skill':>7}",
    ]
    for row in rows:
        lines.append(f"{row['model']:<22}{row['r2_raw']:>8.3f}{row['skill_vs_climatology']:>7.3f}")
    axis.text(
        0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.6,
        family="monospace", color=INK_SECONDARY, linespacing=1.55,
    )

    figure.suptitle(
        "Can a model reproduce a published ice sheet reconstruction from temperature and precipitation alone?",
        fontsize=13.5, x=0.012, ha="left", y=1.075,
    )
    figure.text(
        0.012, 0.995,
        f"Trained on {SPLIT.train[0]}–{SPLIT.train[1]}, tested on {SPLIT.test[0]}–{SPLIT.test[1]}. "
        "Held-out years are contiguous, not randomly sampled: adjacent years are correlated, so a random split "
        "would leak the answer.\nInputs are ERA5 annual and summer temperature, annual precipitation, elevation and "
        "latitude. Target is Box (2013), which is calibrated against RACMO2.3 — so this shows a model can learn "
        "the relationship, not that it has independent skill.",
        fontsize=8.8, color=INK_SECONDARY, linespacing=1.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=165, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "smb_emulator.png")
    arguments = parser.parse_args()

    box = se.load_box_annual()
    era5_on_box = se.sample_era5_onto_box(se.load_era5_annual(), box)
    table = se.build_feature_table(box, era5_on_box)
    table = se.add_climatology(table, SPLIT.train_years())
    table = table.dropna(subset=["climatology_kg_m2"])
    table = se.add_anomaly_features(table, SPLIT.train_years()).dropna()

    test, predictions = fit_models(table, SPLIT)
    rows = score(test, predictions)

    print(f"train {SPLIT.train}  test {SPLIT.test}  samples {len(table):,}")
    print(f"{'model':<20}{'raw R2':>9}{'skill':>9}{'anomaly R2':>12}{'bias Gt/yr':>12}")
    for row in rows:
        print(
            f"{row['model']:<20}{row['r2_raw']:>9.3f}{row['skill_vs_climatology']:>9.3f}"
            f"{row['anomaly_r2']:>12.3f}{row['total_bias_gt']:>12.1f}"
        )

    grid_shape = box["smb"].isel(time=0).shape
    print(f"\nWrote {build_figure(table, test, predictions, rows, grid_shape, arguments.output)}")


if __name__ == "__main__":
    main()
