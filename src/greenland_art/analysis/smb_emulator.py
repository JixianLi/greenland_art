"""Learn the ERA5 climate -> Box SMB reconstruction mapping.

The question this answers for a collaborator is narrow and concrete: given
gridded temperature and precipitation, can a model reproduce a published ice
sheet surface mass balance reconstruction on years it has never seen?

Two design choices carry the honesty of the result, and both are easy to get
wrong in a way that flatters the model:

Held-out years are contiguous, never random. Adjacent years are strongly
correlated, so a random split puts 1971 in training and 1972 in test and the
model half-remembers the answer. Contiguous blocks make the test a real
extrapolation in time.

Skill is reported against climatology, on anomalies. Greenland's SMB spatial
pattern barely changes year to year, so predicting the long-term mean map
already scores R^2 near 0.98. That number is meaningless as evidence of
learning. The honest measure is skill on the anomaly -- what is left after each
pixel's own mean is removed.

Standing caveat: Box SMB is calibrated against RACMO2.3, which is reanalysis
forced, so an ERA5 -> Box mapping is partly circular by construction. This
demonstrates that a model can learn the relationship in their data. It is not
evidence of independent predictive skill.
"""

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"
BOX_SMB_PATH = (
    DATASETS_DIR / "smb_reconstruction"
    / "Box_Greenland_SMB_monthly_1840-2012_5km_cal_ver20141007.nc"
)
ERA5_DIR = DATASETS_DIR / "era5"

# Box is 5 km; coarsening by 4 gives 20 km, matching the MAR grid and cutting
# the sample count ~16x with no loss of the spatial structure at this scale.
COARSEN_FACTOR = 4

# Melt responds to summer warmth, not the annual mean, so June-August
# temperature is carried as its own feature.
SUMMER_MONTHS = [6, 7, 8]

CELL_AREA_M2 = (5_000 * COARSEN_FACTOR) ** 2
KG_PER_GIGATONNE = 1e12


@dataclass(frozen=True)
class YearSplit:
    """Contiguous train/test years. Contiguous is the point -- see module docstring."""

    train: tuple[int, int]
    test: tuple[int, int]

    def train_years(self) -> range:
        return range(self.train[0], self.train[1] + 1)

    def test_years(self) -> range:
        return range(self.test[0], self.test[1] + 1)


def load_box_annual(path: Path | None = None) -> xr.Dataset:
    """Annual SMB (kg/m2/yr) on the coarsened Box grid, with mask and elevation.

    icemask2 is the fractional ice-sheet mask. icemask is not interchangeable:
    it codes 1 for the ice sheet and 2 for peripheral glaciers, so using it as a
    weight double-counts the periphery and inflates the ice sheet area from
    1.62 to 2.0 million km2.
    """
    dataset = xr.open_dataset(path or BOX_SMB_PATH)
    coarsen = {"x": COARSEN_FACTOR, "y": COARSEN_FACTOR}

    annual = dataset["MassFlux"].sum("lev")
    return xr.Dataset(
        {
            "smb": annual.coarsen(**coarsen, boundary="trim").mean(),
            "ice_fraction": dataset["icemask2"].coarsen(**coarsen, boundary="trim").mean(),
            "elevation": dataset["dem"].coarsen(**coarsen, boundary="trim").mean(),
            "latitude": dataset["lat"].coarsen(**coarsen, boundary="trim").mean(),
            "longitude": dataset["lon"].coarsen(**coarsen, boundary="trim").mean(),
        }
    )


def load_era5_annual(era5_dir: Path | None = None) -> xr.Dataset:
    """Annual mean temperature, summer mean temperature, annual precipitation."""
    era5_dir = era5_dir or ERA5_DIR

    def open_variable(short_name: str) -> xr.DataArray:
        paths = sorted(
            p for p in glob.glob(str(era5_dir / f"era5_{short_name}_greenland_*.nc"))
            if "/._" not in p
        )
        if not paths:
            raise FileNotFoundError(
                f"No ERA5 {short_name} files in {era5_dir}. "
                "Run scripts/download_era5_greenland.py first."
            )
        return xr.open_mfdataset(paths, combine="by_coords")[short_name]

    temperature = open_variable("t2m")
    precipitation = open_variable("tp")
    is_summer = temperature["valid_time.month"].isin(SUMMER_MONTHS)

    return xr.Dataset(
        {
            "annual_temperature_k": temperature.groupby("valid_time.year").mean(),
            "summer_temperature_k": (
                temperature.where(is_summer, drop=True).groupby("valid_time.year").mean()
            ),
            # Monthly means are m/day; the annual mean times 365.25 is m/yr.
            "annual_precipitation_m": (
                (precipitation * 365.25).groupby("valid_time.year").mean()
            ),
        }
    ).load()


def sample_era5_onto_box(era5: xr.Dataset, box: xr.Dataset) -> xr.Dataset:
    """Bilinearly sample the ERA5 lat/lon grid at the Box grid's cell centres.

    Box is polar stereographic with 2D lat/lon coordinates, ERA5 is a regular
    lat/lon grid, so this is interpolation of ERA5 at scattered points rather
    than a grid-to-grid regrid.
    """
    target_lat = xr.DataArray(box["latitude"].values, dims=("y", "x"))
    target_lon = xr.DataArray(box["longitude"].values, dims=("y", "x"))
    return era5.interp(latitude=target_lat, longitude=target_lon, method="linear")


def build_feature_table(
    box: xr.Dataset, era5_on_box: xr.Dataset, minimum_ice_fraction: float = 0.5
) -> pd.DataFrame:
    """One row per (year, ice cell). Columns are the features plus the target.

    Cells below minimum_ice_fraction are dropped: a cell that is mostly bare
    rock has an SMB that is not physically comparable to an ice sheet cell, and
    including them lets the model score well by learning the coastline.
    """
    years = sorted(
        set(box["time"].values.astype(int)) & set(era5_on_box["year"].values.astype(int))
    )
    ice_cells = box["ice_fraction"].values >= minimum_ice_fraction

    elevation = box["elevation"].values[ice_cells]
    latitude = box["latitude"].values[ice_cells]
    ice_fraction = box["ice_fraction"].values[ice_cells]
    cell_index = np.flatnonzero(ice_cells.ravel())

    frames = []
    for year in years:
        frames.append(
            pd.DataFrame(
                {
                    "year": year,
                    "cell": cell_index,
                    "elevation_m": elevation,
                    "latitude_deg": latitude,
                    "ice_fraction": ice_fraction,
                    "annual_temperature_k": era5_on_box["annual_temperature_k"]
                    .sel(year=year).values[ice_cells],
                    "summer_temperature_k": era5_on_box["summer_temperature_k"]
                    .sel(year=year).values[ice_cells],
                    "annual_precipitation_m": era5_on_box["annual_precipitation_m"]
                    .sel(year=year).values[ice_cells],
                    "smb_kg_m2": box["smb"].sel(time=year).values[ice_cells],
                }
            )
        )

    table = pd.concat(frames, ignore_index=True)
    return table.dropna().reset_index(drop=True)


FEATURE_COLUMNS = [
    "elevation_m",
    "latitude_deg",
    "annual_temperature_k",
    "summer_temperature_k",
    "annual_precipitation_m",
]
TARGET_COLUMN = "smb_kg_m2"

CLIMATE_COLUMNS = [
    "annual_temperature_k",
    "summer_temperature_k",
    "annual_precipitation_m",
]
# Predicting the anomaly rather than the absolute value forces the model onto
# the question that matters. Absolute SMB is dominated by geography -- a ~40 K
# spatial temperature range against a ~1 K shift between decades -- so a model
# targeting absolutes wins by memorising the map and barely learns the climate
# response. Measured here, the reformulation roughly doubled year-to-year skill.
ANOMALY_FEATURE_COLUMNS = [
    "annual_temperature_k_anomaly",
    "summer_temperature_k_anomaly",
    "annual_precipitation_m_anomaly",
    "elevation_m",
    "latitude_deg",
]
ANOMALY_TARGET_COLUMN = "smb_anomaly_kg_m2"


def add_anomaly_features(table: pd.DataFrame, train_years: range) -> pd.DataFrame:
    """Add per-cell anomalies of each climate field, and of the target.

    Baselines come from training years only. Using all years would leak the test
    period into the anomaly definition.
    """
    training = table[table["year"].isin(train_years)]
    result = table
    for column in CLIMATE_COLUMNS:
        baseline = training.groupby("cell")[column].mean().rename(f"{column}_baseline")
        result = result.merge(baseline, on="cell", how="left")
        result[f"{column}_anomaly"] = result[column] - result[f"{column}_baseline"]
    result[ANOMALY_TARGET_COLUMN] = result[TARGET_COLUMN] - result["climatology_kg_m2"]
    return result


def add_climatology(table: pd.DataFrame, train_years: range) -> pd.DataFrame:
    """Attach each cell's training-period mean SMB.

    This is the baseline to beat and the reference for anomalies. It is computed
    on training years only; using all years would leak the test period's mean
    into the baseline and into the anomaly definition.
    """
    training = table[table["year"].isin(train_years)]
    climatology = training.groupby("cell")[TARGET_COLUMN].mean().rename("climatology_kg_m2")
    return table.merge(climatology, on="cell", how="left")


def total_gigatonnes(smb_kg_m2: np.ndarray, ice_fraction: np.ndarray) -> float:
    """Convert per-cell SMB to an ice-sheet total in Gt/yr."""
    return float(np.sum(smb_kg_m2 * ice_fraction * CELL_AREA_M2) / KG_PER_GIGATONNE)
