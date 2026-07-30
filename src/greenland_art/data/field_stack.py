"""Co-register every gridded field we hold onto one EPSG:3413 grid.

Three products arrive on three different grids: BedMachine at 150 m on
EPSG:3413 metres, Box SMB at 5 km polar stereographic carrying only 2D
lat/lon, and ERA5 on a regular 0.25 degree lat/lon mesh. Comparing them, or
feeding them jointly to a model, requires one grid.

The Box grid coarsened to 20 km is the common target. It is already polar
stereographic, it covers exactly the ice sheet, and at 20 km the stack is small
enough to hold in memory whole.

This is the same array a multi-field autoencoder would consume: one row per
cell, one column per field.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from ..visualization.projection import lonlat_to_polar_stereographic
from ..analysis import smb_emulator as se

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"
BEDMACHINE_PATH = DATASETS_DIR / "bedmachine" / "BedMachineGreenland-v6.nc"

# BedMachine is 150 m; a factor of 8 gives ~1.2 km, which is still far finer
# than the 20 km target and small enough to load whole (about 12 MB per field).
BEDMACHINE_COARSEN = 8

BEDMACHINE_FIELDS = ["thickness", "surface", "bed"]


def load_bedmachine_coarse(
    path: Path | None = None, coarsen: int = BEDMACHINE_COARSEN
) -> xr.Dataset:
    """BedMachine on its native EPSG:3413 metre grid, coarsened and loaded."""
    dataset = xr.open_dataset(path or BEDMACHINE_PATH)
    grounded = dataset["mask"].isin([2, 3])

    fields = {
        name: dataset[name].where(grounded).coarsen(
            x=coarsen, y=coarsen, boundary="trim"
        ).mean()
        for name in BEDMACHINE_FIELDS
    }
    fields["ice_area_fraction"] = grounded.coarsen(
        x=coarsen, y=coarsen, boundary="trim"
    ).mean()
    return xr.Dataset(fields).load()


def sample_bedmachine_onto(bedmachine: xr.Dataset, box: xr.Dataset) -> xr.Dataset:
    """Sample BedMachine at the Box grid's cell centres.

    Box exposes only 2D lat/lon, so its centres are projected into EPSG:3413
    metres first -- the same projection BedMachine is already on.
    """
    x_metres, y_metres = lonlat_to_polar_stereographic(
        box["longitude"].values, box["latitude"].values
    )
    target_x = xr.DataArray(x_metres, dims=("y", "x"))
    target_y = xr.DataArray(y_metres, dims=("y", "x"))
    return bedmachine.interp(x=target_x, y=target_y, method="linear")


def build_field_stack(minimum_ice_fraction: float = 0.5) -> xr.Dataset:
    """Every gridded field on the common 20 km grid, masked to the ice sheet.

    Time-varying fields are reduced to their 1940-2012 means, the window where
    ERA5 and Box overlap. Returned fields:

      thickness_m, surface_m, bed_m       BedMachine v6 (static)
      smb_kg_m2                           Box (2013) mean annual SMB
      summer_temperature_c                ERA5 mean June-August temperature
      annual_precipitation_m              ERA5 mean annual precipitation
      elevation_m                         Box DEM
      ice_fraction                        Box icemask2
    """
    box = se.load_box_annual()
    era5_on_box = se.sample_era5_onto_box(se.load_era5_annual(), box)
    bedmachine = sample_bedmachine_onto(load_bedmachine_coarse(), box)

    overlap = slice(1940, 2012)
    stack = xr.Dataset(
        {
            "thickness_m": bedmachine["thickness"],
            "surface_m": bedmachine["surface"],
            "bed_m": bedmachine["bed"],
            "smb_kg_m2": box["smb"].sel(time=overlap).mean("time"),
            "summer_temperature_c": (
                era5_on_box["summer_temperature_k"].sel(year=overlap).mean("year") - 273.15
            ),
            "annual_precipitation_m": (
                era5_on_box["annual_precipitation_m"].sel(year=overlap).mean("year")
            ),
            "elevation_m": box["elevation"],
            "ice_fraction": box["ice_fraction"],
        }
    )
    return stack.where(stack["ice_fraction"] >= minimum_ice_fraction)


def stack_to_matrix(stack: xr.Dataset, field_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to (n_cells, n_fields) plus the flat cell indices that survived.

    Rows with any missing field are dropped, so the matrix is dense -- which is
    what a model needs and what an ice-sheet-shaped grid does not give for free.
    """
    columns = [stack[name].values.ravel() for name in field_names]
    matrix = np.column_stack(columns)
    complete = np.all(np.isfinite(matrix), axis=1)
    return matrix[complete], np.flatnonzero(complete)
