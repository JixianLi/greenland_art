#!/usr/bin/env python3
"""Export MAR fields as .vti time series for ParaView.

Two modes, because MAR carries two genuinely different geometries:

  surface   73 x 135 x 1   the daily 2D fields -- melt, runoff, albedo, the
                           energy budget. A flat map animated over time.
  snowpack  73 x 135 x 18  the vertical snow profile: density, temperature and
                           liquid water through the firn. Actually volumetric,
                           and the case where ParaView earns its keep over a
                           matplotlib figure.

Both write one .vti per timestep plus a .pvd index, so ParaView loads the whole
series as a single dataset with a real time axis rather than an unordered file
group.

Open the .pvd, not the .vti files.

Grid note: MAR's own x/y are used directly, in kilometres. They are polar
stereographic but not EPSG:3413 -- a different central meridian -- so these
exports do not align with the BedMachine or Box products without reprojection.
Fine for looking at MAR alone, wrong for overlaying.

Vertical note for snowpack: real snow layers are centimetres to metres thick
against a 20 km horizontal cell. Drawn to scale the stack would be invisible,
so the layer axis is given the same 20 km spacing as the horizontal. The
vertical axis is therefore NOT physical -- it is layer index, evenly spaced.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from greenland_art.visualization.vti import write_image_data, write_time_series_index

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
MAR_DIR = DATASETS_DIR / "mar"

ICE_SHEET_SECTOR = 0
MINIMUM_ICE_PERCENT = 50.0
GRID_SPACING_KM = 20.0

# Daily 2D fields worth animating: the mass and energy terms a glaciologist
# reads first, kept short so the files stay small and the ParaView field list
# stays navigable.
SURFACE_EXPORT = [
    "SMB", "ME", "RU", "RZ", "SF", "RF", "SU",
    "ST", "PDD", "AL", "SWD", "LWD", "SHF", "LHF", "CC",
]

# The 18-layer firn profile.
SNOWPACK_EXPORT = ["RO1", "TI1", "WA1"]


def open_year(mar_dir: Path, year: int) -> xr.Dataset:
    matches = [
        path for path in mar_dir.rglob(f"ICE.{year}.*.nc") if not path.name.startswith("._")
    ]
    if not matches:
        raise SystemExit(
            f"No ICE.{year}.*.nc under {mar_dir}. "
            "Run scripts/download_mar_greenland.py and let prepare_mar_training_data.py "
            "extract the archive, or unzip it manually."
        )
    return xr.open_dataset(matches[0], decode_times=False)


def masked(values: np.ndarray, ice: np.ndarray) -> np.ndarray:
    """Blank the non-ice cells so they do not drive ParaView's colour range."""
    out = np.asarray(values, dtype=np.float32).copy()
    out[..., ~ice] = np.nan
    return out


def export_surface(dataset, ice, days, output_dir: Path, year: int) -> list[tuple[float, str]]:
    available = [name for name in SURFACE_EXPORT if name in dataset]
    print(f"  surface fields: {', '.join(available)}")

    planes = {}
    for name in available:
        variable = dataset[name]
        if "SECTOR" in variable.dims:
            variable = variable.isel(SECTOR=ICE_SHEET_SECTOR)
        extra = [d for d in variable.dims if d not in ("TIME", "Y21_155", "X12_84")]
        if extra:
            variable = variable.isel({extra[0]: 0})
        planes[name] = variable.values.astype(np.float32)

    entries = []
    for step, day in enumerate(days):
        fields = {
            name: masked(plane[day], ice)[np.newaxis, :, :] for name, plane in planes.items()
        }
        name = f"mar_surface_{year}_{step:04d}.vti"
        write_image_data(
            output_dir / name,
            fields,
            origin=(float(dataset["X12_84"][0]), float(dataset["Y21_155"][0]), 0.0),
            spacing=(GRID_SPACING_KM, GRID_SPACING_KM, GRID_SPACING_KM),
        )
        entries.append((float(day + 1), name))
    return entries


def export_snowpack(dataset, ice, days, output_dir: Path, year: int) -> list[tuple[float, str]]:
    available = [name for name in SNOWPACK_EXPORT if name in dataset]
    print(f"  snowpack fields: {', '.join(available)}")

    volumes = {}
    for name in available:
        variable = dataset[name]
        if "SECTOR" in variable.dims:
            variable = variable.isel(SECTOR=ICE_SHEET_SECTOR)
        # (TIME, LAYER, Y, X) is already (t, z, y, x).
        volumes[name] = variable.values.astype(np.float32)

    entries = []
    for step, day in enumerate(days):
        fields = {name: masked(volume[day], ice) for name, volume in volumes.items()}
        name = f"mar_snowpack_{year}_{step:04d}.vti"
        write_image_data(
            output_dir / name,
            fields,
            origin=(float(dataset["X12_84"][0]), float(dataset["Y21_155"][0]), 0.0),
            spacing=(GRID_SPACING_KM, GRID_SPACING_KM, GRID_SPACING_KM),
        )
        entries.append((float(day + 1), name))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2009)
    parser.add_argument("--mode", choices=["surface", "snowpack"], default="surface")
    parser.add_argument("--day-stride", type=int, default=5)
    parser.add_argument("--mar-dir", type=Path, default=MAR_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()

    dataset = open_year(arguments.mar_dir, arguments.year)
    ice = dataset["MSK"].values >= MINIMUM_ICE_PERCENT
    days = np.arange(0, dataset.sizes["TIME"], arguments.day_stride)

    output_dir = arguments.output_dir or (
        DATASETS_DIR / "paraview" / f"mar_{arguments.mode}_{arguments.year}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"MAR {arguments.year} -> {output_dir}")
    print(f"  grid {dataset.sizes['X12_84']} x {dataset.sizes['Y21_155']} at {GRID_SPACING_KM} km")
    print(f"  ice cells {int(ice.sum())}, {len(days)} timesteps of {dataset.sizes['TIME']}")

    exporter = export_surface if arguments.mode == "surface" else export_snowpack
    entries = exporter(dataset, ice, days, output_dir, arguments.year)

    index = write_time_series_index(
        output_dir / f"mar_{arguments.mode}_{arguments.year}.pvd", entries
    )
    written = sum(path.stat().st_size for path in output_dir.glob("*.vti"))
    print(f"\nWrote {len(entries)} timesteps, {written / 1e6:.0f} MB")
    print(f"Open this in ParaView: {index}")


if __name__ == "__main__":
    main()
