#!/usr/bin/env python3
"""Turn MAR annual NetCDF files into one compact training matrix.

Reduces ~19 GB of decadal NetCDF to roughly 1 GB of float32, which is what
actually needs to travel to a cluster. Output is a single .npz holding:

    features     (n_samples, n_fields) float32, one row per (day, ice cell)
    field_names  (n_fields,) the MAR short names, in column order
    cell_index   (n_samples,) flat index into the 135 x 73 grid
    day_of_year  (n_samples,) 1-366
    year         (n_samples,)
    latitude, longitude, surface_height, ice_mask   per grid cell
    grid_shape   (135, 73)

Two MAR conventions worth stating because both are easy to get backwards:
SECTOR 0 is the ice sheet and SECTOR 1 is tundra -- verified by surface height,
1357 m mean against 197 m. And MSK is an ice-sheet percentage, not a 0/1 flag.
"""

import argparse
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
MAR_DIR = DATASETS_DIR / "mar"

ICE_SHEET_SECTOR = 0
MINIMUM_ICE_PERCENT = 50.0

# NetCDF's default float _FillValue. These files carry it undecoded, so it
# arrives as a literal 9.969e36 rather than NaN. It appears where a pressure
# level sits below the ice surface -- underground over a 3 km ice sheet -- so
# for some columns most rows are fill. Dropping those rows would discard the
# whole interior; dropping the column is the correct response.
NETCDF_FILL_THRESHOLD = 1e30
MAXIMUM_FILL_FRACTION = 0.02

# Daily 2D surface fields. Chosen to span the coupled system a collaborator
# would recognise: mass fluxes, the surface energy budget, cloud state and
# surface condition. Cloud hydrometeor species (QW/QI/QS/QR) are included
# because they are exactly the kind of near-redundant channel a bottleneck
# should collapse.
SURFACE_FIELDS = [
    "SF", "RF", "CP",
    "SWD", "LWD", "LWU", "SHF", "LHF",
    "AL", "ST", "PDD", "SP",
    "QW", "QI", "QS", "QR",
    "CC", "COD", "CU", "CM", "CD",
    "WVP", "IWP", "CWP",
]

# Fields that carry a SECTOR dimension; taken on the ice sheet sector.
SECTOR_FIELDS = ["SMB", "SU", "RU", "AL1", "SHSN3", "ST2", "Z0", "PBL"]

# Fields with a vertical or sub-daily axis, flattened to one feature per level.
# These carry exactly the structure a bottleneck should find redundant: an
# 18-layer snow temperature profile has nothing like 18 independent degrees of
# freedom. Surface fields alone give 32 columns, which cannot support a
# 100-dimensional latent -- PCA is capped at n_features components, and a
# bottleneck wider than its input is not a bottleneck.
PROFILE_FIELDS = [
    "RO1", "TI1", "WA1",
    "UUH", "VVH", "SWDH", "LWDH", "LWUH", "SPH", "SHFH", "LHFH", "ALH", "CCH",
    "TTP", "QQP", "UUP", "VVP",
]


def annual_files(mar_dir: Path, first_year: int, last_year: int) -> list[Path]:
    found = []
    for path in sorted(mar_dir.rglob("ICE.*.nc")):
        if path.name.startswith("._"):
            continue
        year = int(path.name.split(".")[1])
        if first_year <= year <= last_year:
            found.append(path)
    return found


def extract_archives(mar_dir: Path, first_year: int, last_year: int) -> None:
    """Unzip any decadal archive overlapping the requested years."""
    for archive in sorted(mar_dir.glob("NCEPv1_*_20km.zip")):
        if archive.name.startswith("._"):
            continue
        span = archive.stem.split("_")[1]
        start, end = (int(part) for part in span.split("-"))
        if end < first_year or start > last_year:
            continue
        target = mar_dir / archive.stem
        if target.exists() and any(target.glob("ICE.*.nc")):
            continue
        print(f"  extracting {archive.name} ...", flush=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(mar_dir)


def _level_dim(variable) -> str | None:
    return next(
        (d for d in variable.dims if d not in ("TIME", "SECTOR", "Y21_155", "X12_84")), None
    )


def expand_field_names(path: Path, include_profiles: bool) -> list[str]:
    """Column names, one entry per level for the profile fields."""
    dataset = xr.open_dataset(path, decode_times=False)
    names = list(SURFACE_FIELDS + SECTOR_FIELDS)
    if include_profiles:
        for base in PROFILE_FIELDS:
            if base not in dataset:
                continue
            dim = _level_dim(dataset[base])
            if dim is None:
                names.append(base)
            else:
                names += [f"{base}_L{i:02d}" for i in range(dataset.sizes[dim])]
    dataset.close()
    return names


def load_year(path: Path, include_profiles: bool) -> tuple[np.ndarray, xr.Dataset]:
    """Return (n_days, n_features, ny, nx) for one annual file."""
    dataset = xr.open_dataset(path, decode_times=False)
    planes = []

    for name in SURFACE_FIELDS + SECTOR_FIELDS:
        variable = dataset[name]
        if "SECTOR" in variable.dims:
            variable = variable.isel(SECTOR=ICE_SHEET_SECTOR)
        planes.append(variable.values.astype(np.float32))

    if include_profiles:
        for base in PROFILE_FIELDS:
            if base not in dataset:
                continue
            variable = dataset[base]
            if "SECTOR" in variable.dims:
                variable = variable.isel(SECTOR=ICE_SHEET_SECTOR)
            dim = _level_dim(variable)
            if dim is None:
                planes.append(variable.values.astype(np.float32))
            else:
                for i in range(variable.sizes[dim]):
                    planes.append(variable.isel({dim: i}).values.astype(np.float32))

    return np.stack(planes, axis=1), dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=2000)
    parser.add_argument("--last-year", type=int, default=2009)
    parser.add_argument("--mar-dir", type=Path, default=MAR_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--day-stride", type=int, default=1,
        help="Keep every Nth day. 1 keeps all; 5 cuts the transfer five-fold.",
    )
    parser.add_argument(
        "--surface-only", action="store_true",
        help="Use only the 32 surface fields, omitting vertical and sub-daily profiles.",
    )
    arguments = parser.parse_args()

    extract_archives(arguments.mar_dir, arguments.first_year, arguments.last_year)
    paths = annual_files(arguments.mar_dir, arguments.first_year, arguments.last_year)
    if not paths:
        raise SystemExit(
            f"No ICE.*.nc for {arguments.first_year}-{arguments.last_year} in {arguments.mar_dir}"
        )

    include_profiles = not arguments.surface_only
    field_names = expand_field_names(paths[0], include_profiles)
    print(f"{len(paths)} annual files, {len(field_names)} feature columns")

    feature_blocks, cell_blocks, day_blocks, year_blocks = [], [], [], []
    geometry: xr.Dataset | None = None
    ice_cells: np.ndarray | None = None

    for path in paths:
        stack, dataset = load_year(path, include_profiles)
        if geometry is None:
            geometry = dataset
            ice_cells = (dataset["MSK"].values >= MINIMUM_ICE_PERCENT).ravel()
            print(f"  ice cells: {int(ice_cells.sum())} of {ice_cells.size}")

        n_days = stack.shape[0]
        keep_days = np.arange(0, n_days, arguments.day_stride)
        flat = stack[keep_days].reshape(len(keep_days), len(field_names), -1)[:, :, ice_cells]
        # (days, fields, cells) -> (days*cells, fields)
        rows = np.transpose(flat, (0, 2, 1)).reshape(-1, len(field_names))

        cell_index = np.flatnonzero(ice_cells)
        n_cells = len(cell_index)
        feature_blocks.append(rows)
        cell_blocks.append(np.tile(cell_index, len(keep_days)))
        day_blocks.append(np.repeat(keep_days + 1, n_cells).astype(np.int16))
        year_blocks.append(
            np.full(len(keep_days) * n_cells, int(path.name.split(".")[1]), dtype=np.int16)
        )
        print(f"  {path.name}: {rows.shape[0]:,} rows", flush=True)

    features = np.concatenate(feature_blocks).astype(np.float32)

    features[np.abs(features) >= NETCDF_FILL_THRESHOLD] = np.nan
    fill_fraction = np.isnan(features).mean(axis=0)
    keep_columns = fill_fraction <= MAXIMUM_FILL_FRACTION
    if not keep_columns.all():
        dropped_names = [
            f"{field_names[i]} ({fill_fraction[i]* 100:.0f}%)"
            for i in np.flatnonzero(~keep_columns)
        ]
        print(f"  dropping {int((~keep_columns).sum())} mostly-fill columns: {', '.join(dropped_names)}")
        features = features[:, keep_columns]
        field_names = [n for n, keep in zip(field_names, keep_columns) if keep]

    finite = np.all(np.isfinite(features), axis=1)
    dropped = int((~finite).sum())
    if dropped:
        print(f"  dropping {dropped:,} rows still containing non-finite values")

    output = arguments.output or (
        DATASETS_DIR / "mar" / f"mar_training_{arguments.first_year}_{arguments.last_year}.npz"
    )
    np.savez_compressed(
        output,
        features=features[finite],
        field_names=np.array(field_names),
        cell_index=np.concatenate(cell_blocks)[finite].astype(np.int32),
        day_of_year=np.concatenate(day_blocks)[finite],
        year=np.concatenate(year_blocks)[finite],
        latitude=geometry["LAT"].values.astype(np.float32),
        longitude=geometry["LON"].values.astype(np.float32),
        surface_height=geometry["SH"].values.astype(np.float32),
        ice_mask=geometry["MSK"].values.astype(np.float32),
        grid_shape=np.array(geometry["MSK"].shape),
    )
    print(f"\nWrote {output}")
    print(f"  {features[finite].shape[0]:,} samples x {features.shape[1]} fields")
    print(f"  {output.stat().st_size / 1e9:.2f} GB on disk")


if __name__ == "__main__":
    main()
