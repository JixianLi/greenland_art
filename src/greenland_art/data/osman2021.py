"""Loader for the Osman et al. 2021 Greenland ice core array.

Osman, M.B., Coats, S., Das, S.B., McConnell, J.R., Chellman, N. (2021).
"North Atlantic jet stream projections in the context of the past 1,250 years."
PNAS 118(38). NOAA study, files under
pub/data/paleo/icecore/greenland/osman2021/.

Ten sites, annually resolved d18O (per mil VSMOW) and accumulation
(kg m-2 yr-1), timestamped in Year CE at year midpoints.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"
OSMAN_DIR = DATASETS_DIR / "paleoclimate" / "osman2021"

# NOAA ships these with latin-1 degree symbols in the header prose.
FILE_ENCODING = "latin-1"

_FILENAME_PATTERN = re.compile(r"^(?P<site>.+?)-?2021(?P<variable>accum|d18o)\.txt$")

_HEADER_FIELDS = {
    "site_name": "Site_Name",
    "latitude": "Northernmost_Latitude",
    "longitude": "Easternmost_Longitude",
}


def _parse_noaa_header(path: Path) -> dict[str, str]:
    """Pull site metadata out of the NOAA template comment block."""
    found: dict[str, str] = {}
    with open(path, encoding=FILE_ENCODING) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            for key, noaa_key in _HEADER_FIELDS.items():
                if key not in found and f"{noaa_key}:" in line:
                    found[key] = line.split(":", 1)[1].strip()
    return found


def _read_measurements(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", comment="#", encoding=FILE_ENCODING)
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def load_site_records(osman_dir: Path | None = None) -> pd.DataFrame:
    """Return every site/variable measurement in long format.

    Columns: site, latitude, longitude, variable, age_ce, value.
    Variable is 'd18o' or 'accumulation'.
    """
    osman_dir = osman_dir or OSMAN_DIR
    records = []

    for path in sorted(osman_dir.glob("*2021*.txt")):
        # The dataset volume grows macOS AppleDouble sidecars ("._name"); they
        # share the glob but are binary resource forks, not data.
        if path.name.startswith("._"):
            continue
        match = _FILENAME_PATTERN.match(path.name)
        if match is None:
            continue

        header = _parse_noaa_header(path)
        measurements = _read_measurements(path)
        value_column = [c for c in measurements.columns if c != "age_CE"][0]

        records.append(
            pd.DataFrame(
                {
                    "site": header.get("site_name", match.group("site")),
                    "latitude": float(header["latitude"]),
                    "longitude": float(header["longitude"]),
                    "variable": "d18o" if match.group("variable") == "d18o" else "accumulation",
                    "age_ce": measurements["age_CE"].astype(float),
                    "value": pd.to_numeric(measurements[value_column], errors="coerce"),
                }
            )
        )

    if not records:
        raise FileNotFoundError(
            f"No Osman 2021 files in {osman_dir}. Run scripts/download_preview_data.py first."
        )

    return pd.concat(records, ignore_index=True).dropna(subset=["value"])


def load_site_metadata(records: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per site: coordinates, year span, and which variables exist."""
    records = load_site_records() if records is None else records
    metadata = (
        records.groupby("site")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            first_year=("age_ce", "min"),
            last_year=("age_ce", "max"),
            n_samples=("value", "size"),
        )
        .reset_index()
    )
    variables = records.groupby("site")["variable"].unique().apply(lambda v: ",".join(sorted(v)))
    return metadata.merge(variables.rename("variables"), on="site").sort_values(
        "latitude", ascending=False
    )


def pivot_variable(records: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Site-by-year matrix for one variable, indexed by integer year CE.

    age_ce is a year midpoint (2009.5 means calendar 2009), so floor is the
    correct conversion. round() would be wrong twice over: it shifts every
    label up one year, and numpy's round-half-to-even collapses adjacent pairs
    (2007.5 and 2008.5 both land on 2008), silently halving the record.
    """
    subset = records[records["variable"] == variable].copy()
    subset["year_ce"] = np.floor(subset["age_ce"]).astype(int)
    return subset.pivot_table(index="year_ce", columns="site", values="value", aggfunc="mean")
