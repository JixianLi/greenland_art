"""Loaders for GISP2 Summit multi-field records.

Sources under pub/data/paleo/icecore/greenland/summit/gisp2/:
  chem/ionb.txt      major ions, B core 0-200 m, ~bi-yearly (Mayewski et al. 1997)
  chem/volcano.txt   total and volcanic sulfate on the Meese/Sowers timescale
                     (Zielinski et al. 1994)
  physical/tempertr.txt  borehole temperature profile (Clow et al. 1996)

These are legacy fixed-layout text files, not NOAA template files: a prose
preamble, then a header row, then whitespace-delimited numbers. Missing values
are coded 999999. Ages are years before present with present = 1950 CE, so
negative ages are post-1950.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"
GISP2_DIR = DATASETS_DIR / "paleoclimate" / "gisp2"

MISSING_VALUE = 999999
# NOAA ships these legacy files with latin-1 characters in the prose preamble.
FILE_ENCODING = "latin-1"
BP_REFERENCE_YEAR = 1950

ION_SPECIES = ["sodium", "ammonium", "potassium", "magnesium", "calcium", "chloride", "nitrate", "sulfate"]

_ION_COLUMNS = [
    "depth_top_m",
    "depth_bottom_m",
    *ION_SPECIES,
    "age_top_bp",
    "age_bottom_bp",
]


def _find_data_start(path: Path, marker: str) -> int:
    with open(path, encoding=FILE_ENCODING) as handle:
        for line_number, line in enumerate(handle):
            if line.lstrip().startswith(marker):
                return line_number
    raise ValueError(f"Marker {marker!r} not found in {path}")


def bp_to_year_ce(age_bp: pd.Series) -> pd.Series:
    return BP_REFERENCE_YEAR - age_bp


def load_major_ions(gisp2_dir: Path | None = None) -> pd.DataFrame:
    """Eight major ion concentrations in ppb, one row per bi-yearly sample.

    Columns: depth_top_m, depth_bottom_m, the eight species in ION_SPECIES,
    age_top_bp, age_bottom_bp, year_ce.
    """
    path = (gisp2_dir or GISP2_DIR) / "chem" / "ionb.txt"
    header_row = _find_data_start(path, "depth top")

    frame = pd.read_csv(
        path,
        sep="\t",
        skiprows=header_row + 1,
        names=_ION_COLUMNS,
        engine="python",
        encoding=FILE_ENCODING,
    )
    frame = frame.replace(MISSING_VALUE, np.nan)
    frame["year_ce"] = bp_to_year_ce(frame["age_top_bp"])
    return frame.dropna(subset=["year_ce"]).reset_index(drop=True)


def load_volcanic_sulfate(gisp2_dir: Path | None = None) -> pd.DataFrame:
    """Total and volcanic (background-subtracted) sulfate in ppb.

    Columns: age_bp, total_sulfate_ppb, volcanic_sulfate_ppb, year_ce.
    """
    path = (gisp2_dir or GISP2_DIR) / "chem" / "volcano.txt"
    header_row = _find_data_start(path, "age (yr)")

    frame = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=header_row + 1,
        names=["age_bp", "total_sulfate_ppb", "volcanic_sulfate_ppb"],
        engine="python",
        encoding=FILE_ENCODING,
    )
    frame = frame.apply(pd.to_numeric, errors="coerce").replace(MISSING_VALUE, np.nan)
    frame["year_ce"] = bp_to_year_ce(frame["age_bp"])
    return frame.dropna(subset=["year_ce", "total_sulfate_ppb"]).reset_index(drop=True)


def load_borehole_temperature(gisp2_dir: Path | None = None) -> pd.DataFrame:
    """Borehole temperature profile. Columns: depth_m, temperature_c."""
    path = (gisp2_dir or GISP2_DIR) / "physical" / "tempertr.txt"
    header_row = _find_data_start(path, "TVD")

    frame = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=header_row + 1,
        names=["depth_m", "temperature_c"],
        usecols=[0, 1],
        engine="python",
        encoding=FILE_ENCODING,
    )
    return frame.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def build_multifield_matrix(
    start_year_ce: float = 1750, end_year_ce: float = 2000
) -> pd.DataFrame:
    """Co-register the ion species onto one year axis for the requested window.

    This is the input matrix for multi-field analysis: rows are samples, columns
    are the eight chemical species. Returned indexed by year_ce.
    """
    ions = load_major_ions()
    window = ions[(ions["year_ce"] >= start_year_ce) & (ions["year_ce"] <= end_year_ce)]
    matrix = window.set_index("year_ce")[ION_SPECIES].sort_index()
    return matrix.dropna(how="all")
