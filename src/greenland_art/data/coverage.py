"""What actually backs each year of the 1940-2024 frame.

The project rule is that generated values are never presented as observation.
Over a 1940-2024 span that rule cannot live in a figure caption, because the
answer changes by year and by variable: ERA5 covers the whole span, ice sheet
observation covers only 1992 onward, and the ice core chemistry stops in 1988.

This module makes the provenance structural. Every record carries a tier, and
anything built on the frame can be forced to state which tier it is standing on.
"""

from dataclasses import dataclass

import pandas as pd

FRAME_FIRST_YEAR = 1940
FRAME_LAST_YEAR = 2024

# Ordered worst-to-best; a derived product inherits the weakest tier it touches.
TIER_RECONSTRUCTED = "reconstructed"
TIER_STATIC = "static"
TIER_OBSERVED = "observed"

TIER_ORDER = [TIER_RECONSTRUCTED, TIER_STATIC, TIER_OBSERVED]


@dataclass(frozen=True)
class DataSource:
    """One input record and the years it genuinely covers.

    first_year/last_year are the years the record itself spans. For a static
    composite these are the span of the *input* observations, not a time axis —
    tier distinguishes that case.
    """

    name: str
    quantity: str
    first_year: int
    last_year: int
    tier: str
    note: str


SOURCES = [
    DataSource(
        "ERA5", "2 m temperature, total precipitation",
        1940, 2024, TIER_OBSERVED,
        "reanalysis: observations assimilated into a model, not direct measurement",
    ),
    DataSource(
        "Osman et al. 2021", "d18O, accumulation (10 cores)",
        1940, 2013, TIER_OBSERVED,
        "annual, point records; last site ends 2013",
    ),
    DataSource(
        "GISP2 major ions", "8 chemical species",
        1940, 1988, TIER_OBSERVED,
        "~bi-yearly, single site; record ends 1988",
    ),
    DataSource(
        "BedMachine v6", "absolute ice thickness",
        1993, 2021, TIER_STATIC,
        "ONE composite map; the years are input span, there is no time axis",
    ),
    DataSource(
        "CDS satellite ice sheet elevation change", "dh/dt",
        1992, 2024, TIER_OBSERVED,
        "monthly gridded; earliest gridded ice observation of any kind",
    ),
    DataSource(
        "CDS gravimetric mass balance", "mass change per basin",
        2003, 2022, TIER_OBSERVED,
        "drainage basins, not gridded",
    ),
    DataSource(
        "Kjeldsen et al. 2015 (Nature)", "spatial mass loss",
        1900, 2010, TIER_RECONSTRUCTED,
        "published reconstruction from aerial photography and SMB modelling",
    ),
    DataSource(
        "Fettweis et al. 2016 (TC)", "surface mass balance",
        1900, 2015, TIER_RECONSTRUCTED,
        "published MAR reconstruction",
    ),
]


def build_coverage_frame(sources: list[DataSource] | None = None) -> pd.DataFrame:
    """Year-by-source table of which tier backs each year.

    Rows are years FRAME_FIRST_YEAR..FRAME_LAST_YEAR, columns are source names,
    values are the tier string or None where the source does not reach.
    """
    sources = sources or SOURCES
    years = range(FRAME_FIRST_YEAR, FRAME_LAST_YEAR + 1)
    return pd.DataFrame(
        {
            source.name: [
                source.tier if source.first_year <= year <= source.last_year else None
                for year in years
            ]
            for source in sources
        },
        index=pd.Index(years, name="year_ce"),
    )


def ice_observation_gap(sources: list[DataSource] | None = None) -> tuple[int, int, int]:
    """Years in the frame with no gridded ice observation at all.

    Returns (first_year, last_year, count). This is the span that any
    1940-2024 ice product must reconstruct rather than observe.
    """
    sources = sources or SOURCES
    gridded_ice = [
        source
        for source in sources
        if source.tier == TIER_OBSERVED and "elevation change" in source.name
    ]
    if not gridded_ice:
        raise ValueError("no gridded ice observation source registered")

    earliest = min(source.first_year for source in gridded_ice)
    return FRAME_FIRST_YEAR, earliest - 1, earliest - FRAME_FIRST_YEAR


def summarise() -> pd.DataFrame:
    """One row per source, for printing or putting on a figure."""
    return pd.DataFrame(
        [
            {
                "source": source.name,
                "quantity": source.quantity,
                "covers": f"{source.first_year}-{source.last_year}",
                "tier": source.tier,
                "note": source.note,
            }
            for source in SOURCES
        ]
    )
