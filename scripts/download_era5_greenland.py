#!/usr/bin/env python3
"""Download ERA5 monthly temperature and precipitation fields over Greenland.

Requires a Copernicus CDS account and ~/.cdsapirc. The dataset licence must be
accepted once in the CDS web UI; until it is, the API returns an authorisation
error rather than a licence error, which reads like a bad key.

Each variable is requested separately on purpose. A combined request mixes an
instantaneous field (2 m temperature) with an accumulated one (total
precipitation); CDS then splits them across internal streams and returns a zip
of two files with opaque names, regardless of download_format. One variable per
request returns a plain NetCDF.
"""

import argparse
from pathlib import Path

import cdsapi

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_DIR = DATASETS_DIR / "era5"

CDS_DATASET = "reanalysis-era5-single-levels-monthly-means"

# North, West, South, East. Covers the ice sheet plus surrounding ocean margin.
GREENLAND_AREA = [84, -75, 58, -10]

# CDS name -> short name in the resulting NetCDF.
ERA5_VARIABLES = {
    "2m_temperature": "t2m",
    "total_precipitation": "tp",
}

ALL_MONTHS = [f"{month:02d}" for month in range(1, 13)]

# ERA5 back extension starts in 1940.
FIRST_YEAR = 1940
LAST_YEAR = 2024

DOWNLOAD_ATTEMPTS = 3


def decade_chunks(first_year: int, last_year: int) -> list[list[str]]:
    """Group years into decades.

    One request for 85 years times 12 months exceeds the CDS per-request item
    limit; one request per year would queue 85 times. Decades sit between.
    """
    chunks = []
    for decade_start in range((first_year // 10) * 10, last_year + 1, 10):
        years = [
            str(year)
            for year in range(max(decade_start, first_year), min(decade_start + 9, last_year) + 1)
        ]
        if years:
            chunks.append(years)
    return chunks


def download_chunk(
    client: cdsapi.Client, variable: str, years: list[str], output_path: Path
) -> bool:
    if output_path.exists():
        print(f"  skip (exists)  {output_path.name}")
        return True

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [variable],
        "year": years,
        "month": ALL_MONTHS,
        "time": ["00:00"],
        "area": GREENLAND_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    # CDS truncates transfers often enough to matter: a first pass over 18
    # chunks lost 4 to "File size mismatch". The request itself succeeds, so
    # retrying the transfer is enough; the partial file must be removed first
    # or the existence check above would treat it as complete.
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            client.retrieve(CDS_DATASET, request, str(output_path))
        except Exception as error:
            output_path.unlink(missing_ok=True)
            if attempt == DOWNLOAD_ATTEMPTS:
                print(f"  FAILED  {output_path.name}: {error}")
                return False
            print(f"  retry {attempt}/{DOWNLOAD_ATTEMPTS - 1}  {output_path.name}: {error}")
            continue
        print(f"  ok {output_path.stat().st_size / 1e6:>7.1f} MB  {output_path.name}")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--last-year", type=int, default=LAST_YEAR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    chunks = decade_chunks(arguments.first_year, arguments.last_year)
    succeeded = 0
    total = len(ERA5_VARIABLES) * len(chunks)

    for variable, short_name in ERA5_VARIABLES.items():
        print(f"{variable} ({short_name}) -> {arguments.output_dir}")
        for years in chunks:
            name = f"era5_{short_name}_greenland_{years[0]}_{years[-1]}.nc"
            succeeded += download_chunk(client, variable, years, arguments.output_dir / name)

    print(f"\n{succeeded}/{total} chunks present in {arguments.output_dir}")


if __name__ == "__main__":
    main()
