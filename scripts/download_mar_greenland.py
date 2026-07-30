#!/usr/bin/env python3
"""Download MAR v3.2 Greenland regional climate model output.

Tedesco, M. (2018). "Modele Atmospherique Regional (MAR) three-dimensional
regional climate model (RCM), version 3.2, over Greenland, 1948-2016."
Arctic Data Center, doi:10.18739/A23775W1C. Open access, no account.

This is the multi-field target: ~80 time-varying variables on a 73 x 135
20 km grid at daily resolution, covering mass fluxes (SMB, melt, runoff,
refreezing, snowfall, rainfall, sublimation), the surface energy budget
(shortwave, longwave, sensible and latent heat, albedo), atmospheric state,
and snowpack properties (density, temperature, liquid water, grain shape).

It is forced by NCEPv1 rather than ERA. That matters: an ERA5 -> MAR mapping
is less circular than ERA5 -> a RACMO-calibrated product, because the forcing
reanalysis differs from the predictor reanalysis. The two still assimilate
overlapping observations, so it is not independent, only less entangled.

The full deposit is 130 GB across decadal zips. Fetch only what a given
experiment needs.
"""

import argparse
import subprocess
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_DIR = DATASETS_DIR / "mar"

DATAONE_OBJECT = "https://arcticdata.io/metacat/d1/mn/v2/object/{pid}"

# decade key -> (persistent identifier, filename, exact size in bytes).
# Sizes come from the DataONE index and are checked exactly, so an interrupted
# transfer resumes rather than being mistaken for a complete file.
MAR_DECADES = {
    "1948-1949": ("urn:uuid:8e45a873-28c6-47c3-88f7-34e971943033", "NCEPv1_1948-1949_20km.zip", 4_486_690_883),
    "1950-1959": ("urn:uuid:e539b208-50a4-4b8f-a83b-c441715b3812", "NCEPv1_1950-1959_20km.zip", 19_924_175_012),
    "1960-1969": ("urn:uuid:379b92ef-3577-4079-a4a8-2ed5d8a8c71c", "NCEPv1_1960-1969_20km.zip", 19_313_071_783),
    "1970-1979": ("urn:uuid:2eb93404-1fa8-4439-b19e-92b273b1f187", "NCEPv1_1970-1979_20km.zip", 18_628_785_409),
    "1980-1989": ("urn:uuid:d8574a56-10a9-4944-a045-9364d3b68399", "NCEPv1_1980-1989_20km.zip", 19_019_756_061),
    "1990-1999": ("urn:uuid:db7a7294-7a7c-4ac4-ae96-9237c88d4df7", "NCEPv1_1990-1999_20km.zip", 18_815_125_239),
    "2000-2009": ("urn:uuid:5ebdf4c5-ad3d-422d-9a5e-9f63e78120c4", "NCEPv1_2000-2009_20km.zip", 18_728_016_868),
    "2010-2016": ("urn:uuid:57dfc39a-a0ab-49f4-a4d2-32f1a796c2dc", "NCEPv1_2010-2016_20km.zip", 11_746_563_444),
}

# Variable descriptions (18 KB) and model ancillary files (22 MB).
MAR_SUPPORT = {
    "ncinfo": ("urn:uuid:7f583033-ec4c-482a-b772-bd86c0147bdb", "ncinfo.txt", 18_427),
    "ancillary": ("urn:uuid:4601001e-d4b6-40ee-a51a-9136d00e6f33", "MAR.zip", 22_861_339),
}


def download_object(pid: str, expected_bytes: int, destination: Path) -> bool:
    if destination.exists() and destination.stat().st_size == expected_bytes:
        print(f"  skip (complete)  {destination.name}")
        return True
    if destination.exists():
        print(f"  resuming {destination.name} from {destination.stat().st_size / 1e9:.2f} GB")

    command = [
        "curl", "-fSL", "-C", "-",
        "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
        "-o", str(destination),
        DATAONE_OBJECT.format(pid=pid),
    ]
    result = subprocess.run(command)

    actual = destination.stat().st_size if destination.exists() else 0
    if result.returncode != 0 or actual != expected_bytes:
        print(f"  INCOMPLETE  {destination.name}: {actual:,} of {expected_bytes:,} B (resumable)")
        return False
    print(f"  ok {actual / 1e9:>6.2f} GB  {destination.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--decades", nargs="*", choices=sorted(MAR_DECADES), default=["1948-1949"],
        help="Which decadal archives to fetch (default: the 4.5 GB 1948-1949 sample)",
    )
    parser.add_argument("--skip-support", action="store_true")
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"MAR v3.2 Greenland -> {arguments.output_dir}")

    if not arguments.skip_support:
        for pid, name, size in MAR_SUPPORT.values():
            download_object(pid, size, arguments.output_dir / name)

    requested = sum(MAR_DECADES[d][2] for d in arguments.decades)
    print(f"  requesting {len(arguments.decades)} decade(s), {requested / 1e9:.1f} GB")
    for decade in arguments.decades:
        pid, name, size = MAR_DECADES[decade]
        download_object(pid, size, arguments.output_dir / name)


if __name__ == "__main__":
    main()
