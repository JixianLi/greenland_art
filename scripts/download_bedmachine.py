#!/usr/bin/env python3
"""Download BedMachine Greenland v6 (bed, surface, thickness, mask).

Requires NASA Earthdata credentials in ~/.netrc for urs.earthdata.nasa.gov.
The download goes through an Earthdata Cloud redirect to a signed CloudFront
URL, so redirects must be followed with the netrc credentials still attached.

BedMachine is a single static composite, not a time series. Its granule
timestamps (1993-2021) are the span of the *input* observations that went into
one map, not a temporal dimension. Anything time-varying built on top of this
is a reconstruction and must be labelled as one.

Morlighem et al. (2017), "BedMachine v3: Complete bed topography and ocean
bathymetry mapping of Greenland from multibeam echo sounding combined with mass
conservation", Geophysical Research Letters 44. NSIDC dataset IDBMG4 v6.
"""

import argparse
import subprocess
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_DIR = DATASETS_DIR / "bedmachine"

NSIDC_BASE = (
    "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/"
    "ICEBRIDGE/IDBMG4/6/1993/01/01/"
)
GRANULE = "BedMachineGreenland-v6.nc"
EXPECTED_BYTES = 2_949_000_000  # ~2.8 GiB; used only as a sanity floor


def download_granule(output_path: Path, cookie_jar: Path) -> bool:
    """Fetch via curl.

    curl rather than urllib because the Earthdata URS redirect chain needs
    netrc credentials preserved across hosts plus a cookie jar, and because
    -C - gives a resumable transfer for a multi-gigabyte file.
    """
    command = [
        "curl", "-fSL", "--netrc",
        "-c", str(cookie_jar), "-b", str(cookie_jar),
        "-C", "-",
        "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
        "-o", str(output_path),
        NSIDC_BASE + GRANULE,
    ]
    print(f"Downloading {GRANULE} (~2.8 GB) -> {output_path}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"  curl exited {result.returncode}; partial file left for resume")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_dir / GRANULE

    if output_path.exists() and output_path.stat().st_size >= EXPECTED_BYTES:
        print(f"Already present: {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")
        return

    if download_granule(output_path, arguments.output_dir / ".urs_cookies"):
        print(f"Wrote {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
