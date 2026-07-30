#!/usr/bin/env python3
"""Download the Box (2013) Greenland surface mass balance reconstruction.

Zenodo record 3359192 (doi:10.5281/zenodo.3359192), Box, J.E. Monthly SMB, melt
and accumulation on a 5 km grid, 1840-2012, plus per-month RMSE for the
1960-2012 calibration period. Open access, no account.

This is the reconstruction target for the ERA5 -> SMB emulator. It fully spans
the 1940-1991 window where no gridded ice observation exists.

Important caveat for anything built on it: the product is a calibration of
observational data against RACMO2.3 output, and RACMO is itself reanalysis
forced. So an ERA5 -> Box SMB mapping is partly circular by construction. That
is acceptable for demonstrating that a model can learn the relationship; it is
not evidence of independent predictive skill.
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_DIR = DATASETS_DIR / "smb_reconstruction"

ZENODO_RECORD = "3359192"
ZENODO_FILE_URL = "https://zenodo.org/api/records/{record}/files/{name}/content"

# key -> (filename, exact size in bytes, published md5), all from the Zenodo API.
# Size alone is not enough. A resumed transfer here came back 74 MB LONGER than
# the published size: the content was byte-perfect but the server re-sent a
# trailing chunk. netCDF reads by internal offset, so the file opened fine and
# looked healthy. Only the checksum distinguishes "intact plus trailing junk"
# from "corrupt in the middle", and the two need opposite responses -- truncate
# versus re-download.
BOX_FILES = {
    "smb": (
        "Box_Greenland_SMB_monthly_1840-2012_5km_cal_ver20141007.nc",
        1_405_602_152, "b0ce03a50b7f3cb87405bb1dbc2db07c",
    ),
    "melt": (
        "Box_Greenland_Melt_monthly_1840-2012_5km_cal_ver20140421.nc",
        1_404_924_332, "edeb450cd9817865e51d0afa920d3ea6",
    ),
    "accumulation": (
        "Box_Greenland_C_monthly_1840-2012_5km_cal_ver20140421.nc",
        1_404_924_332, "c85e455c98a2245d1f10469045b09d93",
    ),
    "rmse": (
        "Box_SMB_RMSE_1960-2012_monthly_v20140323.nc",
        10_807_936, "f364a5db49e2e49b9d20395688862f9f",
    ),
}


def md5_of_prefix(path: Path, byte_count: int) -> str:
    digest = hashlib.md5()
    remaining = byte_count
    with open(path, "rb") as handle:
        while remaining > 0:
            block = handle.read(min(8 << 20, remaining))
            if not block:
                break
            remaining -= len(block)
            digest.update(block)
    return digest.hexdigest()


def verify(destination: Path, expected_bytes: int, expected_md5: str) -> bool:
    """True if the file is byte-correct, truncating harmless trailing overrun."""
    actual = destination.stat().st_size
    if actual >= expected_bytes and md5_of_prefix(destination, expected_bytes) == expected_md5:
        if actual > expected_bytes:
            print(f"  trimming {actual - expected_bytes:,} trailing bytes from {destination.name}")
            with open(destination, "r+b") as handle:
                handle.truncate(expected_bytes)
        return True
    return False


def download_file(name: str, expected_bytes: int, expected_md5: str, destination: Path) -> bool:
    """Fetch with resume, then verify against the published checksum.

    curl rather than urllib because urlretrieve cannot resume: a crashed
    transfer leaves a partial file that has to be restarted from zero, and these
    are 1.4 GB each.
    """
    if destination.exists():
        if verify(destination, expected_bytes, expected_md5):
            print(f"  skip (verified)  {destination.name}")
            return True
        actual = destination.stat().st_size
        if actual > expected_bytes:
            print(f"  {destination.name} is oversized and corrupt; restarting")
            destination.unlink()
        else:
            print(f"  resuming {destination.name} from {actual / 1e6:.0f} of {expected_bytes / 1e6:.0f} MB")

    command = [
        "curl", "-fSL", "-C", "-",
        "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
        "-o", str(destination),
        ZENODO_FILE_URL.format(record=ZENODO_RECORD, name=name),
    ]
    result = subprocess.run(command)

    if not destination.exists():
        print(f"  FAILED  {destination.name}: nothing written")
        return False
    if result.returncode != 0 or not verify(destination, expected_bytes, expected_md5):
        actual = destination.stat().st_size
        print(f"  INCOMPLETE  {destination.name}: {actual:,} of {expected_bytes:,} B (resumable)")
        return False

    print(f"  ok {destination.stat().st_size / 1e6:>8.1f} MB  {destination.name}  md5 verified")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--only", nargs="*", choices=sorted(BOX_FILES),
        help="Subset of products to fetch (default: all)",
    )
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = arguments.only or sorted(BOX_FILES)

    print(f"Box (2013) SMB reconstruction -> {arguments.output_dir}")
    downloaded = 0
    for key in wanted:
        name, expected_bytes, expected_md5 = BOX_FILES[key]
        downloaded += download_file(name, expected_bytes, expected_md5, arguments.output_dir / name)
    print(f"\n{downloaded}/{len(wanted)} files complete")


if __name__ == "__main__":
    main()
