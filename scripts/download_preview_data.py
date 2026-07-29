#!/usr/bin/env python3
"""Download the two real NOAA datasets backing the preview.

Osman et al. 2021 gives annually resolved d18O and accumulation at ten
Greenland core sites (the spatial axis). GISP2 Summit chemistry gives eight
co-registered major ions plus dust and volcanic markers at one site (the
multi-field axis). Both are open HTTP, no authentication.
"""

import urllib.error
import urllib.request
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

NOAA_GREENLAND = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/"

# Greenland outline for the map panel. 50m resolution gives 2,240 vertices
# across 17 rings, enough to read the fjords at preview size.
NATURAL_EARTH_50M = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_50m_admin_0_countries.geojson"
)

# Osman et al. 2021, PNAS, "North Atlantic jet stream projections in the
# context of the past 1,250 years". Coordinates are read from each file's
# NOAA template header at parse time, not hardcoded here.
OSMAN_2021_FILES = [
    "act11d-2021accum.txt",
    "act11d-2021d18o.txt",
    "act2-2021accum.txt",
    "b19-2021accum.txt",
    "b19-2021d18o.txt",
    "d4-2021d18o.txt",
    "eurocore2021accum.txt",
    "eurocore2021d18o.txt",
    "humboldt2021accum.txt",
    "neem2021accum.txt",
    "neem2021d18o.txt",
    "nu-2021d18o.txt",
    "summit2021accum.txt",
    "summit2021d18o.txt",
    "tunu2021accum.txt",
    "tunu2021d18o.txt",
]

# GISP2 Summit. ionb.txt is the bi-yearly major-ion series over the top 200 m,
# which is the only one of these that resolves the industrial era.
GISP2_FILES = {
    "chem": ["ionb.txt", "iond.txt", "volcano.txt", "msacored.txt"],
    "physical": ["accum.txt", "tempertr.txt"],
}


def download_file(url: str, destination: Path) -> bool:
    if destination.exists():
        print(f"  skip (exists)  {destination.name}")
        return True
    try:
        urllib.request.urlretrieve(url, destination)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as error:
        print(f"  FAILED         {destination.name}: {error}")
        destination.unlink(missing_ok=True)
        return False
    print(f"  ok {destination.stat().st_size:>9,} B  {destination.name}")
    return True


def download_osman_array(output_root: Path) -> int:
    output_dir = output_root / "osman2021"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Osman et al. 2021 multi-site array -> {output_dir}")
    base = NOAA_GREENLAND + "osman2021/"
    return sum(download_file(base + name, output_dir / name) for name in OSMAN_2021_FILES)


def download_gisp2_multifield(output_root: Path) -> int:
    base = NOAA_GREENLAND + "summit/gisp2/"
    downloaded = 0
    for subdirectory, names in GISP2_FILES.items():
        output_dir = output_root / "gisp2" / subdirectory
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"GISP2 {subdirectory} -> {output_dir}")
        downloaded += sum(
            download_file(f"{base}{subdirectory}/{name}", output_dir / name) for name in names
        )
    return downloaded


def download_coastline(output_root: Path) -> int:
    output_dir = output_root / "naturalearth"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Natural Earth coastline -> {output_dir}")
    return int(download_file(NATURAL_EARTH_50M, output_dir / "ne_50m_admin_0_countries.geojson"))


def main() -> None:
    output_root = DATASETS_DIR / "paleoclimate"
    total_expected = len(OSMAN_2021_FILES) + sum(len(v) for v in GISP2_FILES.values())
    downloaded = download_osman_array(output_root) + download_gisp2_multifield(output_root)
    print(f"\n{downloaded}/{total_expected} paleoclimate files present in {output_root}")
    download_coastline(DATASETS_DIR / "geometry")


if __name__ == "__main__":
    main()
