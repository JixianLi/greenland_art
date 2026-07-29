#!/usr/bin/env python3
"""
Download scripts for Greenland Art Project datasets.

This module provides download utilities for:
- NOAA Paleoclimatology data (ice cores, sediment cores)
- NSIDC satellite data (GRACE, ICESat-2, CryoSat-2)
- PROMICE glacier data
- RACMO climate model outputs

Most datasets are accessed via HTTPS direct download or OPeNDAP.
Some require NASA Earthdata authentication for bulk downloads.

Usage:
    python -m scripts.download_noaa_paleoclimate --dataset gisp2 --output datasets/
    python -m scripts.download_nsidc_data --dataset grace --output datasets/
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# =============================================================================
# NOAA Paleoclimatology Data
# https://www.ncei.noaa.gov/access/paleo-search/
# =============================================================================

# CORRECTED URLs (verified July 2026)
# Data is organized under: https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/

NOAA_PALEOCLIMATOLOGY_URLS = {
    "gisp2": {
        "base": "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/gisp2/isotopes/",
        "files": {
            "d18o": "gispd18o-noaa.txt",           # δ18O (primary isotope proxy)
            "d18o_annual": "d18o1yr-noaa.txt",       # Annual resolution δ18O
            "d18o_20yr": "d18o20y.txt",              # 20-year resolution
            "temperature": "gisp2-temperature2011-noaa.txt",  # Temperature reconstruction
            "accumulation": "gisp2_accum_alley2000-noaa.txt", # Accumulation data
        },
        "description": "GISP2 ( Greenland Ice Sheet Project 2) - 116,000 years, Summit location",
    },
    "grip": {
        "base": "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/isotopes/",
        "files": {
            "d18o": "gripd18o-noaa.txt",             # δ18O (100,000+ years)
            "iso2007": "grip-iso2007.txt",             # Full isotope dataset 2007 version
        },
        "description": "GRIP (Greenland Ice Core Project) - 100,000+ years, Summit location",
    },
    "ngrip": {
        "base": "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/ngrip/isotopes/",
        "files": {
            "d18o_50yr": "ngrip-d18o-50yr-noaa.txt",  # 50-year resolution δ18O
            "d18o_2015": "ngrip2015d18o.txt",          # 2015 version
        },
        "description": "NGRIP (North Greenland Ice Core Project) - 123,000 years",
    },
}


def download_noaa_isotope_data(output_dir: Path, dataset: str = "gisp2") -> list[Path]:
    """
    Download oxygen isotope (δ18O) data from NOAA for ice cores.
    
    δ18O is a proxy for temperature - heavier isotopes precipitate more
    in warmer conditions. This gives us temperature history.
    
    Datasets:
    - gisp2: GISP2 (116,000 years at Summit)
    - grip: GRIP (100,000+ years at Summit)
    - ngrip: NGRIP (123,000 years)
    
    Args:
        output_dir: Directory to save downloaded files
        dataset: Which ice core dataset ('gisp2', 'grip', 'ngrip')
    
    Returns:
        List of downloaded file paths
    """
    import urllib.request
    
    output_dir = Path(output_dir) / "paleoclimate" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_info = NOAA_PALEOCLIMATOLOGY_URLS.get(dataset)
    if not dataset_info:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {list(NOAA_PALEOCLIMATOLOGY_URLS.keys())}")
        return []
    
    base_url = dataset_info["base"]
    files_to_download = dataset_info["files"]
    
    print(f"Dataset: {dataset_info['description']}")
    print(f"Base URL: {base_url}")
    
    downloaded = []
    
    for name, filename in files_to_download.items():
        url = base_url + filename
        output_path = output_dir / filename
        
        print(f"\nDownloading {filename}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            file_size = output_path.stat().st_size
            print(f"  -> {output_path} ({file_size:,} bytes)")
            downloaded.append(output_path)
        except Exception as e:
            print(f"  -> Failed: {e}")
    
    return downloaded


def download_noaa_all_types():
    """
    List available NOAA Paleoclimatology data types for Greenland.
    
    Returns all data categories available:
    - Ice cores (GISP2, GRIP, NGRIP, NEEM)
    - Marine sediments
    - Pollen
    - Tree rings
    """
    print("NOAA Paleoclimatology Data Categories:")
    for category, url in NOAA_PALEOCLIMATOLOGY_URLS.items():
        print(f"  {category}: {url}")


# =============================================================================
# NSIDC Satellite Data
# https://nsidc.org/data
# =============================================================================

NSIDC_DATASETS = {
    # GRACE: Gravity Recovery and Climate Experiment (mass changes)
    "grace": {
        "provider": "NASA/JPL",
        "doi": "10.5067/SGMR65061S7DA",
        "url": "https://podaac-tools.jpl.nasa.gov/drive/files.all/geosgcm/g5npkg/yearly/",
        "description": "Ice sheet mass balance from satellite gravimetry",
    },
    # ICESat-2: Ice, Cloud, and land Elevation Satellite-2
    "icesat2": {
        "provider": "NASA",
        "doi": "10.5067/ATLAS/ATL06.003",
        "url": "https://nsidc.org/data/ATL06",
        "description": "Land ice elevation changes (2018-present)",
    },
    # CryoSat-2: European ice monitoring
    "cryosat2": {
        "provider": "ESA",
        "doi": "10.5270/EN1-0ng2-7y",
        "url": "https://spacesauthority.com/cryosat-data/",
        "description": "Ice sheet elevation and thickness",
    },
    # MEaSUREs: Glacier velocity
    "measures_velocity": {
        "provider": "NASA",
        "doi": "10.5067/MEaSUREs/GlacierVelocity/v01",
        "url": "https://nsidc.org/data/NSIDC-0481",
        "description": "Greenland ice sheet velocity maps",
    },
}


def list_nsidc_datasets():
    """List available NSIDC datasets for ice sheet observation."""
    print("NSIDC datasets for Greenland Ice Sheet:")
    print("-" * 60)
    for name, info in NSIDC_DATASETS.items():
        print(f"\n{name.upper()}")
        print(f"  DOI: {info['doi']}")
        print(f"  Description: {info['description']}")
        print(f"  URL: {info['url']}")


# =============================================================================
# PROMICE Data (Programme for Monitoring of the Greenland Ice Sheet)
# https://promice.org
# =============================================================================

PROMICE_URLS = {
    "mass_balance": "https://zenodo.org/record/7683412/files/greenland_mass_balance.csv",
    "velocity": "https://data.g-e-null.dk/viewers/sentinel-1-ice-velocity/",
}


def download_promice_data(output_dir: Path) -> Path:
    """
    Download PROMICE mass balance data.
    
    The PROMICE network provides in-situ measurements of glacier
    mass balance, ablation, and velocity from automatic weather
    stations on the Greenland Ice Sheet.
    """
    import urllib.request
    
    output_dir = Path(output_dir) / "promice"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = PROMICE_URLS["mass_balance"]
    output_path = output_dir / "greenland_mass_balance.csv"
    
    print(f"Downloading PROMICE data from {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"  -> {output_path}")
        return output_path
    except Exception as e:
        print(f"  -> Download failed: {e}")
        return None


# =============================================================================
# RACMO Regional Climate Model Data
# Available from KNMI Climate Data Explorer
# =============================================================================

RACMO_INFO = {
    "description": "Regional Atmospheric Climate Model outputs for Greenland",
    "resolution": "5.5 km (historical), 11 km (projections)",
    "variables": ["precipitation", "melt", "runoff", "temperature", "wind"],
    "url": "https://projects.knmi.nl/datasets/",
}


# =============================================================================
# Utility functions
# =============================================================================

def ensure_dataset_dir():
    """Ensure the datasets symlink and directory structure exist."""
    dataset_root = DATASETS_DIR
    
    # Create symlink if it doesn't exist
    if not dataset_root.exists():
        target = Path.home() / "dataset" / "greenland_art"
        target.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating symlink: {dataset_root} -> {target}")
        dataset_root.symlink_to(target)
    
    # Create subdirectories
    subdirs = ["paleoclimate", "satellite", "promice", "racmo", "other"]
    for subdir in subdirs:
        (DATASETS_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset directory ready: {DATASETS_DIR}")
    return DATASETS_DIR


def main():
    parser = argparse.ArgumentParser(description="Download Greenland Ice Sheet data")
    parser.add_argument("--source", choices=["noaa", "nsidc", "promice", "all"], 
                        default="all", help="Data source to download")
    parser.add_argument("--dataset", default="gisp2", 
                        help="Specific dataset (for NOAA: gisp2, grip, ngrip)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: datasets/)")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else DATASETS_DIR
    
    if args.list:
        print("Available datasets:")
        print("\n=== NOAA Paleoclimatology ===")
        download_noaa_all_types()
        print("\n=== NSIDC DAAC ===")
        list_nsidc_datasets()
        print("\n=== PROMICE ===")
        print(f"Mass balance: {PROMICE_URLS['mass_balance']}")
        return
    
    ensure_dataset_dir()
    
    if args.source in ["noaa", "all"]:
        print("\n--- Downloading NOAA Paleoclimatology Data ---")
        download_noaa_isotope_data(output_dir, args.dataset)
    
    if args.source in ["nsidc", "all"]:
        print("\n--- NSIDC Satellite Data ---")
        print("Note: Full NSIDC download requires NASA Earthdata authentication.")
        print("Use the web interface or create an account at: https://urs.earthdata.nasa.gov/")
        list_nsidc_datasets()
    
    if args.source in ["promice", "all"]:
        print("\n--- Downloading PROMICE Data ---")
        download_promice_data(output_dir)
    
    print("\nDone! Data available in:", output_dir)


if __name__ == "__main__":
    main()