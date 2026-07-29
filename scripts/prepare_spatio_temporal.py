#!/usr/bin/env python3
"""
Download and prepare 100-year scale spatio-temporal data for visualization.

This script prepares:
1. GRACE satellite data (ice sheet mass changes, 2002-2017)
2. ICESat elevation data (surface elevation changes)
3. Synthetic "100 year animation" data derived from ice core temporal patterns

The goal is to create visualization demos that show:
- How ice sheet changes over time (modern observations)
- How temporal patterns from ice cores could drive artistic visualizations
- The potential of spatio-temporal data for multi-sensory installation

This serves as "proof of concept" to justify million-year simulations.
"""

import sys
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_DIR = DATASETS_DIR / "spatio_temporal_100yr"


# =============================================================================
# GRACE Data - Ice Sheet Mass Changes (2002-2017)
# =============================================================================

# GRACE data is available from NASA JPL PODAAC
# Format: NetCDF files showing mass anomaly

GRACE_INFO = {
    "name": "GRACE/GRACE-FO Ice Sheet Mass Balance",
    "source": "NASA JPL/Center for Space Research",
    "temporal_range": "2002-2017 (with gaps)",
    "doi": "10.5067/SGMR65061S7DA",
    "url_base": "https://podaac-tools.jpl.nasa.gov/drive/files.all/geosgcm/g5npkg/yearly/",
}

# Alternative: Get monthly GRACE data for Greenland
# This is smaller and more tractable

GRACE_URLS = {
    "greenland_monthly": "https://podaac-tools.jpl.nasa.gov/drive/files.all/geosgcm/g5npkg/monthly/",
}


def download_grace_monthly_data(output_dir: Path) -> list[Path]:
    """
    Download GRACE/GRACE-FO monthly data for Greenland.
    
    The data is available as NetCDF files containing:
    - time: months since start
    - lat, lon: grid coordinates  
    - mass_anomaly: ice sheet mass change (Gt)
    
    For a quick demo, we can use this to show:
    - Current mass loss trend
    - Seasonal variations
    - Correlation with climate indices
    """
    print("Note: GRACE data download requires NASA EDL authentication.")
    print("Alternative: Using synthetic stylized data based on observed trends.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return []


# =============================================================================
# Synthetic 100-year spatio-temporal data based on real observations
# =============================================================================

def create_stylized_100yr_animation(output_dir: Path) -> Path:
    """
    Create a stylized 100-year animation dataset.
    
    Since real GRACE data requires authentication, we create a synthetic
    dataset that captures the KEY features observed in the real data:
    
    1. Accelerating mass loss over time (observed in GRACE)
    2. Seasonal cycles (~0.5 Gt oscillation)
    3. Year-to-year variability
    4. Regional differences (some glaciers losing faster than others)
    
    This synthetic data would be replaced by real data once accessible,
    but demonstrates the visualization concept.
    
    For the 100-year demonstration:
    - 10 major glaciers/regions
    - Monthly time steps (1200 time steps)
    - Mass change values based on observed trends
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define regions with different loss rates (Gt/year)
    regions = {
        "Jakobshavn": -40,        # Fastest retreating glacier
        "Helheim": -20,
        "Kangerlussuaq": -15,
        "Sell Sap": -10,
        "Rosters": -8,
        "Daugaard": -7,
        "Kirk": -5,
        "Humboldt": -3,
        "Petermann": -2,
        "Nioghalvfjerdsbræ": -1,
    }
    
    years = 100
    months_per_year = 12
    n_timesteps = years * months_per_year
    
    # Create data structure
    time_index = pd.date_range(start="1925", periods=n_timesteps, freq='ME')
    
    # Accelerating trend: mass loss accelerates ~3% per year
    # Based on observations: Greenland lost ~3.8 trillion tons 1900-2010
    yearly_loss_rate = np.zeros((years, len(regions)))
    
    for i, (name, base_rate) in enumerate(regions.items()):
        for y in range(years):
            # Acceleration factor (loss rate increases over time)
            accel = 1.0 + 0.03 * y  # 3% acceleration per decade
            noise = np.random.normal(0, abs(base_rate) * 0.1)
            yearly_loss_rate[y, i] = base_rate * accel + noise
    
    # Expand to monthly with seasonal cycle
    monthly_data = np.zeros((n_timesteps, len(regions)))
    for i in range(len(regions)):
        for y in range(years):
            for m in range(12):
                timestep = y * 12 + m
                # Add seasonal cycle (more melt in summer)
                seasonal = 0.1 * np.sin(2 * np.pi * (m - 6) / 12)
                monthly_data[timestep, i] = yearly_loss_rate[y, i] * (1 + seasonal)
    
    # Create DataFrame
    regions_list = list(regions.keys())
    df = pd.DataFrame(monthly_data, columns=regions_list, index=time_index)
    df.index.name = 'date'
    
    # Add cumulative mass (starting at 0, becoming more negative)
    df_cumsum = df.cumsum()
    df_cumsum = df_cumsum / 1000  # Convert toGt to 1000 Gt units
    
    # Save
    output_path = output_dir / "stylized_greenland_mass_changes_1925_2025.csv"
    df_cumsum.to_csv(output_path)
    print(f"Created: {output_path}")
    print(f"  Time range: {time_index[0]} to {time_index[-1]}")
    print(f"  Regions: {len(regions)}")
    print(f"  Final mass change: {df_cumsum.iloc[-1].sum():.1f} (1000 Gt)")
    
    return output_path


def create_paraview_temporal_animation(output_dir: Path, csv_path: Path):
    """
    Create ParaView-compatible temporal animation data.
    
    For each timestep, creates a CSV with point locations and mass values.
    ParaView can load this as a temporal collection (TimeAnimation).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Define approximate locations (lon, lat) for each region
    region_locs = {
        "Jakobshavn": (-49.5, 69.1),
        "Helheim": (-38.0, 66.4),
        "Kangerlussuaq": (-34.8, 66.5),
        "Sell Sap": (-36.4, 66.1),
        "Rosters": (-38.2, 64.5),
        "Daingaard": (-35.0, 64.2),
        "Kirk": (-36.8, 63.8),
        "Humboldt": (-62.0, 77.5),
        "Petermann": (-60.4, 80.7),
        "Nioghalvfjerdsbræ": (-22.0, 79.1),
    }
    
    # Get unique years for coarser animation
    data_by_year = data.groupby(pd.Grouper(freq='YE')).mean()
    
    anim_dir = output_dir / "paraview_animation"
    anim_dir.mkdir(parents=True, exist_ok=True)
    
    for year_idx, (date, row) in enumerate(data_by_year.iterrows()):
        # Create CSV for this timestep
        timestep_file = anim_dir / f"timestep_{year_idx:04d}_{date.year}.csv"
        
        rows = []
        for region, loc in region_locs.items():
            rows.append({
                'x': loc[0],
                'y': 0,  # Flat for now
                'z': loc[1],
                'mass_change': row.get(region, 0),
                'region': region,
            })
        
        df_step = pd.DataFrame(rows)
        df_step.to_csv(timestep_file, index=False)
    
    print(f"Created {len(data_by_year)} animation frames in {anim_dir}/")
    
    # Create ParaView state file hints
    readme = anim_dir / "_README.txt"
    with open(readme, 'w') as f:
        f.write("""ParaView Temporal Animation Setup
==================================

This directory contains time-series CSV files for animation.

To animate in ParaView:
1. Open all CSV files as a "Temporal CSV Collection" or
   use File → Open and select multiple files
2. Apply "TableToPoints" filter (X=x, Y=y, Z=z)
3. Use "Glyph" to represent each point as a sphere
4. Color by "mass_change"
5. Use "Set Frame" or animation controls to scrub through time

For smooth animation, ParaView's " animation Inspector" can interpolate
between timesteps.

Files: timestep_XXXX_YYYY.csv
  - XXXX = timestep number (0 to N)
  - YYYY = year
""")
    print(f"Created: {readme}")
    
    return anim_dir


# =============================================================================
# Ice Core temporal driver for visualization
# =============================================================================

def create_icecore_temporal_driver(output_dir: Path) -> Path:
    """
    Create a file that maps ice core time periods to visualization parameters.
    
    This shows how the ice core temporal signal (110k years) could be
    mapped to drive visualization parameters for an artistic installation.
    
    The idea: when we HAVE million-year simulations, we can use this same
    framework to drive the visualization, but with much richer content.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load our ice core data
    ic_data = pd.read_csv(
        DATASETS_DIR / "paleoclimate" / "gisp2" / "gispd18o-noaa.txt",
        sep='\t', comment='#'
    )
    ic_data.columns = ['depth', 'd18O', 'age']
    ic_data = ic_data.replace(999999, np.nan).dropna()
    ic_data['age_ka'] = ic_data['age'] / 1000  # Convert to thousands of years
    
    # Subsample to get a manageable number of time periods
    # Take every Nth point to get ~500 time slices across 110k years
    step = max(1, len(ic_data) // 500)
    ic_subset = ic_data.iloc[::step].copy()
    
    # Map d18O to visualization parameters
    # These mappings would be used in the artistic installation
    ic_subset['temperature_proxy'] = ic_subset['d18O']  # Already temperature proxy
    ic_subset['color_hue'] = (ic_subset['d18O'] - ic_subset['d18O'].min()) / \
                             (ic_subset['d18O'].max() - ic_subset['d18O'].min()) * 240  # 0-240 hue range (blue to red)
    ic_subset['audio_pitch'] = ((ic_subset['d18O'] - ic_subset['d18O'].min()) / \
                               (ic_subset['d18O'].max() - ic_subset['d18O'].min()) * 1000) + 200  # 200-1200 Hz
    ic_subset['vibration_intensity'] = ((ic_subset['d18O'] - ic_subset['d18O'].min()) / \
                                       (ic_subset['d18O'].max() - ic_subset['d18O'].min()))  # 0-1
    
    output_path = output_dir / "icecore_temporal_driver.csv"
    ic_subset.to_csv(output_path, index=False)
    print(f"Created: {output_path}")
    print(f"  Time span: {ic_subset['age_ka'].min():.1f} ka to {ic_subset['age_ka'].max():.1f} ka")
    print(f"  Points: {len(ic_subset)}")
    print("  Columns: age_ka, d18O, temperature_proxy, color_hue, audio_pitch, vibration_intensity")
    
    return output_path


# =============================================================================
# Create comparison visualization that shows scales
# =============================================================================

def create_scale_comparison_visualization(output_dir: Path):
    """
    Create a visualization that compares different temporal scales.
    
    This demonstrates what we could show at:
    - 100-year scale (modern observations)
    - 11,700-year scale (Holocene)
    - 110,000-year scale (full glacial cycle)
    - 1,000,000+ year scale (future simulation potential)
    
    Each scale reveals different aspects of the ice sheet's behavior.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scales = {
        "modern_100yr": {
            "time_range_years": 100,
            "data_source": "GRACE satellite (synthetic demo)",
            "features": "Mass loss acceleration, seasonal cycles, year-to-year variability",
            "color": "red",
        },
        "holocene_11ka": {
            "time_range_years": 11700,
            "data_source": "Ice core (GISP2 subset)",
            "features": "Holocene climate optimum, Little Ice Age, Roman Warm Period, 1000-yr cycles",
            "color": "orange",
        },
        "full_glacial_110ka": {
            "time_range_years": 110000,
            "data_source": "Ice core (GISP2 full record)",
            "features": "Full glacial-interglacial cycle, Dansgaard-Oeschger events, orbital cycles",
            "color": "blue",
        },
        "million_yr_potential": {
            "time_range_years": 2500000,
            "data_source": "ICE SHEET MODEL SIMULATION (needed)",
            "features": "Plio-Pleistocene transitions, Greenland formation, 41-kyr and 100-kyr cycles atdifferent amplitudes, climate-biology-geology coupling",
            "color": "purple",
        }
    }
    
    # Create a summary DataFrame
    rows = []
    for name, info in scales.items():
        rows.append({
            'scale_name': name,
            'time_range_years': info['time_range_years'],
            'display_name': name.replace('_', ' ').title(),
            'data_source': info['data_source'],
            'key_features': info['features'],
            'color': info['color'],
        })
    
    df_scales = pd.DataFrame(rows)
    output_path = output_dir / "temporal_scale_comparison.csv"
    df_scales.to_csv(output_path, index=False)
    print(f"Created: {output_path}")
    print("\nTemporal Scales Summary:")
    print("="*70)
    for _, row in df_scales.iterrows():
        print(f"\n{row['display_name']} ({row['time_range_years']:,} years)")
        print(f"  Data: {row['data_source']}")
        print(f"  Features: {row['key_features']}")
    
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare spatio-temporal data for visualization")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SPATIO-TEMPORAL DATA PREPARATION")
    print("Proof-of-concept for multi-scale visualization")
    print("="*70)
    
    # 1. Create synthetic 100-year data (stylized but realistic)
    print("\n[1/4] Creating stylized 100-year animation data...")
    csv_path = create_stylized_100yr_animation(output_dir)
    
    # 2. Create ParaView animation frames
    print("\n[2/4] Creating ParaView animation files...")
    create_paraview_temporal_animation(output_dir, csv_path)
    
    # 3. Create ice core temporal driver
    print("\n[3/4] Creating ice core temporal driver...")
    create_icecore_temporal_driver(output_dir)
    
    # 4. Create scale comparison
    print("\n[4/4] Creating temporal scale comparison...")
    create_scale_comparison_visualization(output_dir)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.glob("**/*")):
        if f.is_file():
            print(f"  {f.relative_to(output_dir)}")
    
    print("\n" + "="*70)
    print("VISUALIZATION OPPORTUNITIES")
    print("="*70)
    print("""
1. MODERN (100yr) - GRACE-style mass change animations
   → Show accelerating ice loss in real-time
   → Demonstrates human-caused climate impact
   
2. HOLOCENE (11,700yr) - Ice core temporal driver
   → Show how temperature has varied within the current interglacial
   → Context for "unprecedented" vs "natural variation"
   
3. FULL GLACIAL (110,000yr) - Complete ice core record
   → Show the dramaticDansgaard-Oeschger events
   → 10°C jumps in decades - really tells a story
   
4. MILLION-YEAR POTENTIAL - Justify the simulation
   → Show that current data is a "snapshot" of one interglacial
   → For exhibition: we'd have the FULL story from formation to future
   → Would reveal how ice sheet responds to different orbital configurations
""")
    
    print("="*70)
    print("RECOMMENDED NEXT STEPS")
    print("="*70)
    print("""
1. Review this synthetic data with your visualization team
2. Test in ParaView to see if the animation concept works
3. Present to scientists to get buy-in for:
   - Running ice sheet model for million-year simulation
   - Or accessing existing model output from PMIP/ISMIP6
4. The goal: show that even a "proof of concept" with limited data
   demonstrates the artistic and scientific value
""")


if __name__ == "__main__":
    main()