#!/usr/bin/env python3
"""
Exploration script for Greenland ice core data.

This script loads and visualizes the GISP2, GRIP, and NGRIP ice core data
to understand the temporal structure and δ18O patterns.

δ18O (delta 18O) is a proxy for temperature:
- More negative values = cooler temperatures (glacial periods)
- More positive values = warmer temperatures (interglacial periods)

Usage:
    uv run python scripts/explore_ice_core_data.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def load_gisp2_isotope():
    """Load GISP2 isotope data."""
    filepath = DATASETS_DIR / "paleoclimate" / "gisp2" / "gispd18o-noaa.txt"
    
    # Skip header lines (starting with #) and load data
    data = pd.read_csv(filepath, sep='\t', comment='#')
    data.columns = ['depth_top_m', 'd18O_smow', 'age_top_calBP']
    
    # Replace missing values (999999) with NaN
    data = data.replace(999999, np.nan)
    data = data.dropna()
    
    return data


def load_grip_isotope():
    """Load GRIP isotope data."""
    filepath = DATASETS_DIR / "paleoclimate" / "grip" / "gripd18o-noaa.txt"
    
    # Load the data - it should have similar format
    data = pd.read_csv(filepath, sep='\t', comment='#')
    data.columns = ['depth_top_m', 'd18O_smow', 'age_top_calBP']
    
    data = data.replace(999999, np.nan)
    data = data.dropna()
    
    return data


def load_ngrip_isotope():
    """Load NGRIP isotope data."""
    filepath = DATASETS_DIR / "paleoclimate" / "ngrip" / "ngrip-d18o-50yr-noaa.txt"
    
    data = pd.read_csv(filepath, sep='\t', comment='#')
    data.columns = ['depth_top_m', 'd18O_smow', 'age_top_calBP']
    
    data = data.replace(999999, np.nan)
    data = data.dropna()
    
    return data


def plot_timeseries(data, title, filename=None):
    """Plot δ18O time series."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Sort by age (ascending - oldest at left)
    data = data.sort_values('age_top_calBP')
    
    # Plot age in thousands of years
    age_kyr = data['age_top_calBP'] / 1000
    
    ax.plot(age_kyr, data['d18O_smow'], 'b-', linewidth=0.5, alpha=0.8)
    ax.fill_between(age_kyr, data['d18O_smow'], -40, alpha=0.3)
    
    ax.set_xlabel('Age (thousands of years before present)')
    ax.set_ylabel('δ¹⁸O (‰ SMOW)')
    ax.set_title(title)
    
    # Add climate period annotations
    ax.axhline(y=-32, color='gray', linestyle='--', alpha=0.5, label='Glacial threshold')
    ax.axhline(y=-28, color='gray', linestyle='--', alpha=0.5, label='Interglacial threshold')
    
    # Mark key time periods
    ax.annotate('Last Glacial\n(~115-11.7 ka)', xy=(60, -36), fontsize=9, ha='center')
    ax.annotate('Holocene\n(~11.7 ka-present)', xy=(5, -30), fontsize=9, ha='center')
    ax.annotate('Eemian\n(~130-115 ka)', xy=(122, -29), fontsize=9, ha='center')
    
    ax.set_xlim(0, age_kyr.max())
    ax.set_ylim(-40, -26)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig


def analyze_climate_periods(data):
    """Analyze different climate periods in the data."""
    print("\n" + "="*60)
    print("CLIMATE PERIOD ANALYSIS")
    print("="*60)
    
    # Define time periods
    periods = {
        'Last Glacial Maximum (26-19 ka)': (19000, 26000),
        'Deglaciation (11.7-19 ka)': (11700, 19000),
        'Holocene (0-11.7 ka)': (0, 11700),
        'Eemian Interglacial (115-130 ka)': (115000, 130000),
    }
    
    for name, (young_limit, old_limit) in periods.items():
        mask = (data['age_top_calBP'] >= young_limit) & (data['age_top_calBP'] < old_limit)
        period_data = data[mask]['d18O_smow']
        
        if len(period_data) > 0:
            print(f"\n{name}:")
            print(f"  n = {len(period_data)} samples")
            print(f"  δ¹⁸O mean = {period_data.mean():.2f}‰")
            print(f"  δ¹⁸O std = {period_data.std():.2f}‰")
            print(f"  δ¹⁸O range = [{period_data.min():.2f}, {period_data.max():.2f}]")


def find_doe_events(data, threshold=2.0):
    """
    Find Dansgaard-Oeschger events (rapid warming events in the record).
    
    DO events are characterized by rapid transitions from cold (stadial) 
    to warm (interstadial) conditions, typically within a few decades.
    """
    # Sort by age
    data = data.sort_values('age_top_calBP')
    
    # Calculate running mean to reduce noise
    window = 50  # ~50 year running mean
    data['d18O_smooth'] = data['d18O_smow'].rolling(window, center=True, min_periods=1).mean()
    
    # Find large upward jumps (rapid warming)
    diff = data['d18O_smooth'].diff()
    large_jumps = diff > threshold
    
    jump_ages = data.loc[large_jumps, 'age_top_calBP'].values
    
    print(f"\nFound {len(jump_ages)} potential DO events (threshold: {threshold}‰)")
    print("Sample of event ages (ka):")
    for age in jump_ages[:5]:
        print(f"  {age/1000:.1f} ka")


def main():
    print("Loading Greenland ice core data...")
    print("="*60)
    
    # Load data
    try:
        gisp2 = load_gisp2_isotope()
        print(f"GISP2: {len(gisp2)} samples, age range: {gisp2['age_top_calBP'].max()/1000:.1f} ka to {gisp2['age_top_calBP'].min()/1000:.1f} ka")
    except Exception as e:
        print(f"Error loading GISP2: {e}")
        gisp2 = None
    
    try:
        grip = load_grip_isotope()
        print(f"GRIP: {len(grip)} samples, age range: {grip['age_top_calBP'].max()/1000:.1f} ka to {grip['age_top_calBP'].min()/1000:.1f} ka")
    except Exception as e:
        print(f"Error loading GRIP: {e}")
        grip = None
    
    try:
        ngrip = load_ngrip_isotope()
        print(f"NGRIP: {len(ngrip)} samples, age range: {ngrip['age_top_calBP'].max()/1000:.1f} ka to {ngrip['age_top_calBP'].min()/1000:.1f} ka")
    except Exception as e:
        print(f"Error loading NGRIP: {e}")
        ngrip = None
    
    # Analyze GISP2 data
    if gisp2 is not None:
        print("\n" + "="*60)
        print("GISP2 DATA SUMMARY")
        print("="*60)
        print(f"Time span: {gisp2['age_top_calBP'].max()/1000:.0f} ka to {gisp2['age_top_calBP'].min()/1000:.0f} ka")
        print(f"Total duration: {(gisp2['age_top_calBP'].max() - gisp2['age_top_calBP'].min())/1000:.0f} thousand years")
        print(f"δ¹⁸O range: {gisp2['d18O_smow'].min():.2f} to {gisp2['d18O_smow'].max():.2f}‰")
        print(f"Mean δ¹⁸O: {gisp2['d18O_smow'].mean():.2f}‰")
        
        # Analyze climate periods
        analyze_climate_periods(gisp2)
        
        # Find DO events
        find_doe_events(gisp2)
        
        # Create visualization
        print("\nGenerating visualizations...")
        fig = plot_timeseries(
            gisp2, 
            'GISP2 Ice Core δ¹⁸O Record (110,000 years)',
            'gisp2_timeseries.png'
        )


if __name__ == "__main__":
    main()