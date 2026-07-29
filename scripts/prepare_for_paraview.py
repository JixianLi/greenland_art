#!/usr/bin/env python3
"""
Prepare ice core data for ParaView visualization.

ParaView expects CSV or VTK file formats. This script:
1. Loads the ice core isotope data
2. Creates a clean CSV with columns ParaView can easily import
3. Adds derived variables (temperature anomaly, normalized values)
4. Creates a parametric 3D curve representation

Usage:
    uv run python scripts/prepare_for_paraview.py --dataset gisp2
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_DIR = Path(__file__).parent.parent / "datasets" / "paraview"


def load_gisp2():
    """Load GISP2 ice core data."""
    filepath = DATASETS_DIR / "paleoclimate" / "gisp2" / "gispd18o-noaa.txt"
    
    # Skip comment lines (starting with #) 
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the header line (column names)
    data_start = 0
    for i, line in enumerate(lines):
        if not line.startswith('#') and 'depth_top_m' in line:
            data_start = i
            break
    
    # Read the data starting from the header line
    data = pd.read_csv(filepath, sep='\t', skiprows=data_start, comment='#')
    
    # Clean up - rename columns if needed
    data.columns = [c.strip() for c in data.columns]
    
    # Remove any rows with missing values
    data = data.replace(999999, np.nan)
    data = data.dropna()
    data = data.reset_index(drop=True)
    
    return data


def create_paraview_csv(data: pd.DataFrame, output_path: Path, dataset_name: str):
    """
    Create a CSV file optimized for ParaView import.
    
    ParaView can load CSV files via File → Open or drag-and-drop.
    We'll set up:
    - Time (age in ka - thousands of years)
    - δ18O value
    - Temperature anomaly (estimated from δ18O)
    - Normalized position for 3D curve
    """
    
    # Calculate derived values
    output = pd.DataFrame()
    
    # Time axis (convert to thousands of years for cleaner visualization)
    output['age_ka'] = data['age_top_calBP'] / 1000.0
    
    # Original δ18O value
    output['d18O'] = data['d18O_smow']
    
    # Estimated temperature anomaly (rough conversion: ~0.6‰ per °C)
    # More negative δ18O = colder, so we invert this relationship
    # Baseline (Holocene mean) is about -34.9‰
    baseline_d18O = -34.9  # Holocene mean
    output['temp_anomaly_C'] = (output['d18O'] - baseline_d18O) / 0.6
    
    # Normalized time for curve parameterization (0 to 1 over full record)
    t_norm = (output['age_ka'] - output['age_ka'].min()) / (output['age_ka'].max() - output['age_ka'].min())
    output['t_norm'] = t_norm
    
    # 3D curve coordinates
    # Age mapped to Z axis (depth into time)
    # d18O mapped to X
    # A small periodic component in Y for visual interest
    output['X'] = output['d18O']
    output['Y'] = np.sin(output['t_norm'] * np.pi * 20) * 0.5  # 20 oscillations
    output['Z'] = output['age_ka']
    
    # Color mapping value (normalized d18O for coloring)
    # More positive = warmer, more negative = cooler
    output['color_val'] = (output['d18O'] - output['d18O'].min()) / (output['d18O'].max() - output['d18O'].min())
    
    # Save CSV for ParaView
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved ParaView CSV: {output_path}")
    print(f"  Rows: {len(output)}")
    print(f"  Columns: {list(output.columns)}")
    
    return output


def create_vtk_polydata(data: pd.DataFrame, output_path: Path, dataset_name: str):
    """
    Create a VTK file with a parametric curve for 3D visualization.
    
    This creates a VTK file with:
    - Points along a curve where Z = age, X = d18O
    - Point data scalars for d18O, temperature
    - A polyline connecting the points
    
    ParaView can open .vtk files directly and render them as 3D curves.
    """
    try:
        from pyevtk.hl import pointsToVTK, lineToVTK
        print("pyevtk found - can create VTK files")
    except ImportError:
        print("pyevtk not installed - will only create CSV")
        return None
    
    # Prepare arrays - age goes in Z
    n_points = len(data)
    
    # Coordinates
    x = data['d18O'].values
    y = np.sin(np.linspace(0, 2*np.pi, n_points)) * 0.5
    z = data['age_top_calBP'].values / 1000.0  # Convert to ka for reasonable scale
    
    # Point data
    point_data = {
        'd18O': data['d18O_smow'].values,
        'temperature_anomaly': (data['d18O_smow'].values - (-34.9)) / 0.6,
        'age_ka': data['age_top_calBP'].values / 1000.0,
    }
    
    # Create VTK file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use pointsToVTK for scattered points
    pointsToVTK(str(output_path).replace('.vtk', '_points'), x, y, z, pointData=point_data)
    print(f"Saved VTK points: {output_path}")
    
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare ice core data for ParaView")
    parser.add_argument("--dataset", default="gisp2", choices=["gisp2", "grip", "ngrip"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {args.dataset.upper()} data...")
    
    if args.dataset == "gisp2":
        data = load_gisp2()
    else:
        print(f"Loading {args.dataset} not implemented yet - using GISP2 as example")
        data = load_gisp2()
    
    print(f"Loaded {len(data)} samples")
    print(f"Age range: {data['age_top_calBP'].max()/1000:.1f} ka to {data['age_top_calBP'].min()/1000:.1f} ka")
    
    # Create CSV
    csv_path = output_dir / f"{args.dataset}_paraview.csv"
    create_paraview_csv(data, csv_path, args.dataset)
    
    # Try to create VTK
    vtk_path = output_dir / f"{args.dataset}_curve.vtk"
    result = create_vtk_polydata(data, vtk_path, args.dataset)
    
    print(f"\n=== Files created in {output_dir} ===")
    for f in sorted(output_dir.glob(f"*")):
        print(f"  {f.name}")
    
    print("\n=== ParaView Quick Start ===")
    print(f"1. Open ParaView")
    print(f"2. File → Open → Select '{csv_path.name}'")
    print(f"3. In the 'Properties' tab, check 'Have Headers'")
    print(f"4. Click 'Apply'")
    print(f"5. Use 'Spreadsheet View' or create a plot:")
    print(f"   - Select the filter 'Plot Data' to see isotope values vs age")
    print(f"   - Or use 'TableToPoints' to see as 3D scatter")
    print(f"\nFor a curve visualization (more artistic):")
    print(f"   - Use 'TableToPoints' with X=d18O, Y=0, Z=age_ka")
    print(f"   - Then 'Plot Line' to connect points into a curve")
    print(f"   - Color by 'd18O' or 'temp_anomaly_C'")


if __name__ == "__main__":
    main()