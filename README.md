# Greenland Art Project

A multi-sensory artistic installation exploring the Greenland Ice Sheet through simulation/observation data, climate history, and AI-driven data analysis.

## Project Overview

This project presents the Greenland Ice Sheet from multiple perspectives:
- Current ice sheet dynamics from satellite and in-situ observations
- Historical evolution going back millions of years
- Future projections under various climate scenarios
- AI-driven pattern discovery in high-dimensional climate data

The final exhibition aims to create an immersive, multi-sensory experience that helps viewers feel the scale and urgency of Greenland's transformation.

## Project Structure

```
greenland_art/
├── README.md              # This file
├── pyproject.toml         # Python project configuration (uv)
├── scripts/               # Data download and processing scripts
├── datasets/              # Symlink to ~/dataset/greenland_art
│                         # (actual data lives outside the repo)
├── src/                   # Source code for analysis and visualization
│   └── greenland_art/     # Main package
│       ├── __init__.py
│       ├── data/          # Data loading and processing
│       ├── analysis/      # Data analysis modules
│       └── visualization/ # Visualization utilities
└── tests/                 # Test suite
```

## Data Sources

### Ice Sheet Observations
- **NSIDC DAAC**: Satellite data (GRACE, ICESat, CryoSat)
- **PROMICE**: Glacier mass balance and velocity
- **MEaSUREs**: Ice sheet velocity and thickness changes

### Paleoclimate Data (DOWNLOADED ✅)
- **NOAA Paleoclimatology**: Ice cores, sediment cores, proxies
- **Ice core records**: GISP2, GRIP, NGRIP (100,000+ years)
- **Deep sea sediments**: Million-year records

#### Ice Core Data Status (July 2026)
We have successfully downloaded and verified the following datasets:

| Dataset | Time Span | Samples | δ¹⁸O Range | Description |
|---------|-----------|---------|------------|-------------|
| **GISP2** | 111 ka | 1,390 | -43.26 to -33.41‰ | Full glacial-interglacial cycle |
| **GRIP** | 249 ka | 5,425 | (not yet analyzed) | Extended record, multiple glacial cycles |
| **NGRIP** | 123 ka | (loading issue) | - | High-resolution at 50-year intervals |

**Key findings:**
- The GISP2 data shows clear climate shifts between glacial (~-40‰) and interglacial (~-35‰) periods
- Last Glacial Maximum (26-19 ka): mean δ¹⁸O = -40.56‰
- Holocene (0-11.7 ka): mean δ¹⁸O = -34.90‰
- The 10‰ variation encodes temperature changes of ~10-15°C

**Note on time scales:**
- Ice cores from Greenland give us ~100k-250k years (not millions)
- For millions-of-years records, deep sea sediment cores are needed
- Greenland Ice Sheet itself formed ~2.5-3 million years ago
- For ice sheet dynamics (surges, retreats), the 100k record is most relevant

### Ocean & Climate
- **CarbonTracker**: CO2 fluxes and atmospheric carbon
- **AMOC data**: Atlantic Meridional Overturning Circulation
- **RACMO**: Regional climate model outputs

## Technology Stack

- **Python** (managed via `uv`)
- **Xarray**: N-dimensional climate data arrays
- **Dask**: Distributed computing for large datasets
- **Zarr**: Chunked array storage
- **hvPlot/Panel**: Interactive visualization
- **UMAP/t-SNE**: Dimensionality reduction
- **librosa**: Audio/sound analysis

## Setup

```bash
# Install dependencies
uv sync

# Activate environment
uv run python -m greenland_art
```

## Current Status

This project is in the **exploration stage**. The focus is on:
1. Understanding what data is available
2. Exploring Human + AI collaboration for data interpretation
3. Designing multi-sensory exhibition components

## License

TBD