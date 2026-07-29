"""Greenland outline and site placement in EPSG:3413."""

import json
from pathlib import Path

import numpy as np

from .projection import project_to_kilometres

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"
COASTLINE_PATH = (
    DATASETS_DIR / "geometry" / "naturalearth" / "ne_50m_admin_0_countries.geojson"
)


def load_greenland_rings(coastline_path: Path | None = None) -> list[np.ndarray]:
    """Greenland's boundary rings, each an (n, 2) array of EPSG:3413 kilometres.

    Returned as separate rings rather than one array so plotting does not draw
    spurious segments between the mainland and the offshore islands.
    """
    path = coastline_path or COASTLINE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/download_preview_data.py first."
        )

    with open(path, encoding="utf-8") as handle:
        collection = json.load(handle)

    feature = next(
        f for f in collection["features"] if f["properties"].get("NAME") == "Greenland"
    )
    geometry = feature["geometry"]
    polygons = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )

    rings = []
    for polygon in polygons:
        for ring in polygon:
            coordinates = np.asarray(ring, dtype=float)
            x_km, y_km = project_to_kilometres(coordinates[:, 0], coordinates[:, 1])
            rings.append(np.column_stack([x_km, y_km]))
    return rings


def dodge_labels_vertically(anchor_y: np.ndarray, minimum_separation_km: float) -> np.ndarray:
    """Nudge label anchors apart so no two collide.

    Needed because Summit2010 and Eurocore2015 sit at identical coordinates
    (both are Summit cores) and three other pairs fall within a label height of
    each other. Relaxes in sorted order, which is stable and keeps each label
    near its own marker.
    """
    order = np.argsort(anchor_y)
    adjusted = np.asarray(anchor_y, dtype=float).copy()
    for position in range(1, len(order)):
        previous, current = order[position - 1], order[position]
        if adjusted[current] - adjusted[previous] < minimum_separation_km:
            adjusted[current] = adjusted[previous] + minimum_separation_km
    return adjusted


def project_site_metadata(site_metadata):
    """Add EPSG:3413 x_km / y_km columns to an Osman site metadata frame."""
    projected = site_metadata.copy()
    x_km, y_km = project_to_kilometres(
        projected["longitude"].to_numpy(), projected["latitude"].to_numpy()
    )
    projected["x_km"] = x_km
    projected["y_km"] = y_km
    return projected
