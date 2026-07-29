"""Polar stereographic projection for Greenland maps (EPSG:3413).

EPSG:3413 is the NSIDC Sea Ice Polar Stereographic North projection and the
conventional grid for Greenland ice sheet products (BedMachine, MEaSUREs,
RACMO). Plotting lon/lat directly on a Cartesian axis instead badly shears
Greenland, which spans 59 N to 84 N.

Ellipsoidal polar stereographic, north polar aspect, from Snyder, J.P. (1987),
"Map Projections: A Working Manual", USGS Professional Paper 1395, pp. 160-161
(equations 15-9, 14-15, 21-34, 21-35).

Symbols follow Snyder:
  a       WGS84 semi-major axis (m)
  e       first eccentricity
  phi     latitude (rad), lam  longitude (rad)
  phi_c   standard parallel where scale is true (70 N for EPSG:3413)
  lam_0   central meridian / longitude of projection origin (-45 E)
  t, m    Snyder's auxiliary functions
  rho     distance from the pole in the projection plane (m)
"""

import numpy as np

WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
WGS84_FLATTENING = 1 / 298.257223563
WGS84_ECCENTRICITY = np.sqrt(2 * WGS84_FLATTENING - WGS84_FLATTENING**2)

EPSG_3413_STANDARD_PARALLEL_DEG = 70.0
EPSG_3413_CENTRAL_MERIDIAN_DEG = -45.0


def _snyder_t(phi: np.ndarray, e: float) -> np.ndarray:
    """Snyder eq. 15-9."""
    sin_phi = np.sin(phi)
    return np.tan(np.pi / 4 - phi / 2) / ((1 - e * sin_phi) / (1 + e * sin_phi)) ** (e / 2)


def _snyder_m(phi: float, e: float) -> float:
    """Snyder eq. 14-15."""
    return np.cos(phi) / np.sqrt(1 - e**2 * np.sin(phi) ** 2)


def lonlat_to_polar_stereographic(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    standard_parallel_deg: float = EPSG_3413_STANDARD_PARALLEL_DEG,
    central_meridian_deg: float = EPSG_3413_CENTRAL_MERIDIAN_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Project geographic coordinates to EPSG:3413 metres.

    Returns (x, y) in metres. Accepts scalars or arrays.
    """
    a = WGS84_SEMI_MAJOR_AXIS_M
    e = WGS84_ECCENTRICITY

    phi = np.radians(np.asarray(latitude_deg, dtype=float))
    lam = np.radians(np.asarray(longitude_deg, dtype=float))
    phi_c = np.radians(standard_parallel_deg)
    lam_0 = np.radians(central_meridian_deg)

    rho = a * _snyder_m(phi_c, e) * _snyder_t(phi, e) / _snyder_t(np.array(phi_c), e)

    x = rho * np.sin(lam - lam_0)
    y = -rho * np.cos(lam - lam_0)
    return x, y


def project_to_kilometres(
    longitude_deg: np.ndarray, latitude_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """EPSG:3413 in kilometres, which keeps axis tick labels readable."""
    x, y = lonlat_to_polar_stereographic(longitude_deg, latitude_deg)
    return x / 1000.0, y / 1000.0
