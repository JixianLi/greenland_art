"""Window-versus-baseline anomalies, the quantity that links timeline to map."""

import numpy as np
import pandas as pd


def window_anomaly(
    site_by_year: pd.DataFrame,
    window: tuple[int, int],
    baseline: tuple[int, int],
    min_window_samples: int = 3,
    min_baseline_samples: int = 30,
) -> pd.Series:
    """Mean value inside `window` minus each site's own `baseline` mean.

    Differencing against a per-site baseline rather than a single ice-sheet-wide
    mean is what makes the sites comparable: d18O at ACT11d (-24 permil) and
    TUNU (-32 permil) differ mostly by elevation and latitude, so raw values
    would map site geography, not the climate signal.

    Sites with too little data in either period return NaN rather than a
    number computed from two or three years.
    """
    window_slice = site_by_year.loc[window[0] : window[1]]
    baseline_slice = site_by_year.loc[baseline[0] : baseline[1]]

    window_mean = window_slice.mean()
    baseline_mean = baseline_slice.mean()

    too_sparse = (window_slice.count() < min_window_samples) | (
        baseline_slice.count() < min_baseline_samples
    )
    return (window_mean - baseline_mean).mask(too_sparse)


def standardize_fields(field_matrix: pd.DataFrame) -> pd.DataFrame:
    """Z-score each field independently.

    Required before any joint multi-field analysis here: the eight GISP2 ion
    species span two orders of magnitude (magnesium ~1 ppb, nitrate ~80 ppb),
    so an unscaled matrix would let nitrate and sulfate dominate every
    covariance-based method applied to it.
    """
    return (field_matrix - field_matrix.mean()) / field_matrix.std(ddof=0)


def multifield_novelty(field_matrix: pd.DataFrame) -> pd.Series:
    """Per-sample novelty score: distance from the record's typical state.

    Euclidean norm of the standardized field vector. This is the cheap
    stand-in for a learned reconstruction error, and serves the same role in
    the interface: a track under the timeline saying "look here". Anything
    that flags here is a candidate, not a finding.
    """
    standardized = standardize_fields(field_matrix)
    return pd.Series(
        np.linalg.norm(standardized.fillna(0.0).to_numpy(), axis=1),
        index=field_matrix.index,
        name="novelty",
    )
