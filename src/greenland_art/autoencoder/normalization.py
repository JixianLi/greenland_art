"""Input normalisation schemes, fitted on training data only.

Every scheme is the same two steps -- an optional log compression, then a linear
rescale -- so they live in one class rather than four. What varies is whether
the log runs and whether the rescale is mean/std or min/max:

    zscore         (x - mean) / std                       the original baseline
    log1p_zscore   log first, then mean/std
    minmax         (x - min) / (max - min)                to [0, 1]
    log1p_minmax   log first, then min/max

Why the log step is not plain ``log1p``. Two measured reasons:

1. 60 of the 155 MAR columns take negative values -- surface temperature reaches
   -60 C -- and ``log1p`` is undefined below -1. Hence the signed form,
   ``sign(x) * log1p(|x|)``, which is smooth through zero and maps 0 to 0, so
   the zero-inflated columns keep their spike at zero rather than acquiring a
   discontinuity.

2. ``log1p`` only compresses where ``|x| >> 1``, and 26 columns have 99th
   percentile magnitudes below 0.1 in their native units. Cloud ice QI spans
   7.7e-14 to 3.2e-6 kg/kg: ``log1p`` leaves its dynamic range at 3.3 decades,
   completely untouched. Those are precisely the heavy-tailed columns the log is
   meant to fix, so the transform would look like it had failed when in fact it
   never engaged. Dividing by a robust per-column scale first makes the log bite
   regardless of the physical units the field happens to be recorded in.

The scale is the median of |x| over nonzero entries, which is chosen for being
indifferent to the extreme tail -- a mean would be dragged by the same outliers
the transform exists to tame.

All statistics are fitted on the training split alone and applied unchanged to
validation, so nothing about the held-out rows leaks into the scaling.
"""

import numpy as np

SCHEMES = ("zscore", "log1p_zscore", "minmax", "log1p_minmax")

ROBUST_SCALE_ROWS = 200_000


def _robust_scale(features: np.ndarray, seed: int) -> np.ndarray:
    """Median |x| over nonzero entries, per column.

    Subsampled: this is a scale estimate, and a median over three million rows
    costs a full sort of the matrix for digits that do not change the result.
    """
    rows = features
    if len(features) > ROBUST_SCALE_ROWS:
        index = np.random.default_rng(seed).choice(len(features), ROBUST_SCALE_ROWS, replace=False)
        rows = features[index]

    magnitude = np.abs(rows.astype(np.float64))
    scale = np.ones(features.shape[1], dtype=np.float64)
    for column in range(features.shape[1]):
        nonzero = magnitude[:, column][magnitude[:, column] > 0.0]
        if nonzero.size:
            scale[column] = np.median(nonzero)
    # An all-zero column, or one whose median underflows, would divide by zero.
    scale[~(scale > 0.0)] = 1.0
    return scale


class Normalization:
    """Fitted normalisation with an exact inverse.

    The inverse matters as much as the forward direction here: explained
    variance measured in transformed space is not comparable between schemes,
    because each scheme decides how much weight a column carries. Comparing
    schemes at all requires mapping reconstructions back to physical units,
    which is what inverse_transform is for.
    """

    def __init__(self, scheme: str, log_scale, offset: np.ndarray, denominator: np.ndarray):
        self.scheme = scheme
        self.log_scale = log_scale
        self.offset = offset
        self.denominator = denominator

    @classmethod
    def fit(cls, features: np.ndarray, scheme: str = "zscore", seed: int = 0) -> "Normalization":
        if scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")

        log_scale = _robust_scale(features, seed) if scheme.startswith("log1p_") else None
        compressed = cls._compress(features, log_scale)

        if scheme.endswith("zscore"):
            # float64 accumulation: summing millions of float32 values overflows
            # silently to inf, which then propagates as NaN through the division.
            offset = compressed.mean(axis=0, dtype=np.float64)
            denominator = compressed.std(axis=0, dtype=np.float64)
        else:
            offset = compressed.min(axis=0).astype(np.float64)
            denominator = compressed.max(axis=0).astype(np.float64) - offset

        # A constant column would divide by zero; leaving the denominator at 1
        # maps it to a constant column, which every model handles harmlessly.
        denominator[~(denominator > 0.0)] = 1.0
        return cls(scheme, log_scale, offset, denominator)

    @staticmethod
    def _compress(features: np.ndarray, log_scale) -> np.ndarray:
        if log_scale is None:
            return features
        return np.sign(features) * np.log1p(np.abs(features) / log_scale)

    @staticmethod
    def _expand(values: np.ndarray, log_scale) -> np.ndarray:
        if log_scale is None:
            return values
        return np.sign(values) * log_scale * np.expm1(np.abs(values))

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (self._compress(features, self.log_scale) - self.offset) / self.denominator

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return self._expand(values * self.denominator + self.offset, self.log_scale)

    def save(self, path) -> None:
        arrays = {"offset": self.offset, "denominator": self.denominator}
        if self.log_scale is not None:
            arrays["log_scale"] = self.log_scale
        np.savez(path, scheme=np.array(self.scheme), **arrays)

    @classmethod
    def load(cls, path) -> "Normalization":
        archive = np.load(path, allow_pickle=False)
        return cls(
            str(archive["scheme"]),
            archive["log_scale"] if "log_scale" in archive.files else None,
            archive["offset"],
            archive["denominator"],
        )
