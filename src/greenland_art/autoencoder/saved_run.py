"""Reopen a finished run and reconstruct from it on demand.

A job writes six fitted models totalling about 12 MB. Saving what those models
*output* over the full matrix would be 2.1 GB each, 14.6 GB for the set, so the
outputs are not saved -- they are recomputed here from the checkpoints and the
training matrix, which already sits on the analysis machine.

That choice is what makes an interactive view possible at all: reconstructing a
single timestep is 4,604 rows through a small network, a few milliseconds, so a
time slider can drive inference live rather than reading precomputed frames.
"""

import json
from pathlib import Path

import numpy as np

from .baselines import PCAModel
from .mlp import MLPAutoencoder
from .normalization import Normalization


class SavedRun:
    """Models, normalisation and source data for one completed job."""

    def __init__(self, run_dir: Path, training_data: Path, device: str = "auto"):
        self.run_dir = Path(run_dir)
        with open(self.run_dir / "results.json") as handle:
            self.results = json.load(handle)

        self.field_names = [str(name) for name in self.results["field_names"]]
        self.latent_dims = list(self.results["latent_dims"])
        self.normalization = Normalization.load(self.run_dir / "normalization.npz")

        archive = np.load(training_data, allow_pickle=False)
        self.features = archive["features"]
        self.cell_index = archive["cell_index"]
        self.day_of_year = archive["day_of_year"]
        self.year = archive["year"]
        self.grid_shape = tuple(int(v) for v in archive["grid_shape"])
        self.surface_height = archive["surface_height"]
        self.ice_mask = archive["ice_mask"]

        if len(self.field_names) != self.features.shape[1]:
            raise ValueError(
                f"run has {len(self.field_names)} columns but {training_data.name} has "
                f"{self.features.shape[1]}; these are not the same matrix"
            )

        self._device = device
        self._models: dict[str, object] = {}

        # Row order in the matrix is (day, cell), so every timestep is one
        # contiguous block. Recording the slice bounds once turns timestep
        # lookup into a slice rather than a scan of three million rows.
        self.timesteps = []
        self._bounds = {}
        keys = (self.year.astype(np.int64) << 16) | self.day_of_year.astype(np.int64)
        edges = np.flatnonzero(np.diff(keys)) + 1
        for start, stop in zip(
            np.concatenate([[0], edges]), np.concatenate([edges, [len(keys)]])
        ):
            step = (int(self.year[start]), int(self.day_of_year[start]))
            self.timesteps.append(step)
            self._bounds[step] = (int(start), int(stop))

    @property
    def validation_mask(self) -> np.ndarray | None:
        """Per-row held-out flag, or None if this run cannot be matched to it.

        The split is random over samples, so roughly 90 % of any timestep was
        seen in training. Error quoted over every cell would flatter every model,
        and the difference is not small. Returns None when the job subsampled
        with --max-samples, because the saved indices then refer to that
        subsample and cannot be mapped back to full matrix rows.
        """
        if self.results["n_samples"] != len(self.features):
            return None
        if not hasattr(self, "_validation_mask"):
            archive = np.load(self.run_dir / "projections.npz", allow_pickle=False)
            mask = np.zeros(len(self.features), dtype=bool)
            mask[archive["validation_index"]] = True
            self._validation_mask = mask
        return self._validation_mask

    @property
    def model_names(self) -> list[str]:
        return [
            f"{family}{latent_dim}"
            for latent_dim in self.latent_dims
            for family in ("pca", "autoencoder")
        ]

    def model(self, name: str):
        """Fitted model by name, e.g. 'pca15' or 'autoencoder30'. Cached."""
        if name not in self._models:
            if name.startswith("pca"):
                self._models[name] = PCAModel.load(
                    self.run_dir / f"pca_latent{name.removeprefix('pca')}.npz"
                )
            elif name.startswith("autoencoder"):
                self._models[name] = MLPAutoencoder.load(
                    self.run_dir
                    / f"mlp_autoencoder_latent{name.removeprefix('autoencoder')}.pt",
                    device=self._device,
                )
            else:
                raise KeyError(f"unknown model {name!r}; expected one of {self.model_names}")
        return self._models[name]

    def timestep_rows(self, year: int, day: int) -> slice:
        start, stop = self._bounds[(year, day)]
        return slice(start, stop)

    def truth(self, year: int, day: int) -> np.ndarray:
        return self.features[self.timestep_rows(year, day)]

    def reconstruct(self, model_name: str, year: int, day: int) -> np.ndarray:
        """Reconstruction for one timestep, in physical units."""
        rows = self.timestep_rows(year, day)
        transformed = self.normalization.transform(self.features[rows])
        return self.normalization.inverse_transform(self.model(model_name).reconstruct(transformed))

    def absolute_error(self, model_name: str, rows: np.ndarray, chunk: int = 262_144):
        """|reconstruction - truth| in physical units, for the given row indices.

        Chunked over rows rather than looped over timesteps: a timestep is 4,604
        rows, small enough that per-call overhead dominates, and there are 733 of
        them. Larger chunks make this roughly an order of magnitude faster.
        """
        model = self.model(model_name)
        error = np.empty((len(rows), self.features.shape[1]), dtype=np.float32)
        for start in range(0, len(rows), chunk):
            selected = rows[start : start + chunk]
            truth = self.features[selected]
            reconstruction = self.normalization.inverse_transform(
                model.reconstruct(self.normalization.transform(truth))
            )
            error[start : start + chunk] = np.abs(reconstruction - truth)
        return error

    def to_grid(self, values: np.ndarray, year: int, day: int) -> np.ndarray:
        """Scatter per-cell values onto the model grid, off-ice left blank."""
        field = np.full(int(np.prod(self.grid_shape)), np.nan)
        field[self.cell_index[self.timestep_rows(year, day)]] = values
        return field.reshape(self.grid_shape)
