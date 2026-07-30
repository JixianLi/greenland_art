"""PCA and UMAP against the same interface as the autoencoders.

PCA is the benchmark that matters. It is linear, deterministic and takes
seconds; an autoencoder that does not beat it at equal latent width has bought
nothing with its nonlinearity, and saying so is a result rather than a failure.

UMAP answers a different question and is deliberately typed differently. It has
no faithful inverse, so it implements LatentModel only -- see base.py.
"""

import numpy as np
from sklearn.decomposition import PCA

from .base import LatentModel, ReconstructiveModel


class PCAModel(ReconstructiveModel):
    """Linear baseline. Also exposes the per-component variance breakdown."""

    def __init__(self, latent_dim: int, seed: int = 0):
        self._latent_dim = latent_dim
        self._pca = PCA(n_components=latent_dim, random_state=seed)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def name(self) -> str:
        return f"PCA({self._latent_dim})"

    def fit(self, features: np.ndarray) -> "PCAModel":
        self._pca.fit(features)
        return self

    def encode(self, features: np.ndarray) -> np.ndarray:
        return self._pca.transform(features)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        return self._pca.inverse_transform(latents)

    @property
    def explained_variance_ratio_per_component(self) -> np.ndarray:
        """Variance fraction carried by each principal component, in order."""
        return self._pca.explained_variance_ratio_

    @property
    def cumulative_explained_variance_ratio(self) -> np.ndarray:
        return np.cumsum(self._pca.explained_variance_ratio_)


class UMAPProjection(LatentModel):
    """Neighbour-preserving embedding, used for visualisation only.

    No decode method by design. UMAP optimises local neighbourhood structure,
    not reconstruction, so distances in the embedding are not metric and cluster
    sizes and inter-cluster gaps carry little meaning. It is included to show
    the shape of the latent space, never to quantify information retained.
    """

    def __init__(
        self,
        latent_dim: int = 2,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        seed: int = 0,
        subsample: int | None = 200_000,
    ):
        self._latent_dim = latent_dim
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.seed = seed
        # UMAP is O(n log n) but with a large constant; beyond a few hundred
        # thousand rows it dominates the whole comparison for a plot that looks
        # the same either way.
        self.subsample = subsample
        self._reducer = None

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def name(self) -> str:
        return f"UMAP({self._latent_dim})"

    def fit(self, features: np.ndarray) -> "UMAPProjection":
        import umap

        rows = features
        if self.subsample is not None and len(features) > self.subsample:
            generator = np.random.default_rng(self.seed)
            rows = features[generator.choice(len(features), self.subsample, replace=False)]

        self._reducer = umap.UMAP(
            n_components=self._latent_dim,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.seed,
        ).fit(rows)
        return self

    def encode(self, features: np.ndarray) -> np.ndarray:
        if self._reducer is None:
            raise RuntimeError("fit() must be called before encode()")
        return self._reducer.transform(features)
