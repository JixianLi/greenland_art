"""Shared interface for latent-space models.

PCA, UMAP and the autoencoders all reduce a multi-field matrix to a latent
representation, but they are not interchangeable, and the split here is
deliberate rather than cosmetic.

PCA and an autoencoder both learn a mapping *and* its inverse, so reconstruction
error is defined for them and can be compared directly. UMAP does not: it
optimises a neighbour-preserving embedding with no faithful inverse. umap-learn
ships an ``inverse_transform``, but it is an approximation fitted after the fact
and its reconstruction error is not comparable to the other two.

So ``LatentModel`` covers everything that produces an embedding, and
``ReconstructiveModel`` adds decoding for the subset that genuinely supports it.
Asking UMAP for a reconstruction error is then a type error rather than a
misleading number.
"""

from abc import ABC, abstractmethod

import numpy as np


class LatentModel(ABC):
    """Anything that maps a (n_samples, n_features) matrix to a latent space."""

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short label used in figures and result tables."""

    @abstractmethod
    def fit(self, features: np.ndarray) -> "LatentModel":
        ...

    @abstractmethod
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Return the (n_samples, latent_dim) embedding."""

    def fit_encode(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).encode(features)


class ReconstructiveModel(LatentModel):
    """A LatentModel that can also map latents back to feature space."""

    @abstractmethod
    def decode(self, latents: np.ndarray) -> np.ndarray:
        ...

    def reconstruct(self, features: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(features))

    def reconstruction_error(self, features: np.ndarray) -> np.ndarray:
        """Per-sample mean squared error, in the units the model was fitted on."""
        residual = self.reconstruct(features) - features
        return np.mean(residual**2, axis=1)

    def mean_squared_error(self, features: np.ndarray) -> float:
        return float(np.mean(self.reconstruction_error(features)))

    def explained_variance_ratio(self, features: np.ndarray) -> float:
        """Fraction of total variance retained through the bottleneck.

        Defined for any reconstructive model, so an autoencoder and PCA can be
        compared on one axis. For PCA this equals the sum of its component
        ratios; for an autoencoder there is no per-component decomposition, only
        this total.
        """
        residual_variance = float(np.mean((self.reconstruct(features) - features) ** 2))
        total_variance = float(np.mean((features - features.mean(axis=0)) ** 2))
        return 1.0 - residual_variance / total_variance
