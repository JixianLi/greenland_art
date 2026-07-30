"""Latent-space models over multi-field Greenland data.

base.py defines the interface every model implements, so a convolutional
encoder can be added later without changing any caller.
"""

from .base import LatentModel, ReconstructiveModel, StandardScalerState
from .baselines import PCAModel, UMAPProjection
from .mlp import MLPAutoencoder

__all__ = [
    "LatentModel",
    "ReconstructiveModel",
    "StandardScalerState",
    "PCAModel",
    "UMAPProjection",
    "MLPAutoencoder",
]
