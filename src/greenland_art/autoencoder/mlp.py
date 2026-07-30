"""Dense (MLP) autoencoder over per-cell multi-field vectors.

Each sample is one grid cell on one day: a vector of the MAR surface fields at
that place and time. Spatial structure is deliberately not used here -- that is
what a convolutional encoder is for, and it will implement the same interface
from base.py so the comparison stays like-for-like.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import ReconstructiveModel


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick a device. 'auto' prefers CUDA, then Apple MPS, then CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _SymmetricAutoencoder(nn.Module):
    """Encoder and decoder mirror each other around the bottleneck."""

    def __init__(self, n_features: int, latent_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()

        encoder_layers: list[nn.Module] = []
        width = n_features
        for hidden in hidden_sizes:
            encoder_layers += [nn.Linear(width, hidden), nn.BatchNorm1d(hidden), nn.GELU()]
            width = hidden
        encoder_layers.append(nn.Linear(width, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        width = latent_dim
        for hidden in reversed(hidden_sizes):
            decoder_layers += [nn.Linear(width, hidden), nn.BatchNorm1d(hidden), nn.GELU()]
            width = hidden
        # No activation on the output: inputs are standardised, so the target
        # range is unbounded and any squashing would clip the tails.
        decoder_layers.append(nn.Linear(width, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(batch))


class MLPAutoencoder(ReconstructiveModel):
    """Dense autoencoder trained with mean squared error on standardised inputs.

    The caller is expected to have standardised `features` already (see
    StandardScalerState); the model does not rescale internally, so that PCA,
    UMAP and this model all see identical inputs.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 128, 64),
        max_epochs: int = 60,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        validation_fraction: float = 0.1,
        patience: int = 8,
        device: str = "auto",
        seed: int = 0,
        verbose: bool = True,
    ):
        self._latent_dim = latent_dim
        self.hidden_sizes = hidden_sizes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.device = resolve_device(device)
        self.seed = seed
        self.verbose = verbose

        self.module: _SymmetricAutoencoder | None = None
        self.history: list[dict[str, float]] = []

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def name(self) -> str:
        return f"MLP-AE({self._latent_dim})"

    def fit(self, features: np.ndarray) -> "MLPAutoencoder":
        torch.manual_seed(self.seed)
        generator = np.random.default_rng(self.seed)

        n_samples, n_features = features.shape
        # Validation split is random over samples on purpose: unlike the SMB
        # emulator, the question here is representation quality, not temporal
        # generalisation, and every sample is an independent cell-day vector.
        shuffled = generator.permutation(n_samples)
        n_validation = int(n_samples * self.validation_fraction)
        validation_index, train_index = shuffled[:n_validation], shuffled[n_validation:]

        train_tensor = torch.from_numpy(np.ascontiguousarray(features[train_index], dtype=np.float32))
        validation_tensor = torch.from_numpy(
            np.ascontiguousarray(features[validation_index], dtype=np.float32)
        ).to(self.device)

        self.module = _SymmetricAutoencoder(n_features, self._latent_dim, self.hidden_sizes).to(
            self.device
        )
        optimiser = torch.optim.AdamW(
            self.module.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.max_epochs)
        loss_function = nn.MSELoss()

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        best_validation = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            self.module.train()
            running, batches = 0.0, 0
            for (batch,) in loader:
                batch = batch.to(self.device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                loss = loss_function(self.module(batch), batch)
                loss.backward()
                optimiser.step()
                running += float(loss.detach())
                batches += 1
            schedule.step()

            self.module.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_function(self.module(validation_tensor), validation_tensor)
                )
            self.history.append(
                {"epoch": epoch, "train_mse": running / max(batches, 1), "validation_mse": validation_loss}
            )
            if self.verbose:
                print(
                    f"  [{self.name}] epoch {epoch:3d}  train {running / max(batches, 1):.5f}"
                    f"  val {validation_loss:.5f}",
                    flush=True,
                )

            if validation_loss < best_validation - 1e-6:
                best_validation = validation_loss
                best_state = {k: v.detach().clone() for k, v in self.module.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"  [{self.name}] early stop at epoch {epoch}", flush=True)
                    break

        if best_state is not None:
            self.module.load_state_dict(best_state)
        return self

    def _apply(self, network: nn.Module, values: np.ndarray) -> np.ndarray:
        if self.module is None:
            raise RuntimeError("fit() must be called before encode()/decode()")
        self.module.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                chunk = torch.from_numpy(
                    np.ascontiguousarray(values[start : start + self.batch_size], dtype=np.float32)
                ).to(self.device)
                outputs.append(network(chunk).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def encode(self, features: np.ndarray) -> np.ndarray:
        return self._apply(self.module.encoder, features)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        return self._apply(self.module.decoder, latents)

    def save(self, path: Path) -> None:
        if self.module is None:
            raise RuntimeError("nothing to save; fit() first")
        torch.save(
            {
                "state_dict": self.module.state_dict(),
                "latent_dim": self._latent_dim,
                "hidden_sizes": self.hidden_sizes,
                "history": self.history,
            },
            path,
        )
