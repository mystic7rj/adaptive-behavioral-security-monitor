from __future__ import annotations

import logging

from torch import Tensor, nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class BehaviorAutoencoder(nn.Module):
    """Encode and reconstruct behavior vectors for reconstruction-error anomaly detection.

    Autoencoders learn compact normal-behavior representations so reconstruction
    error can flag atypical activity.
    """

    def __init__(self) -> None:
        """Initialize encoder/decoder layers for 64-dim behavior feature inputs."""

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return reconstruction and latent embedding for a behavior batch."""

        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def anomaly_score(self, x: Tensor) -> float:
        """Compute scalar reconstruction-error anomaly score for one input batch."""

        reconstructed, _ = self.forward(x)
        loss = F.mse_loss(reconstructed, x, reduction="mean")
        return float(loss.item())
