from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from models.autoencoder import BehaviorAutoencoder
from models.model_registry import save_model

logger = logging.getLogger(__name__)


def _epoch_loss(model: BehaviorAutoencoder, batches: Iterable[Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one training epoch and return mean reconstruction loss.

    Encapsulating one epoch keeps train loop readable and testable.
    """

    model.train()
    total_loss = 0.0
    count = 0
    for batch in batches:
        optimizer.zero_grad()
        reconstructed, _ = model(batch)
        loss = torch.nn.functional.mse_loss(reconstructed, batch, reduction="mean")
        torch.autograd.backward(loss)
        optimizer.step()
        total_loss += float(loss.item())
        count += 1
    return total_loss / max(count, 1)


def _eval_loss(model: BehaviorAutoencoder, batches: Iterable[Tensor]) -> float:
    """Evaluate model reconstruction loss without gradient updates."""

    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches:
            reconstructed, _ = model(batch)
            loss = torch.nn.functional.mse_loss(reconstructed, batch, reduction="mean")
            total_loss += float(loss.item())
            count += 1
    return total_loss / max(count, 1)


def train_autoencoder(
    model: BehaviorAutoencoder,
    train_batches: Iterable[Tensor],
    val_batches: Iterable[Tensor],
    epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 10,
    save_path: str | None = None,
) -> float:
    """Train autoencoder with early stopping and optional checkpoint persistence.

    This trainer exists to standardize optimization, stopping criteria, and
    metadata capture across model retraining runs.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val = float("inf")
    patience_left = patience

    for epoch in range(epochs):
        train_loss = _epoch_loss(model, train_batches, optimizer)
        val_loss = _eval_loss(model, val_batches)
        logger.info("epoch_completed epoch=%d train_loss=%.6f val_loss=%.6f", epoch + 1, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("early_stopping epoch=%d", epoch + 1)
                break

    if save_path is not None:
        metadata = {
            "val_loss": best_val,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        # Save final trained state and metadata for reproducibility and later rollout.
        save_model(model, Path(save_path), metadata)

    logger.info("training_completed best_val=%.6f", best_val)
    return best_val
