from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from config.settings import load_settings

from .autoencoder import BehaviorAutoencoder

logger = logging.getLogger(__name__)


def _resolve_base_dir() -> Path:
    """Resolve and normalize model-registry base directory from settings."""

    settings = load_settings()
    return Path(settings.MODEL_REGISTRY_DIR).expanduser().resolve()


def _validate_model_path(path: Path) -> Path:
    """Validate and constrain model paths to the configured registry root.

    Path validation prevents accidental or malicious writes outside managed storage.
    """

    if ".." in path.parts:
        raise ValueError("path traversal is not allowed")

    base_dir = _resolve_base_dir()
    resolved = (base_dir / path).expanduser().resolve() if not path.is_absolute() else path.expanduser().resolve()
    if not resolved.is_relative_to(base_dir):
        raise ValueError("model path must be under registry base dir")
    return resolved


def _metadata_path(model_path: Path) -> Path:
    """Return sidecar metadata path stored alongside model checkpoints."""

    return model_path.with_suffix(model_path.suffix + ".json")


def save_model(model: BehaviorAutoencoder, path: Path, metadata: dict[str, Any]) -> None:
    """Persist model weights and structured metadata to the registry filesystem.

    Saving both state and metadata supports reproducibility and audit trails.
    """

    model_path = _validate_model_path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    payload = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        **metadata,
    }
    metadata_path = _metadata_path(model_path)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("model_saved")


def load_model(path: Path) -> tuple[BehaviorAutoencoder, dict[str, Any]]:
    """Load model checkpoint and associated metadata from registry storage."""

    model_path = _validate_model_path(path)
    state = torch.load(model_path, map_location="cpu", weights_only=True)  # nosec B614
    model = BehaviorAutoencoder()
    model.load_state_dict(state)

    metadata_path = _metadata_path(model_path)
    if not metadata_path.exists():
        raise FileNotFoundError("metadata file not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logger.info("model_loaded")
    return model, metadata
