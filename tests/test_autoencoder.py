from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from models.autoencoder import BehaviorAutoencoder
from models.model_registry import load_model, save_model
from models.threshold_calibrator import calibrate


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAFKA_BROKER", "test-broker")
    monkeypatch.setenv("KAFKA_TOPIC", "test-topic")
    monkeypatch.setenv("KAFKA_GROUP_ID", "test-group")
    monkeypatch.setenv("DB_URL", "postgresql://test-user:test-pass@localhost:5432/testdb")
    monkeypatch.setenv("SCRUBBER_HMAC_KEY", "test-hmac-key")
    monkeypatch.setenv("REDIS_URL", "rediss://test-redis")
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_LEVEL", "INFO")


def test_autoencoder_output_shapes() -> None:
    model = BehaviorAutoencoder()
    batch = torch.randn(4, 64)
    reconstructed, latent = model(batch)
    assert reconstructed.shape == batch.shape
    assert latent.shape == (4, 8)


def test_anomaly_score_higher_for_shifted_login_hour() -> None:
    model = BehaviorAutoencoder()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    normal = torch.zeros(1, 64)
    normal[0, 0] = 1.0
    anomalous = torch.zeros(1, 64)
    anomalous[0, 0] = 13.0

    normal_score = model.anomaly_score(normal)
    anomalous_score = model.anomaly_score(anomalous)
    assert anomalous_score > normal_score


def test_calibrate_percentile() -> None:
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    threshold = calibrate(scores, fpr_target=0.5)
    assert threshold == 3.0


def test_save_load_round_trip(tmp_path: Path) -> None:
    model = BehaviorAutoencoder()
    metadata = {"val_loss": 0.25, "trained_at": datetime.now(timezone.utc).isoformat()}
    model_path = tmp_path / "ae.pt"

    save_model(model, model_path, metadata)
    restored, restored_metadata = load_model(model_path)

    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    assert restored_metadata["val_loss"] == metadata["val_loss"]
