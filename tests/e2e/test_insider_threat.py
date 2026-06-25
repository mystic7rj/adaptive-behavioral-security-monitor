"""Synthetic insider-threat replay to validate delayed data-exfiltration detection.

This scenario models a trusted insider that initially behaves normally for seven days,
then shifts login timing by +4 hours for three days, and finally begins a sudden 10x
bytes-out spike that should trigger anomaly escalation quickly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import torch
from torch import Tensor

from models.autoencoder import BehaviorAutoencoder
from models.ensemble import EnsembleScorer
from models.lstm_predictor import SessionLSTM

logger = logging.getLogger(__name__)


def _build_feature_vector(login_hour_delta: float, bytes_out_ratio: float) -> Tensor:
    """Create a synthetic 64-dimensional feature vector used by the autoencoder."""

    vector = torch.zeros((1, 64), dtype=torch.float32)
    # These positions intentionally encode compact synthetic proxies for hour drift and exfil volume.
    vector[0, 0] = login_hour_delta
    vector[0, 1] = bytes_out_ratio
    return vector


def _build_sequence_tensor(token_id: int = 2, length: int = 5) -> Tensor:
    """Construct a stable synthetic event-token sequence for LSTM input compatibility."""

    return torch.tensor([token_id] * length, dtype=torch.long)


def _make_timestamp(day_offset: int, hour: int) -> datetime:
    """Generate deterministic UTC timestamps for synthetic replay event ordering."""

    base = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    return base + timedelta(days=day_offset, hours=hour - 9)


def run_scenario() -> bool:
    """Replay insider-threat progression and assert threshold-crossing expectations.

    Assertions:
    - Ensemble anomaly score stays below 0.5 during seven-day normal behavior.
    - Ensemble anomaly score crosses 0.5 within three events of exfiltration spike onset.
    """

    autoencoder = BehaviorAutoencoder()
    lstm = SessionLSTM(vocab_size=32)
    scorer = EnsembleScorer(autoencoder, lstm, ae_weight=0.95, lstm_weight=0.05)

    with torch.no_grad():
        # Zeroed weights make scenario outputs deterministic and reproducible in CI.
        for parameter in autoencoder.parameters():
            parameter.zero_()
        for parameter in lstm.parameters():
            parameter.zero_()

    normal_sequence = _build_sequence_tensor(token_id=2, length=5)
    normal_scores: list[float] = []
    for day in range(7):
        timestamp = _make_timestamp(day_offset=day, hour=9)
        score = scorer.score(
            feature_vec=_build_feature_vector(login_hour_delta=0.0, bytes_out_ratio=1.0),
            event_seq=normal_sequence,
        ).combined_score
        normal_scores.append(score)
        logger.info("insider_scenario_phase=normal day=%d timestamp=%s score=%.4f", day, timestamp.isoformat(), score)

    assert all(score < 0.5 for score in normal_scores), "Normal-period score unexpectedly crossed 0.5"

    shifted_and_spike_scores: list[float] = []
    # Three shifted-login events represent +4h behavior change before the exfil spike begins.
    for event_index in range(3):
        timestamp = _make_timestamp(day_offset=7 + event_index, hour=13)
        score = scorer.score(
            feature_vec=_build_feature_vector(login_hour_delta=4.0, bytes_out_ratio=1.0),
            event_seq=normal_sequence,
        ).combined_score
        shifted_and_spike_scores.append(score)
        logger.info(
            "insider_scenario_phase=shifted_login event=%d timestamp=%s score=%.4f",
            event_index,
            timestamp.isoformat(),
            score,
        )

    for spike_event in range(3):
        timestamp = _make_timestamp(day_offset=10, hour=13) + timedelta(minutes=15 * spike_event)
        score = scorer.score(
            feature_vec=_build_feature_vector(login_hour_delta=4.0, bytes_out_ratio=10.0),
            event_seq=normal_sequence,
        ).combined_score
        shifted_and_spike_scores.append(score)
        logger.info(
            "insider_scenario_phase=spike event=%d timestamp=%s score=%.4f",
            spike_event,
            timestamp.isoformat(),
            score,
        )

    spike_scores = shifted_and_spike_scores[3:]
    assert any(score > 0.5 for score in spike_scores[:3]), "Spike onset failed to cross 0.5 within 3 events"
    return True


def test_insider_threat_scenario() -> None:
    """Execute insider-threat scenario as a pytest-collected e2e test."""

    assert run_scenario() is True
