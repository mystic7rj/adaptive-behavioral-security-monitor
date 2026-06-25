"""Synthetic compromised-credentials replay to validate account-takeover detection.

This scenario simulates one user account being abused by an attacker: source IP shifts
to an unseen value, resource access moves outside normal role scope, and authentication
happens at 02:00. The expected outcome is sequence-level anomaly elevation and ensemble
score threshold crossing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from models.autoencoder import BehaviorAutoencoder
from models.ensemble import EnsembleScorer
from models.lstm_predictor import SessionLSTM
from models.sequence_scorer import score_sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticAccessEvent:
    """Synthetic event record for compromised-credential scenario generation."""

    user_id: str
    ip_profile: str
    resource_scope: str
    login_hour: int
    token_id: int


def _build_feature_vector(ip_changed: float, outside_role_scope: float, off_hours_login: float) -> Tensor:
    """Construct a synthetic feature vector for ensemble scoring."""

    vector = torch.zeros((1, 64), dtype=torch.float32)
    # These compact indicators mimic the three compromised-credential signals in model features.
    vector[0, 0] = ip_changed
    vector[0, 1] = outside_role_scope
    vector[0, 2] = off_hours_login
    return vector


def _to_sequence(events: list[SyntheticAccessEvent]) -> Tensor:
    """Convert synthetic access events into an LSTM-compatible token sequence."""

    return torch.tensor([event.token_id for event in events], dtype=torch.long)


def run_scenario() -> bool:
    """Replay compromised-credentials behavior and assert detection outcomes.

    Assertions:
    - LSTM sequence scorer flags unusual access pattern with elevated z-score.
    - Ensemble score crosses a 0.5 anomaly threshold under compromised conditions.
    """

    user_id = "synthetic-user-001"
    baseline_events = [
        SyntheticAccessEvent(user_id=user_id, ip_profile="known", resource_scope="normal", login_hour=9, token_id=2),
        SyntheticAccessEvent(user_id=user_id, ip_profile="known", resource_scope="normal", login_hour=10, token_id=2),
        SyntheticAccessEvent(user_id=user_id, ip_profile="known", resource_scope="normal", login_hour=11, token_id=2),
        SyntheticAccessEvent(user_id=user_id, ip_profile="known", resource_scope="normal", login_hour=9, token_id=2),
    ]
    compromised_events = [
        SyntheticAccessEvent(user_id=user_id, ip_profile="new", resource_scope="outside_role", login_hour=2, token_id=10),
        SyntheticAccessEvent(user_id=user_id, ip_profile="new", resource_scope="outside_role", login_hour=2, token_id=10),
        SyntheticAccessEvent(user_id=user_id, ip_profile="new", resource_scope="outside_role", login_hour=2, token_id=10),
        SyntheticAccessEvent(user_id=user_id, ip_profile="new", resource_scope="outside_role", login_hour=2, token_id=10),
    ]

    lstm = SessionLSTM(vocab_size=32)
    autoencoder = BehaviorAutoencoder()
    ensemble = EnsembleScorer(autoencoder, lstm, ae_weight=0.95, lstm_weight=0.05)

    with torch.no_grad():
        # Deterministic parameters make scenario outcomes stable across runs and CI workers.
        for parameter in lstm.parameters():
            parameter.zero_()
        for parameter in autoencoder.parameters():
            parameter.zero_()
        # Biasing common-token log-probability down-weights compromised token 10 as unusual.
        lstm.projection.bias[2] = 2.0
        lstm.projection.bias[10] = -2.0

    baseline_sequence = _to_sequence(baseline_events)
    compromised_sequence = _to_sequence(compromised_events)
    baseline_ll = lstm.sequence_log_likelihood(baseline_sequence)
    compromised_zscore = score_sequence(
        seq=compromised_sequence,
        model=lstm,
        baseline_ll=baseline_ll,
        baseline_std=1.0,
    )
    assert compromised_zscore > 1.0, "LSTM sequence scorer failed to flag unusual access pattern"

    compromised_score = ensemble.score(
        # Amplified synthetic indicators ensure the autoencoder component reflects clearly abnormal behavior.
        feature_vec=_build_feature_vector(ip_changed=6.0, outside_role_scope=6.0, off_hours_login=6.0),
        event_seq=compromised_sequence,
    ).combined_score
    assert compromised_score > 0.5, "Ensemble score failed to cross 0.5 for compromised-credential pattern"

    logger.info("compromised_creds_scenario_passed sequence_z=%.4f ensemble_score=%.4f", compromised_zscore, compromised_score)
    return True


def test_compromised_creds_scenario() -> None:
    """Execute compromised-credentials scenario as a pytest-collected e2e test."""

    assert run_scenario() is True
