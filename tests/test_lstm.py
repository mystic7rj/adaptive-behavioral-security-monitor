from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import torch

from features.pii_scrubber import ScrubbedEvent
from features.sequence_builder import build_session_sequences
from models.ensemble import EnsembleScorer
from models.lstm_predictor import SessionLSTM


def _make_event(ts: datetime, event_type: str) -> ScrubbedEvent:
    return ScrubbedEvent(
        user_id=uuid4(),
        timestamp=ts,
        event_type=event_type,
        login_hour=ts.hour,
        session_duration_s=None,
        bytes_out=None,
        ip_hmac=None,
        email_sha256=None,
    )


def test_sequence_builder_shape_and_split() -> None:
    base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    events = [
        _make_event(base, "auth"),
        _make_event(base + timedelta(seconds=60), "auth"),
        _make_event(base + timedelta(seconds=1900), "network"),
    ]
    sequences = build_session_sequences(events, max_len=5)
    assert sequences.shape == (2, 5)
    assert int((sequences[0] != 0).sum().item()) == 2
    assert sequences[1, 0].item() != 0
    assert sequences[1, 1:].sum().item() == 0


def test_sequence_log_likelihood_lower_for_rare_event() -> None:
    model = SessionLSTM(vocab_size=32)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.projection.bias[2] = 2.0
        model.projection.bias[10] = -2.0

    normal = torch.tensor([2, 2, 2, 2], dtype=torch.long)
    anomalous = torch.tensor([2, 2, 10, 2], dtype=torch.long)

    normal_ll = model.sequence_log_likelihood(normal)
    anomalous_ll = model.sequence_log_likelihood(anomalous)
    assert anomalous_ll < normal_ll


def test_ensemble_weighted_average() -> None:
    class FakeAE:
        def anomaly_score(self, _: torch.Tensor) -> float:
            return 2.0

    class FakeLSTM:
        def sequence_log_likelihood(self, _: torch.Tensor) -> float:
            return 6.0

    scorer = EnsembleScorer(FakeAE(), FakeLSTM(), ae_weight=0.25, lstm_weight=0.75)
    score = scorer.score(torch.zeros(1, 64), torch.tensor([1, 2, 3]))
    assert score.ae_score == 2.0
    assert score.lstm_score == 6.0
    assert score.combined_score == 5.0
