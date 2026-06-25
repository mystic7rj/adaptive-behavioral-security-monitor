from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from alerts.alert_builder import build_alert
from alerts import dispatcher
from enrich.peer_comparator import PeerContext, PeerFeatureRank
from explain.shap_explainer import Explanation, FeatureContribution


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provide required settings plus isolated paths for alert and peer tests."""

    monkeypatch.setenv("KAFKA_BROKER", "test-broker")
    monkeypatch.setenv("KAFKA_TOPIC", "test-topic")
    monkeypatch.setenv("KAFKA_GROUP_ID", "test-group")
    monkeypatch.setenv("DB_URL", "postgresql://test-user:test-pass@localhost:5432/testdb")
    monkeypatch.setenv("SCRUBBER_HMAC_KEY", "test-hmac-key")
    monkeypatch.setenv("REDIS_URL", "rediss://test-redis")
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path / "registry"))
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("RL_FEEDBACK_LOG_PATH", str(tmp_path / "rl_feedback.jsonl"))
    monkeypatch.setenv("RL_Q_TABLE_PATH", str(tmp_path / "rl_q_table.json"))
    monkeypatch.setenv("PEER_STATS_PATH", str(tmp_path / "peer_stats.json"))
    monkeypatch.setenv("ALERT_DEAD_LETTER_PATH", str(tmp_path / "dead_letter.jsonl"))


def _explanation() -> Explanation:
    """Create a deterministic top-3 explanation payload for alert-building tests."""

    return Explanation(
        top_features=[
            FeatureContribution(
                feature_name="login_hour",
                contribution=3.2,
                description="Login hour deviated increased 3.2sigma from baseline",
            ),
            FeatureContribution(
                feature_name="bytes_out",
                contribution=-2.1,
                description="Bytes out deviated decreased 2.1sigma from baseline",
            ),
            FeatureContribution(
                feature_name="session_duration_s",
                contribution=1.4,
                description="Session duration s deviated increased 1.4sigma from baseline",
            ),
        ]
    )


def _peer_context() -> PeerContext:
    """Create deterministic peer context used by alert construction tests."""

    return PeerContext(
        user_id="user-abc",
        role="engineer",
        feature_percentiles=[
            PeerFeatureRank(feature_name="login_hour", percentile_rank=91.0),
            PeerFeatureRank(feature_name="bytes_out", percentile_rank=88.0),
        ],
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.81, "critical"),
        (0.61, "high"),
        (0.41, "medium"),
        (0.2, "low"),
    ],
)
def test_severity_mapping_boundaries(score: float, expected: str) -> None:
    """Verify score thresholds map to the requested severities at boundary-like values."""

    alert = build_alert(score, _explanation(), _peer_context(), user_id="user-abc")
    assert alert.severity == expected


def test_alert_id_is_valid_uuid() -> None:
    """Ensure each built alert has a UUID identifier rather than a sequential number."""

    alert = build_alert(0.9, _explanation(), _peer_context(), user_id="user-abc")
    assert isinstance(alert.alert_id, UUID)


def test_dead_letter_file_written_on_dispatch_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure exhausted dispatch retries are persisted to dead-letter storage."""

    dead_letter_path = tmp_path / "dead_letter.jsonl"
    monkeypatch.setenv("ALERT_DEAD_LETTER_PATH", str(dead_letter_path))

    def _always_fail(alert: object) -> None:
        raise RuntimeError("simulated transport failure")

    monkeypatch.setattr(dispatcher, "_deliver", _always_fail)
    alert = build_alert(0.9, _explanation(), _peer_context(), user_id="user-abc")
    dispatcher.dispatch(alert)

    lines = dead_letter_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["alert_id"] == str(alert.alert_id)
    assert payload["severity"] == "critical"


def test_explanation_contains_exactly_three_features() -> None:
    """Ensure the alert explanation contract always carries exactly three top features."""

    alert = build_alert(0.7, _explanation(), _peer_context(), user_id="user-abc")
    assert len(alert.explanation.top_features) == 3
