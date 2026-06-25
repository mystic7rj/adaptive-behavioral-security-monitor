from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from features.feature_store import get_profile, set_profile
from features.pii_scrubber import ScrubbedEvent, scrub
from features.profile_builder import UserProfile
from features.rolling_aggregates import compute_aggregates
from ingest.schema import AuthEvent


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_BROKER", "test-broker")
    monkeypatch.setenv("KAFKA_TOPIC", "test-topic")
    monkeypatch.setenv("KAFKA_GROUP_ID", "test-group")
    monkeypatch.setenv("DB_URL", "postgresql://test-user:test-pass@localhost:5432/testdb")
    monkeypatch.setenv("SCRUBBER_HMAC_KEY", "test-hmac-key")
    monkeypatch.setenv("REDIS_URL", "rediss://test-redis")
    monkeypatch.setenv("MODEL_REGISTRY_DIR", "models")
    monkeypatch.setenv("LOG_LEVEL", "INFO")


def test_scrubber_deterministic_and_scrubs_pii() -> None:
    event = AuthEvent.model_validate(
        {
            "user_id": uuid4(),
            "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            "event_type": "auth",
            "ip": "203.0.113.42",
            "email": "user@example.com",
            "session_duration_s": 120.0,
            "bytes_out": 2048,
        }
    )

    first = scrub(event)
    second = scrub(event)

    assert first.ip_hmac == second.ip_hmac
    assert first.email_sha256 == second.email_sha256
    assert first.ip_hmac is not None
    assert first.email_sha256 is not None
    assert first.ip_hmac != event.ip
    assert first.email_sha256 != event.email
    assert event.ip not in first.ip_hmac
    assert event.email not in first.email_sha256
    assert len(first.ip_hmac) == 64
    assert len(first.email_sha256) == 64


def test_rolling_aggregates_on_three_events() -> None:
    base_time = datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc)
    events = [
        ScrubbedEvent(
            user_id=uuid4(),
            timestamp=base_time,
            event_type="auth",
            login_hour=12,
            session_duration_s=30.0,
            bytes_out=100,
            ip_hmac="ip-1",
            email_sha256=None,
        ),
        ScrubbedEvent(
            user_id=uuid4(),
            timestamp=base_time - timedelta(minutes=30),
            event_type="auth",
            login_hour=11,
            session_duration_s=60.0,
            bytes_out=200,
            ip_hmac="ip-2",
            email_sha256=None,
        ),
        ScrubbedEvent(
            user_id=uuid4(),
            timestamp=base_time - timedelta(hours=24),
            event_type="auth",
            login_hour=12,
            session_duration_s=90.0,
            bytes_out=300,
            ip_hmac="ip-1",
            email_sha256=None,
        ),
    ]

    features = compute_aggregates(events)

    assert features["login_hour_1h_mean"] == pytest.approx(11.5)
    assert features["login_hour_1h_std"] == pytest.approx(0.5)
    assert features["login_hour_1h_count"] == 2
    assert features["session_duration_s_1h_mean"] == pytest.approx(45.0)
    assert features["session_duration_s_1h_std"] == pytest.approx(15.0)
    assert features["session_duration_s_1h_count"] == 2
    assert features["bytes_out_1h_mean"] == pytest.approx(150.0)
    assert features["bytes_out_1h_std"] == pytest.approx(50.0)
    assert features["bytes_out_1h_count"] == 2
    assert features["unique_ip_count_1h_mean"] == pytest.approx(2.0)
    assert features["unique_ip_count_1h_std"] == pytest.approx(0.0)
    assert features["unique_ip_count_1h_count"] == 2

    assert features["login_hour_24h_mean"] == pytest.approx(11.6666667)
    assert features["login_hour_24h_std"] == pytest.approx(0.4714045)
    assert features["login_hour_24h_count"] == 3
    assert features["session_duration_s_24h_mean"] == pytest.approx(60.0)
    assert features["session_duration_s_24h_std"] == pytest.approx(24.494897)
    assert features["session_duration_s_24h_count"] == 3
    assert features["bytes_out_24h_mean"] == pytest.approx(200.0)
    assert features["bytes_out_24h_std"] == pytest.approx(81.649658)
    assert features["bytes_out_24h_count"] == 3
    assert features["unique_ip_count_24h_mean"] == pytest.approx(2.0)
    assert features["unique_ip_count_24h_std"] == pytest.approx(0.0)
    assert features["unique_ip_count_24h_count"] == 2


def test_feature_store_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self._store: dict[str, bytes] = {}

        def setex(self, key: str, ttl: int, value: bytes) -> bool:
            self._store[key] = value
            return True

        def get(self, key: str) -> bytes | None:
            return self._store.get(key)

    fake_client = FakeRedis()

    def _fake_get_client() -> FakeRedis:
        return fake_client

    monkeypatch.setattr("features.feature_store._get_client", _fake_get_client)

    profile = UserProfile(
        user_id="user-123",
        window_days=7,
        features={"login_hour_1h_mean": 11.0, "unique_ip_count_1h_count": 2},
        generated_at=datetime(2024, 1, 9, 10, 0, tzinfo=timezone.utc),
    )

    set_profile(profile.user_id, profile, ttl=3600)
    restored = get_profile(profile.user_id)

    assert restored == profile
