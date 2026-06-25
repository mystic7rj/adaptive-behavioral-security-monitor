from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient
from jose import jwt


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provide required runtime settings for API/auth tests with isolated storage paths."""

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
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    monkeypatch.setenv("JWT_ISSUER", "absm-issuer")
    monkeypatch.setenv("ALLOWED_ORIGINS", '["https://example.test"]')


def _client() -> TestClient:
    """Build a TestClient after env setup so settings-based app wiring loads correctly."""

    # Removing module cache forces settings to reload from current test env vars.
    sys.modules.pop("api.main", None)
    main_module = import_module("api.main")
    return TestClient(main_module.app)


def _token(sub: str, role: str, expires_delta: timedelta) -> str:
    """Create signed test JWT tokens with deterministic issuer and expiry claims."""

    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "role": role,
        "iss": "absm-issuer",
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, "test-secret", algorithm="HS256")


def test_valid_jwt_allows_access() -> None:
    """Ensure a valid analyst JWT can access analyst-protected alerts endpoints."""

    client = _client()
    response = client.get(
        "/alerts",
        headers={"Authorization": f"Bearer {_token('user-1', 'analyst', timedelta(minutes=5))}"},
    )
    assert response.status_code == 200


def test_expired_jwt_returns_401() -> None:
    """Ensure expired JWT tokens are rejected with strict unauthorized responses."""

    client = _client()
    response = client.get(
        "/alerts",
        headers={"Authorization": f"Bearer {_token('user-1', 'analyst', timedelta(minutes=-1))}"},
    )
    assert response.status_code == 401


def test_wrong_role_returns_403() -> None:
    """Ensure authenticated users with non-analyst role are denied alerts reads."""

    client = _client()
    response = client.get(
        "/alerts",
        headers={"Authorization": f"Bearer {_token('user-2', 'viewer', timedelta(minutes=5))}"},
    )
    assert response.status_code == 403


def test_rate_limiter_returns_429_on_101st_request() -> None:
    """Ensure per-sub limiter enforces 100 requests per minute with 429 on overflow."""

    client = _client()
    headers = {"Authorization": f"Bearer {_token('user-3', 'analyst', timedelta(minutes=5))}"}
    for _ in range(100):
        ok_response = client.get("/alerts", headers=headers)
        assert ok_response.status_code == 200
    overflow_response = client.get("/alerts", headers=headers)
    assert overflow_response.status_code == 429


def test_health_endpoint_returns_200_without_token() -> None:
    """Ensure unauthenticated health checks remain available for uptime monitors."""

    client = _client()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "timestamp" in body
