from __future__ import annotations

import pytest

from rl.feedback_collector import collect_feedback
from rl.policy_guard import GuardedPolicy, PolicyViolationError
from rl.reward_calculator import compute_reward
from rl.threshold_policy import ContextualBanditPolicy


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    """Provide required settings and isolated RL file paths for deterministic tests."""

    base_path = tmp_path  # tmp_path is already a per-test temporary directory.
    monkeypatch.setenv("KAFKA_BROKER", "test-broker")
    monkeypatch.setenv("KAFKA_TOPIC", "test-topic")
    monkeypatch.setenv("KAFKA_GROUP_ID", "test-group")
    monkeypatch.setenv("DB_URL", "postgresql://test-user:test-pass@localhost:5432/testdb")
    monkeypatch.setenv("SCRUBBER_HMAC_KEY", "test-hmac-key")
    monkeypatch.setenv("REDIS_URL", "rediss://test-redis")
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(base_path / "registry"))
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("RL_FEEDBACK_LOG_PATH", str(base_path / "feedback.jsonl"))
    monkeypatch.setenv("RL_Q_TABLE_PATH", str(base_path / "q_table.json"))


def test_invalid_verdict_raises_value_error() -> None:
    """Ensure unsupported verdict labels are rejected early to protect RL signal integrity."""

    with pytest.raises(ValueError, match="verdict must be one of TP, FP, FN"):
        collect_feedback(
            alert_id="alert-001",
            verdict="MAYBE",
            reason="manual-check",
            analyst_id="analyst-01",
        )


@pytest.mark.parametrize(
    ("verdict", "severity", "expected"),
    [
        ("TP", "low", 1.0),
        ("FP", "medium", -0.5),
        ("FN", "low", -2.0),
        ("FN", "critical", -4.0),
    ],
)
def test_reward_values_match_spec(verdict: str, severity: str, expected: float) -> None:
    """Verify all required verdict/severity combinations produce the exact reward contract."""

    assert compute_reward(verdict, severity) == expected


def test_policy_moves_toward_lower_threshold_after_tp_rewards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm repeated positive reward on a lower threshold increases its selection preference."""

    monkeypatch.setattr("rl.threshold_policy.random.random", lambda: 0.99)  # Force exploit mode.
    policy = ContextualBanditPolicy(epsilon=0.0, alpha=0.5)
    state = ("analyst", 9)
    for _ in range(20):
        policy.update(state, 0.3, 1.0)  # Repeatedly reinforce lower threshold action.
    selected = policy.select_threshold(state)
    assert selected == 0.3


def test_guard_raises_policy_violation_on_out_of_range_action() -> None:
    """Ensure unsafe policy outputs are blocked to enforce global threshold safety bounds."""

    class OutOfRangePolicy(ContextualBanditPolicy):
        """Test double that intentionally emits an unsafe threshold."""

        def __init__(self) -> None:
            pass

        def select_threshold(self, state: tuple[str, int]) -> float:
            return 1.2

    guarded = GuardedPolicy(policy=OutOfRangePolicy())
    with pytest.raises(PolicyViolationError):
        guarded.get_threshold(("admin", 12))


def test_guard_allows_valid_action_through() -> None:
    """Ensure guard is transparent when the wrapped policy returns a safe threshold."""

    class ValidPolicy(ContextualBanditPolicy):
        """Test double that emits a valid threshold in the allowed interval."""

        def __init__(self) -> None:
            pass

        def select_threshold(self, state: tuple[str, int]) -> float:
            return 0.6

    guarded = GuardedPolicy(policy=ValidPolicy())
    assert guarded.get_threshold(("engineer", 14)) == 0.6


