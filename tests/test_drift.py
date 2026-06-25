from __future__ import annotations

from datetime import datetime, timezone

import pytest

from drift.adwin_detector import ADWINDetector
from drift.retraining_scheduler import check_and_schedule_retrain
from features.profile_builder import UserProfile
from models.online_updater import update_baseline


class FakeStore:
    def __init__(self, profile: UserProfile) -> None:
        self.profile = profile

    def get_profile(self, user_id: str) -> UserProfile | None:
        return self.profile if self.profile.user_id == user_id else None

    def set_profile(self, user_id: str, profile: UserProfile, ttl: int = 86400) -> None:
        self.profile = profile


def test_ema_update_moves_toward_new_value() -> None:
    profile = UserProfile(
        user_id="user-1",
        window_days=7,
        features={"metric_mean": 10.0, "metric_std": 2.0},
        generated_at=datetime.now(timezone.utc),
    )
    store = FakeStore(profile)

    update_baseline("user-1", {"metric_mean": 20.0}, store, alpha=0.1)
    updated = store.profile.features["metric_mean"]
    assert updated == pytest.approx(11.0)


def test_adversarial_update_is_capped() -> None:
    profile = UserProfile(
        user_id="user-2",
        window_days=7,
        features={"metric_mean": 0.0, "metric_std": 2.0},
        generated_at=datetime.now(timezone.utc),
    )
    store = FakeStore(profile)

    update_baseline("user-2", {"metric_mean": 20.0}, store, alpha=0.2)
    updated = store.profile.features["metric_mean"]
    assert updated == pytest.approx(1.0)


def test_adwin_detects_shift() -> None:
    detector = ADWINDetector()
    detected = False
    for _ in range(50):
        detector.add_element(0.0)
    for _ in range(50):
        if detector.add_element(10.0):
            detected = True
    assert detected


def test_ks_test_detects_distribution_change() -> None:
    detector = ADWINDetector()
    current = [10.0] * 50
    reference = [0.0] * 50
    assert check_and_schedule_retrain(current, reference, detector) is True
