from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from features.profile_builder import UserProfile

logger = logging.getLogger(__name__)


class FeatureStore(Protocol):
    """Define storage contract required by online baseline updater logic."""

    def get_profile(self, user_id: str) -> UserProfile | None: ...

    def set_profile(self, user_id: str, profile: UserProfile, ttl: int = 86400) -> None: ...


@dataclass(frozen=True)
class UpdateResult:
    """Describe baseline update summary metrics for observability and testing."""

    updated_features: int
    capped_features: int


def _std_key(feature_key: str) -> str:
    """Derive companion standard-deviation feature key from a mean feature key."""

    if feature_key.endswith("_mean"):
        return f"{feature_key[:-5]}_std"
    return f"{feature_key}_std"


def update_baseline(
    user_id: str,
    new_feature_vec: dict[str, float],
    store: FeatureStore,
    alpha: float = 0.05,
) -> None:
    """Apply EMA-based baseline updates with outlier capping for robustness.

    This online update keeps profiles adaptive while preventing large anomalous
    jumps from immediately polluting baseline state.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    profile = store.get_profile(user_id)
    if profile is None:
        raise ValueError("profile not found for baseline update")

    updated = dict(profile.features)
    capped_count = 0

    for key, new_val in new_feature_vec.items():
        old_mean = float(updated.get(key, new_val))
        std_key = _std_key(key)
        sigma = float(updated.get(std_key, 0.0))

        if sigma > 0.0 and abs(new_val - old_mean) > 5.0 * sigma:
            cap_delta = 0.5 * sigma if new_val >= old_mean else -0.5 * sigma
            new_mean = old_mean + cap_delta
            capped_count += 1
        else:
            new_mean = alpha * new_val + (1.0 - alpha) * old_mean

        updated[key] = float(new_mean)

    refreshed = UserProfile(
        user_id=profile.user_id,
        window_days=profile.window_days,
        features=updated,
        generated_at=datetime.now(timezone.utc),
    )
    store.set_profile(user_id, refreshed)
    logger.info("baseline_updated features=%d capped=%d", len(new_feature_vec), capped_count)
