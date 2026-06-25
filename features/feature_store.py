from __future__ import annotations

import logging
from datetime import datetime

import msgpack  # type: ignore[import-untyped]
import redis

from config.settings import load_settings

from .profile_builder import UserProfile

logger = logging.getLogger(__name__)


def _profile_key(user_id: str) -> str:
    """Build namespaced cache keys so profile entries remain collision-free."""

    return f"profile:{user_id}"


def _serialize_profile(profile: UserProfile) -> bytes:
    """Serialize user profiles into compact msgpack payloads for Redis storage.

    Compact binary payloads reduce bandwidth and memory overhead in hot-path cache access.
    """

    payload = {
        "user_id": profile.user_id,
        "window_days": profile.window_days,
        "features": profile.features,
        "generated_at": profile.generated_at.isoformat(),
    }
    packed = msgpack.packb(payload, use_bin_type=True)
    if not isinstance(packed, (bytes, bytearray)):
        raise TypeError("msgpack.packb returned unexpected type")
    return bytes(packed)


def _deserialize_profile(data: bytes) -> UserProfile:
    """Deserialize cached profile bytes back into typed UserProfile objects."""

    payload = msgpack.unpackb(data, raw=False)
    generated_at = datetime.fromisoformat(payload["generated_at"])
    return UserProfile(
        user_id=payload["user_id"],
        window_days=int(payload["window_days"]),
        features=payload["features"],
        generated_at=generated_at,
    )


def _get_client() -> redis.Redis:
    """Create a Redis client from configured connection settings."""

    settings = load_settings()
    return redis.Redis.from_url(settings.REDIS_URL, ssl=True)


def set_profile(user_id: str, profile: UserProfile, ttl: int = 86400) -> None:
    """Cache a profile with TTL so behavioral baselines are fast to retrieve.

    TTL-based caching balances freshness and load on primary storage.
    """

    client = _get_client()
    client.setex(_profile_key(user_id), ttl, _serialize_profile(profile))
    logger.info("profile_cached ttl=%d", ttl)


def get_profile(user_id: str) -> UserProfile | None:
    """Fetch a cached profile if available, returning None on cache miss."""

    client = _get_client()
    data = client.get(_profile_key(user_id))
    if data is None:
        logger.info("profile_cache_miss")
        return None
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("redis returned unexpected data type")
    logger.info("profile_cache_hit")
    return _deserialize_profile(bytes(data))
