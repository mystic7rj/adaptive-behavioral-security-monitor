from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from pydantic import ValidationError

from ingest.schema import EVENT_MODEL_BY_TYPE, NormalizedEvent

from .pii_scrubber import ScrubbedEvent, scrub
from .rolling_aggregates import compute_aggregates

logger = logging.getLogger(__name__)

PROFILE_SQL = """
SELECT user_id, timestamp, event_type, ip, email, session_duration_s, bytes_out
FROM security_events
WHERE user_id = $1 AND timestamp >= $2
ORDER BY timestamp ASC
"""


@dataclass(frozen=True)
class UserProfile:
    """Represent a generated user baseline profile over a fixed historical window."""

    user_id: str
    window_days: int
    features: dict[str, float | int]
    generated_at: datetime


def _coerce_event(row: Mapping[str, Any]) -> NormalizedEvent | None:
    """Validate raw database rows into typed normalized event models.

    This helper isolates row-validation concerns so profile construction can skip
    malformed records safely.
    """

    event_type = str(row.get("event_type", "")).lower()
    model = EVENT_MODEL_BY_TYPE.get(event_type)
    if model is None:
        logger.warning("unknown_event_type event_type=%s", event_type)
        return None

    try:
        return model.model_validate(row)
    except (ValidationError, TypeError, ValueError) as exc:
        logger.warning("event_validation_failed error_type=%s", type(exc).__name__)
        return None


async def _await_result(awaitable: Awaitable[Any]) -> Any:
    """Bridge awaited fetch results when async database APIs are used."""

    return await awaitable


def _fetch_rows(conn: Any, query: str, params: Iterable[Any]) -> list[Mapping[str, Any]]:
    """Fetch rows from either asyncpg-style or DB-API style connections.

    Supporting both interfaces keeps profile generation reusable across runtime
    and test adapters.
    """

    if hasattr(conn, "fetch"):
        result = conn.fetch(query, *params)
        if inspect.isawaitable(result):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                awaited = asyncio.run(_await_result(result))
                return list(awaited)
            raise RuntimeError("build_user_profile cannot await fetch in a running event loop")
        return list(result)

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    if not rows:
        return []
    if hasattr(rows[0], "keys"):
        return [dict(row) for row in rows]
    column_names = [desc[0] for desc in cursor.description]
    return [dict(zip(column_names, row)) for row in rows]


def build_user_profile(user_id: str, window_days: int, conn: Any) -> UserProfile:
    """Build a profile by loading recent events, scrubbing PII, and aggregating features.

    This function exists to provide a deterministic baseline snapshot used by
    detection and online adaptation components.
    """

    if window_days <= 0:
        raise ValueError("window_days must be positive")

    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    rows = _fetch_rows(conn, PROFILE_SQL, (user_id, since))
    scrubbed_events: list[ScrubbedEvent] = []

    for row in rows:
        normalized = _coerce_event(row)
        if normalized is None:
            continue
        scrubbed_events.append(scrub(normalized))  # Scrub before aggregation to enforce privacy by design.

    features = compute_aggregates(scrubbed_events)
    profile = UserProfile(
        user_id=user_id,
        window_days=window_days,
        features=features,
        generated_at=datetime.now(timezone.utc),
    )
    logger.info("profile_built window_days=%d events=%d", window_days, len(scrubbed_events))
    return profile
