from __future__ import annotations

import asyncio
import logging

import asyncpg  # type: ignore[import-untyped]

from config.settings import Settings, load_settings
from ingest.schema import NormalizedEvent

logger = logging.getLogger(__name__)

INSERT_SQL = """
INSERT INTO security_events (user_id, timestamp, event_type)
VALUES ($1, $2, $3)
"""


async def connect(settings: Settings | None = None) -> asyncpg.Connection:
    """Open a database connection using configured Timescale/PostgreSQL settings.

    A dedicated connector keeps storage configuration handling centralized and testable.
    """

    resolved = settings or load_settings()
    return await asyncpg.connect(resolved.DB_URL)


async def write_batch(events: list[NormalizedEvent], conn: asyncpg.Connection) -> None:
    """Persist normalized events in batches with bounded retry/backoff.

    Batched writes reduce database round trips and retries absorb transient DB
    pressure without dropping data.
    """

    if not events:
        return  # No-op avoids issuing empty SQL operations.

    records = [(event.user_id, event.timestamp, event.event_type) for event in events]
    attempt = 0
    delay = 0.5

    while True:
        try:
            await conn.executemany(INSERT_SQL, records)
            logger.info("batch_written size=%d", len(records))
            return
        except asyncpg.PostgresError as exc:
            attempt += 1
            logger.warning("write_failed attempt=%d error=%s", attempt, exc)
            if attempt >= 3:
                raise
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff lowers repeated contention on recovery attempts.
