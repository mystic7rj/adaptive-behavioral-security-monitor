from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from config.settings import load_settings
from ingest.schema import NormalizedEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScrubbedEvent:
    """Represent a privacy-preserving event payload used by feature engineering."""

    user_id: UUID
    timestamp: datetime
    event_type: str
    login_hour: int
    session_duration_s: float | None
    bytes_out: int | None
    ip_hmac: str | None
    email_sha256: str | None


def _hmac_sha256(value: str, key: str) -> str:
    """Compute keyed digest for reversible-resistant pseudonymization of sensitive fields."""

    return hmac.new(key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def _sha256(value: str) -> str:
    """Compute one-way digest for fields where keyed linkage is not required."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def scrub(event: NormalizedEvent) -> ScrubbedEvent:
    """Scrub direct identifiers from normalized events while preserving analytic utility.

    This function exists to enforce privacy boundaries before events enter feature
    pipelines and downstream model artifacts.
    """

    settings = load_settings()
    ip_hmac = _hmac_sha256(event.ip, settings.SCRUBBER_HMAC_KEY) if event.ip else None
    email_sha256 = _sha256(event.email) if event.email else None

    return ScrubbedEvent(
        user_id=event.user_id,
        timestamp=event.timestamp,
        event_type=event.event_type,
        login_hour=event.timestamp.hour,
        session_duration_s=event.session_duration_s,
        bytes_out=event.bytes_out,
        ip_hmac=ip_hmac,
        email_sha256=email_sha256,
    )
