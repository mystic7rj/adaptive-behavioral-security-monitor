from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from .schema import EVENT_MODEL_BY_TYPE, NormalizedEvent

logger = logging.getLogger(__name__)

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "user_id": ("userId", "user", "uid"),
    "timestamp": ("time", "ts", "eventTime"),
    "event_type": ("eventType", "type"),
    "ip": ("ipAddress", "ip_address"),
    "email": ("emailAddress", "email_address"),
    "session_duration_s": ("sessionDurationS", "sessionDuration", "session_duration"),
    "bytes_out": ("bytesOut",),
}

EVENT_TYPE_ALIASES: dict[str, str] = {
    "authentication": "auth",
    "login": "auth",
    "net": "network",
    "network_event": "network",
    "file_access": "file",
    "file_event": "file",
}


def _extract_value(raw: dict[str, Any], canonical: str) -> Any | None:
    """Resolve one canonical field from raw payload using known alias keys.

    This helper keeps schema adaptation centralized so upstream producers can vary
    naming conventions without breaking normalization logic.
    """

    if canonical in raw:
        return raw[canonical]
    for alias in FIELD_ALIASES.get(canonical, ()):
        if alias in raw:
            return raw[alias]
    return None


def normalize(raw: dict[str, Any]) -> NormalizedEvent | None:
    """Normalize heterogenous event payloads into strict typed event models.

    This function exists to make downstream feature pipelines resilient to source
    format drift while dropping malformed events safely.
    """

    if not isinstance(raw, dict):
        logger.warning("invalid_payload payload_type=%s", type(raw).__name__)
        return None

    normalized: dict[str, Any] = {}
    # Restrict copied fields to the canonical schema to avoid carrying untrusted extras.
    for field in ("user_id", "timestamp", "event_type", "ip", "email", "session_duration_s", "bytes_out"):
        value = _extract_value(raw, field)
        if value is not None:
            normalized[field] = value

    event_type = normalized.get("event_type")
    if not event_type:
        logger.warning("missing_event_type")
        return None

    normalized_type = EVENT_TYPE_ALIASES.get(str(event_type).lower(), str(event_type).lower())
    normalized["event_type"] = normalized_type

    model = EVENT_MODEL_BY_TYPE.get(normalized_type)
    if model is None:
        logger.warning("unknown_event_type event_type=%s", normalized_type)
        return None

    try:
        return model.model_validate(normalized)
    except ValidationError as exc:
        logger.warning("event_validation_failed error_type=%s", type(exc).__name__)
        return None
