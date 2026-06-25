from __future__ import annotations

from datetime import datetime
from typing import Literal, TypeAlias, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class BaseEvent(BaseModel):
    """Define shared validated fields used by all normalized event categories.

    A common base model avoids schema duplication and enforces consistent typing
    guarantees across ingestion paths.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    timestamp: datetime
    event_type: str
    ip: str | None = None
    email: str | None = None
    session_duration_s: float | None = None
    bytes_out: int | None = None


class AuthEvent(BaseEvent):
    """Represent authentication activity after normalization and validation."""

    event_type: Literal["auth"]


class NetworkEvent(BaseEvent):
    """Represent network activity after normalization and validation."""

    event_type: Literal["network"]


class FileEvent(BaseEvent):
    """Represent file-access activity after normalization and validation."""

    event_type: Literal["file"]


NormalizedEvent = Union[AuthEvent, NetworkEvent, FileEvent]
EventModel: TypeAlias = type[AuthEvent] | type[NetworkEvent] | type[FileEvent]

EVENT_MODEL_BY_TYPE: dict[str, EventModel] = {
    "auth": AuthEvent,
    "network": NetworkEvent,
    "file": FileEvent,
}

__all__ = [
    "AuthEvent",
    "NetworkEvent",
    "FileEvent",
    "NormalizedEvent",
    "EventModel",
    "EVENT_MODEL_BY_TYPE",
]
