from __future__ import annotations

import logging
from datetime import timedelta

import torch
from torch import Tensor

from .pii_scrubber import ScrubbedEvent

logger = logging.getLogger(__name__)

SESSION_GAP = timedelta(seconds=1800)
PAD_INDEX = 0
BASE_EVENT_INDEX: dict[str, int] = {
    "unknown": 1,
    "auth": 2,
    "network": 3,
    "file": 4,
}


def _build_event_index(events: list[ScrubbedEvent]) -> dict[str, int]:
    """Build a stable token index for event types, extending defaults for unseen types.

    Dynamic indexing keeps sequence models robust as new event categories appear.
    """

    index = dict(BASE_EVENT_INDEX)
    next_index = max(index.values()) + 1
    unseen = sorted({event.event_type for event in events if event.event_type not in index})
    for event_type in unseen:
        index[event_type] = next_index
        next_index += 1
    return index


def _pad_or_truncate(tokens: list[int], max_len: int) -> list[int]:
    """Enforce fixed-length token sequences required by batch-oriented sequence models."""

    if len(tokens) >= max_len:
        return tokens[:max_len]
    return tokens + [PAD_INDEX] * (max_len - len(tokens))


def build_session_sequences(events: list[ScrubbedEvent], max_len: int = 50) -> Tensor:
    """Split events into sessions and convert them into padded token sequences.

    Sessionization captures temporal behavior structure while fixed-length encoding
    supports efficient LSTM-style model inputs.
    """

    if max_len <= 0:
        raise ValueError("max_len must be positive")

    if not events:
        return torch.zeros((0, max_len), dtype=torch.long)

    event_index = _build_event_index(events)
    ordered = sorted(events, key=lambda event: event.timestamp)
    sessions: list[list[int]] = []
    current: list[int] = []
    last_time = ordered[0].timestamp

    for event in ordered:
        if event.timestamp - last_time > SESSION_GAP:
            if current:
                sessions.append(current)
            current = []
        token = event_index.get(event.event_type, BASE_EVENT_INDEX["unknown"])
        current.append(token)
        last_time = event.timestamp
    if current:
        sessions.append(current)

    padded = [_pad_or_truncate(tokens, max_len) for tokens in sessions]
    logger.info("sessions_built sessions=%d max_len=%d", len(padded), max_len)
    return torch.tensor(padded, dtype=torch.long)
