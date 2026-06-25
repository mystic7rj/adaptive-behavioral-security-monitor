from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ingest.normalizer import normalize
from ingest.schema import AuthEvent


def test_valid_event_parses_correctly() -> None:
    event = AuthEvent.model_validate(
        {
            "user_id": uuid4(),
            "timestamp": datetime.now(timezone.utc),
            "event_type": "auth",
        }
    )
    assert event.event_type == "auth"


def test_invalid_event_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        AuthEvent.model_validate(
            {
                "timestamp": datetime.now(timezone.utc),
                "event_type": "auth",
            }
        )


def test_normalizer_returns_none_on_unknown_type() -> None:
    result = normalize(
        {
            "user_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc),
            "event_type": "unknown",
        }
    )
    assert result is None


def test_normalizer_maps_known_alias() -> None:
    uid = uuid4()
    result = normalize(
        {
            "userId": str(uid),
            "time": datetime.now(timezone.utc),
            "eventType": "auth",
        }
    )
    assert result is not None
    assert result.user_id == uid
    assert result.event_type == "auth"
