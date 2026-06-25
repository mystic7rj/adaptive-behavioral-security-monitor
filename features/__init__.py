from .feature_store import get_profile, set_profile
from .pii_scrubber import ScrubbedEvent, scrub
from .profile_builder import UserProfile, build_user_profile
from .rolling_aggregates import compute_aggregates
from .sequence_builder import build_session_sequences

"""Feature-engineering package exports.

This module provides a stable import surface for profile and sequence feature workflows.
"""

__all__ = [
    "ScrubbedEvent",
    "UserProfile",
    "scrub",
    "compute_aggregates",
    "build_session_sequences",
    "build_user_profile",
    "set_profile",
    "get_profile",
]
