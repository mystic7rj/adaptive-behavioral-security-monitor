from __future__ import annotations

import logging
from datetime import timedelta
from statistics import fmean, pstdev

from .pii_scrubber import ScrubbedEvent

logger = logging.getLogger(__name__)

WINDOWS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}

METRICS = ("login_hour", "session_duration_s", "bytes_out", "unique_ip_count")


def _stats(values: list[float]) -> tuple[float, float, int]:
    """Compute mean/std/count summary stats used by rolling feature generation.

    This helper centralizes empty-list handling so aggregate code stays compact and consistent.
    """

    count = len(values)
    if count == 0:
        return 0.0, 0.0, 0
    mean = fmean(values)
    std = pstdev(values) if count > 1 else 0.0
    return mean, std, count


def _empty_features() -> dict[str, float | int]:
    """Build a zero-initialized feature vector for all metric/window combinations.

    A complete zero vector keeps downstream models shape-stable when no events exist.
    """

    features: dict[str, float | int] = {}
    for window in WINDOWS:
        for metric in METRICS:
            features[f"{metric}_{window}_mean"] = 0.0
            features[f"{metric}_{window}_std"] = 0.0
            features[f"{metric}_{window}_count"] = 0
    return features


def compute_aggregates(events: list[ScrubbedEvent]) -> dict[str, float | int]:
    """Generate rolling-window statistical features from scrubbed user events.

    Rolling summaries provide compact behavioral baselines for anomaly models while
    avoiding storage of raw event-level timelines.
    """

    if not events:
        return _empty_features()

    reference_time = max(event.timestamp for event in events)
    features: dict[str, float | int] = {}

    for window_name, window_delta in WINDOWS.items():
        window_start = reference_time - window_delta
        window_events = [
            event for event in events if window_start <= event.timestamp <= reference_time
        ]

        login_hours = [float(event.login_hour) for event in window_events]
        session_durations = [
            float(event.session_duration_s)
            for event in window_events
            if event.session_duration_s is not None
        ]
        bytes_out_values = [
            float(event.bytes_out)
            for event in window_events
            if event.bytes_out is not None
        ]
        unique_ips = {event.ip_hmac for event in window_events if event.ip_hmac}  # Set deduplicates IP hashes.

        mean, std, count = _stats(login_hours)
        features[f"login_hour_{window_name}_mean"] = mean
        features[f"login_hour_{window_name}_std"] = std
        features[f"login_hour_{window_name}_count"] = count

        mean, std, count = _stats(session_durations)
        features[f"session_duration_s_{window_name}_mean"] = mean
        features[f"session_duration_s_{window_name}_std"] = std
        features[f"session_duration_s_{window_name}_count"] = count

        mean, std, count = _stats(bytes_out_values)
        features[f"bytes_out_{window_name}_mean"] = mean
        features[f"bytes_out_{window_name}_std"] = std
        features[f"bytes_out_{window_name}_count"] = count

        unique_count = len(unique_ips)
        features[f"unique_ip_count_{window_name}_mean"] = float(unique_count)
        features[f"unique_ip_count_{window_name}_std"] = 0.0
        features[f"unique_ip_count_{window_name}_count"] = unique_count

    logger.info("aggregates_computed windows=%d events=%d", len(WINDOWS), len(events))
    return features
