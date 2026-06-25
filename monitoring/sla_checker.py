"""SLA evaluation for ingest-to-alert pipeline latency."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from config.settings import MONITORING_SLA_THRESHOLD_SECONDS_DEFAULT

from .metrics import pipeline_lag_seconds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SLAStatus:
    """Result object returned after evaluating one ingest-to-alert latency sample."""

    breached: bool
    latency_seconds: float
    threshold_seconds: float


def check_sla(
    ingest_time: datetime,
    alert_time: datetime,
    threshold_seconds: float = MONITORING_SLA_THRESHOLD_SECONDS_DEFAULT,
) -> SLAStatus:
    """Check SLA compliance and record latency telemetry for the processing pipeline.

    60 seconds is the default because many SOC workflows treat one-minute response as
    the practical upper bound for near-real-time triage. A breach means operations may
    be reacting too late to contain risky behavior before impact expands.
    """

    latency_seconds = (alert_time - ingest_time).total_seconds()
    pipeline_lag_seconds.observe(latency_seconds)
    breached = latency_seconds > threshold_seconds
    if breached:
        # Structured logging intentionally avoids any raw user payload fields.
        logger.warning("sla_breached latency_seconds=%.3f threshold_seconds=%.3f", latency_seconds, threshold_seconds)
    return SLAStatus(
        breached=breached,
        latency_seconds=latency_seconds,
        threshold_seconds=threshold_seconds,
    )
