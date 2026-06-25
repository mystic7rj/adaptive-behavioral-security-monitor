from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("prometheus_client")

from monitoring.data_drift import detect_feature_drift
from monitoring.model_health import ModelHealthMonitor
from monitoring.sla_checker import check_sla


def test_drift_detected_when_mean_diff_exceeds_two_sigma() -> None:
    """Drift should trigger when mean shift is larger than 2 * reference stdev."""

    reference = [1.0, 2.0, 3.0, 4.0, 5.0]
    current = [9.0, 10.0, 11.0]
    result = detect_feature_drift(current=current, reference=reference, feature_name="login_frequency")
    assert result.drifted is True


def test_drift_not_detected_when_distributions_are_similar() -> None:
    """Drift should not trigger for close means under the same spread."""

    reference = [10.0, 11.0, 12.0, 13.0, 14.0]
    current = [10.5, 11.5, 12.5, 13.5]
    result = detect_feature_drift(current=current, reference=reference, feature_name="session_duration")
    assert result.drifted is False


def test_model_health_unhealthy_when_p95_doubles() -> None:
    """Model health should turn unhealthy when p95 crosses 2x baseline."""

    monitor = ModelHealthMonitor(baseline_p95=1.0)
    for _ in range(100):
        # Uniformly high errors make p95 deterministic and above degradation threshold.
        monitor.record_error(2.2)
    status = monitor.check_health()
    assert status.healthy is False
    assert status.p95_error > 2.0


def test_sla_breached_when_latency_exceeds_threshold() -> None:
    """SLA should be breached when latency is above threshold."""

    ingest_time = datetime.now(timezone.utc)
    alert_time = ingest_time + timedelta(seconds=90)
    status = check_sla(ingest_time=ingest_time, alert_time=alert_time, threshold_seconds=60.0)
    assert status.breached is True


def test_sla_not_breached_when_latency_within_threshold() -> None:
    """SLA should not be breached when latency is within threshold."""

    ingest_time = datetime.now(timezone.utc)
    alert_time = ingest_time + timedelta(seconds=30)
    status = check_sla(ingest_time=ingest_time, alert_time=alert_time, threshold_seconds=60.0)
    assert status.breached is False
