"""Model-health monitoring utilities based on reconstruction-error behavior."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import quantiles

from config.settings import load_settings

from .metrics import reconstruction_error_histogram


@dataclass(frozen=True)
class HealthStatus:
    """Health-check output for model reconstruction quality."""

    healthy: bool
    p95_error: float
    status_message: str


class ModelHealthMonitor:
    """Track reconstruction errors and detect degradation/permissiveness failures.

    Two failure modes are monitored:
    - p95 > 2x baseline: model error inflation suggests degraded fit or concept drift.
    - p95 < 0.1x baseline: model may be too permissive, reducing anomaly sensitivity.
    Both conditions matter because either can break alert quality in opposite directions.
    """

    def __init__(self, baseline_p95: float | None = None, max_errors: int = 1000) -> None:
        """Initialize monitor with configured baseline and bounded rolling history."""

        resolved_baseline = baseline_p95
        if resolved_baseline is None:
            # Pulling from settings keeps thresholds centrally configurable and auditable.
            resolved_baseline = load_settings().MONITORING_MODEL_BASELINE_P95
        if resolved_baseline <= 0:
            raise ValueError("baseline_p95 must be positive")
        if max_errors <= 0:
            raise ValueError("max_errors must be positive")

        self._baseline_p95 = float(resolved_baseline)
        self._max_errors = int(max_errors)
        self._errors: list[float] = []

    def record_error(self, error: float) -> None:
        """Record one reconstruction error and update histogram telemetry."""

        value = float(error)
        self._errors.append(value)
        if len(self._errors) > self._max_errors:
            # Drop oldest value so the monitor reflects recent behavior only.
            self._errors.pop(0)
        reconstruction_error_histogram.observe(value)

    def check_health(self) -> HealthStatus:
        """Evaluate p95 against baseline bands and return a health decision."""

        if not self._errors:
            return HealthStatus(healthy=True, p95_error=0.0, status_message="healthy: no errors recorded yet")

        if len(self._errors) == 1:
            p95_error = self._errors[0]
        else:
            # statistics.quantiles with n=100 returns percentile cut points; index 94 is p95.
            percentiles = quantiles(self._errors, n=100, method="inclusive")
            p95_error = percentiles[94]

        high_threshold = self._baseline_p95 * 2.0
        low_threshold = self._baseline_p95 * 0.1
        if p95_error > high_threshold:
            return HealthStatus(
                healthy=False,
                p95_error=p95_error,
                status_message="unhealthy: p95 reconstruction error exceeded 2x baseline",
            )
        if p95_error < low_threshold:
            return HealthStatus(
                healthy=False,
                p95_error=p95_error,
                status_message="unhealthy: p95 reconstruction error fell below 0.1x baseline",
            )
        return HealthStatus(healthy=True, p95_error=p95_error, status_message="healthy: p95 within baseline bounds")
